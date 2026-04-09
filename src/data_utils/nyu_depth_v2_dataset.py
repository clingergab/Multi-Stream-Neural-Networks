"""
NYU Depth V2 Dataset Loader for scene classification.

Loads preprocessed NYU Depth V2 dataset with RGB and depth images.
Class names and normalization stats are loaded dynamically from the data root.

Tensors are stored at 256x256. At load time:
  - Train: RandomCrop(crop_size) + horizontal flip + augmentations
  - Val/Test: CenterCrop(crop_size)

Note: NYU Depth V2 labeled data has inpainted (hole-filled) depth, so depth=0
sentinel pixels are rare/absent. The sentinel replacement logic is kept for
robustness but typically has no effect.
"""

import json
import os
import random
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F2

from src.training.augmentation_config import (
    BASE_FLIP_P,
    BASE_COLOR_JITTER_P,
    BASE_BLUR_P,
    BASE_GRAYSCALE_P,
    BASE_RGB_ERASING_P,
    BASE_DEPTH_AUG_P,
    BASE_DEPTH_ERASING_P,
    BASE_BRIGHTNESS,
    BASE_CONTRAST,
    BASE_SATURATION,
    BASE_HUE,
    BASE_BLUR_SIGMA_MIN,
    BASE_BLUR_SIGMA_MAX,
    BASE_ERASING_SCALE_MIN,
    BASE_ERASING_SCALE_MAX,
    BASE_ERASING_RATIO_MIN,
    BASE_ERASING_RATIO_MAX,
    BASE_DEPTH_BRIGHTNESS,
    BASE_DEPTH_CONTRAST,
    BASE_DEPTH_NOISE_STD,
    MAX_PROBABILITY,
    MAX_BRIGHTNESS,
    MAX_CONTRAST,
    MAX_SATURATION,
    MAX_HUE,
    MAX_BLUR_SIGMA,
    MAX_DEPTH_BRIGHTNESS,
    MAX_DEPTH_CONTRAST,
    MAX_DEPTH_NOISE_STD,
    MAX_ERASING_SCALE,
)


def _load_class_names(data_root: str) -> list[str]:
    """Load class names from class_names.txt in data_root.

    Expected format per line: '0: bathroom'
    """
    path = os.path.join(data_root, 'class_names.txt')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"class_names.txt not found in {data_root}. "
            f"Run scripts/preprocess_nyu_depth_v2.py first."
        )
    names = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                names.append(line.split(': ', 1)[1])
    return names


def _load_norm_stats(data_root: str) -> dict:
    """Load normalization statistics from norm_stats.json in data_root.

    Returns dict with keys: rgb_mean, rgb_std, depth_mean, depth_std.
    """
    path = os.path.join(data_root, 'norm_stats.json')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"norm_stats.json not found in {data_root}. "
            f"Run scripts/preprocess_nyu_depth_v2.py first."
        )
    with open(path, 'r') as f:
        return json.load(f)


class NYUDepthV2Dataset(Dataset):
    """
    NYU Depth V2 dataset for scene classification.

    Class names are loaded dynamically from class_names.txt in data_root.
    Normalization stats are loaded from norm_stats.json in data_root.

    Tensors are stored at 256x256. crop_size controls the output:
      - Train: RandomCrop(crop_size) from 256x256
      - Val/Test: CenterCrop(crop_size) from 256x256

    Directory structure:
        data_root/
            class_names.txt
            norm_stats.json
            train/ or val/ or test/
                rgb_tensors.pt    # [N, 3, 256, 256] uint8
                depth_tensors.pt  # [N, 1, 256, 256] uint16 (mm)
                labels.txt
    """

    VALID_SPLITS = ('train', 'val', 'test')

    def __init__(
        self,
        data_root='data/nyu_depth_v2_10_traintest',
        split='train',
        crop_size: int = 224,
        normalize: bool = True,
        rgb_aug_prob: float = 1.0,
        rgb_aug_mag: float = 1.0,
        depth_aug_prob: float = 1.0,
        depth_aug_mag: float = 1.0,
    ):
        """
        Args:
            data_root: Root directory of preprocessed dataset
            split: One of 'train', 'val', or 'test'
            crop_size: Output crop size. Train uses RandomCrop, val/test use CenterCrop.
            normalize: If True, apply normalization in __getitem__().
                      Set to False when using GPU augmentation (which handles
                      normalization on GPU after augmentation).
            rgb_aug_prob: Scales probability of RGB augmentations (default: 1.0 = baseline)
            rgb_aug_mag: Scales magnitude of RGB augmentations (default: 1.0 = baseline)
            depth_aug_prob: Scales probability of Depth augmentations (default: 1.0 = baseline)
            depth_aug_mag: Scales magnitude of Depth augmentations (default: 1.0 = baseline)
        """
        if split not in self.VALID_SPLITS:
            raise ValueError(f"split must be one of {self.VALID_SPLITS}, got '{split}'")

        self.data_root = data_root
        self.split = split
        self.crop_size = crop_size
        self.normalize = normalize

        self.CLASS_NAMES = _load_class_names(data_root)
        self.num_classes = len(self.CLASS_NAMES)

        self._norm_stats = _load_norm_stats(data_root)

        self.rgb_aug_prob = rgb_aug_prob
        self.rgb_aug_mag = rgb_aug_mag
        self.depth_aug_prob = depth_aug_prob
        self.depth_aug_mag = depth_aug_mag

        self._compute_scaled_aug_values()

        self.split_dir = os.path.join(data_root, split)

        labels_file = os.path.join(self.split_dir, 'labels.txt')
        with open(labels_file, 'r') as f:
            self.labels = [int(line.strip()) for line in f.readlines()]

        self.num_samples = len(self.labels)

        rgb_tensor_path = os.path.join(self.split_dir, 'rgb_tensors.pt')
        depth_tensor_path = os.path.join(self.split_dir, 'depth_tensors.pt')
        self.rgb_tensors = torch.load(rgb_tensor_path, weights_only=True, mmap=True)
        self.depth_tensors = torch.load(depth_tensor_path, weights_only=True, mmap=True)
        print(f"Loaded NYU Depth V2 {split}: {self.num_samples} samples, "
              f"{self.num_classes} classes (tensors, mmap)")

        if split == 'train' and any(p != 1.0 for p in [rgb_aug_prob, rgb_aug_mag, depth_aug_prob, depth_aug_mag]):
            self._log_augmentation_config()

    def __len__(self):
        return self.num_samples

    def _compute_scaled_aug_values(self):
        """Pre-compute scaled augmentation values based on aug_prob and aug_mag parameters."""
        sync_prob = (self.rgb_aug_prob + self.depth_aug_prob) / 2

        # === SYNCHRONIZED (flip) ===
        self._flip_p = min(BASE_FLIP_P * sync_prob, MAX_PROBABILITY)

        # === RGB-ONLY ===
        self._color_jitter_p = min(BASE_COLOR_JITTER_P * self.rgb_aug_prob, MAX_PROBABILITY)
        self._brightness = min(BASE_BRIGHTNESS * self.rgb_aug_mag, MAX_BRIGHTNESS)
        self._contrast = min(BASE_CONTRAST * self.rgb_aug_mag, MAX_CONTRAST)
        self._saturation = min(BASE_SATURATION * self.rgb_aug_mag, MAX_SATURATION)
        self._hue = min(BASE_HUE * self.rgb_aug_mag, MAX_HUE)

        self._blur_p = min(BASE_BLUR_P * self.rgb_aug_prob, MAX_PROBABILITY)
        self._blur_sigma_min = BASE_BLUR_SIGMA_MIN
        self._blur_sigma_max = min(BASE_BLUR_SIGMA_MAX * self.rgb_aug_mag, MAX_BLUR_SIGMA)

        self._grayscale_p = min(BASE_GRAYSCALE_P * self.rgb_aug_prob, MAX_PROBABILITY)

        self._rgb_erasing_p = min(BASE_RGB_ERASING_P * self.rgb_aug_prob, MAX_PROBABILITY)
        self._rgb_erasing_scale_min = BASE_ERASING_SCALE_MIN
        self._rgb_erasing_scale_max = min(BASE_ERASING_SCALE_MAX * self.rgb_aug_mag, MAX_ERASING_SCALE)

        # === DEPTH-ONLY ===
        self._depth_aug_p = min(BASE_DEPTH_AUG_P * self.depth_aug_prob, MAX_PROBABILITY)
        self._depth_brightness = min(BASE_DEPTH_BRIGHTNESS * self.depth_aug_mag, MAX_DEPTH_BRIGHTNESS)
        self._depth_contrast = min(BASE_DEPTH_CONTRAST * self.depth_aug_mag, MAX_DEPTH_CONTRAST)
        self._depth_noise_std = min(BASE_DEPTH_NOISE_STD * self.depth_aug_mag, MAX_DEPTH_NOISE_STD)

        self._depth_erasing_p = min(BASE_DEPTH_ERASING_P * self.depth_aug_prob, MAX_PROBABILITY)
        self._depth_erasing_scale_min = BASE_ERASING_SCALE_MIN
        self._depth_erasing_scale_max = min(BASE_ERASING_SCALE_MAX * self.depth_aug_mag, MAX_ERASING_SCALE)

        # === PRE-CREATE REUSABLE TRANSFORM INSTANCES ===
        self._color_jitter_transform = v2.ColorJitter(
            brightness=self._brightness,
            contrast=self._contrast,
            saturation=self._saturation,
            hue=self._hue,
        )
        self._rgb_erasing_transform = v2.RandomErasing(
            p=1.0,
            scale=(self._rgb_erasing_scale_min, self._rgb_erasing_scale_max),
            ratio=(BASE_ERASING_RATIO_MIN, BASE_ERASING_RATIO_MAX),
        )
        self._depth_erasing_transform = v2.RandomErasing(
            p=1.0,
            scale=(self._depth_erasing_scale_min, self._depth_erasing_scale_max),
            ratio=(BASE_ERASING_RATIO_MIN, BASE_ERASING_RATIO_MAX),
        )

    def _log_augmentation_config(self):
        """Log computed augmentation values when scaling is applied."""
        print(f"\nAugmentation scaling applied:")
        print(f"  RGB:   prob={self.rgb_aug_prob:.2f}, mag={self.rgb_aug_mag:.2f}")
        print(f"  Depth: prob={self.depth_aug_prob:.2f}, mag={self.depth_aug_mag:.2f}")
        print(f"  Computed values:")
        print(f"    [Sync]  Flip prob: {BASE_FLIP_P:.2f} -> {self._flip_p:.3f}")
        print(f"    [RGB]   ColorJitter prob: {BASE_COLOR_JITTER_P:.2f} -> {self._color_jitter_p:.3f}")
        print(f"    [RGB]   Brightness: ±{BASE_BRIGHTNESS:.2f} -> ±{self._brightness:.3f}")
        print(f"    [RGB]   Blur prob: {BASE_BLUR_P:.2f} -> {self._blur_p:.3f}")
        print(f"    [RGB]   Grayscale prob: {BASE_GRAYSCALE_P:.2f} -> {self._grayscale_p:.3f}")
        print(f"    [RGB]   Erasing prob: {BASE_RGB_ERASING_P:.2f} -> {self._rgb_erasing_p:.3f}")
        print(f"    [Depth] Aug prob: {BASE_DEPTH_AUG_P:.2f} -> {self._depth_aug_p:.3f}")
        print(f"    [Depth] Brightness: ±{BASE_DEPTH_BRIGHTNESS:.2f} -> ±{self._depth_brightness:.3f}")
        print(f"    [Depth] Noise std: {BASE_DEPTH_NOISE_STD:.3f} -> {self._depth_noise_std:.3f}")
        print(f"    [Depth] Erasing prob: {BASE_DEPTH_ERASING_P:.2f} -> {self._depth_erasing_p:.3f}")

    def __getitem__(self, idx):
        """
        Returns:
            rgb: RGB image tensor [3, crop_size, crop_size] float32
            depth: Depth image tensor [1, crop_size, crop_size] float32
            label: Class label (0 to num_classes-1)
        """
        rgb = self.rgb_tensors[idx]      # [3, H, W] uint8
        depth = self.depth_tensors[idx]  # [1, H, W] uint16 (mm)

        # Convert depth early: uint16 mm -> float32 meters
        depth = depth.float() / 1000.0

        # ==================== TRAINING AUGMENTATION ====================
        if self.split == 'train':
            # 1. Synchronized Random Horizontal Flip
            if np.random.random() < self._flip_p:
                rgb = F2.horizontal_flip(rgb)
                depth = F2.horizontal_flip(depth)

            # 2. Synchronized RandomCrop (256 -> crop_size)
            i, j, h, w = v2.RandomCrop.get_params(
                rgb, output_size=(self.crop_size, self.crop_size)
            )
            rgb = F2.crop(rgb, i, j, h, w)
            depth = F2.crop(depth, i, j, h, w)

            # 3-5. RGB-Only Appearance Augmentation
            if self.normalize:
                # 3. Color Jitter
                if np.random.random() < self._color_jitter_p:
                    rgb = self._color_jitter_transform(rgb)

                # 4. Gaussian Blur
                if np.random.random() < self._blur_p:
                    kernel_size = int(np.random.choice([3, 5, 7]))
                    sigma = float(np.random.uniform(self._blur_sigma_min, self._blur_sigma_max))
                    rgb = F2.gaussian_blur(rgb, kernel_size=kernel_size, sigma=sigma)

                # 5. Grayscale
                if np.random.random() < self._grayscale_p:
                    rgb = F2.rgb_to_grayscale(rgb, num_output_channels=3)

            # 6. Depth-Only: Combined Appearance Augmentation
            if np.random.random() < self._depth_aug_p:
                depth_missing_mask = (depth == 0.0)

                d_min = depth.min()
                d_max = depth.max()
                d_range = d_max - d_min
                if d_range > 1e-6:
                    depth_01 = (depth - d_min) / d_range
                else:
                    depth_01 = torch.zeros_like(depth)

                brightness_factor = np.random.uniform(
                    1.0 - self._depth_brightness,
                    1.0 + self._depth_brightness,
                )
                contrast_factor = np.random.uniform(
                    1.0 - self._depth_contrast,
                    1.0 + self._depth_contrast,
                )

                depth_01 = (depth_01 - 0.5) * contrast_factor + 0.5
                depth_01 = depth_01 * brightness_factor
                depth_01 = depth_01 + torch.randn_like(depth_01) * self._depth_noise_std

                if d_range > 1e-6:
                    depth = depth_01.clamp(0.0, 1.0) * d_range + d_min

                if depth_missing_mask.any():
                    depth[depth_missing_mask] = 0.0

        else:
            # Val/Test: CenterCrop (256 -> crop_size)
            rgb = F2.center_crop(rgb, (self.crop_size, self.crop_size))
            depth = F2.center_crop(depth, (self.crop_size, self.crop_size))

        # ==================== TO FLOAT32 ====================
        if rgb.dtype == torch.uint8:
            rgb = rgb.float() / 255.0

        # ==================== SENTINEL REPLACEMENT ====================
        zero_mask = (depth == 0.0)
        if zero_mask.any():
            depth[zero_mask] = self._norm_stats['depth_mean'][0]

        # ==================== NORMALIZATION ====================
        if self.normalize:
            rgb = F2.normalize(
                rgb,
                mean=self._norm_stats['rgb_mean'],
                std=self._norm_stats['rgb_std'],
            )
            depth = F2.normalize(
                depth,
                mean=self._norm_stats['depth_mean'],
                std=self._norm_stats['depth_std'],
            )

            # 7. Post-normalization Random Erasing (CPU mode only)
            if self.split == 'train':
                if np.random.random() < self._rgb_erasing_p:
                    rgb = self._rgb_erasing_transform(rgb)

                if np.random.random() < self._depth_erasing_p:
                    depth = self._depth_erasing_transform(depth)

        label = self.labels[idx]
        return rgb, depth, label

    def get_class_weights(self):
        """
        Calculate class weights for weighted loss (inverse frequency).

        Returns:
            Tensor of shape [num_classes] with weights
        """
        label_counts = Counter(self.labels)

        weights = torch.zeros(self.num_classes)
        total = len(self.labels)

        for class_idx in range(self.num_classes):
            count = label_counts.get(class_idx, 0)
            if count > 0:
                weights[class_idx] = total / (self.num_classes * count)
            else:
                weights[class_idx] = 0.0

        return weights

    def get_class_distribution(self):
        """
        Get class distribution statistics.

        Returns:
            Dictionary with class counts and percentages
        """
        label_counts = Counter(self.labels)

        distribution = {}
        for class_idx in range(self.num_classes):
            count = label_counts.get(class_idx, 0)
            percentage = (count / self.num_samples) * 100
            distribution[self.CLASS_NAMES[class_idx]] = {
                'count': count,
                'percentage': percentage
            }

        return distribution

    def get_norm_stats(self):
        """Return the normalization statistics dict loaded from norm_stats.json."""
        return self._norm_stats


class _WorkerInitFn:
    """
    Callable class for DataLoader worker initialization.

    This is a class instead of a nested function to allow pickling
    for multiprocessing when num_workers > 0.
    """
    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, worker_id: int):
        worker_seed = self.seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)


def get_nyu_depth_v2_dataloaders(
    data_root='data/nyu_depth_v2_10_traintest',
    batch_size=32,
    num_workers=4,
    crop_size: int = 224,
    use_class_weights=False,
    stratified=False,
    seed=None,
    normalize: bool = True,
    rgb_aug_prob: float = 1.0,
    rgb_aug_mag: float = 1.0,
    depth_aug_prob: float = 1.0,
    depth_aug_mag: float = 1.0,
):
    """
    Create train, validation, and test dataloaders for NYU Depth V2.

    Automatically detects available splits: if val/ directory exists, creates a
    val_loader; otherwise val_loader is None.

    Args:
        data_root: Root directory of preprocessed dataset
        batch_size: Batch size
        num_workers: Number of dataloader workers
        crop_size: Output crop size (RandomCrop for train, CenterCrop for val/test)
        use_class_weights: If True, return class weights for loss
        stratified: If True, use stratified sampling for training
        seed: Random seed for reproducible data loading
        normalize: If True, apply normalization in dataset __getitem__()
        rgb_aug_prob: Scales probability of RGB augmentations (default: 1.0 = baseline)
        rgb_aug_mag: Scales magnitude of RGB augmentations (default: 1.0 = baseline)
        depth_aug_prob: Scales probability of Depth augmentations (default: 1.0 = baseline)
        depth_aug_mag: Scales magnitude of Depth augmentations (default: 1.0 = baseline)

    Returns:
        train_loader, val_loader, test_loader, (optional) class_weights
        val_loader is None if no val/ directory exists in data_root.
    """
    train_dataset = NYUDepthV2Dataset(
        data_root=data_root,
        split='train',
        crop_size=crop_size,
        normalize=normalize,
        rgb_aug_prob=rgb_aug_prob,
        rgb_aug_mag=rgb_aug_mag,
        depth_aug_prob=depth_aug_prob,
        depth_aug_mag=depth_aug_mag,
    )

    has_val = os.path.isdir(os.path.join(data_root, 'val'))
    val_dataset = None
    if has_val:
        val_dataset = NYUDepthV2Dataset(
            data_root=data_root,
            split='val',
            crop_size=crop_size,
            normalize=normalize,
        )

    test_dataset = NYUDepthV2Dataset(
        data_root=data_root,
        split='test',
        crop_size=crop_size,
        normalize=normalize,
    )

    worker_init_fn = None
    generator = None

    if seed is not None:
        worker_init_fn = _WorkerInitFn(seed)
        generator = torch.Generator().manual_seed(seed)

    train_sampler = None
    train_shuffle = True

    if stratified:
        label_counts = Counter(train_dataset.labels)
        num_samples = len(train_dataset.labels)

        sample_weights = [1.0 / label_counts[label] for label in train_dataset.labels]
        sample_weights = torch.tensor(sample_weights, dtype=torch.float64)

        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=num_samples,
            replacement=True,
            generator=generator
        )
        train_shuffle = False

        print(f"\nStratified sampling enabled (training only):")
        print(f"  Train class imbalance: {max(label_counts.values())/min(label_counts.values()):.1f}x")
        print(f"  Each training batch will have balanced class representation")

    mp_kwargs = {}
    if num_workers > 0:
        mp_kwargs = dict(
            pin_memory=True,
            persistent_workers=True,
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        prefetch_factor=4 if num_workers > 0 else None,
        worker_init_fn=worker_init_fn,
        generator=generator if train_sampler is None else None,
        **mp_kwargs,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            prefetch_factor=2 if num_workers > 0 else None,
            worker_init_fn=worker_init_fn,
            **mp_kwargs,
        )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=worker_init_fn,
        **mp_kwargs,
    )

    print(f"\nDataLoader Info:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader) if val_loader else 'N/A (no val split)'}")
    print(f"  Test batches: {len(test_loader)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Stratified: {stratified}")

    if use_class_weights:
        class_weights = train_dataset.get_class_weights()
        print(f"\nClass weights computed (inverse frequency)")
        return train_loader, val_loader, test_loader, class_weights

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    """Test the dataset loader."""

    print("Testing NYU Depth V2 Dataset Loader")
    print("=" * 80)

    train_dataset = NYUDepthV2Dataset(split='train')

    print(f"\nTrain set: {len(train_dataset)} samples, {train_dataset.num_classes} classes")
    print(f"Classes: {train_dataset.CLASS_NAMES}")

    # Test getting a sample
    rgb, depth, label = train_dataset[0]
    print(f"\nSample 0:")
    print(f"  RGB shape: {rgb.shape}")
    print(f"  Depth shape: {depth.shape}")
    print(f"  Label: {label} ({train_dataset.CLASS_NAMES[label]})")

    # Test class distribution
    print("\nTrain class distribution:")
    train_dist = train_dataset.get_class_distribution()
    for class_name in train_dataset.CLASS_NAMES:
        info = train_dist[class_name]
        print(f"  {class_name:20s}: {info['count']:5d} ({info['percentage']:5.2f}%)")

    # Test dataloader
    print("\nTesting dataloaders...")
    train_loader, val_loader, test_loader = get_nyu_depth_v2_dataloaders(
        batch_size=16, num_workers=0
    )

    rgb_batch, depth_batch, labels_batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  RGB: {rgb_batch.shape}")
    print(f"  Depth: {depth_batch.shape}")
    print(f"  Labels: {labels_batch.shape}")

    print("\nDataset loader working correctly!")
