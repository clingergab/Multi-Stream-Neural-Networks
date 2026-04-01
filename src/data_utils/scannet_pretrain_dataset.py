"""
ScanNet Pretrain Dataset Loader for LINet pretraining.

Loads per-sample tensor files from a pre-split directory layout with
paired RGB and depth data.  The dataset root must contain ``train/``
and ``val/`` subdirectories (each with class folders) plus metadata
files ``class_names.txt`` and ``norm_stats.json`` at the root.

Tensors are stored at 256x256. At load time:
  - Train: RandomCrop(crop_size) + horizontal flip + augmentations
  - Val: CenterCrop(crop_size)

Differences from OmniPretrainDataset:
  - Synchronized random erasing: when both RGB and depth are erased,
    they share the same box coordinates (prevents spatially impossible
    occlusion patterns that harm fusion layers).
  - val_loader num_workers bug fix: prefetch_factor and persistent_workers
    checks use the halved val worker count, not the original num_workers.
"""

import json
import os
import random
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F2

from src.training.augmentation_config import (
    # Probability baselines
    BASE_FLIP_P,
    BASE_COLOR_JITTER_P,
    BASE_BLUR_P,
    BASE_GRAYSCALE_P,
    BASE_RGB_ERASING_P,
    BASE_DEPTH_AUG_P,
    BASE_DEPTH_ERASING_P,
    BASE_DEPTH_SCALE_JITTER_P,
    BASE_DEPTH_SCALE_MIN,
    BASE_DEPTH_SCALE_MAX,
    BASE_HOLE_DROPOUT_P,
    BASE_HOLE_DROPOUT_NUM_MIN,
    BASE_HOLE_DROPOUT_NUM_MAX,
    BASE_HOLE_DROPOUT_SIZE_MIN,
    BASE_HOLE_DROPOUT_SIZE_MAX,
    # Magnitude baselines
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
    # Caps
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

    Handles both formats:
      - Plain: 'bathroom'
      - Indexed: '0: bathroom'

    Args:
        data_root: Root directory containing class_names.txt.

    Returns:
        List of class name strings in order.

    Raises:
        FileNotFoundError: If class_names.txt does not exist.
    """
    path = os.path.join(data_root, 'class_names.txt')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"class_names.txt not found in {data_root}. "
            f"Run the preprocessing script first."
        )
    names = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Handle "0: bathroom" format
            if ': ' in line and line.split(': ', 1)[0].strip().isdigit():
                names.append(line.split(': ', 1)[1])
            else:
                names.append(line)
    return names


def _load_norm_stats(data_root: str) -> dict:
    """Load normalization statistics from norm_stats.json in data_root.

    The JSON file must contain four keys with the following types:
      - ``rgb_mean``: ``list[float]`` (3 elements, channel means)
      - ``rgb_std``:  ``list[float]`` (3 elements, channel stds)
      - ``depth_mean``: ``list[float]`` (1 element, mean depth in meters)
      - ``depth_std``:  ``list[float]`` (1 element, depth std in meters)

    Args:
        data_root: Root directory containing norm_stats.json.

    Returns:
        Dict with keys: rgb_mean, rgb_std, depth_mean, depth_std.

    Raises:
        FileNotFoundError: If norm_stats.json does not exist.
    """
    path = os.path.join(data_root, 'norm_stats.json')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"norm_stats.json not found in {data_root}. "
            f"Run the preprocessing script with stats computation first."
        )
    with open(path, 'r') as f:
        return json.load(f)


def _discover_samples(
    data_root: str,
    class_names: list[str],
) -> list[tuple[str, str, int]]:
    """Walk class folders and discover paired RGB/depth sample files.

    Args:
        data_root: Root directory with class subfolders.
        class_names: Canonical class list from class_names.txt.

    Returns:
        List of (rgb_path, depth_path, label) tuples sorted by
        (class_name, rgb_filename) for deterministic ordering.

    Raises:
        ValueError: If unpaired rgb/depth files are found.
    """
    class_to_label = {name: i for i, name in enumerate(class_names)}
    samples = []

    for folder_name in sorted(os.listdir(data_root)):
        folder_path = os.path.join(data_root, folder_name)
        if not os.path.isdir(folder_path):
            continue
        if folder_name not in class_to_label:
            continue

        label = class_to_label[folder_name]

        rgb_files = {}
        depth_files = {}

        for fname in os.listdir(folder_path):
            if not fname.endswith('.pt'):
                continue
            if fname.endswith('_rgb.pt'):
                stem = fname[:-len('_rgb.pt')]
                rgb_files[stem] = os.path.join(folder_path, fname)
            elif fname.endswith('_depth.pt'):
                stem = fname[:-len('_depth.pt')]
                depth_files[stem] = os.path.join(folder_path, fname)

        rgb_only = set(rgb_files.keys()) - set(depth_files.keys())
        depth_only = set(depth_files.keys()) - set(rgb_files.keys())
        if rgb_only or depth_only:
            msg_parts = []
            if rgb_only:
                msg_parts.append(
                    f"RGB without depth: {sorted(rgb_only)[:5]}"
                )
            if depth_only:
                msg_parts.append(
                    f"Depth without RGB: {sorted(depth_only)[:5]}"
                )
            raise ValueError(
                f"Unpaired files in {folder_path}: {'; '.join(msg_parts)}"
            )

        for stem in sorted(rgb_files.keys()):
            samples.append((rgb_files[stem], depth_files[stem], label))

    return samples


class ScanNetPretrainDataset(Dataset):
    """
    ScanNet pretraining dataset for LINet.

    Loads per-sample .pt tensor files from a pre-split class folder layout.
    Each sample has paired *_rgb.pt and *_depth.pt files.

    Same augmentation pipeline as OmniPretrainDataset except random erasing
    is spatially synchronized between RGB and depth (same box coordinates
    when both modalities are erased).

    Tensor format:
    - RGB: uint8 [3, 256, 256] on disk, float32 [3, crop_size, crop_size] returned
    - Depth: uint16 [1, 256, 256] on disk (mm), float32 [1, crop_size, crop_size] (meters)
    """

    VALID_SPLITS = ('train', 'val')

    def __init__(
        self,
        data_root: str,
        split: str,
        samples: list[tuple[str, str, int]],
        class_names: list[str],
        norm_stats: dict,
        crop_size: int = 224,
        normalize: bool = True,
        rgb_aug_prob: float = 1.0,
        rgb_aug_mag: float = 1.0,
        depth_aug_prob: float = 1.0,
        depth_aug_mag: float = 1.0,
    ):
        if split not in self.VALID_SPLITS:
            raise ValueError(
                f"split must be one of {self.VALID_SPLITS}, got '{split}'"
            )

        self.data_root = data_root
        self.split = split
        self.crop_size = crop_size
        self.normalize = normalize

        self.CLASS_NAMES = class_names
        self.num_classes = len(class_names)
        self._norm_stats = norm_stats

        self.samples = samples
        self.labels = [s[2] for s in samples]
        self.num_samples = len(samples)

        self.rgb_aug_prob = rgb_aug_prob
        self.rgb_aug_mag = rgb_aug_mag
        self.depth_aug_prob = depth_aug_prob
        self.depth_aug_mag = depth_aug_mag

        self._compute_scaled_aug_values()

        if split == 'train' and any(
            p != 1.0
            for p in [rgb_aug_prob, rgb_aug_mag, depth_aug_prob, depth_aug_mag]
        ):
            self._log_augmentation_config()

    def __len__(self) -> int:
        return self.num_samples

    def _compute_scaled_aug_values(self):
        """Pre-compute scaled augmentation values based on aug_prob and aug_mag."""
        sync_prob = (self.rgb_aug_prob + self.depth_aug_prob) / 2

        # === SYNCHRONIZED (flip) ===
        self._flip_p = min(BASE_FLIP_P * sync_prob, MAX_PROBABILITY)

        # === RGB-ONLY ===
        self._color_jitter_p = min(
            BASE_COLOR_JITTER_P * self.rgb_aug_prob, MAX_PROBABILITY
        )
        self._brightness = min(BASE_BRIGHTNESS * self.rgb_aug_mag, MAX_BRIGHTNESS)
        self._contrast = min(BASE_CONTRAST * self.rgb_aug_mag, MAX_CONTRAST)
        self._saturation = min(BASE_SATURATION * self.rgb_aug_mag, MAX_SATURATION)
        self._hue = min(BASE_HUE * self.rgb_aug_mag, MAX_HUE)

        self._blur_p = min(BASE_BLUR_P * self.rgb_aug_prob, MAX_PROBABILITY)
        self._blur_sigma_min = BASE_BLUR_SIGMA_MIN
        self._blur_sigma_max = min(
            BASE_BLUR_SIGMA_MAX * self.rgb_aug_mag, MAX_BLUR_SIGMA
        )

        self._grayscale_p = min(
            BASE_GRAYSCALE_P * self.rgb_aug_prob, MAX_PROBABILITY
        )

        self._rgb_erasing_p = min(
            BASE_RGB_ERASING_P * self.rgb_aug_prob, MAX_PROBABILITY
        )
        self._rgb_erasing_scale_min = BASE_ERASING_SCALE_MIN
        self._rgb_erasing_scale_max = min(
            BASE_ERASING_SCALE_MAX * self.rgb_aug_mag, MAX_ERASING_SCALE
        )

        # === DEPTH-ONLY ===
        self._depth_aug_p = min(
            BASE_DEPTH_AUG_P * self.depth_aug_prob, MAX_PROBABILITY
        )
        self._depth_brightness = min(
            BASE_DEPTH_BRIGHTNESS * self.depth_aug_mag, MAX_DEPTH_BRIGHTNESS
        )
        self._depth_contrast = min(
            BASE_DEPTH_CONTRAST * self.depth_aug_mag, MAX_DEPTH_CONTRAST
        )
        self._depth_noise_std = min(
            BASE_DEPTH_NOISE_STD * self.depth_aug_mag, MAX_DEPTH_NOISE_STD
        )

        self._depth_erasing_p = min(
            BASE_DEPTH_ERASING_P * self.depth_aug_prob, MAX_PROBABILITY
        )
        self._depth_erasing_scale_min = BASE_ERASING_SCALE_MIN
        self._depth_erasing_scale_max = min(
            BASE_ERASING_SCALE_MAX * self.depth_aug_mag, MAX_ERASING_SCALE
        )

        # === DEPTH SCALE JITTER ===
        self._depth_scale_jitter_p = min(
            BASE_DEPTH_SCALE_JITTER_P * self.depth_aug_prob, MAX_PROBABILITY
        )

        # === DEPTH HOLE DROPOUT ===
        self._hole_dropout_p = min(
            BASE_HOLE_DROPOUT_P * self.depth_aug_prob, MAX_PROBABILITY
        )

        # === PRE-CREATE REUSABLE TRANSFORM INSTANCES ===
        self._color_jitter_transform = v2.ColorJitter(
            brightness=self._brightness,
            contrast=self._contrast,
            saturation=self._saturation,
            hue=self._hue,
        )
        # Single erasing transform for parameter sampling (shared coordinates)
        self._erasing_scale = (
            self._rgb_erasing_scale_min, self._rgb_erasing_scale_max
        )
        self._erasing_ratio = (BASE_ERASING_RATIO_MIN, BASE_ERASING_RATIO_MAX)

    def _log_augmentation_config(self):
        """Log computed augmentation values when scaling is applied."""
        print(f"\nScanNetPretrain augmentation scaling applied:")
        print(f"  RGB:   prob={self.rgb_aug_prob:.2f}, mag={self.rgb_aug_mag:.2f}")
        print(f"  Depth: prob={self.depth_aug_prob:.2f}, mag={self.depth_aug_mag:.2f}")
        print(f"  Computed values:")
        print(f"    [Sync]  Flip prob: {BASE_FLIP_P:.2f} -> {self._flip_p:.3f}")
        print(f"    [RGB]   ColorJitter prob: {BASE_COLOR_JITTER_P:.2f} -> {self._color_jitter_p:.3f}")
        print(f"    [RGB]   Brightness: +/-{BASE_BRIGHTNESS:.2f} -> +/-{self._brightness:.3f}")
        print(f"    [RGB]   Blur prob: {BASE_BLUR_P:.2f} -> {self._blur_p:.3f}")
        print(f"    [RGB]   Grayscale prob: {BASE_GRAYSCALE_P:.2f} -> {self._grayscale_p:.3f}")
        print(f"    [RGB]   Erasing prob: {BASE_RGB_ERASING_P:.2f} -> {self._rgb_erasing_p:.3f}")
        print(f"    [Depth] Aug prob: {BASE_DEPTH_AUG_P:.2f} -> {self._depth_aug_p:.3f}")
        print(f"    [Depth] Brightness: +/-{BASE_DEPTH_BRIGHTNESS:.2f} -> +/-{self._depth_brightness:.3f}")
        print(f"    [Depth] Noise std: {BASE_DEPTH_NOISE_STD:.3f} -> {self._depth_noise_std:.3f}")
        print(f"    [Depth] Erasing prob: {BASE_DEPTH_ERASING_P:.2f} -> {self._depth_erasing_p:.3f}")
        print(f"    [Depth] Scale jitter prob: {BASE_DEPTH_SCALE_JITTER_P:.2f} -> {self._depth_scale_jitter_p:.3f}")
        print(f"    [Depth] Hole dropout prob: {BASE_HOLE_DROPOUT_P:.2f} -> {self._hole_dropout_p:.3f}")

    def _apply_hole_dropout(self, depth: torch.Tensor) -> torch.Tensor:
        """Apply random rectangular hole dropout to depth tensor.

        Zeros out multiple small rectangular patches to simulate sensor
        missing data. Applied BEFORE normalization (sets pixels to 0 = sentinel).

        Modifies depth in-place and returns it.
        """
        _, h, w = depth.shape
        num_holes = np.random.randint(
            BASE_HOLE_DROPOUT_NUM_MIN, BASE_HOLE_DROPOUT_NUM_MAX + 1
        )
        for _ in range(num_holes):
            hole_h = np.random.randint(
                BASE_HOLE_DROPOUT_SIZE_MIN, BASE_HOLE_DROPOUT_SIZE_MAX + 1
            )
            hole_w = np.random.randint(
                BASE_HOLE_DROPOUT_SIZE_MIN, BASE_HOLE_DROPOUT_SIZE_MAX + 1
            )
            top = np.random.randint(0, max(1, h - hole_h + 1))
            left = np.random.randint(0, max(1, w - hole_w + 1))
            depth[:, top:top + hole_h, left:left + hole_w] = 0.0
        return depth

    def _sample_erase_params(self, img: torch.Tensor) -> tuple | None:
        """Sample random erasing parameters from a reference tensor.

        Returns (i, j, h, w, v) or None if sampling fails
        (e.g., no valid region found after max attempts).
        """
        try:
            i, j, h, w, v = v2.RandomErasing.get_params(
                img,
                scale=self._erasing_scale,
                ratio=self._erasing_ratio,
                value=[0.0],
            )
            return i, j, h, w, v
        except ValueError:
            return None

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Load and return a single sample.

        Returns:
            rgb: float32 [3, crop_size, crop_size]
            depth: float32 [1, crop_size, crop_size]
            label: int (0 to num_classes-1)
        """
        rgb_path, depth_path, label = self.samples[idx]

        rgb = torch.load(rgb_path, weights_only=True)
        depth = torch.load(depth_path, weights_only=True)

        assert rgb.shape[0] == 3 and rgb.ndim == 3, (
            f"Bad RGB shape {rgb.shape} at {rgb_path}"
        )
        assert depth.shape[0] == 1 and depth.ndim == 3, (
            f"Bad depth shape {depth.shape} at {depth_path}"
        )

        # Convert depth: uint16 mm -> float32 meters
        depth = depth.float() / 1000.0

        # ==================== TRAINING AUGMENTATION ====================
        if self.split == 'train':
            zero_mask = (depth == 0.0)

            # 1. Synchronized Random Horizontal Flip
            if np.random.random() < self._flip_p:
                rgb = F2.horizontal_flip(rgb)
                depth = F2.horizontal_flip(depth)
                zero_mask = zero_mask.flip(-1)

            # 2. Synchronized RandomCrop (256 -> crop_size)
            i, j, h, w = v2.RandomCrop.get_params(
                rgb, output_size=(self.crop_size, self.crop_size)
            )
            rgb = F2.crop(rgb, i, j, h, w)
            depth = F2.crop(depth, i, j, h, w)
            zero_mask = zero_mask[:, i:i+h, j:j+w]

            # 3-5. RGB-Only Appearance Augmentation
            if self.normalize:
                if np.random.random() < self._color_jitter_p:
                    rgb = self._color_jitter_transform(rgb)

                if np.random.random() < self._blur_p:
                    kernel_size = int(np.random.choice([3, 5, 7]))
                    sigma = float(
                        np.random.uniform(
                            self._blur_sigma_min, self._blur_sigma_max
                        )
                    )
                    rgb = F2.gaussian_blur(rgb, kernel_size=kernel_size, sigma=sigma)

                if np.random.random() < self._grayscale_p:
                    rgb = F2.rgb_to_grayscale(rgb, num_output_channels=3)

            # 6. Depth Scale Jitter
            if self.normalize and np.random.random() < self._depth_scale_jitter_p:
                scale_factor = np.random.uniform(
                    BASE_DEPTH_SCALE_MIN, BASE_DEPTH_SCALE_MAX
                )
                depth = depth * scale_factor

            # 7. Depth Appearance Augmentation
            if self.normalize and np.random.random() < self._depth_aug_p:
                d_min = depth[~zero_mask].min() if (~zero_mask).any() else 0.0
                d_max = depth[~zero_mask].max() if (~zero_mask).any() else 1.0
                d_range = d_max - d_min

                depth_01 = torch.zeros_like(depth)
                if d_range > 1e-6:
                    depth_01[~zero_mask] = (depth[~zero_mask] - d_min) / d_range

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
                    depth[~zero_mask] = depth_01[~zero_mask].clamp(0.0, 1.0) * d_range + d_min

            # 8. Random Hole Dropout
            if np.random.random() < self._hole_dropout_p:
                depth = self._apply_hole_dropout(depth)
                zero_mask = zero_mask | (depth == 0.0)

            # Restore original 0-sentinel pixels
            depth[zero_mask] = 0.0

        else:
            # Val: CenterCrop (256 -> crop_size)
            rgb = F2.center_crop(rgb, (self.crop_size, self.crop_size))
            depth = F2.center_crop(depth, (self.crop_size, self.crop_size))
            zero_mask = (depth == 0.0)

        # ==================== TO FLOAT32 ====================
        if rgb.dtype == torch.uint8:
            rgb = rgb.float() / 255.0

        # ==================== SENTINEL REPLACEMENT ====================
        depth_mean_val = self._norm_stats['depth_mean'][0]
        depth[zero_mask] = depth_mean_val

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

            # Post-normalization Synchronized Random Erasing (CPU mode only)
            if self.split == 'train':
                do_rgb_erase = np.random.random() < self._rgb_erasing_p
                do_depth_erase = np.random.random() < self._depth_erasing_p

                if do_rgb_erase or do_depth_erase:
                    # Sample erase params once from RGB (reference tensor)
                    params = self._sample_erase_params(rgb)
                    if params is not None:
                        ei, ej, eh, ew, ev = params
                        if do_rgb_erase:
                            rgb = F2.erase(rgb, ei, ej, eh, ew, ev)
                        if do_depth_erase:
                            depth = F2.erase(depth, ei, ej, eh, ew, ev)

        return rgb, depth, label

    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for weighted loss (inverse frequency)."""
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

    def get_sample_weights(self) -> torch.Tensor:
        """Calculate per-sample weights for WeightedRandomSampler."""
        label_counts = Counter(self.labels)
        sample_weights = torch.tensor(
            [1.0 / label_counts[label] for label in self.labels],
            dtype=torch.float64,
        )
        return sample_weights

    def get_class_distribution(self) -> dict[str, dict[str, int | float]]:
        """Get class distribution statistics."""
        label_counts = Counter(self.labels)
        distribution = {}
        for class_idx in range(self.num_classes):
            count = label_counts.get(class_idx, 0)
            percentage = (count / self.num_samples * 100) if self.num_samples > 0 else 0.0
            distribution[self.CLASS_NAMES[class_idx]] = {
                'count': count,
                'percentage': percentage,
            }
        return distribution

    def get_norm_stats(self) -> dict:
        """Return the normalization statistics dict loaded from norm_stats.json."""
        return self._norm_stats


class _WorkerInitFn:
    """Callable for DataLoader worker initialization (picklable)."""

    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, worker_id: int):
        worker_seed = self.seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)


def get_scannet_pretrain_dataloaders(
    data_root: str = 'data/scannet_pretrain_256',
    batch_size: int = 32,
    num_workers: int = 4,
    crop_size: int = 224,
    use_class_weights: bool = False,
    seed: int = 42,
    normalize: bool = True,
    balanced_sampling: bool = True,
    rgb_aug_prob: float = 1.0,
    rgb_aug_mag: float = 1.0,
    depth_aug_prob: float = 1.0,
    depth_aug_mag: float = 1.0,
) -> tuple:
    """Create train and val dataloaders for ScanNet pretraining.

    Same return signature as get_omnipretrain_dataloaders:
      (train_loader, val_loader, num_classes) if use_class_weights=False
      (train_loader, val_loader, num_classes, class_weights) if True

    Bug fix over Omni: val_workers uses halved count for
    persistent_workers and prefetch_factor guards.
    """
    class_names = _load_class_names(data_root)
    norm_stats = _load_norm_stats(data_root)

    train_dir = os.path.join(data_root, 'train')
    val_dir = os.path.join(data_root, 'val')

    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        raise FileNotFoundError(
            f"Expected train/ and val/ subdirectories in {data_root}. "
            f"Run the preprocessing notebook to create the split."
        )

    train_samples = _discover_samples(train_dir, class_names)
    val_samples = _discover_samples(val_dir, class_names)

    if len(train_samples) == 0:
        raise ValueError(f"No training samples found in {train_dir}")
    if len(val_samples) == 0:
        raise ValueError(f"No validation samples found in {val_dir}")

    train_dataset = ScanNetPretrainDataset(
        data_root=data_root,
        split='train',
        samples=train_samples,
        class_names=class_names,
        norm_stats=norm_stats,
        crop_size=crop_size,
        normalize=normalize,
        rgb_aug_prob=rgb_aug_prob,
        rgb_aug_mag=rgb_aug_mag,
        depth_aug_prob=depth_aug_prob,
        depth_aug_mag=depth_aug_mag,
    )
    val_dataset = ScanNetPretrainDataset(
        data_root=data_root,
        split='val',
        samples=val_samples,
        class_names=class_names,
        norm_stats=norm_stats,
        crop_size=crop_size,
        normalize=normalize,
    )

    num_classes = len(class_names)

    worker_init_fn = _WorkerInitFn(seed)

    if balanced_sampling:
        sample_weights = train_dataset.get_sample_weights()
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
        train_shuffle = False
    else:
        train_sampler = None
        train_shuffle = True

    # Train loader
    train_prefetch = 4 if num_workers > 0 else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        prefetch_factor=train_prefetch,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        worker_init_fn=worker_init_fn,
    )

    # Val loader — bug fix: use val_workers for all guards
    val_workers = max(num_workers // 2, 0)
    val_prefetch = 2 if val_workers > 0 else None
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=val_workers,
        prefetch_factor=val_prefetch,
        pin_memory=True,
        persistent_workers=True if val_workers > 0 else False,
        worker_init_fn=worker_init_fn,
    )

    sampling_mode = "weighted" if balanced_sampling else "shuffle"
    total = len(train_dataset) + len(val_dataset)
    print(f"\nScanNetPretrain Dataset:")
    print(f"  Total samples: {total}")
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print(f"  Classes: {num_classes}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sampling: {sampling_mode}")

    if use_class_weights:
        class_weights = train_dataset.get_class_weights()
        print(f"  Class weights computed (inverse frequency)")
        return train_loader, val_loader, num_classes, class_weights

    return train_loader, val_loader, num_classes
