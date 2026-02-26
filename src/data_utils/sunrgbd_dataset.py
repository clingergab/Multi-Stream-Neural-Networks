"""
SUN RGB-D Dataset Loader for 15-category scene classification.

Loads preprocessed SUN RGB-D dataset with RGB and depth images.
"""

import os
import random
from collections import Counter

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms

from src.training.augmentation_config import (
    # Probability baselines
    BASE_FLIP_P,
    BASE_CROP_P,
    BASE_COLOR_JITTER_P,
    BASE_BLUR_P,
    BASE_GRAYSCALE_P,
    BASE_RGB_ERASING_P,
    BASE_DEPTH_AUG_P,
    BASE_DEPTH_ERASING_P,
    # Magnitude baselines
    BASE_CROP_SCALE_MIN,
    BASE_CROP_SCALE_MAX,
    BASE_CROP_RATIO_MIN,
    BASE_CROP_RATIO_MAX,
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
    MIN_CROP_SCALE,
    MAX_ERASING_SCALE,
)


class SUNRGBDDataset(Dataset):
    """
    SUN RGB-D dataset for scene classification (15 categories).

    Directory structure:
        data_root/
            train/ or val/ or test/
                rgb/
                    00000.png
                    00001.png
                    ...
                depth/
                    00000.png
                    00001.png
                    ...
                labels.txt
    """

    VALID_SPLITS = ('train', 'val', 'test')

    # 15 scene categories
    CLASS_NAMES = [
        'bathroom', 'bedroom', 'classroom', 'computer_room', 'corridor',
        'dining_area', 'dining_room', 'discussion_area', 'furniture_store',
        'kitchen', 'lab', 'library', 'office', 'rest_space', 'study_space'
    ]

    def __init__(
        self,
        data_root='data/sunrgbd_15',
        split='train',
        target_size=(416, 544),
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
            target_size: Target image size (H, W)
            normalize: If True, apply normalization in __getitem__().
                      Set to False when using GPU augmentation (which handles
                      normalization on GPU after augmentation).
            rgb_aug_prob: Scales probability of RGB augmentations (default: 1.0 = baseline)
            rgb_aug_mag: Scales magnitude of RGB augmentations (default: 1.0 = baseline)
            depth_aug_prob: Scales probability of Depth augmentations (default: 1.0 = baseline)
            depth_aug_mag: Scales magnitude of Depth augmentations (default: 1.0 = baseline)

        Note: All augmentation is performed in __getitem__() to enable synchronized
        transforms between RGB and depth modalities.
        """
        if split not in self.VALID_SPLITS:
            raise ValueError(f"split must be one of {self.VALID_SPLITS}, got '{split}'")

        self.data_root = data_root
        self.split = split
        self.target_size = target_size
        self.normalize = normalize

        # Store augmentation scaling parameters
        self.rgb_aug_prob = rgb_aug_prob
        self.rgb_aug_mag = rgb_aug_mag
        self.depth_aug_prob = depth_aug_prob
        self.depth_aug_mag = depth_aug_mag

        # Pre-compute scaled augmentation values (avoids repeated computation in __getitem__)
        self._compute_scaled_aug_values()

        # Set split directory
        self.split_dir = os.path.join(data_root, split)

        # Load labels
        labels_file = os.path.join(self.split_dir, 'labels.txt')
        with open(labels_file, 'r') as f:
            self.labels = [int(line.strip()) for line in f.readlines()]

        self.num_samples = len(self.labels)

        # RGB and depth directories
        self.rgb_dir = os.path.join(self.split_dir, 'rgb')
        self.depth_dir = os.path.join(self.split_dir, 'depth')

        # Verify directories exist
        assert os.path.exists(self.rgb_dir), f"RGB directory not found: {self.rgb_dir}"
        assert os.path.exists(self.depth_dir), f"Depth directory not found: {self.depth_dir}"

        print(f"Loaded SUN RGB-D {split} set: {self.num_samples} samples, 15 classes")

        # Log augmentation config if scaling is applied
        if split == 'train' and any(p != 1.0 for p in [rgb_aug_prob, rgb_aug_mag, depth_aug_prob, depth_aug_mag]):
            self._log_augmentation_config()

    def __len__(self):
        return self.num_samples

    def _compute_scaled_aug_values(self):
        """Pre-compute scaled augmentation values based on aug_prob and aug_mag parameters."""
        # Synchronized augmentations use average of RGB and Depth params
        sync_prob = (self.rgb_aug_prob + self.depth_aug_prob) / 2
        sync_mag = (self.rgb_aug_mag + self.depth_aug_mag) / 2

        # === SYNCHRONIZED (flip, crop) ===
        self._flip_p = min(BASE_FLIP_P * sync_prob, MAX_PROBABILITY)
        self._crop_p = min(BASE_CROP_P * sync_prob, MAX_PROBABILITY)
        # Crop scale: higher mag = more aggressive crop (lower min scale)
        self._crop_scale_min = max(
            MIN_CROP_SCALE,
            1.0 - (1.0 - BASE_CROP_SCALE_MIN) * sync_mag
        )
        self._crop_scale_max = BASE_CROP_SCALE_MAX
        self._crop_ratio_min = BASE_CROP_RATIO_MIN
        self._crop_ratio_max = BASE_CROP_RATIO_MAX

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

    def _log_augmentation_config(self):
        """Log computed augmentation values when scaling is applied."""
        print(f"\nAugmentation scaling applied:")
        print(f"  RGB:   prob={self.rgb_aug_prob:.2f}, mag={self.rgb_aug_mag:.2f}")
        print(f"  Depth: prob={self.depth_aug_prob:.2f}, mag={self.depth_aug_mag:.2f}")
        print(f"  Computed values:")
        print(f"    [Sync]  Flip prob: {BASE_FLIP_P:.2f} -> {self._flip_p:.3f}")
        print(f"    [Sync]  Crop prob: {BASE_CROP_P:.2f} -> {self._crop_p:.3f}")
        print(f"    [Sync]  Crop scale min: {BASE_CROP_SCALE_MIN:.2f} -> {self._crop_scale_min:.3f}")
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
            rgb: RGB image tensor [3, H, W]
            depth: Depth image tensor [1, H, W]
            label: Class label (0-14)
        """
        # Load RGB image
        rgb_path = os.path.join(self.rgb_dir, f'{idx:05d}.png')
        rgb = Image.open(rgb_path).convert('RGB')

        # Load depth image
        depth_path = os.path.join(self.depth_dir, f'{idx:05d}.png')
        depth = Image.open(depth_path)

        # Convert Depth to Mode F (Float32) with global normalization
        # Use global normalization to maintain semantic consistency across images
        # (same depth value should normalize to same output value regardless of image content)
        if depth.mode in ('I', 'I;16', 'I;16B'):
            depth_arr = np.array(depth, dtype=np.float32)
            # Global normalization: divide by max possible depth value (16-bit)
            # Using 65535.0 to cover full range and avoid clipping (max observed: 65528)
            depth_arr = np.clip(depth_arr / 65535.0, 0.0, 1.0)
            depth = Image.fromarray(depth_arr, mode='F')
        else:
            # Fallback for other modes
            depth = depth.convert('F')
            # If it was 0-255, scale to 0-1
            if np.array(depth).max() > 1.0:
                depth_arr = np.array(depth)
                depth = Image.fromarray(depth_arr / 255.0, mode='F')

        # ==================== TRAINING AUGMENTATION ====================
        if self.split == 'train':
            # 1. Synchronized Random Horizontal Flip
            # Uses pre-computed _flip_p (scaled by sync_prob)
            if np.random.random() < self._flip_p:
                rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)
                depth = depth.transpose(Image.FLIP_LEFT_RIGHT)

            # 2. Synchronized Random Resized Crop
            # Uses pre-computed _crop_p and _crop_scale_min (scaled by sync_prob/sync_mag)
            # Scene classification needs context, so crop is probabilistic
            if np.random.random() < self._crop_p:
                i, j, h, w = transforms.RandomResizedCrop.get_params(
                    rgb,
                    scale=(self._crop_scale_min, self._crop_scale_max),
                    ratio=(self._crop_ratio_min, self._crop_ratio_max)
                )
                rgb = transforms.functional.resized_crop(rgb, i, j, h, w, self.target_size)
                depth = transforms.functional.resized_crop(depth, i, j, h, w, self.target_size)
            else:
                # No crop - preserve full scene context
                rgb = transforms.functional.resize(rgb, self.target_size)
                depth = transforms.functional.resize(depth, self.target_size)

            # 3-5. RGB-Only Appearance Augmentation (ColorJitter, Blur, Grayscale)
            # Skip these when normalize=False (GPU augmentation mode) because
            # GPUAugmentation will apply them on GPU after transfer.
            if self.normalize:
                # 3. RGB-Only: Color Jitter
                # Uses pre-computed _color_jitter_p and magnitude values (scaled by rgb_aug_prob/mag)
                if np.random.random() < self._color_jitter_p:
                    color_jitter = transforms.ColorJitter(
                        brightness=self._brightness,
                        contrast=self._contrast,
                        saturation=self._saturation,
                        hue=self._hue
                    )
                    rgb = color_jitter(rgb)

                # 4. RGB-Only: Gaussian Blur
                # Uses pre-computed _blur_p and _blur_sigma_max (scaled by rgb_aug_prob/mag)
                if np.random.random() < self._blur_p:
                    kernel_size = int(np.random.choice([3, 5, 7]))
                    sigma = float(np.random.uniform(self._blur_sigma_min, self._blur_sigma_max))
                    rgb = transforms.functional.gaussian_blur(rgb, kernel_size=kernel_size, sigma=sigma)

                # 5. RGB-Only: Occasional Grayscale
                # Uses pre-computed _grayscale_p (scaled by rgb_aug_prob)
                if np.random.random() < self._grayscale_p:
                    rgb = transforms.functional.to_grayscale(rgb, num_output_channels=3)

            # 6. Depth-Only: Combined Appearance Augmentation
            # Uses pre-computed _depth_aug_p and magnitude values (scaled by depth_aug_prob/mag)
            if np.random.random() < self._depth_aug_p:
                depth_array = np.array(depth, dtype=np.float32)

                # Apply brightness and contrast using pre-computed scaled magnitudes
                brightness_factor = np.random.uniform(
                    1.0 - self._depth_brightness,
                    1.0 + self._depth_brightness
                )
                contrast_factor = np.random.uniform(
                    1.0 - self._depth_contrast,
                    1.0 + self._depth_contrast
                )

                # Apply contrast then brightness (same order as ColorJitter)
                # Note: depth_array is in [0, 1] range, so use 0.5 as midpoint
                depth_array = (depth_array - 0.5) * contrast_factor + 0.5
                depth_array = depth_array * brightness_factor

                # Add Gaussian noise using pre-computed scaled std
                noise = np.random.normal(0, self._depth_noise_std, depth_array.shape).astype(np.float32)
                depth_array = depth_array + noise

                depth_array = np.clip(depth_array, 0.0, 1.0).astype(np.float32)
                depth = Image.fromarray(depth_array, mode='F')

        else:
            # Validation: just resize (no augmentation)
            rgb = transforms.functional.resize(rgb, self.target_size)
            depth = transforms.functional.resize(depth, self.target_size)

        # ==================== TO TENSOR ====================
        # Convert to tensor (always needed)
        rgb = transforms.functional.to_tensor(rgb)
        depth = transforms.functional.to_tensor(depth)

        # ==================== NORMALIZATION ====================
        # Statistics computed from training samples at (416, 544) resolution
        # after scaling to [0, 1] range
        #
        # When normalize=False (GPU augmentation mode), skip normalization here.
        # GPU augmentation will handle normalization after applying augmentations.
        if self.normalize:
            # RGB: Use exact computed training statistics (official split)
            rgb = transforms.functional.normalize(
                rgb,
                mean=[0.4974685511366709, 0.4657685752251157, 0.4418713446646282],
                std=[0.2772972605813588, 0.2859611184863525, 0.2896814863955933]
            )

            # Depth: Use exact computed training statistics (official split)
            depth = transforms.functional.normalize(
                depth, mean=[0.2911], std=[0.1514]
            )

            # 7. Post-normalization Random Erasing (CPU mode only)
            # When using GPU augmentation, erasing is done on GPU after normalization
            # Uses pre-computed _rgb_erasing_p and _depth_erasing_p (scaled by aug_prob/mag)
            if self.split == 'train':
                # RGB random erasing
                if np.random.random() < self._rgb_erasing_p:
                    erasing = transforms.RandomErasing(
                        p=1.0,
                        scale=(self._rgb_erasing_scale_min, self._rgb_erasing_scale_max),
                        ratio=(BASE_ERASING_RATIO_MIN, BASE_ERASING_RATIO_MAX)
                    )
                    rgb = erasing(rgb)

                # Depth random erasing
                if np.random.random() < self._depth_erasing_p:
                    erasing = transforms.RandomErasing(
                        p=1.0,
                        scale=(self._depth_erasing_scale_min, self._depth_erasing_scale_max),
                        ratio=(BASE_ERASING_RATIO_MIN, BASE_ERASING_RATIO_MAX)
                    )
                    depth = erasing(depth)

        # Get label
        label = self.labels[idx]

        return rgb, depth, label

    def get_class_weights(self):
        """
        Calculate class weights for weighted loss (inverse frequency).

        Returns:
            Tensor of shape [num_classes] with weights
        """
        label_counts = Counter(self.labels)

        weights = torch.zeros(len(self.CLASS_NAMES))
        total = len(self.labels)

        for class_idx in range(len(self.CLASS_NAMES)):
            count = label_counts.get(class_idx, 0)
            if count > 0:
                weights[class_idx] = total / (len(self.CLASS_NAMES) * count)
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
        for class_idx in range(len(self.CLASS_NAMES)):
            count = label_counts.get(class_idx, 0)
            percentage = (count / self.num_samples) * 100
            distribution[self.CLASS_NAMES[class_idx]] = {
                'count': count,
                'percentage': percentage
            }

        return distribution


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


def get_sunrgbd_dataloaders(
    data_root='data/sunrgbd_15',
    batch_size=32,
    num_workers=4,
    target_size=(416, 544),
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
    Create train, validation, and test dataloaders for SUN RGB-D.

    Args:
        data_root: Root directory of preprocessed dataset
        batch_size: Batch size
        num_workers: Number of dataloader workers
        target_size: Target image size (H, W)
        use_class_weights: If True, return class weights for loss
        stratified: If True, use stratified sampling for training to ensure
                   balanced class representation in each batch. Recommended for
                   imbalanced datasets. Note: this oversamples minority classes.
        seed: Random seed for reproducible data loading. If None, non-reproducible.
              When set, ensures reproducible shuffle order and worker initialization.
        normalize: If True, apply normalization in dataset __getitem__().
                  Set to False when using GPU augmentation (which handles
                  normalization on GPU after augmentation).
        rgb_aug_prob: Scales probability of RGB augmentations (default: 1.0 = baseline)
        rgb_aug_mag: Scales magnitude of RGB augmentations (default: 1.0 = baseline)
        depth_aug_prob: Scales probability of Depth augmentations (default: 1.0 = baseline)
        depth_aug_mag: Scales magnitude of Depth augmentations (default: 1.0 = baseline)

    Returns:
        train_loader, val_loader, test_loader, (optional) class_weights

    Example:
        >>> # Reproducible dataloaders with stratified sampling
        >>> from src.utils.seed import set_seed
        >>> set_seed(42)
        >>> train_loader, val_loader, test_loader = get_sunrgbd_dataloaders(seed=42, stratified=True)
        >>>
        >>> # For GPU augmentation mode with custom augmentation scaling
        >>> from src.training.augmentation_config import AugmentationConfig
        >>> aug_config = AugmentationConfig(rgb_aug_prob=1.5, rgb_aug_mag=1.2)
        >>> train_loader, val_loader, test_loader = get_sunrgbd_dataloaders(
        ...     normalize=False,
        ...     **aug_config.to_dict()
        ... )
    """
    # Create datasets
    # Note: Augmentation params only affect training set (split='train')
    # Val and test sets ignore these params (no augmentation applied)
    train_dataset = SUNRGBDDataset(
        data_root=data_root,
        split='train',
        target_size=target_size,
        normalize=normalize,
        rgb_aug_prob=rgb_aug_prob,
        rgb_aug_mag=rgb_aug_mag,
        depth_aug_prob=depth_aug_prob,
        depth_aug_mag=depth_aug_mag,
    )

    val_dataset = SUNRGBDDataset(
        data_root=data_root,
        split='val',
        target_size=target_size,
        normalize=normalize,
    )

    test_dataset = SUNRGBDDataset(
        data_root=data_root,
        split='test',
        target_size=target_size,
        normalize=normalize,
    )

    # Setup reproducibility if seed is provided
    worker_init_fn = None
    generator = None

    if seed is not None:
        worker_init_fn = _WorkerInitFn(seed)
        generator = torch.Generator().manual_seed(seed)

    # Setup stratified sampling if requested
    train_sampler = None
    train_shuffle = True

    if stratified:
        # Compute sample weights (inverse class frequency)
        label_counts = Counter(train_dataset.labels)
        num_samples = len(train_dataset.labels)

        # Weight for each sample = 1 / (number of samples in that class)
        # This makes each class equally likely to be sampled
        sample_weights = [1.0 / label_counts[label] for label in train_dataset.labels]
        sample_weights = torch.tensor(sample_weights, dtype=torch.float64)

        # Create sampler - replacement=True allows oversampling minority classes
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=num_samples,  # Same epoch size as original
            replacement=True,
            generator=generator
        )
        train_shuffle = False  # Sampler handles randomization

        # Print stratification info
        print(f"\nStratified sampling enabled (training only):")
        print(f"  Train class imbalance: {max(label_counts.values())/min(label_counts.values()):.1f}x")
        print(f"  Each training batch will have balanced class representation")

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        prefetch_factor=4,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        worker_init_fn=worker_init_fn,
        generator=generator if train_sampler is None else None  # Generator used by sampler
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=2,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=2,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    print(f"\nDataLoader Info:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
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

    # Test dataset loading
    print("Testing SUN RGB-D Dataset Loader")
    print("=" * 80)

    train_dataset = SUNRGBDDataset(split='train')
    val_dataset = SUNRGBDDataset(split='val')
    test_dataset = SUNRGBDDataset(split='test')

    print(f"\nTrain set: {len(train_dataset)} samples")
    print(f"Val set: {len(val_dataset)} samples")
    print(f"Test set: {len(test_dataset)} samples")

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
    train_loader, val_loader, test_loader = get_sunrgbd_dataloaders(batch_size=16, num_workers=0)

    # Get one batch
    rgb_batch, depth_batch, labels_batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  RGB: {rgb_batch.shape}")
    print(f"  Depth: {depth_batch.shape}")
    print(f"  Labels: {labels_batch.shape}")

    print("\n✓ Dataset loader working correctly!")
