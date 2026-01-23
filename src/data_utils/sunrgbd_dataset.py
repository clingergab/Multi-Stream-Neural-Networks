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


class SUNRGBDDataset(Dataset):
    """
    SUN RGB-D dataset for scene classification (15 categories).

    Directory structure:
        data_root/
            train/ or val/
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

    # 15 scene categories
    CLASS_NAMES = [
        'bathroom', 'bedroom', 'classroom', 'computer_room', 'corridor',
        'dining_area', 'dining_room', 'discussion_area', 'furniture_store',
        'kitchen', 'lab', 'library', 'office', 'rest_space', 'study_space'
    ]

    def __init__(
        self,
        data_root='data/sunrgbd_15',
        train=True,
        target_size=(416, 544),
        normalize: bool = True,
    ):
        """
        Args:
            data_root: Root directory of preprocessed dataset
            train: If True, load training set; else load validation set
            target_size: Target image size (H, W)
            normalize: If True, apply normalization in __getitem__().
                      Set to False when using GPU augmentation (which handles
                      normalization on GPU after augmentation).

        Note: All augmentation is performed in __getitem__() to enable synchronized
        transforms between RGB and depth modalities.
        """
        self.data_root = data_root
        self.train = train
        self.target_size = target_size
        self.normalize = normalize

        # Set split directory
        split = 'train' if train else 'val'
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

    def __len__(self):
        return self.num_samples

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
        if self.train:
            # 1. Synchronized Random Horizontal Flip (50%)
            if np.random.random() < 0.5:
                rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)
                depth = depth.transpose(Image.FLIP_LEFT_RIGHT)

            # 2. Synchronized Random Resized Crop (50% probability)
            # IMPORTANT: Scene classification needs context, so crop is probabilistic
            # 50% get crop (scale variance), 50% get full scene (context preserved)
            if np.random.random() < 0.5:
                i, j, h, w = transforms.RandomResizedCrop.get_params(
                    rgb, scale=(0.9, 1.0), ratio=(0.95, 1.05)  # Gentle crop 90-100%
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
                # 3. RGB-Only: Color Jitter (43% probability - BALANCED for 1.5x ratio)
                # Applies brightness, contrast, saturation, hue adjustments
                # Balanced augmentation to achieve ~1.5x RGB/Depth ratio (was 75%)
                if np.random.random() < 0.43:
                    color_jitter = transforms.ColorJitter(
                        brightness=0.37,  # ±37% (balanced for 1.5x ratio)
                        contrast=0.37,    # ±37% (balanced for 1.5x ratio)
                        saturation=0.37,  # ±37% (balanced for 1.5x ratio)
                        hue=0.11          # ±11% (balanced for 1.5x ratio)
                    )
                    rgb = color_jitter(rgb)

                # 4. RGB-Only: Gaussian Blur (25% probability - BALANCED for 1.5x ratio)
                # Reduces reliance on fine textures/edges, forces focus on spatial structure
                # Balanced augmentation to achieve ~1.5x RGB/Depth ratio (was 50%)
                if np.random.random() < 0.25:
                    # Kernel size: random odd number between 3, 5, and 7
                    # Sigma: random between 0.1 and 1.7 (balanced for 1.5x ratio)
                    kernel_size = int(np.random.choice([3, 5, 7]))
                    sigma = float(np.random.uniform(0.1, 1.7))
                    rgb = transforms.functional.gaussian_blur(rgb, kernel_size=kernel_size, sigma=sigma)

                # 5. RGB-Only: Occasional Grayscale (17% - BALANCED for 1.5x ratio)
                # Forces RGB stream to learn from structure, not just color
                # Balanced augmentation to achieve ~1.5x RGB/Depth ratio (was 35%)
                if np.random.random() < 0.17:
                    rgb = transforms.functional.to_grayscale(rgb, num_output_channels=3)

            # 6. Depth-Only: Combined Appearance Augmentation (50% probability)
            # IMPORTANT: Matches RGB's color jitter - single augmentation block
            # Applies brightness + contrast + noise together (like RGB does color jitter)
            if np.random.random() < 0.5:
                depth_array = np.array(depth, dtype=np.float32)

                # Apply brightness and contrast
                # Slightly higher than RGB (±25% vs ±20%) to compensate for:
                #   1. Depth having 1 channel vs RGB's 3 channels
                #   2. Reduced crop probability (50% vs previous 100%)
                brightness_factor = np.random.uniform(0.75, 1.25)  # ±25%
                contrast_factor = np.random.uniform(0.75, 1.25)    # ±25%

                # Apply contrast then brightness (same order as ColorJitter)
                # Note: depth_array is in [0, 1] range, so use 0.5 as midpoint
                depth_array = (depth_array - 0.5) * contrast_factor + 0.5
                depth_array = depth_array * brightness_factor

                # Add Gaussian noise (simulates sensor noise)
                # Moderate std to avoid excessive noise while providing robustness
                # Scale noise to [0, 1] range (15/255 ≈ 0.059)
                noise = np.random.normal(0, 0.059, depth_array.shape).astype(np.float32)
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
        # Statistics computed from 8041 training samples at (416, 544) resolution
        # after scaling to [0, 1] range
        #
        # When normalize=False (GPU augmentation mode), skip normalization here.
        # GPU augmentation will handle normalization after applying augmentations.
        if self.normalize:
            # RGB: Use exact computed training statistics
            rgb = transforms.functional.normalize(
                rgb,
                mean=[0.4905626144214781, 0.4564359471868703, 0.43112756716677114],
                std=[0.27944652961530003, 0.2868739703756949, 0.29222326115669395]
            )

            # Depth: Use exact computed training statistics
            depth = transforms.functional.normalize(
                depth, mean=[0.2912], std=[0.1472]
            )

            # 7. Post-normalization Random Erasing (CPU mode only)
            # When using GPU augmentation, erasing is done on GPU after normalization
            if self.train:
                # RGB random erasing (17% - BALANCED for 1.5x ratio)
                if np.random.random() < 0.17:
                    erasing = transforms.RandomErasing(
                        p=1.0,
                        scale=(0.02, 0.10),    # Small patches (2-10% of image)
                        ratio=(0.5, 2.0)       # Reasonable aspect ratios
                    )
                    rgb = erasing(rgb)

                # Depth random erasing (10% - separate and equal)
                if np.random.random() < 0.1:
                    erasing = transforms.RandomErasing(
                        p=1.0,
                        scale=(0.02, 0.1),
                        ratio=(0.5, 2.0)
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
):
    """
    Create train and validation dataloaders for SUN RGB-D.

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

    Returns:
        train_loader, val_loader, (optional) class_weights

    Example:
        >>> # Reproducible dataloaders with stratified sampling
        >>> from src.utils.seed import set_seed
        >>> set_seed(42)
        >>> train_loader, val_loader = get_sunrgbd_dataloaders(seed=42, stratified=True)
        >>>
        >>> # For GPU augmentation mode
        >>> train_loader, val_loader = get_sunrgbd_dataloaders(normalize=False)
    """
    # Create datasets
    train_dataset = SUNRGBDDataset(
        data_root=data_root,
        train=True,
        target_size=target_size,
        normalize=normalize,
    )

    val_dataset = SUNRGBDDataset(
        data_root=data_root,
        train=False,
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

    print(f"\nDataLoader Info:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Stratified: {stratified}")

    if use_class_weights:
        class_weights = train_dataset.get_class_weights()
        print(f"\nClass weights computed (inverse frequency)")
        return train_loader, val_loader, class_weights

    return train_loader, val_loader


if __name__ == "__main__":
    """Test the dataset loader."""

    # Test dataset loading
    print("Testing SUN RGB-D Dataset Loader")
    print("=" * 80)

    train_dataset = SUNRGBDDataset(train=True)
    val_dataset = SUNRGBDDataset(train=False)

    print(f"\nTrain set: {len(train_dataset)} samples")
    print(f"Val set: {len(val_dataset)} samples")

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
    train_loader, val_loader = get_sunrgbd_dataloaders(batch_size=16, num_workers=0)

    # Get one batch
    rgb_batch, depth_batch, labels_batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  RGB: {rgb_batch.shape}")
    print(f"  Depth: {depth_batch.shape}")
    print(f"  Labels: {labels_batch.shape}")

    print("\n✓ Dataset loader working correctly!")
