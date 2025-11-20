"""
SUN RGB-D 3-Stream Dataset Loader for 15-category scene classification.

Loads preprocessed SUN RGB-D dataset with RGB, Depth, and Orthogonal images.
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from collections import Counter


class SUNRGBD3StreamDataset(Dataset):
    """
    SUN RGB-D 3-Stream dataset for scene classification (15 categories).

    Directory structure:
        data_root/
            train/ or val/
                rgb/
                    00000.png
                    ...
                depth/
                    00000.png
                    ...
                orth/
                    00000.png
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
        target_size=(224, 224),
    ):
        """
        Args:
            data_root: Root directory of preprocessed dataset
            train: If True, load training set; else load validation set
            target_size: Target image size (H, W)

        Note: All augmentation is performed in __getitem__() to enable synchronized
        transforms between RGB, Depth, and Orthogonal modalities.
        """
        self.data_root = data_root
        self.train = train
        self.target_size = target_size

        # Set split directory
        split = 'train' if train else 'val'
        self.split_dir = os.path.join(data_root, split)

        # Load labels
        labels_file = os.path.join(self.split_dir, 'labels.txt')
        with open(labels_file, 'r') as f:
            self.labels = [int(line.strip()) for line in f.readlines()]

        self.num_samples = len(self.labels)

        # RGB, Depth, and Orthogonal directories
        self.rgb_dir = os.path.join(self.split_dir, 'rgb')
        self.depth_dir = os.path.join(self.split_dir, 'depth')
        self.orth_dir = os.path.join(self.split_dir, 'orth')

        # Verify directories exist
        assert os.path.exists(self.rgb_dir), f"RGB directory not found: {self.rgb_dir}"
        assert os.path.exists(self.depth_dir), f"Depth directory not found: {self.depth_dir}"
        assert os.path.exists(self.orth_dir), f"Orthogonal directory not found: {self.orth_dir}"

        print(f"Loaded SUN RGB-D 3-Stream {split} set: {self.num_samples} samples, 15 classes")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Returns:
            rgb: RGB image tensor [3, H, W]
            depth: Depth image tensor [1, H, W]
            orth: Orthogonal image tensor [1, H, W]
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
            # Global normalization: divide by max possible depth value
            # Using 65000 to cover 99th percentile (65392) while avoiding clipping
            depth_arr = np.clip(depth_arr / 65000.0, 0.0, 1.0)
            depth = Image.fromarray(depth_arr, mode='F')
        else:
            # Fallback for other modes
            depth = depth.convert('F')
            # If it was 0-255, scale to 0-1
            if np.array(depth).max() > 1.0:
                depth_arr = np.array(depth)
                depth = Image.fromarray(depth_arr / 255.0, mode='F')

        # Load orthogonal image
        orth_path = os.path.join(self.orth_dir, f'{idx:05d}.png')
        orth = Image.open(orth_path)

        # Convert Orth to Mode F (Float32) with global normalization
        # Use global normalization to avoid artifacts from per-image normalization after cropping
        if orth.mode in ('I', 'I;16', 'I;16B'):
            orth_arr = np.array(orth, dtype=np.float32)
            # Global normalization using dataset statistics (1-99 percentile range)
            # Computed from full dataset: min=4368, max=60859 (covers 98% of data)
            # Using slightly wider range for safety: [0, 65000] to avoid clipping outliers
            orth_arr = np.clip(orth_arr / 65000.0, 0.0, 1.0)
            orth = Image.fromarray(orth_arr, mode='F')
        else:
            # Fallback for other modes
            orth = orth.convert('F')
            # If it was 0-255, scale to 0-1
            if np.array(orth).max() > 1.0:
                orth_arr = np.array(orth)
                orth = Image.fromarray(orth_arr / 255.0, mode='F')


        # ==================== TRAINING AUGMENTATION ====================
        if self.train:
            # 1. Synchronized Random Horizontal Flip (50%)
            # Applied to ALL streams to maintain geometric consistency
            if np.random.random() < 0.5:
                rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)
                depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
                orth = orth.transpose(Image.FLIP_LEFT_RIGHT)

            # 2. Synchronized Random Resized Crop (50% probability)
            # Applied to ALL streams to maintain geometric consistency
            if np.random.random() < 0.5:
                i, j, h, w = transforms.RandomResizedCrop.get_params(
                    rgb, scale=(0.9, 1.0), ratio=(0.95, 1.05)
                )
                rgb = transforms.functional.resized_crop(rgb, i, j, h, w, self.target_size)
                depth = transforms.functional.resized_crop(depth, i, j, h, w, self.target_size)
                orth = transforms.functional.resized_crop(orth, i, j, h, w, self.target_size)
            else:
                # No crop - preserve full scene context
                rgb = transforms.functional.resize(rgb, self.target_size)
                depth = transforms.functional.resize(depth, self.target_size)
                orth = transforms.functional.resize(orth, self.target_size)

            # 3. RGB-Only: Color Jitter (43% probability)
            # Appearance only - does not affect geometry
            if np.random.random() < 0.43:
                color_jitter = transforms.ColorJitter(
                    brightness=0.37,
                    contrast=0.37,
                    saturation=0.37,
                    hue=0.11
                )
                rgb = color_jitter(rgb)

            # 4. RGB-Only: Gaussian Blur (25% probability)
            # Appearance only
            if np.random.random() < 0.25:
                kernel_size = int(np.random.choice([3, 5, 7]))
                sigma = float(np.random.uniform(0.1, 1.7))
                rgb = transforms.functional.gaussian_blur(rgb, kernel_size=kernel_size, sigma=sigma)

            # 5. RGB-Only: Occasional Grayscale (17%)
            # Appearance only
            if np.random.random() < 0.17:
                rgb = transforms.functional.to_grayscale(rgb, num_output_channels=3)

            # 6. Depth & Orth: Appearance Augmentation (50% probability each)
            # Independent augmentation for each modality to increase training diversity
            # Each modality gets its own brightness/contrast/noise variations

            # Depth appearance augmentation (50%)
            if np.random.random() < 0.5:
                depth_array = np.array(depth, dtype=np.float32)  # Ensure float32

                # Apply brightness and contrast
                # Slightly higher than RGB (±25% vs ±20%) to compensate for:
                #   1. Depth having 1 channel vs RGB's 3 channels
                #   2. Reduced crop probability (50% vs previous 100%)
                brightness_factor = np.random.uniform(0.75, 1.25)  # ±25%
                contrast_factor = np.random.uniform(0.75, 1.25)    # ±25%

                # Apply contrast then brightness (same order as ColorJitter)
                # Center is 0.5 for contrast (0-1 range)
                depth_array = (depth_array - 0.5) * contrast_factor + 0.5
                depth_array = depth_array * brightness_factor

                # Add Gaussian noise (simulates sensor noise)
                # Scaled for 0-1 range: sigma=0.06 on 0-1 (~6%)
                noise = np.random.normal(0, 0.06, depth_array.shape).astype(np.float32)
                depth_array = depth_array + noise

                depth_array = np.clip(depth_array, 0.0, 1.0).astype(np.float32)
                depth = Image.fromarray(depth_array, mode='F')

            # Orth appearance augmentation (50% - independent from depth)
            if np.random.random() < 0.5:
                orth_array = np.array(orth, dtype=np.float32)  # Ensure float32

                # Apply independent brightness and contrast factors
                brightness_factor = np.random.uniform(0.75, 1.25)  # ±25%
                contrast_factor = np.random.uniform(0.75, 1.25)    # ±25%

                # Apply contrast then brightness
                orth_array = (orth_array - 0.5) * contrast_factor + 0.5
                orth_array = orth_array * brightness_factor

                # Add Gaussian noise
                noise = np.random.normal(0, 0.06, orth_array.shape).astype(np.float32)
                orth_array = orth_array + noise

                orth_array = np.clip(orth_array, 0.0, 1.0).astype(np.float32)
                orth = Image.fromarray(orth_array, mode='F')

        else:
            # Validation: just resize (no augmentation)
            rgb = transforms.functional.resize(rgb, self.target_size)
            depth = transforms.functional.resize(depth, self.target_size)
            orth = transforms.functional.resize(orth, self.target_size)

        # ==================== NORMALIZATION ====================
        # Convert to tensor and normalize ALL modalities to ~[-1, 1]
        # Using mean=0.5, std=0.5 for consistency across all streams
        # This ensures all inputs have similar scale to the neural network

        rgb = transforms.functional.to_tensor(rgb)
        rgb = transforms.functional.normalize(
            rgb, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        )

        depth = transforms.functional.to_tensor(depth)
        depth = transforms.functional.normalize(
            depth, mean=[0.5], std=[0.5]
        )

        orth = transforms.functional.to_tensor(orth)
        orth = transforms.functional.normalize(
            orth, mean=[0.5], std=[0.5]
        )

        # 7. Post-normalization Random Erasing
        if self.train:
            # RGB random erasing (17%)
            if np.random.random() < 0.17:
                erasing = transforms.RandomErasing(
                    p=1.0,
                    scale=(0.02, 0.10),
                    ratio=(0.5, 2.0)
                )
                rgb = erasing(rgb)

            # Depth random erasing (10%)
            if np.random.random() < 0.1:
                erasing = transforms.RandomErasing(
                    p=1.0,
                    scale=(0.02, 0.1),
                    ratio=(0.5, 2.0)
                )
                depth = erasing(depth)
                
            # Orthogonal random erasing (10%)
            if np.random.random() < 0.1:
                erasing = transforms.RandomErasing(
                    p=1.0,
                    scale=(0.02, 0.1),
                    ratio=(0.5, 2.0)
                )
                orth = erasing(orth)

        # Get label
        label = self.labels[idx]

        return rgb, depth, orth, label

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


def get_sunrgbd_3stream_dataloaders(
    data_root='data/sunrgbd_15',
    batch_size=32,
    num_workers=4,
    target_size=(224, 224),
    use_class_weights=False,
):
    """
    Create train and validation dataloaders for SUN RGB-D 3-Stream.

    Args:
        data_root: Root directory of preprocessed dataset
        batch_size: Batch size
        num_workers: Number of dataloader workers
        target_size: Target image size (H, W)
        use_class_weights: If True, return class weights for loss

    Returns:
        train_loader, val_loader, (optional) class_weights
    """
    # Create datasets
    train_dataset = SUNRGBD3StreamDataset(
        data_root=data_root,
        train=True,
        target_size=target_size,
    )

    val_dataset = SUNRGBD3StreamDataset(
        data_root=data_root,
        train=False,
        target_size=target_size,
    )

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )

    print(f"\nDataLoader Info:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Batch size: {batch_size}")

    if use_class_weights:
        class_weights = train_dataset.get_class_weights()
        print(f"\nClass weights computed (inverse frequency)")
        return train_loader, val_loader, class_weights

    return train_loader, val_loader


if __name__ == "__main__":
    """Test the dataset loader."""

    # Test dataset loading
    print("Testing SUN RGB-D 3-Stream Dataset Loader")
    print("=" * 80)

    try:
        train_dataset = SUNRGBD3StreamDataset(train=True)
        val_dataset = SUNRGBD3StreamDataset(train=False)

        print(f"\nTrain set: {len(train_dataset)} samples")
        print(f"Val set: {len(val_dataset)} samples")

        # Test getting a sample
        rgb, depth, orth, label = train_dataset[0]
        print(f"\nSample 0:")
        print(f"  RGB shape: {rgb.shape}")
        print(f"  Depth shape: {depth.shape}")
        print(f"  Orth shape: {orth.shape}")
        print(f"  Label: {label} ({train_dataset.CLASS_NAMES[label]})")

        # Test class distribution
        print("\nTrain class distribution:")
        train_dist = train_dataset.get_class_distribution()
        for class_name in train_dataset.CLASS_NAMES:
            info = train_dist[class_name]
            print(f"  {class_name:20s}: {info['count']:5d} ({info['percentage']:5.2f}%)")

        # Test dataloader
        print("\nTesting dataloaders...")
        train_loader, val_loader = get_sunrgbd_3stream_dataloaders(batch_size=16, num_workers=0)

        # Get one batch
        rgb_batch, depth_batch, orth_batch, labels_batch = next(iter(train_loader))
        print(f"\nBatch shapes:")
        print(f"  RGB: {rgb_batch.shape}")
        print(f"  Depth: {depth_batch.shape}")
        print(f"  Orth: {orth_batch.shape}")
        print(f"  Labels: {labels_batch.shape}")

        print("\n✓ Dataset loader working correctly!")
    
    except Exception as e:
        print(f"\nError testing dataset: {e}")
        print("Make sure the 'orth' directory exists in data/sunrgbd_15/train/ and data/sunrgbd_15/val/")
