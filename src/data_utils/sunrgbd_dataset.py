"""
SUN RGB-D Dataset Loader for 15-category scene classification.

Loads preprocessed SUN RGB-D dataset with RGB and depth images.
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
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
        rgb_transform=None,
        depth_transform=None,
        shared_transform=None,
        target_size=(224, 224),
    ):
        """
        Args:
            data_root: Root directory of preprocessed dataset
            train: If True, load training set; else load validation set
            rgb_transform: Transforms for RGB images (applied after shared)
            depth_transform: Transforms for depth images (applied after shared)
            shared_transform: Transforms applied to both RGB and depth (before individual)
            target_size: Target image size (H, W)
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

        # RGB and depth directories
        self.rgb_dir = os.path.join(self.split_dir, 'rgb')
        self.depth_dir = os.path.join(self.split_dir, 'depth')

        # Verify directories exist
        assert os.path.exists(self.rgb_dir), f"RGB directory not found: {self.rgb_dir}"
        assert os.path.exists(self.depth_dir), f"Depth directory not found: {self.depth_dir}"

        # Set up transforms
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform
        self.shared_transform = shared_transform

        # Default transforms if not provided
        # IMPORTANT: RandomHorizontalFlip must be applied BEFORE individual transforms
        # to ensure RGB and Depth are flipped together
        if self.rgb_transform is None:
            # REMOVED ColorJitter - RGB and Depth now have same augmentation difficulty
            # Only difference is normalization (ImageNet stats for RGB)
            self.rgb_transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        if self.depth_transform is None:
            # Depth-specific transforms (applied AFTER shared flip)
            # Normalize to same scale as RGB for balanced gradient flow
            self.depth_transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5027], std=[0.2197])  # SUN RGB-D dataset stats
            ])

        # Store whether we're in training mode (for synchronized flipping)
        self.train = train

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

        # Convert depth to grayscale/float format
        if depth.mode in ('I', 'I;16', 'I;16B'):
            depth_array = np.array(depth, dtype=np.float32)
            if depth_array.max() > 0:
                depth_array = (depth_array / depth_array.max() * 255).astype(np.uint8)
            else:
                depth_array = depth_array.astype(np.uint8)
            depth = Image.fromarray(depth_array, mode='L')
        elif depth.mode == 'RGB':
            depth = depth.convert('L')
        elif depth.mode != 'L':
            depth = depth.convert('L')

        # ==================== TRAINING AUGMENTATION ====================
        if self.train:
            # 1. Synchronized Random Horizontal Flip (50%)
            if np.random.random() < 0.5:
                rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)
                depth = depth.transpose(Image.FLIP_LEFT_RIGHT)

            # 2. Synchronized Random Resized Crop (CRITICAL - always apply)
            # Scale 0.8-1.0 is conservative (not too aggressive)
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                rgb, scale=(0.8, 1.0), ratio=(0.95, 1.05)
            )
            rgb = transforms.functional.resized_crop(rgb, i, j, h, w, self.target_size)
            depth = transforms.functional.resized_crop(depth, i, j, h, w, self.target_size)

            # 3. RGB-Only: Moderate Color Jitter (50% probability)
            # Conservative values to avoid over-augmentation
            if np.random.random() < 0.5:
                color_jitter = transforms.ColorJitter(
                    brightness=0.2,  # ±20% (conservative)
                    contrast=0.2,    # ±20%
                    saturation=0.2,  # ±20%
                    hue=0.05         # ±5% (small hue shift)
                )
                rgb = color_jitter(rgb)

            # 4. RGB-Only: Occasional Grayscale (5% - rare, just for robustness)
            if np.random.random() < 0.05:
                rgb = transforms.functional.to_grayscale(rgb, num_output_channels=3)

            # 5. Depth-Only: Light Gaussian Noise (20% probability)
            # Simulates sensor noise without being too aggressive
            if np.random.random() < 0.2:
                depth_array = np.array(depth, dtype=np.float32)
                noise = np.random.normal(0, 3, depth_array.shape)  # Std=3 (light noise)
                depth_array = np.clip(depth_array + noise, 0, 255)
                depth = Image.fromarray(depth_array.astype(np.uint8), mode='L')

        else:
            # Validation: just resize (no augmentation)
            rgb = transforms.functional.resize(rgb, self.target_size)
            depth = transforms.functional.resize(depth, self.target_size)

        # ==================== NORMALIZATION ====================
        # Convert to tensor and normalize
        rgb = transforms.functional.to_tensor(rgb)
        rgb = transforms.functional.normalize(
            rgb, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        depth = transforms.functional.to_tensor(depth)
        depth = transforms.functional.normalize(
            depth, mean=[0.5027], std=[0.2197]
        )

        # 6. Post-normalization Random Erasing (10% - conservative)
        # Applied after normalization for both modalities
        if self.train and np.random.random() < 0.1:
            erasing = transforms.RandomErasing(
                p=1.0,
                scale=(0.02, 0.1),     # Small patches (2-10% of image)
                ratio=(0.5, 2.0)       # Reasonable aspect ratios
            )
            rgb = erasing(rgb)
            # Depth gets separate random erasing (different patches)
            if np.random.random() < 0.5:  # Only 5% overall (10% * 50%)
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
        from collections import Counter
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
        from collections import Counter
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


def get_sunrgbd_dataloaders(
    data_root='data/sunrgbd_15',
    batch_size=32,
    num_workers=4,
    target_size=(224, 224),
    use_class_weights=False,
):
    """
    Create train and validation dataloaders for SUN RGB-D.

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
    train_dataset = SUNRGBDDataset(
        data_root=data_root,
        train=True,
        target_size=target_size,
    )

    val_dataset = SUNRGBDDataset(
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
        persistent_workers=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
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
