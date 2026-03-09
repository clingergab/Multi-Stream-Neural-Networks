"""
SUN RGB-D 3-Stream Dataset Loader for scene classification.

Loads preprocessed SUN RGB-D dataset with RGB, Depth, and Orthogonal images.
Class names and normalization stats are loaded dynamically from the data root.

Tensors are stored at 256x256. At load time:
  - Train: RandomCrop(crop_size) + horizontal flip + augmentations
  - Val/Test: CenterCrop(crop_size)
"""

import json
import os
from collections import Counter

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F2

from src.data_utils.sunrgbd_dataset import _load_class_names, _load_norm_stats


class SUNRGBD3StreamDataset(Dataset):
    """
    SUN RGB-D 3-Stream dataset for scene classification.

    Class names are loaded dynamically from class_names.txt in data_root.
    Normalization stats are loaded from norm_stats.json in data_root.

    Directory structure:
        data_root/
            class_names.txt
            norm_stats.json
            train/ or val/ or test/
                rgb_tensors.pt
                depth_tensors.pt
                orth_tensors.pt
                labels.txt
    """

    VALID_SPLITS = ('train', 'val', 'test')

    def __init__(
        self,
        data_root='data/sunrgbd_19',
        split='train',
        crop_size: int = 224,
    ):
        """
        Args:
            data_root: Root directory of preprocessed dataset
            split: One of 'train', 'val', or 'test'
            crop_size: Output crop size. Train uses RandomCrop, val/test use CenterCrop.
        """
        if split not in self.VALID_SPLITS:
            raise ValueError(f"split must be one of {self.VALID_SPLITS}, got '{split}'")

        self.data_root = data_root
        self.split = split
        self.crop_size = crop_size

        # Load class names dynamically from data root
        self.CLASS_NAMES = _load_class_names(data_root)
        self.num_classes = len(self.CLASS_NAMES)

        # Load normalization statistics from data root
        self._norm_stats = _load_norm_stats(data_root)

        # Set split directory
        self.split_dir = os.path.join(data_root, split)

        # Load labels
        labels_file = os.path.join(self.split_dir, 'labels.txt')
        with open(labels_file, 'r') as f:
            self.labels = [int(line.strip()) for line in f.readlines()]

        self.num_samples = len(self.labels)

        # Try to load pre-resized tensor files (fast path)
        # mmap=True avoids copy-on-write overhead when DataLoader forks workers
        rgb_tensor_path = os.path.join(self.split_dir, 'rgb_tensors.pt')
        depth_tensor_path = os.path.join(self.split_dir, 'depth_tensors.pt')
        orth_tensor_path = os.path.join(self.split_dir, 'orth_tensors.pt')
        if (os.path.exists(rgb_tensor_path) and os.path.exists(depth_tensor_path)
                and os.path.exists(orth_tensor_path)):
            self.rgb_tensors = torch.load(rgb_tensor_path, weights_only=True, mmap=True)
            self.depth_tensors = torch.load(depth_tensor_path, weights_only=True, mmap=True)
            self.orth_tensors = torch.load(orth_tensor_path, weights_only=True, mmap=True)
            self.use_tensors = True
            print(f"Loaded SUN RGB-D 3-Stream {self.split}: {self.num_samples} samples, "
                  f"{self.num_classes} classes (tensors, mmap)")
        else:
            self.rgb_tensors = None
            self.depth_tensors = None
            self.orth_tensors = None
            self.use_tensors = False
            # Fallback PNG directories
            self.rgb_dir = os.path.join(self.split_dir, 'rgb')
            self.depth_dir = os.path.join(self.split_dir, 'depth')
            self.orth_dir = os.path.join(self.split_dir, 'orth')
            assert os.path.exists(self.rgb_dir), f"RGB directory not found: {self.rgb_dir}"
            assert os.path.exists(self.depth_dir), f"Depth directory not found: {self.depth_dir}"
            assert os.path.exists(self.orth_dir), f"Orthogonal directory not found: {self.orth_dir}"
            print(f"Loaded SUN RGB-D 3-Stream {self.split}: {self.num_samples} samples, "
                  f"{self.num_classes} classes (PNG fallback)")

        # Pre-create reusable transform instances (avoids per-__getitem__ construction)
        self._color_jitter_transform = v2.ColorJitter(
            brightness=0.37, contrast=0.37, saturation=0.37, hue=0.11,
        )
        self._rgb_erasing_transform = v2.RandomErasing(
            p=1.0, scale=(0.02, 0.10), ratio=(0.5, 2.0),
        )
        self._depth_erasing_transform = v2.RandomErasing(
            p=1.0, scale=(0.02, 0.1), ratio=(0.5, 2.0),
        )
        self._orth_erasing_transform = v2.RandomErasing(
            p=1.0, scale=(0.02, 0.1), ratio=(0.5, 2.0),
        )

    def __len__(self):
        return self.num_samples

    def _load_images_tensor(self, idx):
        """Load pre-resized uint8 images from tensor files as tensors (no PIL)."""
        rgb = self.rgb_tensors[idx]      # [3, H, W] uint8
        depth = self.depth_tensors[idx]  # [1, H, W] uint8
        orth = self.orth_tensors[idx]    # [1, H, W] uint8
        return rgb, depth, orth

    def _load_images_png(self, idx):
        """Load images from PNG files (fallback path), return as tensors."""
        # RGB: load PIL, convert to uint8 tensor immediately
        rgb_path = os.path.join(self.rgb_dir, f'{idx:05d}.png')
        rgb_pil = Image.open(rgb_path).convert('RGB')
        rgb = F2.pil_to_tensor(rgb_pil)  # [3, H, W] uint8

        # Depth: handle various PIL modes, normalize to [0,1], then quantize to uint8
        depth_path = os.path.join(self.depth_dir, f'{idx:05d}.png')
        depth_pil = Image.open(depth_path)
        if depth_pil.mode in ('I', 'I;16', 'I;16B'):
            depth_arr = np.array(depth_pil, dtype=np.float32)
            depth_arr = np.clip(depth_arr / 65535.0, 0.0, 1.0)
        else:
            depth_arr = np.array(depth_pil.convert('L'), dtype=np.float32)
            if depth_arr.max() > 1.0:
                depth_arr = depth_arr / 255.0
        depth = torch.from_numpy(
            (depth_arr * 255).clip(0, 255).astype(np.uint8)
        ).unsqueeze(0)  # [1, H, W] uint8

        # Orth: same handling as depth
        orth_path = os.path.join(self.orth_dir, f'{idx:05d}.png')
        orth_pil = Image.open(orth_path)
        if orth_pil.mode in ('I', 'I;16', 'I;16B'):
            orth_arr = np.array(orth_pil, dtype=np.float32)
            orth_arr = np.clip(orth_arr / 65535.0, 0.0, 1.0)
        else:
            orth_arr = np.array(orth_pil.convert('L'), dtype=np.float32)
            if orth_arr.max() > 1.0:
                orth_arr = orth_arr / 255.0
        orth = torch.from_numpy(
            (orth_arr * 255).clip(0, 255).astype(np.uint8)
        ).unsqueeze(0)  # [1, H, W] uint8

        return rgb, depth, orth

    def __getitem__(self, idx):
        """
        Returns:
            rgb: RGB image tensor [3, crop_size, crop_size] float32
            depth: Depth image tensor [1, crop_size, crop_size] float32
            orth: Orthogonal image tensor [1, crop_size, crop_size] float32
            label: Class label (0 to num_classes-1)
        """
        # Load images as uint8 tensors (no PIL conversion)
        if self.use_tensors:
            rgb, depth, orth = self._load_images_tensor(idx)
        else:
            rgb, depth, orth = self._load_images_png(idx)

        # At this point: rgb [3, H, W] uint8, depth [1, H, W] uint8, orth [1, H, W] uint8

        # ==================== TRAINING AUGMENTATION ====================
        if self.split == 'train':
            # 1. Synchronized Random Horizontal Flip (50%)
            if np.random.random() < 0.5:
                rgb = F2.horizontal_flip(rgb)
                depth = F2.horizontal_flip(depth)
                orth = F2.horizontal_flip(orth)

            # 2. Synchronized RandomCrop (256 -> crop_size)
            i, j, h, w = v2.RandomCrop.get_params(
                rgb, output_size=(self.crop_size, self.crop_size)
            )
            rgb = F2.crop(rgb, i, j, h, w)
            depth = F2.crop(depth, i, j, h, w)
            orth = F2.crop(orth, i, j, h, w)

            # 3. RGB-Only: Color Jitter (43% probability, pre-created instance)
            if np.random.random() < 0.43:
                rgb = self._color_jitter_transform(rgb)

            # 4. RGB-Only: Gaussian Blur (25% probability)
            if np.random.random() < 0.25:
                kernel_size = int(np.random.choice([3, 5, 7]))
                sigma = float(np.random.uniform(0.1, 1.7))
                rgb = F2.gaussian_blur(rgb, kernel_size=kernel_size, sigma=sigma)

            # 5. RGB-Only: Occasional Grayscale (17%)
            if np.random.random() < 0.17:
                rgb = F2.rgb_to_grayscale(rgb, num_output_channels=3)

            # 6. Depth & Orth: Appearance Augmentation (50% probability each)
            if np.random.random() < 0.5:
                depth = depth.float() / 255.0
                brightness_factor = np.random.uniform(0.75, 1.25)
                contrast_factor = np.random.uniform(0.75, 1.25)
                depth = (depth - 0.5) * contrast_factor + 0.5
                depth = depth * brightness_factor
                depth = depth + torch.randn_like(depth) * 0.06
                depth = depth.clamp(0.0, 1.0)

            if np.random.random() < 0.5:
                orth = orth.float() / 255.0
                brightness_factor = np.random.uniform(0.75, 1.25)
                contrast_factor = np.random.uniform(0.75, 1.25)
                orth = (orth - 0.5) * contrast_factor + 0.5
                orth = orth * brightness_factor
                orth = orth + torch.randn_like(orth) * 0.06
                orth = orth.clamp(0.0, 1.0)

        else:
            # Val/Test: CenterCrop (256 -> crop_size)
            rgb = F2.center_crop(rgb, (self.crop_size, self.crop_size))
            depth = F2.center_crop(depth, (self.crop_size, self.crop_size))
            orth = F2.center_crop(orth, (self.crop_size, self.crop_size))

        # ==================== TO FLOAT32 ====================
        if rgb.dtype == torch.uint8:
            rgb = rgb.float() / 255.0
        if depth.dtype == torch.uint8:
            depth = depth.float() / 255.0
        if orth.dtype == torch.uint8:
            orth = orth.float() / 255.0

        # ==================== NORMALIZATION ====================
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

        # Orth stats: use if available, otherwise fall back to depth stats
        orth_mean = self._norm_stats.get('orth_mean', self._norm_stats['depth_mean'])
        orth_std = self._norm_stats.get('orth_std', self._norm_stats['depth_std'])
        orth = F2.normalize(orth, mean=orth_mean, std=orth_std)

        # 7. Post-normalization Random Erasing (pre-created instances)
        if self.split == 'train':
            if np.random.random() < 0.17:
                rgb = self._rgb_erasing_transform(rgb)

            if np.random.random() < 0.1:
                depth = self._depth_erasing_transform(depth)

            if np.random.random() < 0.1:
                orth = self._orth_erasing_transform(orth)

        label = self.labels[idx]

        return rgb, depth, orth, label

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


def get_sunrgbd_3stream_dataloaders(
    data_root='data/sunrgbd_19',
    batch_size=32,
    num_workers=4,
    crop_size: int = 224,
    use_class_weights=False,
):
    """
    Create train, validation, and test dataloaders for SUN RGB-D 3-Stream.

    Args:
        data_root: Root directory of preprocessed dataset
        batch_size: Batch size
        num_workers: Number of dataloader workers
        crop_size: Output crop size (RandomCrop for train, CenterCrop for val/test)
        use_class_weights: If True, return class weights for loss

    Returns:
        train_loader, val_loader, test_loader, (optional) class_weights
    """
    train_dataset = SUNRGBD3StreamDataset(
        data_root=data_root,
        split='train',
        crop_size=crop_size,
    )

    # Val split is optional
    has_val = os.path.isdir(os.path.join(data_root, 'val'))
    val_dataset = None
    if has_val:
        val_dataset = SUNRGBD3StreamDataset(
            data_root=data_root,
            split='val',
            crop_size=crop_size,
        )

    test_dataset = SUNRGBD3StreamDataset(
        data_root=data_root,
        split='test',
        crop_size=crop_size,
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

    val_loader = None
    if val_dataset is not None:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0)
        )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )

    print(f"\nDataLoader Info:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader) if val_loader else 'N/A (no val split)'}")
    print(f"  Test batches: {len(test_loader)}")
    print(f"  Batch size: {batch_size}")

    if use_class_weights:
        class_weights = train_dataset.get_class_weights()
        print(f"\nClass weights computed (inverse frequency)")
        return train_loader, val_loader, test_loader, class_weights

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    """Test the dataset loader."""

    print("Testing SUN RGB-D 3-Stream Dataset Loader")
    print("=" * 80)

    try:
        train_dataset = SUNRGBD3StreamDataset(split='train')

        print(f"\nTrain set: {len(train_dataset)} samples, {train_dataset.num_classes} classes")
        print(f"Classes: {train_dataset.CLASS_NAMES}")

        rgb, depth, orth, label = train_dataset[0]
        print(f"\nSample 0:")
        print(f"  RGB shape: {rgb.shape}")
        print(f"  Depth shape: {depth.shape}")
        print(f"  Orth shape: {orth.shape}")
        print(f"  Label: {label} ({train_dataset.CLASS_NAMES[label]})")

        print("\nTrain class distribution:")
        train_dist = train_dataset.get_class_distribution()
        for class_name in train_dataset.CLASS_NAMES:
            info = train_dist[class_name]
            print(f"  {class_name:20s}: {info['count']:5d} ({info['percentage']:5.2f}%)")

        print("\nTesting dataloaders...")
        train_loader, val_loader, test_loader = get_sunrgbd_3stream_dataloaders(batch_size=16, num_workers=0)

        rgb_batch, depth_batch, orth_batch, labels_batch = next(iter(train_loader))
        print(f"\nBatch shapes:")
        print(f"  RGB: {rgb_batch.shape}")
        print(f"  Depth: {depth_batch.shape}")
        print(f"  Orth: {orth_batch.shape}")
        print(f"  Labels: {labels_batch.shape}")

        print("\nDataset loader working correctly!")

    except Exception as e:
        print(f"\nError testing dataset: {e}")
        print("Make sure the 'orth' directory exists in the dataset splits")
