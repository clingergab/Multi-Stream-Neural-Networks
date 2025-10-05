"""
NYU Depth V2 dataset for dual-stream MCResNet training.
Provides RGB images + Depth maps for scene classification.
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from typing import Optional, Tuple, Callable
import random


class NYUDepthV2Dataset(Dataset):
    """
    NYU Depth V2 dataset for RGB-D scene classification.

    Provides synchronized RGB and Depth images with consistent augmentation.
    Compatible with MCResNet dual-stream architecture.
    """

    def __init__(
        self,
        h5_file_path: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_size: Tuple[int, int] = (224, 224),
        num_classes: int = 13  # Scene classification
    ):
        """
        Initialize NYU Depth V2 dataset.

        Args:
            h5_file_path: Path to nyu_depth_v2_labeled.mat file
            train: If True, use training split; else use test split
            transform: Augmentation transforms (applied to both RGB and depth)
            target_size: Resize images to this size (height, width)
            num_classes: Number of scene classes (13 for room classification)
        """
        self.h5_file = h5py.File(h5_file_path, 'r')
        self.train = train
        self.transform = transform
        self.target_size = target_size
        self.num_classes = num_classes

        # Load images and depths
        # NYU Depth V2 HDF5 format: [N, C, H, W] for images, [N, H, W] for depths
        self.images = self.h5_file['images']  # Shape: [1449, 3, 640, 480]
        self.depths = self.h5_file['depths']  # Shape: [1449, 640, 480]
        self.labels = self.h5_file['labels']  # Shape: [1449, 640, 480]

        # Get scene labels (if available) or create from semantic labels
        if 'scenes' in self.h5_file:
            self.scenes = self.h5_file['scenes']
        else:
            # Create scene labels from dominant semantic class
            self.scenes = self._create_scene_labels()

        # Train/test split (80/20)
        num_samples = self.images.shape[0]  # 1449 total samples
        split_idx = int(num_samples * 0.8)  # 1159 train, 290 val

        if train:
            self.indices = list(range(0, split_idx))
        else:
            self.indices = list(range(split_idx, num_samples))

    def _create_scene_labels(self):
        """Create scene labels from semantic segmentation (fallback)."""
        # This is a simplified approach - you may need to customize
        # based on actual NYU Depth V2 label structure
        num_samples = self.labels.shape[0]
        scene_labels = np.zeros(num_samples, dtype=np.int64)

        # Map semantic labels to scene categories (simplified)
        # You'll need to implement proper mapping based on NYU label definitions
        for i in range(num_samples):
            label_img = self.labels[i, :, :]  # [H, W]
            # Use mode (most common label) as scene indicator
            unique, counts = np.unique(label_img, return_counts=True)
            dominant_label = unique[np.argmax(counts)]
            scene_labels[i] = min(dominant_label % self.num_classes, self.num_classes - 1)

        return scene_labels

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single RGB-Depth sample.

        Returns:
            Tuple of (rgb_tensor, depth_tensor, scene_label)
        """
        real_idx = self.indices[idx]

        # Load RGB image - Format: [N, C, H, W] = [1449, 3, 640, 480]
        rgb = self.images[real_idx, :, :, :]  # Shape: [3, 640, 480]

        # Transpose from [C, H, W] to [H, W, C] for PIL
        rgb = np.transpose(rgb, (1, 2, 0))  # [640, 480, 3]

        # Handle potential data type issues
        if rgb.dtype != np.uint8:
            # Scale to 0-255 if needed
            if rgb.max() <= 1.0:
                rgb = (rgb * 255).astype(np.uint8)
            else:
                rgb = rgb.astype(np.uint8)

        rgb = Image.fromarray(rgb, mode='RGB')

        # Load Depth map - Format: [N, H, W] = [1449, 640, 480]
        depth = self.depths[real_idx, :, :]  # Shape: [640, 480]

        # Normalize depth to 0-255 range for visualization/processing
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth = (depth * 255).astype(np.uint8)
        depth = Image.fromarray(depth, mode='L')  # Grayscale

        # Get scene label
        if isinstance(self.scenes, np.ndarray):
            label = self.scenes[real_idx]
        else:
            label = self.scenes[real_idx, 0]
        label = torch.tensor(label, dtype=torch.long)

        # Resize to target size
        rgb = rgb.resize(self.target_size, Image.BILINEAR)
        depth = depth.resize(self.target_size, Image.BILINEAR)

        # Convert to tensors
        rgb = transforms.ToTensor()(rgb)  # [3, 224, 224]
        depth = transforms.ToTensor()(depth)  # [1, 224, 224]

        # Apply synchronized augmentation
        if self.transform:
            seed = random.randint(0, 2**32 - 1)

            # Apply to RGB
            torch.manual_seed(seed)
            random.seed(seed)
            rgb = self.transform(rgb)

            # Apply to depth with same seed
            torch.manual_seed(seed)
            random.seed(seed)
            depth = self.transform(depth)

        return rgb, depth, label

    def __del__(self):
        """Clean up h5 file handle."""
        if hasattr(self, 'h5_file'):
            self.h5_file.close()


def get_nyu_transforms(train: bool = True):
    """
    Get standard ImageNet-style transforms for NYU Depth V2.

    Args:
        train: If True, return training transforms with augmentation

    Returns:
        Transform pipeline for RGB and depth
    """
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def create_nyu_dataloaders(
    h5_file_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    target_size: Tuple[int, int] = (224, 224),
    num_classes: int = 13
):
    """
    Create train and validation dataloaders for NYU Depth V2.

    Args:
        h5_file_path: Path to NYU Depth V2 .mat file
        batch_size: Training batch size (validation uses 2x)
        num_workers: Number of data loading workers
        target_size: Image resize dimensions
        num_classes: Number of scene classes

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = NYUDepthV2Dataset(
        h5_file_path,
        train=True,
        transform=get_nyu_transforms(train=True),
        target_size=target_size,
        num_classes=num_classes
    )

    val_dataset = NYUDepthV2Dataset(
        h5_file_path,
        train=False,
        transform=get_nyu_transforms(train=False),
        target_size=target_size,
        num_classes=num_classes
    )

    # Create dataloaders (A100 optimized)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  # Larger batch for validation
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    return train_loader, val_loader
