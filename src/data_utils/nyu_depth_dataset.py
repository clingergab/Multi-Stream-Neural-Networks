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
        self.h5_file_path = h5_file_path
        self.train = train
        self.transform = transform
        self.target_size = target_size
        self.num_classes = num_classes

        # Open file temporarily to get metadata
        with h5py.File(h5_file_path, 'r') as f:
            # Get dataset shapes
            num_samples = f['images'].shape[0]  # 1449 total samples

            # Get scene labels (if available) or create from semantic labels
            if 'scenes' in f:
                scenes_data = f['scenes']

                # Check if it contains HDF5 references (MATLAB format)
                if scenes_data.dtype == h5py.ref_dtype:
                    # Dereference each element
                    self.scenes = np.zeros((1, num_samples), dtype=np.int64)
                    for i in range(num_samples):
                        ref = scenes_data[0, i]
                        self.scenes[0, i] = int(f[ref][0]) if ref else 0
                else:
                    self.scenes = np.array(scenes_data)  # Load into memory
            else:
                # Create scene labels from dominant semantic class
                self.scenes = self._create_scene_labels_from_file(f)

        # Train/test split (80/20)
        split_idx = int(num_samples * 0.8)  # 1159 train, 290 val

        if train:
            self.indices = list(range(0, split_idx))
        else:
            self.indices = list(range(split_idx, num_samples))

        # HDF5 file handle - will be opened per worker
        self._h5_file = None
        self._images = None
        self._depths = None
        self._labels = None

        # Pre-create normalization transform (reusable)
        self.rgb_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def _create_scene_labels_from_file(self, h5_file):
        """Create scene labels from semantic segmentation (fallback)."""
        # This is a simplified approach - you may need to customize
        # based on actual NYU Depth V2 label structure
        labels_dataset = h5_file['labels']
        num_samples = labels_dataset.shape[0]
        scene_labels = np.zeros((1, num_samples), dtype=np.int64)

        # Map semantic labels to scene categories (simplified)
        # You'll need to implement proper mapping based on NYU label definitions
        for i in range(num_samples):
            label_img = np.array(labels_dataset[i])  # [H, W]
            # Use mode (most common label) as scene indicator
            unique, counts = np.unique(label_img, return_counts=True)
            dominant_label = unique[np.argmax(counts)]
            scene_labels[0, i] = min(dominant_label % self.num_classes, self.num_classes - 1)

        return scene_labels

    def _open_hdf5(self):
        """Open HDF5 file in current process/worker."""
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_file_path, 'r')
            self._images = self._h5_file['images']
            self._depths = self._h5_file['depths']
            self._labels = self._h5_file['labels']

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single RGB-Depth sample.

        Returns:
            Tuple of (rgb_tensor, depth_tensor, scene_label)
        """
        # Ensure HDF5 file is open in this worker
        self._open_hdf5()

        real_idx = self.indices[idx]

        # Load RGB image - Format: [N, C, H, W] = [1449, 3, 640, 480]
        # HDF5 requires explicit conversion to numpy array
        rgb = np.array(self._images[real_idx])  # Shape: [3, 640, 480]

        # Transpose from [C, H, W] to [H, W, C] for PIL
        rgb = np.transpose(rgb, (1, 2, 0))  # [640, 480, 3]

        # NYU Depth V2 images are already uint8, but handle edge cases
        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8)

        rgb = Image.fromarray(rgb, mode='RGB')

        # Load Depth map - Format: [N, H, W] = [1449, 640, 480]
        # HDF5 requires explicit conversion to numpy array
        depth = np.array(self._depths[real_idx])  # Shape: [640, 480]

        # Normalize depth to 0-255 range for visualization/processing
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth = (depth * 255).astype(np.uint8)
        depth = Image.fromarray(depth, mode='L')  # Grayscale

        # Get scene label - scenes shape is (1, N) or (N,)
        if self.scenes.ndim == 2:
            label = int(self.scenes[0, real_idx])
        else:
            label = int(self.scenes[real_idx])
        label = torch.tensor(label, dtype=torch.long)

        # Resize to target size
        rgb = rgb.resize(self.target_size, Image.BILINEAR)
        depth = depth.resize(self.target_size, Image.BILINEAR)

        # Convert to tensors
        rgb = transforms.ToTensor()(rgb)  # [3, 224, 224]
        depth = transforms.ToTensor()(depth)  # [1, 224, 224]

        # Apply synchronized geometric augmentation
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

        # Apply ImageNet normalization to RGB only
        rgb = self.rgb_normalize(rgb)

        return rgb, depth, label

    def __del__(self):
        """Clean up h5 file handle."""
        if self._h5_file is not None:
            self._h5_file.close()


def get_nyu_transforms(train: bool = True):
    """
    Get geometric transforms for NYU Depth V2.

    Note: These are geometric transforms only (flip, rotation) that apply to both RGB and depth.
    Color-specific transforms and normalization are handled separately in the dataset.

    Args:
        train: If True, return training transforms with augmentation

    Returns:
        Transform pipeline for synchronized RGB and depth augmentation
    """
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10)
        ])
    else:
        return None  # No augmentation for validation


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
    # Note: persistent_workers and prefetch_factor only work with num_workers > 0
    train_loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': num_workers,
        'pin_memory': True if num_workers > 0 else False
    }
    if num_workers > 0:
        train_loader_kwargs['persistent_workers'] = True
        train_loader_kwargs['prefetch_factor'] = 2

    val_loader_kwargs = {
        'batch_size': batch_size * 2,  # Larger batch for validation
        'shuffle': False,
        'num_workers': num_workers,
        'pin_memory': True if num_workers > 0 else False
    }
    if num_workers > 0:
        val_loader_kwargs['persistent_workers'] = True
        val_loader_kwargs['prefetch_factor'] = 2

    train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    val_loader = DataLoader(val_dataset, **val_loader_kwargs)

    return train_loader, val_loader
