"""
SUN RGB-D 3-Stream Dataset Loader for 15-category scene classification.

Loads preprocessed SUN RGB-D dataset with RGB, Depth, and Orthogonal images.
"""

import os
from collections import Counter

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F2


class SUNRGBD3StreamDataset(Dataset):
    """
    SUN RGB-D 3-Stream dataset for scene classification (15 categories).

    Directory structure:
        data_root/
            train/ or val/ or test/
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
    ):
        """
        Args:
            data_root: Root directory of preprocessed dataset
            split: One of 'train', 'val', or 'test'
            target_size: Target image size (H, W)

        Note: All augmentation is performed in __getitem__() to enable synchronized
        transforms between RGB, Depth, and Orthogonal modalities.
        """
        if split not in self.VALID_SPLITS:
            raise ValueError(f"split must be one of {self.VALID_SPLITS}, got '{split}'")

        self.data_root = data_root
        self.split = split
        self.target_size = target_size

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
            print(f"Loaded SUN RGB-D 3-Stream {self.split} set: {self.num_samples} samples, 15 classes (pre-resized tensors, mmap)")
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
            print(f"Loaded SUN RGB-D 3-Stream {self.split} set: {self.num_samples} samples, 15 classes (PNG fallback)")

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
        # Convert to uint8 tensor to match tensor fast path format.
        # Note: for mode 'L' with values 0-255, this divides by 255 then multiplies
        # by 255 (a no-op with rounding). Intentional — normalizes all depth formats
        # to the same uint8 output regardless of input mode.
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
            rgb: RGB image tensor [3, H, W] float32
            depth: Depth image tensor [1, H, W] float32
            orth: Orthogonal image tensor [1, H, W] float32
            label: Class label (0-14)
        """
        # Load images as uint8 tensors (no PIL conversion)
        if self.use_tensors:
            rgb, depth, orth = self._load_images_tensor(idx)
            images_already_resized = True
        else:
            rgb, depth, orth = self._load_images_png(idx)
            images_already_resized = False

        # At this point: rgb [3, H, W] uint8, depth [1, H, W] uint8, orth [1, H, W] uint8

        # ==================== TRAINING AUGMENTATION ====================
        if self.split == 'train':
            # 1. Synchronized Random Horizontal Flip (50%)
            if np.random.random() < 0.5:
                rgb = F2.horizontal_flip(rgb)
                depth = F2.horizontal_flip(depth)
                orth = F2.horizontal_flip(orth)

            # 2. Synchronized Random Resized Crop (50% probability)
            if np.random.random() < 0.5:
                i, j, h, w = v2.RandomResizedCrop.get_params(
                    rgb, scale=(0.9, 1.0), ratio=(0.95, 1.05)
                )
                rgb = F2.resized_crop(rgb, i, j, h, w, self.target_size)
                depth = F2.resized_crop(depth, i, j, h, w, self.target_size)
                orth = F2.resized_crop(orth, i, j, h, w, self.target_size)
            elif not images_already_resized:
                rgb = F2.resize(rgb, self.target_size)
                depth = F2.resize(depth, self.target_size)
                orth = F2.resize(orth, self.target_size)

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
            # Uses torch ops instead of numpy/PIL
            if np.random.random() < 0.5:
                depth = depth.float() / 255.0  # uint8 → float32 [0, 1]
                brightness_factor = np.random.uniform(0.75, 1.25)
                contrast_factor = np.random.uniform(0.75, 1.25)
                depth = (depth - 0.5) * contrast_factor + 0.5
                depth = depth * brightness_factor
                depth = depth + torch.randn_like(depth) * 0.06
                depth = depth.clamp(0.0, 1.0)
                # depth is now float32 [0, 1] — skips later uint8→float conversion

            if np.random.random() < 0.5:
                orth = orth.float() / 255.0  # uint8 → float32 [0, 1]
                brightness_factor = np.random.uniform(0.75, 1.25)
                contrast_factor = np.random.uniform(0.75, 1.25)
                orth = (orth - 0.5) * contrast_factor + 0.5
                orth = orth * brightness_factor
                orth = orth + torch.randn_like(orth) * 0.06
                orth = orth.clamp(0.0, 1.0)
                # orth is now float32 [0, 1] — skips later uint8→float conversion

        elif not images_already_resized:
            # Validation: just resize (no augmentation) — only needed for PNG path
            rgb = F2.resize(rgb, self.target_size)
            depth = F2.resize(depth, self.target_size)
            orth = F2.resize(orth, self.target_size)

        # ==================== TO FLOAT32 ====================
        # Convert uint8 → float32 [0, 1] (replaces to_tensor() which did CHW permute + /255;
        # our tensors are already CHW, so just need dtype conversion).
        # Dtype guard: if depth/orth aug triggered, they are already float32 — no double-division.
        if rgb.dtype == torch.uint8:
            rgb = rgb.float() / 255.0
        if depth.dtype == torch.uint8:
            depth = depth.float() / 255.0
        if orth.dtype == torch.uint8:
            orth = orth.float() / 255.0

        # ==================== NORMALIZATION ====================
        rgb = F2.normalize(
            rgb,
            mean=[0.49829878533942046, 0.4667760665084003, 0.44289694564460663],
            std=[0.27731416732781294, 0.28601699847044426, 0.2899506179157605],
        )

        depth = F2.normalize(depth, mean=[0.2908], std=[0.1504])

        orth = F2.normalize(orth, mean=[0.4944], std=[0.2065])

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
    target_size=(416, 544),
    use_class_weights=False,
):
    """
    Create train, validation, and test dataloaders for SUN RGB-D 3-Stream.

    Args:
        data_root: Root directory of preprocessed dataset
        batch_size: Batch size
        num_workers: Number of dataloader workers
        target_size: Target image size (H, W)
        use_class_weights: If True, return class weights for loss

    Returns:
        train_loader, val_loader, test_loader, (optional) class_weights
    """
    # Create datasets
    train_dataset = SUNRGBD3StreamDataset(
        data_root=data_root,
        split='train',
        target_size=target_size,
    )

    val_dataset = SUNRGBD3StreamDataset(
        data_root=data_root,
        split='val',
        target_size=target_size,
    )

    test_dataset = SUNRGBD3StreamDataset(
        data_root=data_root,
        split='test',
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
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    print(f"  Batch size: {batch_size}")

    if use_class_weights:
        class_weights = train_dataset.get_class_weights()
        print(f"\nClass weights computed (inverse frequency)")
        return train_loader, val_loader, test_loader, class_weights

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    """Test the dataset loader."""

    # Test dataset loading
    print("Testing SUN RGB-D 3-Stream Dataset Loader")
    print("=" * 80)

    try:
        train_dataset = SUNRGBD3StreamDataset(split='train')
        val_dataset = SUNRGBD3StreamDataset(split='val')
        test_dataset = SUNRGBD3StreamDataset(split='test')

        print(f"\nTrain set: {len(train_dataset)} samples")
        print(f"Val set: {len(val_dataset)} samples")
        print(f"Test set: {len(test_dataset)} samples")

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
        train_loader, val_loader, test_loader = get_sunrgbd_3stream_dataloaders(batch_size=16, num_workers=0)

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
        print("Make sure the 'orth' directory exists in data/sunrgbd_15/train/, val/, and test/")
