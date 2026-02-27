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
from torchvision import transforms


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

    def __len__(self):
        return self.num_samples

    def _load_images_tensor(self, idx):
        """Load pre-resized uint8 images from tensor files and convert to PIL."""
        # RGB: [3, H, W] uint8 → [H, W, 3] numpy → PIL RGB
        rgb_arr = self.rgb_tensors[idx].numpy().transpose(1, 2, 0)
        rgb = Image.fromarray(np.ascontiguousarray(rgb_arr), mode='RGB')

        # Depth: [1, H, W] uint8 → [H, W] float32 in [0, 1] → PIL mode F
        depth_arr = self.depth_tensors[idx, 0].numpy()
        depth = Image.fromarray(depth_arr.astype(np.float32) / 255.0, mode='F')

        # Orth: [1, H, W] uint8 → [H, W] float32 in [0, 1] → PIL mode F
        orth_arr = self.orth_tensors[idx, 0].numpy()
        orth = Image.fromarray(orth_arr.astype(np.float32) / 255.0, mode='F')

        return rgb, depth, orth

    def _load_images_png(self, idx):
        """Load images from PNG files (fallback path)."""
        # RGB
        rgb_path = os.path.join(self.rgb_dir, f'{idx:05d}.png')
        rgb = Image.open(rgb_path).convert('RGB')

        # Depth
        depth_path = os.path.join(self.depth_dir, f'{idx:05d}.png')
        depth = Image.open(depth_path)
        if depth.mode in ('I', 'I;16', 'I;16B'):
            depth_arr = np.array(depth, dtype=np.float32)
            depth_arr = np.clip(depth_arr / 65535.0, 0.0, 1.0)
            depth = Image.fromarray(depth_arr, mode='F')
        else:
            depth = depth.convert('F')
            if np.array(depth).max() > 1.0:
                depth_arr = np.array(depth)
                depth = Image.fromarray(depth_arr / 255.0, mode='F')

        # Orth
        orth_path = os.path.join(self.orth_dir, f'{idx:05d}.png')
        orth = Image.open(orth_path)
        if orth.mode in ('I', 'I;16', 'I;16B'):
            orth_arr = np.array(orth, dtype=np.float32)
            orth_arr = np.clip(orth_arr / 65535.0, 0.0, 1.0)
            orth = Image.fromarray(orth_arr, mode='F')
        else:
            orth = orth.convert('F')
            if np.array(orth).max() > 1.0:
                orth_arr = np.array(orth)
                orth = Image.fromarray(orth_arr / 255.0, mode='F')

        return rgb, depth, orth

    def __getitem__(self, idx):
        """
        Returns:
            rgb: RGB image tensor [3, H, W]
            depth: Depth image tensor [1, H, W]
            orth: Orthogonal image tensor [1, H, W]
            label: Class label (0-14)
        """
        # Load images (tensor fast path or PNG fallback)
        if self.use_tensors:
            rgb, depth, orth = self._load_images_tensor(idx)
            images_already_resized = True
        else:
            rgb, depth, orth = self._load_images_png(idx)
            images_already_resized = False

        # ==================== TRAINING AUGMENTATION ====================
        if self.split == 'train':
            # 1. Synchronized Random Horizontal Flip (50%)
            if np.random.random() < 0.5:
                rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)
                depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
                orth = orth.transpose(Image.FLIP_LEFT_RIGHT)

            # 2. Synchronized Random Resized Crop (50% probability)
            if np.random.random() < 0.5:
                i, j, h, w = transforms.RandomResizedCrop.get_params(
                    rgb, scale=(0.9, 1.0), ratio=(0.95, 1.05)
                )
                rgb = transforms.functional.resized_crop(rgb, i, j, h, w, self.target_size)
                depth = transforms.functional.resized_crop(depth, i, j, h, w, self.target_size)
                orth = transforms.functional.resized_crop(orth, i, j, h, w, self.target_size)
            elif not images_already_resized:
                rgb = transforms.functional.resize(rgb, self.target_size)
                depth = transforms.functional.resize(depth, self.target_size)
                orth = transforms.functional.resize(orth, self.target_size)

            # 3. RGB-Only: Color Jitter (43% probability)
            if np.random.random() < 0.43:
                color_jitter = transforms.ColorJitter(
                    brightness=0.37,
                    contrast=0.37,
                    saturation=0.37,
                    hue=0.11
                )
                rgb = color_jitter(rgb)

            # 4. RGB-Only: Gaussian Blur (25% probability)
            if np.random.random() < 0.25:
                kernel_size = int(np.random.choice([3, 5, 7]))
                sigma = float(np.random.uniform(0.1, 1.7))
                rgb = transforms.functional.gaussian_blur(rgb, kernel_size=kernel_size, sigma=sigma)

            # 5. RGB-Only: Occasional Grayscale (17%)
            if np.random.random() < 0.17:
                rgb = transforms.functional.to_grayscale(rgb, num_output_channels=3)

            # 6. Depth & Orth: Appearance Augmentation (50% probability each)
            if np.random.random() < 0.5:
                depth_array = np.array(depth, dtype=np.float32)
                brightness_factor = np.random.uniform(0.75, 1.25)
                contrast_factor = np.random.uniform(0.75, 1.25)
                depth_array = (depth_array - 0.5) * contrast_factor + 0.5
                depth_array = depth_array * brightness_factor
                noise = np.random.normal(0, 0.06, depth_array.shape).astype(np.float32)
                depth_array = depth_array + noise
                depth_array = np.clip(depth_array, 0.0, 1.0).astype(np.float32)
                depth = Image.fromarray(depth_array, mode='F')

            if np.random.random() < 0.5:
                orth_array = np.array(orth, dtype=np.float32)
                brightness_factor = np.random.uniform(0.75, 1.25)
                contrast_factor = np.random.uniform(0.75, 1.25)
                orth_array = (orth_array - 0.5) * contrast_factor + 0.5
                orth_array = orth_array * brightness_factor
                noise = np.random.normal(0, 0.06, orth_array.shape).astype(np.float32)
                orth_array = orth_array + noise
                orth_array = np.clip(orth_array, 0.0, 1.0).astype(np.float32)
                orth = Image.fromarray(orth_array, mode='F')

        elif not images_already_resized:
            # Validation: just resize (no augmentation) — only needed for PNG path
            rgb = transforms.functional.resize(rgb, self.target_size)
            depth = transforms.functional.resize(depth, self.target_size)
            orth = transforms.functional.resize(orth, self.target_size)

        # ==================== NORMALIZATION ====================
        rgb = transforms.functional.to_tensor(rgb)
        rgb = transforms.functional.normalize(
            rgb,
            mean=[0.49829878533942046, 0.4667760665084003, 0.44289694564460663],
            std=[0.27731416732781294, 0.28601699847044426, 0.2899506179157605]
        )

        depth = transforms.functional.to_tensor(depth)
        depth = transforms.functional.normalize(
            depth, mean=[0.2908], std=[0.1504]
        )

        orth = transforms.functional.to_tensor(orth)
        orth = transforms.functional.normalize(
            orth, mean=[0.4944], std=[0.2065]
        )

        # 7. Post-normalization Random Erasing
        if self.split == 'train':
            if np.random.random() < 0.17:
                erasing = transforms.RandomErasing(
                    p=1.0, scale=(0.02, 0.10), ratio=(0.5, 2.0)
                )
                rgb = erasing(rgb)

            if np.random.random() < 0.1:
                erasing = transforms.RandomErasing(
                    p=1.0, scale=(0.02, 0.1), ratio=(0.5, 2.0)
                )
                depth = erasing(depth)

            if np.random.random() < 0.1:
                erasing = transforms.RandomErasing(
                    p=1.0, scale=(0.02, 0.1), ratio=(0.5, 2.0)
                )
                orth = erasing(orth)

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
