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

        # Load orthogonal image
        orth_path = os.path.join(self.orth_dir, f'{idx:05d}.png')
        orth = Image.open(orth_path)

        # Helper to convert single channel images
        def process_single_channel(img):
            if img.mode in ('I', 'I;16', 'I;16B'):
                img_array = np.array(img, dtype=np.float32)
                if img_array.max() > 0:
                    img_array = (img_array / img_array.max() * 255).astype(np.uint8)
                else:
                    img_array = img_array.astype(np.uint8)
                return Image.fromarray(img_array, mode='L')
            elif img.mode == 'RGB':
                return img.convert('L')
            elif img.mode != 'L':
                return img.convert('L')
            return img

        depth = process_single_channel(depth)
        # orth is kept as 16-bit (I;16) to preserve global consistency
        # orth = process_single_channel(orth)  <-- REMOVED

        # ==================== TRAINING AUGMENTATION ====================
        if self.train:
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
            else:
                # No crop - preserve full scene context
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

            # 6. Depth Only: Appearance Augmentation
            # (Skipping orth appearance aug to preserve geometric meaning for now)
            if np.random.random() < 0.5:
                img_array = np.array(depth, dtype=np.float32)

                # Apply brightness and contrast
                brightness_factor = np.random.uniform(0.75, 1.25)  # ±25%
                contrast_factor = np.random.uniform(0.75, 1.25)    # ±25%

                # Apply contrast then brightness
                img_array = (img_array - 127.5) * contrast_factor + 127.5
                img_array = img_array * brightness_factor

                # Add Gaussian noise
                noise = np.random.normal(0, 15, img_array.shape)
                img_array = img_array + noise

                img_array = np.clip(img_array, 0, 255)
                depth = Image.fromarray(img_array.astype(np.uint8), mode='L')

        else:
            # Validation: just resize (no augmentation)
            rgb = transforms.functional.resize(rgb, self.target_size)
            depth = transforms.functional.resize(depth, self.target_size)
            orth = transforms.functional.resize(orth, self.target_size)

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
        
        # Orthogonal normalization
        # 1. Convert 16-bit PIL to float tensor [0, 1]
        # Note: to_tensor handles uint8 but not uint16 correctly for scaling
        orth_arr = np.array(orth, dtype=np.float32)
        orth = torch.from_numpy(orth_arr).unsqueeze(0) / 65535.0
        
        # 2. Normalize to [-1, 1]
        # Since 0 maps to 0.5 in [0,1] range, we use mean=0.5
        # std=0.5 maps [0, 1] -> [-1, 1]
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
