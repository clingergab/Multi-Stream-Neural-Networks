#!/usr/bin/env python3
"""
Simple test script for the unified fit method with both array input and DataLoader input.
Tests both BaseMultiChannelNetwork and MultiChannelResNetNetwork models.
"""

import sys
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import models
from src.models.basic_multi_channel.base_multi_channel_network import BaseMultiChannelNetwork
from src.models.basic_multi_channel.multi_channel_resnet_network import MultiChannelResNetNetwork
from src.transforms.augmentation import CIFAR100Augmentation

# Create small random datasets
def generate_test_data(n_samples=100, feature_dim=32, n_classes=5, image_data=False):
    if image_data:
        # Image data: (samples, channels, height, width)
        color_data = np.random.rand(n_samples, 3, feature_dim, feature_dim).astype(np.float32)
        brightness_data = np.random.rand(n_samples, 1, feature_dim, feature_dim).astype(np.float32)
    else:
        # Flat data: (samples, features)
        color_data = np.random.rand(n_samples, feature_dim * feature_dim * 3).astype(np.float32)
        brightness_data = np.random.rand(n_samples, feature_dim * feature_dim).astype(np.float32)
    
    # Random labels
    labels = np.random.randint(0, n_classes, size=n_samples).astype(np.int64)
    
    return color_data, brightness_data, labels

def create_dataloader(color_data, brightness_data, labels, batch_size=16, apply_augmentation=False):
    """Create a DataLoader with or without augmentation."""
    # Convert to tensors
    color_tensor = torch.tensor(color_data, dtype=torch.float32)
    brightness_tensor = torch.tensor(brightness_data, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    if apply_augmentation and len(color_data.shape) == 4:  # For image data only
        # Apply augmentation transform
        print("Creating DataLoader with augmentation...")
        augmentation = CIFAR100Augmentation(apply_to_brightness=True)
        
        # Create dataset with augmentation
        class AugmentedDataset(torch.utils.data.Dataset):
            def __init__(self, color, brightness, labels, transform=None):
                self.color = color
                self.brightness = brightness
                self.labels = labels
                self.transform = transform
                
            def __len__(self):
                return len(self.labels)
                
            def __getitem__(self, idx):
                c, b = self.color[idx], self.brightness[idx]
                if self.transform:
                    # Use the updated interface that accepts both images
                    c, b = self.transform(c, b)
                return c, b, self.labels[idx]
        
        dataset = AugmentedDataset(color_tensor, brightness_tensor, labels_tensor, transform=augmentation)
    else:
        # Create regular dataset without augmentation
        print("Creating DataLoader without augmentation...")
        dataset = TensorDataset(color_tensor, brightness_tensor, labels_tensor)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    return dataloader

def test_base_multi_channel_network():
    """Test BaseMultiChannelNetwork with both input methods."""
    print("\n=== Testing BaseMultiChannelNetwork ===")
    
    # Generate test data (flat)
    color_data, brightness_data, labels = generate_test_data(n_samples=100, feature_dim=16, n_classes=5)
    
    # Split into train/val
    train_idx = int(len(labels) * 0.8)
    train_color, train_brightness, train_labels = color_data[:train_idx], brightness_data[:train_idx], labels[:train_idx]
    val_color, val_brightness, val_labels = color_data[train_idx:], brightness_data[train_idx:], labels[train_idx:]
    
    # Create model
    model = BaseMultiChannelNetwork(
        color_input_size=train_color.shape[1],
        brightness_input_size=train_brightness.shape[1],
        hidden_sizes=[64, 32],
        num_classes=5,
        use_shared_classifier=True,
        device='cpu'
    )
    
    # Test 1: Train with direct array data
    print("\n[Test 1] Training with direct array data:")
    history1 = model.fit(
        train_color_data=train_color,
        train_brightness_data=train_brightness,
        train_labels=train_labels,
        val_color_data=val_color,
        val_brightness_data=val_brightness,
        val_labels=val_labels,
        batch_size=16,
        epochs=2,
        verbose=1
    )
    
    # Test 2: Train with DataLoader
    print("\n[Test 2] Training with DataLoader:")
    train_loader = create_dataloader(train_color, train_brightness, train_labels, batch_size=16)
    val_loader = create_dataloader(val_color, val_brightness, val_labels, batch_size=16)
    
    history2 = model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=2,
        verbose=1
    )
    
    print("\n✅ BaseMultiChannelNetwork tests completed successfully!")
    return True

def test_multi_channel_resnet_network():
    """Test MultiChannelResNetNetwork with both input methods."""
    print("\n=== Testing MultiChannelResNetNetwork ===")
    
    # Generate test data (image format)
    color_data, brightness_data, labels = generate_test_data(
        n_samples=100, feature_dim=16, n_classes=5, image_data=True
    )
    
    # Split into train/val
    train_idx = int(len(labels) * 0.8)
    train_color, train_brightness, train_labels = color_data[:train_idx], brightness_data[:train_idx], labels[:train_idx]
    val_color, val_brightness, val_labels = color_data[train_idx:], brightness_data[train_idx:], labels[train_idx:]
    
    # Create model
    model = MultiChannelResNetNetwork(
        num_classes=5,
        color_input_channels=3,
        brightness_input_channels=1,
        num_blocks=[1, 1, 1, 1],  # Small model for testing
        use_shared_classifier=True,
        device='cpu'
    )
    
    # Test 1: Train with direct array data
    print("\n[Test 1] Training with direct array data:")
    history1 = model.fit(
        train_color_data=train_color,
        train_brightness_data=train_brightness,
        train_labels=train_labels,
        val_color_data=val_color,
        val_brightness_data=val_brightness,
        val_labels=val_labels,
        batch_size=16,
        epochs=2,
        verbose=1
    )
    
    # Test 2: Train with regular DataLoader (no augmentation)
    print("\n[Test 2] Training with DataLoader (no augmentation):")
    train_loader = create_dataloader(train_color, train_brightness, train_labels, batch_size=16)
    val_loader = create_dataloader(val_color, val_brightness, val_labels, batch_size=16)
    
    history2 = model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=2,
        verbose=1
    )
    
    # Test 3: Train with augmented DataLoader
    print("\n[Test 3] Training with DataLoader (with augmentation):")
    train_loader_aug = create_dataloader(
        train_color, train_brightness, train_labels, batch_size=16, apply_augmentation=True
    )
    
    history3 = model.fit(
        train_loader=train_loader_aug,
        val_loader=val_loader,
        epochs=2,
        verbose=1
    )
    
    print("\n✅ MultiChannelResNetNetwork tests completed successfully!")
    return True

def main():
    """Run all tests."""
    print("🚀 Starting unified fit method tests")
    
    success = test_base_multi_channel_network()
    if success:
        success = test_multi_channel_resnet_network()
        
    if success:
        print("\n🎉 All tests passed! The unified fit method works correctly.")
        return 0
    else:
        print("\n❌ Tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
