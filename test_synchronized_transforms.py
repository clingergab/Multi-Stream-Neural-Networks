#!/usr/bin/env python3
"""
Test script to verify synchronized augmentation in DualChannelDataset.
"""

import torch
import torchvision.transforms as transforms
import numpy as np
from src.data_utils.dual_channel_dataset import DualChannelDataset
from torch.utils.data import DataLoader

def test_synchronized_augmentation():
    """Test that RGB and brightness channels receive identical geometric transforms."""
    print("Testing synchronized augmentation...")
    
    # Create synthetic data - simple patterns to easily verify synchronization
    batch_size = 4
    height, width = 32, 32
    
    # Create test images with distinct patterns
    rgb_data = torch.zeros(batch_size, 3, height, width)
    labels = torch.arange(batch_size)
    
    # Create distinctive patterns for each image
    for i in range(batch_size):
        # Create vertical stripes in red channel
        rgb_data[i, 0, :, ::4] = 1.0
        # Create horizontal stripes in green channel  
        rgb_data[i, 1, ::4, :] = 1.0
        # Create diagonal pattern in blue channel
        for j in range(min(height, width)):
            if j < height and j < width:
                rgb_data[i, 2, j, j] = 1.0
    
    # Define augmentation that should be synchronized
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),  # Always flip for predictable test
        transforms.RandomRotation(degrees=90),   # Large rotation to see effect
    ])
    
    # Create dataset
    dataset = DualChannelDataset(
        rgb_data=rgb_data,
        labels=labels,
        transform=transform,
        auto_brightness=True
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    # Test multiple samples to check consistency
    for idx in range(min(3, len(dataset))):
        print(f"\nTesting sample {idx}:")
        
        # Get the same sample multiple times to check transform randomness
        rgb1, brightness1, label1 = dataset[idx]
        rgb2, brightness2, label2 = dataset[idx]
        
        print(f"  RGB shape: {rgb1.shape}, Brightness shape: {brightness1.shape}")
        print(f"  Label: {label1}")
        
        # Check that transforms were applied (should be different from original)
        original_rgb = rgb_data[idx]
        original_brightness = dataset._rgb_converter.get_brightness(original_rgb)
        
        # Transforms should change the data
        rgb_changed = not torch.allclose(rgb1, original_rgb, atol=1e-6)
        brightness_changed = not torch.allclose(brightness1, original_brightness, atol=1e-6)
        
        print(f"  RGB transformed: {rgb_changed}")
        print(f"  Brightness transformed: {brightness_changed}")
        
        # Different calls should give different results (due to random transforms)
        different_results = not torch.allclose(rgb1, rgb2, atol=1e-6)
        print(f"  Random transforms working: {different_results}")
        
        # Verify brightness is correctly computed from RGB after transforms
        # (This is the key test - brightness should match the transformed RGB)
        expected_brightness = dataset._rgb_converter.get_brightness(rgb1)
        brightness_consistent = torch.allclose(brightness1, expected_brightness, atol=1e-4)
        print(f"  RGB-brightness consistency: {brightness_consistent}")
        
        if not brightness_consistent:
            print(f"    WARNING: Brightness not consistent with RGB!")
            print(f"    Max difference: {torch.max(torch.abs(brightness1 - expected_brightness))}")

def test_dataloader_integration():
    """Test that the dataset works properly with DataLoader."""
    print("\n" + "="*50)
    print("Testing DataLoader integration...")
    
    # Create larger synthetic dataset
    batch_size = 8
    num_samples = 20
    height, width = 32, 32
    
    rgb_data = torch.randn(num_samples, 3, height, width)
    labels = torch.randint(0, 10, (num_samples,))
    
    # Simple augmentation
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
    ])
    
    dataset = DualChannelDataset(
        rgb_data=rgb_data,
        labels=labels,
        transform=transform,
        auto_brightness=True
    )
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    print(f"DataLoader created with {len(dataloader)} batches")
    
    # Test a few batches
    for i, (rgb_batch, brightness_batch, label_batch) in enumerate(dataloader):
        print(f"  Batch {i}: RGB {rgb_batch.shape}, Brightness {brightness_batch.shape}, Labels {label_batch.shape}")
        
        # Verify brightness is single channel
        assert brightness_batch.shape[1] == 1, f"Expected 1 brightness channel, got {brightness_batch.shape[1]}"
        
        # Verify batch dimensions match
        assert rgb_batch.shape[0] == brightness_batch.shape[0] == label_batch.shape[0], \
            "Batch dimensions don't match"
        
        if i >= 2:  # Test just a few batches
            break
    
    print("DataLoader integration test passed!")

def test_no_transform():
    """Test dataset without transforms."""
    print("\n" + "="*50)
    print("Testing dataset without transforms...")
    
    batch_size = 4
    height, width = 32, 32
    
    rgb_data = torch.randn(batch_size, 3, height, width)
    labels = torch.arange(batch_size)
    
    dataset = DualChannelDataset(
        rgb_data=rgb_data,
        labels=labels,
        transform=None,  # No transforms
        auto_brightness=True
    )
    
    # Test that data is unchanged
    for idx in range(len(dataset)):
        rgb, brightness, label = dataset[idx]
        
        # RGB should be identical to original
        assert torch.allclose(rgb, rgb_data[idx]), "RGB should be unchanged without transforms"
        
        # Brightness should match computed brightness
        expected_brightness = dataset._rgb_converter.get_brightness(rgb_data[idx])
        assert torch.allclose(brightness, expected_brightness), "Brightness should match computed value"
        
        print(f"  Sample {idx}: No-transform test passed")

if __name__ == "__main__":
    test_synchronized_augmentation()
    test_dataloader_integration()
    test_no_transform()
    
    print("\n" + "="*50)
    print("All tests completed!")
