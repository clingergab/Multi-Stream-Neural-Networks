#!/usr/bin/env python3
"""
Test script using canonical dataset wrappers and refactored transforms.

This test demonstrates the proper use of:
1. Canonical MultiStreamWrapper from src.datasets
2. Refactored RGBtoRGBL (now a simple preprocessing class, not nn.Module)  
3. Proper separation of preprocessing and model computation
4. Loading data from the actual project data folder
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

# Add src to path
sys.path.insert(0, '/Users/gclinger/Documents/projects/Multi-Stream-Neural-Networks/src')

# Import canonical components
from src.models.basic_multi_channel.multi_channel_model import MultiChannelNetwork
from src.utils.device_utils import get_device, DeviceManager
from src.transforms.rgb_to_rgbl import RGBtoRGBL
from src.datasets.dataset_wrappers import MultiStreamWrapper

def test_canonical_components():
    """Test the canonical dataset wrapper and refactored transforms."""
    print("=== Testing Canonical Dataset Wrapper and Refactored Transforms ===\n")
    
    # Initialize device
    device = get_device()
    device_manager = DeviceManager()
    print(f"Using device: {device}")
    print(f"Device manager initialized successfully\n")
    
    # Data paths
    data_root = "/Users/gclinger/Documents/projects/Multi-Stream-Neural-Networks/data"
    mnist_path = os.path.join(data_root, "MNIST")
    
    print(f"Loading MNIST from: {mnist_path}")
    
    # Standard transforms for base dataset
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
    ])
    
    # Load MNIST dataset (download if needed for testing)
    mnist_train = datasets.MNIST(
        root=mnist_path, 
        train=True, 
        download=True,  # Download if not in PyTorch format
        transform=transforms.ToTensor()  # We'll handle normalization later
    )
    
    # Create a small subset for testing
    subset_size = 1000
    indices = torch.randperm(len(mnist_train))[:subset_size]
    mnist_subset = torch.utils.data.Subset(mnist_train, indices)
    
    print(f"Created subset with {len(mnist_subset)} samples")
    
    # Initialize transforms (now they're not nn.Modules)
    rgbl_transform = RGBtoRGBL()
    
    # Test the refactored transforms
    print("\n=== Testing Refactored Transforms ===")
    
    # Get a sample
    sample_image, sample_label = mnist_subset[0]
    print(f"Original sample shape: {sample_image.shape}, label: {sample_label}")
    
    # Convert grayscale to RGB for testing
    if sample_image.shape[0] == 1:
        sample_rgb = sample_image.repeat(3, 1, 1)
        print(f"Converted to RGB shape: {sample_rgb.shape}")
    
    # Test RGBtoRGBL (should work as a simple callable, not nn.Module)
    try:
        rgb_part, l_part = rgbl_transform(sample_rgb)
        print(f"RGB stream shape: {rgb_part.shape}, L stream shape: {l_part.shape}")
        assert rgb_part.shape[0] == 3, f"Expected 3 RGB channels, got {rgb_part.shape[0]}"
        assert l_part.shape[0] == 1, f"Expected 1 L channel, got {l_part.shape[0]}"
        
        print("✅ Refactored transforms work correctly!")
        
    except Exception as e:
        print(f"❌ Transform test failed: {e}")
        return False
    
    # Create canonical MultiStreamWrapper
    print("\n=== Testing Canonical MultiStreamWrapper ===")
    
    def grayscale_to_rgb_transform(img):
        """Convert grayscale MNIST to RGB."""
        if isinstance(img, torch.Tensor) and img.shape[0] == 1:
            return img.repeat(3, 1, 1)
        return img
    
    def normalize_transform(img):
        """Apply MNIST normalization."""
        return transforms.Normalize((0.1307,), (0.3081,))(img)
    
    # Use the canonical MultiStreamWrapper
    multistream_dataset = MultiStreamWrapper(
        base_dataset=mnist_subset,
        color_transform=transforms.Compose([
            grayscale_to_rgb_transform,
            normalize_transform
        ]),
        brightness_transform=normalize_transform
    )
    
    print(f"Created MultiStreamWrapper with {len(multistream_dataset)} samples")
    
    # Test the wrapper
    try:
        sample_data = multistream_dataset[0]
        print(f"MultiStream sample keys: {sample_data.keys()}")
        print(f"Color input shape: {sample_data['color'].shape}")
        print(f"Brightness input shape: {sample_data['brightness'].shape}")
        print(f"Target: {sample_data['target']}")
        
        assert 'color' in sample_data, "Missing 'color' key"
        assert 'brightness' in sample_data, "Missing 'brightness' key"
        assert 'target' in sample_data, "Missing 'target' key"
        
        print("✅ Canonical MultiStreamWrapper works correctly!")
        
    except Exception as e:
        print(f"❌ MultiStreamWrapper test failed: {e}")
        return False
    
    # Create dataloader
    print("\n=== Testing DataLoader Integration ===")
    
    def multistream_collate_fn(batch):
        """Custom collate function for multistream data."""
        color_inputs = torch.stack([item['color'] for item in batch])
        brightness_inputs = torch.stack([item['brightness'] for item in batch])
        targets = torch.tensor([item['target'] for item in batch])
        
        return {
            'color': color_inputs,
            'brightness': brightness_inputs,
            'target': targets
        }
    
    dataloader = DataLoader(
        multistream_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=multistream_collate_fn
    )
    
    # Test dataloader
    try:
        batch = next(iter(dataloader))
        print(f"Batch keys: {batch.keys()}")
        print(f"Color batch shape: {batch['color'].shape}")
        print(f"Brightness batch shape: {batch['brightness'].shape}")
        print(f"Target batch shape: {batch['target'].shape}")
        
        assert batch['color'].shape[1] == 3, f"Expected 3 color channels, got {batch['color'].shape[1]}"
        assert batch['brightness'].shape[1] == 1, f"Expected 1 brightness channel, got {batch['brightness'].shape[1]}"
        
        print("✅ DataLoader integration works correctly!")
        
    except Exception as e:
        print(f"❌ DataLoader test failed: {e}")
        return False
    
    # Test model integration
    print("\n=== Testing Model Integration ===")
    
    try:
        # Create model using canonical class
        model = MultiChannelNetwork(
            num_classes=10,
            input_channels=3,
            hidden_channels=64
        ).to(device)
        
        print(f"Created model on device: {next(model.parameters()).device}")
        
        # Test forward pass
        batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                       for k, v in batch.items()}
        
        with torch.no_grad():
            outputs = model(batch_device['color'], batch_device['brightness'])
            print(f"Model output shape: {outputs.shape}")
            
            assert outputs.shape[0] == batch['color'].shape[0], "Batch size mismatch"
            assert outputs.shape[1] == 10, f"Expected 10 classes, got {outputs.shape[1]}"
        
        print("✅ Model integration works correctly!")
        
    except Exception as e:
        print(f"❌ Model integration test failed: {e}")
        return False
    
    # Quick training test
    print("\n=== Quick Training Test ===")
    
    try:
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Single training step
        batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                       for k, v in batch.items()}
        
        optimizer.zero_grad()
        outputs = model(batch_device['color'], batch_device['brightness'])
        loss = criterion(outputs, batch_device['target'])
        loss.backward()
        optimizer.step()
        
        print(f"Training step completed. Loss: {loss.item():.4f}")
        print("✅ Training integration works correctly!")
        
    except Exception as e:
        print(f"❌ Training test failed: {e}")
        return False
    
    print("\n=== All Tests Passed! ===")
    print("✅ Canonical dataset wrapper and refactored transforms are working correctly")
    print("✅ All components are properly integrated and using canonical implementations")
    
    return True

if __name__ == "__main__":
    success = test_canonical_components()
    if success:
        print(f"\n🎉 SUCCESS: All canonical components are working properly!")
        sys.exit(0)
    else:
        print(f"\n❌ FAILURE: Some tests failed")
        sys.exit(1)
