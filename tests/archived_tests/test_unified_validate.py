#!/usr/bin/env python3
"""
Test script to verify the unified _validate method works correctly for both model types.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models.basic_multi_channel.base_multi_channel_network import BaseMultiChannelNetwork
from models.basic_multi_channel.multi_channel_resnet_network import MultiChannelResNetNetwork


def create_test_data(num_samples=100, image_format=True):
    """Create test data in either image or tabular format."""
    if image_format:
        # Image format: (N, C, H, W)
        color_data = torch.randn(num_samples, 3, 32, 32)
        brightness_data = torch.randn(num_samples, 1, 32, 32)
    else:
        # Tabular format: (N, features)
        color_data = torch.randn(num_samples, 3072)  # 3*32*32
        brightness_data = torch.randn(num_samples, 1024)  # 1*32*32
    
    labels = torch.randint(0, 10, (num_samples,))
    return color_data, brightness_data, labels


def test_base_multi_channel_validate():
    """Test _validate method for BaseMultiChannelNetwork (tabular model)."""
    print("Testing BaseMultiChannelNetwork _validate method...")
    
    # Create model
    model = BaseMultiChannelNetwork(
        color_input_size=3072,
        brightness_input_size=1024,
        hidden_sizes=[256, 128],
        num_classes=10
    )
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='cross_entropy',
        metrics=['accuracy']
    )
    
    # Create test data in image format (will be reshaped automatically by _validate)
    color_data, brightness_data, labels = create_test_data(50, image_format=True)
    
    # Create DataLoader
    dataset = TensorDataset(color_data, brightness_data, labels)
    val_loader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    # Test validation
    try:
        val_loss, val_acc = model._validate(val_loader)
        print("  ✓ BaseMultiChannelNetwork validation successful")
        print(f"    Validation loss: {val_loss:.4f}")
        print(f"    Validation accuracy: {val_acc:.4f}")
        return True
    except Exception as e:
        print(f"  ✗ BaseMultiChannelNetwork validation failed: {e}")
        return False


def test_resnet_validate():
    """Test _validate method for MultiChannelResNetNetwork (CNN model)."""
    print("Testing MultiChannelResNetNetwork _validate method...")
    
    # Create model
    model = MultiChannelResNetNetwork(
        num_classes=10,
        color_channels=3,
        brightness_channels=1,
        initial_filters=16,
        block_structure=[1, 1, 1],
        input_height=32,
        input_width=32
    )
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='cross_entropy',
        metrics=['accuracy']
    )
    
    # Create test data in image format
    color_data, brightness_data, labels = create_test_data(50, image_format=True)
    
    # Create DataLoader
    dataset = TensorDataset(color_data, brightness_data, labels)
    val_loader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    # Test validation
    try:
        val_loss, val_acc = model._validate(val_loader)
        print("  ✓ MultiChannelResNetNetwork validation successful")
        print(f"    Validation loss: {val_loss:.4f}")
        print(f"    Validation accuracy: {val_acc:.4f}")
        return True
    except Exception as e:
        print(f"  ✗ MultiChannelResNetNetwork validation failed: {e}")
        return False


def test_inheritance_validation():
    """Test that both models inherit the _validate method from parent."""
    print("Testing inheritance of _validate method...")
    
    # Create models
    tabular_model = BaseMultiChannelNetwork(
        color_input_size=3072,
        brightness_input_size=1024,
        hidden_sizes=[128],
        num_classes=10
    )
    
    cnn_model = MultiChannelResNetNetwork(
        num_classes=10,
        color_channels=3,
        brightness_channels=1,
        initial_filters=16,
        block_structure=[1],
        input_height=32,
        input_width=32
    )
    
    # Check that _validate method exists and is inherited from parent
    from models.base import BaseMultiStreamModel
    
    tabular_validate = getattr(tabular_model.__class__, '_validate', None)
    cnn_validate = getattr(cnn_model.__class__, '_validate', None)
    parent_validate = getattr(BaseMultiStreamModel, '_validate', None)
    
    if tabular_validate is parent_validate:
        print("  ✓ BaseMultiChannelNetwork inherits _validate from parent")
    else:
        print("  ✗ BaseMultiChannelNetwork has its own _validate method")
        return False
    
    if cnn_validate is parent_validate:
        print("  ✓ MultiChannelResNetNetwork inherits _validate from parent")
    else:
        print("  ✗ MultiChannelResNetNetwork has its own _validate method")
        return False
    
    return True


def test_mixed_precision_support():
    """Test that mixed precision works correctly in unified _validate."""
    print("Testing mixed precision support in unified _validate...")
    
    if not torch.cuda.is_available():
        print("  ⚠ CUDA not available, skipping mixed precision test")
        return True
    
    # Create model on GPU
    model = BaseMultiChannelNetwork(
        color_input_size=3072,
        brightness_input_size=1024,
        hidden_sizes=[128],
        num_classes=10,
        device='cuda'
    )
    
    # Enable mixed precision
    model.use_mixed_precision = True
    model.scaler = torch.cuda.amp.GradScaler()
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='cross_entropy',
        metrics=['accuracy']
    )
    
    # Create test data
    color_data, brightness_data, labels = create_test_data(32, image_format=True)
    dataset = TensorDataset(color_data, brightness_data, labels)
    val_loader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    try:
        val_loss, val_acc = model._validate(val_loader)
        print("  ✓ Mixed precision validation successful")
        print(f"    Validation loss: {val_loss:.4f}")
        print(f"    Validation accuracy: {val_acc:.4f}")
        return True
    except Exception as e:
        print(f"  ✗ Mixed precision validation failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Testing Unified _validate Method Implementation")
    print("=" * 50)
    
    tests = [
        test_inheritance_validation,
        test_base_multi_channel_validate,
        test_resnet_validate,
        test_mixed_precision_support,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"Test {test.__name__} failed with exception: {e}")
            print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The unified _validate method is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
