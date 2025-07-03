#!/usr/bin/env python
"""
Simple test script to verify predict() method in our multi-stream models.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from src.models.basic_multi_channel.base_multi_channel_network import BaseMultiChannelNetwork
from src.models.basic_multi_channel.multi_channel_resnet_network import MultiChannelResNetNetwork

# Test BaseMultiChannelNetwork predict() with both direct data and DataLoader
def test_base_multi_channel_predict():
    print("\n=== Testing BaseMultiChannelNetwork predict() ===")
    
    # Create a small model
    model = BaseMultiChannelNetwork(
        color_input_size=3*32*32,  # CIFAR dimensions flattened
        brightness_input_size=32*32,
        hidden_sizes=[64, 32],
        num_classes=10,
        dropout=0.2,
        device='cpu'  # Use CPU for testing
    )
    model.compile()
    
    # Test with direct data in flattened format (2D)
    print("Testing with direct 2D data:")
    batch_size = 4
    color_data = torch.rand(batch_size, 3*32*32)
    brightness_data = torch.rand(batch_size, 32*32)
    
    predictions = model.predict(color_data, brightness_data)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions: {predictions}")
    
    # Test with direct data in image format (4D)
    print("\nTesting with direct 4D data (should be flattened automatically):")
    color_data_4d = torch.rand(batch_size, 3, 32, 32)
    brightness_data_4d = torch.rand(batch_size, 1, 32, 32)
    
    predictions = model.predict(color_data_4d, brightness_data_4d)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions: {predictions}")
    
    # Test with DataLoader
    print("\nTesting with DataLoader:")
    dataset = TensorDataset(color_data_4d, brightness_data_4d)
    loader = DataLoader(dataset, batch_size=2)
    
    predictions = model.predict(loader)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions: {predictions}")
    
    # Test predict_proba
    print("\nTesting predict_proba:")
    proba = model.predict_proba(color_data_4d, brightness_data_4d)
    print(f"Probabilities shape: {proba.shape}")
    print(f"Probabilities sum to 1: {np.allclose(np.sum(proba, axis=1), 1.0)}")
    
    return True

# Test MultiChannelResNetNetwork predict() with both direct data and DataLoader
def test_multi_channel_resnet_predict():
    print("\n=== Testing MultiChannelResNetNetwork predict() ===")
    
    # Create a small model
    model = MultiChannelResNetNetwork(
        num_classes=10,
        color_input_channels=3,
        brightness_input_channels=1,
        num_blocks=[1, 1, 1, 1],  # Minimal blocks for testing
        reduce_architecture=True,  # Use smaller architecture
        device='cpu'  # Use CPU for testing
    )
    model.compile()
    
    # Test with direct data in image format (4D)
    print("Testing with direct 4D data:")
    batch_size = 4
    color_data_4d = torch.rand(batch_size, 3, 32, 32)
    brightness_data_4d = torch.rand(batch_size, 1, 32, 32)
    
    predictions = model.predict(color_data_4d, brightness_data_4d)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions: {predictions}")
    
    # Test with DataLoader
    print("\nTesting with DataLoader:")
    dataset = TensorDataset(color_data_4d, brightness_data_4d)
    loader = DataLoader(dataset, batch_size=2)
    
    predictions = model.predict(loader)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions: {predictions}")
    
    # Test predict_proba
    print("\nTesting predict_proba:")
    proba = model.predict_proba(color_data_4d, brightness_data_4d)
    print(f"Probabilities shape: {proba.shape}")
    print(f"Probabilities sum to 1: {np.allclose(np.sum(proba, axis=1), 1.0)}")
    
    # Try with 2D data - should raise an error
    print("\nTesting with 2D data (should raise an error):")
    color_data_2d = torch.rand(batch_size, 3*32*32)
    brightness_data_2d = torch.rand(batch_size, 32*32)
    
    try:
        predictions = model.predict(color_data_2d, brightness_data_2d)
        print("ERROR: Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"Successfully caught error: {e}")
    
    return True

if __name__ == "__main__":
    print("Testing predict() methods in multi-stream models...")
    
    success_base = test_base_multi_channel_predict()
    success_resnet = test_multi_channel_resnet_predict()
    
    if success_base and success_resnet:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
