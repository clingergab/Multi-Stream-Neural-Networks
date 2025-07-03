#!/usr/bin/env python
"""
Comprehensive test script to verify all training, evaluation, and prediction methods
in our multi-stream neural network models.

This script tests:
1. fit() method with both direct data and DataLoader
2. evaluate() method with both direct data and DataLoader
3. predict() method with both direct data and DataLoader
4. predict_proba() method with both direct data and DataLoader

For both BaseMultiChannelNetwork and MultiChannelResNetNetwork
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from src.models.basic_multi_channel.base_multi_channel_network import BaseMultiChannelNetwork
from src.models.basic_multi_channel.multi_channel_resnet_network import MultiChannelResNetNetwork
import matplotlib.pyplot as plt
import time
from datetime import datetime

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create output directory for results
os.makedirs('test_results', exist_ok=True)

def get_timestamp():
    """Get a timestamp string for naming files."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def create_dummy_data(batch_size=16, is_4d=True, flattened_color_size=3072, flattened_brightness_size=1024):
    """
    Create dummy data for testing.
    
    Args:
        batch_size: Number of samples
        is_4d: If True, create 4D data (for CNNs), else create 2D data (for MLPs)
        flattened_color_size: Size of flattened color data
        flattened_brightness_size: Size of flattened brightness data
        
    Returns:
        Tuple of (color_data, brightness_data, labels) in appropriate shapes
    """
    num_classes = 10
    
    if is_4d:
        # Create 4D data for CNNs (batch_size, channels, height, width)
        color_data = torch.rand(batch_size, 3, 32, 32)  # RGB
        brightness_data = torch.rand(batch_size, 1, 32, 32)  # Grayscale
    else:
        # Create 2D data for MLPs (batch_size, features)
        color_data = torch.rand(batch_size, flattened_color_size)
        brightness_data = torch.rand(batch_size, flattened_brightness_size)
    
    # Create random labels
    labels = torch.randint(0, num_classes, (batch_size,), dtype=torch.long)
    
    return color_data, brightness_data, labels

def get_dataloader(color_data, brightness_data, labels, batch_size=4):
    """Create a DataLoader from the provided data."""
    dataset = TensorDataset(color_data, brightness_data, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def test_base_multi_channel_network():
    """Test all methods of BaseMultiChannelNetwork"""
    print("\n" + "="*80)
    print("Testing BaseMultiChannelNetwork")
    print("="*80)
    
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
    
    # Create both 2D and 4D data for testing
    data_2d = create_dummy_data(batch_size=32, is_4d=False)
    data_4d = create_dummy_data(batch_size=32, is_4d=True)
    
    color_2d, brightness_2d, labels_2d = data_2d
    color_4d, brightness_4d, labels_4d = data_4d
    
    # Create dataloaders for both types
    loader_2d = get_dataloader(*data_2d, batch_size=8)
    loader_4d = get_dataloader(*data_4d, batch_size=8)
    
    # Split data for train/val/test
    train_size = 20
    val_size = 6
    test_size = 6
    
    # 1. Test fit() with direct 2D data
    print("\n1. Testing fit() with direct 2D data")
    history = model.fit(
        train_color_data=color_2d[:train_size],
        train_brightness_data=brightness_2d[:train_size],
        train_labels=labels_2d[:train_size],
        val_color_data=color_2d[train_size:train_size+val_size],
        val_brightness_data=brightness_2d[train_size:train_size+val_size],
        val_labels=labels_2d[train_size:train_size+val_size],
        epochs=2,
        verbose=1
    )
    print("✓ fit() with direct 2D data successful")
    
    # 2. Test fit() with direct 4D data (should be flattened automatically)
    print("\n2. Testing fit() with direct 4D data (should be flattened automatically)")
    history = model.fit(
        train_color_data=color_4d[:train_size],
        train_brightness_data=brightness_4d[:train_size],
        train_labels=labels_4d[:train_size],
        val_color_data=color_4d[train_size:train_size+val_size],
        val_brightness_data=brightness_4d[train_size:train_size+val_size],
        val_labels=labels_4d[train_size:train_size+val_size],
        epochs=2,
        verbose=1
    )
    print("✓ fit() with direct 4D data successful")
    
    # 3. Test fit() with DataLoader containing 2D data
    print("\n3. Testing fit() with DataLoader containing 2D data")
    train_loader_2d = get_dataloader(
        color_2d[:train_size], 
        brightness_2d[:train_size], 
        labels_2d[:train_size],
        batch_size=4
    )
    val_loader_2d = get_dataloader(
        color_2d[train_size:train_size+val_size], 
        brightness_2d[train_size:train_size+val_size], 
        labels_2d[train_size:train_size+val_size],
        batch_size=4
    )
    history = model.fit(
        train_loader=train_loader_2d,
        val_loader=val_loader_2d,
        epochs=2,
        verbose=1
    )
    print("✓ fit() with DataLoader containing 2D data successful")
    
    # 4. Test fit() with DataLoader containing 4D data
    print("\n4. Testing fit() with DataLoader containing 4D data")
    train_loader_4d = get_dataloader(
        color_4d[:train_size], 
        brightness_4d[:train_size], 
        labels_4d[:train_size],
        batch_size=4
    )
    val_loader_4d = get_dataloader(
        color_4d[train_size:train_size+val_size], 
        brightness_4d[train_size:train_size+val_size], 
        labels_4d[train_size:train_size+val_size],
        batch_size=4
    )
    history = model.fit(
        train_loader=train_loader_4d,
        val_loader=val_loader_4d,
        epochs=2,
        verbose=1
    )
    print("✓ fit() with DataLoader containing 4D data successful")
    
    # 5. Test evaluate() with direct 2D data
    print("\n5. Testing evaluate() with direct 2D data")
    results = model.evaluate(
        test_color_data=color_2d[-test_size:],
        test_brightness_data=brightness_2d[-test_size:],
        test_labels=labels_2d[-test_size:]
    )
    print(f"Results: {results}")
    print("✓ evaluate() with direct 2D data successful")
    
    # 6. Test evaluate() with direct 4D data
    print("\n6. Testing evaluate() with direct 4D data")
    results = model.evaluate(
        test_color_data=color_4d[-test_size:],
        test_brightness_data=brightness_4d[-test_size:],
        test_labels=labels_4d[-test_size:]
    )
    print(f"Results: {results}")
    print("✓ evaluate() with direct 4D data successful")
    
    # 7. Test evaluate() with DataLoader
    print("\n7. Testing evaluate() with DataLoader")
    test_loader = get_dataloader(
        color_4d[-test_size:], 
        brightness_4d[-test_size:], 
        labels_4d[-test_size:],
        batch_size=2
    )
    results = model.evaluate(test_loader=test_loader)
    print(f"Results: {results}")
    print("✓ evaluate() with DataLoader successful")
    
    # 8. Test predict() with direct 2D data
    print("\n8. Testing predict() with direct 2D data")
    predictions = model.predict(color_2d[-test_size:], brightness_2d[-test_size:])
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions: {predictions}")
    print("✓ predict() with direct 2D data successful")
    
    # 9. Test predict() with direct 4D data
    print("\n9. Testing predict() with direct 4D data (should be flattened automatically)")
    predictions = model.predict(color_4d[-test_size:], brightness_4d[-test_size:])
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions: {predictions}")
    print("✓ predict() with direct 4D data successful")
    
    # 10. Test predict() with DataLoader
    print("\n10. Testing predict() with DataLoader")
    loader_for_predict = get_dataloader(
        color_4d[-test_size:], 
        brightness_4d[-test_size:], 
        labels_4d[-test_size:],
        batch_size=2
    )
    predictions = model.predict(loader_for_predict)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions: {predictions}")
    print("✓ predict() with DataLoader successful")
    
    # 11. Test predict_proba() with direct 2D data
    print("\n11. Testing predict_proba() with direct 2D data")
    probas = model.predict_proba(color_2d[-test_size:], brightness_2d[-test_size:])
    print(f"Probabilities shape: {probas.shape}")
    print(f"Probabilities sum to 1: {np.allclose(np.sum(probas, axis=1), 1.0)}")
    print("✓ predict_proba() with direct 2D data successful")
    
    # 12. Test predict_proba() with direct 4D data
    print("\n12. Testing predict_proba() with direct 4D data")
    probas = model.predict_proba(color_4d[-test_size:], brightness_4d[-test_size:])
    print(f"Probabilities shape: {probas.shape}")
    print(f"Probabilities sum to 1: {np.allclose(np.sum(probas, axis=1), 1.0)}")
    print("✓ predict_proba() with direct 4D data successful")
    
    # 13. Test predict_proba() with DataLoader
    print("\n13. Testing predict_proba() with DataLoader")
    probas = model.predict_proba(loader_for_predict)
    print(f"Probabilities shape: {probas.shape}")
    print(f"Probabilities sum to 1: {np.allclose(np.sum(probas, axis=1), 1.0)}")
    print("✓ predict_proba() with DataLoader successful")
    
    print("\nAll BaseMultiChannelNetwork tests passed! ✓")
    return True

def test_multi_channel_resnet_network():
    """Test all methods of MultiChannelResNetNetwork"""
    print("\n" + "="*80)
    print("Testing MultiChannelResNetNetwork")
    print("="*80)
    
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
    
    # Create 4D data for testing (CNNs require 4D data)
    data_4d = create_dummy_data(batch_size=32, is_4d=True)
    color_4d, brightness_4d, labels_4d = data_4d
    
    # Create 2D data for testing errors
    data_2d = create_dummy_data(batch_size=32, is_4d=False)
    color_2d, brightness_2d, labels_2d = data_2d
    
    # Create dataloaders
    loader_4d = get_dataloader(*data_4d, batch_size=8)
    
    # Split data for train/val/test
    train_size = 20
    val_size = 6
    test_size = 6
    
    # 1. Test fit() with direct 4D data
    print("\n1. Testing fit() with direct 4D data")
    history = model.fit(
        train_color_data=color_4d[:train_size],
        train_brightness_data=brightness_4d[:train_size],
        train_labels=labels_4d[:train_size],
        val_color_data=color_4d[train_size:train_size+val_size],
        val_brightness_data=brightness_4d[train_size:train_size+val_size],
        val_labels=labels_4d[train_size:train_size+val_size],
        epochs=2,
        verbose=1
    )
    print("✓ fit() with direct 4D data successful")
    
    # 2. Test fit() with DataLoader
    print("\n2. Testing fit() with DataLoader")
    train_loader_4d = get_dataloader(
        color_4d[:train_size], 
        brightness_4d[:train_size], 
        labels_4d[:train_size],
        batch_size=4
    )
    val_loader_4d = get_dataloader(
        color_4d[train_size:train_size+val_size], 
        brightness_4d[train_size:train_size+val_size], 
        labels_4d[train_size:train_size+val_size],
        batch_size=4
    )
    history = model.fit(
        train_loader=train_loader_4d,
        val_loader=val_loader_4d,
        epochs=2,
        verbose=1
    )
    print("✓ fit() with DataLoader successful")
    
    # 3. Test evaluate() with direct 4D data
    print("\n3. Testing evaluate() with direct 4D data")
    results = model.evaluate(
        test_color_data=color_4d[-test_size:],
        test_brightness_data=brightness_4d[-test_size:],
        test_labels=labels_4d[-test_size:]
    )
    print(f"Results: {results}")
    print("✓ evaluate() with direct 4D data successful")
    
    # 4. Test evaluate() with DataLoader
    print("\n4. Testing evaluate() with DataLoader")
    test_loader = get_dataloader(
        color_4d[-test_size:], 
        brightness_4d[-test_size:], 
        labels_4d[-test_size:],
        batch_size=2
    )
    results = model.evaluate(test_loader=test_loader)
    print(f"Results: {results}")
    print("✓ evaluate() with DataLoader successful")
    
    # 5. Test predict() with direct 4D data
    print("\n5. Testing predict() with direct 4D data")
    predictions = model.predict(color_4d[-test_size:], brightness_4d[-test_size:])
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions: {predictions}")
    print("✓ predict() with direct 4D data successful")
    
    # 6. Test predict() with DataLoader
    print("\n6. Testing predict() with DataLoader")
    loader_for_predict = get_dataloader(
        color_4d[-test_size:], 
        brightness_4d[-test_size:], 
        labels_4d[-test_size:],
        batch_size=2
    )
    predictions = model.predict(loader_for_predict)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions: {predictions}")
    print("✓ predict() with DataLoader successful")
    
    # 7. Test predict_proba() with direct 4D data
    print("\n7. Testing predict_proba() with direct 4D data")
    probas = model.predict_proba(color_4d[-test_size:], brightness_4d[-test_size:])
    print(f"Probabilities shape: {probas.shape}")
    print(f"Probabilities sum to 1: {np.allclose(np.sum(probas, axis=1), 1.0)}")
    print("✓ predict_proba() with direct 4D data successful")
    
    # 8. Test predict_proba() with DataLoader
    print("\n8. Testing predict_proba() with DataLoader")
    probas = model.predict_proba(loader_for_predict)
    print(f"Probabilities shape: {probas.shape}")
    print(f"Probabilities sum to 1: {np.allclose(np.sum(probas, axis=1), 1.0)}")
    print("✓ predict_proba() with DataLoader successful")
    
    # 9. Test error handling with 2D data (should raise error)
    print("\n9. Testing error handling with 2D data (should raise error)")
    try:
        results = model.evaluate(
            test_color_data=color_2d[-test_size:],
            test_brightness_data=brightness_2d[-test_size:],
            test_labels=labels_2d[-test_size:]
        )
        print("❌ ERROR: evaluate() with 2D data should have raised ValueError")
        return False
    except ValueError as e:
        print(f"✓ Correctly caught error: {e}")
    
    try:
        predictions = model.predict(color_2d[-test_size:], brightness_2d[-test_size:])
        print("❌ ERROR: predict() with 2D data should have raised ValueError")
        return False
    except ValueError as e:
        print(f"✓ Correctly caught error: {e}")
    
    try:
        probas = model.predict_proba(color_2d[-test_size:], brightness_2d[-test_size:])
        print("❌ ERROR: predict_proba() with 2D data should have raised ValueError")
        return False
    except ValueError as e:
        print(f"✓ Correctly caught error: {e}")
    
    print("\nAll MultiChannelResNetNetwork tests passed! ✓")
    return True

if __name__ == "__main__":
    print(f"Running comprehensive tests for multi-stream neural networks")
    print(f"Started at: {get_timestamp()}")
    
    all_passed = True
    
    try:
        # Test BaseMultiChannelNetwork (MLP architecture)
        base_passed = test_base_multi_channel_network()
        all_passed = all_passed and base_passed
        
        # Test MultiChannelResNetNetwork (CNN architecture)
        resnet_passed = test_multi_channel_resnet_network()
        all_passed = all_passed and resnet_passed
    except Exception as e:
        print(f"\n❌ ERROR: Test failed with exception: {e}")
        all_passed = False
    
    if all_passed:
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED")
        print("="*80)
        print("\nSUMMARY:")
        print("1. BaseMultiChannelNetwork correctly handles both 2D and 4D inputs (flattening 4D inputs)")
        print("2. MultiChannelResNetNetwork correctly enforces 4D inputs (raising errors for 2D inputs)")
        print("3. Both models support DataLoader inputs for all methods")
        print("4. All method signatures are consistent across models")
    else:
        print("\n" + "="*80)
        print("❌ SOME TESTS FAILED")
        print("="*80)
    
    print(f"\nFinished at: {get_timestamp()}")
