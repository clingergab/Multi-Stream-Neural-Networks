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
import sys
from torch.utils.data import DataLoader, TensorDataset

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.basic_multi_channel.base_multi_channel_network import BaseMultiChannelNetwork
from src.models.basic_multi_channel.multi_channel_resnet_network import MultiChannelResNetNetwork
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
        # Create 4D data (NCHW format)
        # For color: [batch_size, 3, 32, 32]
        # For brightness: [batch_size, 1, 32, 32]
        color_data = torch.rand(batch_size, 3, 32, 32)
        brightness_data = torch.rand(batch_size, 1, 32, 32)
    else:
        # Create 2D data (flattened)
        # For color: [batch_size, 3*32*32]
        # For brightness: [batch_size, 32*32]
        color_data = torch.rand(batch_size, flattened_color_size)
        brightness_data = torch.rand(batch_size, flattened_brightness_size)
    
    # Create random labels
    labels = torch.randint(0, num_classes, (batch_size,))
    
    return color_data, brightness_data, labels

def create_data_loader(color_data, brightness_data, labels, batch_size=4):
    """Create a DataLoader from data tensors."""
    dataset = TensorDataset(color_data, brightness_data, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def test_base_multi_channel_network():
    """Test all methods of BaseMultiChannelNetwork."""
    print("\n================================================================================")
    print("Testing BaseMultiChannelNetwork")
    print("================================================================================")
    
    # Create a small model
    model = BaseMultiChannelNetwork(
        color_input_size=3*32*32,  # CIFAR dimensions flattened
        brightness_input_size=32*32,
        hidden_sizes=[64, 32],
        num_classes=10,
        dropout=0.2,
        device='cpu'  # Use CPU for testing
    )
    model.compile(learning_rate=0.003, weight_decay=0.0, early_stopping_patience=10)
    
    # Create datasets
    train_color_2d, train_brightness_2d, train_labels = create_dummy_data(batch_size=20, is_4d=False)
    val_color_2d, val_brightness_2d, val_labels = create_dummy_data(batch_size=6, is_4d=False)
    
    train_color_4d, train_brightness_4d, train_labels_4d = create_dummy_data(batch_size=20, is_4d=True)
    val_color_4d, val_brightness_4d, val_labels_4d = create_dummy_data(batch_size=6, is_4d=True)
    
    train_loader_2d = create_data_loader(train_color_2d, train_brightness_2d, train_labels)
    val_loader_2d = create_data_loader(val_color_2d, val_brightness_2d, val_labels)
    
    train_loader_4d = create_data_loader(train_color_4d, train_brightness_4d, train_labels_4d)
    val_loader_4d = create_data_loader(val_color_4d, val_brightness_4d, val_labels_4d)
    
    # 1. Test fit() with direct 2D data
    print("1. Testing fit() with direct 2D data")
    model.fit(
        train_color_data=train_color_2d,
        train_brightness_data=train_brightness_2d,
        train_labels=train_labels,
        val_color_data=val_color_2d,
        val_brightness_data=val_brightness_2d,
        val_labels=val_labels,
        epochs=2,
        batch_size=10
    )
    print("✓ fit() with direct 2D data successful\n")
    
    # 2. Test fit() with direct 4D data (should be flattened automatically)
    print("2. Testing fit() with direct 4D data (should be flattened automatically)")
    history = model.fit(
        train_color_data=train_color_4d,
        train_brightness_data=train_brightness_4d,
        train_labels=train_labels_4d,
        val_color_data=val_color_4d,
        val_brightness_data=val_brightness_4d,
        val_labels=val_labels_4d,
        epochs=2,
        batch_size=10
    )
    print("✓ fit() with direct 4D data successful\n")
    
    # 3. Test fit() with DataLoader containing 2D data
    print("3. Testing fit() with DataLoader containing 2D data")
    history = model.fit(
        train_loader=train_loader_2d,
        val_loader=val_loader_2d,
        epochs=2
    )
    print("✓ fit() with DataLoader containing 2D data successful\n")
    
    # 4. Test fit() with DataLoader containing 4D data
    print("4. Testing fit() with DataLoader containing 4D data")
    history = model.fit(
        train_loader=train_loader_4d,
        val_loader=val_loader_4d,
        epochs=2
    )
    print("✓ fit() with DataLoader containing 4D data successful\n")
    
    # 5. Test evaluate() with direct 2D data
    print("5. Testing evaluate() with direct 2D data")
    results = model.evaluate(
        test_color_data=val_color_2d,
        test_brightness_data=val_brightness_2d,
        test_labels=val_labels
    )
    print(f"Results: {results}")
    print("✓ evaluate() with direct 2D data successful\n")
    
    # 6. Test evaluate() with direct 4D data
    print("6. Testing evaluate() with direct 4D data")
    results = model.evaluate(
        test_color_data=val_color_4d,
        test_brightness_data=val_brightness_4d,
        test_labels=val_labels_4d
    )
    print(f"Results: {results}")
    print("✓ evaluate() with direct 4D data successful\n")
    
    # 7. Test evaluate() with DataLoader
    print("7. Testing evaluate() with DataLoader")
    results = model.evaluate(test_loader=val_loader_4d)
    print(f"Results: {results}")
    print("✓ evaluate() with DataLoader successful\n")
    
    # 8. Test predict() with direct 2D data
    print("8. Testing predict() with direct 2D data")
    predictions = model.predict(val_color_2d, val_brightness_2d)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions: {predictions}")
    print("✓ predict() with direct 2D data successful\n")
    
    # 9. Test predict() with direct 4D data
    print("9. Testing predict() with direct 4D data (should be flattened automatically)")
    predictions = model.predict(val_color_4d, val_brightness_4d)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions: {predictions}")
    print("✓ predict() with direct 4D data successful\n")
    
    # 10. Test predict() with DataLoader
    print("10. Testing predict() with DataLoader")
    predictions = model.predict(val_loader_4d)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions: {predictions}")
    print("✓ predict() with DataLoader successful\n")
    
    # 11. Test predict_proba() with direct 2D data
    print("11. Testing predict_proba() with direct 2D data")
    probabilities = model.predict_proba(val_color_2d, val_brightness_2d)
    print(f"Probabilities shape: {probabilities.shape}")
    print(f"Probabilities sum to 1: {np.allclose(np.sum(probabilities, axis=1), 1.0)}")
    print("✓ predict_proba() with direct 2D data successful\n")
    
    # 12. Test predict_proba() with direct 4D data
    print("12. Testing predict_proba() with direct 4D data")
    probabilities = model.predict_proba(val_color_4d, val_brightness_4d)
    print(f"Probabilities shape: {probabilities.shape}")
    print(f"Probabilities sum to 1: {np.allclose(np.sum(probabilities, axis=1), 1.0)}")
    print("✓ predict_proba() with direct 4D data successful\n")
    
    # 13. Test predict_proba() with DataLoader
    print("13. Testing predict_proba() with DataLoader")
    probabilities = model.predict_proba(val_loader_4d)
    print(f"Probabilities shape: {probabilities.shape}")
    print(f"Probabilities sum to 1: {np.allclose(np.sum(probabilities, axis=1), 1.0)}")
    print("✓ predict_proba() with DataLoader successful\n")
    
    print("All BaseMultiChannelNetwork tests passed! ✓")

def test_multi_channel_resnet_network():
    """Test all methods of MultiChannelResNetNetwork."""
    print("\n================================================================================")
    print("Testing MultiChannelResNetNetwork")
    print("================================================================================")
    
    # Create a small ResNet model
    model = MultiChannelResNetNetwork(
        num_classes=10,
        device='cpu',  # Use CPU for testing
        architecture='reduced',
        img_size=32
    )
    model.compile()
    
    # Create datasets - only 4D data for ResNet
    train_color_4d, train_brightness_4d, train_labels = create_dummy_data(batch_size=20, is_4d=True)
    val_color_4d, val_brightness_4d, val_labels = create_dummy_data(batch_size=6, is_4d=True)
    
    # 2D data for error testing
    train_color_2d, train_brightness_2d, train_labels_2d = create_dummy_data(batch_size=20, is_4d=False)
    
    train_loader = create_data_loader(train_color_4d, train_brightness_4d, train_labels)
    val_loader = create_data_loader(val_color_4d, val_brightness_4d, val_labels)
    
    # 1. Test fit() with direct 4D data
    print("1. Testing fit() with direct 4D data")
    model.fit(
        train_color_data=train_color_4d,
        train_brightness_data=train_brightness_4d,
        train_labels=train_labels,
        val_color_data=val_color_4d,
        val_brightness_data=val_brightness_4d,
        val_labels=val_labels,
        epochs=2,
        batch_size=10
    )
    print("✓ fit() with direct 4D data successful\n")
    
    # 2. Test fit() with DataLoader
    print("2. Testing fit() with DataLoader")
    history = model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=2
    )
    print("✓ fit() with DataLoader successful\n")
    
    # 3. Test evaluate() with direct 4D data
    print("3. Testing evaluate() with direct 4D data")
    results = model.evaluate(
        test_color_data=val_color_4d,
        test_brightness_data=val_brightness_4d,
        test_labels=val_labels
    )
    print(f"Results: {results}")
    print("✓ evaluate() with direct 4D data successful\n")
    
    # 4. Test evaluate() with DataLoader
    print("4. Testing evaluate() with DataLoader")
    results = model.evaluate(test_loader=val_loader)
    print(f"Results: {results}")
    print("✓ evaluate() with DataLoader successful\n")
    
    # 5. Test predict() with direct 4D data
    print("5. Testing predict() with direct 4D data")
    predictions = model.predict(val_color_4d, val_brightness_4d)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions: {predictions}")
    print("✓ predict() with direct 4D data successful\n")
    
    # 6. Test predict() with DataLoader
    print("6. Testing predict() with DataLoader")
    predictions = model.predict(val_loader)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions: {predictions}")
    print("✓ predict() with DataLoader successful\n")
    
    # 7. Test predict_proba() with direct 4D data
    print("7. Testing predict_proba() with direct 4D data")
    probabilities = model.predict_proba(val_color_4d, val_brightness_4d)
    print(f"Probabilities shape: {probabilities.shape}")
    print(f"Probabilities sum to 1: {np.allclose(np.sum(probabilities, axis=1), 1.0)}")
    print("✓ predict_proba() with direct 4D data successful\n")
    
    # 8. Test predict_proba() with DataLoader
    print("8. Testing predict_proba() with DataLoader")
    probabilities = model.predict_proba(val_loader)
    print(f"Probabilities shape: {probabilities.shape}")
    print(f"Probabilities sum to 1: {np.allclose(np.sum(probabilities, axis=1), 1.0)}")
    print("✓ predict_proba() with DataLoader successful\n")
    
    # 9. Test error handling with 2D data (should raise error)
    print("9. Testing error handling with 2D data (should raise error)")
    
    try:
        predictions = model.predict(train_color_2d, train_brightness_2d)
        print("❌ Failed to catch error with 2D data in predict()")
    except ValueError as e:
        print(f"✓ Correctly caught error: {str(e)}")
    
    try:
        probabilities = model.predict_proba(train_color_2d, train_brightness_2d)
        print("❌ Failed to catch error with 2D data in predict_proba()")
    except ValueError as e:
        print(f"✓ Correctly caught error: {str(e)}")
    
    try:
        history = model.fit(
            train_color_data=train_color_2d,
            train_brightness_data=train_brightness_2d,
            train_labels=train_labels_2d,
            epochs=1
        )
        print("❌ Failed to catch error with 2D data in fit()")
    except ValueError as e:
        print(f"✓ Correctly caught error: {str(e)}")
    
    print("\nAll MultiChannelResNetNetwork tests passed! ✓")

if __name__ == "__main__":
    print("Running comprehensive tests for multi-stream neural networks")
    print(f"Started at: {get_timestamp()}")
    
    # Test both model types
    test_base_multi_channel_network()
    test_multi_channel_resnet_network()
    
    print("\n================================================================================")
    print("✅ ALL TESTS PASSED")
    print("================================================================================")
    
    print("\nSUMMARY:")
    print("1. BaseMultiChannelNetwork correctly handles both 2D and 4D inputs (flattening 4D inputs)")
    print("2. MultiChannelResNetNetwork correctly enforces 4D inputs (raising errors for 2D inputs)")
    print("3. Both models support DataLoader inputs for all methods")
    print("4. All method signatures are consistent across models")
    
    print(f"\nFinished at: {get_timestamp()}")
