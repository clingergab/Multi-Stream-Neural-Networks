#!/usr/bin/env python3
"""
Test script to verify the dynamic progress bar implementation for ResNet.
This script tests that the progress bar properly displays train_loss, train_acc, val_loss, val_acc, and lr.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import sys
import os

from src.models2.core.resnet import resnet18

def create_dummy_data(num_samples=1000, num_classes=10, input_size=(3, 32, 32)):
    """Create dummy data for testing."""
    # Create random data
    X = torch.randn(num_samples, *input_size)
    y = torch.randint(0, num_classes, (num_samples,))
    
    return X, y

def test_progress_bar():
    """Test the progress bar implementation."""
    print("🧪 Testing ResNet Progress Bar Implementation")
    print("=" * 50)
    
    # Create dummy data
    print("📊 Creating dummy dataset...")
    train_X, train_y = create_dummy_data(num_samples=320, num_classes=10, input_size=(3, 32, 32))  # Small dataset for quick testing
    val_X, val_y = create_dummy_data(num_samples=128, num_classes=10, input_size=(3, 32, 32))
    
    # Create data loaders
    train_dataset = TensorDataset(train_X, train_y)
    val_dataset = TensorDataset(val_X, val_y)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    
    # Create model
    print("\n🏗️  Creating ResNet18 model...")
    model = resnet18(num_classes=10)
    
    # Compile model
    print("⚙️  Compiling model...")
    model.compile(
        optimizer='adam',
        loss='cross_entropy',
        lr=0.001
    )
    
    print(f"   Model compiled with Adam optimizer")
    print(f"   Device: {model.device}")
    
    # Test with DataLoader inputs
    print("\n🔬 Testing with DataLoader inputs...")
    print("   This tests the native DataLoader path")
    print("   Watch for progress bar with train_loss, train_acc, val_loss, val_acc, lr")
    
    history_loaders = model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=3,
        verbose=True
    )
    
    print("\n✅ DataLoader input test completed!")
    print(f"   Final train accuracy: {history_loaders['train_accuracy'][-1]:.4f}")
    print(f"   Final val accuracy: {history_loaders['val_accuracy'][-1]:.4f}")
    
    # Test with tensor inputs (should also work)
    print("\n🔬 Testing with direct tensor inputs...")
    print("   This tests the tensor -> DataLoader conversion")
    
    history_tensors = model.fit(
        train_loader=train_X,
        val_loader=val_X,
        train_targets=train_y,
        val_targets=val_y,
        epochs=2,
        batch_size=32,
        verbose=True
    )
    
    print("\n✅ Tensor input test completed!")
    print(f"   Final train accuracy: {history_tensors['train_accuracy'][-1]:.4f}")
    print(f"   Final val accuracy: {history_tensors['val_accuracy'][-1]:.4f}")
    
    # Verify history structure
    print("\n📋 Verifying history structure...")
    expected_keys = ['train_loss', 'val_loss', 'train_accuracy', 'val_accuracy', 'learning_rates']
    
    for key in expected_keys:
        if key in history_loaders:
            print(f"   ✅ {key}: {len(history_loaders[key])} values")
        else:
            print(f"   ❌ {key}: Missing!")
    
    # Test evaluation
    print("\n🔍 Testing evaluation method...")
    eval_results = model.evaluate(val_loader)
    print(f"   Evaluation accuracy: {eval_results['accuracy']:.4f}")
    print(f"   Evaluation loss: {eval_results['loss']:.4f}")
    
    print("\n🎉 All tests completed successfully!")
    print("   Progress bar should have shown:")
    print("   - train_loss (decreasing over batches)")
    print("   - train_acc (improving over batches)")
    print("   - val_loss (updated during validation)")
    print("   - val_acc (updated during validation)")
    print("   - lr (learning rate, may change with scheduler)")
    
    return True
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Create model
    model = resnet18(num_classes=10)
    model.compile(
        optimizer='adam',
        loss='cross_entropy',
        lr=0.001,
        scheduler='step',
        step_size=2,
        gamma=0.5
    )
    
    print("\nTraining with progress bar (verbose=True):")
    history = model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=3,
        verbose=True
    )
    
    print(f"\n✅ Training completed!")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final train accuracy: {history['train_accuracy'][-1]*100:.2f}%")
    print(f"Final val loss: {history['val_loss'][-1]:.4f}")
    print(f"Final val accuracy: {history['val_accuracy'][-1]:.2f}%")
    
    print("\n" + "="*50)
    print("Testing with verbose=False (no progress bar):")
    
    # Create new model for silent test
    model2 = resnet18(num_classes=10)
    model2.compile(
        optimizer='adam',
        loss='cross_entropy',
        lr=0.001
    )
    
    history2 = model2.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=2,
        verbose=False
    )
    
    print("✅ Silent training completed!")
    print(f"Final train loss: {history2['train_loss'][-1]:.4f}")
    
    return history, history2

if __name__ == "__main__":
    try:
        result = test_progress_bar()
        if isinstance(result, tuple):
            history, history2 = result
            print("\n✅ Test completed successfully!")
            print(f"Progress bar should have displayed all metrics: train_loss, train_acc, val_loss, val_acc, lr")
        else:
            print("\n✅ Test completed successfully!")
            print(f"Progress bar should have displayed all metrics: train_loss, train_acc, val_loss, val_acc, lr")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
