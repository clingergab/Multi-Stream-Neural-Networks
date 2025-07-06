"""
Demonstration of early stopping functionality in ResNet.

This script shows how to use the early stopping feature with different configurations.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import sys
import os

# Add src to path for imports
src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src')
sys.path.insert(0, src_path)

from models2.core.resnet import resnet18


def create_dummy_data(num_samples=200, num_classes=10):
    """Create dummy CIFAR-like dataset for demonstration."""
    # Generate random data
    X = torch.randn(num_samples, 3, 32, 32)
    y = torch.randint(0, num_classes, (num_samples,))
    
    # Split into train/val
    split_idx = int(0.8 * num_samples)
    
    train_X, val_X = X[:split_idx], X[split_idx:]
    train_y, val_y = y[:split_idx], y[split_idx:]
    
    # Create DataLoaders
    train_dataset = TensorDataset(train_X, train_y)
    val_dataset = TensorDataset(val_X, val_y)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    return train_loader, val_loader


def demo_early_stopping_val_loss():
    """Demonstrate early stopping with validation loss monitoring."""
    print("🚀 Demo: Early Stopping with Validation Loss Monitoring")
    print("=" * 60)
    
    # Create model and data
    model = resnet18(num_classes=10)
    model.compile(optimizer='adam', loss='cross_entropy', lr=0.01, device='cpu')
    
    train_loader, val_loader = create_dummy_data()
    
    # Train with early stopping
    history = model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=20,
        early_stopping=True,
        patience=3,
        monitor='val_loss',
        min_delta=0.001,
        restore_best_weights=True,
        verbose=True
    )
    
    # Display results
    if 'early_stopping' in history:
        es_info = history['early_stopping']
        print(f"\n📊 Early Stopping Results:")
        print(f"   Stopped Early: {es_info['stopped_early']}")
        print(f"   Best Epoch: {es_info['best_epoch']}")
        print(f"   Best {es_info['monitor']}: {es_info['best_metric']:.4f}")
        print(f"   Patience Used: {es_info['patience']}")
        print(f"   Min Delta: {es_info['min_delta']}")
    
    print(f"\n📈 Training Completed:")
    print(f"   Total Epochs Run: {len(history['train_loss'])}")
    print(f"   Final Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"   Final Val Loss: {history['val_loss'][-1]:.4f}")
    
    return history


def demo_early_stopping_val_accuracy():
    """Demonstrate early stopping with validation accuracy monitoring."""
    print("\n\n🚀 Demo: Early Stopping with Validation Accuracy Monitoring")
    print("=" * 60)
    
    # Create model and data
    model = resnet18(num_classes=10)
    model.compile(optimizer='adam', loss='cross_entropy', lr=0.01, device='cpu')
    
    train_loader, val_loader = create_dummy_data()
    
    # Train with early stopping
    history = model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=15,
        early_stopping=True,
        patience=5,
        monitor='val_accuracy',
        min_delta=0.01,  # 1% improvement required
        restore_best_weights=True,
        verbose=True
    )
    
    # Display results
    if 'early_stopping' in history:
        es_info = history['early_stopping']
        print(f"\n📊 Early Stopping Results:")
        print(f"   Stopped Early: {es_info['stopped_early']}")
        print(f"   Best Epoch: {es_info['best_epoch']}")
        print(f"   Best {es_info['monitor']}: {es_info['best_metric']:.4f}")
        print(f"   Patience Used: {es_info['patience']}")
        print(f"   Min Delta: {es_info['min_delta']}")
    
    print(f"\n📈 Training Completed:")
    print(f"   Total Epochs Run: {len(history['train_loss'])}")
    print(f"   Final Train Accuracy: {history['train_accuracy'][-1]:.4f}")
    print(f"   Final Val Accuracy: {history['val_accuracy'][-1]:.4f}")
    
    return history


def demo_no_early_stopping():
    """Demonstrate normal training without early stopping."""
    print("\n\n🚀 Demo: Normal Training (No Early Stopping)")
    print("=" * 60)
    
    # Create model and data
    model = resnet18(num_classes=10)
    model.compile(optimizer='adam', loss='cross_entropy', lr=0.01, device='cpu')
    
    train_loader, val_loader = create_dummy_data()
    
    # Train without early stopping
    history = model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=8,
        early_stopping=False,  # Explicitly disabled
        verbose=True
    )
    
    # Display results
    print(f"\n📊 Training Results:")
    print(f"   Early Stopping: {'Enabled' if 'early_stopping' in history else 'Disabled'}")
    print(f"   Total Epochs Run: {len(history['train_loss'])}")
    print(f"   Final Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"   Final Val Loss: {history['val_loss'][-1]:.4f}")
    print(f"   Final Train Accuracy: {history['train_accuracy'][-1]:.4f}")
    print(f"   Final Val Accuracy: {history['val_accuracy'][-1]:.4f}")
    
    return history


def demo_tensor_inputs():
    """Demonstrate early stopping with tensor inputs instead of DataLoaders."""
    print("\n\n🚀 Demo: Early Stopping with Tensor Inputs")
    print("=" * 60)
    
    # Create model
    model = resnet18(num_classes=10)
    model.compile(optimizer='adam', loss='cross_entropy', lr=0.01, device='cpu')
    
    # Create tensor data directly
    num_samples = 128
    train_X = torch.randn(num_samples, 3, 32, 32)
    train_y = torch.randint(0, 10, (num_samples,))
    val_X = torch.randn(32, 3, 32, 32)
    val_y = torch.randint(0, 10, (32,))
    
    # Train with tensor inputs
    history = model.fit(
        train_loader=train_X,
        train_targets=train_y,
        val_loader=val_X,
        val_targets=val_y,
        epochs=10,
        batch_size=16,
        early_stopping=True,
        patience=3,
        monitor='val_loss',
        verbose=True
    )
    
    # Display results
    if 'early_stopping' in history:
        es_info = history['early_stopping']
        print(f"\n📊 Early Stopping Results:")
        print(f"   Stopped Early: {es_info['stopped_early']}")
        print(f"   Best Epoch: {es_info['best_epoch']}")
        print(f"   Best {es_info['monitor']}: {es_info['best_metric']:.4f}")
    
    print(f"\n📈 Training with Tensors Completed:")
    print(f"   Total Epochs Run: {len(history['train_loss'])}")
    print(f"   Data Type: Tensor inputs converted to DataLoaders")
    
    return history


if __name__ == "__main__":
    print("🎯 ResNet Early Stopping Demonstration")
    print("This demo shows different early stopping configurations.\n")
    
    # Run all demonstrations
    try:
        demo_early_stopping_val_loss()
        demo_early_stopping_val_accuracy()
        demo_no_early_stopping()
        demo_tensor_inputs()
        
        print("\n\n✅ All demonstrations completed successfully!")
        print("\n🔍 Key Features Demonstrated:")
        print("   • Early stopping with validation loss monitoring")
        print("   • Early stopping with validation accuracy monitoring")
        print("   • Normal training without early stopping")
        print("   • Early stopping with tensor inputs")
        print("   • Best weight restoration")
        print("   • Configurable patience and min_delta")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
