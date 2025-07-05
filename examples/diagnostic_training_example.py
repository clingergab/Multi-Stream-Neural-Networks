#!/usr/bin/env python3
"""
Example script demonstrating comprehensive diagnostic training with Multi-Stream Neural Networks.

This script shows how to use the enhanced fit() methods with integrated diagnostics
to get comprehensive insights into model training behavior.

Usage:
    python examples/diagnostic_training_example.py

Features demonstrated:
- Comprehensive diagnostic tracking during training
- Automatic generation of diagnostic plots and reports
- Analysis of gradient flow, weight norms, and pathway balance
- Dead neuron detection and training stability monitoring
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.models.basic_multi_channel.base_multi_channel_network import base_multi_channel_large
from src.models.basic_multi_channel.multi_channel_resnet_network import multi_channel_resnet50
from src.utils.cifar100_loader import get_cifar100_datasets
from src.transforms.rgb_to_rgbl import RGBtoRGBL


def generate_sample_data(n_samples: int = 1000, input_size: int = 512):
    """Generate sample tabular data for BaseMultiChannelNetwork demonstration."""
    np.random.seed(42)
    
    # Generate color features (RGB-like features)
    color_data = np.random.randn(n_samples, input_size).astype(np.float32)
    
    # Generate brightness features (simpler, derived from color)
    brightness_data = np.random.randn(n_samples, input_size // 4).astype(np.float32)
    
    # Generate labels (10 classes)
    labels = np.random.randint(0, 10, n_samples)
    
    # Split into train/val
    split_idx = int(0.8 * n_samples)
    
    return {
        'train_color': color_data[:split_idx],
        'train_brightness': brightness_data[:split_idx],
        'train_labels': labels[:split_idx],
        'val_color': color_data[split_idx:],
        'val_brightness': brightness_data[split_idx:],
        'val_labels': labels[split_idx:]
    }


def demo_base_multi_channel_diagnostics():
    """Demonstrate diagnostic training with BaseMultiChannelNetwork."""
    print("=" * 60)
    print("🔬 BaseMultiChannelNetwork Diagnostic Training Demo")
    print("=" * 60)
    
    # Generate sample data
    data = generate_sample_data(n_samples=2000, input_size=512)
    
    # Create model
    model = base_multi_channel_large(
        color_input_size=512,
        brightness_input_size=128,
        num_classes=10,
        dropout=0.2
    )
    
    # Compile model
    model.compile(
        optimizer='adamw',
        learning_rate=0.001,
        weight_decay=1e-4,
        scheduler='cosine',
        early_stopping_patience=3
    )
    
    print(f"📊 Model created: {model.__class__.__name__}")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Device: {model.device}")
    
    # Train with comprehensive diagnostics
    print("\n🚀 Starting diagnostic training...")
    
    history = model.fit(
        train_color_data=data['train_color'],
        train_brightness_data=data['train_brightness'],
        train_labels=data['train_labels'],
        val_color_data=data['val_color'],
        val_brightness_data=data['val_brightness'],
        val_labels=data['val_labels'],
        batch_size=64,
        epochs=5,
        verbose=1,
        enable_diagnostics=True,
        diagnostic_output_dir="diagnostics/base_multi_channel"
    )
    
    print("\n📈 Training Results:")
    print(f"   Final train accuracy: {history['train_accuracy'][-1]:.4f}")
    print(f"   Final val accuracy: {history['val_accuracy'][-1]:.4f}")
    
    if 'gradient_norms' in history:
        print(f"   Final gradient norm: {history['gradient_norms'][-1]:.6f}")
        print(f"   Final weight norm: {history['weight_norms'][-1]:.4f}")
        print(f"   Final pathway balance: {history['pathway_balance'][-1]:.4f}")
    
    # Analyze pathways
    print("\n🔍 Pathway Analysis:")
    pathway_weights = model.analyze_pathway_weights()
    for key, value in pathway_weights.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")
    
    return model, history


def demo_resnet_diagnostics():
    """Demonstrate diagnostic training with MultiChannelResNetNetwork."""
    print("\n" + "=" * 60)
    print("🔬 MultiChannelResNetNetwork Diagnostic Training Demo")
    print("=" * 60)
    
    try:
        # Try to load CIFAR-100 data
        print("📂 Loading CIFAR-100 data...")
        train_dataset, val_dataset = get_cifar100_datasets(
            data_dir="data/cifar-100",
            download=True,
            train_transform=RGBtoRGBL(),
            val_transform=RGBtoRGBL()
        )
        
        # Create data loaders with small batches for demo
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=0,  # Keep it simple for demo
            pin_memory=False
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
        
    except Exception as e:
        print(f"⚠️  Could not load CIFAR-100 data: {e}")
        print("   Skipping ResNet diagnostic demo")
        return None, None
    
    # Create ResNet model with reduced architecture for faster demo
    model = multi_channel_resnet50(
        num_classes=100,
        reduce_architecture=True,  # Use smaller architecture for demo
        dropout=0.2
    )
    
    # Compile model
    model.compile(
        optimizer='adamw',
        learning_rate=0.0003,  # Lower learning rate for CNN
        weight_decay=1e-4,
        scheduler='cosine',
        early_stopping_patience=3
    )
    
    print(f"📊 Model created: {model.__class__.__name__}")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Device: {model.device}")
    print(f"   Architecture: {'Reduced' if model.reduce_architecture else 'Full'}")
    
    # Train with comprehensive diagnostics (short demo)
    print("\n🚀 Starting diagnostic training...")
    
    history = model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=3,  # Short demo
        verbose=1,
        enable_diagnostics=True,
        diagnostic_output_dir="diagnostics/multi_channel_resnet"
    )
    
    print("\n📈 Training Results:")
    print(f"   Final train accuracy: {history['train_accuracy'][-1]:.4f}")
    print(f"   Final val accuracy: {history['val_accuracy'][-1]:.4f}")
    
    if 'gradient_norms' in history:
        print(f"   Final gradient norm: {history['gradient_norms'][-1]:.6f}")
        print(f"   Final weight norm: {history['weight_norms'][-1]:.4f}")
        print(f"   Final pathway balance: {history['pathway_balance'][-1]:.4f}")
    
    # Analyze pathways
    print("\n🔍 Pathway Analysis:")
    pathway_weights = model.analyze_pathway_weights()
    for key, value in pathway_weights.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")
    
    return model, history


def compare_diagnostic_results(base_model, base_history, resnet_model, resnet_history):
    """Compare diagnostic results between models."""
    print("\n" + "=" * 60)
    print("🔬 Diagnostic Comparison")
    print("=" * 60)
    
    models = [
        ("BaseMultiChannelNetwork", base_model, base_history),
        ("MultiChannelResNetNetwork", resnet_model, resnet_history)
    ]
    
    for name, model, history in models:
        if model is None or history is None:
            print(f"⚠️  {name}: No data available")
            continue
            
        print(f"\n📊 {name}:")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Final accuracy: {history['val_accuracy'][-1]:.4f}")
        
        if 'gradient_norms' in history:
            print(f"   Gradient stability: {np.std(history['gradient_norms']):.6f}")
            print(f"   Weight stability: {np.std(history['weight_norms']):.6f}")
            print(f"   Pathway balance: {np.mean(history['pathway_balance']):.4f}")


def main():
    """Main function to run diagnostic training demos."""
    print("🔬 Multi-Stream Neural Network Diagnostic Training Examples")
    print("This script demonstrates comprehensive diagnostic capabilities.")
    print()
    
    try:
        # Demo 1: BaseMultiChannelNetwork with tabular data
        base_model, base_history = demo_base_multi_channel_diagnostics()
        
        # Demo 2: MultiChannelResNetNetwork with image data
        resnet_model, resnet_history = demo_resnet_diagnostics()
        
        # Compare results
        compare_diagnostic_results(base_model, base_history, resnet_model, resnet_history)
        
        print("\n" + "=" * 60)
        print("✅ Diagnostic Training Examples Completed!")
        print("=" * 60)
        print("📁 Check the 'diagnostics/' directory for:")
        print("   • Diagnostic plots (gradient flow, weight norms, etc.)")
        print("   • JSON summary reports with detailed statistics")
        print("   • Training history with comprehensive metrics")
        print()
        print("💡 Usage Tips:")
        print("   • Use enable_diagnostics=True for comprehensive monitoring")
        print("   • Diagnostic plots help identify training issues early")
        print("   • Pathway balance metrics show multi-stream effectiveness")
        print("   • Dead neuron detection helps with architecture tuning")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
