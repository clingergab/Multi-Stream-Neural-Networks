#!/usr/bin/env python3
"""
Test multi_channel_resnet50 model on brightness stream ONLY to see if it can learn.
This will help determine if the brightness channel contains useful information.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.basic_multi_channel.multi_channel_resnet_network import multi_channel_resnet50
from src.utils.cifar100_loader import get_cifar100_datasets

def test_brightness_only_training():
    print("🌟 Testing Multi-Channel ResNet50 on Brightness Stream ONLY")
    print("=" * 60)
    
    # Load CIFAR-100 data
    print("📊 Loading CIFAR-100 data...")
    train_dataset, test_dataset, class_names = get_cifar100_datasets()
    
    # Convert datasets to numpy arrays and extract brightness
    train_data = []
    train_labels = []
    for i in range(len(train_dataset)):
        data, label = train_dataset[i]
        # Convert RGB to brightness using standard luminance formula
        rgb_data = data.numpy()  # [3, 32, 32]
        # Brightness = 0.299*R + 0.587*G + 0.114*B
        brightness = 0.299 * rgb_data[0] + 0.587 * rgb_data[1] + 0.114 * rgb_data[2]
        brightness = brightness[np.newaxis, :, :]  # Add channel dimension [1, 32, 32]
        train_data.append(brightness)
        train_labels.append(label)
    
    test_data = []
    test_labels = []
    for i in range(len(test_dataset)):
        data, label = test_dataset[i]
        rgb_data = data.numpy()
        brightness = 0.299 * rgb_data[0] + 0.587 * rgb_data[1] + 0.114 * rgb_data[2]
        brightness = brightness[np.newaxis, :, :]
        test_data.append(brightness)
        test_labels.append(label)
    
    # Convert to tensors
    train_brightness = torch.stack([torch.from_numpy(x) for x in train_data])  # [N, 1, 32, 32]
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_brightness = torch.stack([torch.from_numpy(x) for x in test_data])
    test_labels = torch.tensor(test_labels, dtype=torch.long)
    
    # Create validation split (10%)
    val_size = int(0.1 * len(train_brightness))
    val_brightness = train_brightness[:val_size]
    val_labels = train_labels[:val_size]
    train_brightness = train_brightness[val_size:]
    train_labels = train_labels[val_size:]
    
    print(f"✅ Brightness data loaded:")
    print(f"   Training: {train_brightness.shape}, labels: {len(train_labels)}")
    print(f"   Validation: {val_brightness.shape}, labels: {len(val_labels)}")
    print(f"   Test: {test_brightness.shape}, labels: {len(test_labels)}")
    print(f"   Brightness range: [{train_brightness.min():.3f}, {train_brightness.max():.3f}]")
    
    # Create the EXACT same model as used in diagnostics
    print("\n🏗️  Creating multi_channel_resnet50 model (FULL architecture)...")
    model = multi_channel_resnet50(
        num_classes=100,
        device='auto'  # Note: reduce_architecture=False by default
    )
    
    print(f"✅ Model created:")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Architecture: {'Reduced' if model.reduce_architecture else 'Full ImageNet-style'}")
    print(f"   Device: {model.device}")
    print(f"   Block type: {model.block_type}")
    print(f"   Num blocks: {model.num_blocks}")
    
    # Compile model with EXACT same settings as diagnostics
    print("\n⚙️  Compiling model with diagnostic settings...")
    model.compile(
        optimizer='adamw',
        learning_rate=0.0003,  # Same as diagnostics
        weight_decay=1e-4,
        early_stopping_patience=5,
        loss='cross_entropy',
        metrics=['accuracy']
    )
    
    # Create dummy color streams (all zeros)
    print("\n🔧 Creating dummy color streams (zeros)...")
    train_color = torch.zeros(len(train_brightness), 3, 32, 32)
    val_color = torch.zeros(len(val_brightness), 3, 32, 32)
    
    print("✅ Brightness-only setup complete:")
    print("   Color stream: All zeros (no contribution)")
    print("   Brightness stream: Real brightness data from RGB conversion")
    print("   This effectively makes it a single-channel brightness model")
    
    # Debug forward pass before training
    print(f"\n🔍 Debugging forward pass...")
    try:
        # Test with a small batch
        test_color = train_color[:2]  # 2 samples
        test_brightness = train_brightness[:2]
        
        print(f"Input shapes:")
        print(f"   Color: {test_color.shape} (all zeros)")
        print(f"   Brightness: {test_brightness.shape} (real data)")
        print(f"   Brightness sample stats: min={test_brightness.min():.3f}, max={test_brightness.max():.3f}")
        
        # IMPORTANT: Move test data to model's device
        test_color = test_color.to(model.device)
        test_brightness = test_brightness.to(model.device)
        
        print(f"Input devices after moving:")
        print(f"   Color: {test_color.device}")
        print(f"   Brightness: {test_brightness.device}")
        print(f"   Model: {model.device}")
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            output = model(test_color, test_brightness)
            print(f"✅ Forward pass successful!")
            print(f"   Output shape: {output.shape}")
            print(f"   Output sample: {output[0, :5]}")  # First 5 logits of first sample
            
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        print("This confirms there's an issue with brightness-only processing")
        import traceback
        traceback.print_exc()
        return None

    # Train the model
    print(f"\n🚀 Starting training (brightness stream only)...")
    print("=" * 40)
    
    try:
        history = model.fit(
            train_color_data=train_color,
            train_brightness_data=train_brightness,
            train_labels=train_labels,
            val_color_data=val_color,
            val_brightness_data=val_brightness,
            val_labels=val_labels,
            batch_size=32,  # Same as best-performing batch size
            epochs=15,
            verbose=1,
            enable_diagnostics=False  # Keep it simple for this test
        )
        
        print(f"\n📊 Training Results:")
        print(f"   Final training accuracy: {history['train_accuracy'][-1]:.4f}")
        print(f"   Final validation accuracy: {history['val_accuracy'][-1]:.4f}")
        print(f"   Training epochs completed: {len(history['train_accuracy'])}")
        
        # Analyze brightness-only performance
        val_acc = history['val_accuracy'][-1]
        random_acc = 0.01  # 1% for 100 classes
        
        print(f"\n🔍 Brightness Stream Analysis:")
        print(f"   Random accuracy (baseline): {random_acc:.1%}")
        print(f"   Brightness-only accuracy: {val_acc:.1%}")
        print(f"   Improvement over random: {val_acc/random_acc:.1f}x")
        
        if val_acc > 0.15:  # 15% accuracy
            print("✅ GOOD: Brightness stream contains useful information!")
            print("   → Problem might be in how RGB and brightness streams interact")
        elif val_acc > 0.05:  # 5% accuracy (5x better than random)
            print("⚠️  MODERATE: Brightness stream has some useful information")
            print("   → But not as much as RGB stream")
        else:
            print("❌ POOR: Brightness stream contains little useful information")
            print("   → Confirms that brightness extraction might be problematic")
            
        return history
        
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_results():
    """
    Compare all three configurations we've tested
    """
    print("\n📊 COMPARISON SUMMARY")
    print("=" * 50)
    print("Configuration                    | Val Accuracy | Status")
    print("-" * 50)
    print("RGB + Brightness (multi-stream) | ~12.78%      | ❌ Broken")
    print("RGB only (zeros brightness)     | ~46.00%      | ✅ Works!")
    print("Brightness only (zeros RGB)     | [Testing...]  | 🧪 Current")
    print()
    print("This will help us understand:")
    print("1. Is brightness information useful on its own?")
    print("2. Why does multi-stream fusion fail?")
    print("3. Should we use different streams or stick to RGB?")

if __name__ == "__main__":
    # Run the brightness-only test
    history = test_brightness_only_training()
    
    # Show comparison
    compare_results()
    
    if history:
        print(f"\n🎯 BRIGHTNESS-ONLY CONCLUSION:")
        print(f"   This test reveals whether brightness contains useful info")
        print(f"   and helps us understand the multi-stream fusion problem.")
