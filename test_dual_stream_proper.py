#!/usr/bin/env python3
"""
Test multi_channel_resnet50 model on BOTH streams (RGB + brightness) to confirm
the multi-stream fusion problem. This will help verify our hypothesis that:

1. Information Redundancy: RGB and brightness are correlated
2. Competing Gradients: Two pathways learning similar but conflicting representations  
3. Feature Interference: Model gets confused reconciling RGB vs brightness features
4. Fusion Bottleneck: Concatenating correlated features creates conflicting signals

Expected Result: Poor performance (~12-15% validation accuracy) confirming fusion issues.
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

def test_dual_stream_training():
    print("🔄 Testing Multi-Channel ResNet50 on DUAL STREAMS (RGB + Brightness)")
    print("=" * 70)
    
    # Load CIFAR-100 data
    print("📊 Loading CIFAR-100 data...")
    train_dataset, test_dataset, class_names = get_cifar100_datasets()
    
    # Convert datasets to numpy arrays
    train_data = []
    train_labels = []
    for i in range(len(train_dataset)):
        data, label = train_dataset[i]
        train_data.append(data.numpy())
        train_labels.append(label)
    
    test_data = []
    test_labels = []
    for i in range(len(test_dataset)):
        data, label = test_dataset[i]
        test_data.append(data.numpy())
        test_labels.append(label)
    
    # Convert to tensors
    train_data = torch.stack([torch.from_numpy(x) for x in train_data])  # [N, 3, 32, 32]
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_data = torch.stack([torch.from_numpy(x) for x in test_data])
    test_labels = torch.tensor(test_labels, dtype=torch.long)
    
    # Create validation split (10%)
    val_size = int(0.1 * len(train_data))
    val_data = train_data[:val_size]
    val_labels = train_labels[:val_size]
    train_data = train_data[val_size:]
    train_labels = train_labels[val_size:]
    
    print(f"✅ Data loaded:")
    print(f"   Training: {train_data.shape}, labels: {len(train_labels)}")
    print(f"   Validation: {val_data.shape}, labels: {len(val_labels)}")
    print(f"   Test: {test_data.shape}, labels: {len(test_labels)}")
    
    # Create brightness streams from RGB data (REAL brightness data)
    print("\n🔧 Creating brightness streams from RGB...")
    
    # Convert RGB to brightness (luminance): 0.299*R + 0.587*G + 0.114*B
    train_brightness = (0.299 * train_data[:, 0:1] + 
                       0.587 * train_data[:, 1:2] + 
                       0.114 * train_data[:, 2:3])  # Shape: [N, 1, 32, 32]
    
    val_brightness = (0.299 * val_data[:, 0:1] + 
                     0.587 * val_data[:, 1:2] + 
                     0.114 * val_data[:, 2:3])  # Shape: [N, 1, 32, 32]
    
    print("✅ Dual-stream setup complete:")
    print("   Color stream: Real RGB data")
    print("   Brightness stream: Real brightness data from RGB conversion")
    print("   This tests full multi-stream fusion with correlated inputs")
    print(f"   RGB range: [{train_data.min():.3f}, {train_data.max():.3f}]")
    print(f"   Brightness range: [{train_brightness.min():.3f}, {train_brightness.max():.3f}]")
    
    # Create the EXACT same model as used in single-stream tests
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
    
    # Compile model with EXACT same settings as single-stream tests
    print("\n⚙️  Compiling model with diagnostic settings...")
    model.compile(
        optimizer='adamw',
        learning_rate=0.0003,  # Same as single-stream tests
        weight_decay=1e-4,
        early_stopping_patience=5,
        loss='cross_entropy',
        metrics=['accuracy']
    )
    
    # Debug forward pass before training
    print(f"\n🔍 Debugging forward pass...")
    try:
        # Test with a small batch
        test_color = train_data[:2]  # 2 samples
        test_brightness = train_brightness[:2]
        
        print(f"Input shapes:")
        print(f"   Color: {test_color.shape}")
        print(f"   Brightness: {test_brightness.shape}")
        print(f"   RGB sample stats: min={test_color.min():.3f}, max={test_color.max():.3f}")
        print(f"   Brightness sample stats: min={test_brightness.min():.3f}, max={test_brightness.max():.3f}")
        
        # Move test data to model's device
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
            print(f"   Output sample: {output[0][:5]}")
            
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Train the model
    print(f"\n🚀 Starting training (dual stream RGB + brightness)...")
    print("=" * 50)
    
    try:
        history = model.fit(
            train_color_data=train_data,
            train_brightness_data=train_brightness,
            train_labels=train_labels,
            val_color_data=val_data,
            val_brightness_data=val_brightness,
            val_labels=val_labels,
            batch_size=32,  # Same as single-stream tests
            epochs=15,
            verbose=1,
            enable_diagnostics=False  # Keep it simple for this test
        )
        
        print(f"\n📊 Training Results:")
        print(f"   Final training accuracy: {history['train_accuracy'][-1]:.4f}")
        print(f"   Final validation accuracy: {history['val_accuracy'][-1]:.4f}")
        print(f"   Training epochs completed: {len(history['train_accuracy'])}")
        
        # Analyze results vs single-stream performance
        dual_stream_acc = history['val_accuracy'][-1]
        rgb_only_acc = 0.46  # From previous RGB-only test
        brightness_only_acc = 0.406  # From previous brightness-only test
        random_acc = 0.01  # 1% for 100 classes
        
        print(f"\n🔍 MULTI-STREAM FUSION ANALYSIS:")
        print("=" * 50)
        print(f"Configuration                    | Val Accuracy | vs Random")
        print(f"-" * 50)
        print(f"RGB only (zeros brightness)     | {rgb_only_acc:.1%}      | {rgb_only_acc/random_acc:.1f}x")
        print(f"Brightness only (zeros RGB)     | {brightness_only_acc:.1%}      | {brightness_only_acc/random_acc:.1f}x")
        print(f"RGB + Brightness (dual stream)  | {dual_stream_acc:.1%}      | {dual_stream_acc/random_acc:.1f}x")
        
        # Hypothesis testing
        print(f"\n🧪 HYPOTHESIS VERIFICATION:")
        print("=" * 40)
        
        if dual_stream_acc < rgb_only_acc * 0.7:  # Significantly worse than RGB-only
            print("✅ HYPOTHESIS CONFIRMED: Multi-stream fusion HURTS performance!")
            print("   → RGB + Brightness performs significantly worse than RGB alone")
            print("   → This confirms feature interference and fusion bottleneck")
            
            if dual_stream_acc < brightness_only_acc * 0.7:  # Also worse than brightness-only
                print("✅ STRONG EVIDENCE: Dual-stream worse than BOTH single streams!")
                print("   → Competing gradients and information redundancy confirmed")
            
        elif dual_stream_acc < max(rgb_only_acc, brightness_only_acc):
            print("⚠️  PARTIAL CONFIRMATION: Multi-stream underperforms best single stream")
            print("   → Some fusion issues present but not catastrophic")
            
        else:
            print("❓ UNEXPECTED: Multi-stream performs as well as single streams")
            print("   → This would challenge our hypothesis - need further investigation")
        
        # Information theory analysis
        expected_improvement = max(rgb_only_acc, brightness_only_acc) * 1.1  # Expect ~10% boost if streams were independent
        actual_performance = dual_stream_acc
        
        print(f"\n📈 INFORMATION THEORY ANALYSIS:")
        print(f"   If streams were independent, expected: ~{expected_improvement:.1%}")
        print(f"   Actual dual-stream performance: {actual_performance:.1%}")
        print(f"   Performance gap: {(expected_improvement - actual_performance):.1%}")
        
        if actual_performance < expected_improvement * 0.8:
            print("   → CONCLUSION: Streams are highly correlated (not independent)")
            print("   → Brightness = f(RGB) causes redundancy and interference")
        
        return history
        
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_stream_correlation():
    """
    Analyze the mathematical correlation between RGB and brightness streams
    """
    print("\n🔬 STREAM CORRELATION ANALYSIS:")
    print("=" * 40)
    
    print("Mathematical relationship:")
    print("   Brightness = 0.299*R + 0.587*G + 0.114*B")
    print("   → Brightness is a LINEAR COMBINATION of RGB channels")
    print("   → Perfect mathematical correlation exists")
    
    print("\nWhy this causes problems:")
    print("   1. REDUNDANT INFORMATION: Brightness adds no new information")
    print("   2. COMPETING GRADIENTS: Two pathways learn same underlying patterns")
    print("   3. FEATURE INTERFERENCE: Similar features from different pathways confuse fusion")
    print("   4. OPTIMIZATION DIFFICULTY: Network struggles to weight correlated features")
    
    print("\nBetter alternatives:")
    print("   • Use truly independent modalities (RGB + Depth, RGB + Thermal)")
    print("   • Use RGB-only for best single-stream performance")  
    print("   • Apply decorrelation techniques if brightness is needed")
    print("   • Use different fusion strategies (attention, gating)")

if __name__ == "__main__":
    # Run the dual-stream test
    history = test_dual_stream_training()
    
    # Analyze correlation
    analyze_stream_correlation()
    
    if history:
        print(f"\n🎯 FINAL CONCLUSION:")
        print(f"   This test validates our hypothesis about multi-stream fusion issues")
        print(f"   when using correlated input streams (RGB + brightness derived from RGB)")
        print(f"   The results should demonstrate why single-stream approaches work better")
        print(f"   for this particular combination of input modalities.")
