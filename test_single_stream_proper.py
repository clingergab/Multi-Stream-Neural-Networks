#!/usr/bin/env python3
"""
Test multi_channel_resnet50 model on brightness stream only to evaluate if brightness
contains useful information for classification. This zeros out RGB and uses only brightness.
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

def test_single_stream_training():
    print("🧪 Testing Multi-Channel ResNet50 on Single Stream (RGB Only)")
    print("=" * 60)
    
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
    
    # Create brightness streams from RGB data and zero out RGB
    print("\n🔧 Creating brightness streams and zeroing RGB...")
    
    # Convert RGB to brightness (luminance): 0.299*R + 0.587*G + 0.114*B
    train_brightness = (0.299 * train_data[:, 0:1] + 
                       0.587 * train_data[:, 1:2] + 
                       0.114 * train_data[:, 2:3])  # Shape: [N, 1, 32, 32]
    
    val_brightness = (0.299 * val_data[:, 0:1] + 
                     0.587 * val_data[:, 1:2] + 
                     0.114 * val_data[:, 2:3])  # Shape: [N, 1, 32, 32]
    
    # Zero out RGB channels (keep only brightness)
    train_data = torch.zeros_like(train_data)  # All zeros for RGB
    val_data = torch.zeros_like(val_data)      # All zeros for RGB
    
    print("✅ Brightness-only setup complete:")
    print("   Color stream: All zeros (no contribution)")
    print("   Brightness stream: Real brightness data from RGB")
    print("   This effectively makes it a brightness-only model")
    
    # Debug forward pass before training
    print(f"\n🔍 Debugging forward pass...")
    try:
        # Test with a small batch
        test_color = train_data[:2]  # 2 samples
        test_brightness = train_brightness[:2]
        
        print(f"Input shapes:")
        print(f"   Color: {test_color.shape}")
        print(f"   Brightness: {test_brightness.shape}")
        
        # IMPORTANT: Move test data to model's device
        test_color = test_color.to(model.device)
        test_brightness = test_brightness.to(model.device)
        
        print(f"Input devices after moving:")
        print(f"   Color: {test_color.device}")
        print(f"   Brightness: {test_brightness.device}")
        print(f"   Model: {model.device}")
        
        # Test forward pass with detailed debugging
        model.eval()
        with torch.no_grad():
            print(f"🔍 Starting forward pass debugging...")
            
            # Add debugging to trace tensor shapes
            # Temporarily monkey-patch the forward method to add shape tracking
            original_forward = model.forward
            
            def debug_forward(color_input, brightness_input):
                print(f"  Input shapes: color={color_input.shape}, brightness={brightness_input.shape}")
                
                # Basic input normalization
                if color_input.max() > 1.0:
                    color_input = color_input / 255.0
                if brightness_input.max() > 1.0:
                    brightness_input = brightness_input / 255.0
                    
                # Initial layers
                color_x, brightness_x = model.conv1(color_input, brightness_input)
                print(f"  After conv1: color={color_x.shape}, brightness={brightness_x.shape}")
                
                color_x, brightness_x = model.bn1(color_x, brightness_x)
                color_x, brightness_x = model.activation_initial(color_x, brightness_x)
                
                # Apply maxpool to both streams
                color_x = model.maxpool(color_x)
                brightness_x = model.maxpool(brightness_x)
                print(f"  After maxpool: color={color_x.shape}, brightness={brightness_x.shape}")
                
                # ResNet layers
                color_x, brightness_x = model.layer1(color_x, brightness_x)
                print(f"  After layer1: color={color_x.shape}, brightness={brightness_x.shape}")
                
                color_x, brightness_x = model.layer2(color_x, brightness_x)
                print(f"  After layer2: color={color_x.shape}, brightness={brightness_x.shape}")
                
                color_x, brightness_x = model.layer3(color_x, brightness_x)
                print(f"  After layer3: color={color_x.shape}, brightness={brightness_x.shape}")
                
                color_x, brightness_x = model.layer4(color_x, brightness_x)
                print(f"  After layer4: color={color_x.shape}, brightness={brightness_x.shape}")
                
                # Global average pooling
                color_x, brightness_x = model.avgpool(color_x, brightness_x)
                print(f"  After avgpool: color={color_x.shape}, brightness={brightness_x.shape}")
                
                # Flatten features
                color_x = torch.flatten(color_x, 1)
                brightness_x = torch.flatten(brightness_x, 1)
                print(f"  After flatten: color={color_x.shape}, brightness={brightness_x.shape}")
                
                # Apply dropout if enabled
                if model.dropout_layer is not None:
                    color_x = model.dropout_layer(color_x)
                    brightness_x = model.dropout_layer(brightness_x)
                
                # Use shared classifier for optimal fusion
                fused_features = torch.cat([color_x, brightness_x], dim=1)
                print(f"  Fused features shape: {fused_features.shape}")
                print(f"  Expected classifier input: {model.shared_classifier.in_features}")
                print(f"  Actual classifier input: {fused_features.shape[1]}")
                
                logits = model.shared_classifier(fused_features)
                print(f"  Final output shape: {logits.shape}")
                return logits
            
            # Use debug forward
            output = debug_forward(test_color, test_brightness)
            print(f"✅ Forward pass successful!")
            print(f"   Output shape: {output.shape}")
            
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        print("This confirms there's a dimension mismatch in the model architecture")
        import traceback
        traceback.print_exc()
        return None

    # Train the model
    print(f"\n🚀 Starting training (single stream RGB)...")
    print("=" * 40)
    
    try:
        history = model.fit(
            train_color_data=train_data,
            train_brightness_data=train_brightness,
            train_labels=train_labels,
            val_color_data=val_data,
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
        
        # Test if it can achieve reasonable performance
        if history['val_accuracy'][-1] > 0.3:  # 30% accuracy (3x better than our current 12%)
            print("✅ SUCCESS: Model achieved good performance on single stream!")
            print("   → Problem is likely in multi-stream fusion/data processing")
        elif history['val_accuracy'][-1] > 0.15:  # 15% accuracy (better than current)
            print("⚠️  PARTIAL: Model performed better on single stream")
            print("   → Multi-stream approach is hurting performance")
        else:
            print("❌ FAILURE: Model still performs poorly on single stream")
            print("   → Fundamental issue with model architecture or training")
            
        return history
        
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_with_standard_resnet():
    """
    Optional: Create a standard single-stream ResNet for comparison
    """
    print("\n🔬 Comparison Test: Standard ResNet50")
    print("=" * 40)
    
    # Create a standard ResNet50 using torchvision for comparison
    try:
        import torchvision.models as models
        
        # Create standard ResNet50
        standard_resnet = models.resnet50(pretrained=False, num_classes=100)
        
        print(f"Standard ResNet50 parameters: {sum(p.numel() for p in standard_resnet.parameters()):,}")
        print("This gives us a baseline for comparison")
        
    except ImportError:
        print("torchvision not available, skipping standard ResNet comparison")

if __name__ == "__main__":
    # Run the single-stream test
    history = test_single_stream_training()
    
    # Optional comparison
    compare_with_standard_resnet()
    
    if history:
        print(f"\n🎯 CONCLUSION:")
        print(f"   This test isolates whether the problem is:")
        print(f"   1. Multi-stream fusion issues (if single-stream works well)")
        print(f"   2. Fundamental model/training issues (if single-stream also fails)")
        print(f"   3. Data preprocessing problems (if single-stream also fails)")
