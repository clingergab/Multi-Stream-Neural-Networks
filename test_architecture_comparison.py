#!/usr/bin/env python3
"""
Quick test to compare architectural choices for CIFAR-100.
This will demonstrate why the current approach fails.
"""

import torch
from src.models.basic_multi_channel.multi_channel_resnet_network import (
    MultiChannelResNetNetwork, 
    multi_channel_resnet50
)

def test_architectures():
    print("🔬 Architecture Comparison for CIFAR-100")
    print("=" * 50)
    
    # Test 1: Current approach (WRONG for CIFAR-100)
    print("\n1️⃣ Current Diagnostic Script Approach (multi_channel_resnet50):")
    model_wrong = multi_channel_resnet50(num_classes=100)
    params_wrong = sum(p.numel() for p in model_wrong.parameters())
    print(f"   Architecture: {'Reduced' if model_wrong.reduce_architecture else 'Full ImageNet-style'}")
    print(f"   Block type: {model_wrong.block_type}")
    print(f"   Num blocks: {model_wrong.num_blocks}")
    print(f"   Parameters: {params_wrong:,}")
    
    # Test 2: Correct approach for CIFAR-100
    print("\n2️⃣ Correct CIFAR-100 Approach (for_cifar100):")
    model_right = MultiChannelResNetNetwork.for_cifar100(dropout=0.3)
    params_right = sum(p.numel() for p in model_right.parameters())
    print(f"   Architecture: {'Reduced' if model_right.reduce_architecture else 'Full ImageNet-style'}")
    print(f"   Block type: {model_right.block_type}")
    print(f"   Num blocks: {model_right.num_blocks}")
    print(f"   Parameters: {params_right:,}")
    
    # Analysis
    print(f"\n📊 Analysis:")
    print(f"   Parameter reduction: {(params_wrong - params_right) / params_wrong * 100:.1f}%")
    print(f"   Size ratio: {params_wrong / params_right:.1f}x smaller")
    
    # Test a forward pass to see memory usage
    print(f"\n🧪 Memory Test (batch_size=32):")
    dummy_color = torch.randn(32, 3, 32, 32)
    dummy_brightness = torch.randn(32, 1, 32, 32)
    
    try:
        with torch.no_grad():
            out_wrong = model_wrong(dummy_color, dummy_brightness)
            print(f"   Full ResNet50: ✅ Forward pass successful, output shape: {out_wrong.shape}")
    except Exception as e:
        print(f"   Full ResNet50: ❌ Forward pass failed: {e}")
    
    try:
        with torch.no_grad():
            out_right = model_right(dummy_color, dummy_brightness)
            print(f"   CIFAR-100 optimized: ✅ Forward pass successful, output shape: {out_right.shape}")
    except Exception as e:
        print(f"   CIFAR-100 optimized: ❌ Forward pass failed: {e}")
    
    print(f"\n💡 Recommendation:")
    print(f"   Use MultiChannelResNetNetwork.for_cifar100() instead of multi_channel_resnet50()")
    print(f"   This should dramatically improve training performance!")

if __name__ == "__main__":
    test_architectures()
