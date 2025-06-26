#!/usr/bin/env python3
"""
Quick test script to verify MultiChannelResNetNetwork API consistency.
This test ensures the new simplified API works correctly.
"""

import sys
import os
sys.path.append('/Users/gclinger/Documents/projects/Multi-Stream-Neural-Networks')

import torch
import numpy as np
from src.models.basic_multi_channel.multi_channel_resnet_network import MultiChannelResNetNetwork

def test_resnet_api():
    """Test the new simplified API for MultiChannelResNetNetwork."""
    print("Testing MultiChannelResNetNetwork API...")
    
    # Create a small model for testing
    model = MultiChannelResNetNetwork(
        num_classes=5,
        color_input_channels=3,
        brightness_input_channels=1,
        num_blocks=[1, 1, 1, 1],  # Small model for testing
        device='cpu'
    )
    
    # Create test data
    batch_size = 2
    color_data = torch.randn(batch_size, 3, 32, 32)
    brightness_data = torch.randn(batch_size, 1, 32, 32)
    
    # Test 1: forward() method (main API for training/inference)
    print("\n1. Testing forward() method (training/inference)...")
    model.eval()
    
    # Test using model.__call__ (which calls forward)
    output = model(color_data, brightness_data)
    print(f"   Output shape: {output.shape}")
    print(f"   Output type: {type(output)}")
    assert output.shape == (batch_size, 5), f"Expected shape ({batch_size}, 5), got {output.shape}"
    assert isinstance(output, torch.Tensor), "Output should be a single tensor"
    print("   ✓ forward() returns single tensor as expected")
    
    # Test 2: analyze_pathways() method (research API)
    print("\n2. Testing analyze_pathways() method (research)...")
    color_logits, brightness_logits = model.analyze_pathways(color_data, brightness_data)
    print(f"   Color logits shape: {color_logits.shape}")
    print(f"   Brightness logits shape: {brightness_logits.shape}")
    assert color_logits.shape == (batch_size, 5), f"Expected shape ({batch_size}, 5), got {color_logits.shape}"
    assert brightness_logits.shape == (batch_size, 5), f"Expected shape ({batch_size}, 5), got {brightness_logits.shape}"
    print("   ✓ analyze_pathways() returns tuple of tensors as expected")
    
    # Test 3: Verify outputs are consistent
    print("\n3. Testing output consistency...")
    # The combined output should be the sum of individual pathways
    combined_manual = color_logits + brightness_logits
    forward_output = model(color_data, brightness_data)
    
    difference = torch.max(torch.abs(combined_manual - forward_output)).item()
    print(f"   Max difference between manual sum and forward(): {difference:.6f}")
    assert difference < 1e-5, f"Outputs should be identical, but differ by {difference}"
    print("   ✓ Outputs are consistent between forward() and analyze_pathways()")
    
    # Test 4: Verify training compatibility
    print("\n4. Testing training compatibility...")
    model.train()
    
    # Create fake labels
    labels = torch.randint(0, 5, (batch_size,))
    
    # Test that forward pass works with loss computation
    output = model(color_data, brightness_data)
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(output, labels)
    print(f"   Loss computed successfully: {loss.item():.4f}")
    print("   ✓ Training compatibility confirmed")
    
    print("\n✅ All MultiChannelResNetNetwork API tests passed!")
    print("\nAPI Summary:")
    print("- Use model(color, brightness) for training, inference, and evaluation")
    print("- Use model.analyze_pathways(color, brightness) for research analysis")
    print("- No other forward methods needed - clean and simple API!")

if __name__ == "__main__":
    test_resnet_api()
