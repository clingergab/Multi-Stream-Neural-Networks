#!/usr/bin/env python3
"""
Simple test to verify that our models can be imported and instantiated.
"""

import os
import sys
import torch

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import the models
from src.models.basic_multi_channel.base_multi_channel_network import BaseMultiChannelNetwork
from src.models.basic_multi_channel.multi_channel_resnet_network import MultiChannelResNetNetwork

def test_base_multi_channel_network():
    """Test that BaseMultiChannelNetwork can be instantiated and forward pass works."""
    print("Testing BaseMultiChannelNetwork...")
    
    # Create the model
    model = BaseMultiChannelNetwork(
        color_input_size=3*32*32,  # Flattened RGB image
        brightness_input_size=32*32,  # Flattened brightness image
        hidden_sizes=[128, 64],
        num_classes=10,
        device='cpu'
    )
    
    # Create random input tensors
    color_input = torch.randn(4, 3*32*32)  # 4 samples, flattened RGB images
    brightness_input = torch.randn(4, 32*32)  # 4 samples, flattened brightness images
    
    # Test forward pass
    with torch.no_grad():
        output = model(color_input, brightness_input)
        print(f"  Forward pass output shape: {output.shape}")
        assert output.shape == (4, 10), f"Expected output shape (4, 10), got {output.shape}"
        
        # Test pathway analysis
        color_out, brightness_out = model.analyze_pathways(color_input, brightness_input)
        print(f"  Pathway analysis shapes: color {color_out.shape}, brightness {brightness_out.shape}")
        assert color_out.shape == (4, 10), f"Expected color output shape (4, 10), got {color_out.shape}"
        assert brightness_out.shape == (4, 10), f"Expected brightness output shape (4, 10), got {brightness_out.shape}"
    
    print("  ✅ BaseMultiChannelNetwork tests passed!")
    return True

def test_multi_channel_resnet_network():
    """Test that MultiChannelResNetNetwork can be instantiated and forward pass works."""
    print("\nTesting MultiChannelResNetNetwork...")
    
    # Create the model
    model = MultiChannelResNetNetwork(
        num_classes=10,
        color_input_channels=3,  # RGB channels
        brightness_input_channels=1,  # Brightness channel
        num_blocks=[1, 1, 1, 1],  # Smaller ResNet for testing
        device='cpu'
    )
    
    # Create random input tensors (batch, channels, height, width)
    color_input = torch.randn(4, 3, 32, 32)  # 4 samples, RGB images
    brightness_input = torch.randn(4, 1, 32, 32)  # 4 samples, brightness images
    
    # Test forward pass
    with torch.no_grad():
        output = model(color_input, brightness_input)
        print(f"  Forward pass output shape: {output.shape}")
        assert output.shape == (4, 10), f"Expected output shape (4, 10), got {output.shape}"
        
        # Test pathway analysis
        color_out, brightness_out = model.analyze_pathways(color_input, brightness_input)
        print(f"  Pathway analysis shapes: color {color_out.shape}, brightness {brightness_out.shape}")
        assert color_out.shape == (4, 10), f"Expected color output shape (4, 10), got {color_out.shape}"
        assert brightness_out.shape == (4, 10), f"Expected brightness output shape (4, 10), got {brightness_out.shape}"
    
    print("  ✅ MultiChannelResNetNetwork tests passed!")
    return True

def main():
    """Run all tests."""
    print("🚀 Running simple model tests\n")
    
    # Test both models
    base_success = test_base_multi_channel_network()
    resnet_success = test_multi_channel_resnet_network()
    
    # Print summary
    if base_success and resnet_success:
        print("\n🎉 All model tests passed! 🎉")
        return True
    else:
        print("\n❌ Some tests failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
