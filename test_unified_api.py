#!/usr/bin/env python3
"""
Test script to verify API consistency between dense and ResNet models.
Demonstrates shared classifier fusion in both architectures.
"""

import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import with proper path handling
try:
    from src.models.basic_multi_channel.base_multi_channel_network import BaseMultiChannelNetwork
    from src.models.basic_multi_channel.multi_channel_resnet_network import MultiChannelResNetNetwork
except ImportError:
    # Alternative import method
    sys.path.insert(0, os.path.dirname(__file__))
    from src.models.basic_multi_channel.base_multi_channel_network import BaseMultiChannelNetwork
    from src.models.basic_multi_channel.multi_channel_resnet_network import MultiChannelResNetNetwork

def test_api_consistency():
    """Test that both models have the same API and fusion capabilities."""
    print("Testing API consistency between Dense and ResNet models...")
    
    # Test parameters
    batch_size = 4
    num_classes = 10
    color_channels = 3
    brightness_channels = 1
    img_size = 32  # Smaller for faster testing
    
    # Create test data - use CPU for testing to avoid device conflicts
    device = torch.device('cpu')
    color_data = torch.randn(batch_size, color_channels, img_size, img_size).to(device)
    brightness_data = torch.randn(batch_size, brightness_channels, img_size, img_size).to(device)
    
    print(f"\nTest data shapes:")
    print(f"Color: {color_data.shape}")
    print(f"Brightness: {brightness_data.shape}")
    
    # Test Dense model with both fusion strategies
    print("\n" + "="*60)
    print("TESTING DENSE MODEL")
    print("="*60)
    
    for use_shared in [True, False]:
        print(f"\nDense Model - use_shared_classifier={use_shared}")
        dense_model = BaseMultiChannelNetwork(
            color_input_size=(color_channels * img_size * img_size),  # Flattened size for dense model
            brightness_input_size=(brightness_channels * img_size * img_size),
            hidden_sizes=[64, 32],
            num_classes=num_classes,
            use_shared_classifier=use_shared,
            device='cpu'  # Force CPU for testing
        )
        
        # Dense model expects flattened input
        color_flat = color_data.view(batch_size, -1)
        brightness_flat = brightness_data.view(batch_size, -1)
        
        # Test forward method (training/inference)
        with torch.no_grad():
            output = dense_model(color_flat, brightness_flat)
            print(f"  forward() output shape: {output.shape}")
            print(f"  Expected shape: ({batch_size}, {num_classes})")
            assert output.shape == (batch_size, num_classes), f"Wrong output shape: {output.shape}"
            
            # Test analyze_pathways method (research)
            color_out, brightness_out = dense_model.analyze_pathways(color_flat, brightness_flat)
            print(f"  analyze_pathways() - color: {color_out.shape}, brightness: {brightness_out.shape}")
            assert color_out.shape == (batch_size, num_classes), f"Wrong color analysis shape: {color_out.shape}"
            assert brightness_out.shape == (batch_size, num_classes), f"Wrong brightness analysis shape: {brightness_out.shape}"
        
        print(f"  ✓ Dense model with use_shared_classifier={use_shared} working correctly")
    
    # Test ResNet model with both fusion strategies
    print("\n" + "="*60)
    print("TESTING RESNET MODEL")
    print("="*60)
    
    for use_shared in [True, False]:
        print(f"\nResNet Model - use_shared_classifier={use_shared}")
        resnet_model = MultiChannelResNetNetwork(
            num_classes=num_classes,
            color_input_channels=color_channels,
            brightness_input_channels=brightness_channels,
            num_blocks=[1, 1, 1, 1],  # Smaller for faster testing
            use_shared_classifier=use_shared,
            device='cpu'  # Force CPU for testing
        )
        
        # Test forward method (training/inference)
        with torch.no_grad():
            output = resnet_model(color_data, brightness_data)
            print(f"  forward() output shape: {output.shape}")
            print(f"  Expected shape: ({batch_size}, {num_classes})")
            assert output.shape == (batch_size, num_classes), f"Wrong output shape: {output.shape}"
            
            # Test analyze_pathways method (research)
            color_out, brightness_out = resnet_model.analyze_pathways(color_data, brightness_data)
            print(f"  analyze_pathways() - color: {color_out.shape}, brightness: {brightness_out.shape}")
            assert color_out.shape == (batch_size, num_classes), f"Wrong color analysis shape: {color_out.shape}"
            assert brightness_out.shape == (batch_size, num_classes), f"Wrong brightness analysis shape: {brightness_out.shape}"
        
        print(f"  ✓ ResNet model with use_shared_classifier={use_shared} working correctly")
    
    print("\n" + "="*60)
    print("API CONSISTENCY TEST PASSED!")
    print("="*60)
    print("✓ Both models support the same API:")
    print("  - forward() returns single tensor for training/inference")
    print("  - analyze_pathways() returns tuple for research")
    print("  - Both support use_shared_classifier parameter")
    print("  - Feature-level fusion (shared=True) vs output-level fusion (shared=False)")

def test_fusion_difference():
    """Demonstrate the difference between fusion strategies."""
    print("\n" + "="*60)
    print("DEMONSTRATING FUSION STRATEGIES")
    print("="*60)
    
    # Test parameters
    batch_size = 2
    num_classes = 5
    color_channels = 3
    brightness_channels = 1
    img_size = 32
    
    # Create test data - use CPU for testing to avoid device conflicts
    device = torch.device('cpu')
    color_data = torch.randn(batch_size, color_channels, img_size, img_size).to(device)
    brightness_data = torch.randn(batch_size, brightness_channels, img_size, img_size).to(device)
    
    # Test with ResNet (similar patterns apply to Dense model)
    print("\nUsing ResNet model to demonstrate fusion differences:")
    
    # Shared classifier (feature-level fusion)
    model_shared = MultiChannelResNetNetwork(
        num_classes=num_classes,
        color_input_channels=color_channels,
        brightness_input_channels=brightness_channels,
        num_blocks=[1, 1, 1, 1],
        use_shared_classifier=True,
        device='cpu'  # Force CPU for testing
    )
    
    # Separate classifiers (output-level fusion)
    model_separate = MultiChannelResNetNetwork(
        num_classes=num_classes,
        color_input_channels=color_channels,
        brightness_input_channels=brightness_channels,
        num_blocks=[1, 1, 1, 1],
        use_shared_classifier=False,
        device='cpu'  # Force CPU for testing
    )
    
    with torch.no_grad():
        # Get outputs from both fusion strategies
        shared_output = model_shared(color_data, brightness_data)
        separate_output = model_separate(color_data, brightness_data)
        
        print(f"\nShared classifier output shape: {shared_output.shape}")
        print(f"Separate classifier output shape: {separate_output.shape}")
        print(f"Both outputs have same shape: {shared_output.shape == separate_output.shape}")
        
        # Show pathway analysis
        shared_color, shared_brightness = model_shared.analyze_pathways(color_data, brightness_data)
        separate_color, separate_brightness = model_separate.analyze_pathways(color_data, brightness_data)
        
        print(f"\nPathway analysis shapes:")
        print(f"Shared model - color: {shared_color.shape}, brightness: {shared_brightness.shape}")
        print(f"Separate model - color: {separate_color.shape}, brightness: {separate_brightness.shape}")
        
        print(f"\nKey differences:")
        print(f"1. Shared classifier: Features from both streams are concatenated,")
        print(f"   then passed through a single classifier for unified decision-making")
        print(f"2. Separate classifiers: Each stream gets its own classifier,")
        print(f"   outputs are added for final prediction")
        print(f"3. Both strategies produce single tensor for training/inference")

if __name__ == "__main__":
    test_api_consistency()
    test_fusion_difference()
    print(f"\n🎉 All tests passed! Both models have consistent APIs and fusion capabilities.")
