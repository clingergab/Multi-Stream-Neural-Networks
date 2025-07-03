#!/usr/bin/env python3
"""
Test script to verify API consistency after refactoring.
"""

import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import with proper Python path setup
from src.models.basic_multi_channel.base_multi_channel_network import BaseMultiChannelNetwork
from src.models.basic_multi_channel.multi_channel_resnet_network import MultiChannelResNetNetwork

def test_base_multi_channel_network():
    """Test BaseMultiChannelNetwork API consistency."""
    print("🔍 Testing BaseMultiChannelNetwork API...")
    
    # Create model with CPU device for testing
    model = BaseMultiChannelNetwork(
        color_input_size=100,
        brightness_input_size=50,
        num_classes=10,
        hidden_sizes=[64, 32],
        device='cpu'  # Force CPU for consistent testing
    )
    
    # Test data
    color_data = torch.randn(4, 100)
    brightness_data = torch.randn(4, 50)
    
    # Test forward pass
    print("  ✓ Testing forward pass...")
    output = model(color_data, brightness_data)
    assert output.shape == (4, 10), f"Expected (4, 10), got {output.shape}"
    
    # Test extract_features (should return concatenated features)
    print("  ✓ Testing extract_features (concatenated)...")
    fused_features = model.extract_features(color_data, brightness_data)
    expected_feature_size = 32 * 2  # Last hidden size * 2 streams
    assert fused_features.shape == (4, expected_feature_size), f"Expected (4, {expected_feature_size}), got {fused_features.shape}"
    
    # Test get_separate_features (should return separate features)
    print("  ✓ Testing get_separate_features (separate)...")
    color_feats, brightness_feats = model.get_separate_features(color_data, brightness_data)
    assert color_feats.shape == (4, 32), f"Expected (4, 32), got {color_feats.shape}"
    assert brightness_feats.shape == (4, 32), f"Expected (4, 32), got {brightness_feats.shape}"
    
    # Test analyze_pathways
    print("  ✓ Testing analyze_pathways...")
    color_logits, brightness_logits = model.analyze_pathways(color_data, brightness_data)
    assert color_logits.shape == (4, 10), f"Expected (4, 10), got {color_logits.shape}"
    assert brightness_logits.shape == (4, 10), f"Expected (4, 10), got {brightness_logits.shape}"
    
    print("  ✅ BaseMultiChannelNetwork API tests passed!")
    return True

def test_multi_channel_resnet_network():
    """Test MultiChannelResNetNetwork API consistency."""
    print("🔍 Testing MultiChannelResNetNetwork API...")
    
    # Create model (small configuration) with CPU device for testing
    model = MultiChannelResNetNetwork(
        num_classes=10,
        color_input_channels=3,
        brightness_input_channels=1,
        num_blocks=[1, 1, 1, 1],  # Small for testing
        reduce_architecture=True,
        device='cpu'  # Force CPU for consistent testing
    )
    
    # Test data (small images)
    color_data = torch.randn(2, 3, 32, 32)
    brightness_data = torch.randn(2, 1, 32, 32)
    
    # Test forward pass
    print("  ✓ Testing forward pass...")
    output = model(color_data, brightness_data)
    assert output.shape == (2, 10), f"Expected (2, 10), got {output.shape}"
    
    # Test extract_features (should return concatenated features)
    print("  ✓ Testing extract_features (concatenated)...")
    fused_features = model.extract_features(color_data, brightness_data)
    # For reduced ResNet, final features should be 256 * expansion * 2 streams
    expected_feature_size = 256 * 1 * 2  # expansion=1 for basic blocks, reduced arch
    assert fused_features.shape == (2, expected_feature_size), f"Expected (2, {expected_feature_size}), got {fused_features.shape}"
    
    # Test get_separate_features (should return separate features)
    print("  ✓ Testing get_separate_features (separate)...")
    color_feats, brightness_feats = model.get_separate_features(color_data, brightness_data)
    expected_single_feature_size = 256 * 1  # expansion=1 for basic blocks, reduced arch
    assert color_feats.shape == (2, expected_single_feature_size), f"Expected (2, {expected_single_feature_size}), got {color_feats.shape}"
    assert brightness_feats.shape == (2, expected_single_feature_size), f"Expected (2, {expected_single_feature_size}), got {brightness_feats.shape}"
    
    # Test analyze_pathways
    print("  ✓ Testing analyze_pathways...")
    color_logits, brightness_logits = model.analyze_pathways(color_data, brightness_data)
    assert color_logits.shape == (2, 10), f"Expected (2, 10), got {color_logits.shape}"
    assert brightness_logits.shape == (2, 10), f"Expected (2, 10), got {brightness_logits.shape}"
    
    print("  ✅ MultiChannelResNetNetwork API tests passed!")
    return True

def test_api_consistency():
    """Test that both models have consistent APIs."""
    print("🔍 Testing API consistency between models...")
    
    # Both models should have the same public methods
    base_methods = set(dir(BaseMultiChannelNetwork))
    resnet_methods = set(dir(MultiChannelResNetNetwork))
    
    # Key methods that should be consistent
    key_methods = {
        'forward', 'extract_features', 'get_separate_features', 
        'analyze_pathways', 'compile', 'fit'
    }
    
    for method in key_methods:
        assert method in base_methods, f"BaseMultiChannelNetwork missing {method}"
        assert method in resnet_methods, f"MultiChannelResNetNetwork missing {method}"
    
    print("  ✅ API consistency tests passed!")
    return True

def main():
    """Run all tests."""
    print("🚀 Testing API Consistency After Refactoring")
    print("=" * 50)
    
    try:
        # Test individual models
        test_base_multi_channel_network()
        print()
        test_multi_channel_resnet_network()
        print()
        test_api_consistency()
        
        print()
        print("🎉 ALL TESTS PASSED!")
        print("✅ API refactoring successful - methods are now consistent")
        print()
        print("📊 New Consistent API:")
        print("  • extract_features() - Returns concatenated features for external classifiers")
        print("  • get_separate_features() - Returns separate pathway features for research")
        print("  • analyze_pathways() - Returns pathway-specific classification logits")
        print("  • forward() - Standard training/inference method")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
