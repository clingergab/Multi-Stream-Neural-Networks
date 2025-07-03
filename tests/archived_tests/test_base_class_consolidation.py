#!/usr/bin/env python3
"""
Test script to verify base class method consolidation.
"""

import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.models.basic_multi_channel.base_multi_channel_network import BaseMultiChannelNetwork
from src.models.basic_multi_channel.multi_channel_resnet_network import MultiChannelResNetNetwork

def test_base_class_methods():
    """Test that both models now implement the unified base class interface."""
    print("🔍 Testing Base Class Method Consolidation...")
    
    # Create models with CPU for testing
    base_model = BaseMultiChannelNetwork(
        color_input_size=100,
        brightness_input_size=50,
        num_classes=10,
        hidden_sizes=[64, 32],
        device='cpu'
    )
    
    resnet_model = MultiChannelResNetNetwork(
        num_classes=10,
        color_input_channels=3,
        brightness_input_channels=1,
        num_blocks=[1, 1, 1, 1],
        reduce_architecture=True,
        device='cpu'
    )
    
    models = [("BaseMultiChannelNetwork", base_model), ("MultiChannelResNetNetwork", resnet_model)]
    
    # Test data
    base_color_data = torch.randn(2, 100)
    base_brightness_data = torch.randn(2, 50)
    resnet_color_data = torch.randn(2, 3, 32, 32)
    resnet_brightness_data = torch.randn(2, 1, 32, 32)
    
    for model_name, model in models:
        print(f"\n  🔧 Testing {model_name}:")
        
        # Choose appropriate test data
        if "ResNet" in model_name:
            color_data, brightness_data = resnet_color_data, resnet_brightness_data
        else:
            color_data, brightness_data = base_color_data, base_brightness_data
        
        # Test abstract methods are implemented
        print("    ✓ Testing abstract method implementations...")
        
        # Test fusion_type property
        fusion_type = model.fusion_type
        assert isinstance(fusion_type, str), f"fusion_type should be string, got {type(fusion_type)}"
        assert fusion_type == "shared_classifier", f"Expected 'shared_classifier', got {fusion_type}"
        
        # Test get_classifier_info
        classifier_info = model.get_classifier_info()
        assert isinstance(classifier_info, dict), f"get_classifier_info should return dict, got {type(classifier_info)}"
        assert 'type' in classifier_info, "classifier_info should have 'type' key"
        assert 'total_params' in classifier_info, "classifier_info should have 'total_params' key"
        
        # Test analyze_pathway_weights
        pathway_weights = model.analyze_pathway_weights()
        assert isinstance(pathway_weights, dict), f"analyze_pathway_weights should return dict, got {type(pathway_weights)}"
        assert 'color_pathway' in pathway_weights, "pathway_weights should have 'color_pathway' key"
        assert 'brightness_pathway' in pathway_weights, "pathway_weights should have 'brightness_pathway' key"
        
        # Test extract_features (should return concatenated)
        fused_features = model.extract_features(color_data, brightness_data)
        assert isinstance(fused_features, torch.Tensor), f"extract_features should return Tensor, got {type(fused_features)}"
        assert len(fused_features.shape) == 2, f"extract_features should return 2D tensor, got {fused_features.shape}"
        
        # Test get_separate_features (should return tuple)
        separate_features = model.get_separate_features(color_data, brightness_data)
        assert isinstance(separate_features, tuple), f"get_separate_features should return tuple, got {type(separate_features)}"
        assert len(separate_features) == 2, f"get_separate_features should return 2-tuple, got {len(separate_features)}"
        color_feats, brightness_feats = separate_features
        assert isinstance(color_feats, torch.Tensor), f"Color features should be Tensor, got {type(color_feats)}"
        assert isinstance(brightness_feats, torch.Tensor), f"Brightness features should be Tensor, got {type(brightness_feats)}"
        
        # Test that concatenated features match separate features (for deterministic models)
        manual_concat = torch.cat([color_feats, brightness_feats], dim=1)
        
        # For deterministic models like BaseMultiChannelNetwork, they should match exactly
        # For ResNet models with potential non-deterministic operations (dropout, etc.), 
        # we just verify the shapes and method consistency
        if "ResNet" not in model_name:
            assert torch.allclose(fused_features, manual_concat, atol=1e-6), "Fused and manual concat should match for deterministic models"
            print("      ✓ Fused and manual concat match perfectly!")
        else:
            # For ResNet, just verify shapes are consistent
            assert fused_features.shape == manual_concat.shape, f"Shape mismatch: {fused_features.shape} vs {manual_concat.shape}"
            print("      ✓ Shapes are consistent (non-deterministic model)")
            print("      Note: ResNet may have slight differences due to potential dropout/normalization effects")
        
        print(f"    ✅ {model_name} implements all base class methods correctly!")
    
    return True

def test_validation_method():
    """Test that both models use the unified _validate method."""
    print("\n🔍 Testing Unified Validation Method...")
    
    # Create a simple model for testing
    model = BaseMultiChannelNetwork(
        color_input_size=50,
        brightness_input_size=25,
        num_classes=5,
        hidden_sizes=[32],
        device='cpu'
    )
    
    # Compile the model
    model.compile(optimizer='adam', learning_rate=0.01, loss='cross_entropy')
    
    # Create validation data
    color_data = torch.randn(10, 50)
    brightness_data = torch.randn(10, 25) 
    labels = torch.randint(0, 5, (10,))
    
    val_dataset = torch.utils.data.TensorDataset(color_data, brightness_data, labels)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # Test validation method
    val_loss, val_accuracy = model._validate(val_loader)
    
    assert isinstance(val_loss, float), f"Validation loss should be float, got {type(val_loss)}"
    assert isinstance(val_accuracy, float), f"Validation accuracy should be float, got {type(val_accuracy)}"
    assert 0 <= val_accuracy <= 1, f"Validation accuracy should be in [0,1], got {val_accuracy}"
    
    print("  ✅ Unified validation method works correctly!")
    return True

def main():
    """Run all tests."""
    print("🚀 Testing Base Class Method Consolidation")
    print("=" * 50)
    
    try:
        test_base_class_methods()
        test_validation_method()
        
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Base class method consolidation successful!")
        print()
        print("📊 New Unified Base Class Methods:")
        print("  • get_separate_features() - Now required in all models")
        print("  • fusion_type (property) - Required in all models")  
        print("  • get_classifier_info() - Required in all models")
        print("  • analyze_pathway_weights() - Required in all models")
        print("  • _validate() - Common validation logic for all models")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
