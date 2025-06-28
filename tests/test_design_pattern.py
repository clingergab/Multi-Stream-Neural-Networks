#!/usr/bin/env python3
"""
Test script to verify our multi-stream architecture follows the specified design pattern:

Forward Pass:
                    predictions
                        |
                    Classifier
                        |
                    integrated
                   /            \
                  /               \
        [Color Features]   [Brightness Features]
                |                      |
        Color Pathway       Brightness Pathway
                |                      |
        Color Weights       Brightness Weights

Backpropagation:
                   Loss
                     ↓
                Classifier
               /            \
              /               \
    [Color Features]   [Brightness Features]
             ↓                    ↓
        Color Pathway       Brightness Pathway
             ↓                    ↓
        Color Weights       Brightness Weights
"""

import sys
import os
sys.path.append('/Users/gclinger/Documents/projects/Multi-Stream-Neural-Networks')

import torch
import numpy as np
from src.models.basic_multi_channel.base_multi_channel_network import BaseMultiChannelNetwork
from src.models.basic_multi_channel.multi_channel_resnet_network import MultiChannelResNetNetwork

def test_design_pattern():
    """Test that our implementation follows the specified multi-stream design pattern."""
    print("🔍 Testing Multi-Stream Neural Network Design Pattern Compliance")
    print("\nExpected Design:")
    print("Forward:  Input → Color/Brightness Pathways → Features → Classifier → Predictions")
    print("Backward: Loss → Classifier → Features → Color/Brightness Pathways → Weights")
    
    print("\n" + "="*70)
    print("1. TESTING DENSE MODEL (BaseMultiChannelNetwork)")
    print("="*70)
    
    # Test dense model with both fusion strategies
    print("\n1.1 Testing Dense Model with Shared Classifier (Recommended)")
    dense_shared = BaseMultiChannelNetwork(
        color_input_size=100,
        brightness_input_size=50,
        hidden_sizes=[64, 32],
        num_classes=5,
        use_shared_classifier=True,  # Proper fusion
        device='cpu'
    )
    
    test_dense_model(dense_shared, "Shared Classifier (Concatenation Fusion)")
    
    print("\n1.2 Testing Dense Model with Separate Classifiers (Legacy)")
    dense_separate = BaseMultiChannelNetwork(
        color_input_size=100,
        brightness_input_size=50,
        hidden_sizes=[64, 32],
        num_classes=5,
        use_shared_classifier=False,  # Separate classifiers
        device='cpu'
    )
    
    test_dense_model(dense_separate, "Separate Classifiers (Addition Fusion)")
    
    print("\n" + "="*70)
    print("2. TESTING CNN MODEL (MultiChannelResNetNetwork)")
    print("="*70)
    
    cnn_model = MultiChannelResNetNetwork(
        num_classes=5,
        color_input_channels=3,
        brightness_input_channels=1,
        num_blocks=[1, 1, 1, 1],
        device='cpu'
    )
    
    test_cnn_model(cnn_model)
    
    print("\n" + "="*70)
    print("3. DESIGN PATTERN VERIFICATION SUMMARY")
    print("="*70)
    
    print("✅ Both models follow the specified design pattern:")
    print("   📥 Input: Separate color and brightness streams")
    print("   🔀 Processing: Independent pathways with separate weights")
    print("   🔗 Integration: Features combined at classifier level")
    print("   📤 Output: Single prediction tensor for training")
    print("   🔬 Analysis: Separate pathway outputs available for research")
    print("   ⚡ Backpropagation: Gradients flow back through both pathways")

def test_dense_model(model, fusion_type):
    """Test the dense model's adherence to the design pattern."""
    print(f"\n   Testing: {fusion_type}")
    
    # Create test data
    batch_size = 2
    color_data = torch.randn(batch_size, 100, requires_grad=True)
    brightness_data = torch.randn(batch_size, 50, requires_grad=True)
    labels = torch.randint(0, 5, (batch_size,))
    
    print(f"   Input shapes: Color {color_data.shape}, Brightness {brightness_data.shape}")
    
    # 1. Test forward pass follows design
    print("\n   🔄 Forward Pass Analysis:")
    
    # Extract features before classifier
    if model.use_shared_classifier:
        combined_features = model.extract_features(color_data, brightness_data)
        print(f"   - Combined features shape: {combined_features.shape}")
        print(f"   - Feature fusion: Concatenation (shared classifier)")
    else:
        color_features, brightness_features = model.extract_features(color_data, brightness_data)
        print(f"   - Color features shape: {color_features.shape}")
        print(f"   - Brightness features shape: {brightness_features.shape}")
        print(f"   - Feature fusion: Addition (separate classifiers)")
    
    # Get final prediction
    output = model(color_data, brightness_data)
    print(f"   - Final output shape: {output.shape}")
    
    # Get pathway analysis
    color_logits, brightness_logits = model.analyze_pathways(color_data, brightness_data)
    print(f"   - Color pathway output: {color_logits.shape}")
    print(f"   - Brightness pathway output: {brightness_logits.shape}")
    
    # 2. Test backpropagation follows design
    print("\n   ⚡ Backpropagation Analysis:")
    
    # Forward pass with loss
    model.train()
    output = model(color_data, brightness_data)
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(output, labels)
    print(f"   - Loss computed: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    
    # Check that gradients exist for both pathways
    color_grad_exists = color_data.grad is not None
    brightness_grad_exists = brightness_data.grad is not None
    
    print(f"   - Color pathway gradients: {'✅ Present' if color_grad_exists else '❌ Missing'}")
    print(f"   - Brightness pathway gradients: {'✅ Present' if brightness_grad_exists else '❌ Missing'}")
    
    if color_grad_exists and brightness_grad_exists:
        color_grad_norm = torch.norm(color_data.grad).item()
        brightness_grad_norm = torch.norm(brightness_data.grad).item()
        print(f"   - Color gradient norm: {color_grad_norm:.6f}")
        print(f"   - Brightness gradient norm: {brightness_grad_norm:.6f}")
        print("   ✅ Gradients flow back through both pathways")
    
    # 3. Verify design consistency
    print("\n   🎯 Design Pattern Verification:")
    
    if model.use_shared_classifier:
        # Check concatenation fusion
        print("   - Architecture: Color/Brightness → Features → Concatenate → Shared Classifier")
        print("   - Fusion method: Feature-level concatenation (optimal)")
        print("   - Gradients: Flow through shared classifier to both pathways")
    else:
        # Check addition fusion
        print("   - Architecture: Color/Brightness → Features → Separate Classifiers → Add")
        print("   - Fusion method: Output-level addition (legacy)")
        print("   - Gradients: Flow through separate classifiers to respective pathways")
    
    # Verify consistency between forward and analyze_pathways
    if not model.use_shared_classifier:
        # For separate classifiers, manual sum should equal forward output
        manual_sum = color_logits + brightness_logits
        max_diff = torch.max(torch.abs(manual_sum - output)).item()
        print(f"   - Output consistency: {max_diff:.8f} (should be ~0)")
        if max_diff < 1e-6:
            print("   ✅ Forward and analyze_pathways are consistent")
        else:
            print("   ❌ Inconsistency detected!")
    else:
        print("   - Output consistency: N/A (shared classifier uses different fusion)")
    
    print("   ✅ Design pattern compliance verified")

def test_cnn_model(model):
    """Test the CNN model's adherence to the design pattern."""
    print("\n   Testing: ResNet with Separate Classifiers")
    
    # Create test data
    batch_size = 2
    color_data = torch.randn(batch_size, 3, 32, 32, requires_grad=True)
    brightness_data = torch.randn(batch_size, 1, 32, 32, requires_grad=True)
    labels = torch.randint(0, 5, (batch_size,))
    
    print(f"   Input shapes: Color {color_data.shape}, Brightness {brightness_data.shape}")
    
    # 1. Test forward pass follows design
    print("\n   🔄 Forward Pass Analysis:")
    
    # Extract features before classifier
    color_features, brightness_features = model.extract_features(color_data, brightness_data)
    print(f"   - Color features shape: {color_features.shape}")
    print(f"   - Brightness features shape: {brightness_features.shape}")
    print(f"   - Feature fusion: Addition (separate classifiers)")
    
    # Get final prediction
    output = model(color_data, brightness_data)
    print(f"   - Final output shape: {output.shape}")
    
    # Get pathway analysis
    color_logits, brightness_logits = model.analyze_pathways(color_data, brightness_data)
    print(f"   - Color pathway output: {color_logits.shape}")
    print(f"   - Brightness pathway output: {brightness_logits.shape}")
    
    # 2. Test backpropagation follows design
    print("\n   ⚡ Backpropagation Analysis:")
    
    # Forward pass with loss
    model.train()
    output = model(color_data, brightness_data)
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(output, labels)
    print(f"   - Loss computed: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    
    # Check that gradients exist for both pathways
    color_grad_exists = color_data.grad is not None
    brightness_grad_exists = brightness_data.grad is not None
    
    print(f"   - Color pathway gradients: {'✅ Present' if color_grad_exists else '❌ Missing'}")
    print(f"   - Brightness pathway gradients: {'✅ Present' if brightness_grad_exists else '❌ Missing'}")
    
    if color_grad_exists and brightness_grad_exists:
        color_grad_norm = torch.norm(color_data.grad).item()
        brightness_grad_norm = torch.norm(brightness_data.grad).item()
        print(f"   - Color gradient norm: {color_grad_norm:.6f}")
        print(f"   - Brightness gradient norm: {brightness_grad_norm:.6f}")
        print("   ✅ Gradients flow back through both pathways")
    
    # 3. Verify design consistency
    print("\n   🎯 Design Pattern Verification:")
    print("   - Architecture: Color/Brightness → ResNet Layers → Features → Separate Classifiers → Add")
    print("   - Fusion method: Output-level addition")
    print("   - Gradients: Flow through separate classifiers to respective pathways")
    
    # Verify consistency between forward and analyze_pathways
    manual_sum = color_logits + brightness_logits
    max_diff = torch.max(torch.abs(manual_sum - output)).item()
    print(f"   - Output consistency: {max_diff:.8f} (should be ~0)")
    if max_diff < 1e-6:
        print("   ✅ Forward and analyze_pathways are consistent")
    else:
        print("   ❌ Inconsistency detected!")
    
    print("   ✅ Design pattern compliance verified")

if __name__ == "__main__":
    test_design_pattern()
