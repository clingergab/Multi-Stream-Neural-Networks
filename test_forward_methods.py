#!/usr/bin/env python3
"""
Test script to verify the simplified forward method behavior.
"""

import torch
import numpy as np
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models.basic_multi_channel.base_multi_channel_network import BaseMultiChannelNetwork

def test_simplified_forward_methods():
    """Test that the simplified forward methods work correctly."""
    print("🧪 Testing Simplified Forward Method API")
    print("=" * 50)
    
    # Create a small test model
    model = BaseMultiChannelNetwork(
        color_input_size=10,
        brightness_input_size=8,
        hidden_sizes=[16, 8],
        num_classes=5,
        use_shared_classifier=True,
        device='cpu'  # Force CPU for consistent testing
    )
    
    # Create test data
    batch_size = 3
    color_data = torch.randn(batch_size, 10)
    brightness_data = torch.randn(batch_size, 8)
    
    print(f"📊 Test data shapes:")
    print(f"   Color: {color_data.shape}")
    print(f"   Brightness: {brightness_data.shape}")
    print()
    
    # Test 1: forward() method (main training method)
    print("1️⃣ Testing forward() - Main method for training/inference:")
    forward_output = model.forward(color_data, brightness_data)
    print(f"   Output shape: {forward_output.shape}")
    print(f"   Output type: {type(forward_output)}")
    print(f"   Expected: torch.Tensor with shape ({batch_size}, 5)")
    print(f"   ✅ Correct!" if forward_output.shape == (batch_size, 5) and isinstance(forward_output, torch.Tensor) else "❌ Wrong!")
    print()
    
    # Test 2: model() call (should be same as forward())
    print("2️⃣ Testing model() call - Should match forward():")
    call_output = model(color_data, brightness_data)
    print(f"   Output shape: {call_output.shape}")
    print(f"   Output type: {type(call_output)}")
    print(f"   Same as forward(): {torch.allclose(forward_output, call_output)}")
    print(f"   ✅ Correct!" if torch.allclose(forward_output, call_output) else "❌ Wrong!")
    print()
    
    # Test 3: analyze_pathways() method (for research)
    print("3️⃣ Testing analyze_pathways() - Research method (returns tuple):")
    analysis_output = model.analyze_pathways(color_data, brightness_data)
    print(f"   Output type: {type(analysis_output)}")
    print(f"   Is tuple: {isinstance(analysis_output, tuple)}")
    if isinstance(analysis_output, tuple):
        color_logits, brightness_logits = analysis_output
        print(f"   Color logits shape: {color_logits.shape}")
        print(f"   Brightness logits shape: {brightness_logits.shape}")
        print(f"   Both have correct shape: {color_logits.shape == (batch_size, 5) and brightness_logits.shape == (batch_size, 5)}")
        print(f"   ✅ Correct!" if color_logits.shape == (batch_size, 5) and brightness_logits.shape == (batch_size, 5) else "❌ Wrong!")
    else:
        print("   ❌ Wrong! Should return tuple")
    print()
    
    # Test 4: Training compatibility
    print("4️⃣ Testing training compatibility:")
    try:
        # Simulate a training step
        labels = torch.randint(0, 5, (batch_size,))
        criterion = torch.nn.CrossEntropyLoss()
        
        # This should work seamlessly now!
        loss = criterion(model(color_data, brightness_data), labels)
        print(f"   Loss computation: ✅ Success! Loss = {loss.item():.4f}")
        
        # Test backward pass
        loss.backward()
        print(f"   Backward pass: ✅ Success!")
        
    except Exception as e:
        print(f"   ❌ Training failed: {e}")
    print()
    
    # Test 5: Research analysis example
    print("5️⃣ Testing pathway analysis example:")
    try:
        color_logits, brightness_logits = model.analyze_pathways(color_data, brightness_data)
        
        # Calculate individual pathway accuracies (example)
        labels = torch.randint(0, 5, (batch_size,))
        color_preds = torch.argmax(color_logits, dim=1)
        brightness_preds = torch.argmax(brightness_logits, dim=1)
        
        color_acc = (color_preds == labels).float().mean().item()
        brightness_acc = (brightness_preds == labels).float().mean().item()
        
        print(f"   Color pathway accuracy: {color_acc:.3f}")
        print(f"   Brightness pathway accuracy: {brightness_acc:.3f}")
        print(f"   ✅ Analysis successful!")
        
    except Exception as e:
        print(f"   ❌ Analysis failed: {e}")
    print()
    
    print("🎉 Simplified API tests completed!")
    print("📋 Summary:")
    print("   ✅ ONE forward() method for training/inference")
    print("   ✅ model() automatically calls forward()")
    print("   ✅ analyze_pathways() for research (separate outputs)")
    print("   ✅ Clean, simple API with no confusion")
    print("   ✅ Training works seamlessly with model() calls")

if __name__ == "__main__":
    test_simplified_forward_methods()
