#!/usr/bin/env python3
"""
Test _validate method with tensor inputs.
"""

import torch
from src.models2.core.resnet import resnet18

def test_validate_with_tensors():
    """Test that _validate method can handle tensor inputs."""
    print("Testing _validate method with tensor inputs...")
    
    # Create test data
    X_test = torch.randn(20, 3, 32, 32)
    y_test = torch.randint(0, 10, (20,))
    
    # Create and compile model
    model = resnet18(num_classes=10)
    model.compile(optimizer='adam', loss='cross_entropy', lr=0.001)
    
    print("1. Testing _validate with tensor inputs...")
    # Test _validate with tensors directly
    loss, accuracy = model._validate(X_test, targets=y_test, batch_size=8)
    print(f"✓ _validate with tensors: Loss={loss:.4f}, Accuracy={accuracy:.2f}%")
    
    # Test error handling
    try:
        model._validate(X_test)  # Missing targets
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Correctly caught error in _validate: {e}")
    
    print("✅ _validate tensor input test passed!")
    return loss, accuracy

if __name__ == "__main__":
    loss, accuracy = test_validate_with_tensors()
