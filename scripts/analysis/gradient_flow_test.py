"""
Gradient Flow Test - Verify gradients flow properly through our ResNet implementation.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.builders.model_factory import create_model

def test_gradient_flow():
    """Test that gradients flow properly through the ResNet model."""
    print("🔍 Testing Gradient Flow in MultiChannelResNetNetwork")
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Create model
    model = create_model(
        'multi_channel_resnet18',
        num_classes=10,
        color_input_channels=3,
        brightness_input_channels=1,
        activation='relu'
    ).to(device)
    
    # Create dummy data
    batch_size = 4
    color_input = torch.randn(batch_size, 3, 28, 28, device=device, requires_grad=True)
    brightness_input = torch.randn(batch_size, 1, 28, 28, device=device, requires_grad=True)
    targets = torch.randint(0, 10, (batch_size,), device=device)
    
    print(f"   Input shapes: color={color_input.shape}, brightness={brightness_input.shape}")
    
    # Store initial weights for comparison
    initial_weights = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            initial_weights[name] = param.data.clone()
    
    # Forward pass
    outputs = model.forward_combined(color_input, brightness_input)
    print(f"   Output shape: {outputs.shape}")
    
    # Compute loss
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, targets)
    print(f"   Loss: {loss.item():.6f}")
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    grad_info = {}
    zero_grad_count = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            total_params += 1
            if param.grad is not None:
                grad_norm = param.grad.data.norm().item()
                grad_info[name] = grad_norm
                if grad_norm == 0:
                    zero_grad_count += 1
            else:
                print(f"   ❌ No gradient for parameter: {name}")
                zero_grad_count += 1
    
    print(f"\n📊 Gradient Analysis:")
    print(f"   Total parameters: {total_params}")
    print(f"   Parameters with zero gradients: {zero_grad_count}")
    print(f"   Parameters with non-zero gradients: {total_params - zero_grad_count}")
    
    # Show gradient norms for key layers
    key_layers = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'classifier']
    print(f"\n🔍 Key Layer Gradient Norms:")
    
    for layer_name in key_layers:
        matching_params = [name for name in grad_info.keys() if layer_name in name]
        if matching_params:
            avg_grad = np.mean([grad_info[name] for name in matching_params])
            print(f"   {layer_name}: {avg_grad:.6f} (avg across {len(matching_params)} params)")
    
    # Test weight updates (simulate optimizer step)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.step()
    
    # Check if weights actually changed
    weight_changes = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            if not torch.equal(initial_weights[name], param.data):
                weight_changes += 1
    
    print(f"\n⚡ Weight Update Analysis:")
    print(f"   Parameters that changed: {weight_changes}/{total_params}")
    
    # Test input gradients (should flow back to inputs)
    if color_input.grad is not None:
        color_grad_norm = color_input.grad.norm().item()
        print(f"   Color input gradient norm: {color_grad_norm:.6f}")
    else:
        print(f"   ❌ No gradient for color input")
    
    if brightness_input.grad is not None:
        brightness_grad_norm = brightness_input.grad.norm().item()
        print(f"   Brightness input gradient norm: {brightness_grad_norm:.6f}")
    else:
        print(f"   ❌ No gradient for brightness input")
    
    # Test data flow integrity
    print(f"\n🔄 Data Flow Test:")
    model.eval()
    with torch.no_grad():
        # Test that different inputs produce different outputs
        outputs1 = model.forward_combined(color_input, brightness_input)
        outputs2 = model.forward_combined(color_input * 0.5, brightness_input * 0.5)
        
        output_diff = (outputs1 - outputs2).norm().item()
        print(f"   Output difference for different inputs: {output_diff:.6f}")
        
        if output_diff > 1e-6:
            print(f"   ✅ Model responds to input changes")
        else:
            print(f"   ❌ Model output doesn't change with different inputs")
    
    # Test channel-specific processing
    print(f"\n🎨 Channel-Specific Processing Test:")
    with torch.no_grad():
        # Test color-only
        color_only = model.forward_combined(color_input, torch.zeros_like(brightness_input))
        # Test brightness-only  
        brightness_only = model.forward_combined(torch.zeros_like(color_input), brightness_input)
        # Test both
        both_channels = model.forward_combined(color_input, brightness_input)
        
        color_norm = color_only.norm().item()
        brightness_norm = brightness_only.norm().item()
        both_norm = both_channels.norm().item()
        
        print(f"   Color-only output norm: {color_norm:.6f}")
        print(f"   Brightness-only output norm: {brightness_norm:.6f}")
        print(f"   Both channels output norm: {both_norm:.6f}")
        
        if color_norm > 1e-6 and brightness_norm > 1e-6:
            print(f"   ✅ Both channels contribute to output")
        else:
            print(f"   ❌ Channel isolation issue detected")
    
    # Overall assessment
    success = (
        zero_grad_count == 0 and 
        weight_changes > 0 and 
        output_diff > 1e-6 and
        color_norm > 1e-6 and 
        brightness_norm > 1e-6
    )
    
    print(f"\n{'✅ GRADIENT FLOW TEST PASSED' if success else '❌ GRADIENT FLOW TEST FAILED'}")
    
    return success

if __name__ == "__main__":
    test_gradient_flow()
