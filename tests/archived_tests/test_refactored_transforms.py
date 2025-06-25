#!/usr/bin/env python3
"""
Simple test to demonstrate the canonical RGBtoRGBL transform:

1. RGBtoRGBL is a simple preprocessing class (not nn.Module)
2. AdaptiveBrightnessExtraction is a learnable nn.Module for training
3. Shows the clean separation between preprocessing and learnable components
"""

import sys
import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, '/Users/gclinger/Documents/projects/Multi-Stream-Neural-Networks/src')

from src.transforms.rgb_to_rgbl import RGBtoRGBL, AdaptiveBrightnessExtraction

def demonstrate_canonical_design():
    """Demonstrate the canonical RGBtoRGBL transform design."""
    
    print("=== Demonstrating Canonical Transform Design ===\n")
    
    # Create some sample RGB data
    batch_size = 4
    channels = 3
    height, width = 32, 32
    
    # Sample RGB batch
    rgb_batch = torch.randn(batch_size, channels, height, width)
    print(f"Input RGB batch shape: {rgb_batch.shape}")
    
    # =============================================================================
    # 1. RGBtoRGBL - Canonical preprocessing class
    # =============================================================================
    
    print("\n1. RGBtoRGBL Transform (Canonical Preprocessing)")
    print("   - Simple preprocessing class (not nn.Module)")
    print("   - Uses __call__ method for direct stream output")
    print("   - No learnable parameters")  
    print("   - Used for data preprocessing only")
    
    rgbl_transform = RGBtoRGBL()
    
    print(f"   - Transform type: {type(rgbl_transform)}")
    print(f"   - Has __call__: {hasattr(rgbl_transform, '__call__')}")
    print(f"   - Has forward: {hasattr(rgbl_transform, 'forward')}")
    print(f"   - Is nn.Module: {isinstance(rgbl_transform, nn.Module)}")
    
    # Process each image in batch - returns separate streams directly
    rgb_parts = []
    brightness_parts = []
    
    for i in range(batch_size):
        rgb_stream, brightness_stream = rgbl_transform(rgb_batch[i])
        rgb_parts.append(rgb_stream)
        brightness_parts.append(brightness_stream)
    
    rgb_batch_streams = torch.stack(rgb_parts)
    brightness_batch_streams = torch.stack(brightness_parts)
    
    print(f"   - RGB stream shape: {rgb_batch_streams.shape}")
    print(f"   - Brightness stream shape: {brightness_batch_streams.shape}")
    
    # Alternative: get 4-channel RGBL if needed
    rgbl_combined = rgbl_transform.get_rgbl(rgb_batch[0])
    print(f"   - Alternative RGBL output shape: {rgbl_combined.shape}")
    
    # =============================================================================
    # 2. AdaptiveBrightnessExtraction - Learnable component
    # =============================================================================
    
    print("\n2. AdaptiveBrightnessExtraction (Learnable Model Component)")
    print("   - Inherits from nn.Module")
    print("   - Has learnable parameters for adaptive brightness")
    print("   - Used as part of model, not preprocessing")
    
    adaptive_brightness = AdaptiveBrightnessExtraction()
    
    print(f"   - Transform type: {type(adaptive_brightness)}")
    print(f"   - Is nn.Module: {isinstance(adaptive_brightness, nn.Module)}")
    print(f"   - Has parameters: {len(list(adaptive_brightness.parameters())) > 0}")
    print(f"   - Parameter shape: {list(adaptive_brightness.parameters())[0].shape}")
    
    # This would be used inside a model's forward pass
    with torch.no_grad():
        adaptive_brightness_output = adaptive_brightness(rgb_batch)
    print(f"   - Output shape: {adaptive_brightness_output.shape}")
    
    # =============================================================================
    # 3. Why this canonical design makes sense
    # =============================================================================
    
    print("\n=== Why This Canonical Design Makes Sense ===")
    
    print("\n✅ PREPROCESSING (RGBtoRGBL):")
    print("   - Single class for all RGB-to-brightness preprocessing")
    print("   - Applied during data loading")
    print("   - No gradients needed")
    print("   - Fixed mathematical transformations")
    print("   - Returns streams directly (no intermediate tensors)")
    print("   - Clean, intuitive API")
    
    print("\n✅ MODEL COMPONENTS (AdaptiveBrightnessExtraction):")
    print("   - Part of model's forward pass")
    print("   - Have learnable parameters")
    print("   - Need gradients for backpropagation")
    print("   - Inherit from nn.Module for proper parameter handling")
    
    # =============================================================================
    # 4. Demonstrate gradient behavior
    # =============================================================================
    
    print("\n=== Gradient Behavior Demonstration ===")
    
    # Test with gradients enabled
    rgb_test = torch.randn(3, 32, 32, requires_grad=True)
    
    print(f"\nInput requires_grad: {rgb_test.requires_grad}")
    
    # Preprocessing transform preserves gradients
    rgb_stream, brightness_stream = rgbl_transform(rgb_test)
    print(f"RGBtoRGBL RGB output requires_grad: {rgb_stream.requires_grad}")
    print(f"RGBtoRGBL brightness output requires_grad: {brightness_stream.requires_grad}")
    
    # Model component should preserve gradients
    adaptive_result = adaptive_brightness(rgb_test)
    print(f"AdaptiveBrightnessExtraction output requires_grad: {adaptive_result.requires_grad}")
    
    # Test gradient flow through adaptive component
    loss = adaptive_result.sum()
    loss.backward()
    
    print(f"✅ Gradients successfully computed for learnable component")
    print(f"   Input grad shape: {rgb_test.grad.shape if rgb_test.grad is not None else 'None'}")
    print(f"   Parameter grad: {adaptive_brightness.rgb_weights.grad is not None}")
    
    # =============================================================================
    # 5. API Comparison
    # =============================================================================
    
    print("\n=== API Demonstration ===")
    
    print("\n📋 MAIN USAGE:")
    print("   transform = RGBtoRGBL()")
    print("   rgb_stream, brightness_stream = transform(rgb_image)")
    print("   → Direct stream output for multi-stream models")
    
    print("\n📋 ALTERNATIVE USAGE:")
    print("   transform = RGBtoRGBL()")
    print("   rgbl_tensor = transform.get_rgbl(rgb_image)")
    print("   → 4-channel tensor if needed")
    
    print("\n=== Summary ===")
    print("✅ RGBtoRGBL is the single canonical transform for RGB-to-brightness")
    print("✅ Clean separation between preprocessing and learnable components")
    print("✅ Efficient direct stream output")
    print("✅ No deprecated code or backward compatibility complexity")
    print("✅ Intuitive, maintainable design")
    
    return True

if __name__ == "__main__":
    success = demonstrate_canonical_design()
    if success:
        print(f"\n🎉 Canonical design demonstration completed successfully!")
    else:
        print(f"\n❌ Something went wrong")
