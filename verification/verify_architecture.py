#!/usr/bin/env python3
"""
Verify the actual BaseMultiChannelNetwork architecture and parameter counts.
"""

import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.basic_multi_channel.base_multi_channel_network import BaseMultiChannelNetwork


def analyze_base_multi_channel():
    """Analyze the actual BaseMultiChannelNetwork implementation."""
    
    # Create a model with the configuration mentioned in comparisons.md
    model = BaseMultiChannelNetwork(
        color_input_size=3072,      # 28x28x3 = 2352, but let's use what's in comparisons.md
        brightness_input_size=1024, # 28x28x1 = 784, but let's use what's in comparisons.md
        hidden_sizes=[512, 256],    # Two hidden layers as described
        num_classes=10
    )
    
    print("=== BaseMultiChannelNetwork Architecture Analysis ===\n")
    
    # Print model structure
    print("Model Structure:")
    for name, module in model.named_modules():
        if name:  # Skip root module
            print(f"  {name}: {module}")
    
    print("\n" + "="*60 + "\n")
    
    # Analyze each layer's parameters
    print("Parameter Breakdown:")
    total_params = 0
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        print(f"  {name}: {param.shape} = {num_params:,} params")
    
    print(f"\nTotal Parameters: {total_params:,}")
    
    print("\n" + "="*60 + "\n")
    
    # Test forward pass to understand the architecture
    print("Forward Pass Analysis:")
    
    # Create dummy inputs
    batch_size = 2
    color_input = torch.randn(batch_size, 3072)
    brightness_input = torch.randn(batch_size, 1024)
    
    print(f"Input shapes:")
    print(f"  Color: {color_input.shape}")
    print(f"  Brightness: {brightness_input.shape}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(color_input, brightness_input)
    
    print(f"\nOutput:")
    if isinstance(outputs, tuple):
        print(f"  Type: Tuple of {len(outputs)} tensors")
        for i, output in enumerate(outputs):
            print(f"  Output {i}: {output.shape}")
    else:
        print(f"  Type: Single tensor")
        print(f"  Shape: {outputs.shape}")
    
    # Check if there's a forward_combined method
    if hasattr(model, 'forward_combined'):
        print(f"\nTesting forward_combined method:")
        with torch.no_grad():
            combined_output = model.forward_combined(color_input, brightness_input)
        print(f"  Combined output shape: {combined_output.shape}")
    
    print("\n" + "="*60 + "\n")
    
    # Detailed layer analysis
    print("Detailed Layer Analysis:")
    
    layer_idx = 0
    for layer in model.layers:
        if hasattr(layer, 'color_weights'):
            print(f"\nLayer {layer_idx} (BasicMultiChannelLayer):")
            print(f"  Color weights: {layer.color_weights.shape}")
            print(f"  Brightness weights: {layer.brightness_weights.shape}")
            print(f"  Color params: {layer.color_weights.numel():,}")
            print(f"  Brightness params: {layer.brightness_weights.numel():,}")
            if layer.color_bias is not None:
                print(f"  Bias params: {layer.color_bias.numel() + layer.brightness_bias.numel():,}")
            layer_idx += 1
    
    # Classifier analysis
    print(f"\nClassifier Analysis:")
    classifier = model.classifier
    if hasattr(classifier, 'color_weights'):
        print(f"  Type: BasicMultiChannelLayer (separate outputs)")
        print(f"  Color weights: {classifier.color_weights.shape}")
        print(f"  Brightness weights: {classifier.brightness_weights.shape}")
        print(f"  Color params: {classifier.color_weights.numel():,}")
        print(f"  Brightness params: {classifier.brightness_weights.numel():,}")
        if classifier.color_bias is not None:
            print(f"  Bias params: {classifier.color_bias.numel() + classifier.brightness_bias.numel():,}")
    else:
        print(f"  Type: Standard Linear layer (shared)")
        print(f"  Weight shape: {classifier.weight.shape}")
        print(f"  Parameters: {classifier.weight.numel():,}")
    
    return model, total_params


def compare_to_separate_models():
    """Compare to equivalent separate models."""
    
    print("\n" + "="*60)
    print("COMPARISON TO SEPARATE MODELS")
    print("="*60 + "\n")
    
    # Create equivalent separate models
    color_model = torch.nn.Sequential(
        torch.nn.Linear(3072, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 10)
    )
    
    brightness_model = torch.nn.Sequential(
        torch.nn.Linear(1024, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 10)
    )
    
    color_params = sum(p.numel() for p in color_model.parameters())
    brightness_params = sum(p.numel() for p in brightness_model.parameters())
    total_separate = color_params + brightness_params
    
    print("Separate Models Analysis:")
    print(f"  Color model parameters: {color_params:,}")
    print(f"  Brightness model parameters: {brightness_params:,}")
    print(f"  Total separate models: {total_separate:,}")
    
    # Compare to multi-channel
    _, multi_channel_params = analyze_base_multi_channel()
    
    print(f"\nComparison:")
    print(f"  Multi-channel: {multi_channel_params:,}")
    print(f"  Separate models: {total_separate:,}")
    print(f"  Difference: {multi_channel_params - total_separate:,}")
    print(f"  Percentage difference: {((multi_channel_params - total_separate) / total_separate) * 100:.6f}%")


if __name__ == "__main__":
    analyze_base_multi_channel()
    compare_to_separate_models()
