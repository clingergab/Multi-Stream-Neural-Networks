#!/usr/bin/env python3
"""
Test and compare the shared classifier vs separate classifier architectures.
"""

import torch
import sys
import os

# Add project root to path
sys.path.append('/Users/gclinger/Documents/projects/Multi-Stream-Neural-Networks')

from models.basic_multi_channel.base_multi_channel_network import BaseMultiChannelNetwork

def count_parameters(model):
    """Count total parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def analyze_model_architecture(model, model_name):
    """Analyze model architecture and parameter distribution."""
    print(f"\n{'='*60}")
    print(f"ANALYSIS: {model_name}")
    print(f"{'='*60}")
    
    # Basic info
    total_params = count_parameters(model)
    fusion_type = model.fusion_type
    classifier_info = model.get_classifier_info()
    
    print(f"Fusion Type: {fusion_type}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Classifier Info: {classifier_info}")
    
    # Layer-by-layer breakdown
    print(f"\nLayer-by-layer Parameter Breakdown:")
    print(f"{'Layer':<20} {'Parameters':<15} {'Shape'}")
    print(f"{'-'*50}")
    
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            params = module.weight.numel()
            if hasattr(module, 'bias') and module.bias is not None:
                params += module.bias.numel()
            print(f"{name:<20} {params:<15,} {str(module.weight.shape)}")
    
    # Test forward pass
    print(f"\nTesting Forward Pass:")
    color_input = torch.randn(4, model.color_input_size)
    brightness_input = torch.randn(4, model.brightness_input_size)
    
    # Test regular forward
    output = model.forward(color_input, brightness_input)
    if isinstance(output, tuple):
        print(f"Forward output: Tuple of shapes {output[0].shape}, {output[1].shape}")
    else:
        print(f"Forward output: Single tensor of shape {output.shape}")
    
    # Test combined forward
    combined_output = model.forward_combined(color_input, brightness_input)
    print(f"Combined output shape: {combined_output.shape}")
    
    # Test feature extraction
    features = model.extract_features(color_input, brightness_input)
    if isinstance(features, tuple):
        print(f"Features: Tuple of shapes {features[0].shape}, {features[1].shape}")
    else:
        print(f"Features: Single tensor of shape {features.shape}")
    
    return total_params

def compare_architectures():
    """Compare shared vs separate classifier architectures."""
    print("COMPARING MULTI-CHANNEL NETWORK ARCHITECTURES")
    print("=" * 80)
    
    # Test configuration
    color_input_size = 3072  # RGB flattened 32x32x3
    brightness_input_size = 1024  # Brightness flattened 32x32x1
    hidden_sizes = [512, 256]
    num_classes = 10
    
    # Create models
    print(f"\nConfiguration:")
    print(f"Color input size: {color_input_size}")
    print(f"Brightness input size: {brightness_input_size}")
    print(f"Hidden sizes: {hidden_sizes}")
    print(f"Output classes: {num_classes}")
    
    # Shared classifier model (recommended)
    shared_model = BaseMultiChannelNetwork(
        color_input_size=color_input_size,
        brightness_input_size=brightness_input_size,
        hidden_sizes=hidden_sizes,
        num_classes=num_classes,
        use_shared_classifier=True
    )
    
    # Separate classifier model (legacy)
    separate_model = BaseMultiChannelNetwork(
        color_input_size=color_input_size,
        brightness_input_size=brightness_input_size,
        hidden_sizes=hidden_sizes,
        num_classes=num_classes,
        use_shared_classifier=False
    )
    
    # Analyze both
    shared_params = analyze_model_architecture(shared_model, "SHARED CLASSIFIER MODEL")
    separate_params = analyze_model_architecture(separate_model, "SEPARATE CLASSIFIERS MODEL")
    
    # Calculate equivalent separate models for comparison
    print(f"\n{'='*60}")
    print(f"EQUIVALENT SEPARATE MODELS COMPARISON")
    print(f"{'='*60}")
    
    # Color-only model parameters
    color_layer1 = color_input_size * hidden_sizes[0] + hidden_sizes[0]  # weights + bias
    color_layer2 = hidden_sizes[0] * hidden_sizes[1] + hidden_sizes[1]
    color_classifier = hidden_sizes[1] * num_classes + num_classes
    color_total = color_layer1 + color_layer2 + color_classifier
    
    # Brightness-only model parameters
    brightness_layer1 = brightness_input_size * hidden_sizes[0] + hidden_sizes[0]
    brightness_layer2 = hidden_sizes[0] * hidden_sizes[1] + hidden_sizes[1]  
    brightness_classifier = hidden_sizes[1] * num_classes + num_classes
    brightness_total = brightness_layer1 + brightness_layer2 + brightness_classifier
    
    separate_models_total = color_total + brightness_total
    
    print(f"Color-only model parameters: {color_total:,}")
    print(f"Brightness-only model parameters: {brightness_total:,}")
    print(f"Total separate models: {separate_models_total:,}")
    
    # Summary comparison
    print(f"\n{'='*60}")
    print(f"PARAMETER COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Architecture':<25} {'Parameters':<15} {'Difference'}")
    print(f"{'-'*55}")
    print(f"{'Shared Classifier':<25} {shared_params:<15,} {'Baseline'}")
    print(f"{'Separate Classifiers':<25} {separate_params:<15,} {separate_params - shared_params:+,}")
    print(f"{'Equivalent Separate':<25} {separate_models_total:<15,} {separate_models_total - shared_params:+,}")
    
    # Parameter efficiency analysis
    shared_vs_separate = ((separate_params - shared_params) / shared_params) * 100
    shared_vs_equiv = ((separate_models_total - shared_params) / shared_params) * 100
    
    print(f"\nParameter Efficiency:")
    print(f"Shared vs Separate Classifiers: {shared_vs_separate:+.1f}%")
    print(f"Shared vs Equivalent Separate Models: {shared_vs_equiv:+.1f}%")
    
    # Calculate classifier parameter differences
    print(f"\nClassifier Parameter Analysis:")
    shared_classifier_params = hidden_sizes[1] * 2 * num_classes + num_classes  # concatenated inputs
    separate_classifier_params = 2 * (hidden_sizes[1] * num_classes + num_classes)  # two classifiers
    
    print(f"Shared classifier parameters: {shared_classifier_params:,}")
    print(f"Separate classifiers parameters: {separate_classifier_params:,}")
    print(f"Parameter reduction with shared: {separate_classifier_params - shared_classifier_params:,}")

if __name__ == "__main__":
    compare_architectures()
