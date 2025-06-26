#!/usr/bin/env python3
"""
Demonstration of the two fusion strategies in BaseMultiChannelNetwork.
Shows the difference between shared classifier and separate classifiers.
"""

import sys
import os
sys.path.append('/Users/gclinger/Documents/projects/Multi-Stream-Neural-Networks')

import torch
import numpy as np
from src.models.basic_multi_channel.base_multi_channel_network import BaseMultiChannelNetwork

def demonstrate_fusion_strategies():
    """Demonstrate the difference between shared and separate classifier fusion."""
    print("🔍 Demonstrating Fusion Strategy Differences")
    print("="*60)
    
    # Create test data
    batch_size = 3
    color_features = torch.randn(batch_size, 64)
    brightness_features = torch.randn(batch_size, 32)
    
    print(f"Input Features:")
    print(f"  Color: {color_features.shape}")
    print(f"  Brightness: {brightness_features.shape}")
    
    print("\n" + "="*60)
    print("STRATEGY 1: SHARED CLASSIFIER (Recommended)")
    print("="*60)
    
    # Shared classifier model
    shared_model = BaseMultiChannelNetwork(
        color_input_size=64,
        brightness_input_size=32,
        hidden_sizes=[16],  # Small for demo
        num_classes=5,
        use_shared_classifier=True,  # Feature-level fusion
        device='cpu'
    )
    
    print("\nArchitecture:")
    print("  Color [64] ─┐")
    print("              ├─ Concat [96] ─ Shared Classifier ─ Predictions [5]")
    print("  Brightness [32] ─┘")
    
    # Extract features and show fusion
    color_feat, brightness_feat = shared_model._extract_features(color_features, brightness_features)
    print(f"\nAfter feature extraction:")
    print(f"  Color features: {color_feat.shape}")
    print(f"  Brightness features: {brightness_feat.shape}")
    
    # Show concatenation
    fused_features = torch.cat([color_feat, brightness_feat], dim=1)
    print(f"  Concatenated features: {fused_features.shape}")
    
    # Final output
    output_shared = shared_model(color_features, brightness_features)
    print(f"  Final output: {output_shared.shape}")
    
    # Show classifier parameters
    shared_classifier_params = shared_model.shared_classifier.weight.shape
    print(f"  Shared classifier weights: {shared_classifier_params}")
    print(f"  Can learn interactions: ✅ YES")
    
    print("\n" + "="*60)
    print("STRATEGY 2: SEPARATE CLASSIFIERS (Legacy)")
    print("="*60)
    
    # Separate classifiers model
    separate_model = BaseMultiChannelNetwork(
        color_input_size=64,
        brightness_input_size=32,
        hidden_sizes=[16],  # Small for demo
        num_classes=5,
        use_shared_classifier=False,  # Output-level fusion
        device='cpu'
    )
    
    print("\nArchitecture:")
    print("  Color [64] ─ Color Classifier ─ Color Logits [5] ─┐")
    print("                                                    ├─ Add ─ Predictions [5]")
    print("  Brightness [32] ─ Brightness Classifier ─ Brightness Logits [5] ─┘")
    
    # Extract features (same as before)
    color_feat2, brightness_feat2 = separate_model._extract_features(color_features, brightness_features)
    print(f"\nAfter feature extraction:")
    print(f"  Color features: {color_feat2.shape}")
    print(f"  Brightness features: {brightness_feat2.shape}")
    
    # Show separate classification
    color_logits, brightness_logits = separate_model.analyze_pathways(color_features, brightness_features)
    print(f"  Color logits: {color_logits.shape}")
    print(f"  Brightness logits: {brightness_logits.shape}")
    
    # Final output (addition)
    output_separate = separate_model(color_features, brightness_features)
    print(f"  Final output (sum): {output_separate.shape}")
    
    # Show classifier parameters
    color_classifier_params = separate_model.multi_channel_classifier.color_weights.shape
    brightness_classifier_params = separate_model.multi_channel_classifier.brightness_weights.shape
    print(f"  Color classifier weights: {color_classifier_params}")
    print(f"  Brightness classifier weights: {brightness_classifier_params}")
    print(f"  Can learn interactions: ❌ NO (independent)")
    
    print("\n" + "="*60)
    print("COMPARISON ANALYSIS")
    print("="*60)
    
    # Parameter count comparison
    shared_params = sum(p.numel() for p in shared_model.shared_classifier.parameters())
    separate_params = sum(p.numel() for p in separate_model.multi_channel_classifier.parameters())
    
    print(f"\nParameter Count:")
    print(f"  Shared classifier: {shared_params} parameters")
    print(f"  Separate classifiers: {separate_params} parameters")
    print(f"  Difference: {shared_params - separate_params} ({((shared_params/separate_params - 1)*100):+.1f}%)")
    
    # Show the actual difference in computation
    print(f"\nComputational Difference:")
    print(f"  Shared: Can learn f(color, brightness) - full cross-modal interaction")
    print(f"  Separate: Only f1(color) + f2(brightness) - no interaction")
    
    # Demonstrate with a concrete example
    print(f"\nConcrete Example:")
    sample_idx = 0
    color_sample = color_feat[sample_idx:sample_idx+1]
    brightness_sample = brightness_feat[sample_idx:sample_idx+1]
    
    # Shared classifier sees both
    concat_sample = torch.cat([color_sample, brightness_sample], dim=1)
    shared_output_sample = shared_model.shared_classifier(concat_sample)
    
    # Separate classifiers see only their own stream
    color_only_output = shared_model.color_head(color_sample)
    brightness_only_output = shared_model.brightness_head(brightness_sample)
    
    print(f"  For sample {sample_idx}:")
    print(f"    Shared classifier output: {shared_output_sample[0][:3].detach().numpy()}")
    print(f"    Color-only output: {color_only_output[0][:3].detach().numpy()}")
    print(f"    Brightness-only output: {brightness_only_output[0][:3].detach().numpy()}")
    print(f"    Sum of separate: {(color_only_output[0] + brightness_only_output[0])[:3].detach().numpy()}")
    
    print(f"\n📊 Which is better?")
    print(f"  🏆 Shared Classifier (Recommended):")
    print(f"    ✅ Can learn feature interactions")
    print(f"    ✅ More expressive (higher capacity)")
    print(f"    ✅ Better fusion for complex relationships")
    print(f"    ❌ More parameters (higher risk of overfitting)")
    
    print(f"\n  📋 Separate Classifiers (Legacy):")
    print(f"    ✅ Simpler architecture")
    print(f"    ✅ Fewer parameters (less overfitting risk)")
    print(f"    ✅ Independent pathway analysis")
    print(f"    ❌ Cannot learn cross-modal interactions")
    print(f"    ❌ Limited fusion capability")
    
    print(f"\n🎯 Recommendation: Use shared_classifier=True for better performance!")

if __name__ == "__main__":
    demonstrate_fusion_strategies()
