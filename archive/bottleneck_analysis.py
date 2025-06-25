"""
ResNet Bottleneck Analysis - Check for performance bottlenecks and inefficiencies.
"""

import os
import sys
import torch
import torch.nn as nn
import time
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.builders.model_factory import create_model

def analyze_model_bottlenecks():
    """Comprehensive analysis of ResNet model for bottlenecks."""
    print("🔍 ResNet Bottleneck Analysis")
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Create models for comparison
    resnet_model = create_model(
        'multi_channel_resnet18',
        num_classes=10,
        color_input_channels=3,
        brightness_input_channels=1,
        activation='relu'
    ).to(device)
    
    dense_model = create_model(
        'base_multi_channel',
        color_input_size=28*28*3,
        brightness_input_size=28*28,
        hidden_sizes=[128, 64, 32],
        num_classes=10,
        activation='relu',
        dropout_rate=0.2
    ).to(device)
    
    # Create test data
    batch_size = 32
    color_input = torch.randn(batch_size, 3, 28, 28, device=device)
    brightness_input = torch.randn(batch_size, 1, 28, 28, device=device)
    
    # For dense model, flatten inputs
    color_flat = color_input.view(batch_size, -1)
    brightness_flat = brightness_input.view(batch_size, -1)
    
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    
    # 1. Parameter Analysis
    print(f"\n📊 Parameter Analysis:")
    resnet_params = sum(p.numel() for p in resnet_model.parameters())
    dense_params = sum(p.numel() for p in dense_model.parameters())
    
    print(f"   ResNet parameters: {resnet_params:,}")
    print(f"   Dense parameters: {dense_params:,}")
    print(f"   Parameter ratio (ResNet/Dense): {resnet_params/dense_params:.1f}x")
    
    # 2. Memory Usage Analysis
    print(f"\n💾 Memory Usage Analysis:")
    
    # Forward pass memory
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    with torch.no_grad():
        # ResNet memory
        resnet_outputs = resnet_model.forward_combined(color_input, brightness_input)
        resnet_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else "N/A (CPU)"
        
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Dense memory
        dense_outputs = dense_model(color_flat, brightness_flat)
        dense_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else "N/A (CPU)"
    
    print(f"   ResNet memory usage: {resnet_memory}")
    print(f"   Dense memory usage: {dense_memory}")
    
    # 3. Forward Pass Timing
    print(f"\n⏱️ Forward Pass Timing (100 iterations):")
    
    def time_forward_pass(model, color_in, brightness_in, is_dense=False):
        model.eval()
        times = []
        
        with torch.no_grad():
            # Warmup
            for _ in range(10):
                if is_dense:
                    _ = model(color_in, brightness_in)
                else:
                    _ = model.forward_combined(color_in, brightness_in)
            
            # Timing
            for _ in range(100):
                start_time = time.time()
                if is_dense:
                    _ = model(color_in, brightness_in)
                else:
                    _ = model.forward_combined(color_in, brightness_in)
                times.append(time.time() - start_time)
        
        return np.mean(times), np.std(times)
    
    resnet_time, resnet_std = time_forward_pass(resnet_model, color_input, brightness_input, is_dense=False)
    dense_time, dense_std = time_forward_pass(dense_model, color_flat, brightness_flat, is_dense=True)
    
    print(f"   ResNet forward pass: {resnet_time*1000:.2f} ± {resnet_std*1000:.2f} ms")
    print(f"   Dense forward pass: {dense_time*1000:.2f} ± {dense_std*1000:.2f} ms")
    print(f"   Speed ratio (ResNet/Dense): {resnet_time/dense_time:.1f}x")
    
    # 4. Backward Pass Timing
    print(f"\n⏱️ Backward Pass Timing (50 iterations):")
    
    def time_backward_pass(model, color_in, brightness_in, is_dense=False):
        model.train()
        criterion = nn.CrossEntropyLoss()
        targets = torch.randint(0, 10, (batch_size,), device=device)
        times = []
        
        for _ in range(50):
            model.zero_grad()
            
            start_time = time.time()
            if is_dense:
                outputs = model(color_in, brightness_in)
            else:
                outputs = model.forward_combined(color_in, brightness_in)
            
            loss = criterion(outputs, targets)
            loss.backward()
            times.append(time.time() - start_time)
        
        return np.mean(times), np.std(times)
    
    resnet_back_time, resnet_back_std = time_backward_pass(resnet_model, color_input, brightness_input, is_dense=False)
    dense_back_time, dense_back_std = time_backward_pass(dense_model, color_flat, brightness_flat, is_dense=True)
    
    print(f"   ResNet backward pass: {resnet_back_time*1000:.2f} ± {resnet_back_std*1000:.2f} ms")
    print(f"   Dense backward pass: {dense_back_time*1000:.2f} ± {dense_back_std*1000:.2f} ms")
    print(f"   Speed ratio (ResNet/Dense): {resnet_back_time/dense_back_time:.1f}x")
    
    # 5. Layer-by-Layer Analysis
    print(f"\n🏗️ ResNet Layer-by-Layer Analysis:")
    
    def analyze_layer_outputs(model, color_in, brightness_in):
        model.eval()
        layer_stats = {}
        
        with torch.no_grad():
            # Initial layers
            color_x, brightness_x = model.conv1(color_in, brightness_in)
            layer_stats['conv1'] = {
                'color_shape': color_x.shape,
                'brightness_shape': brightness_x.shape,
                'color_norm': color_x.norm().item(),
                'brightness_norm': brightness_x.norm().item()
            }
            
            color_x, brightness_x = model.bn1(color_x, brightness_x)
            color_x, brightness_x = model.activation_initial(color_x, brightness_x)
            color_x = model.maxpool(color_x)
            brightness_x = model.maxpool(brightness_x)
            
            layer_stats['after_maxpool'] = {
                'color_shape': color_x.shape,
                'brightness_shape': brightness_x.shape,
                'color_norm': color_x.norm().item(),
                'brightness_norm': brightness_x.norm().item()
            }
            
            # ResNet layers
            for i, layer in enumerate([model.layer1, model.layer2, model.layer3, model.layer4], 1):
                color_x, brightness_x = layer(color_x, brightness_x)
                layer_stats[f'layer{i}'] = {
                    'color_shape': color_x.shape,
                    'brightness_shape': brightness_x.shape,
                    'color_norm': color_x.norm().item(),
                    'brightness_norm': brightness_x.norm().item()
                }
            
            # Final pooling
            color_x, brightness_x = model.avgpool(color_x, brightness_x)
            layer_stats['after_avgpool'] = {
                'color_shape': color_x.shape,
                'brightness_shape': brightness_x.shape,
                'color_norm': color_x.norm().item(),
                'brightness_norm': brightness_x.norm().item()
            }
        
        return layer_stats
    
    layer_stats = analyze_layer_outputs(resnet_model, color_input, brightness_input)
    
    for layer_name, stats in layer_stats.items():
        print(f"   {layer_name}:")
        print(f"     Color: {stats['color_shape']} (norm: {stats['color_norm']:.3f})")
        print(f"     Brightness: {stats['brightness_shape']} (norm: {stats['brightness_norm']:.3f})")
    
    # 6. Channel Efficiency Analysis
    print(f"\n🎨 Channel Efficiency Analysis:")
    
    # Test with different input configurations
    with torch.no_grad():
        # Full inputs
        full_output = resnet_model.forward_combined(color_input, brightness_input)
        
        # Color only
        color_only_output = resnet_model.forward_combined(color_input, torch.zeros_like(brightness_input))
        
        # Brightness only
        brightness_only_output = resnet_model.forward_combined(torch.zeros_like(color_input), brightness_input)
        
        # Calculate contributions
        full_norm = full_output.norm().item()
        color_contrib = color_only_output.norm().item() / full_norm
        brightness_contrib = brightness_only_output.norm().item() / full_norm
        
        print(f"   Color contribution: {color_contrib:.3f} ({color_contrib*100:.1f}%)")
        print(f"   Brightness contribution: {brightness_contrib:.3f} ({brightness_contrib*100:.1f}%)")
        print(f"   Channel utilization balance: {color_contrib/brightness_contrib:.2f} (color/brightness)")
    
    # 7. Activation Statistics
    print(f"\n📈 Activation Statistics:")
    
    with torch.no_grad():
        # Get intermediate activations
        resnet_model.eval()
        color_x, brightness_x = resnet_model.conv1(color_input, brightness_input)
        
        # Check for dead neurons
        color_dead = (color_x <= 0).float().mean().item()
        brightness_dead = (brightness_x <= 0).float().mean().item()
        
        print(f"   Dead neurons after conv1:")
        print(f"     Color stream: {color_dead*100:.1f}%")
        print(f"     Brightness stream: {brightness_dead*100:.1f}%")
        
        # Check activation ranges
        print(f"   Activation ranges after conv1:")
        print(f"     Color: [{color_x.min().item():.3f}, {color_x.max().item():.3f}]")
        print(f"     Brightness: [{brightness_x.min().item():.3f}, {brightness_x.max().item():.3f}]")
    
    # 8. Final Assessment
    print(f"\n🎯 Bottleneck Assessment:")
    
    bottlenecks = []
    
    if resnet_params / dense_params > 50:
        bottlenecks.append(f"Parameter count too high ({resnet_params/dense_params:.1f}x dense model)")
    
    if resnet_time / dense_time > 10:
        bottlenecks.append(f"Forward pass too slow ({resnet_time/dense_time:.1f}x dense model)")
    
    if resnet_back_time / dense_back_time > 10:
        bottlenecks.append(f"Backward pass too slow ({resnet_back_time/dense_back_time:.1f}x dense model)")
    
    if color_dead > 0.5 or brightness_dead > 0.5:
        bottlenecks.append("Too many dead neurons detected")
    
    if abs(color_contrib - brightness_contrib) > 0.8:
        bottlenecks.append("Severe channel imbalance detected")
    
    if bottlenecks:
        print("   ❌ Bottlenecks found:")
        for bottleneck in bottlenecks:
            print(f"     - {bottleneck}")
    else:
        print("   ✅ No significant bottlenecks detected")
        print("   ✅ Model architecture is sound")
        print("   ✅ Performance differences likely due to training dynamics")
    
    return len(bottlenecks) == 0

if __name__ == "__main__":
    analyze_model_bottlenecks()
