"""
Diagnostic script to analyze multi-channel neural network training issues.

This script runs a comprehensive diagnosis of model learning problems,
focusing particularly on the MultiChannelResNetNetwork architecture.
It checks for issues in gradient flow, dead neurons, parameter updates,
and various other factors that could impede learning.
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

# Add project root to path to ensure imports work
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import after sys.path is updated
from src.models.basic_multi_channel.multi_channel_resnet_network import MultiChannelResNetNetwork
from src.models.basic_multi_channel.base_multi_channel_network import BaseMultiChannelNetwork
from src.data_utils.dataset_utils import get_cifar100_datasets
from src.data_utils.rgb_to_rgbl import RGBtoRGBL
from src.utils.debug_utils import (
    analyze_gradient_flow, 
    analyze_parameter_magnitudes,
    debug_forward_pass,
    check_for_dead_neurons,
    check_pathway_gradients,
    add_diagnostic_hooks
)


def load_model_with_hooks(model_path, model_class, device='cuda'):
    """Load model and add diagnostic hooks"""
    
    # Create model based on its class type
    if model_class == MultiChannelResNetNetwork:
        model = model_class(
            num_classes=100,  # CIFAR-100
            color_input_channels=3,  # RGB
            brightness_input_channels=1,  # L channel
            reduce_architecture=True,  # Use smaller architecture for CIFAR
            debug_mode=True,
        ).to(device)
    elif model_class == BaseMultiChannelNetwork:
        # For BaseMultiChannelNetwork, we need to provide input sizes
        # CIFAR images are 32x32x3, flattened would be 3072 for color
        # Brightness channel is 32x32x1, flattened would be 1024
        model = model_class(
            num_classes=100,  # CIFAR-100
            color_input_size=3072,  # Flattened 32x32x3 RGB
            brightness_input_size=1024,  # Flattened 32x32x1 L channel
        ).to(device)
    else:
        raise ValueError(f"Unsupported model class: {model_class.__name__}")
    
    try:
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device)
            # Try to load state dict but don't raise error if it fails
            try:
                model.load_state_dict(state_dict)
                print(f"Loaded model from {model_path}")
            except Exception as e:
                print(f"Could not load model from {model_path}, using fresh initialization")
                print(f"Error: {e}")
        else:
            print(f"No saved model found at {model_path}, using fresh initialization")
    except Exception as e:
        print(f"Error when loading model: {e}")
        print("Using freshly initialized model")
    
    # Add hooks for monitoring
    hooks = add_diagnostic_hooks(model)
    
    return model, hooks
    
    try:
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device)
            # Try to load state dict but don't raise error if it fails
            try:
                model.load_state_dict(state_dict)
                print(f"Loaded model from {model_path}")
            except Exception as e:
                print(f"Could not load model from {model_path}, using fresh initialization")
                print(f"Error: {e}")
        else:
            print(f"No saved model found at {model_path}, using fresh initialization")
    except Exception as e:
        print(f"Error when loading model: {e}")
        print("Using freshly initialized model")
    
    # Add hooks for monitoring
    hooks = add_diagnostic_hooks(model)
    
    return model, hooks


def compare_model_initializations(resnet_model, base_model):
    """Compare weight initialization between models"""
    resnet_weights = {}
    base_weights = {}
    
    # Collect weights by layer type
    for name, param in resnet_model.named_parameters():
        if "weight" in name:
            layer_type = name.split('.')[-2]
            if layer_type not in resnet_weights:
                resnet_weights[layer_type] = []
            resnet_weights[layer_type].append(param.abs().mean().item())
    
    for name, param in base_model.named_parameters():
        if "weight" in name:
            layer_type = name.split('.')[-2]
            if layer_type not in base_weights:
                base_weights[layer_type] = []
            base_weights[layer_type].append(param.abs().mean().item())
    
    # Compare weights for each layer type
    print("Weight magnitude comparison by layer type:")
    all_layer_types = set(list(resnet_weights.keys()) + list(base_weights.keys()))
    
    for layer_type in all_layer_types:
        resnet_avg = np.mean(resnet_weights.get(layer_type, [0])) if layer_type in resnet_weights else 0
        base_avg = np.mean(base_weights.get(layer_type, [0])) if layer_type in base_weights else 0
        
        if resnet_avg > 0 and base_avg > 0:
            ratio = resnet_avg / base_avg
            print(f"  {layer_type}: ResNet={resnet_avg:.6f}, Base={base_avg:.6f}, Ratio={ratio:.2f}")
        else:
            print(f"  {layer_type}: ResNet={resnet_avg:.6f}, Base={base_avg:.6f}")


def perform_input_gradient_analysis(model, loader, device, is_base_model=False):
    """Analyze how gradients flow from inputs to early layers"""
    # Get a batch
    for batch in loader:
        # Unpack data
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            color, brightness, labels = batch
        else:
            raise ValueError("DataLoader must provide (color, brightness, labels) tuples")
        
        # Move to device
        color = color.to(device, non_blocking=True)
        brightness = brightness.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # For BaseMultiChannelNetwork, flatten inputs
        if is_base_model:
            # Flatten from [batch, channels, height, width] to [batch, channels*height*width]
            color_flat = color.view(color.size(0), -1)
            brightness_flat = brightness.view(brightness.size(0), -1)
            
            # Enable gradients for inputs
            color_flat.requires_grad_(True)
            brightness_flat.requires_grad_(True)
            
            # Forward pass
            model.train()
            outputs = model(color_flat, brightness_flat)
            
            # Get loss
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Analyze input gradients
            if color_flat.grad is not None:
                color_grad_mean = color_flat.grad.abs().mean().item()
                color_grad_std = color_flat.grad.abs().std().item()
                print(f"Color input gradients: Mean={color_grad_mean:.8f}, Std={color_grad_std:.8f}")
            else:
                print("Color gradients are None - gradient not flowing back to inputs!")
                
            if brightness_flat.grad is not None:
                brightness_grad_mean = brightness_flat.grad.abs().mean().item()
                brightness_grad_std = brightness_flat.grad.abs().std().item()
                print(f"Brightness input gradients: Mean={brightness_grad_mean:.8f}, Std={brightness_grad_std:.8f}")
            else:
                print("Brightness gradients are None - gradient not flowing back to inputs!")
        else:
            # Normal CNN input processing
            # Enable gradients for inputs
            color.requires_grad_(True)
            brightness.requires_grad_(True)
            
            # Forward pass
            model.train()
            outputs = model(color, brightness)
            
            # Get loss
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Analyze input gradients
            if color.grad is not None:
                color_grad_mean = color.grad.abs().mean().item()
                color_grad_std = color.grad.abs().std().item()
                print(f"Color input gradients: Mean={color_grad_mean:.8f}, Std={color_grad_std:.8f}")
            else:
                print("Color gradients are None - gradient not flowing back to inputs!")
                
            if brightness.grad is not None:
                brightness_grad_mean = brightness.grad.abs().mean().item()
                brightness_grad_std = brightness.grad.abs().std().item()
                print(f"Brightness input gradients: Mean={brightness_grad_mean:.8f}, Std={brightness_grad_std:.8f}")
            else:
                print("Brightness gradients are None - gradient not flowing back to inputs!")
        
        break  # Just need one batch


def compare_forward_activations(resnet_model, base_model, loader, device):
    """Compare activation patterns between ResNet and Base models"""
    # Get a batch
    for batch in loader:
        # Unpack data
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            color, brightness, labels = batch
        else:
            raise ValueError("DataLoader must provide (color, brightness, labels) tuples")
        
        # Move to device
        color = color.to(device, non_blocking=True)
        brightness = brightness.to(device, non_blocking=True)
        
        # Process inputs for BaseMultiChannelNetwork (flatten)
        color_flat = color.view(color.size(0), -1)
        brightness_flat = brightness.view(brightness.size(0), -1)
        
        # Get activations from ResNet model
        resnet_model.eval()
        resnet_stats = debug_forward_pass(resnet_model, (color, brightness))
        
        # Get activations from Base model
        base_model.eval()
        base_stats = debug_forward_pass(base_model, (color_flat, brightness_flat))
        
        # Compare key activation layers
        print("\nForward pass activation comparison:")
        
        # For ResNet model
        print("\nResNet Model Activations:")
        for layer_name, stats in resnet_stats.items():
            if isinstance(stats, dict):
                if "color_mean" in stats:  # Multi-channel layer
                    print(f"  {layer_name}:")
                    print(f"    Color: mean={stats['color_mean']:.6f}, std={stats['color_std']:.6f}, zeros={stats['color_zeros']*100:.2f}%")
                    print(f"    Brightness: mean={stats['brightness_mean']:.6f}, std={stats['brightness_std']:.6f}, zeros={stats['brightness_zeros']*100:.2f}%")
                else:  # Standard layer
                    print(f"  {layer_name}: mean={stats['mean']:.6f}, std={stats['std']:.6f}, zeros={stats['zeros']*100:.2f}%")
        
        # For Base model
        print("\nBase Model Activations:")
        for layer_name, stats in base_stats.items():
            if isinstance(stats, dict):
                if "color_mean" in stats:  # Multi-channel layer
                    print(f"  {layer_name}:")
                    print(f"    Color: mean={stats['color_mean']:.6f}, std={stats['color_std']:.6f}, zeros={stats['color_zeros']*100:.2f}%")
                    print(f"    Brightness: mean={stats['brightness_mean']:.6f}, std={stats['brightness_std']:.6f}, zeros={stats['brightness_zeros']*100:.2f}%")
                else:  # Standard layer
                    print(f"  {layer_name}: mean={stats['mean']:.6f}, std={stats['std']:.6f}, zeros={stats['zeros']*100:.2f}%")
        
        break  # Just need one batch


def check_gradient_norms_during_training(model, loader, device, num_batches=3, is_base_model=False):
    """Check gradient norms during a few batches of training"""
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    print("\nGradient norms during training:")
    
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= num_batches:
            break
            
        # Unpack data
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            color, brightness, labels = batch
        else:
            raise ValueError("DataLoader must provide (color, brightness, labels) tuples")
        
        # Move to device
        color = color.to(device, non_blocking=True)
        brightness = brightness.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Process inputs for BaseMultiChannelNetwork if needed
        if is_base_model:
            # Flatten inputs for base model
            color = color.view(color.size(0), -1)
            brightness = brightness.view(brightness.size(0), -1)
            
        # Forward and backward pass
        optimizer.zero_grad()
        outputs = model(color, brightness)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Check pathway gradients
        pathway_stats = check_pathway_gradients(model)
        if pathway_stats and 'pathway_balance' in pathway_stats:
            balance = pathway_stats['pathway_balance']
            print(f"Batch {batch_idx+1}:")
            print(f"  Color pathway gradient: {balance['color_pathway_grad_percent']:.2f}%")
            print(f"  Brightness pathway gradient: {balance['brightness_pathway_grad_percent']:.2f}%")
            print(f"  Color/Brightness ratio: {balance['ratio_color_to_brightness']:.4f}")
        
        # Check for dead neurons
        if batch_idx == 0:
            if is_base_model:
                dead_neuron_stats = check_for_dead_neurons(model, (color, brightness))
            else:
                # For CNN model, need to use original shape
                color_original = color.to(device, non_blocking=True)
                brightness_original = brightness.to(device, non_blocking=True)
                dead_neuron_stats = check_for_dead_neurons(model, (color_original, brightness_original))
                
            print("\nDead neuron check:")
            for layer_name, stats in dead_neuron_stats.items():
                if isinstance(stats, dict):
                    print(f"  {layer_name}:")
                    print(f"    Color pathway: {stats['color_dead_percent']:.2f}% dead")
                    print(f"    Brightness pathway: {stats['brightness_dead_percent']:.2f}% dead")
                else:
                    print(f"  {layer_name}: {stats:.2f}% dead")
        
        # Check layer-by-layer gradient norms
        print(f"\nBatch {batch_idx+1} gradient norms by layer:")
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                if "weight" in name or "bias" in name:
                    print(f"  {name}: {grad_norm:.8f}")
        
        # Don't actually update the model
        optimizer.zero_grad()


# Define a dataset class for RGBL transformation
class CIFAR100WithRGBL(torch.utils.data.Dataset):
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform
        
    def __len__(self):
        return len(self.base_dataset)
        
    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        # Apply RGB to RGBL transformation
        color, brightness = self.transform(image)
        return color, brightness, label


def main():
    parser = argparse.ArgumentParser(description='Diagnose Multi-Stream Neural Network issues')
    parser.add_argument('--resnet-model', type=str, default='best_multichannelresnetnetwork_model.pth',
                        help='Path to the ResNet model checkpoint')
    parser.add_argument('--base-model', type=str, default='best_basemultichannelnetwork_model.pth',
                        help='Path to the Base model checkpoint')
    parser.add_argument('--data-dir', type=str, default='data/cifar-100',
                        help='Path to the CIFAR-100 data directory')
    parser.add_argument('--output-dir', type=str, default='diagnostics',
                        help='Directory to save diagnostic plots')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run analysis on')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for data loader')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize models with hooks
    resnet_model, resnet_hooks = load_model_with_hooks(
        args.resnet_model, 
        MultiChannelResNetNetwork,
        args.device
    )
    
    base_model, base_hooks = load_model_with_hooks(
        args.base_model, 
        BaseMultiChannelNetwork,
        args.device
    )
    
    # Load a small subset of the data for testing
    try:
        # Get the CIFAR-100 datasets
        _, test_dataset, class_names = get_cifar100_datasets(
            data_dir=args.data_dir,
        )
        
        # Create RGB to RGBL transform
        rgb_to_rgbl = RGBtoRGBL()
        
        # Wrap the test dataset with our transform
        rgbl_test_dataset = CIFAR100WithRGBL(test_dataset, rgb_to_rgbl)
        
        test_loader = DataLoader(
            rgbl_test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0  # Use single process to avoid serialization issues
        )
        print(f"Loaded CIFAR-100 test dataset with {len(rgbl_test_dataset)} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        test_loader = None
    
    # Run diagnostic tests
    if test_loader is not None:
        # 1. Compare weight initializations
        print("\n=== COMPARING MODEL WEIGHT INITIALIZATIONS ===")
        compare_model_initializations(resnet_model, base_model)
        
        # 2. Generate and save gradient flow visualizations
        print("\n=== ANALYZING GRADIENT FLOW ===")
        analyze_gradient_flow(resnet_model, os.path.join(args.output_dir, 'resnet_gradient_flow.png'))
        analyze_gradient_flow(base_model, os.path.join(args.output_dir, 'base_gradient_flow.png'))
        
        # 3. Generate and save parameter magnitude visualizations
        print("\n=== ANALYZING PARAMETER MAGNITUDES ===")
        analyze_parameter_magnitudes(resnet_model, os.path.join(args.output_dir, 'resnet_parameter_magnitudes.png'))
        analyze_parameter_magnitudes(base_model, os.path.join(args.output_dir, 'base_parameter_magnitudes.png'))
        
        # 4. Compare forward pass activations
        print("\n=== COMPARING FORWARD PASS ACTIVATIONS ===")
        compare_forward_activations(resnet_model, base_model, test_loader, args.device)
        
        # 5. Check input gradient flow
        print("\n=== CHECKING INPUT GRADIENT FLOW FOR RESNET MODEL ===")
        perform_input_gradient_analysis(resnet_model, test_loader, args.device, is_base_model=False)
        
        print("\n=== CHECKING INPUT GRADIENT FLOW FOR BASE MODEL ===")
        perform_input_gradient_analysis(base_model, test_loader, args.device, is_base_model=True)
        
        # 6. Check gradient norms during training batches
        print("\n=== CHECKING GRADIENT NORMS DURING TRAINING FOR RESNET MODEL ===")
        check_gradient_norms_during_training(resnet_model, test_loader, args.device, is_base_model=False)
        
        print("\n=== CHECKING GRADIENT NORMS DURING TRAINING FOR BASE MODEL ===")
        check_gradient_norms_during_training(base_model, test_loader, args.device, is_base_model=True)
    
    # Clean up hooks
    for hook in resnet_hooks:
        hook.remove()
    for hook in base_hooks:
        hook.remove()
    
    print("\nDiagnostic tests completed. Check the output directory for plots and examine the console output.")


if __name__ == "__main__":
    main()
