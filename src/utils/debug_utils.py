"""
Debugging utilities for multi-stream neural networks.

These utilities help diagnose issues with model training, particularly 
for MultiChannelResNetNetwork and other multi-stream architectures.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


def analyze_gradient_flow(model: nn.Module, save_path: Optional[str] = None) -> None:
    """
    Visualize gradient flow through the network layers.
    
    Args:
        model: The model to analyze
        save_path: If provided, saves the plot to this path
    """
    # Create lists to hold data
    avg_grads = []
    max_grads = []
    layers = []
    
    # Collect gradient data from all parameters
    for n, p in model.named_parameters():
        if p.requires_grad and ("weight" in n or "bias" in n) and p.grad is not None:
            layers.append(n)
            avg_grads.append(p.grad.abs().mean().item())
            max_grads.append(p.grad.abs().max().item())
    
    # Create plot
    plt.figure(figsize=(12, 8))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), avg_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(avg_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(avg_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(avg_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # Zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("Gradient Magnitude")
    plt.title("Gradient Flow")
    plt.grid(True)
    plt.legend([
        "Max Gradients", 
        "Mean Gradients"
    ], loc="upper right")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def analyze_parameter_magnitudes(model: nn.Module, save_path: Optional[str] = None) -> None:
    """
    Visualize parameter magnitudes across the network.
    
    Args:
        model: The model to analyze
        save_path: If provided, saves the plot to this path
    """
    # Create lists to hold data
    avg_weights = []
    max_weights = []
    layers = []
    
    # Collect weight data from all parameters
    for n, p in model.named_parameters():
        if "weight" in n:  # Only analyze weights
            layers.append(n)
            avg_weights.append(p.abs().mean().item())
            max_weights.append(p.abs().max().item())
    
    # Create plot
    plt.figure(figsize=(12, 8))
    plt.bar(np.arange(len(max_weights)), max_weights, alpha=0.1, lw=1, color="r")
    plt.bar(np.arange(len(max_weights)), avg_weights, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(avg_weights) + 1, lw=2, color="k")
    plt.xticks(range(0, len(avg_weights), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(avg_weights))
    plt.xlabel("Layers")
    plt.ylabel("Weight Magnitude")
    plt.title("Parameter Magnitudes")
    plt.grid(True)
    plt.legend([
        "Max Weights", 
        "Mean Weights"
    ], loc="upper right")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def debug_forward_pass(model: nn.Module, sample_batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, Dict[str, float]]:
    """
    Debug a forward pass through the model by tracking activations at each layer.
    
    This should be used with hooks registered on the model.
    
    Args:
        model: The model to debug
        sample_batch: A sample (color, brightness) input batch
        
    Returns:
        Dictionary with activation statistics for each layer
    """
    # Set up hooks to capture activations
    activation_stats = {}
    hooks = []
    
    def hook_fn(name):
        def hook(module, input, output):
            # For multi-channel modules that return tuples
            if isinstance(output, tuple) and len(output) == 2:
                color_output, brightness_output = output
                activation_stats[name] = {
                    "color_mean": color_output.abs().mean().item(),
                    "color_std": color_output.std().item() if color_output.numel() > 1 else 0.0,
                    "color_max": color_output.abs().max().item() if color_output.numel() > 0 else 0.0,
                    "color_min": color_output.abs().min().item() if color_output.numel() > 0 else 0.0,
                    "color_zeros": (color_output == 0).float().mean().item(),
                    "brightness_mean": brightness_output.abs().mean().item(),
                    "brightness_std": brightness_output.std().item() if brightness_output.numel() > 1 else 0.0,
                    "brightness_max": brightness_output.abs().max().item() if brightness_output.numel() > 0 else 0.0,
                    "brightness_min": brightness_output.abs().min().item() if brightness_output.numel() > 0 else 0.0,
                    "brightness_zeros": (brightness_output == 0).float().mean().item(),
                }
            else:
                # For standard modules
                activation_stats[name] = {
                    "mean": output.abs().mean().item(),
                    "std": output.std().item() if output.numel() > 1 else 0.0,
                    "max": output.abs().max().item() if output.numel() > 0 else 0.0,
                    "min": output.abs().min().item() if output.numel() > 0 else 0.0,
                    "zeros": (output == 0).float().mean().item(),
                }
        return hook
    
    # Register hooks for all modules
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.Linear, nn.ReLU)) or \
           "MultiChannel" in module.__class__.__name__:
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    # Do a forward pass
    with torch.no_grad():
        color_input, brightness_input = sample_batch
        model.eval()
        _ = model(color_input, brightness_input)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return activation_stats


def track_parameter_updates(model: nn.Module, optimizer: torch.optim.Optimizer, 
                           num_batches: int = 5) -> Dict[str, List[float]]:
    """
    Track parameter updates over a few batches to detect issues.
    
    Args:
        model: The model to track
        optimizer: The optimizer being used
        num_batches: Number of batches to track
        
    Returns:
        Dictionary with parameter update statistics
    """
    # Initialize stats tracking
    update_stats = defaultdict(list)
    param_copy = {}
    
    # Store initial parameter values
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_copy[name] = param.data.clone()
    
    # Track changes over batches
    for batch_idx in range(num_batches):
        # After optimizer.step() is called elsewhere
        for name, param in model.named_parameters():
            if param.requires_grad:
                if name in param_copy:
                    # Calculate update magnitude
                    update = (param.data - param_copy[name]).abs().mean().item()
                    update_stats[name].append(update)
                    # Store new value for next comparison
                    param_copy[name] = param.data.clone()
    
    return update_stats


def detect_vanishing_or_exploding(stats: Dict[str, List[float]], 
                                 threshold_vanish: float = 1e-7, 
                                 threshold_explode: float = 1.0) -> Dict[str, str]:
    """
    Analyze parameter update statistics to detect vanishing or exploding values.
    
    Args:
        stats: Dictionary of parameter update statistics from track_parameter_updates
        threshold_vanish: Threshold below which updates are considered vanishing
        threshold_explode: Threshold above which updates are considered exploding
        
    Returns:
        Dictionary with detection results for each parameter
    """
    results = {}
    
    for name, updates in stats.items():
        avg_update = sum(updates) / len(updates)
        
        if avg_update < threshold_vanish:
            results[name] = "VANISHING"
        elif avg_update > threshold_explode:
            results[name] = "EXPLODING"
        else:
            results[name] = "NORMAL"
    
    return results


def check_for_dead_neurons(model: nn.Module, sample_batch: Tuple[torch.Tensor, torch.Tensor], 
                          num_batches: int = 5, threshold: float = 0.05) -> Dict[str, float]:
    """
    Check for dead neurons (those that never activate) in the model.
    
    Args:
        model: The model to check
        sample_batch: A sample (color, brightness) input batch
        num_batches: Number of batches to check
        threshold: Percentage threshold to consider neurons dead
        
    Returns:
        Dictionary with dead neuron percentages for each layer
    """
    # Set up activation tracking
    activation_counts = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, tuple) and len(output) == 2:
                color_output, brightness_output = output
                
                if name not in activation_counts:
                    # Handle different tensor dimensions
                    if len(color_output.shape) == 4:  # 4D tensor (B, C, H, W)
                        activation_counts[name] = {
                            "color": torch.zeros_like(color_output[0, :, 0, 0], dtype=torch.float32),
                            "brightness": torch.zeros_like(brightness_output[0, :, 0, 0], dtype=torch.float32)
                        }
                    elif len(color_output.shape) == 2:  # 2D tensor (B, F)
                        activation_counts[name] = {
                            "color": torch.zeros_like(color_output[0, :], dtype=torch.float32),
                            "brightness": torch.zeros_like(brightness_output[0, :], dtype=torch.float32)
                        }
                    else:
                        # Skip other dimensions
                        return
                
                # Track which neurons activate (different handling for different layer types)
                if isinstance(module, nn.ReLU) or "Activation" in module.__class__.__name__:
                    if len(color_output.shape) == 4:  # 4D tensor
                        activation_counts[name]["color"] += (color_output.abs().sum(dim=(0, 2, 3)) > 0).float()
                        activation_counts[name]["brightness"] += (brightness_output.abs().sum(dim=(0, 2, 3)) > 0).float()
                    elif len(color_output.shape) == 2:  # 2D tensor
                        activation_counts[name]["color"] += (color_output.abs().sum(dim=0) > 0).float()
                        activation_counts[name]["brightness"] += (brightness_output.abs().sum(dim=0) > 0).float()
                else:
                    # For non-activation layers, just check for non-zero outputs
                    if len(color_output.shape) == 4:  # 4D tensor
                        activation_counts[name]["color"] += (color_output.abs().sum(dim=(0, 2, 3)) > 0.01).float()
                        activation_counts[name]["brightness"] += (brightness_output.abs().sum(dim=(0, 2, 3)) > 0.01).float()
                    elif len(color_output.shape) == 2:  # 2D tensor
                        activation_counts[name]["color"] += (color_output.abs().sum(dim=0) > 0.01).float()
                        activation_counts[name]["brightness"] += (brightness_output.abs().sum(dim=0) > 0.01).float()
            else:
                # Standard modules
                if name not in activation_counts:
                    if len(output.shape) > 1:  # Ensure it has a channel dimension
                        # For 4D tensors (B, C, H, W) we want the channel dimension
                        if len(output.shape) == 4:
                            activation_counts[name] = torch.zeros_like(output[0, :, 0, 0], dtype=torch.float32)
                        # For 2D tensors (B, F) we want the feature dimension
                        elif len(output.shape) == 2:
                            activation_counts[name] = torch.zeros_like(output[0, :], dtype=torch.float32)
                
                if name in activation_counts:
                    if isinstance(module, nn.ReLU) or "Activation" in module.__class__.__name__:
                        if len(output.shape) == 4:  # (B, C, H, W)
                            activation_counts[name] += (output.abs().sum(dim=(0, 2, 3)) > 0).float()
                        elif len(output.shape) == 2:  # (B, F)
                            activation_counts[name] += (output.abs().sum(dim=0) > 0).float()
                    else:
                        # For non-activation layers
                        if len(output.shape) == 4:  # (B, C, H, W)
                            activation_counts[name] += (output.abs().sum(dim=(0, 2, 3)) > 0.01).float()
                        elif len(output.shape) == 2:  # (B, F)
                            activation_counts[name] += (output.abs().sum(dim=0) > 0.01).float()
                    
        return hook
    
    # Register hooks on activation layers and certain other layers
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.ReLU, nn.Conv2d, nn.Linear)) or \
           "MultiChannel" in module.__class__.__name__:
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    # Run several forward passes
    with torch.no_grad():
        color_input, brightness_input = sample_batch
        model.eval()
        for _ in range(num_batches):
            _ = model(color_input, brightness_input)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Calculate dead neuron percentages
    results = {}
    for name, counts in activation_counts.items():
        if isinstance(counts, dict):  # Multi-channel layers
            color_dead = (counts["color"] == 0).float().mean().item()
            brightness_dead = (counts["brightness"] == 0).float().mean().item()
            results[name] = {
                "color_dead_percent": color_dead * 100,
                "brightness_dead_percent": brightness_dead * 100
            }
        else:  # Standard layers
            dead = (counts == 0).float().mean().item()
            results[name] = dead * 100
    
    return results


def check_pathway_gradients(model: nn.Module) -> Dict[str, Dict[str, float]]:
    """
    Check for imbalanced gradients between color and brightness pathways.
    
    This is especially useful for multi-stream networks where one pathway
    might dominate the learning.
    
    Args:
        model: The model to check
        
    Returns:
        Dictionary with gradient balance statistics
    """
    results = {}
    
    # Group parameters by pathway
    color_params = []
    brightness_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            if "color" in name.lower():
                color_params.append(param)
            elif "brightness" in name.lower():
                brightness_params.append(param)
    
    # Calculate gradient statistics
    if color_params and brightness_params:
        color_grad_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach()) for p in color_params])
        ).item()
        
        brightness_grad_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach()) for p in brightness_params])
        ).item()
        
        total_grad_norm = color_grad_norm + brightness_grad_norm
        
        if total_grad_norm > 0:
            results["pathway_balance"] = {
                "color_pathway_grad_percent": (color_grad_norm / total_grad_norm) * 100,
                "brightness_pathway_grad_percent": (brightness_grad_norm / total_grad_norm) * 100,
                "ratio_color_to_brightness": color_grad_norm / max(brightness_grad_norm, 1e-8)
            }
    
    return results


def add_diagnostic_hooks(model: nn.Module) -> List[torch.utils.hooks.RemovableHandle]:
    """
    Add diagnostic hooks to monitor NaN/Inf values during training.
    
    Args:
        model: The model to monitor
        
    Returns:
        List of hook handles that can be removed later
    """
    hooks = []
    
    def check_nan_inf_hook(module, inp, output):
        if isinstance(output, tuple):
            for i, out in enumerate(output):
                if torch.isnan(out).any() or torch.isinf(out).any():
                    print(f"NaN/Inf detected in output {i} of {module.__class__.__name__}")
        else:
            if torch.isnan(output).any() or torch.isinf(output).any():
                print(f"NaN/Inf detected in output of {module.__class__.__name__}")
    
    # Add hooks to all modules
    for name, module in model.named_modules():
        if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList):
            hooks.append(module.register_forward_hook(check_nan_inf_hook))
    
    return hooks


def plot_diagnostic_history(history: Dict[str, List[float]], output_path: str, model_name: str):
    """Plot diagnostic history from training."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Gradient norms
    if history['gradient_norms']:
        axes[0, 0].plot(history['gradient_norms'])
        axes[0, 0].set_title('Gradient Norms')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Gradient Norm')
        axes[0, 0].grid(True)
    
    # Weight norms
    if history['weight_norms']:
        axes[0, 1].plot(history['weight_norms'])
        axes[0, 1].set_title('Weight Norms')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Weight Norm')
        axes[0, 1].grid(True)
    
    # Pathway balance
    if history['pathway_balance']:
        axes[1, 0].plot(history['pathway_balance'])
        axes[1, 0].set_title('Pathway Balance')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Balance Ratio')
        axes[1, 0].grid(True)
    
    # Epoch times
    if history['epoch_times']:
        axes[1, 1].plot(history['epoch_times'])
        axes[1, 1].set_title('Epoch Times')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].grid(True)
    
    plt.suptitle(f'{model_name} - Diagnostic History')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
