"""
Diagnostic script to analyze full-sized multi-channel ResNet network.

This script runs comprehensive diagnostics on the full-sized (non-reduced) 
MultiChannelResNetNetwork architecture to evaluate gradient flow, dead neurons, 
parameter updates, and other training factors.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import logging
from datetime import datetime

# Add project root to path to ensure imports work
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import after sys.path is updated
from src.models.basic_multi_channel.multi_channel_resnet_network import MultiChannelResNetNetwork
from src.data_utils.dataset_utils import get_cifar100_datasets
from src.data_utils.rgb_to_rgbl import RGBtoRGBL
from src.utils.debug_utils import (
    analyze_gradient_flow, 
    analyze_parameter_magnitudes,
    add_diagnostic_hooks
)


class DualOutput:
    """
    Class to handle output to both console and file simultaneously.
    """
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self.log_file = open(log_file_path, 'w', encoding='utf-8')
        
    def write(self, message):
        """Write message to both console and file"""
        print(message, end='')  # Console output
        self.log_file.write(message)  # File output
        self.log_file.flush()  # Ensure immediate write
        
    def close(self):
        """Close the log file"""
        if hasattr(self, 'log_file') and self.log_file:
            self.log_file.close()
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def setup_logging(results_dir='results'):
    """
    Set up dual output logging to both console and file.
    Returns the DualOutput instance and log file path.
    """
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"diagnostic_results_{timestamp}.log"
    log_path = os.path.join(results_dir, log_filename)
    
    # Create dual output handler
    dual_output = DualOutput(log_path)
    
    return dual_output, log_path


# Custom print function that uses dual output
_dual_output = None

def dprint(*args, **kwargs):
    """
    Enhanced print function that outputs to both console and log file.
    """
    global _dual_output
    
    # Convert arguments to string format like print would
    import io
    import sys
    from contextlib import redirect_stdout
    
    # Capture the print output to get exact formatting
    string_buffer = io.StringIO()
    with redirect_stdout(string_buffer):
        print(*args, **kwargs)
    message = string_buffer.getvalue()
    
    # Print to console
    sys.stdout.write(message)
    sys.stdout.flush()
    
    # Also write to log file if dual output is set up
    if _dual_output and hasattr(_dual_output, 'log_file') and _dual_output.log_file:
        try:
            _dual_output.log_file.write(message)
            _dual_output.log_file.flush()
        except Exception as e:
            # Fallback to regular print if logging fails
            print(f"[LOGGING ERROR]: {e}", file=sys.stderr)


class CIFAR100WithRGBL:
    """
    Dataset wrapper to transform CIFAR-100 to RGBL format for multi-channel networks.
    """
    def __init__(self, base_dataset, transform=None):
        self.base_dataset = base_dataset
        self.transform = transform or RGBtoRGBL()
        
    def __len__(self):
        return len(self.base_dataset)
        
    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        # Apply RGB to RGBL transformation
        color, brightness = self.transform(image)
        return color, brightness, label


def load_full_resnet_model(device='auto'):
    """Load the full-sized ResNet model with diagnostic hooks"""
    
    dprint("Creating full-sized MultiChannelResNetNetwork (reduce_architecture=False)...")
    model = MultiChannelResNetNetwork(
        num_classes=100,  # CIFAR-100
        color_input_channels=3,  # RGB
        brightness_input_channels=1,  # L channel
        num_blocks=[2, 2, 2, 2],  # ResNet-18 style
        block_type='basic',
        activation='relu',
        dropout=0.3,
        use_shared_classifier=True,
        reduce_architecture=False,  # Use FULL architecture (not reduced)
        debug_mode=True,
        device=device
    )
    
    # Add diagnostic hooks
    add_diagnostic_hooks(model)
    
    # Print model architecture summary
    params = sum(p.numel() for p in model.parameters())
    dprint(f"Full-sized model created with {params:,} parameters")
    dprint(f"Device: {model.device}")
    
    return model


def load_full_resnet50_model(device='auto'):
    """Load the full-sized ResNet-50 style model with bottleneck blocks"""
    
    dprint("Creating full-sized MultiChannelResNetNetwork ResNet-50 style (bottleneck blocks)...")
    model = MultiChannelResNetNetwork(
        num_classes=100,  # CIFAR-100
        color_input_channels=3,  # RGB
        brightness_input_channels=1,  # L channel
        num_blocks=[3, 4, 6, 3],  # ResNet-50 style
        block_type='bottleneck',  # Bottleneck blocks instead of basic
        activation='relu',
        dropout=0.3,
        use_shared_classifier=True,
        reduce_architecture=False,  # Use FULL architecture (not reduced)
        debug_mode=True,
        device=device
    )
    
    # Add diagnostic hooks
    add_diagnostic_hooks(model)
    
    # Print model architecture summary
    params = sum(p.numel() for p in model.parameters())
    dprint(f"Full-sized ResNet-50 style model created with {params:,} parameters")
    dprint(f"Device: {model.device}")
    
    return model


def load_data(batch_size=32):
    """Load CIFAR-100 dataset and prepare data loaders"""
    
    data_dir = 'data/cifar-100'
    dprint(f"Loading CIFAR-100 from: {data_dir}")
    
    # Load datasets
    train_dataset, test_dataset, class_names = get_cifar100_datasets(data_dir=data_dir)
    
    # Create RGB to RGBL transform
    rgb_to_rgbl = RGBtoRGBL()
    
    # Wrap datasets with RGBL transform
    rgbl_train_dataset = CIFAR100WithRGBL(train_dataset, rgb_to_rgbl)
    rgbl_test_dataset = CIFAR100WithRGBL(test_dataset, rgb_to_rgbl)
    
    # Create data loaders with small batch size for diagnostics
    train_loader = DataLoader(
        rgbl_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing issues for diagnostics
        pin_memory=False  # Disable pin_memory to avoid MPS warnings
    )
    
    test_loader = DataLoader(
        rgbl_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Avoid multiprocessing issues for diagnostics
        pin_memory=False  # Disable pin_memory to avoid MPS warnings
    )
    
    dprint(f"Created data loaders with batch size {batch_size}")
    dprint(f"Train samples: {len(rgbl_train_dataset)}, Test samples: {len(rgbl_test_dataset)}")
    
    return train_loader, test_loader, class_names


def run_gradient_flow_test(model, data_loader, device, model_name=""):
    """Test gradient flow through the model during backpropagation"""
    
    dprint("\n=== Gradient Flow Test ===")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Get a batch of data
    model.train()
    color_data, brightness_data, labels = next(iter(data_loader))
    color_data = color_data.to(device)
    brightness_data = brightness_data.to(device)
    labels = labels.to(device)
    
    # Forward pass
    optimizer.zero_grad()
    outputs = model(color_data, brightness_data)
    loss = criterion(outputs, labels)
    
    # Backward pass
    loss.backward()
    
    # Collect gradient statistics manually
    max_grad = 0.0
    min_grad = float('inf')
    max_grad_layer = ""
    min_grad_layer = ""
    
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_mag = param.grad.abs().mean().item()
            if grad_mag > max_grad:
                max_grad = grad_mag
                max_grad_layer = name
            if grad_mag < min_grad and grad_mag > 0:
                min_grad = grad_mag
                min_grad_layer = name
    
    grad_ratio = max_grad / min_grad if min_grad > 0 else float('inf')
    
    # Compute accuracy
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    
    dprint(f"Mini-batch loss: {loss.item():.4f}, accuracy: {accuracy:.2f}%")
    dprint("Gradient statistics:")
    dprint(f"  Largest gradient: {max_grad:.4e} (in {max_grad_layer})")
    dprint(f"  Smallest gradient: {min_grad:.4e} (in {min_grad_layer})")
    dprint(f"  Largest/smallest ratio: {grad_ratio:.4e}")
    
    # Check for color and brightness pathways
    color_has_grad = False
    brightness_has_grad = False
    
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            if 'color' in name:
                if param.grad.abs().sum().item() > 0:
                    color_has_grad = True
            if 'brightness' in name:
                if param.grad.abs().sum().item() > 0:
                    brightness_has_grad = True
    
    dprint("\nPathway gradient check:")
    dprint(f"  Color pathway: {'✅ Active' if color_has_grad else '❌ Inactive'}")
    dprint(f"  Brightness pathway: {'✅ Active' if brightness_has_grad else '❌ Inactive'}")
    
    # Visualize gradient flow
    # Save plot to results directory with model-specific naming
    os.makedirs('results', exist_ok=True)
    plot_filename = f'results/{model_name}_gradient_flow.png' if model_name else 'results/full_resnet_gradient_flow.png'
    analyze_gradient_flow(model, save_path=plot_filename)
    dprint(f"Gradient flow plot saved to {plot_filename}")
    

def analyze_activations(model, data_loader, device):
    """Analyze activations throughout the model"""
    
    dprint("\n=== Activation Analysis ===")
    model.eval()
    
    # Get a batch of data
    color_data, brightness_data, _ = next(iter(data_loader))
    color_data = color_data.to(device)
    brightness_data = brightness_data.to(device)
    
    # Prepare to collect activation statistics
    activation_hooks = []
    activation_stats = {}
    
    # Function to collect statistics from activations
    def hook_fn(name):
        def hook(module, input, output):
            # Handle different output types
            if isinstance(output, tuple):
                # For multi-channel outputs (color, brightness)
                for idx, out in enumerate(output):
                    pathway = "color" if idx == 0 else "brightness"
                    key = f"{name}_{pathway}"
                    
                    if out is not None:
                        activation_stats[key] = {
                            'mean': out.abs().mean().item(),
                            'std': out.std().item(),
                            'min': out.min().item(),
                            'max': out.max().item(),
                            'zero_percent': 100 * (out == 0).float().mean().item()
                        }
            else:
                # For single outputs
                activation_stats[name] = {
                    'mean': output.abs().mean().item(),
                    'std': output.std().item(),
                    'min': output.min().item(),
                    'max': output.max().item(),
                    'zero_percent': 100 * (output == 0).float().mean().item()
                }
        return hook
    
    # Register hooks for key layers
    for name, module in model.named_modules():
        # Skip non-interesting layers
        if not name or isinstance(module, nn.Sequential) or '.' in name:
            continue
            
        # Register hook for this layer
        hook = module.register_forward_hook(hook_fn(name))
        activation_hooks.append(hook)
    
    # Run forward pass
    with torch.no_grad():
        _ = model(color_data, brightness_data)
    
    # Remove hooks
    for hook in activation_hooks:
        hook.remove()
    
    # Display activation statistics
    dprint("\nActivation statistics:")
    for layer_name, stats in activation_stats.items():
        dprint(f"  {layer_name}:")
        dprint(f"    Mean: {stats['mean']:.4f}, StdDev: {stats['std']:.4f}")
        dprint(f"    Min: {stats['min']:.4f}, Max: {stats['max']:.4f}")
        dprint(f"    Zero %: {stats['zero_percent']:.2f}%")
    
    # Check for possible dead neurons (high percentage of zeros)
    dead_layers = {}
    for layer_name, stats in activation_stats.items():
        if stats['zero_percent'] > 20:  # More than 20% zeros
            dead_layers[layer_name] = stats['zero_percent']
            
    if dead_layers:
        dprint("\n⚠️ Warning: Potential dead neurons detected:")
        for layer_name, percent in dead_layers.items():
            dprint(f"  {layer_name}: {percent:.2f}% zeros")
    else:
        dprint("\n✅ No dead neurons detected (all layers < 20% zeros)")


def run_mini_train(model, train_loader, test_loader, device, epochs=2, model_name=""):
    """Run a mini training loop to test model learning capability"""
    
    dprint("\n=== Mini Training Test ===")
    
    # Compile model
    model.compile(
        optimizer='adamw',
        learning_rate=0.001,
        weight_decay=0.0001,
        loss='cross_entropy',
        gradient_clip=1.0,
        scheduler='onecycle'
    )
    
    # Train for a few epochs
    history = model.fit(
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=epochs,
        early_stopping_patience=epochs + 1,  # Disable early stopping
        verbose=1
    )
    
    # Plot learning curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], 'b-', label='Training')
    plt.plot(history['val_loss'], 'r-', label='Validation')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_accuracy'], 'b-', label='Training')
    plt.plot(history['val_accuracy'], 'r-', label='Validation')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plot_filename = f'results/{model_name}_learning_curves.png' if model_name else 'results/full_resnet_learning_curves.png'
    plt.savefig(plot_filename)
    dprint(f"Learning curves saved to {plot_filename}")
    
    # Analyze final metrics
    final_train_acc = history['train_accuracy'][-1] * 100
    final_val_acc = history['val_accuracy'][-1] * 100
    final_train_loss = history['train_loss'][-1]
    final_val_loss = history['val_loss'][-1]
    
    dprint(f"\nFinal training accuracy: {final_train_acc:.2f}%")
    dprint(f"Final validation accuracy: {final_val_acc:.2f}%")
    dprint(f"Final training loss: {final_train_loss:.4f}")
    dprint(f"Final validation loss: {final_val_loss:.4f}")
    
    # Check for overfitting
    acc_gap = final_train_acc - final_val_acc
    loss_gap = final_val_loss - final_train_loss
    
    if acc_gap > 20 or loss_gap > 1.0:
        dprint("⚠️ Warning: Model shows signs of overfitting (expected for full architecture)")
    else:
        dprint("✅ Model is learning without excessive overfitting")


def analyze_parameters(model, model_name=""):
    """Analyze parameter distributions and magnitudes"""
    
    dprint("\n=== Parameter Analysis ===")
    
    # Calculate parameter statistics manually
    total_params = sum(p.numel() for p in model.parameters())
    
    color_params = []
    brightness_params = []
    param_magnitudes = []
    max_magnitude = 0.0
    min_magnitude = float('inf')
    max_magnitude_layer = ""
    min_magnitude_layer = ""
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            magnitude = param.abs().mean().item()
            param_magnitudes.append(magnitude)
            
            if magnitude > max_magnitude:
                max_magnitude = magnitude
                max_magnitude_layer = name
            if magnitude < min_magnitude and magnitude > 0:
                min_magnitude = magnitude
                min_magnitude_layer = name
                
            # Check for pathway-specific parameters
            if 'color' in name:
                color_params.append(magnitude)
            elif 'brightness' in name:
                brightness_params.append(magnitude)
    
    # Calculate average parameter magnitude
    avg_magnitude = sum(param_magnitudes) / len(param_magnitudes) if param_magnitudes else 0
    
    # Display basic stats
    dprint("\nParameter statistics:")
    dprint(f"  Total parameters: {total_params:,}")
    dprint(f"  Average magnitude: {avg_magnitude:.4f}")
    dprint(f"  Largest magnitude: {max_magnitude:.4f} (in {max_magnitude_layer})")
    dprint(f"  Smallest magnitude: {min_magnitude:.4f} (in {min_magnitude_layer})")
    
    # Check for pathway imbalance
    if color_params and brightness_params:
        color_magnitude = sum(color_params) / len(color_params)
        brightness_magnitude = sum(brightness_params) / len(brightness_params)
        
        ratio = color_magnitude / brightness_magnitude if brightness_magnitude > 1e-10 else float('inf')
        dprint("\nPathway magnitude comparison:")
        dprint(f"  Color pathway: {color_magnitude:.4f}")
        dprint(f"  Brightness pathway: {brightness_magnitude:.4f}")
        dprint(f"  Ratio (color/brightness): {ratio:.4f}")
        
        if ratio > 5 or ratio < 0.2:
            dprint("⚠️ Warning: Significant imbalance between pathways")
        else:
            dprint("✅ Pathways have reasonably balanced parameter magnitudes")
    
    # Visualize parameter magnitudes
    os.makedirs('results', exist_ok=True)
    plot_filename = f'results/{model_name}_parameter_magnitudes.png' if model_name else 'results/full_resnet_parameter_magnitudes.png'
    analyze_parameter_magnitudes(model, save_path=plot_filename)
    dprint(f"Parameter magnitude plot saved to {plot_filename}")


def analyze_input_gradients(model, data_loader, device):
    """Analyze gradient flow from loss back to input tensors"""
    
    dprint("\n=== Input Gradient Analysis ===")
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    # Get a batch of data
    color_data, brightness_data, labels = next(iter(data_loader))
    color_data = color_data.to(device)
    brightness_data = brightness_data.to(device)
    labels = labels.to(device)
    
    # Enable gradient computation for inputs
    color_data.requires_grad_(True)
    brightness_data.requires_grad_(True)
    
    # Forward pass
    model.zero_grad()
    outputs = model(color_data, brightness_data)
    loss = criterion(outputs, labels)
    
    # Backward pass
    loss.backward()
    
    # Analyze input gradients
    if color_data.grad is not None:
        color_grad_stats = {
            'mean': color_data.grad.abs().mean().item(),
            'std': color_data.grad.abs().std().item(),
            'min': color_data.grad.abs().min().item(),
            'max': color_data.grad.abs().max().item(),
            'norm': color_data.grad.norm().item()
        }
        dprint("Color input gradients:")
        dprint(f"  Mean: {color_grad_stats['mean']:.8f}")
        dprint(f"  Std: {color_grad_stats['std']:.8f}")
        dprint(f"  Min: {color_grad_stats['min']:.8f}")
        dprint(f"  Max: {color_grad_stats['max']:.8f}")
        dprint(f"  L2 Norm: {color_grad_stats['norm']:.8f}")
    else:
        dprint("❌ Color input gradients are None - gradient not flowing back to color inputs!")
        
    if brightness_data.grad is not None:
        brightness_grad_stats = {
            'mean': brightness_data.grad.abs().mean().item(),
            'std': brightness_data.grad.abs().std().item(),
            'min': brightness_data.grad.abs().min().item(),
            'max': brightness_data.grad.abs().max().item(),
            'norm': brightness_data.grad.norm().item()
        }
        dprint("Brightness input gradients:")
        dprint(f"  Mean: {brightness_grad_stats['mean']:.8f}")
        dprint(f"  Std: {brightness_grad_stats['std']:.8f}")
        dprint(f"  Min: {brightness_grad_stats['min']:.8f}")
        dprint(f"  Max: {brightness_grad_stats['max']:.8f}")
        dprint(f"  L2 Norm: {brightness_grad_stats['norm']:.8f}")
    else:
        dprint("❌ Brightness input gradients are None - gradient not flowing back to brightness inputs!")
    
    # Compare gradient magnitudes between pathways
    if color_data.grad is not None and brightness_data.grad is not None:
        color_magnitude = color_data.grad.abs().mean().item()
        brightness_magnitude = brightness_data.grad.abs().mean().item()
        
        if color_magnitude > 0 and brightness_magnitude > 0:
            ratio = color_magnitude / brightness_magnitude if brightness_magnitude > 1e-10 else float('inf')
            dprint("\nInput gradient pathway comparison:")
            dprint(f"  Color/Brightness ratio: {ratio:.4f}")
            
            if ratio > 10 or ratio < 0.1:
                dprint("⚠️ Warning: Significant imbalance in input gradient magnitudes")
            else:
                dprint("✅ Input gradients are reasonably balanced between pathways")
        else:
            dprint("⚠️ Warning: One or both input pathways have zero gradients")
    
    # Check for vanishing gradients
    total_input_grad_norm = 0
    if color_data.grad is not None:
        total_input_grad_norm += color_data.grad.norm().item()
    if brightness_data.grad is not None:
        total_input_grad_norm += brightness_data.grad.norm().item()
    
    dprint(f"\nTotal input gradient norm: {total_input_grad_norm:.8f}")
    if total_input_grad_norm < 1e-6:
        dprint("⚠️ Warning: Very small input gradients - possible vanishing gradient problem")
    elif total_input_grad_norm > 100:
        dprint("⚠️ Warning: Very large input gradients - possible exploding gradient problem")
    else:
        dprint("✅ Input gradient magnitudes appear healthy")


def analyze_weight_initialization(model):
    """Analyze weight initialization patterns and distributions"""
    
    dprint("\n=== Weight Initialization Analysis ===")
    
    # Collect weights by layer type
    conv_weights = []
    linear_weights = []
    bn_weights = []
    conv_biases = []
    linear_biases = []
    bn_biases = []
    
    layer_stats = {}
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        param_data = param.detach().cpu().numpy().flatten()
        
        # Categorize by layer type and parameter type
        if 'conv' in name.lower():
            if 'weight' in name:
                conv_weights.extend(param_data)
            elif 'bias' in name:
                conv_biases.extend(param_data)
        elif any(x in name.lower() for x in ['linear', 'fc', 'classifier']):
            if 'weight' in name:
                linear_weights.extend(param_data)
            elif 'bias' in name:
                linear_biases.extend(param_data)
        elif any(x in name.lower() for x in ['bn', 'batchnorm', 'norm']):
            if 'weight' in name:
                bn_weights.extend(param_data)
            elif 'bias' in name:
                bn_biases.extend(param_data)
        
        # Store individual layer statistics
        layer_stats[name] = {
            'mean': param_data.mean(),
            'std': param_data.std(),
            'min': param_data.min(),
            'max': param_data.max(),
            'shape': list(param.shape)
        }
    
    # Analyze by parameter type
    param_types = [
        ('Convolutional Weights', conv_weights),
        ('Linear Weights', linear_weights),
        ('BatchNorm Weights', bn_weights),
        ('Convolutional Biases', conv_biases),
        ('Linear Biases', linear_biases),
        ('BatchNorm Biases', bn_biases)
    ]
    
    dprint("\nWeight distribution by parameter type:")
    for type_name, weights in param_types:
        if weights:
            weights = np.array(weights)
            dprint(f"\n{type_name}:")
            dprint(f"  Count: {len(weights):,} parameters")
            dprint(f"  Mean: {weights.mean():.6f}")
            dprint(f"  Std: {weights.std():.6f}")
            dprint(f"  Min: {weights.min():.6f}")
            dprint(f"  Max: {weights.max():.6f}")
            dprint(f"  |Mean|/Std ratio: {abs(weights.mean())/weights.std():.4f}" if weights.std() > 1e-10 else "  |Mean|/Std ratio: inf (std too small)")
            
            # Check for potential initialization issues
            if abs(weights.mean()) > 0.1:
                dprint("  ⚠️ Warning: Large mean value, may indicate bias in initialization")
            if weights.std() < 0.001:
                dprint("  ⚠️ Warning: Very small std, weights may be too uniform")
            elif weights.std() > 1.0:
                dprint("  ⚠️ Warning: Very large std, weights may be too spread out")
    
    # Analyze specific layers that might be problematic
    dprint("\nPotential initialization issues by layer:")
    issues_found = False
    
    for name, stats in layer_stats.items():
        issues = []
        
        # Check for common initialization problems
        if abs(stats['mean']) > 0.1:
            issues.append(f"Large mean ({stats['mean']:.4f})")
        if stats['std'] < 0.001:
            issues.append(f"Very small std ({stats['std']:.6f})")
        if stats['std'] > 2.0:
            issues.append(f"Very large std ({stats['std']:.4f})")
        if stats['min'] == stats['max']:
            issues.append("All weights identical")
        
        # Special handling for BatchNorm layers
        if any(x in name.lower() for x in ['bn', 'batchnorm', 'norm']):
            if 'weight' in name and abs(stats['mean'] - 0.5) < 0.001:
                issues.append("BatchNorm weights not properly initialized (should be ~1.0, not 0.5)")
            if 'bias' in name and abs(stats['mean']) < 0.001 and stats['std'] < 0.001:
                issues.append("BatchNorm biases initialized to zero (normal, but may need monitoring)")
        
        if issues:
            issues_found = True
            dprint(f"  {name}: {', '.join(issues)}")
    
    if not issues_found:
        dprint("  ✅ No obvious initialization issues detected")
    
    # Check pathway-specific initialization
    color_layers = []
    brightness_layers = []
    shared_layers = []
    
    for name, stats in layer_stats.items():
        if 'color' in name.lower():
            color_layers.append(stats['std'])
        elif 'brightness' in name.lower():
            brightness_layers.append(stats['std'])
        else:
            shared_layers.append(stats['std'])
    
    if color_layers and brightness_layers:
        color_avg_std = np.mean(color_layers)
        brightness_avg_std = np.mean(brightness_layers)
        
        dprint("\nPathway initialization comparison:")
        dprint(f"  Color pathway avg std: {color_avg_std:.6f}")
        dprint(f"  Brightness pathway avg std: {brightness_avg_std:.6f}")
        
        if color_avg_std > 0 and brightness_avg_std > 0:
            ratio = color_avg_std / brightness_avg_std if brightness_avg_std > 1e-10 else float('inf')
            dprint(f"  Ratio (color/brightness): {ratio:.4f}")
            
            if ratio > 2.0 or ratio < 0.5:
                dprint("  ⚠️ Warning: Significant initialization imbalance between pathways")
            else:
                dprint("  ✅ Pathway initializations are reasonably balanced")
    

def analyze_gradient_norms_during_training(model, train_loader, device, num_batches=3):
    """Analyze gradient norms during multiple training batches"""
    
    dprint("\n=== Gradient Norms During Training ===")
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    batch_stats = []
    
    for batch_idx, (color_data, brightness_data, labels) in enumerate(train_loader):
        if batch_idx >= num_batches:
            break
            
        color_data = color_data.to(device)
        brightness_data = brightness_data.to(device)
        labels = labels.to(device)
        
        # Forward and backward pass
        optimizer.zero_grad()
        outputs = model(color_data, brightness_data)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Collect gradient statistics
        batch_grad_stats = {'batch': batch_idx + 1}
        total_norm = 0.0
        color_pathway_norm = 0.0
        brightness_pathway_norm = 0.0
        shared_norm = 0.0
        param_count = 0
        
        layer_norms = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.norm().item()
                total_norm += param_norm ** 2
                param_count += 1
                
                # Categorize by pathway
                if 'color' in name.lower():
                    color_pathway_norm += param_norm ** 2
                elif 'brightness' in name.lower():
                    brightness_pathway_norm += param_norm ** 2
                else:
                    shared_norm += param_norm ** 2
                
                # Store layer-specific norms for key layers
                if any(layer_type in name for layer_type in ['conv', 'linear', 'classifier']):
                    layer_norms[name] = param_norm
        
        # Calculate final norms
        total_norm = total_norm ** 0.5
        color_pathway_norm = color_pathway_norm ** 0.5
        brightness_pathway_norm = brightness_pathway_norm ** 0.5
        shared_norm = shared_norm ** 0.5
        
        batch_grad_stats.update({
            'total_norm': total_norm,
            'color_pathway_norm': color_pathway_norm,
            'brightness_pathway_norm': brightness_pathway_norm,
            'shared_norm': shared_norm,
            'param_count': param_count,
            'layer_norms': layer_norms,
            'loss': loss.item()
        })
        
        batch_stats.append(batch_grad_stats)
        
        # Print batch statistics
        dprint(f"\nBatch {batch_idx + 1}:")
        dprint(f"  Loss: {loss.item():.4f}")
        dprint(f"  Total gradient norm: {total_norm:.6f}")
        dprint(f"  Color pathway norm: {color_pathway_norm:.6f}")
        dprint(f"  Brightness pathway norm: {brightness_pathway_norm:.6f}")
        dprint(f"  Shared layers norm: {shared_norm:.6f}")
        
        # Show pathway balance
        if color_pathway_norm > 0 and brightness_pathway_norm > 0:
            pathway_ratio = color_pathway_norm / brightness_pathway_norm if brightness_pathway_norm > 1e-10 else float('inf')
            dprint(f"  Pathway ratio (color/brightness): {pathway_ratio:.4f}")
        
        # Show top gradient norms by layer
        if layer_norms:
            sorted_layers = sorted(layer_norms.items(), key=lambda x: x[1], reverse=True)
            dprint("  Top 3 layer gradient norms:")
            for layer_name, norm in sorted_layers[:3]:
                dprint(f"    {layer_name}: {norm:.6f}")
        
        # Reset gradients for next batch
        optimizer.zero_grad()
    
    # Summary across batches
    if len(batch_stats) > 1:
        dprint(f"\nSummary across {len(batch_stats)} batches:")
        
        total_norms = [b['total_norm'] for b in batch_stats]
        color_norms = [b['color_pathway_norm'] for b in batch_stats]
        brightness_norms = [b['brightness_pathway_norm'] for b in batch_stats]
        
        dprint(f"  Total norm - Mean: {np.mean(total_norms):.6f}, Std: {np.std(total_norms):.6f}")
        dprint(f"  Color norm - Mean: {np.mean(color_norms):.6f}, Std: {np.std(color_norms):.6f}")
        dprint(f"  Brightness norm - Mean: {np.mean(brightness_norms):.6f}, Std: {np.std(brightness_norms):.6f}")
        
        # Check for gradient stability
        total_norm_std = np.std(total_norms)
        total_norm_mean = np.mean(total_norms)
        
        if total_norm_std / total_norm_mean > 0.5:
            dprint("  ⚠️ Warning: High gradient norm variability between batches")
        else:
            dprint("  ✅ Gradient norms are relatively stable across batches")
        
        # Check for vanishing/exploding gradients
        avg_total_norm = np.mean(total_norms)
        if avg_total_norm < 1e-4:
            dprint("  ⚠️ Warning: Very small gradient norms - possible vanishing gradients")
        elif avg_total_norm > 10:
            dprint("  ⚠️ Warning: Very large gradient norms - possible exploding gradients")
        else:
            dprint("  ✅ Gradient norm magnitudes appear healthy")


def analyze_model_architecture(model):
    """Analyze and display model architecture details"""
    
    dprint("\n=== Model Architecture Analysis ===")
    
    # Basic architecture info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    dprint("Model: MultiChannelResNetNetwork (Full Architecture)")
    dprint(f"Total parameters: {total_params:,}")
    dprint(f"Trainable parameters: {trainable_params:,}")
    dprint(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    # Count parameters by pathway
    color_params = 0
    brightness_params = 0
    shared_params = 0
    
    # Count layers by type
    conv_layers = 0
    linear_layers = 0
    bn_layers = 0
    
    pathway_breakdown = {
        'color_pathway': [],
        'brightness_pathway': [],
        'shared_layers': []
    }
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        
        # Categorize by pathway
        if 'color' in name.lower():
            color_params += param_count
            pathway_breakdown['color_pathway'].append((name, param_count))
        elif 'brightness' in name.lower():
            brightness_params += param_count
            pathway_breakdown['brightness_pathway'].append((name, param_count))
        else:
            shared_params += param_count
            pathway_breakdown['shared_layers'].append((name, param_count))
    
    # Count layer types
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers += 1
        elif isinstance(module, (nn.Linear, nn.LazyLinear)):
            linear_layers += 1
        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            bn_layers += 1
    
    dprint("\nParameter distribution:")
    dprint(f"  Color pathway: {color_params:,} ({100*color_params/total_params:.1f}%)")
    dprint(f"  Brightness pathway: {brightness_params:,} ({100*brightness_params/total_params:.1f}%)")
    dprint(f"  Shared layers: {shared_params:,} ({100*shared_params/total_params:.1f}%)")
    
    dprint("\nLayer type distribution:")
    dprint(f"  Convolutional layers: {conv_layers}")
    dprint(f"  Linear layers: {linear_layers}")
    dprint(f"  BatchNorm layers: {bn_layers}")
    
    # Check for pathway balance
    if color_params > 0 and brightness_params > 0:
        pathway_ratio = color_params / brightness_params
        dprint(f"\nPathway parameter ratio (color/brightness): {pathway_ratio:.4f}")
        
        if pathway_ratio > 2.0 or pathway_ratio < 0.5:
            dprint("⚠️ Warning: Significant parameter imbalance between pathways")
        else:
            dprint("✅ Pathways have reasonably balanced parameter counts")
    
    # Display largest layers
    all_layers = []
    for pathway, layers in pathway_breakdown.items():
        for layer_name, param_count in layers:
            all_layers.append((layer_name, param_count, pathway))
    
    all_layers.sort(key=lambda x: x[1], reverse=True)
    
    dprint("\nLargest layers by parameter count:")
    for i, (layer_name, param_count, pathway) in enumerate(all_layers[:5]):
        dprint(f"  {i+1}. {layer_name}: {param_count:,} ({pathway})")
    
    # Model input/output info
    dprint("\nInput/Output information:")
    dprint("  Expected inputs: color (RGB, 3 channels), brightness (L, 1 channel)")
    dprint("  Input resolution: 32x32 (CIFAR-100)")
    dprint("  Output classes: 100 (CIFAR-100)")
    
    # Check if model has specific components
    components = {
        'Residual connections': any('residual' in name.lower() or 'skip' in name.lower() 
                                 for name, _ in model.named_modules()),
        'Dropout layers': any(isinstance(module, nn.Dropout) 
                            for _, module in model.named_modules()),
        'Attention mechanisms': any('attention' in name.lower() 
                                  for name, _ in model.named_modules()),
        'Skip connections': any('skip' in name.lower() 
                              for name, _ in model.named_modules())
    }
    
    dprint("\nArchitectural components:")
    for component, present in components.items():
        dprint(f"  {component}: {'✅ Present' if present else '❌ Not present'}")
    

def verify_architectural_data_flow(model, data_loader, device, context=""):
    """
    Explicitly verify that the model's forward and backward data flow matches 
    the intended architectural design from your diagram:
    
    Forward: Color/Brightness Pathways → Integration → Classifier
    Backward: Loss gradients flow back through classifier → integration → both pathways
    
    Args:
        model: The model to test
        data_loader: DataLoader for test data
        device: Device to run on
        context: Optional context string to add to output headers
    """
    
    header = "=== Architectural Data Flow Verification ==="
    if context:
        header = f"=== Architectural Data Flow Verification ({context}) ==="
    
    dprint(f"\n{header}")
    dprint("Testing explicit pathway separation, integration, and gradient flow...")
    
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    # Get a batch of data
    color_data, brightness_data, labels = next(iter(data_loader))
    color_data = color_data.to(device)
    brightness_data = brightness_data.to(device)
    labels = labels.to(device)
    
    # Enable gradient tracking for inputs
    color_data.requires_grad_(True)
    brightness_data.requires_grad_(True)
    
    dprint("\n1. Testing Forward Pass Pathway Separation:")
    
    # Test 1: Pathway Independence - verify color and brightness process independently
    model.zero_grad()
    
    # Test color-only input (zero brightness)
    zero_brightness = torch.zeros_like(brightness_data)
    color_only_output = model(color_data, zero_brightness)
    
    # Test brightness-only input (zero color) 
    zero_color = torch.zeros_like(color_data)
    brightness_only_output = model(zero_color, brightness_data)
    
    # Test combined input
    combined_output = model(color_data, brightness_data)
    
    # Verify pathways produce different outputs
    color_norm = color_only_output.norm().item()
    brightness_norm = brightness_only_output.norm().item()
    combined_norm = combined_output.norm().item()
    
    dprint(f"   Color-only pathway output norm: {color_norm:.6f}")
    dprint(f"   Brightness-only pathway output norm: {brightness_norm:.6f}")
    dprint(f"   Combined pathways output norm: {combined_norm:.6f}")
    
    # Verify both pathways contribute
    if color_norm > 1e-4 and brightness_norm > 1e-4:
        dprint("   ✅ Both pathways independently produce meaningful outputs")
    else:
        dprint("   ❌ One or both pathways not contributing to output")
    
    # Verify integration (combined should differ from individual pathways)
    color_vs_combined = (color_only_output - combined_output).norm().item()
    brightness_vs_combined = (brightness_only_output - combined_output).norm().item()
    
    dprint(f"   Color vs Combined difference: {color_vs_combined:.6f}")
    dprint(f"   Brightness vs Combined difference: {brightness_vs_combined:.6f}")
    
    if color_vs_combined > 1e-4 and brightness_vs_combined > 1e-4:
        dprint("   ✅ Integration layer successfully combines pathway outputs")
    else:
        dprint("   ❌ Integration may not be properly combining pathways")
    
    dprint("\n2. Testing Backward Pass Gradient Flow:")
    
    # Test 2: Gradient Flow Verification - ensure gradients flow back through all pathways
    model.zero_grad()
    
    # Forward pass with both inputs
    outputs = model(color_data, brightness_data)
    loss = criterion(outputs, labels)
    
    # Backward pass
    loss.backward()
    
    # Collect pathway-specific gradient statistics
    color_pathway_grads = []
    brightness_pathway_grads = []
    integration_grads = []
    classifier_grads = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            
            if any(keyword in name.lower() for keyword in ['color', 'rgb']):
                color_pathway_grads.append(grad_norm)
            elif any(keyword in name.lower() for keyword in ['brightness', 'luminance', 'lum']):
                brightness_pathway_grads.append(grad_norm)
            elif any(keyword in name.lower() for keyword in ['alpha', 'beta', 'gamma', 'mix', 'integration']):
                integration_grads.append(grad_norm)
            elif any(keyword in name.lower() for keyword in ['classifier', 'fc', 'final']):
                classifier_grads.append(grad_norm)
    
    # Calculate pathway gradient statistics
    color_grad_sum = sum(color_pathway_grads) if color_pathway_grads else 0
    brightness_grad_sum = sum(brightness_pathway_grads) if brightness_pathway_grads else 0
    integration_grad_sum = sum(integration_grads) if integration_grads else 0
    classifier_grad_sum = sum(classifier_grads) if classifier_grads else 0
    
    # Calculate total gradients for reference (not used in current analysis)
    # total_grads = color_grad_sum + brightness_grad_sum + integration_grad_sum + classifier_grad_sum
    
    dprint(f"   Color pathway gradient sum: {color_grad_sum:.6f}")
    dprint(f"   Brightness pathway gradient sum: {brightness_grad_sum:.6f}")
    dprint(f"   Integration weights gradient sum: {integration_grad_sum:.6f}")
    dprint(f"   Classifier gradient sum: {classifier_grad_sum:.6f}")
    
    # Verify gradient flow to all components
    if color_grad_sum > 1e-6:
        dprint("   ✅ Gradients flowing to color pathway")
    else:
        dprint("   ❌ No gradients flowing to color pathway")
        
    if brightness_grad_sum > 1e-6:
        dprint("   ✅ Gradients flowing to brightness pathway") 
    else:
        dprint("   ❌ No gradients flowing to brightness pathway")
        
    if classifier_grad_sum > 1e-6:
        dprint("   ✅ Gradients flowing to classifier")
    else:
        dprint("   ❌ No gradients flowing to classifier")
    
    # Verify gradient balance between pathways
    if color_grad_sum > 0 and brightness_grad_sum > 0:
        pathway_balance = color_grad_sum / (color_grad_sum + brightness_grad_sum)
        dprint(f"   Pathway gradient balance: Color {pathway_balance:.3f}, Brightness {1-pathway_balance:.3f}")
        
        if 0.2 <= pathway_balance <= 0.8:
            dprint("   ✅ Reasonably balanced gradient flow between pathways")
        else:
            dprint("   ⚠️ Warning: Imbalanced gradient flow between pathways")
    
    dprint("\n3. Testing Input Gradient Flow (End-to-End):")
    
    # Verify gradients flow all the way back to inputs
    if color_data.grad is not None:
        color_input_grad_norm = color_data.grad.norm().item()
        dprint(f"   Color input gradient norm: {color_input_grad_norm:.8f}")
        if color_input_grad_norm > 1e-8:
            dprint("   ✅ Gradients successfully flow back to color inputs")
        else:
            dprint("   ❌ Gradients not reaching color inputs")
    else:
        dprint("   ❌ No gradients computed for color inputs")
    
    if brightness_data.grad is not None:
        brightness_input_grad_norm = brightness_data.grad.norm().item()
        dprint(f"   Brightness input gradient norm: {brightness_input_grad_norm:.8f}")
        if brightness_input_grad_norm > 1e-8:
            dprint("   ✅ Gradients successfully flow back to brightness inputs")
        else:
            dprint("   ❌ Gradients not reaching brightness inputs")
    else:
        dprint("   ❌ No gradients computed for brightness inputs")
    
    dprint("\n4. Architectural Design Compliance Summary:")
    
    # Overall assessment
    forward_pass_ok = (color_norm > 1e-4 and brightness_norm > 1e-4 and 
                      color_vs_combined > 1e-4 and brightness_vs_combined > 1e-4)
    
    backward_pass_ok = (color_grad_sum > 1e-6 and brightness_grad_sum > 1e-6 and 
                       classifier_grad_sum > 1e-6)
    
    input_gradients_ok = (color_data.grad is not None and brightness_data.grad is not None and
                         color_data.grad.norm().item() > 1e-8 and brightness_data.grad.norm().item() > 1e-8)
    
    if forward_pass_ok:
        dprint("   ✅ Forward pass: Pathway separation and integration working correctly")
    else:
        dprint("   ❌ Forward pass: Issues with pathway separation or integration")
    
    if backward_pass_ok:
        dprint("   ✅ Backward pass: Gradients flowing through all pathways and classifier")
    else:
        dprint("   ❌ Backward pass: Missing gradient flow to one or more components")
        
    if input_gradients_ok:
        dprint("   ✅ End-to-end gradient flow: Loss gradients reach input tensors")
    else:
        dprint("   ❌ End-to-end gradient flow: Gradients not reaching inputs")
    
    overall_compliance = forward_pass_ok and backward_pass_ok and input_gradients_ok
    
    if overall_compliance:
        dprint("\n   🎉 ARCHITECTURAL VERIFICATION PASSED: Model data flow matches intended design!")
    else:
        dprint("\n   ⚠️ ARCHITECTURAL VERIFICATION ISSUES: Model may not match intended design")
    
    return {
        'forward_pass_ok': forward_pass_ok,
        'backward_pass_ok': backward_pass_ok, 
        'input_gradients_ok': input_gradients_ok,
        'overall_compliance': overall_compliance,
        'pathway_stats': {
            'color_output_norm': color_norm,
            'brightness_output_norm': brightness_norm,
            'combined_output_norm': combined_norm,
            'color_grad_sum': color_grad_sum,
            'brightness_grad_sum': brightness_grad_sum,
            'integration_grad_sum': integration_grad_sum,
            'classifier_grad_sum': classifier_grad_sum
        }
    }


def run_diagnostics_on_model(model, model_name, train_loader, test_loader, device, epochs=2):
    """Run full diagnostic suite on a specific model"""
    
    dprint(f"\n{'='*60}")
    dprint(f"RUNNING DIAGNOSTICS ON: {model_name}")
    dprint(f"{'='*60}")
    
    # Create a safe filename from model name
    safe_name = model_name.lower().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
    
    # Run diagnostics in logical order
    analyze_model_architecture(model)
    analyze_weight_initialization(model)
    analyze_parameters(model, safe_name)
    analyze_activations(model, test_loader, device)
    analyze_input_gradients(model, train_loader, device)
    run_gradient_flow_test(model, train_loader, device, safe_name)
    
    # Test architectural compliance with initialized model
    verify_architectural_data_flow(model, train_loader, device, f"{model_name} - Pre-Training")
    
    analyze_gradient_norms_during_training(model, train_loader, device, num_batches=3)
    run_mini_train(model, train_loader, test_loader, device, epochs=epochs, model_name=safe_name)
    
    # Test architectural compliance after mini training to ensure training doesn't break architecture
    verify_architectural_data_flow(model, train_loader, device, f"{model_name} - Post-Training")
    
    dprint(f"\n{'='*60}")
    dprint(f"COMPLETED DIAGNOSTICS ON: {model_name}")
    dprint(f"{'='*60}")


def main():
    global _dual_output
    
    # Set up dual output logging
    dual_output, log_path = setup_logging()
    _dual_output = dual_output
    
    try:
        # Log script start time and info
        dprint("="*80)
        dprint(f"MULTI-CHANNEL RESNET DIAGNOSTICS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        dprint("="*80)
        dprint(f"Log file: {log_path}")
        dprint("="*80)
        
        # Set up device
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        dprint(f"Using device: {device}")
        
        # Load data once for both models
        train_loader, test_loader, _ = load_data(batch_size=16)  # Small batch size for diagnostics
        
        dprint("\n" + "="*80)
        dprint("MULTI-CHANNEL RESNET ARCHITECTURE COMPARISON DIAGNOSTICS")
        dprint("="*80)
        dprint("Testing both ResNet-18 and ResNet-50 style architectures")
        dprint("="*80)
        
        # Test 1: ResNet-18 style model (basic blocks)
        dprint(f"\n{'#'*60}")
        dprint("# TESTING RESNET-18 STYLE MODEL (Basic Blocks)")
        dprint(f"{'#'*60}")
        
        resnet18_model = load_full_resnet_model(device)
        resnet18_model = resnet18_model.to(device)
        
        run_diagnostics_on_model(
            model=resnet18_model,
            model_name="ResNet-18 Style (Basic Blocks)",
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            epochs=2  # 2 epochs for smaller model to show learning trends
        )
        
        # Clean up memory
        del resnet18_model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        elif device.type == 'mps':
            torch.mps.empty_cache()
        
        # Test 2: ResNet-50 style model (bottleneck blocks)
        dprint(f"\n{'#'*60}")
        dprint("# TESTING RESNET-50 STYLE MODEL (Bottleneck Blocks)")
        dprint(f"{'#'*60}")
        
        resnet50_model = load_full_resnet50_model(device)
        resnet50_model = resnet50_model.to(device)
        
        run_diagnostics_on_model(
            model=resnet50_model,
            model_name="ResNet-50 Style (Bottleneck Blocks)",
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            epochs=1  # Reduced training for larger model
        )
        
        # Clean up memory
        del resnet50_model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        elif device.type == 'mps':
            torch.mps.empty_cache()
        
        dprint("\n" + "="*80)
        dprint("ALL DIAGNOSTICS COMPLETE")
        dprint("="*80)
        dprint("Results saved to 'results' directory.")
        dprint("Both ResNet-18 and ResNet-50 style architectures have been tested.")
        dprint(f"Full diagnostic log saved to: {log_path}")
        dprint("="*80)
        
    finally:
        # Clean up dual output
        if _dual_output:
            _dual_output.close()
            _dual_output = None


if __name__ == "__main__":
    main()
