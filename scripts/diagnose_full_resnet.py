"""
Diagnostic script to analyze full-sized multi-channel ResNet network.

This script runs comprehensive diagnostics on the full-sized (non-reduced) 
MultiChannelResNetNetwork architecture to evaluate gradient flow, dead neurons, 
parameter updates, and other training factors.
"""

import os
import sys
import torch
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
from src.utils.cifar100_loader import get_cifar100_datasets
from src.transforms.rgb_to_rgbl import RGBtoRGBL
from src.utils.debug_utils import (
    analyze_gradient_flow, 
    analyze_parameter_magnitudes,
    add_diagnostic_hooks
)


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
    
    print("Creating full-sized MultiChannelResNetNetwork (reduce_architecture=False)...")
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
    print(f"Full-sized model created with {params:,} parameters")
    print(f"Device: {model.device}")
    
    return model


def load_data(batch_size=32):
    """Load CIFAR-100 dataset and prepare data loaders"""
    
    data_dir = 'data/cifar-100'
    print(f"Loading CIFAR-100 from: {data_dir}")
    
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
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        rgbl_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"Created data loaders with batch size {batch_size}")
    print(f"Train samples: {len(rgbl_train_dataset)}, Test samples: {len(rgbl_test_dataset)}")
    
    return train_loader, test_loader, class_names


def run_gradient_flow_test(model, data_loader, device):
    """Test gradient flow through the model during backpropagation"""
    
    print("\n=== Gradient Flow Test ===")
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
    
    print(f"Mini-batch loss: {loss.item():.4f}, accuracy: {accuracy:.2f}%")
    print("Gradient statistics:")
    print(f"  Largest gradient: {max_grad:.4e} (in {max_grad_layer})")
    print(f"  Smallest gradient: {min_grad:.4e} (in {min_grad_layer})")
    print(f"  Largest/smallest ratio: {grad_ratio:.4e}")
    
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
    
    print("\nPathway gradient check:")
    print(f"  Color pathway: {'✅ Active' if color_has_grad else '❌ Inactive'}")
    print(f"  Brightness pathway: {'✅ Active' if brightness_has_grad else '❌ Inactive'}")
    
    # Visualize gradient flow
    # Save plot to results directory
    os.makedirs('results', exist_ok=True)
    analyze_gradient_flow(model, save_path='results/full_resnet_gradient_flow.png')
    print("Gradient flow plot saved to results/full_resnet_gradient_flow.png")
    

def analyze_activations(model, data_loader, device):
    """Analyze activations throughout the model"""
    
    print("\n=== Activation Analysis ===")
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
    print("\nActivation statistics:")
    for layer_name, stats in activation_stats.items():
        print(f"  {layer_name}:")
        print(f"    Mean: {stats['mean']:.4f}, StdDev: {stats['std']:.4f}")
        print(f"    Min: {stats['min']:.4f}, Max: {stats['max']:.4f}")
        print(f"    Zero %: {stats['zero_percent']:.2f}%")
    
    # Check for possible dead neurons (high percentage of zeros)
    dead_layers = {}
    for layer_name, stats in activation_stats.items():
        if stats['zero_percent'] > 20:  # More than 20% zeros
            dead_layers[layer_name] = stats['zero_percent']
            
    if dead_layers:
        print("\n⚠️ Warning: Potential dead neurons detected:")
        for layer_name, percent in dead_layers.items():
            print(f"  {layer_name}: {percent:.2f}% zeros")
    else:
        print("\n✅ No dead neurons detected (all layers < 20% zeros)")


def run_mini_train(model, train_loader, test_loader, device, epochs=2):
    """Run a mini training loop to test model learning capability"""
    
    print("\n=== Mini Training Test ===")
    
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
    plt.savefig('results/full_resnet_learning_curves.png')
    print("Learning curves saved to results/full_resnet_learning_curves.png")
    
    # Analyze final metrics
    final_train_acc = history['train_accuracy'][-1] * 100
    final_val_acc = history['val_accuracy'][-1] * 100
    final_train_loss = history['train_loss'][-1]
    final_val_loss = history['val_loss'][-1]
    
    print(f"\nFinal training accuracy: {final_train_acc:.2f}%")
    print(f"Final validation accuracy: {final_val_acc:.2f}%")
    print(f"Final training loss: {final_train_loss:.4f}")
    print(f"Final validation loss: {final_val_loss:.4f}")
    
    # Check for overfitting
    acc_gap = final_train_acc - final_val_acc
    loss_gap = final_val_loss - final_train_loss
    
    if acc_gap > 20 or loss_gap > 1.0:
        print("⚠️ Warning: Model shows signs of overfitting (expected for full architecture)")
    else:
        print("✅ Model is learning without excessive overfitting")


def analyze_parameters(model):
    """Analyze parameter distributions and magnitudes"""
    
    print("\n=== Parameter Analysis ===")
    
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
    print("\nParameter statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Average magnitude: {avg_magnitude:.4f}")
    print(f"  Largest magnitude: {max_magnitude:.4f} (in {max_magnitude_layer})")
    print(f"  Smallest magnitude: {min_magnitude:.4f} (in {min_magnitude_layer})")
    
    # Check for pathway imbalance
    if color_params and brightness_params:
        color_magnitude = sum(color_params) / len(color_params)
        brightness_magnitude = sum(brightness_params) / len(brightness_params)
        
        ratio = color_magnitude / brightness_magnitude
        print("\nPathway magnitude comparison:")
        print(f"  Color pathway: {color_magnitude:.4f}")
        print(f"  Brightness pathway: {brightness_magnitude:.4f}")
        print(f"  Ratio (color/brightness): {ratio:.4f}")
        
        if ratio > 5 or ratio < 0.2:
            print("⚠️ Warning: Significant imbalance between pathways")
        else:
            print("✅ Pathways have reasonably balanced parameter magnitudes")
    
    # Visualize parameter magnitudes
    os.makedirs('results', exist_ok=True)
    analyze_parameter_magnitudes(model, save_path='results/full_resnet_parameter_magnitudes.png')
    print("Parameter magnitude plot saved to results/full_resnet_parameter_magnitudes.png")


def main():
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_full_resnet_model(device)
    model = model.to(device)
    
    # Load data
    train_loader, test_loader, _ = load_data(batch_size=16)  # Small batch size for diagnostics
    
    # Run diagnostics
    analyze_parameters(model)
    analyze_activations(model, test_loader, device)
    run_gradient_flow_test(model, train_loader, device)
    run_mini_train(model, train_loader, test_loader, device, epochs=2)
    
    print("\n=== Diagnostics Complete ===")
    print("Results saved to 'results' directory.")


if __name__ == "__main__":
    main()
