"""Training visualization utilities."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path


def plot_training_curves(metrics: Dict[str, List[float]], 
                        save_path: Optional[str] = None,
                        title: str = "Training Progress") -> None:
    """Plot training curves for various metrics."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=16)
    
    # Loss curves
    if 'train_loss' in metrics or 'val_loss' in metrics:
        ax = axes[0, 0]
        if 'train_loss' in metrics:
            ax.plot(metrics['train_loss'], label='Train Loss', color='blue')
        if 'val_loss' in metrics:
            ax.plot(metrics['val_loss'], label='Val Loss', color='red')
        ax.set_title('Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)
    
    # Accuracy curves
    if 'train_acc' in metrics or 'val_acc' in metrics:
        ax = axes[0, 1]
        if 'train_acc' in metrics:
            ax.plot(metrics['train_acc'], label='Train Acc', color='blue')
        if 'val_acc' in metrics:
            ax.plot(metrics['val_acc'], label='Val Acc', color='red')
        ax.set_title('Accuracy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.legend()
        ax.grid(True)
    
    # Learning rate
    if 'learning_rate' in metrics:
        ax = axes[1, 0]
        ax.plot(metrics['learning_rate'], color='green')
        ax.set_title('Learning Rate')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('LR')
        ax.grid(True)
    
    # Additional metrics
    additional_metrics = [k for k in metrics.keys() 
                         if k not in ['train_loss', 'val_loss', 'train_acc', 'val_acc', 'learning_rate']]
    
    if additional_metrics:
        ax = axes[1, 1]
        for metric in additional_metrics[:3]:  # Plot up to 3 additional metrics
            ax.plot(metrics[metric], label=metric)
        ax.set_title('Additional Metrics')
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")
    
    plt.show()


def plot_pathway_importance(importance_history: List[Dict[str, float]],
                           save_path: Optional[str] = None) -> None:
    """Plot pathway importance evolution over training."""
    
    if not importance_history:
        print("No pathway importance data available")
        return
    
    # Extract data
    epochs = list(range(len(importance_history)))
    pathways = list(importance_history[0].keys())
    
    plt.figure(figsize=(10, 6))
    
    for pathway in pathways:
        values = [entry[pathway] for entry in importance_history]
        plt.plot(epochs, values, label=pathway, linewidth=2, marker='o', markersize=4)
    
    plt.title('Pathway Importance Evolution', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Relative Importance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Pathway importance plot saved to: {save_path}")
    
    plt.show()


def plot_mixing_weights_evolution(weight_history: Dict[str, List[float]],
                                 save_path: Optional[str] = None) -> None:
    """Plot evolution of mixing weights during training."""
    
    if not weight_history:
        print("No mixing weights data available")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Mixing Weights Evolution', fontsize=16)
    
    # Alpha (color pathway weight)
    if 'alpha' in weight_history:
        axes[0, 0].plot(weight_history['alpha'], 'r-', linewidth=2)
        axes[0, 0].set_title('α (Color Weight)')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('α Value')
        axes[0, 0].grid(True)
    
    # Beta (brightness pathway weight)
    if 'beta' in weight_history:
        axes[0, 1].plot(weight_history['beta'], 'g-', linewidth=2)
        axes[0, 1].set_title('β (Brightness Weight)')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('β Value')
        axes[0, 1].grid(True)
    
    # Gamma (interaction weight)
    if 'gamma' in weight_history:
        axes[1, 0].plot(weight_history['gamma'], 'purple', linewidth=2)
        axes[1, 0].set_title('γ (Interaction Weight)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('γ Value')
        axes[1, 0].grid(True)
    
    # Pathway balance
    if 'alpha' in weight_history and 'beta' in weight_history:
        alpha_vals = np.array(weight_history['alpha'])
        beta_vals = np.array(weight_history['beta'])
        total = alpha_vals + beta_vals
        color_ratio = alpha_vals / total
        brightness_ratio = beta_vals / total
        
        axes[1, 1].plot(color_ratio, 'r-', label='Color', linewidth=2)
        axes[1, 1].plot(brightness_ratio, 'g-', label='Brightness', linewidth=2)
        axes[1, 1].set_title('Pathway Balance')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Relative Weight')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        axes[1, 1].set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Mixing weights plot saved to: {save_path}")
    
    plt.show()


def plot_loss_landscape(model, data_loader, 
                       param_names: List[str] = ['alpha', 'beta'],
                       param_ranges: Dict[str, Tuple[float, float]] = None,
                       resolution: int = 50,
                       save_path: Optional[str] = None) -> None:
    """Plot loss landscape around current parameters."""
    
    if param_ranges is None:
        param_ranges = {
            'alpha': (0.1, 2.0),
            'beta': (0.1, 2.0)
        }
    
    if len(param_names) != 2:
        print("Loss landscape visualization requires exactly 2 parameters")
        return
    
    param1_name, param2_name = param_names
    param1_range = param_ranges[param1_name]
    param2_range = param_ranges[param2_name]
    
    # Get original parameters
    original_params = {}
    param_objects = {}
    
    for name, param in model.named_parameters():
        if param1_name in name or param2_name in name:
            original_params[name] = param.data.clone()
            param_objects[name] = param
    
    # Create parameter grids
    param1_vals = np.linspace(param1_range[0], param1_range[1], resolution)
    param2_vals = np.linspace(param2_range[0], param2_range[1], resolution)
    P1, P2 = np.meshgrid(param1_vals, param2_vals)
    
    losses = np.zeros_like(P1)
    
    model.eval()
    with torch.no_grad():
        for i in range(resolution):
            for j in range(resolution):
                # Set parameters
                for name, param in param_objects.items():
                    if param1_name in name:
                        param.data.fill_(P1[i, j])
                    elif param2_name in name:
                        param.data.fill_(P2[i, j])
                
                # Compute loss
                total_loss = 0.0
                num_batches = 0
                
                for batch in data_loader:
                    if len(batch) == 2:
                        inputs, targets = batch
                        outputs = model(inputs)
                    else:
                        # Multi-input model
                        color_input, brightness_input, targets = batch
                        outputs = model(color_input, brightness_input)
                    
                    loss = torch.nn.functional.cross_entropy(outputs, targets)
                    total_loss += loss.item()
                    num_batches += 1
                    
                    if num_batches >= 5:  # Limit batches for speed
                        break
                
                losses[i, j] = total_loss / num_batches
    
    # Restore original parameters
    for name, param in param_objects.items():
        param.data.copy_(original_params[name])
    
    # Plot
    plt.figure(figsize=(10, 8))
    contour = plt.contour(P1, P2, losses, levels=20)
    plt.contourf(P1, P2, losses, levels=20, alpha=0.6, cmap='viridis')
    plt.colorbar(label='Loss')
    
    # Mark current position
    current_p1 = original_params[param1_name].item()
    current_p2 = original_params[param2_name].item()
    plt.plot(current_p1, current_p2, 'r*', markersize=15, label='Current')
    
    plt.xlabel(param1_name)
    plt.ylabel(param2_name)
    plt.title('Loss Landscape')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Loss landscape saved to: {save_path}")
    
    plt.show()


def plot_gradient_flow(model, save_path: Optional[str] = None) -> None:
    """Plot gradient flow through the model."""
    
    ave_grads = []
    max_grads = []
    layers = []
    
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            layers.append(name)
            ave_grads.append(param.grad.abs().mean().item())
            max_grads.append(param.grad.abs().max().item())
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Average gradients
    ax1.bar(range(len(ave_grads)), ave_grads, alpha=0.7, color='blue')
    ax1.set_title('Average Gradient Magnitude')
    ax1.set_xlabel('Layers')
    ax1.set_ylabel('Average Gradient')
    ax1.set_xticks(range(len(layers)))
    ax1.set_xticklabels(layers, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Max gradients
    ax2.bar(range(len(max_grads)), max_grads, alpha=0.7, color='red')
    ax2.set_title('Maximum Gradient Magnitude')
    ax2.set_xlabel('Layers')
    ax2.set_ylabel('Max Gradient')
    ax2.set_xticks(range(len(layers)))
    ax2.set_xticklabels(layers, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Gradient flow plot saved to: {save_path}")
    
    plt.show()


def create_training_summary(metrics: Dict[str, List[float]],
                           weight_history: Dict[str, List[float]],
                           importance_history: List[Dict[str, float]],
                           save_dir: Optional[str] = None) -> None:
    """Create a comprehensive training summary plot."""
    
    fig = plt.figure(figsize=(16, 12))
    
    # Training curves
    ax1 = plt.subplot(3, 3, 1)
    if 'train_loss' in metrics:
        ax1.plot(metrics['train_loss'], 'b-', label='Train')
    if 'val_loss' in metrics:
        ax1.plot(metrics['val_loss'], 'r-', label='Val')
    ax1.set_title('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2 = plt.subplot(3, 3, 2)
    if 'train_acc' in metrics:
        ax2.plot(metrics['train_acc'], 'b-', label='Train')
    if 'val_acc' in metrics:
        ax2.plot(metrics['val_acc'], 'r-', label='Val')
    ax2.set_title('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    # Mixing weights
    ax3 = plt.subplot(3, 3, 3)
    if 'alpha' in weight_history:
        ax3.plot(weight_history['alpha'], 'r-', label='α')
    if 'beta' in weight_history:
        ax3.plot(weight_history['beta'], 'g-', label='β')
    if 'gamma' in weight_history:
        ax3.plot(weight_history['gamma'], 'purple', label='γ')
    ax3.set_title('Mixing Weights')
    ax3.legend()
    ax3.grid(True)
    
    # Pathway importance
    if importance_history:
        ax4 = plt.subplot(3, 3, 4)
        epochs = list(range(len(importance_history)))
        for pathway in importance_history[0].keys():
            values = [entry[pathway] for entry in importance_history]
            ax4.plot(epochs, values, label=pathway, marker='o', markersize=3)
        ax4.set_title('Pathway Importance')
        ax4.legend()
        ax4.grid(True)
    
    # Additional analysis plots can be added here
    
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / 'training_summary.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training summary saved to: {save_path}")
    
    plt.show()
