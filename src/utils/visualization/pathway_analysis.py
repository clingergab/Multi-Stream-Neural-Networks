"""Pathway analysis visualization utilities."""

import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path


def visualize_pathway_activations(model,
                                 color_input: torch.Tensor,
                                 brightness_input: torch.Tensor,
                                 layer_names: Optional[List[str]] = None,
                                 save_path: Optional[str] = None) -> None:
    """Visualize activations in different pathways."""
    
    activations = {}
    hooks = []
    
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    
    # Register hooks
    if layer_names is None:
        layer_names = []
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                layer_names.append(name)
    
    for name in layer_names:
        for module_name, module in model.named_modules():
            if module_name == name:
                hook = module.register_forward_hook(get_activation(name))
                hooks.append(hook)
    
    # Forward pass
    with torch.no_grad():
        model.eval()
        output = model(color_input, brightness_input)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Visualize activations
    num_layers = len(activations)
    if num_layers == 0:
        print("No activations captured")
        return
    
    fig, axes = plt.subplots(2, (num_layers + 1) // 2, figsize=(15, 8))
    if num_layers == 1:
        axes = [axes]
    elif num_layers <= 2:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i, (layer_name, activation) in enumerate(activations.items()):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Handle different activation shapes
        if len(activation.shape) == 4:  # Conv output [B, C, H, W]
            # Show mean activation across channels
            mean_activation = activation.mean(dim=1).squeeze().cpu().numpy()
            im = ax.imshow(mean_activation, cmap='viridis')
            plt.colorbar(im, ax=ax)
        elif len(activation.shape) == 2:  # Linear output [B, F]
            # Show activation histogram
            activation_flat = activation.flatten().cpu().numpy()
            ax.hist(activation_flat, bins=50, alpha=0.7)
            ax.set_ylabel('Frequency')
        
        ax.set_title(f'{layer_name}')
    
    # Hide unused subplots
    for i in range(len(activations), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Pathway activations saved to: {save_path}")
    
    plt.show()


def plot_pathway_comparison(results: Dict[str, Dict[str, float]],
                           metric: str = 'accuracy',
                           save_path: Optional[str] = None) -> None:
    """Plot comparison between different pathway configurations."""
    
    pathways = list(results.keys())
    values = [results[pathway][metric] for pathway in pathways]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(pathways, values, alpha=0.7, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'][:len(pathways)])
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.title(f'Pathway Comparison - {metric.title()}')
    plt.xlabel('Pathway Type')
    plt.ylabel(metric.title())
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Pathway comparison saved to: {save_path}")
    
    plt.show()


def analyze_pathway_contributions(model,
                                dataloader,
                                pathway_names: List[str] = ['color', 'brightness'],
                                save_path: Optional[str] = None) -> Dict[str, float]:
    """Analyze the contribution of each pathway to the final prediction."""
    
    contributions = {pathway: [] for pathway in pathway_names}
    
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 10:  # Limit for speed
                break
            
            if len(batch) == 3:
                color_input, brightness_input, targets = batch
            else:
                continue
            
            # Normal prediction
            normal_output = model(color_input, brightness_input)
            normal_pred = torch.softmax(normal_output, dim=1)
            
            # Zero out color pathway
            zero_color = torch.zeros_like(color_input)
            brightness_only = model(zero_color, brightness_input)
            brightness_pred = torch.softmax(brightness_only, dim=1)
            
            # Zero out brightness pathway
            zero_brightness = torch.zeros_like(brightness_input)
            color_only = model(color_input, zero_brightness)
            color_pred = torch.softmax(color_only, dim=1)
            
            # Compute contributions
            color_contrib = torch.norm(normal_pred - brightness_pred, dim=1).mean().item()
            brightness_contrib = torch.norm(normal_pred - color_pred, dim=1).mean().item()
            
            contributions['color'].append(color_contrib)
            contributions['brightness'].append(brightness_contrib)
    
    # Average contributions
    avg_contributions = {pathway: np.mean(contribs) 
                        for pathway, contribs in contributions.items()}
    
    # Normalize
    total = sum(avg_contributions.values())
    if total > 0:
        avg_contributions = {pathway: contrib / total 
                           for pathway, contrib in avg_contributions.items()}
    
    # Visualize
    plt.figure(figsize=(8, 6))
    pathways = list(avg_contributions.keys())
    values = list(avg_contributions.values())
    
    colors = ['lightblue', 'lightcoral', 'lightgreen', 'gold'][:len(pathways)]
    wedges, texts, autotexts = plt.pie(values, labels=pathways, colors=colors, 
                                      autopct='%1.1f%%', startangle=90)
    
    plt.title('Pathway Contributions to Prediction')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Pathway contributions saved to: {save_path}")
    
    plt.show()
    
    return avg_contributions


def create_pathway_summary_plot(pathway_analysis: Dict[str, Any],
                               save_path: Optional[str] = None) -> None:
    """Create a comprehensive pathway analysis summary."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Pathway Analysis Summary', fontsize=16)
    
    # Pathway importance over time
    if 'importance_history' in pathway_analysis:
        ax = axes[0, 0]
        importance_history = pathway_analysis['importance_history']
        epochs = list(range(len(importance_history)))
        
        for pathway in importance_history[0].keys():
            values = [entry[pathway] for entry in importance_history]
            ax.plot(epochs, values, label=pathway, marker='o', markersize=3)
        
        ax.set_title('Importance Evolution')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Importance')
        ax.legend()
        ax.grid(True)
    
    # Current pathway contributions
    if 'contributions' in pathway_analysis:
        ax = axes[0, 1]
        contributions = pathway_analysis['contributions']
        pathways = list(contributions.keys())
        values = list(contributions.values())
        
        bars = ax.bar(pathways, values, alpha=0.7)
        ax.set_title('Current Contributions')
        ax.set_ylabel('Contribution')
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}', ha='center', va='bottom')
    
    # Mixing weights distribution
    if 'mixing_weights' in pathway_analysis:
        ax = axes[1, 0]
        weights = pathway_analysis['mixing_weights']
        
        if isinstance(weights, dict):
            weight_names = list(weights.keys())
            weight_values = [w.item() if torch.is_tensor(w) else w for w in weights.values()]
            
            bars = ax.bar(weight_names, weight_values, alpha=0.7, 
                         color=['red', 'green', 'purple'][:len(weight_names)])
            ax.set_title('Mixing Weights')
            ax.set_ylabel('Weight Value')
            
            # Add value labels
            for bar, value in zip(bars, weight_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom')
    
    # Performance comparison
    if 'performance' in pathway_analysis:
        ax = axes[1, 1]
        performance = pathway_analysis['performance']
        
        metrics = list(performance.keys())
        values = list(performance.values())
        
        bars = ax.bar(metrics, values, alpha=0.7, color='skyblue')
        ax.set_title('Performance Metrics')
        ax.set_ylabel('Value')
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Pathway summary saved to: {save_path}")
    
    plt.show()


def plot_pathway_correlations(activations: Dict[str, torch.Tensor],
                             save_path: Optional[str] = None) -> None:
    """Plot correlations between pathway activations."""
    
    # Flatten activations
    flat_activations = {}
    for pathway, activation in activations.items():
        flat_activations[pathway] = activation.flatten().cpu().numpy()
    
    # Compute correlation matrix
    pathways = list(flat_activations.keys())
    correlation_matrix = np.zeros((len(pathways), len(pathways)))
    
    for i, pathway1 in enumerate(pathways):
        for j, pathway2 in enumerate(pathways):
            correlation = np.corrcoef(flat_activations[pathway1], 
                                    flat_activations[pathway2])[0, 1]
            correlation_matrix[i, j] = correlation
    
    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, 
                xticklabels=pathways, 
                yticklabels=pathways,
                annot=True, 
                cmap='coolwarm', 
                center=0,
                square=True)
    
    plt.title('Pathway Activation Correlations')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Pathway correlations saved to: {save_path}")
    
    plt.show()


def visualize_pathway_features(model, 
                              color_input: torch.Tensor,
                              brightness_input: torch.Tensor,
                              layer_name: str,
                              num_features: int = 16,
                              save_path: Optional[str] = None) -> None:
    """Visualize individual features from a specific layer."""
    
    activation = None
    
    def get_activation(name):
        def hook(model, input, output):
            nonlocal activation
            activation = output.detach()
        return hook
    
    # Register hook
    hook_registered = False
    for module_name, module in model.named_modules():
        if module_name == layer_name:
            hook = module.register_forward_hook(get_activation(layer_name))
            hook_registered = True
            break
    
    if not hook_registered:
        print(f"Layer {layer_name} not found")
        return
    
    # Forward pass
    with torch.no_grad():
        model.eval()
        output = model(color_input, brightness_input)
    
    # Remove hook
    hook.remove()
    
    if activation is None:
        print("No activation captured")
        return
    
    # Visualize features
    if len(activation.shape) == 4:  # Conv features [B, C, H, W]
        batch_size, channels, height, width = activation.shape
        num_features = min(num_features, channels)
        
        cols = int(np.ceil(np.sqrt(num_features)))
        rows = int(np.ceil(num_features / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
        fig.suptitle(f'Features from {layer_name}')
        
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i in range(num_features):
            feature_map = activation[0, i].cpu().numpy()  # First batch
            ax = axes[i] if num_features > 1 else axes
            
            im = ax.imshow(feature_map, cmap='viridis')
            ax.set_title(f'Feature {i}')
            ax.axis('off')
        
        # Hide unused subplots
        for i in range(num_features, len(axes)):
            axes[i].set_visible(False)
    
    else:
        print(f"Cannot visualize features for activation shape: {activation.shape}")
        return
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Pathway features saved to: {save_path}")
    
    plt.show()
