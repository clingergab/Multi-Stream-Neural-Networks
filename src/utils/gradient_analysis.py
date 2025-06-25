"""Gradient analysis utilities for Multi-Stream Neural Networks."""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


class GradientAnalyzer:
    """Analyze gradients in multi-stream neural networks."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.gradient_history: Dict[str, List[float]] = defaultdict(list)
        self.hooks = []
        
    def register_hooks(self):
        """Register hooks to capture gradients during backpropagation."""
        def make_hook(name):
            def hook(grad):
                if grad is not None:
                    grad_norm = grad.norm().item()
                    self.gradient_history[name].append(grad_norm)
                return grad
            return hook
        
        # Register hooks on model parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                hook = param.register_hook(make_hook(name))
                self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def analyze_pathway_gradients(self, 
                                color_input: torch.Tensor, 
                                brightness_input: torch.Tensor,
                                target: torch.Tensor) -> Dict[str, float]:
        """Analyze gradient flow through different pathways."""
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(color_input, brightness_input)
        loss = nn.functional.cross_entropy(output, target)
        
        # Backward pass
        loss.backward()
        
        # Analyze gradients
        pathway_grads = {
            'color_pathway': 0.0,
            'brightness_pathway': 0.0,
            'integration': 0.0
        }
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                
                # Categorize gradients by pathway
                if 'color' in name.lower():
                    pathway_grads['color_pathway'] += grad_norm
                elif 'brightness' in name.lower():
                    pathway_grads['brightness_pathway'] += grad_norm
                elif any(x in name.lower() for x in ['alpha', 'beta', 'gamma', 'integration']):
                    pathway_grads['integration'] += grad_norm
        
        return pathway_grads
    
    def compute_gradient_balance(self) -> Dict[str, float]:
        """Compute gradient balance across pathways."""
        if not self.gradient_history:
            return {}
        
        balance = {}
        total_grads = {}
        
        # Group gradients by pathway
        for param_name, grad_norms in self.gradient_history.items():
            if not grad_norms:
                continue
                
            avg_grad = np.mean(grad_norms[-10:])  # Last 10 values
            
            if 'color' in param_name.lower():
                total_grads['color'] = total_grads.get('color', 0) + avg_grad
            elif 'brightness' in param_name.lower():
                total_grads['brightness'] = total_grads.get('brightness', 0) + avg_grad
            elif any(x in param_name.lower() for x in ['alpha', 'beta', 'gamma']):
                total_grads['integration'] = total_grads.get('integration', 0) + avg_grad
        
        # Compute balance ratios
        total = sum(total_grads.values())
        if total > 0:
            for pathway, grad_sum in total_grads.items():
                balance[f'{pathway}_ratio'] = grad_sum / total
        
        return balance
    
    def plot_gradient_evolution(self, save_path: Optional[str] = None):
        """Plot gradient evolution over training."""
        if not self.gradient_history:
            print("No gradient history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Group parameters by type
        pathway_grads = {
            'Color Pathway': [],
            'Brightness Pathway': [],
            'Integration': [],
            'Other': []
        }
        
        for param_name, grad_norms in self.gradient_history.items():
            if 'color' in param_name.lower():
                pathway_grads['Color Pathway'].extend(grad_norms)
            elif 'brightness' in param_name.lower():
                pathway_grads['Brightness Pathway'].extend(grad_norms)
            elif any(x in param_name.lower() for x in ['alpha', 'beta', 'gamma']):
                pathway_grads['Integration'].extend(grad_norms)
            else:
                pathway_grads['Other'].extend(grad_norms)
        
        # Plot gradient magnitudes
        for i, (pathway, grads) in enumerate(pathway_grads.items()):
            if grads:
                ax = axes[i//2, i%2]
                ax.plot(grads, label=pathway)
                ax.set_title(f'{pathway} Gradients')
                ax.set_xlabel('Step')
                ax.set_ylabel('Gradient Norm')
                ax.grid(True)
                ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def detect_gradient_problems(self) -> Dict[str, List[str]]:
        """Detect common gradient problems."""
        problems = {
            'vanishing': [],
            'exploding': [],
            'dead_neurons': []
        }
        
        for param_name, grad_norms in self.gradient_history.items():
            if not grad_norms:
                continue
            
            recent_grads = grad_norms[-10:]  # Last 10 gradients
            
            # Check for vanishing gradients (very small)
            if np.mean(recent_grads) < 1e-7:
                problems['vanishing'].append(param_name)
            
            # Check for exploding gradients (very large)
            if np.max(recent_grads) > 100:
                problems['exploding'].append(param_name)
            
            # Check for dead neurons (consistently zero)
            if len(recent_grads) > 5 and all(g < 1e-8 for g in recent_grads):
                problems['dead_neurons'].append(param_name)
        
        return problems


def analyze_pathway_gradients(model: nn.Module, 
                            color_input: torch.Tensor,
                            brightness_input: torch.Tensor,
                            target: torch.Tensor) -> Dict[str, float]:
    """Quick gradient analysis for pathway balance."""
    analyzer = GradientAnalyzer(model)
    return analyzer.analyze_pathway_gradients(color_input, brightness_input, target)


def compare_gradient_flows(models: Dict[str, nn.Module],
                          color_input: torch.Tensor,
                          brightness_input: torch.Tensor,
                          target: torch.Tensor) -> Dict[str, Dict[str, float]]:
    """Compare gradient flows across different models."""
    results = {}
    
    for model_name, model in models.items():
        results[model_name] = analyze_pathway_gradients(
            model, color_input, brightness_input, target
        )
    
    return results
