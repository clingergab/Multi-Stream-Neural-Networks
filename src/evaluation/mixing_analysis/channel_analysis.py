"""Channel-wise mixing analysis."""

import torch
import numpy as np
import matplotlib.pyplot as plt


class ChannelMixingAnalyzer:
    """Analyze channel-wise mixing weights and patterns."""
    
    def __init__(self, model):
        self.model = model
    
    def analyze_channel_weights(self):
        """Analyze distribution of channel-wise mixing weights."""
        weights = {}
        
        for name, param in self.model.named_parameters():
            if 'alpha' in name or 'beta' in name or 'gamma' in name:
                weights[name] = param.data.clone()
        
        analysis = {}
        for name, weight_tensor in weights.items():
            if weight_tensor.numel() > 1:  # Channel-wise weights
                analysis[name] = {
                    'mean': weight_tensor.mean().item(),
                    'std': weight_tensor.std().item(),
                    'min': weight_tensor.min().item(),
                    'max': weight_tensor.max().item(),
                    'distribution': weight_tensor.cpu().numpy()
                }
        
        return analysis
    
    def plot_channel_distributions(self, analysis, save_path=None):
        """Plot channel weight distributions."""
        n_weights = len(analysis)
        if n_weights == 0:
            return
        
        fig, axes = plt.subplots(1, n_weights, figsize=(5*n_weights, 4))
        if n_weights == 1:
            axes = [axes]
        
        for i, (name, stats) in enumerate(analysis.items()):
            axes[i].hist(stats['distribution'], bins=20, alpha=0.7)
            axes[i].set_title(f'{name.capitalize()} Distribution')
            axes[i].set_xlabel('Weight Value')
            axes[i].set_ylabel('Frequency')
            axes[i].axvline(stats['mean'], color='red', linestyle='--', label=f'Mean: {stats["mean"]:.3f}')
            axes[i].legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()