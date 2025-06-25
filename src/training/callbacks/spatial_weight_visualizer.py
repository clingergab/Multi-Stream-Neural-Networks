"""Spatial weight visualization callback."""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


class SpatialWeightVisualizer:
    """Visualize spatial attention weights."""
    
    def __init__(self, save_dir='results/spatial_weights', vis_freq=100):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.vis_freq = vis_freq
    
    def __call__(self, model, metrics):
        """Visualize spatial weights periodically."""
        epoch = metrics.get('epoch', 0)
        
        if epoch % self.vis_freq == 0:
            spatial_weights = self._extract_spatial_weights(model)
            if spatial_weights:
                self._save_visualization(spatial_weights, epoch)
    
    def _extract_spatial_weights(self, model):
        """Extract spatial attention maps from model."""
        # This is model-specific - implement for spatial adaptive mixing
        spatial_weights = {}
        
        # Create dummy input to get attention maps
        dummy_color = torch.randn(1, 3, 32, 32)
        dummy_brightness = torch.randn(1, 1, 32, 32)
        
        if torch.cuda.is_available():
            dummy_color = dummy_color.cuda()
            dummy_brightness = dummy_brightness.cuda()
        
        model.eval()
        with torch.no_grad():
            # For spatial adaptive mixing models
            if hasattr(model, 'alpha_attention'):
                concat_features = torch.cat([dummy_color, dummy_brightness], dim=1)
                alpha_map = model.alpha_attention(concat_features)
                beta_map = model.beta_attention(concat_features)
                gamma_map = model.gamma_attention(concat_features)
                
                spatial_weights = {
                    'alpha': alpha_map.cpu().numpy()[0, 0],
                    'beta': beta_map.cpu().numpy()[0, 0],
                    'gamma': gamma_map.cpu().numpy()[0, 0]
                }
        
        return spatial_weights
    
    def _save_visualization(self, spatial_weights, epoch):
        """Save spatial weight visualizations."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, (name, weights) in enumerate(spatial_weights.items()):
            im = axes[i].imshow(weights, cmap='viridis')
            axes[i].set_title(f'{name.capitalize()} Attention Map')
            axes[i].set_xlabel('Width')
            axes[i].set_ylabel('Height')
            plt.colorbar(im, ax=axes[i])
        
        plt.tight_layout()
        plt.savefig(self.save_dir / f'spatial_weights_epoch_{epoch}.png', dpi=150)
        plt.close()
        
        print(f"Saved spatial weight visualization for epoch {epoch}")