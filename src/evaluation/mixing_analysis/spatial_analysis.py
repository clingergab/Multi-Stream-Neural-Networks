"""Spatial mixing analysis."""

import torch
import numpy as np
import matplotlib.pyplot as plt


class SpatialMixingAnalyzer:
    """Analyze spatial attention patterns in mixing."""
    
    def __init__(self, model):
        self.model = model
    
    def analyze_spatial_patterns(self, sample_inputs=None):
        """Analyze spatial attention patterns."""
        if sample_inputs is None:
            # Create sample inputs
            color_input = torch.randn(4, 3, 32, 32)
            brightness_input = torch.randn(4, 1, 32, 32)
        else:
            color_input, brightness_input = sample_inputs
        
        if torch.cuda.is_available():
            color_input = color_input.cuda()
            brightness_input = brightness_input.cuda()
        
        self.model.eval()
        attention_maps = {}
        
        with torch.no_grad():
            if hasattr(self.model, 'alpha_attention'):
                # For spatial adaptive mixing models
                color_features = self.model.color_pathway(color_input)
                brightness_features = self.model.brightness_pathway(brightness_input)
                concat_features = torch.cat([color_features, brightness_features], dim=1)
                
                alpha_map = self.model.alpha_attention(concat_features)
                beta_map = self.model.beta_attention(concat_features)
                gamma_map = self.model.gamma_attention(concat_features)
                
                attention_maps = {
                    'alpha': alpha_map.cpu().numpy(),
                    'beta': beta_map.cpu().numpy(),
                    'gamma': gamma_map.cpu().numpy()
                }
        
        return self._analyze_attention_statistics(attention_maps)
    
    def _analyze_attention_statistics(self, attention_maps):
        """Compute statistics on attention maps."""
        analysis = {}
        
        for name, maps in attention_maps.items():
            if maps.size > 0:
                analysis[name] = {
                    'mean_attention': np.mean(maps),
                    'std_attention': np.std(maps),
                    'max_attention': np.max(maps),
                    'min_attention': np.min(maps),
                    'spatial_variance': np.var(maps, axis=(2, 3)).mean(),
                    'center_bias': self._compute_center_bias(maps)
                }
        
        return analysis
    
    def _compute_center_bias(self, attention_maps):
        """Compute bias towards center of spatial attention."""
        if len(attention_maps.shape) < 4:
            return 0.0
        
        batch_size, channels, height, width = attention_maps.shape
        center_h, center_w = height // 2, width // 2
        
        # Create distance matrix from center
        y, x = np.meshgrid(range(height), range(width), indexing='ij')
        distances = np.sqrt((y - center_h)**2 + (x - center_w)**2)
        
        # Compute correlation between attention and distance from center
        correlations = []
        for b in range(batch_size):
            for c in range(channels):
                attention_flat = attention_maps[b, c].flatten()
                distance_flat = distances.flatten()
                correlation = np.corrcoef(attention_flat, distance_flat)[0, 1]
                if not np.isnan(correlation):
                    correlations.append(correlation)
        
        return np.mean(correlations) if correlations else 0.0
    
    def visualize_attention_patterns(self, attention_maps, save_path=None):
        """Visualize spatial attention patterns."""
        if not attention_maps:
            return
        
        n_maps = len(attention_maps)
        fig, axes = plt.subplots(1, n_maps, figsize=(5*n_maps, 4))
        if n_maps == 1:
            axes = [axes]
        
        for i, (name, maps) in enumerate(attention_maps.items()):
            # Average across batch and channels
            avg_map = np.mean(maps, axis=(0, 1))
            
            im = axes[i].imshow(avg_map, cmap='viridis')
            axes[i].set_title(f'{name.capitalize()} Attention')
            axes[i].set_xlabel('Width')
            axes[i].set_ylabel('Height')
            plt.colorbar(im, ax=axes[i])
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()