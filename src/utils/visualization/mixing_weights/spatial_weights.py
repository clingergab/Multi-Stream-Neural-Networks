"""
Spatial weight visualization utilities for Multi-Stream Neural Networks.

This module provides tools for visualizing spatial mixing weights that apply
different scaling factors based on spatial locations in feature maps.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import List, Optional, Dict, Any
import seaborn as sns


class SpatialWeightVisualizer:
    """Visualizer for spatial mixing weights in MSNNs."""
    
    def __init__(self, figsize: tuple = (12, 8)):
        """
        Initialize the spatial weight visualizer.
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        self.cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
    
    def plot_spatial_weight_maps(self,
                               spatial_weights: torch.Tensor,
                               stream_names: Optional[List[str]] = None,
                               title: str = "Spatial Weight Maps",
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot spatial weight maps for multiple streams.
        
        Args:
            spatial_weights: Tensor of shape (num_streams, height, width)
            stream_names: Names of the streams
            title: Plot title
            save_path: Path to save the figure
            
        Returns:
            The matplotlib figure
        """
        weights_np = spatial_weights.detach().cpu().numpy()
        num_streams, height, width = weights_np.shape
        
        if stream_names is None:
            stream_names = [f"Stream {i+1}" for i in range(num_streams)]
        
        # Calculate subplot grid
        cols = min(3, num_streams)
        rows = (num_streams + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        if num_streams == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        # Plot each stream's spatial weights
        for i in range(num_streams):
            row = i // cols
            col = i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            cmap = self.cmaps[i % len(self.cmaps)]
            im = ax.imshow(weights_np[i], cmap=cmap, aspect='auto')
            
            # Add colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            
            ax.set_title(f'{stream_names[i]}')
            ax.set_xlabel('Width')
            ax.set_ylabel('Height')
        
        # Hide unused subplots
        for i in range(num_streams, rows * cols):
            row = i // cols
            col = i % cols
            if rows > 1:
                axes[row, col].set_visible(False)
            else:
                axes[col].set_visible(False)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_spatial_weight_comparison(self,
                                     spatial_weights: torch.Tensor,
                                     stream_idx1: int,
                                     stream_idx2: int,
                                     stream_names: Optional[List[str]] = None,
                                     title: Optional[str] = None,
                                     save_path: Optional[str] = None) -> plt.Figure:
        """
        Compare spatial weights between two streams.
        
        Args:
            spatial_weights: Tensor of shape (num_streams, height, width)
            stream_idx1: Index of first stream to compare
            stream_idx2: Index of second stream to compare
            stream_names: Names of the streams
            title: Plot title
            save_path: Path to save the figure
            
        Returns:
            The matplotlib figure
        """
        weights_np = spatial_weights.detach().cpu().numpy()
        
        if stream_names is None:
            stream_names = [f"Stream {i+1}" for i in range(weights_np.shape[0])]
        
        if title is None:
            title = f"Spatial Weight Comparison: {stream_names[stream_idx1]} vs {stream_names[stream_idx2]}"
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # Plot individual weight maps
        im1 = axes[0, 0].imshow(weights_np[stream_idx1], cmap='viridis', aspect='auto')
        axes[0, 0].set_title(f'{stream_names[stream_idx1]}')
        plt.colorbar(im1, ax=axes[0, 0])
        
        im2 = axes[0, 1].imshow(weights_np[stream_idx2], cmap='viridis', aspect='auto')
        axes[0, 1].set_title(f'{stream_names[stream_idx2]}')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Plot difference map
        diff = weights_np[stream_idx1] - weights_np[stream_idx2]
        im3 = axes[1, 0].imshow(diff, cmap='RdBu_r', aspect='auto')
        axes[1, 0].set_title('Difference (Stream1 - Stream2)')
        plt.colorbar(im3, ax=axes[1, 0])
        
        # Plot correlation scatter
        flat1 = weights_np[stream_idx1].flatten()
        flat2 = weights_np[stream_idx2].flatten()
        axes[1, 1].scatter(flat1, flat2, alpha=0.6, s=10)
        
        # Add correlation line
        correlation = np.corrcoef(flat1, flat2)[0, 1]
        z = np.polyfit(flat1, flat2, 1)
        p = np.poly1d(z)
        x_line = np.linspace(flat1.min(), flat1.max(), 100)
        axes[1, 1].plot(x_line, p(x_line), "r--", alpha=0.8)
        
        axes[1, 1].set_xlabel(f'{stream_names[stream_idx1]} Weights')
        axes[1, 1].set_ylabel(f'{stream_names[stream_idx2]} Weights')
        axes[1, 1].set_title(f'Correlation: {correlation:.3f}')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_spatial_statistics(self,
                              spatial_weights: torch.Tensor,
                              stream_names: Optional[List[str]] = None,
                              title: str = "Spatial Weight Statistics",
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot statistical analysis of spatial weights.
        
        Args:
            spatial_weights: Tensor of shape (num_streams, height, width)
            stream_names: Names of the streams
            title: Plot title
            save_path: Path to save the figure
            
        Returns:
            The matplotlib figure
        """
        weights_np = spatial_weights.detach().cpu().numpy()
        num_streams = weights_np.shape[0]
        
        if stream_names is None:
            stream_names = [f"Stream {i+1}" for i in range(num_streams)]
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # 1. Mean spatial weights per stream
        mean_weights = [np.mean(weights_np[i]) for i in range(num_streams)]
        std_weights = [np.std(weights_np[i]) for i in range(num_streams)]
        
        x_pos = np.arange(num_streams)
        axes[0, 0].bar(x_pos, mean_weights, yerr=std_weights, capsize=5, alpha=0.7)
        axes[0, 0].set_xlabel('Stream')
        axes[0, 0].set_ylabel('Mean Weight')
        axes[0, 0].set_title('Mean Spatial Weight per Stream')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(stream_names)
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # 2. Weight distribution across all spatial locations
        all_weights = [weights_np[i].flatten() for i in range(num_streams)]
        axes[0, 1].hist(all_weights, bins=30, alpha=0.7, label=stream_names)
        axes[0, 1].set_xlabel('Weight Value')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Weight Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Spatial variance (how much weights vary spatially)
        spatial_variance = [np.var(weights_np[i]) for i in range(num_streams)]
        axes[1, 0].bar(x_pos, spatial_variance, alpha=0.7)
        axes[1, 0].set_xlabel('Stream')
        axes[1, 0].set_ylabel('Spatial Variance')
        axes[1, 0].set_title('Spatial Weight Variance')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(stream_names)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 4. Edge vs center weight analysis
        height, width = weights_np.shape[1], weights_np.shape[2]
        edge_weights = []
        center_weights = []
        
        for i in range(num_streams):
            # Edge: first/last rows and columns
            edge_mask = np.zeros((height, width), dtype=bool)
            edge_mask[0, :] = True
            edge_mask[-1, :] = True
            edge_mask[:, 0] = True
            edge_mask[:, -1] = True
            
            # Center: middle region
            center_h_start, center_h_end = height//4, 3*height//4
            center_w_start, center_w_end = width//4, 3*width//4
            center_mask = np.zeros((height, width), dtype=bool)
            center_mask[center_h_start:center_h_end, center_w_start:center_w_end] = True
            
            edge_weights.append(np.mean(weights_np[i][edge_mask]))
            center_weights.append(np.mean(weights_np[i][center_mask]))
        
        x_width = 0.35
        axes[1, 1].bar(x_pos - x_width/2, edge_weights, x_width, label='Edge', alpha=0.7)
        axes[1, 1].bar(x_pos + x_width/2, center_weights, x_width, label='Center', alpha=0.7)
        axes[1, 1].set_xlabel('Stream')
        axes[1, 1].set_ylabel('Mean Weight')
        axes[1, 1].set_title('Edge vs Center Weights')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(stream_names)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def analyze_spatial_patterns(self,
                               spatial_weights: torch.Tensor,
                               stream_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze spatial patterns in the weight maps.
        
        Args:
            spatial_weights: Tensor of shape (num_streams, height, width)
            stream_names: Names of the streams
            
        Returns:
            Dictionary containing spatial pattern analysis
        """
        weights_np = spatial_weights.detach().cpu().numpy()
        num_streams, height, width = weights_np.shape
        
        if stream_names is None:
            stream_names = [f"Stream {i+1}" for i in range(num_streams)]
        
        analysis = {}
        
        # Overall statistics
        analysis['overall'] = {
            'num_streams': num_streams,
            'spatial_dimensions': (height, width),
            'total_spatial_elements': height * width,
            'global_mean': float(np.mean(weights_np)),
            'global_std': float(np.std(weights_np)),
            'global_min': float(np.min(weights_np)),
            'global_max': float(np.max(weights_np))
        }
        
        # Per-stream analysis
        analysis['streams'] = {}
        for i, stream_name in enumerate(stream_names):
            stream_weights = weights_np[i]
            
            # Basic statistics
            stream_stats = {
                'mean': float(np.mean(stream_weights)),
                'std': float(np.std(stream_weights)),
                'min': float(np.min(stream_weights)),
                'max': float(np.max(stream_weights)),
                'variance': float(np.var(stream_weights)),
                'range': float(np.max(stream_weights) - np.min(stream_weights))
            }
            
            # Spatial gradients (measure of spatial variation)
            grad_y, grad_x = np.gradient(stream_weights)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            stream_stats['mean_gradient'] = float(np.mean(gradient_magnitude))
            stream_stats['max_gradient'] = float(np.max(gradient_magnitude))
            
            # Center vs edge analysis
            edge_mask = np.zeros((height, width), dtype=bool)
            edge_mask[0, :] = True
            edge_mask[-1, :] = True
            edge_mask[:, 0] = True
            edge_mask[:, -1] = True
            
            center_h_start, center_h_end = height//4, 3*height//4
            center_w_start, center_w_end = width//4, 3*width//4
            center_mask = np.zeros((height, width), dtype=bool)
            center_mask[center_h_start:center_h_end, center_w_start:center_w_end] = True
            
            stream_stats['edge_mean'] = float(np.mean(stream_weights[edge_mask]))
            stream_stats['center_mean'] = float(np.mean(stream_weights[center_mask]))
            stream_stats['edge_center_ratio'] = float(stream_stats['edge_mean'] / (stream_stats['center_mean'] + 1e-8))
            
            # Find peak locations
            max_idx = np.unravel_index(np.argmax(stream_weights), stream_weights.shape)
            min_idx = np.unravel_index(np.argmin(stream_weights), stream_weights.shape)
            
            stream_stats['max_location'] = {'y': int(max_idx[0]), 'x': int(max_idx[1])}
            stream_stats['min_location'] = {'y': int(min_idx[0]), 'x': int(min_idx[1])}
            
            analysis['streams'][stream_name] = stream_stats
        
        # Cross-stream spatial correlations
        analysis['spatial_correlations'] = {}
        for i in range(num_streams):
            for j in range(i+1, num_streams):
                flat_i = weights_np[i].flatten()
                flat_j = weights_np[j].flatten()
                corr = np.corrcoef(flat_i, flat_j)[0, 1]
                analysis['spatial_correlations'][f'{stream_names[i]}_vs_{stream_names[j]}'] = float(corr)
        
        return analysis
