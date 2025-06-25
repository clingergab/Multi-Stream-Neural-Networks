"""
Channel-wise weight visualization utilities for Multi-Stream Neural Networks.

This module provides tools for visualizing channel-wise mixing weights that apply
different scaling factors to individual channels within feature streams.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from typing import List, Optional, Dict, Any, Tuple


class ChannelWeightVisualizer:
    """Visualizer for channel-wise mixing weights in MSNNs."""
    
    def __init__(self, figsize: tuple = (12, 8)):
        """
        Initialize the channel weight visualizer.
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_channel_heatmap(self,
                           weights: torch.Tensor,
                           stream_names: Optional[List[str]] = None,
                           title: str = "Channel-wise Weight Heatmap",
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot a heatmap of channel-wise weights across streams.
        
        Args:
            weights: Weight tensor of shape (num_streams, num_channels)
            stream_names: Names of the streams
            title: Plot title
            save_path: Path to save the figure
            
        Returns:
            The matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        weights_np = weights.detach().cpu().numpy()
        num_streams, num_channels = weights_np.shape
        
        if stream_names is None:
            stream_names = [f"Stream {i+1}" for i in range(num_streams)]
        
        # Create heatmap
        im = ax.imshow(weights_np, cmap='viridis', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(np.arange(num_channels))
        ax.set_yticks(np.arange(num_streams))
        ax.set_xticklabels([f"Ch{i+1}" for i in range(num_channels)])
        ax.set_yticklabels(stream_names)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Weight Value', rotation=270, labelpad=20)
        
        # Add text annotations
        for i in range(num_streams):
            for j in range(num_channels):
                text = ax.text(j, i, f'{weights_np[i, j]:.2f}',
                             ha="center", va="center", color="white" if weights_np[i, j] < 0.5 else "black")
        
        ax.set_xlabel('Channel')
        ax.set_ylabel('Stream')
        ax.set_title(title)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_channel_distribution(self,
                                weights: torch.Tensor,
                                stream_idx: int = 0,
                                stream_name: Optional[str] = None,
                                title: Optional[str] = None,
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot the distribution of channel weights for a specific stream.
        
        Args:
            weights: Weight tensor of shape (num_streams, num_channels)
            stream_idx: Index of the stream to visualize
            stream_name: Name of the stream
            title: Plot title
            save_path: Path to save the figure
            
        Returns:
            The matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        weights_np = weights.detach().cpu().numpy()
        stream_weights = weights_np[stream_idx]
        
        if stream_name is None:
            stream_name = f"Stream {stream_idx + 1}"
        
        if title is None:
            title = f"Channel Weight Distribution - {stream_name}"
        
        # Bar plot of channel weights
        channels = np.arange(len(stream_weights))
        bars = ax1.bar(channels, stream_weights, alpha=0.7)
        ax1.set_xlabel('Channel')
        ax1.set_ylabel('Weight Value')
        ax1.set_title(f'{stream_name} - Channel Weights')
        ax1.grid(True, alpha=0.3)
        
        # Histogram of weight values
        ax2.hist(stream_weights, bins=20, alpha=0.7, edgecolor='black')
        ax2.axvline(np.mean(stream_weights), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(stream_weights):.3f}')
        ax2.set_xlabel('Weight Value')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'{stream_name} - Weight Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_cross_stream_comparison(self,
                                   weights: torch.Tensor,
                                   stream_names: Optional[List[str]] = None,
                                   title: str = "Cross-Stream Channel Weight Comparison",
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Compare channel weights across different streams.
        
        Args:
            weights: Weight tensor of shape (num_streams, num_channels)
            stream_names: Names of the streams
            title: Plot title
            save_path: Path to save the figure
            
        Returns:
            The matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        weights_np = weights.detach().cpu().numpy()
        num_streams, num_channels = weights_np.shape
        
        if stream_names is None:
            stream_names = [f"Stream {i+1}" for i in range(num_streams)]
        
        # Plot each stream's channel weights
        x = np.arange(num_channels)
        width = 0.8 / num_streams
        
        for i, stream_name in enumerate(stream_names):
            offset = (i - num_streams/2 + 0.5) * width
            ax.bar(x + offset, weights_np[i], width, 
                  label=stream_name, alpha=0.8)
        
        ax.set_xlabel('Channel')
        ax.set_ylabel('Weight Value')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([f'Ch{i+1}' for i in range(num_channels)])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def analyze_channel_importance(self,
                                 weights: torch.Tensor,
                                 stream_names: Optional[List[str]] = None,
                                 top_k: int = 5) -> Dict[str, Any]:
        """
        Analyze the importance of channels across streams.
        
        Args:
            weights: Weight tensor of shape (num_streams, num_channels)
            stream_names: Names of the streams
            top_k: Number of top channels to identify
            
        Returns:
            Dictionary containing channel importance analysis
        """
        weights_np = weights.detach().cpu().numpy()
        num_streams, num_channels = weights_np.shape
        
        if stream_names is None:
            stream_names = [f"Stream {i+1}" for i in range(num_streams)]
        
        # Overall channel importance (sum across streams)
        channel_importance = np.sum(weights_np, axis=0)
        top_channels = np.argsort(channel_importance)[-top_k:][::-1]
        
        # Per-stream analysis
        stream_analysis = {}
        for i, stream_name in enumerate(stream_names):
            stream_weights = weights_np[i]
            top_stream_channels = np.argsort(stream_weights)[-top_k:][::-1]
            
            stream_analysis[stream_name] = {
                'mean_weight': float(np.mean(stream_weights)),
                'std_weight': float(np.std(stream_weights)),
                'max_weight': float(np.max(stream_weights)),
                'min_weight': float(np.min(stream_weights)),
                'top_channels': [int(ch) for ch in top_stream_channels],
                'top_channel_weights': [float(stream_weights[ch]) for ch in top_stream_channels]
            }
        
        analysis = {
            'overall_top_channels': [int(ch) for ch in top_channels],
            'overall_channel_importance': [float(channel_importance[ch]) for ch in top_channels],
            'total_weight_sum': float(np.sum(weights_np)),
            'stream_analysis': stream_analysis,
            'weight_statistics': {
                'global_mean': float(np.mean(weights_np)),
                'global_std': float(np.std(weights_np)),
                'global_min': float(np.min(weights_np)),
                'global_max': float(np.max(weights_np))
            }
        }
        
        return analysis
