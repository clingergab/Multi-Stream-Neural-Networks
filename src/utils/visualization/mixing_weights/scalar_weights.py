"""
Scalar weight visualization utilities for Multi-Stream Neural Networks.

This module provides tools for visualizing scalar mixing weights that apply
uniform scaling to entire feature streams.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import List, Optional, Dict, Any


class ScalarWeightVisualizer:
    """Visualizer for scalar mixing weights in MSNNs."""
    
    def __init__(self, figsize: tuple = (10, 6)):
        """
        Initialize the scalar weight visualizer.
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    def plot_weight_evolution(self, 
                            weights_history: List[torch.Tensor],
                            stream_names: Optional[List[str]] = None,
                            title: str = "Scalar Weight Evolution",
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot the evolution of scalar weights during training.
        
        Args:
            weights_history: List of weight tensors over training steps
            stream_names: Names of the streams
            title: Plot title
            save_path: Path to save the figure
            
        Returns:
            The matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Convert to numpy for plotting
        weights_np = [w.detach().cpu().numpy() for w in weights_history]
        weights_array = np.array(weights_np)
        
        num_streams = weights_array.shape[1]
        if stream_names is None:
            stream_names = [f"Stream {i+1}" for i in range(num_streams)]
        
        # Plot each stream's weight evolution
        for i in range(num_streams):
            color = self.colors[i % len(self.colors)]
            ax.plot(weights_array[:, i], 
                   label=stream_names[i],
                   color=color,
                   linewidth=2)
        
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Weight Value')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_weight_distribution(self,
                               weights: torch.Tensor,
                               stream_names: Optional[List[str]] = None,
                               title: str = "Scalar Weight Distribution",
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot the distribution of scalar weights across streams.
        
        Args:
            weights: Current weight tensor
            stream_names: Names of the streams
            title: Plot title
            save_path: Path to save the figure
            
        Returns:
            The matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        weights_np = weights.detach().cpu().numpy()
        num_streams = len(weights_np)
        
        if stream_names is None:
            stream_names = [f"Stream {i+1}" for i in range(num_streams)]
        
        # Create bar plot
        x_pos = np.arange(num_streams)
        bars = ax.bar(x_pos, weights_np, 
                     color=[self.colors[i % len(self.colors)] for i in range(num_streams)],
                     alpha=0.7)
        
        # Add value labels on bars
        for bar, weight in zip(bars, weights_np):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{weight:.3f}',
                   ha='center', va='bottom')
        
        ax.set_xlabel('Stream')
        ax.set_ylabel('Weight Value')
        ax.set_title(title)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(stream_names)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_weight_summary(self,
                            weights: torch.Tensor,
                            stream_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Create a summary of scalar weight statistics.
        
        Args:
            weights: Current weight tensor
            stream_names: Names of the streams
            
        Returns:
            Dictionary containing weight statistics
        """
        weights_np = weights.detach().cpu().numpy()
        num_streams = len(weights_np)
        
        if stream_names is None:
            stream_names = [f"Stream {i+1}" for i in range(num_streams)]
        
        summary = {
            'total_streams': num_streams,
            'mean_weight': float(np.mean(weights_np)),
            'std_weight': float(np.std(weights_np)),
            'min_weight': float(np.min(weights_np)),
            'max_weight': float(np.max(weights_np)),
            'weight_range': float(np.max(weights_np) - np.min(weights_np)),
            'dominant_stream': stream_names[np.argmax(weights_np)],
            'weakest_stream': stream_names[np.argmin(weights_np)],
            'weights_per_stream': {
                name: float(weight) for name, weight in zip(stream_names, weights_np)
            }
        }
        
        return summary
