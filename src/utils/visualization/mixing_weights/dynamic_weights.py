"""
Dynamic weight visualization utilities for Multi-Stream Neural Networks.

This module provides tools for visualizing dynamic mixing weights that change
based on input content, learned attention mechanisms, or other adaptive factors.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation
from typing import List, Optional, Dict, Any, Tuple
import seaborn as sns


class DynamicWeightVisualizer:
    """Visualizer for dynamic mixing weights in MSNNs."""
    
    def __init__(self, figsize: tuple = (12, 8)):
        """
        Initialize the dynamic weight visualizer.
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        self.colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    def plot_temporal_weights(self,
                            weight_sequence: List[torch.Tensor],
                            stream_names: Optional[List[str]] = None,
                            title: str = "Dynamic Weight Evolution",
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot how dynamic weights change over time/samples.
        
        Args:
            weight_sequence: List of weight tensors over time
            stream_names: Names of the streams
            title: Plot title
            save_path: Path to save the figure
            
        Returns:
            The matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Convert to numpy
        weights_np = [w.detach().cpu().numpy() for w in weight_sequence]
        weights_array = np.array(weights_np)  # Shape: (time_steps, num_streams)
        
        num_streams = weights_array.shape[1]
        time_steps = len(weight_sequence)
        
        if stream_names is None:
            stream_names = [f"Stream {i+1}" for i in range(num_streams)]
        
        # Plot each stream's weight evolution
        for i in range(num_streams):
            ax.plot(range(time_steps), weights_array[:, i], 
                   label=stream_names[i],
                   color=self.colors[i % len(self.colors)],
                   linewidth=2,
                   marker='o',
                   markersize=4)
        
        ax.set_xlabel('Time Step / Sample')
        ax.set_ylabel('Weight Value')
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_attention_heatmap(self,
                             attention_weights: torch.Tensor,
                             input_names: Optional[List[str]] = None,
                             stream_names: Optional[List[str]] = None,
                             title: str = "Dynamic Attention Weights",
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot attention-based dynamic weights as a heatmap.
        
        Args:
            attention_weights: Tensor of shape (num_inputs, num_streams)
            input_names: Names for input samples/features
            stream_names: Names of the streams
            title: Plot title
            save_path: Path to save the figure
            
        Returns:
            The matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        weights_np = attention_weights.detach().cpu().numpy()
        num_inputs, num_streams = weights_np.shape
        
        if input_names is None:
            input_names = [f"Input {i+1}" for i in range(num_inputs)]
        if stream_names is None:
            stream_names = [f"Stream {i+1}" for i in range(num_streams)]
        
        # Create heatmap with seaborn for better styling
        sns.heatmap(weights_np.T, 
                   xticklabels=input_names,
                   yticklabels=stream_names,
                   annot=True,
                   fmt='.3f',
                   cmap='YlOrRd',
                   ax=ax)
        
        ax.set_xlabel('Input Sample')
        ax.set_ylabel('Stream')
        ax.set_title(title)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_weight_variance(self,
                           weight_sequence: List[torch.Tensor],
                           stream_names: Optional[List[str]] = None,
                           title: str = "Weight Variance Analysis",
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Analyze the variance of dynamic weights to understand adaptivity.
        
        Args:
            weight_sequence: List of weight tensors over time
            stream_names: Names of the streams
            title: Plot title
            save_path: Path to save the figure
            
        Returns:
            The matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        weights_np = [w.detach().cpu().numpy() for w in weight_sequence]
        weights_array = np.array(weights_np)
        
        num_streams = weights_array.shape[1]
        if stream_names is None:
            stream_names = [f"Stream {i+1}" for i in range(num_streams)]
        
        # Calculate statistics
        mean_weights = np.mean(weights_array, axis=0)
        std_weights = np.std(weights_array, axis=0)
        
        # Plot 1: Mean and standard deviation
        x_pos = np.arange(num_streams)
        bars1 = ax1.bar(x_pos, mean_weights, yerr=std_weights, 
                       capsize=5, alpha=0.7,
                       color=[self.colors[i % len(self.colors)] for i in range(num_streams)])
        
        ax1.set_xlabel('Stream')
        ax1.set_ylabel('Weight Value')
        ax1.set_title('Mean Weight ± Std Dev')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(stream_names)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Coefficient of variation (relative variability)
        cv = std_weights / (mean_weights + 1e-8)  # Add small epsilon to avoid division by zero
        bars2 = ax2.bar(x_pos, cv, alpha=0.7,
                       color=[self.colors[i % len(self.colors)] for i in range(num_streams)])
        
        ax2.set_xlabel('Stream')
        ax2.set_ylabel('Coefficient of Variation')
        ax2.set_title('Relative Weight Variability')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(stream_names)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_weight_animation(self,
                              weight_sequence: List[torch.Tensor],
                              stream_names: Optional[List[str]] = None,
                              interval: int = 200,
                              title: str = "Dynamic Weight Animation",
                              save_path: Optional[str] = None) -> FuncAnimation:
        """
        Create an animated visualization of dynamic weight changes.
        
        Args:
            weight_sequence: List of weight tensors over time
            stream_names: Names of the streams
            interval: Animation interval in milliseconds
            title: Animation title
            save_path: Path to save the animation (as GIF)
            
        Returns:
            The matplotlib animation object
        """
        weights_np = [w.detach().cpu().numpy() for w in weight_sequence]
        weights_array = np.array(weights_np)
        
        num_streams = weights_array.shape[1]
        if stream_names is None:
            stream_names = [f"Stream {i+1}" for i in range(num_streams)]
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Initialize the bar plot
        x_pos = np.arange(num_streams)
        bars = ax.bar(x_pos, weights_array[0], 
                     color=[self.colors[i % len(self.colors)] for i in range(num_streams)],
                     alpha=0.7)
        
        ax.set_xlabel('Stream')
        ax.set_ylabel('Weight Value')
        ax.set_title(f'{title} - Step 0')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(stream_names)
        ax.set_ylim(0, np.max(weights_array) * 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Animation function
        def animate(frame):
            for bar, height in zip(bars, weights_array[frame]):
                bar.set_height(height)
            ax.set_title(f'{title} - Step {frame}')
            return bars
        
        # Create animation
        anim = FuncAnimation(fig, animate, frames=len(weight_sequence),
                           interval=interval, blit=False, repeat=True)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=1000//interval)
        
        return anim
    
    def analyze_dynamic_patterns(self,
                               weight_sequence: List[torch.Tensor],
                               stream_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze patterns in dynamic weight changes.
        
        Args:
            weight_sequence: List of weight tensors over time
            stream_names: Names of the streams
            
        Returns:
            Dictionary containing dynamic pattern analysis
        """
        weights_np = [w.detach().cpu().numpy() for w in weight_sequence]
        weights_array = np.array(weights_np)
        
        num_streams = weights_array.shape[1]
        if stream_names is None:
            stream_names = [f"Stream {i+1}" for i in range(num_streams)]
        
        analysis = {}
        
        # Overall statistics
        analysis['overall'] = {
            'total_time_steps': len(weight_sequence),
            'num_streams': num_streams,
            'global_mean': float(np.mean(weights_array)),
            'global_std': float(np.std(weights_array)),
            'global_min': float(np.min(weights_array)),
            'global_max': float(np.max(weights_array))
        }
        
        # Per-stream analysis
        analysis['streams'] = {}
        for i, stream_name in enumerate(stream_names):
            stream_weights = weights_array[:, i]
            
            # Basic statistics
            stream_stats = {
                'mean': float(np.mean(stream_weights)),
                'std': float(np.std(stream_weights)),
                'min': float(np.min(stream_weights)),
                'max': float(np.max(stream_weights)),
                'range': float(np.max(stream_weights) - np.min(stream_weights)),
                'cv': float(np.std(stream_weights) / (np.mean(stream_weights) + 1e-8))
            }
            
            # Trend analysis
            time_steps = np.arange(len(stream_weights))
            correlation = np.corrcoef(time_steps, stream_weights)[0, 1]
            stream_stats['trend_correlation'] = float(correlation)
            
            # Change frequency (number of significant changes)
            changes = np.abs(np.diff(stream_weights))
            significant_changes = np.sum(changes > np.std(changes))
            stream_stats['significant_changes'] = int(significant_changes)
            stream_stats['change_frequency'] = float(significant_changes / len(stream_weights))
            
            analysis['streams'][stream_name] = stream_stats
        
        # Cross-stream correlations
        analysis['correlations'] = {}
        for i in range(num_streams):
            for j in range(i+1, num_streams):
                corr = np.corrcoef(weights_array[:, i], weights_array[:, j])[0, 1]
                analysis['correlations'][f'{stream_names[i]}_vs_{stream_names[j]}'] = float(corr)
        
        return analysis
