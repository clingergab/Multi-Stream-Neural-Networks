"""
Gradient monitoring utilities for multi-stream neural networks.

This module provides lightweight, zero-overhead gradient analysis tools
that can be optionally enabled during training to detect pathway collapse
or gradient imbalance between streams.

Usage:
    # Only incurs overhead when explicitly called
    monitor = GradientMonitor(model)
    stats = monitor.compute_pathway_stats()  # Call after loss.backward()

    # Or use during training
    if epoch % 10 == 0:  # Only check every 10 epochs
        stats = monitor.compute_pathway_stats()
        if stats['ratio'] > 10.0:
            print("Warning: Pathway imbalance detected!")
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
from collections import defaultdict


class GradientMonitor:
    """
    Lightweight gradient monitoring for multi-stream networks.

    Zero overhead when not actively used - only computes statistics
    when explicitly called.
    """

    def __init__(self, model: nn.Module, stream1_name: str = "stream1", stream2_name: str = "stream2"):
        """
        Initialize gradient monitor.

        Args:
            model: The neural network model to monitor
            stream1_name: Name pattern for first stream parameters (e.g., "color", "stream1")
            stream2_name: Name pattern for second stream parameters (e.g., "brightness", "stream2")
        """
        self.model = model
        self.stream1_name = stream1_name
        self.stream2_name = stream2_name

    def compute_pathway_stats(self) -> Dict[str, float]:
        """
        Compute gradient statistics for each pathway.

        Call this AFTER loss.backward() to analyze gradient magnitudes.

        Returns:
            Dictionary with gradient statistics:
            - stream1_grad_norm: L2 norm of stream1 gradients
            - stream2_grad_norm: L2 norm of stream2 gradients
            - ratio: stream1_norm / stream2_norm (indicates imbalance)
            - stream1_mean: Mean gradient magnitude for stream1
            - stream2_mean: Mean gradient magnitude for stream2
        """
        stream1_grad_norm = 0.0
        stream2_grad_norm = 0.0
        stream1_count = 0
        stream2_count = 0

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()

                if self.stream1_name in name:
                    stream1_grad_norm += grad_norm ** 2
                    stream1_count += 1
                elif self.stream2_name in name:
                    stream2_grad_norm += grad_norm ** 2
                    stream2_count += 1

        # Compute L2 norms
        stream1_grad_norm = stream1_grad_norm ** 0.5
        stream2_grad_norm = stream2_grad_norm ** 0.5

        # Compute ratio (add epsilon to avoid division by zero)
        ratio = stream1_grad_norm / (stream2_grad_norm + 1e-8)

        return {
            'stream1_grad_norm': stream1_grad_norm,
            'stream2_grad_norm': stream2_grad_norm,
            'ratio': ratio,
            'stream1_mean': stream1_grad_norm / max(stream1_count, 1),
            'stream2_mean': stream2_grad_norm / max(stream2_count, 1),
            'stream1_param_count': stream1_count,
            'stream2_param_count': stream2_count,
        }

    def detect_pathway_collapse(self, threshold: float = 10.0) -> tuple[bool, str]:
        """
        Detect if one pathway is dominating training.

        Args:
            threshold: Ratio threshold for detecting collapse (default: 10.0)
                      ratio > threshold indicates potential collapse

        Returns:
            Tuple of (is_collapsed, warning_message)
        """
        stats = self.compute_pathway_stats()
        ratio = stats['ratio']

        if ratio > threshold:
            return True, f"⚠️  Stream1 dominance detected (ratio: {ratio:.2f})"
        elif ratio < 1.0 / threshold:
            return True, f"⚠️  Stream2 dominance detected (ratio: {ratio:.2f})"
        else:
            return False, f"✓ Pathways balanced (ratio: {ratio:.2f})"

    def get_layer_wise_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get gradient statistics grouped by layer.

        Returns:
            Dictionary mapping layer names to their gradient statistics
        """
        layer_stats = defaultdict(lambda: {'stream1': 0.0, 'stream2': 0.0})

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # Extract layer name (e.g., "layer1", "layer2", etc.)
                layer_name = name.split('.')[0]
                grad_norm = param.grad.norm().item()

                if self.stream1_name in name:
                    layer_stats[layer_name]['stream1'] += grad_norm ** 2
                elif self.stream2_name in name:
                    layer_stats[layer_name]['stream2'] += grad_norm ** 2

        # Convert to L2 norms and compute ratios
        result = {}
        for layer_name, stats in layer_stats.items():
            stream1_norm = stats['stream1'] ** 0.5
            stream2_norm = stats['stream2'] ** 0.5
            result[layer_name] = {
                'stream1_grad_norm': stream1_norm,
                'stream2_grad_norm': stream2_norm,
                'ratio': stream1_norm / (stream2_norm + 1e-8)
            }

        return result

    def print_summary(self):
        """Print a formatted summary of gradient statistics."""
        stats = self.compute_pathway_stats()

        print("\n" + "="*60)
        print("GRADIENT MONITORING SUMMARY")
        print("="*60)
        print(f"Stream1 ({self.stream1_name}):")
        print(f"  Total gradient norm: {stats['stream1_grad_norm']:.6f}")
        print(f"  Mean gradient norm:  {stats['stream1_mean']:.6f}")
        print(f"  Parameter count:     {stats['stream1_param_count']}")
        print()
        print(f"Stream2 ({self.stream2_name}):")
        print(f"  Total gradient norm: {stats['stream2_grad_norm']:.6f}")
        print(f"  Mean gradient norm:  {stats['stream2_mean']:.6f}")
        print(f"  Parameter count:     {stats['stream2_param_count']}")
        print()
        print(f"Ratio (Stream1/Stream2): {stats['ratio']:.4f}")

        # Provide interpretation
        if stats['ratio'] > 5.0:
            print("⚠️  WARNING: Stream1 appears to dominate training")
        elif stats['ratio'] < 0.2:
            print("⚠️  WARNING: Stream2 appears to dominate training")
        else:
            print("✓ Pathways appear balanced")
        print("="*60 + "\n")


class GradientLogger:
    """
    Accumulate gradient statistics over multiple training steps.

    Useful for tracking gradient trends during training.
    """

    def __init__(self, model: nn.Module, stream1_name: str = "stream1", stream2_name: str = "stream2"):
        """
        Initialize gradient logger.

        Args:
            model: The neural network model to monitor
            stream1_name: Name pattern for first stream parameters
            stream2_name: Name pattern for second stream parameters
        """
        self.monitor = GradientMonitor(model, stream1_name, stream2_name)
        self.history = []

    def log(self):
        """Log current gradient statistics."""
        stats = self.monitor.compute_pathway_stats()
        self.history.append(stats)

    def get_history(self) -> List[Dict[str, float]]:
        """Get logged gradient history."""
        return self.history

    def clear(self):
        """Clear logged history."""
        self.history = []

    def plot_history(self, save_path: Optional[str] = None):
        """
        Plot gradient history over time.

        Args:
            save_path: Optional path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available for plotting")
            return

        if not self.history:
            print("No history to plot")
            return

        steps = list(range(len(self.history)))
        stream1_norms = [h['stream1_grad_norm'] for h in self.history]
        stream2_norms = [h['stream2_grad_norm'] for h in self.history]
        ratios = [h['ratio'] for h in self.history]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Plot gradient norms
        ax1.plot(steps, stream1_norms, label='Stream1', marker='o')
        ax1.plot(steps, stream2_norms, label='Stream2', marker='s')
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Gradient Norm')
        ax1.set_title('Gradient Magnitudes by Stream')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot ratio
        ax2.plot(steps, ratios, label='Stream1/Stream2 Ratio', marker='^', color='purple')
        ax2.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Perfect Balance')
        ax2.axhline(y=5.0, color='red', linestyle='--', alpha=0.3, label='Warning Threshold')
        ax2.axhline(y=0.2, color='red', linestyle='--', alpha=0.3)
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Ratio')
        ax2.set_title('Pathway Balance Ratio')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
