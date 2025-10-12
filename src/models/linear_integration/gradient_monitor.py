"""
Gradient monitoring utilities for Linear Integration Neural Networks (LINet).

This module provides lightweight gradient analysis tools for 3-stream architectures
to detect pathway collapse or gradient imbalance between streams.

Usage:
    monitor = GradientMonitor(model)
    stats = monitor.compute_pathway_stats()  # Call after loss.backward()
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
from collections import defaultdict


class GradientMonitor:
    """
    Gradient monitoring for 3-stream Linear Integration networks.

    Tracks gradients for:
    - Stream1 (RGB) pathway
    - Stream2 (Depth) pathway
    - Integrated pathway
    - Shared parameters (classifier, etc.)
    """

    def __init__(self, model: nn.Module):
        """
        Initialize gradient monitor.

        Args:
            model: The LINet model to monitor
        """
        self.model = model

    def compute_pathway_stats(self) -> Dict[str, float]:
        """
        Compute gradient statistics for each pathway.

        Call this AFTER loss.backward() to analyze gradient magnitudes.

        Returns:
            Dictionary with gradient statistics for all 3 streams:
            - stream1_grad_norm: L2 norm of stream1 gradients
            - stream2_grad_norm: L2 norm of stream2 gradients
            - integrated_grad_norm: L2 norm of integrated gradients
            - shared_grad_norm: L2 norm of shared gradients
            - stream1_to_stream2_ratio: Gradient imbalance indicator
            - stream1_to_integrated_ratio: Stream1 vs integrated balance
            - stream2_to_integrated_ratio: Stream2 vs integrated balance
        """
        stream1_grad_norm = 0.0
        stream2_grad_norm = 0.0
        integrated_grad_norm = 0.0
        shared_grad_norm = 0.0

        stream1_count = 0
        stream2_count = 0
        integrated_count = 0
        shared_count = 0

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm_sq = param.grad.norm().item() ** 2

                if 'stream1' in name:
                    stream1_grad_norm += grad_norm_sq
                    stream1_count += 1
                elif 'stream2' in name:
                    stream2_grad_norm += grad_norm_sq
                    stream2_count += 1
                elif 'integrated' in name or 'integration' in name:
                    integrated_grad_norm += grad_norm_sq
                    integrated_count += 1
                else:
                    # Shared parameters (fc, etc.)
                    shared_grad_norm += grad_norm_sq
                    shared_count += 1

        # Compute L2 norms
        stream1_grad_norm = stream1_grad_norm ** 0.5
        stream2_grad_norm = stream2_grad_norm ** 0.5
        integrated_grad_norm = integrated_grad_norm ** 0.5
        shared_grad_norm = shared_grad_norm ** 0.5

        # Compute ratios (add epsilon to avoid division by zero)
        eps = 1e-8
        s1_to_s2_ratio = stream1_grad_norm / (stream2_grad_norm + eps)
        s1_to_int_ratio = stream1_grad_norm / (integrated_grad_norm + eps)
        s2_to_int_ratio = stream2_grad_norm / (integrated_grad_norm + eps)

        return {
            'stream1_grad_norm': stream1_grad_norm,
            'stream2_grad_norm': stream2_grad_norm,
            'integrated_grad_norm': integrated_grad_norm,
            'shared_grad_norm': shared_grad_norm,
            'stream1_to_stream2_ratio': s1_to_s2_ratio,
            'stream1_to_integrated_ratio': s1_to_int_ratio,
            'stream2_to_integrated_ratio': s2_to_int_ratio,
            'stream1_mean': stream1_grad_norm / max(stream1_count, 1),
            'stream2_mean': stream2_grad_norm / max(stream2_count, 1),
            'integrated_mean': integrated_grad_norm / max(integrated_count, 1),
            'stream1_param_count': stream1_count,
            'stream2_param_count': stream2_count,
            'integrated_param_count': integrated_count,
            'shared_param_count': shared_count,
        }

    def detect_pathway_collapse(self, threshold: float = 10.0) -> tuple[bool, str]:
        """
        Detect if one pathway is dominating training.

        Args:
            threshold: Ratio threshold for detecting collapse (default: 10.0)

        Returns:
            Tuple of (is_collapsed, warning_message)
        """
        stats = self.compute_pathway_stats()

        s1_to_s2 = stats['stream1_to_stream2_ratio']
        s1_to_int = stats['stream1_to_integrated_ratio']
        s2_to_int = stats['stream2_to_integrated_ratio']

        warnings = []
        is_collapsed = False

        # Check stream1 vs stream2
        if s1_to_s2 > threshold:
            warnings.append(f"⚠️  Stream1 dominates Stream2 (ratio: {s1_to_s2:.2f})")
            is_collapsed = True
        elif s1_to_s2 < 1.0 / threshold:
            warnings.append(f"⚠️  Stream2 dominates Stream1 (ratio: {s1_to_s2:.2f})")
            is_collapsed = True

        # Check integrated stream
        if s1_to_int > threshold:
            warnings.append(f"⚠️  Stream1 dominates Integrated (ratio: {s1_to_int:.2f})")
            is_collapsed = True
        elif s1_to_int < 1.0 / threshold:
            warnings.append(f"⚠️  Integrated dominates Stream1 (ratio: {s1_to_int:.2f})")
            is_collapsed = True

        if s2_to_int > threshold:
            warnings.append(f"⚠️  Stream2 dominates Integrated (ratio: {s2_to_int:.2f})")
            is_collapsed = True
        elif s2_to_int < 1.0 / threshold:
            warnings.append(f"⚠️  Integrated dominates Stream2 (ratio: {s2_to_int:.2f})")
            is_collapsed = True

        if is_collapsed:
            return True, "\n".join(warnings)
        else:
            return False, f"✓ All pathways balanced (S1/S2: {s1_to_s2:.2f}, S1/Int: {s1_to_int:.2f}, S2/Int: {s2_to_int:.2f})"

    def get_layer_wise_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get gradient statistics grouped by layer.

        Returns:
            Dictionary mapping layer names to their gradient statistics
        """
        layer_stats = defaultdict(lambda: {'stream1': 0.0, 'stream2': 0.0, 'integrated': 0.0})

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # Extract layer name (e.g., "layer1", "layer2", etc.)
                layer_name = name.split('.')[0]
                grad_norm_sq = param.grad.norm().item() ** 2

                if 'stream1' in name:
                    layer_stats[layer_name]['stream1'] += grad_norm_sq
                elif 'stream2' in name:
                    layer_stats[layer_name]['stream2'] += grad_norm_sq
                elif 'integrated' in name or 'integration' in name:
                    layer_stats[layer_name]['integrated'] += grad_norm_sq

        # Convert to L2 norms and compute ratios
        result = {}
        for layer_name, stats in layer_stats.items():
            s1_norm = stats['stream1'] ** 0.5
            s2_norm = stats['stream2'] ** 0.5
            int_norm = stats['integrated'] ** 0.5

            eps = 1e-8
            result[layer_name] = {
                'stream1_grad_norm': s1_norm,
                'stream2_grad_norm': s2_norm,
                'integrated_grad_norm': int_norm,
                's1_to_s2_ratio': s1_norm / (s2_norm + eps),
                's1_to_int_ratio': s1_norm / (int_norm + eps),
                's2_to_int_ratio': s2_norm / (int_norm + eps),
            }

        return result

    def print_summary(self):
        """Print a formatted summary of gradient statistics."""
        stats = self.compute_pathway_stats()

        print("\n" + "="*70)
        print("LI-NET GRADIENT MONITORING SUMMARY")
        print("="*70)
        print(f"Stream1 (RGB):")
        print(f"  Total gradient norm: {stats['stream1_grad_norm']:.6f}")
        print(f"  Mean gradient norm:  {stats['stream1_mean']:.6f}")
        print(f"  Parameter count:     {stats['stream1_param_count']}")
        print()
        print(f"Stream2 (Depth):")
        print(f"  Total gradient norm: {stats['stream2_grad_norm']:.6f}")
        print(f"  Mean gradient norm:  {stats['stream2_mean']:.6f}")
        print(f"  Parameter count:     {stats['stream2_param_count']}")
        print()
        print(f"Integrated:")
        print(f"  Total gradient norm: {stats['integrated_grad_norm']:.6f}")
        print(f"  Mean gradient norm:  {stats['integrated_mean']:.6f}")
        print(f"  Parameter count:     {stats['integrated_param_count']}")
        print()
        print(f"Shared (Classifier, etc.):")
        print(f"  Total gradient norm: {stats['shared_grad_norm']:.6f}")
        print(f"  Parameter count:     {stats['shared_param_count']}")
        print()
        print(f"Ratios:")
        print(f"  Stream1/Stream2:     {stats['stream1_to_stream2_ratio']:.4f}")
        print(f"  Stream1/Integrated:  {stats['stream1_to_integrated_ratio']:.4f}")
        print(f"  Stream2/Integrated:  {stats['stream2_to_integrated_ratio']:.4f}")

        # Provide interpretation
        s1_to_s2 = stats['stream1_to_stream2_ratio']
        s1_to_int = stats['stream1_to_integrated_ratio']
        s2_to_int = stats['stream2_to_integrated_ratio']

        print()
        if s1_to_s2 > 5.0 or s1_to_int > 5.0:
            print("⚠️  WARNING: Stream1 appears to dominate training")
        elif s1_to_s2 < 0.2 or s2_to_int > 5.0:
            print("⚠️  WARNING: Stream2 appears to dominate training")
        elif s1_to_int < 0.2 or s2_to_int < 0.2:
            print("⚠️  WARNING: Integrated stream appears to dominate training")
        else:
            print("✓ All pathways appear balanced")
        print("="*70 + "\n")


class GradientLogger:
    """
    Accumulate gradient statistics over multiple training steps for LINet.
    """

    def __init__(self, model: nn.Module):
        """
        Initialize gradient logger.

        Args:
            model: The LINet model to monitor
        """
        self.monitor = GradientMonitor(model)
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
        integrated_norms = [h['integrated_grad_norm'] for h in self.history]
        s1_to_s2_ratios = [h['stream1_to_stream2_ratio'] for h in self.history]
        s1_to_int_ratios = [h['stream1_to_integrated_ratio'] for h in self.history]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot gradient norms
        ax1.plot(steps, stream1_norms, label='Stream1 (RGB)', marker='o')
        ax1.plot(steps, stream2_norms, label='Stream2 (Depth)', marker='s')
        ax1.plot(steps, integrated_norms, label='Integrated', marker='^')
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Gradient Norm')
        ax1.set_title('Gradient Magnitudes by Stream (LINet)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot ratios
        ax2.plot(steps, s1_to_s2_ratios, label='Stream1/Stream2', marker='o', color='blue')
        ax2.plot(steps, s1_to_int_ratios, label='Stream1/Integrated', marker='s', color='green')
        ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect Balance')
        ax2.axhline(y=5.0, color='red', linestyle='--', alpha=0.3, label='Warning Threshold')
        ax2.axhline(y=0.2, color='red', linestyle='--', alpha=0.3)
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Ratio (log scale)')
        ax2.set_title('Pathway Balance Ratios (LINet)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
