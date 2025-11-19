"""
Gradient monitoring utilities for Linear Integration Neural Networks (LINet).

This module provides lightweight gradient analysis tools for N-stream architectures
to detect pathway collapse or gradient imbalance between streams.

Usage:
    monitor = GradientMonitor(model)
    stats = monitor.compute_pathway_stats()  # Call after loss.backward()
"""

import re
import torch
import torch.nn as nn
from typing import Optional
from collections import defaultdict

# Optional matplotlib import for plotting
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class GradientMonitor:
    """
    Gradient monitoring for N-stream Linear Integration networks.

    Tracks gradients for:
    - Individual stream pathways (stream_0, stream_1, ..., streamN-1)
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
        # Get num_streams directly from model if available, otherwise detect
        self.num_streams = getattr(model, 'num_streams', self._detect_num_streams())

    def _detect_num_streams(self) -> int:
        """
        Detect the number of streams in the model by examining parameter names.

        Returns:
            Number of streams detected
        """
        stream_indices = set()
        for name, _ in self.model.named_parameters():
            # Match patterns like .stream_weights.0 or .stream_weights.0.weight
            match = re.search(r'\.stream_weights\.(\d+)(?:\.|$)', name)
            if match:
                stream_indices.add(int(match.group(1)))
            # Match patterns like .integration_from_streams.0 or .integration_from_streams.0.weight
            match = re.search(r'\.integration_from_streams\.(\d+)(?:\.|$)', name)
            if match:
                stream_indices.add(int(match.group(1)))

        if stream_indices:
            return max(stream_indices) + 1

        # Fallback: assume 2 streams if no stream_weights pattern found
        return 2

    def _get_stream_index(self, param_name: str) -> Optional[int]:
        """
        Extract stream index from parameter name.

        Args:
            param_name: Name of the parameter

        Returns:
            Stream index if found, None otherwise
        """
        # Match patterns like .stream_weights.0 or .stream_weights.0.weight
        match = re.search(r'\.stream_weights\.(\d+)(?:\.|$)', param_name)
        if match:
            return int(match.group(1))

        # Match patterns like .integration_from_streams.0 or .integration_from_streams.0.weight
        match = re.search(r'\.integration_from_streams\.(\d+)(?:\.|$)', param_name)
        if match:
            return int(match.group(1))

        return None

    def compute_pathway_stats(self) -> dict[str, float]:
        """
        Compute gradient statistics for each pathway.

        Call this AFTER loss.backward() to analyze gradient magnitudes.

        Returns:
            Dictionary with gradient statistics for all streams:
            - stream{i}_grad_norm: L2 norm of stream{i} gradients for each stream
            - integrated_grad_norm: L2 norm of integrated gradients
            - shared_grad_norm: L2 norm of shared gradients
            - stream{i}_to_stream0_ratio: Gradient ratio for each stream vs stream0
            - stream{i}_to_integrated_ratio: Stream vs integrated balance
            - stream{i}_mean: Mean gradient norm per parameter for each stream
            - stream{i}_param_count: Number of parameters for each stream
        """
        # Initialize accumulators for each stream
        stream_grad_norms = [0.0] * self.num_streams
        stream_counts = [0] * self.num_streams
        integrated_grad_norm = 0.0
        shared_grad_norm = 0.0
        integrated_count = 0
        shared_count = 0

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm_sq = param.grad.norm().item() ** 2

                # Check if this is a stream-specific parameter
                stream_idx = self._get_stream_index(name)
                if stream_idx is not None and stream_idx < self.num_streams:
                    stream_grad_norms[stream_idx] += grad_norm_sq
                    stream_counts[stream_idx] += 1
                elif 'integrated' in name:
                    integrated_grad_norm += grad_norm_sq
                    integrated_count += 1
                else:
                    # Shared parameters (fc, etc.)
                    shared_grad_norm += grad_norm_sq
                    shared_count += 1

        # Compute L2 norms
        stream_grad_norms = [norm ** 0.5 for norm in stream_grad_norms]
        integrated_grad_norm = integrated_grad_norm ** 0.5
        shared_grad_norm = shared_grad_norm ** 0.5

        # Build result dictionary
        result = {
            'integrated_grad_norm': integrated_grad_norm,
            'shared_grad_norm': shared_grad_norm,
            'integrated_mean': integrated_grad_norm / max(integrated_count, 1),
            'integrated_param_count': integrated_count,
            'shared_param_count': shared_count,
        }

        # Add per-stream statistics
        eps = 1e-8
        for i in range(self.num_streams):
            result[f'stream_{i}_grad_norm'] = stream_grad_norms[i]
            result[f'stream_{i}_mean'] = stream_grad_norms[i] / max(stream_counts[i], 1)
            result[f'stream_{i}_param_count'] = stream_counts[i]

        # Compute ratios (compare each stream to stream0 and integrated)
        for i in range(self.num_streams):
            if i > 0:
                result[f'stream_{i}_to_stream0_ratio'] = stream_grad_norms[i] / (stream_grad_norms[0] + eps)
            result[f'stream_{i}_to_integrated_ratio'] = stream_grad_norms[i] / (integrated_grad_norm + eps)

        return result

    def detect_pathway_collapse(self, threshold: float = 10.0) -> tuple[bool, str]:
        """
        Detect if one pathway is dominating training.

        Args:
            threshold: Ratio threshold for detecting collapse (default: 10.0)

        Returns:
            Tuple of (is_collapsed, warning_message)
        """
        stats = self.compute_pathway_stats()

        warnings = []
        is_collapsed = False

        # Check pairwise stream comparisons (compare each stream to stream0)
        for i in range(1, self.num_streams):
            ratio_key = f'stream_{i}_to_stream0_ratio'
            if ratio_key in stats:
                ratio = stats[ratio_key]
                if ratio > threshold:
                    warnings.append(f"Stream{i} dominates Stream0 (ratio: {ratio:.2f})")
                    is_collapsed = True
                elif ratio < 1.0 / threshold:
                    warnings.append(f"Stream0 dominates Stream{i} (ratio: {ratio:.2f})")
                    is_collapsed = True

        # Check each stream vs integrated
        for i in range(self.num_streams):
            ratio_key = f'stream_{i}_to_integrated_ratio'
            if ratio_key in stats:
                ratio = stats[ratio_key]
                if ratio > threshold:
                    warnings.append(f"Stream{i} dominates Integrated (ratio: {ratio:.2f})")
                    is_collapsed = True
                elif ratio < 1.0 / threshold:
                    warnings.append(f"Integrated dominates Stream{i} (ratio: {ratio:.2f})")
                    is_collapsed = True

        if is_collapsed:
            return True, "\n".join(warnings)
        else:
            # Build balance summary
            balance_parts = []
            for i in range(1, self.num_streams):
                ratio_key = f'stream_{i}_to_stream0_ratio'
                if ratio_key in stats:
                    balance_parts.append(f"S{i}/S0: {stats[ratio_key]:.2f}")

            for i in range(self.num_streams):
                ratio_key = f'stream_{i}_to_integrated_ratio'
                if ratio_key in stats:
                    balance_parts.append(f"S{i}/Int: {stats[ratio_key]:.2f}")

            return False, f"All pathways balanced ({', '.join(balance_parts)})"

    def get_layer_wise_stats(self) -> dict[str, dict[str, float]]:
        """
        Get gradient statistics grouped by layer.

        Returns:
            Dictionary mapping layer names to their gradient statistics
        """
        # Initialize default dict with dynamic stream count
        layer_stats = defaultdict(lambda: {f'stream_{i}': 0.0 for i in range(self.num_streams)} | {'integrated': 0.0})

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # Extract layer name (e.g., "layer1", "layer2", etc.)
                layer_name = name.split('.')[0]
                grad_norm_sq = param.grad.norm().item() ** 2

                # Check if this is a stream-specific parameter
                stream_idx = self._get_stream_index(name)
                if stream_idx is not None and stream_idx < self.num_streams:
                    layer_stats[layer_name][f'stream{stream_idx}'] += grad_norm_sq
                elif 'integrated' in name:
                    layer_stats[layer_name]['integrated'] += grad_norm_sq

        # Convert to L2 norms and compute ratios
        result = {}
        eps = 1e-8
        for layer_name, stats in layer_stats.items():
            layer_result = {}

            # Get norms for all streams
            stream_norms = [stats[f'stream_{i}'] ** 0.5 for i in range(self.num_streams)]
            int_norm = stats['integrated'] ** 0.5

            # Add per-stream norms
            for i in range(self.num_streams):
                layer_result[f'stream_{i}_grad_norm'] = stream_norms[i]

            layer_result['integrated_grad_norm'] = int_norm

            # Compute ratios (compare each stream to stream0 and integrated)
            for i in range(self.num_streams):
                if i > 0:
                    layer_result[f's{i}_to_s0_ratio'] = stream_norms[i] / (stream_norms[0] + eps)
                layer_result[f's{i}_to_int_ratio'] = stream_norms[i] / (int_norm + eps)

            result[layer_name] = layer_result

        return result

    def print_summary(self):
        """Print a formatted summary of gradient statistics."""
        stats = self.compute_pathway_stats()

        print("\n" + "="*70)
        print("LI-NET GRADIENT MONITORING SUMMARY")
        print("="*70)

        # Print stats for each stream
        for i in range(self.num_streams):
            print(f"Stream{i}:")
            print(f"  Total gradient norm: {stats[f'stream_{i}_grad_norm']:.6f}")
            print(f"  Mean gradient norm:  {stats[f'stream_{i}_mean']:.6f}")
            print(f"  Parameter count:     {stats[f'stream_{i}_param_count']}")
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
        # Print stream-to-stream0 ratios
        for i in range(1, self.num_streams):
            ratio_key = f'stream_{i}_to_stream0_ratio'
            if ratio_key in stats:
                print(f"  Stream_{i}/Stream_0:    {stats[ratio_key]:.4f}")

        # Print stream-to-integrated ratios
        for i in range(self.num_streams):
            ratio_key = f'stream_{i}_to_integrated_ratio'
            if ratio_key in stats:
                print(f"  Stream{i}/Integrated: {stats[ratio_key]:.4f}")

        # Provide interpretation
        print()
        warnings = []
        for i in range(1, self.num_streams):
            ratio_key = f'stream_{i}_to_stream0_ratio'
            if ratio_key in stats:
                ratio = stats[ratio_key]
                if ratio > 5.0:
                    warnings.append(f"Stream{i} appears to dominate Stream0")
                elif ratio < 0.2:
                    warnings.append(f"Stream0 appears to dominate Stream{i}")

        for i in range(self.num_streams):
            ratio_key = f'stream_{i}_to_integrated_ratio'
            if ratio_key in stats:
                ratio = stats[ratio_key]
                if ratio > 5.0:
                    warnings.append(f"Stream{i} appears to dominate Integrated")
                elif ratio < 0.2:
                    warnings.append(f"Integrated appears to dominate Stream{i}")

        if warnings:
            for warning in warnings:
                print(f"WARNING: {warning}")
        else:
            print("All pathways appear balanced")
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

    def get_history(self) -> list[dict[str, float]]:
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
        if not HAS_MATPLOTLIB:
            print("matplotlib not available for plotting")
            return

        if not self.history:
            print("No history to plot")
            return

        # Detect number of streams from first history entry
        num_streams = self.monitor.num_streams

        steps = list(range(len(self.history)))

        # Collect data for all streams
        stream_norms = []
        for i in range(num_streams):
            stream_norms.append([h[f'stream_{i}_grad_norm'] for h in self.history])

        integrated_norms = [h['integrated_grad_norm'] for h in self.history]

        # Collect ratio data
        stream_to_stream0_ratios = []
        for i in range(1, num_streams):
            ratio_key = f'stream_{i}_to_stream0_ratio'
            if ratio_key in self.history[0]:
                stream_to_stream0_ratios.append(([h[ratio_key] for h in self.history], f'Stream_{i}/Stream_0'))

        stream_to_int_ratios = []
        for i in range(num_streams):
            ratio_key = f'stream_{i}_to_integrated_ratio'
            if ratio_key in self.history[0]:
                stream_to_int_ratios.append(([h[ratio_key] for h in self.history], f'Stream{i}/Integrated'))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot gradient norms
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
        for i in range(num_streams):
            ax1.plot(steps, stream_norms[i], label=f'Stream{i}', marker=markers[i % len(markers)])
        ax1.plot(steps, integrated_norms, label='Integrated', marker=markers[num_streams % len(markers)])
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Gradient Norm')
        ax1.set_title('Gradient Magnitudes by Stream (LINet)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot ratios
        colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        color_idx = 0
        for ratios, label in stream_to_stream0_ratios:
            ax2.plot(steps, ratios, label=label, marker='o', color=colors[color_idx % len(colors)])
            color_idx += 1
        for ratios, label in stream_to_int_ratios:
            ax2.plot(steps, ratios, label=label, marker='s', color=colors[color_idx % len(colors)])
            color_idx += 1

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
