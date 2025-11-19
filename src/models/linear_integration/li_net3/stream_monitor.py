"""
Stream-specific monitoring utilities for Linear Integration Neural Networks (LINet).

Provides tools to monitor individual stream learning, detect overfitting,
and gain insights into N-stream behavior during training.
"""

import torch
import torch.nn as nn
from typing import Optional
import numpy as np
from collections import defaultdict
import re


class StreamMonitor:
    """
    Monitor individual stream behavior during LINet training.

    Tracks:
    - Stream-specific gradients (N streams)
    - Stream-specific activations (N streams)
    - Stream-specific performance metrics
    - Overfitting indicators per stream
    - Integration effectiveness
    """

    def __init__(self, model: nn.Module):
        """
        Initialize stream monitor.

        Args:
            model: The LINet model to monitor
        """
        self.model = model
        self.history = defaultdict(list)
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

    def compute_stream_gradients(self) -> dict[str, float]:
        """
        Compute gradient statistics for each of the N streams.

        Returns:
            Dictionary with gradient norms and statistics per stream
        """
        # Initialize accumulators for each stream
        stream_grad_norms = [0.0] * self.num_streams
        stream_grad_maxs = [0.0] * self.num_streams
        stream_params = [0] * self.num_streams
        integrated_grad_norm = 0.0
        shared_grad_norm = 0.0
        integrated_grad_max = 0.0
        integrated_params = 0
        shared_params = 0

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_max = param.grad.abs().max().item()

                # Check if this is a stream-specific parameter
                stream_idx = self._get_stream_index(name)
                if stream_idx is not None and stream_idx < self.num_streams:
                    stream_grad_norms[stream_idx] += grad_norm ** 2
                    stream_grad_maxs[stream_idx] = max(stream_grad_maxs[stream_idx], grad_max)
                    stream_params[stream_idx] += 1
                elif 'integrated' in name:
                    integrated_grad_norm += grad_norm ** 2
                    integrated_grad_max = max(integrated_grad_max, grad_max)
                    integrated_params += 1
                else:
                    shared_grad_norm += grad_norm ** 2
                    shared_params += 1

        # Compute L2 norms
        stream_grad_norms = [np.sqrt(norm) for norm in stream_grad_norms]
        integrated_grad_norm = np.sqrt(integrated_grad_norm)
        shared_grad_norm = np.sqrt(shared_grad_norm)

        # Build result dictionary
        result = {
            'integrated_grad_norm': integrated_grad_norm,
            'shared_grad_norm': shared_grad_norm,
            'integrated_grad_max': integrated_grad_max,
            'integrated_params': integrated_params,
            'shared_params': shared_params
        }

        # Add per-stream statistics
        eps = 1e-8
        for i in range(self.num_streams):
            result[f'stream_{i}_grad_norm'] = stream_grad_norms[i]
            result[f'stream_{i}_grad_max'] = stream_grad_maxs[i]
            result[f'stream_{i}_params'] = stream_params[i]

        # Compute ratios (compare each stream to stream0 and integrated)
        for i in range(self.num_streams):
            if i > 0:
                result[f'stream_{i}_to_stream0_ratio'] = stream_grad_norms[i] / (stream_grad_norms[0] + eps)
            result[f'stream_{i}_to_integrated_ratio'] = stream_grad_norms[i] / (integrated_grad_norm + eps)

        return result

    def compute_stream_weights(self) -> dict[str, float]:
        """
        Compute weight statistics for each of the N streams.

        Returns:
            Dictionary with weight norms and statistics per stream
        """
        # Initialize accumulators for each stream
        stream_weight_norms = [0.0] * self.num_streams
        stream_weight_means = [[] for _ in range(self.num_streams)]
        integrated_weight_norm = 0.0
        integrated_weight_mean = []

        for name, param in self.model.named_parameters():
            if 'weight' in name:
                # Check if this is a stream-specific parameter
                stream_idx = self._get_stream_index(name)
                if stream_idx is not None and stream_idx < self.num_streams:
                    stream_weight_norms[stream_idx] += param.norm().item() ** 2
                    stream_weight_means[stream_idx].append(param.mean().item())
                elif 'integrated' in name:
                    integrated_weight_norm += param.norm().item() ** 2
                    integrated_weight_mean.append(param.mean().item())

        # Compute L2 norms
        stream_weight_norms = [np.sqrt(norm) for norm in stream_weight_norms]
        integrated_weight_norm = np.sqrt(integrated_weight_norm)

        # Build result dictionary
        result = {
            'integrated_weight_norm': integrated_weight_norm,
            'integrated_weight_mean': np.mean(integrated_weight_mean) if integrated_weight_mean else 0,
        }

        # Add per-stream statistics
        eps = 1e-8
        for i in range(self.num_streams):
            result[f'stream_{i}_weight_norm'] = stream_weight_norms[i]
            result[f'stream_{i}_weight_mean'] = np.mean(stream_weight_means[i]) if stream_weight_means[i] else 0

        # Compute ratios (compare each stream to stream0 and integrated)
        for i in range(self.num_streams):
            if i > 0:
                result[f'stream_{i}_to_stream0_weight_ratio'] = stream_weight_norms[i] / (stream_weight_norms[0] + eps)
            result[f'stream_{i}_to_integrated_weight_ratio'] = stream_weight_norms[i] / (integrated_weight_norm + eps)

        return result

    def compute_stream_overfitting_indicators(self,
                                             train_loss: float,
                                             val_loss: float,
                                             train_acc: float,
                                             val_acc: float,
                                             train_loader: torch.utils.data.DataLoader,
                                             val_loader: torch.utils.data.DataLoader,
                                             max_batches: int = 5) -> dict[str, float]:
        """
        Compute overfitting indicators for each of the N streams separately.

        For LINet: Tests individual stream pathways by running stream-specific forward passes.
        This shows how well each stream has learned to extract useful features.

        Args:
            train_loss: Overall training loss
            val_loss: Overall validation loss
            train_acc: Overall training accuracy
            val_acc: Overall validation accuracy
            train_loader: Training data loader
            val_loader: Validation data loader
            max_batches: Maximum number of batches to evaluate (for speed)

        Returns:
            Dictionary with overfitting indicators per stream (including integrated)
        """
        self.model.eval()

        # Accumulators for train set (per stream)
        stream_train_correct = [0] * self.num_streams
        stream_train_total = [0] * self.num_streams
        stream_train_loss_sum = [0.0] * self.num_streams
        integrated_train_correct = 0
        integrated_train_total = 0
        integrated_train_loss_sum = 0.0

        # Accumulators for val set (per stream)
        stream_val_correct = [0] * self.num_streams
        stream_val_total = [0] * self.num_streams
        stream_val_loss_sum = [0.0] * self.num_streams
        integrated_val_correct = 0
        integrated_val_total = 0
        integrated_val_loss_sum = 0.0

        with torch.no_grad():
            # Evaluate on train set
            train_batches_processed = 0
            train_iter = iter(train_loader)
            for _ in range(max_batches):
                try:
                    batch_data = next(train_iter)
                except StopIteration:
                    break

                # Unpack batch data (expects N streams + targets)
                *stream_inputs, targets_train = batch_data
                stream_inputs = [s.to(self.model.device) for s in stream_inputs]
                targets_train = targets_train.to(self.model.device)

                # Evaluate each stream individually
                for i in range(self.num_streams):
                    # Use the unified _forward_stream_pathway method with stream index
                    stream_features = self.model._forward_stream_pathway(i, stream_inputs[i])
                    stream_features = self.model.dropout(stream_features)
                    stream_out = self.model.fc(stream_features)
                    stream_train_loss_sum[i] += self.model.criterion(stream_out, targets_train).item()
                    stream_train_correct[i] += (stream_out.argmax(1) == targets_train).sum().item()
                    stream_train_total[i] += targets_train.size(0)

                # Integrated pathway performance
                integrated_train_features = self.model._forward_integrated_pathway(stream_inputs)
                integrated_train_features = self.model.dropout(integrated_train_features)
                integrated_train_out = self.model.fc(integrated_train_features)
                integrated_train_loss_sum += self.model.criterion(integrated_train_out, targets_train).item()
                integrated_train_correct += (integrated_train_out.argmax(1) == targets_train).sum().item()
                integrated_train_total += targets_train.size(0)

                train_batches_processed += 1

            # Evaluate on val set
            val_batches_processed = 0
            val_iter = iter(val_loader)
            for _ in range(max_batches):
                try:
                    batch_data = next(val_iter)
                except StopIteration:
                    break

                # Unpack batch data (expects N streams + targets)
                *stream_inputs, targets_val = batch_data
                stream_inputs = [s.to(self.model.device) for s in stream_inputs]
                targets_val = targets_val.to(self.model.device)

                # Evaluate each stream individually
                for i in range(self.num_streams):
                    # Use the unified _forward_stream_pathway method with stream index
                    stream_features = self.model._forward_stream_pathway(i, stream_inputs[i])
                    stream_features = self.model.dropout(stream_features)
                    stream_out = self.model.fc(stream_features)
                    stream_val_loss_sum[i] += self.model.criterion(stream_out, targets_val).item()
                    stream_val_correct[i] += (stream_out.argmax(1) == targets_val).sum().item()
                    stream_val_total[i] += targets_val.size(0)

                # Integrated pathway performance
                integrated_val_features = self.model._forward_integrated_pathway(stream_inputs)
                integrated_val_features = self.model.dropout(integrated_val_features)
                integrated_val_out = self.model.fc(integrated_val_features)
                integrated_val_loss_sum += self.model.criterion(integrated_val_out, targets_val).item()
                integrated_val_correct += (integrated_val_out.argmax(1) == targets_val).sum().item()
                integrated_val_total += targets_val.size(0)

                val_batches_processed += 1

        # Calculate averages for each stream
        stream_train_losses = [stream_train_loss_sum[i] / max(train_batches_processed, 1) for i in range(self.num_streams)]
        stream_train_accs = [stream_train_correct[i] / max(stream_train_total[i], 1) for i in range(self.num_streams)]
        stream_val_losses = [stream_val_loss_sum[i] / max(val_batches_processed, 1) for i in range(self.num_streams)]
        stream_val_accs = [stream_val_correct[i] / max(stream_val_total[i], 1) for i in range(self.num_streams)]

        integrated_train_loss = integrated_train_loss_sum / max(train_batches_processed, 1)
        integrated_train_acc = integrated_train_correct / max(integrated_train_total, 1)
        integrated_val_loss = integrated_val_loss_sum / max(val_batches_processed, 1)
        integrated_val_acc = integrated_val_correct / max(integrated_val_total, 1)

        # Calculate overfitting indicators for each stream
        stream_loss_gaps = [stream_val_losses[i] - stream_train_losses[i] for i in range(self.num_streams)]
        stream_acc_gaps = [stream_train_accs[i] - stream_val_accs[i] for i in range(self.num_streams)]
        integrated_loss_gap = integrated_val_loss - integrated_train_loss
        integrated_acc_gap = integrated_train_acc - integrated_val_acc
        full_loss_gap = val_loss - train_loss
        full_acc_gap = train_acc - val_acc

        # Build result dictionary
        result = {
            # Integrated stream indicators
            'integrated_train_loss': integrated_train_loss,
            'integrated_val_loss': integrated_val_loss,
            'integrated_loss_gap': integrated_loss_gap,
            'integrated_train_acc': integrated_train_acc,
            'integrated_val_acc': integrated_val_acc,
            'integrated_acc_gap': integrated_acc_gap,
            'integrated_overfitting_score': integrated_loss_gap + integrated_acc_gap,

            # Full model indicators
            'full_model_loss_gap': full_loss_gap,
            'full_model_acc_gap': full_acc_gap,
        }

        # Add per-stream indicators
        for i in range(self.num_streams):
            result[f'stream_{i}_train_loss'] = stream_train_losses[i]
            result[f'stream_{i}_val_loss'] = stream_val_losses[i]
            result[f'stream_{i}_loss_gap'] = stream_loss_gaps[i]
            result[f'stream_{i}_train_acc'] = stream_train_accs[i]
            result[f'stream_{i}_val_acc'] = stream_val_accs[i]
            result[f'stream_{i}_acc_gap'] = stream_acc_gaps[i]
            result[f'stream_{i}_overfitting_score'] = stream_loss_gaps[i] + stream_acc_gaps[i]

        # Add stream comparison ratios (compare each stream to stream0)
        eps = 1e-8
        for i in range(1, self.num_streams):
            stream0_score = stream_loss_gaps[0] + stream_acc_gaps[0]
            stream_i_score = stream_loss_gaps[i] + stream_acc_gaps[i]
            result[f'stream_{i}_vs_stream0_overfit_ratio'] = stream_i_score / (stream0_score + eps)

        # Add stream vs integrated ratios
        for i in range(self.num_streams):
            stream_i_score = stream_loss_gaps[i] + stream_acc_gaps[i]
            integrated_score = integrated_loss_gap + integrated_acc_gap
            result[f'integrated_vs_stream{i}_overfit_ratio'] = integrated_score / (stream_i_score + eps)

        return result

    def log_metrics(self, epoch: int, metrics: dict[str, float]):
        """
        Log metrics for this epoch.

        Args:
            epoch: Current epoch number
            metrics: Dictionary of metrics to log
        """
        self.history['epoch'].append(epoch)
        for key, value in metrics.items():
            self.history[key].append(value)

    def get_summary(self) -> str:
        """
        Get a summary of stream behavior.

        Returns:
            Formatted string with summary
        """
        if not self.history['epoch']:
            return "No monitoring data available"

        summary = []
        summary.append("=" * 70)
        summary.append("LINet Stream Monitoring Summary")
        summary.append("=" * 70)

        # Gradient analysis
        if f'stream_0_grad_norm' in self.history:
            summary.append(f"\nGradient Norms (average):")
            stream_grad_avgs = []
            for i in range(self.num_streams):
                key = f'stream_{i}_grad_norm'
                if key in self.history:
                    avg = np.mean(self.history[key])
                    stream_grad_avgs.append(avg)
                    summary.append(f"  Stream{i}: {avg:.4f}")

            if 'integrated_grad_norm' in self.history:
                avg_int_grad = np.mean(self.history['integrated_grad_norm'])
                summary.append(f"  Integrated: {avg_int_grad:.4f}")

                # Show ratios
                for i in range(1, self.num_streams):
                    if len(stream_grad_avgs) > i:
                        ratio = stream_grad_avgs[i] / (stream_grad_avgs[0] + 1e-8)
                        summary.append(f"  Ratio (S{i}/S0): {ratio:.2f}")

        # Weight analysis
        if f'stream_0_weight_norm' in self.history:
            summary.append(f"\nWeight Norms (latest):")
            for i in range(self.num_streams):
                key = f'stream_{i}_weight_norm'
                if key in self.history:
                    latest = self.history[key][-1]
                    summary.append(f"  Stream{i}: {latest:.4f}")

            if 'integrated_weight_norm' in self.history:
                latest_int_weight = self.history['integrated_weight_norm'][-1]
                summary.append(f"  Integrated: {latest_int_weight:.4f}")

        # Overfitting analysis
        if f'stream_0_overfitting_score' in self.history:
            summary.append(f"\nOverfitting Indicators (latest):")
            stream_overfit_scores = []
            for i in range(self.num_streams):
                key = f'stream_{i}_overfitting_score'
                if key in self.history:
                    latest = self.history[key][-1]
                    stream_overfit_scores.append(latest)
                    summary.append(f"  Stream{i} score: {latest:.4f}")

            # Check for imbalances
            warnings = []
            for i in range(1, len(stream_overfit_scores)):
                if stream_overfit_scores[i] > stream_overfit_scores[0] * 1.5:
                    warnings.append(f"Stream{i} overfitting more than Stream0")
                elif stream_overfit_scores[0] > stream_overfit_scores[i] * 1.5:
                    warnings.append(f"Stream0 overfitting more than Stream{i}")

            if warnings:
                for warning in warnings:
                    summary.append(f"  WARNING: {warning}")
            else:
                summary.append(f"  Balanced overfitting")

        summary.append("=" * 70)
        return "\n".join(summary)

    def get_recommendations(self) -> list[str]:
        """
        Get recommendations based on monitoring data.

        Returns:
            List of recommendation strings
        """
        recommendations = []

        if not self.history['epoch']:
            return ["Insufficient data for recommendations"]

        # Gradient recommendations
        if f'stream_0_grad_norm' in self.history and len(self.history['stream_0_grad_norm']) > 0:
            # Calculate averages for last 5 epochs
            stream_grad_avgs = []
            for i in range(self.num_streams):
                key = f'stream_{i}_grad_norm'
                if key in self.history:
                    avg = np.mean(self.history[key][-5:])
                    stream_grad_avgs.append(avg)

            if 'integrated_grad_norm' in self.history:
                avg_int_grad = np.mean(self.history['integrated_grad_norm'][-5:])

                # Check stream-to-stream0 ratios
                for i in range(1, len(stream_grad_avgs)):
                    ratio = stream_grad_avgs[i] / (stream_grad_avgs[0] + 1e-8)
                    if ratio > 3.0:
                        recommendations.append(f"Stream{i} gradients {ratio:.1f}x larger than Stream0 - consider stream-specific LR")
                    elif ratio < 0.33:
                        recommendations.append(f"Stream0 gradients {1/ratio:.1f}x larger than Stream{i} - consider stream-specific LR")

                # Check stream-to-integrated ratios
                for i in range(len(stream_grad_avgs)):
                    ratio = stream_grad_avgs[i] / (avg_int_grad + 1e-8)
                    if ratio > 3.0:
                        recommendations.append(f"Stream{i} gradients {ratio:.1f}x larger than Integrated - check integration weights")
                    elif ratio < 0.33:
                        recommendations.append(f"Integrated gradients {1/ratio:.1f}x larger than Stream{i} - check integration weights")

        # Overfitting recommendations
        if f'stream_0_overfitting_score' in self.history and len(self.history['stream_0_overfitting_score']) > 0:
            for i in range(self.num_streams):
                key = f'stream_{i}_overfitting_score'
                if key in self.history:
                    latest_overfit = self.history[key][-1]
                    if latest_overfit > 0.2:
                        recommendations.append(f"Stream{i} overfitting (score: {latest_overfit:.2f}) - increase regularization")

        if not recommendations:
            recommendations.append("All metrics look healthy")

        return recommendations


def create_stream_monitor(model: nn.Module) -> StreamMonitor:
    """
    Factory function to create a StreamMonitor.

    Args:
        model: The LINet model to monitor

    Returns:
        StreamMonitor instance
    """
    return StreamMonitor(model)
