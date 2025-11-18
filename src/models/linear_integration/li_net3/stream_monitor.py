"""
Stream-specific monitoring utilities for Linear Integration Neural Networks (LINet).

Provides tools to monitor individual stream learning, detect overfitting,
and gain insights into 3-stream behavior during training.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict


class StreamMonitor:
    """
    Monitor individual stream behavior during LINet training.

    Tracks:
    - Stream-specific gradients (3 streams)
    - Stream-specific activations (3 streams)
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

    def compute_stream_gradients(self) -> Dict[str, float]:
        """
        Compute gradient statistics for each of the 3 streams.

        Returns:
            Dictionary with gradient norms and statistics per stream
        """
        stream1_grad_norm = 0.0
        stream2_grad_norm = 0.0
        integrated_grad_norm = 0.0
        shared_grad_norm = 0.0

        stream1_grad_max = 0.0
        stream2_grad_max = 0.0
        integrated_grad_max = 0.0

        stream1_params = 0
        stream2_params = 0
        integrated_params = 0
        shared_params = 0

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_max = param.grad.abs().max().item()

                if 'stream1' in name:
                    stream1_grad_norm += grad_norm ** 2
                    stream1_grad_max = max(stream1_grad_max, grad_max)
                    stream1_params += 1
                elif 'stream2' in name:
                    stream2_grad_norm += grad_norm ** 2
                    stream2_grad_max = max(stream2_grad_max, grad_max)
                    stream2_params += 1
                elif 'integrated' in name or 'integration' in name:
                    integrated_grad_norm += grad_norm ** 2
                    integrated_grad_max = max(integrated_grad_max, grad_max)
                    integrated_params += 1
                else:
                    shared_grad_norm += grad_norm ** 2
                    shared_params += 1

        stream1_grad_norm = np.sqrt(stream1_grad_norm)
        stream2_grad_norm = np.sqrt(stream2_grad_norm)
        integrated_grad_norm = np.sqrt(integrated_grad_norm)
        shared_grad_norm = np.sqrt(shared_grad_norm)

        eps = 1e-8
        return {
            'stream1_grad_norm': stream1_grad_norm,
            'stream2_grad_norm': stream2_grad_norm,
            'integrated_grad_norm': integrated_grad_norm,
            'shared_grad_norm': shared_grad_norm,
            'stream1_grad_max': stream1_grad_max,
            'stream2_grad_max': stream2_grad_max,
            'integrated_grad_max': integrated_grad_max,
            'stream1_to_stream2_ratio': stream1_grad_norm / (stream2_grad_norm + eps),
            'stream1_to_integrated_ratio': stream1_grad_norm / (integrated_grad_norm + eps),
            'stream2_to_integrated_ratio': stream2_grad_norm / (integrated_grad_norm + eps),
            'stream1_params': stream1_params,
            'stream2_params': stream2_params,
            'integrated_params': integrated_params,
            'shared_params': shared_params
        }

    def compute_stream_weights(self) -> Dict[str, float]:
        """
        Compute weight statistics for each of the 3 streams.

        Returns:
            Dictionary with weight norms and statistics per stream
        """
        stream1_weight_norm = 0.0
        stream2_weight_norm = 0.0
        integrated_weight_norm = 0.0

        stream1_weight_mean = []
        stream2_weight_mean = []
        integrated_weight_mean = []

        for name, param in self.model.named_parameters():
            if 'weight' in name:
                if 'stream1' in name:
                    stream1_weight_norm += param.norm().item() ** 2
                    stream1_weight_mean.append(param.mean().item())
                elif 'stream2' in name:
                    stream2_weight_norm += param.norm().item() ** 2
                    stream2_weight_mean.append(param.mean().item())
                elif 'integrated' in name or 'integration' in name:
                    integrated_weight_norm += param.norm().item() ** 2
                    integrated_weight_mean.append(param.mean().item())

        stream1_weight_norm = np.sqrt(stream1_weight_norm)
        stream2_weight_norm = np.sqrt(stream2_weight_norm)
        integrated_weight_norm = np.sqrt(integrated_weight_norm)

        eps = 1e-8
        return {
            'stream1_weight_norm': stream1_weight_norm,
            'stream2_weight_norm': stream2_weight_norm,
            'integrated_weight_norm': integrated_weight_norm,
            'stream1_weight_mean': np.mean(stream1_weight_mean) if stream1_weight_mean else 0,
            'stream2_weight_mean': np.mean(stream2_weight_mean) if stream2_weight_mean else 0,
            'integrated_weight_mean': np.mean(integrated_weight_mean) if integrated_weight_mean else 0,
            'stream1_to_stream2_weight_ratio': stream1_weight_norm / (stream2_weight_norm + eps),
            'stream1_to_integrated_weight_ratio': stream1_weight_norm / (integrated_weight_norm + eps),
        }

    def compute_stream_overfitting_indicators(self,
                                             train_loss: float,
                                             val_loss: float,
                                             train_acc: float,
                                             val_acc: float,
                                             train_loader: torch.utils.data.DataLoader,
                                             val_loader: torch.utils.data.DataLoader,
                                             max_batches: int = 5) -> Dict[str, float]:
        """
        Compute overfitting indicators for each of the 3 streams separately.

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

        # Accumulators for train set
        stream1_train_correct = 0
        stream1_train_total = 0
        stream1_train_loss_sum = 0.0
        stream2_train_correct = 0
        stream2_train_total = 0
        stream2_train_loss_sum = 0.0
        integrated_train_correct = 0
        integrated_train_total = 0
        integrated_train_loss_sum = 0.0

        # Accumulators for val set
        stream1_val_correct = 0
        stream1_val_total = 0
        stream1_val_loss_sum = 0.0
        stream2_val_correct = 0
        stream2_val_total = 0
        stream2_val_loss_sum = 0.0
        integrated_val_correct = 0
        integrated_val_total = 0
        integrated_val_loss_sum = 0.0

        with torch.no_grad():
            # Evaluate on train set
            train_batches_processed = 0
            train_iter = iter(train_loader)
            for _ in range(max_batches):
                try:
                    stream1_train, stream2_train, targets_train = next(train_iter)
                except StopIteration:
                    break

                stream1_train = stream1_train.to(self.model.device)
                stream2_train = stream2_train.to(self.model.device)
                targets_train = targets_train.to(self.model.device)

                # Stream1 only performance
                stream1_train_features = self.model._forward_stream1_pathway(stream1_train)
                stream1_train_features = self.model.dropout(stream1_train_features)
                stream1_train_out = self.model.fc(stream1_train_features)
                stream1_train_loss_sum += self.model.criterion(stream1_train_out, targets_train).item()
                stream1_train_correct += (stream1_train_out.argmax(1) == targets_train).sum().item()
                stream1_train_total += targets_train.size(0)

                # Stream2 only performance
                stream2_train_features = self.model._forward_stream2_pathway(stream2_train)
                stream2_train_features = self.model.dropout(stream2_train_features)
                stream2_train_out = self.model.fc(stream2_train_features)
                stream2_train_loss_sum += self.model.criterion(stream2_train_out, targets_train).item()
                stream2_train_correct += (stream2_train_out.argmax(1) == targets_train).sum().item()
                stream2_train_total += targets_train.size(0)

                # Integrated pathway performance
                integrated_train_features = self.model._forward_integrated_pathway(stream1_train, stream2_train)
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
                    stream1_val, stream2_val, targets_val = next(val_iter)
                except StopIteration:
                    break

                stream1_val = stream1_val.to(self.model.device)
                stream2_val = stream2_val.to(self.model.device)
                targets_val = targets_val.to(self.model.device)

                # Stream1 only performance
                stream1_val_features = self.model._forward_stream1_pathway(stream1_val)
                stream1_val_features = self.model.dropout(stream1_val_features)
                stream1_val_out = self.model.fc(stream1_val_features)
                stream1_val_loss_sum += self.model.criterion(stream1_val_out, targets_val).item()
                stream1_val_correct += (stream1_val_out.argmax(1) == targets_val).sum().item()
                stream1_val_total += targets_val.size(0)

                # Stream2 only performance
                stream2_val_features = self.model._forward_stream2_pathway(stream2_val)
                stream2_val_features = self.model.dropout(stream2_val_features)
                stream2_val_out = self.model.fc(stream2_val_features)
                stream2_val_loss_sum += self.model.criterion(stream2_val_out, targets_val).item()
                stream2_val_correct += (stream2_val_out.argmax(1) == targets_val).sum().item()
                stream2_val_total += targets_val.size(0)

                # Integrated pathway performance
                integrated_val_features = self.model._forward_integrated_pathway(stream1_val, stream2_val)
                integrated_val_features = self.model.dropout(integrated_val_features)
                integrated_val_out = self.model.fc(integrated_val_features)
                integrated_val_loss_sum += self.model.criterion(integrated_val_out, targets_val).item()
                integrated_val_correct += (integrated_val_out.argmax(1) == targets_val).sum().item()
                integrated_val_total += targets_val.size(0)

                val_batches_processed += 1

        # Calculate averages
        stream1_train_loss = stream1_train_loss_sum / max(train_batches_processed, 1)
        stream1_train_acc = stream1_train_correct / max(stream1_train_total, 1)
        stream2_train_loss = stream2_train_loss_sum / max(train_batches_processed, 1)
        stream2_train_acc = stream2_train_correct / max(stream2_train_total, 1)
        integrated_train_loss = integrated_train_loss_sum / max(train_batches_processed, 1)
        integrated_train_acc = integrated_train_correct / max(integrated_train_total, 1)

        stream1_val_loss = stream1_val_loss_sum / max(val_batches_processed, 1)
        stream1_val_acc = stream1_val_correct / max(stream1_val_total, 1)
        stream2_val_loss = stream2_val_loss_sum / max(val_batches_processed, 1)
        stream2_val_acc = stream2_val_correct / max(stream2_val_total, 1)
        integrated_val_loss = integrated_val_loss_sum / max(val_batches_processed, 1)
        integrated_val_acc = integrated_val_correct / max(integrated_val_total, 1)

        # Calculate overfitting indicators
        stream1_loss_gap = stream1_val_loss - stream1_train_loss
        stream2_loss_gap = stream2_val_loss - stream2_train_loss
        integrated_loss_gap = integrated_val_loss - integrated_train_loss
        full_loss_gap = val_loss - train_loss

        stream1_acc_gap = stream1_train_acc - stream1_val_acc
        stream2_acc_gap = stream2_train_acc - stream2_val_acc
        integrated_acc_gap = integrated_train_acc - integrated_val_acc
        full_acc_gap = train_acc - val_acc

        return {
            # Stream1 indicators
            'stream1_train_loss': stream1_train_loss,
            'stream1_val_loss': stream1_val_loss,
            'stream1_loss_gap': stream1_loss_gap,
            'stream1_train_acc': stream1_train_acc,
            'stream1_val_acc': stream1_val_acc,
            'stream1_acc_gap': stream1_acc_gap,
            'stream1_overfitting_score': stream1_loss_gap + stream1_acc_gap,

            # Stream2 indicators
            'stream2_train_loss': stream2_train_loss,
            'stream2_val_loss': stream2_val_loss,
            'stream2_loss_gap': stream2_loss_gap,
            'stream2_train_acc': stream2_train_acc,
            'stream2_val_acc': stream2_val_acc,
            'stream2_acc_gap': stream2_acc_gap,
            'stream2_overfitting_score': stream2_loss_gap + stream2_acc_gap,

            # Integrated stream indicators
            'integrated_train_loss': integrated_train_loss,
            'integrated_val_loss': integrated_val_loss,
            'integrated_loss_gap': integrated_loss_gap,
            'integrated_train_acc': integrated_train_acc,
            'integrated_val_acc': integrated_val_acc,
            'integrated_acc_gap': integrated_acc_gap,
            'integrated_overfitting_score': integrated_loss_gap + integrated_acc_gap,

            # Comparison
            'stream1_vs_stream2_overfit_ratio': (stream1_loss_gap + stream1_acc_gap) / (stream2_loss_gap + stream2_acc_gap + 1e-8),
            'integrated_vs_stream1_overfit_ratio': (integrated_loss_gap + integrated_acc_gap) / (stream1_loss_gap + stream1_acc_gap + 1e-8),
            'full_model_loss_gap': full_loss_gap,
            'full_model_acc_gap': full_acc_gap,
        }

    def log_metrics(self, epoch: int, metrics: Dict[str, float]):
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
        if 'stream1_grad_norm' in self.history:
            avg_s1_grad = np.mean(self.history['stream1_grad_norm'])
            avg_s2_grad = np.mean(self.history['stream2_grad_norm'])
            avg_int_grad = np.mean(self.history['integrated_grad_norm'])
            summary.append(f"\nGradient Norms (average):")
            summary.append(f"  Stream1 (RGB):  {avg_s1_grad:.4f}")
            summary.append(f"  Stream2 (Depth): {avg_s2_grad:.4f}")
            summary.append(f"  Integrated:     {avg_int_grad:.4f}")
            summary.append(f"  Ratio (S1/S2):  {avg_s1_grad/(avg_s2_grad+1e-8):.2f}")
            summary.append(f"  Ratio (S1/Int): {avg_s1_grad/(avg_int_grad+1e-8):.2f}")

        # Weight analysis
        if 'stream1_weight_norm' in self.history:
            latest_s1_weight = self.history['stream1_weight_norm'][-1]
            latest_s2_weight = self.history['stream2_weight_norm'][-1]
            latest_int_weight = self.history['integrated_weight_norm'][-1]
            summary.append(f"\nWeight Norms (latest):")
            summary.append(f"  Stream1 (RGB):  {latest_s1_weight:.4f}")
            summary.append(f"  Stream2 (Depth): {latest_s2_weight:.4f}")
            summary.append(f"  Integrated:     {latest_int_weight:.4f}")

        # Overfitting analysis
        if 'stream1_overfitting_score' in self.history:
            latest_s1_overfit = self.history['stream1_overfitting_score'][-1]
            latest_s2_overfit = self.history['stream2_overfitting_score'][-1]
            summary.append(f"\nOverfitting Indicators (latest):")
            summary.append(f"  Stream1 score: {latest_s1_overfit:.4f}")
            summary.append(f"  Stream2 score: {latest_s2_overfit:.4f}")

            if latest_s1_overfit > latest_s2_overfit * 1.5:
                summary.append(f"  ⚠️  Stream1 overfitting more than Stream2")
            elif latest_s2_overfit > latest_s1_overfit * 1.5:
                summary.append(f"  ⚠️  Stream2 overfitting more than Stream1")
            else:
                summary.append(f"  ✓ Balanced overfitting")

        summary.append("=" * 70)
        return "\n".join(summary)

    def get_recommendations(self) -> List[str]:
        """
        Get recommendations based on monitoring data.

        Returns:
            List of recommendation strings
        """
        recommendations = []

        if not self.history['epoch']:
            return ["Insufficient data for recommendations"]

        # Gradient recommendations
        if 'stream1_grad_norm' in self.history and len(self.history['stream1_grad_norm']) > 0:
            avg_s1_grad = np.mean(self.history['stream1_grad_norm'][-5:])  # Last 5 epochs
            avg_s2_grad = np.mean(self.history['stream2_grad_norm'][-5:])
            avg_int_grad = np.mean(self.history['integrated_grad_norm'][-5:])

            s1_to_s2_ratio = avg_s1_grad / (avg_s2_grad + 1e-8)
            s1_to_int_ratio = avg_s1_grad / (avg_int_grad + 1e-8)

            if s1_to_s2_ratio > 3.0:
                recommendations.append(f"⚠️  Stream1 gradients {s1_to_s2_ratio:.1f}x larger than Stream2 - consider stream-specific LR")
            elif s1_to_s2_ratio < 0.33:
                recommendations.append(f"⚠️  Stream2 gradients {1/s1_to_s2_ratio:.1f}x larger than Stream1 - consider stream-specific LR")

            if s1_to_int_ratio > 3.0:
                recommendations.append(f"⚠️  Stream1 gradients {s1_to_int_ratio:.1f}x larger than Integrated - check integration weights")
            elif s1_to_int_ratio < 0.33:
                recommendations.append(f"⚠️  Integrated gradients {1/s1_to_int_ratio:.1f}x larger than Stream1 - check integration weights")

        # Overfitting recommendations
        if 'stream1_overfitting_score' in self.history and len(self.history['stream1_overfitting_score']) > 0:
            latest_s1_overfit = self.history['stream1_overfitting_score'][-1]
            latest_s2_overfit = self.history['stream2_overfitting_score'][-1]

            if latest_s1_overfit > 0.2:
                recommendations.append(f"⚠️  Stream1 overfitting (score: {latest_s1_overfit:.2f}) - increase regularization")

            if latest_s2_overfit > 0.2:
                recommendations.append(f"⚠️  Stream2 overfitting (score: {latest_s2_overfit:.2f}) - increase regularization")

        if not recommendations:
            recommendations.append("✓ All metrics look healthy")

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
