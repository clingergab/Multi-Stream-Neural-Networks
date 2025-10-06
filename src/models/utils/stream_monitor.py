"""
Stream-specific monitoring utilities for multi-stream models.

Provides tools to monitor individual stream learning, detect overfitting,
and gain insights into stream-specific behavior during training.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict


class StreamMonitor:
    """
    Monitor individual stream behavior during training.

    Tracks:
    - Stream-specific gradients
    - Stream-specific activations
    - Stream-specific loss contributions
    - Overfitting indicators per stream
    - Learning rate effectiveness
    """

    def __init__(self, model: nn.Module):
        """
        Initialize stream monitor.

        Args:
            model: The multi-stream model to monitor
        """
        self.model = model
        self.history = defaultdict(list)

    def compute_stream_gradients(self) -> Dict[str, float]:
        """
        Compute gradient statistics for each stream.

        Returns:
            Dictionary with gradient norms and statistics per stream
        """
        stream1_grad_norm = 0.0
        stream2_grad_norm = 0.0
        shared_grad_norm = 0.0

        stream1_grad_max = 0.0
        stream2_grad_max = 0.0

        stream1_params = 0
        stream2_params = 0
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
                else:
                    shared_grad_norm += grad_norm ** 2
                    shared_params += 1

        stream1_grad_norm = np.sqrt(stream1_grad_norm)
        stream2_grad_norm = np.sqrt(stream2_grad_norm)
        shared_grad_norm = np.sqrt(shared_grad_norm)

        return {
            'stream1_grad_norm': stream1_grad_norm,
            'stream2_grad_norm': stream2_grad_norm,
            'shared_grad_norm': shared_grad_norm,
            'stream1_grad_max': stream1_grad_max,
            'stream2_grad_max': stream2_grad_max,
            'stream1_to_stream2_ratio': stream1_grad_norm / stream2_grad_norm if stream2_grad_norm > 0 else 0,
            'stream1_params': stream1_params,
            'stream2_params': stream2_params,
            'shared_params': shared_params
        }

    def compute_stream_weights(self) -> Dict[str, float]:
        """
        Compute weight statistics for each stream.

        Returns:
            Dictionary with weight norms and statistics per stream
        """
        stream1_weight_norm = 0.0
        stream2_weight_norm = 0.0

        stream1_weight_mean = []
        stream2_weight_mean = []

        for name, param in self.model.named_parameters():
            if 'stream1' in name and 'weight' in name:
                stream1_weight_norm += param.norm().item() ** 2
                stream1_weight_mean.append(param.mean().item())
            elif 'stream2' in name and 'weight' in name:
                stream2_weight_norm += param.norm().item() ** 2
                stream2_weight_mean.append(param.mean().item())

        stream1_weight_norm = np.sqrt(stream1_weight_norm)
        stream2_weight_norm = np.sqrt(stream2_weight_norm)

        return {
            'stream1_weight_norm': stream1_weight_norm,
            'stream2_weight_norm': stream2_weight_norm,
            'stream1_weight_mean': np.mean(stream1_weight_mean) if stream1_weight_mean else 0,
            'stream2_weight_mean': np.mean(stream2_weight_mean) if stream2_weight_mean else 0,
            'weight_norm_ratio': stream1_weight_norm / stream2_weight_norm if stream2_weight_norm > 0 else 0
        }

    def compute_stream_activations(self, stream1_input: torch.Tensor,
                                   stream2_input: torch.Tensor) -> Dict[str, float]:
        """
        Compute activation statistics for each stream.

        Args:
            stream1_input: Input to stream1
            stream2_input: Input to stream2

        Returns:
            Dictionary with activation statistics per stream
        """
        activations = {}

        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    # Multi-stream layer output
                    stream1_out, stream2_out = output
                    activations[f'{name}_stream1_mean'] = stream1_out.mean().item()
                    activations[f'{name}_stream1_std'] = stream1_out.std().item()
                    activations[f'{name}_stream2_mean'] = stream2_out.mean().item()
                    activations[f'{name}_stream2_std'] = stream2_out.std().item()
                else:
                    # Regular layer output
                    activations[f'{name}_mean'] = output.mean().item()
                    activations[f'{name}_std'] = output.std().item()
            return hook

        # Register hooks
        hooks = []
        for name, module in self.model.named_modules():
            if 'layer' in name and len(name.split('.')) == 1:  # Only top-level layers
                hooks.append(module.register_forward_hook(hook_fn(name)))

        # Forward pass
        self.model.eval()
        with torch.no_grad():
            _ = self.model(stream1_input, stream2_input)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return activations

    def compute_stream_overfitting_indicators(self,
                                             train_loss: float,
                                             val_loss: float,
                                             train_acc: float,
                                             val_acc: float,
                                             train_loader: torch.utils.data.DataLoader,
                                             val_loader: torch.utils.data.DataLoader,
                                             max_batches: int = 5,
                                             train_loader_no_aug: torch.utils.data.DataLoader = None) -> Dict[str, float]:
        """
        Compute overfitting indicators for each stream separately.

        IMPORTANT: For fair comparison, train_loader_no_aug should be provided with
        the same data as train_loader but WITHOUT augmentation. Otherwise, metrics
        will be misleading (augmented train data is harder than val data).

        Args:
            train_loss: Training loss
            val_loss: Validation loss
            train_acc: Training accuracy
            val_acc: Validation accuracy
            train_loader: Training data loader (WITH augmentation)
            val_loader: Validation data loader (no augmentation)
            max_batches: Maximum number of batches to evaluate (for speed)
            train_loader_no_aug: Training data loader WITHOUT augmentation (recommended!)
                If None, will use train_loader (but results may be misleading)

        Returns:
            Dictionary with overfitting indicators per stream
        """
        # Note: For weighted/gated fusion, isolated stream testing measures how much
        # the FC layer can classify using each stream alone (with zeros for the other).
        # Low accuracy means the FC layer didn't learn to use that stream's features.
        # This IS meaningful - it shows stream contribution/reliance!

        # Use non-augmented train loader if provided, otherwise fall back to train_loader
        effective_train_loader = train_loader_no_aug if train_loader_no_aug is not None else train_loader

        fusion_type = self.model.fusion.__class__.__name__
        self.model.eval()

        # Accumulators for train set
        stream1_train_correct = 0
        stream1_train_total = 0
        stream1_train_loss_sum = 0.0
        stream2_train_correct = 0
        stream2_train_total = 0
        stream2_train_loss_sum = 0.0

        # Accumulators for val set
        stream1_val_correct = 0
        stream1_val_total = 0
        stream1_val_loss_sum = 0.0
        stream2_val_correct = 0
        stream2_val_total = 0
        stream2_val_loss_sum = 0.0

        with torch.no_grad():
            # Evaluate on train set (multiple batches)
            # Create fresh iterator to avoid caching issues between epochs
            train_batches_processed = 0
            train_iter = iter(effective_train_loader)
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
                stream2_dummy_train = torch.zeros_like(stream1_train_features)
                stream1_train_fused = self.model.fusion(stream1_train_features, stream2_dummy_train)
                stream1_train_fused = self.model.dropout(stream1_train_fused)
                stream1_train_out = self.model.fc(stream1_train_fused)
                stream1_train_loss_sum += self.model.criterion(stream1_train_out, targets_train).item()
                stream1_train_correct += (stream1_train_out.argmax(1) == targets_train).sum().item()
                stream1_train_total += targets_train.size(0)

                # Stream2 only performance
                stream2_train_features = self.model._forward_stream2_pathway(stream2_train)
                stream1_dummy_train = torch.zeros_like(stream2_train_features)
                stream2_train_fused = self.model.fusion(stream1_dummy_train, stream2_train_features)
                stream2_train_fused = self.model.dropout(stream2_train_fused)
                stream2_train_out = self.model.fc(stream2_train_fused)
                stream2_train_loss_sum += self.model.criterion(stream2_train_out, targets_train).item()
                stream2_train_correct += (stream2_train_out.argmax(1) == targets_train).sum().item()
                stream2_train_total += targets_train.size(0)

                train_batches_processed += 1

            # Evaluate on val set (multiple batches)
            # Create fresh iterator to avoid caching issues between epochs
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
                stream2_dummy_val = torch.zeros_like(stream1_val_features)
                stream1_val_fused = self.model.fusion(stream1_val_features, stream2_dummy_val)
                stream1_val_fused = self.model.dropout(stream1_val_fused)
                stream1_val_out = self.model.fc(stream1_val_fused)
                stream1_val_loss_sum += self.model.criterion(stream1_val_out, targets_val).item()
                stream1_val_correct += (stream1_val_out.argmax(1) == targets_val).sum().item()
                stream1_val_total += targets_val.size(0)

                # Stream2 only performance
                stream2_val_features = self.model._forward_stream2_pathway(stream2_val)
                stream1_dummy_val = torch.zeros_like(stream2_val_features)
                stream2_val_fused = self.model.fusion(stream1_dummy_val, stream2_val_features)
                stream2_val_fused = self.model.dropout(stream2_val_fused)
                stream2_val_out = self.model.fc(stream2_val_fused)
                stream2_val_loss_sum += self.model.criterion(stream2_val_out, targets_val).item()
                stream2_val_correct += (stream2_val_out.argmax(1) == targets_val).sum().item()
                stream2_val_total += targets_val.size(0)

                val_batches_processed += 1

        # Calculate averages - use correct batch counts
        stream1_train_loss = stream1_train_loss_sum / max(train_batches_processed, 1)
        stream1_train_acc = stream1_train_correct / max(stream1_train_total, 1)
        stream2_train_loss = stream2_train_loss_sum / max(train_batches_processed, 1)
        stream2_train_acc = stream2_train_correct / max(stream2_train_total, 1)

        stream1_val_loss = stream1_val_loss_sum / max(val_batches_processed, 1)
        stream1_val_acc = stream1_val_correct / max(stream1_val_total, 1)
        stream2_val_loss = stream2_val_loss_sum / max(val_batches_processed, 1)
        stream2_val_acc = stream2_val_correct / max(stream2_val_total, 1)

        # Calculate overfitting indicators
        stream1_loss_gap = stream1_val_loss - stream1_train_loss
        stream2_loss_gap = stream2_val_loss - stream2_train_loss
        full_loss_gap = val_loss - train_loss

        stream1_acc_gap = stream1_train_acc - stream1_val_acc
        stream2_acc_gap = stream2_train_acc - stream2_val_acc
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

            # Comparison
            'stream1_vs_stream2_overfit_ratio': (stream1_loss_gap + stream1_acc_gap) / (stream2_loss_gap + stream2_acc_gap) if (stream2_loss_gap + stream2_acc_gap) > 0 else 0,
            'full_model_loss_gap': full_loss_gap,
            'full_model_acc_gap': full_acc_gap,
            'fusion_type': fusion_type
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
        summary.append("Stream Monitoring Summary")
        summary.append("=" * 70)

        # Gradient analysis
        if 'stream1_grad_norm' in self.history:
            avg_s1_grad = np.mean(self.history['stream1_grad_norm'])
            avg_s2_grad = np.mean(self.history['stream2_grad_norm'])
            summary.append(f"\nGradient Norms (average):")
            summary.append(f"  Stream1: {avg_s1_grad:.4f}")
            summary.append(f"  Stream2: {avg_s2_grad:.4f}")
            summary.append(f"  Ratio (S1/S2): {avg_s1_grad/avg_s2_grad:.2f}")

        # Weight analysis
        if 'stream1_weight_norm' in self.history:
            latest_s1_weight = self.history['stream1_weight_norm'][-1]
            latest_s2_weight = self.history['stream2_weight_norm'][-1]
            summary.append(f"\nWeight Norms (latest):")
            summary.append(f"  Stream1: {latest_s1_weight:.4f}")
            summary.append(f"  Stream2: {latest_s2_weight:.4f}")
            summary.append(f"  Ratio (S1/S2): {latest_s1_weight/latest_s2_weight:.2f}")

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

            ratio = avg_s1_grad / avg_s2_grad if avg_s2_grad > 0 else 0

            if ratio > 3.0:
                recommendations.append(f"⚠️  Stream1 gradients {ratio:.1f}x larger than Stream2 - consider reducing stream1_lr")
            elif ratio < 0.33:
                recommendations.append(f"⚠️  Stream2 gradients {1/ratio:.1f}x larger than Stream1 - consider reducing stream2_lr")
            else:
                recommendations.append(f"✓ Gradient ratio balanced ({ratio:.2f})")

        # Overfitting recommendations
        if 'stream1_overfitting_score' in self.history and len(self.history['stream1_overfitting_score']) > 0:
            latest_s1_overfit = self.history['stream1_overfitting_score'][-1]
            latest_s2_overfit = self.history['stream2_overfitting_score'][-1]

            if latest_s1_overfit > 1.0:
                recommendations.append(f"⚠️  Stream1 overfitting (score: {latest_s1_overfit:.2f}) - increase stream1_weight_decay or add dropout")

            if latest_s2_overfit > 1.0:
                recommendations.append(f"⚠️  Stream2 overfitting (score: {latest_s2_overfit:.2f}) - increase stream2_weight_decay or add dropout")

            if latest_s1_overfit > latest_s2_overfit * 2:
                recommendations.append(f"⚠️  Stream1 overfitting significantly more - boost stream1 regularization")
            elif latest_s2_overfit > latest_s1_overfit * 2:
                recommendations.append(f"⚠️  Stream2 overfitting significantly more - boost stream2 regularization")

        # Learning effectiveness
        if 'stream1_train_acc' in self.history and len(self.history['stream1_train_acc']) > 5:
            s1_acc_trend = np.mean(np.diff(self.history['stream1_train_acc'][-5:]))
            s2_acc_trend = np.mean(np.diff(self.history['stream2_train_acc'][-5:]))

            if s1_acc_trend < 0.001:
                recommendations.append(f"⚠️  Stream1 learning stalled - consider increasing stream1_lr")

            if s2_acc_trend < 0.001:
                recommendations.append(f"⚠️  Stream2 learning stalled - consider increasing stream2_lr")

        if not recommendations:
            recommendations.append("✓ All metrics look healthy")

        return recommendations


def create_stream_monitor(model: nn.Module) -> StreamMonitor:
    """
    Factory function to create a StreamMonitor.

    Args:
        model: The multi-stream model to monitor

    Returns:
        StreamMonitor instance
    """
    return StreamMonitor(model)
