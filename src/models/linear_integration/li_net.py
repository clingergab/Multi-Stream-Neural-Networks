"""
Linear Integration ResNet (LINet) implementations.

Extends MCResNet from 2 streams to 3 streams with integrated pathway.
"""

from sched import scheduler
from typing import Any, Callable, Optional, Union, TYPE_CHECKING
import time
import numpy as np

if TYPE_CHECKING:
    from tqdm import tqdm as TqdmType

from src.models.abstracts.abstract_model import BaseModel
import torch
import torch.nn as nn
from torch import Tensor
from torch.amp import autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts
from src.training.schedulers import setup_scheduler
from src.models.common import (
    save_checkpoint,
    create_progress_bar,
    finalize_progress_bar,
    update_history
)
# StreamMonitor no longer needed - monitoring is done during main training/validation loops

# Smart tqdm import - detect environment
try:
    from IPython import get_ipython
    if get_ipython() is not None and get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
        try:
            from tqdm.notebook import tqdm
            # Test if widgets work
            test_bar = tqdm(total=1, disable=True)
            test_bar.close()
        except ImportError:
            from tqdm import tqdm
    else:
        from tqdm import tqdm
except:
    from tqdm import tqdm

# Import the linear integration components
from src.models.linear_integration.conv import LIConv2d, LIBatchNorm2d
from src.models.linear_integration.blocks import LIBasicBlock, LIBottleneck
from src.models.linear_integration.container import LISequential, LIReLU
from src.models.linear_integration.pooling import LIMaxPool2d, LIAdaptiveAvgPool2d
# Note: LINet doesn't use fusion - it uses the integrated stream directly


class LINet(BaseModel):
    """
    Linear Integration ResNet (LINet) implementation.

    Extends MCResNet from 2 streams to 3 streams:
    - Stream1: RGB/Color pathway (independent processing)
    - Stream2: Depth/Brightness pathway (independent processing)
    - Integrated: Learned fusion pathway (combines stream outputs during convolution)

    Uses unified LIConv2d neurons where integration happens INSIDE the neuron.
    Final predictions come from the integrated stream (no separate fusion layer needed).
    """

    def __init__(
        self,
        block: type[Union[LIBasicBlock, LIBottleneck]],
        layers: list[int],
        num_classes: int,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[list[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        device: Optional[str] = None,
        use_amp: bool = False,
        # LINet-specific parameters come AFTER all ResNet parameters
        stream1_input_channels: int = 3,
        stream2_input_channels: int = 1,
        dropout_p: float = 0.0,  # Dropout probability (0.0 = no dropout, 0.5 = 50% dropout)
        **kwargs
    ) -> None:
        # Store LINet-specific parameters BEFORE calling super().__init__
        # because _build_network() (called by super()) needs these attributes
        self.stream1_input_channels = stream1_input_channels
        self.stream2_input_channels = stream2_input_channels
        self.dropout_p = dropout_p

        # Set LINet default norm layer if not specified
        if norm_layer is None:
            norm_layer = LIBatchNorm2d

        # Initialize BaseModel with all ResNet-compatible parameters in exact order
        # BaseModel will handle all the standard ResNet setup (device, inplanes, dilation, etc.)
        super().__init__(
            block=block,
            layers=layers,
            num_classes=num_classes,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            replace_stride_with_dilation=replace_stride_with_dilation,
            norm_layer=norm_layer,
            device=device,
            use_amp=use_amp
        )
    
    def _build_network(
        self,
        block: type[Union[LIBasicBlock, LIBottleneck]],
        layers: list[int],
        replace_stride_with_dilation: list[bool]
    ):
        """Build the Linear Integration ResNet network architecture."""
        # BaseModel already initialized self.stream1_inplanes and self.stream2_inplanes to 64
        # Now add integrated_inplanes tracking for 3rd stream
        self.integrated_inplanes = 64

        # Network architecture - exactly like ResNet but with 3-stream LI components
        # First conv: no integrated input yet (integrated starts after first block)
        self.conv1 = LIConv2d(
            self.stream1_input_channels, self.stream2_input_channels, 0,  # 0 for integrated_in (doesn't exist yet)
            self.stream1_inplanes, self.stream2_inplanes, self.integrated_inplanes,
            kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = self._norm_layer(self.stream1_inplanes, self.stream2_inplanes, self.integrated_inplanes)
        self.relu = LIReLU(inplace=True)
        self.maxpool = LIMaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers - equal scaling for all 3 streams
        self.layer1 = self._make_layer(block, 64, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        # Adaptive average pooling for integrated stream only
        # (Stream1/Stream2 pooling happens in LIMaxPool2d layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Add dropout for regularization (configurable, critical for small datasets)
        self.dropout = nn.Dropout(p=self.dropout_p) if self.dropout_p > 0.0 else nn.Identity()

        # Single classifier for integrated stream features
        # No fusion needed - integrated stream is the final representation!
        feature_dim = 512 * block.expansion
        self.fc = nn.Linear(feature_dim, self.num_classes)

        # Auxiliary classifiers for stream monitoring (gradient-isolated)
        # These learn to classify from stream features but DON'T affect stream training via .detach()
        # Only used when stream_monitoring=True
        self.fc_stream1 = nn.Linear(feature_dim, self.num_classes)
        self.fc_stream2 = nn.Linear(feature_dim, self.num_classes)
    
    def _initialize_weights(self, zero_init_residual: bool):
        """Initialize network weights for 3-stream architecture."""
        # Weight initialization for LIConv2d (already handled in LIConv2d.reset_parameters())
        # But we can double-check or add custom initialization here if needed
        for m in self.modules():
            if isinstance(m, LIConv2d):
                # LIConv2d.reset_parameters() already initializes all 5 weight matrices
                # using Kaiming initialization, so no need to do anything here
                pass
            elif isinstance(m, LIBatchNorm2d):
                # LIBatchNorm2d uses standard nn.BatchNorm2d internally,
                # which initializes itself, so no need to do anything here
                pass

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, LIBottleneck) and hasattr(m.bn3, 'integrated_bn'):
                    # Zero-init integrated stream's last BN weight
                    nn.init.constant_(m.bn3.integrated_bn.weight, 0)
                elif isinstance(m, LIBasicBlock) and hasattr(m.bn2, 'integrated_bn'):
                    # Zero-init integrated stream's last BN weight
                    nn.init.constant_(m.bn2.integrated_bn.weight, 0)

    def _make_layer(
        self,
        block: type[Union[LIBasicBlock, LIBottleneck]],
        stream1_planes: int,
        stream2_planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> LISequential:
        """
        Create a layer composed of multiple residual blocks with 3-stream support.
        Fully compliant with ResNet implementation but using LI blocks with 3-stream channel tracking.
        """
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        # Check if downsampling is needed for any stream
        need_downsample = (stride != 1 or
                          self.stream1_inplanes != stream1_planes * block.expansion or
                          self.stream2_inplanes != stream2_planes * block.expansion or
                          self.integrated_inplanes != stream1_planes * block.expansion)

        if need_downsample:
            # Downsample all 3 streams together using unified LI neurons
            downsample = LISequential(
                LIConv2d(
                    self.stream1_inplanes, self.stream2_inplanes, self.integrated_inplanes,
                    stream1_planes * block.expansion, stream2_planes * block.expansion, stream1_planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                norm_layer(stream1_planes * block.expansion, stream2_planes * block.expansion, stream1_planes * block.expansion)
            )

        layers = []
        # First block with potential downsampling
        layers.append(
            block(
                self.stream1_inplanes,
                self.stream2_inplanes,
                self.integrated_inplanes,
                stream1_planes,
                stream2_planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer
            )
        )

        # Update channel tracking for all 3 streams
        self.stream1_inplanes = stream1_planes * block.expansion
        self.stream2_inplanes = stream2_planes * block.expansion
        self.integrated_inplanes = stream1_planes * block.expansion

        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.stream1_inplanes,
                    self.stream2_inplanes,
                    self.integrated_inplanes,
                    stream1_planes,
                    stream2_planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return LISequential(*layers)
    
    def forward(self, stream1_input: Tensor, stream2_input: Tensor) -> Tensor:
        """
        Forward pass through the Linear Integration ResNet.

        Args:
            stream1_input: RGB input tensor [batch_size, channels, height, width]
            stream2_input: Depth input tensor [batch_size, channels, height, width]

        Returns:
            Classification logits [batch_size, num_classes] from integrated stream
        """
        # Initial convolution (creates integrated stream from stream1 + stream2)
        s1, s2, ic = self.conv1(stream1_input, stream2_input, None)  # integrated_input=None for first layer
        s1, s2, ic = self.bn1(s1, s2, ic)
        s1, s2, ic = self.relu(s1, s2, ic)

        # Max pooling (all 3 streams)
        s1, s2, ic = self.maxpool(s1, s2, ic)

        # ResNet layers (all 3 streams, integration happens inside LIConv2d neurons!)
        s1, s2, ic = self.layer1(s1, s2, ic)
        s1, s2, ic = self.layer2(s1, s2, ic)
        s1, s2, ic = self.layer3(s1, s2, ic)
        s1, s2, ic = self.layer4(s1, s2, ic)

        # Global average pooling for integrated stream only
        # (We only use integrated stream for final prediction)
        integrated_pooled = self.avgpool(ic)

        # Flatten integrated features
        integrated_features = torch.flatten(integrated_pooled, 1)

        # Apply dropout (only active during training if dropout_p > 0)
        integrated_features = self.dropout(integrated_features)

        # Classify using integrated stream features
        # No fusion needed - integrated stream IS the fused representation!
        logits = self.fc(integrated_features)
        return logits

    # ==================== Early Stopping Helper Methods ====================

    def _setup_early_stopping(self, enabled: bool, val_loader, monitor: str,
                             patience: int, min_delta: float, verbose: bool) -> dict:
        """
        Initialize early stopping state.

        Returns a dictionary with:
        - enabled: bool
        - patience: int
        - best_metric: float (initialized based on monitor)
        - patience_counter: int
        - best_epoch: int
        - best_weights: dict or None
        """
        if enabled and val_loader is None:
            if verbose:
                print("⚠️  Early stopping requested but no validation data provided. Disabling.")
            enabled = False

        if not enabled:
            return {'enabled': False}

        # Validate monitor
        if monitor not in ('val_loss', 'val_accuracy'):
            raise ValueError(f"Invalid monitor: {monitor}. Use 'val_loss' or 'val_accuracy'.")

        # Initialize best metric based on monitor type
        best_metric = float('inf') if monitor == 'val_loss' else 0.0

        if verbose:
            print(f"🛑 Early stopping enabled: monitoring {monitor} with patience={patience}, min_delta={min_delta}")

        return {
            'enabled': True,
            'patience': patience,
            'best_metric': best_metric,
            'patience_counter': 0,
            'best_epoch': 0,
            'best_weights': None
        }

    def _setup_stream_early_stopping(self, enabled: bool, monitor: str,
                                     stream1_patience: int, stream2_patience: int,
                                     min_delta: float, verbose: bool) -> dict:
        """
        Initialize stream-specific early stopping state.

        Returns a dictionary with per-stream state and a full model checkpoint
        for when all streams freeze.
        """
        if not enabled:
            return {'enabled': False}

        # Validate monitor
        if monitor not in ('val_loss', 'val_accuracy'):
            raise ValueError(f"Invalid monitor: {monitor}. Use 'val_loss' or 'val_accuracy'.")

        # Initialize best metric based on monitor type
        best_metric = float('inf') if monitor == 'val_loss' else 0.0

        if verbose:
            print(f"❄️  Stream-specific early stopping enabled:")
            print(f"   Monitor: {monitor}")
            print(f"   Stream1 patience: {stream1_patience}, Stream2 patience: {stream2_patience}")
            print(f"   Min delta: {min_delta}")

        return {
            'enabled': True,
            'stream1': {
                'best_metric': best_metric,
                'patience': stream1_patience,
                'patience_counter': 0,
                'best_epoch': 0,
                'frozen': False,
                'best_weights': None
            },
            'stream2': {
                'best_metric': best_metric,
                'patience': stream2_patience,
                'patience_counter': 0,
                'best_epoch': 0,
                'frozen': False,
                'best_weights': None
            },
            'all_frozen': False,
            'best_full_model': {
                'best_metric': best_metric,
                'epoch': 0,
                'weights': None
            }
        }

    def _should_update_checkpoint(self, current_metric: float, best_metric: float,
                                   monitor: str, min_delta: float) -> bool:
        """Check if current metric is better than best metric."""
        if monitor == 'val_loss':
            return current_metric < (best_metric - min_delta)
        else:  # val_accuracy
            return current_metric > (best_metric + min_delta)

    def _save_checkpoint(self) -> dict:
        """Save current model state as checkpoint."""
        return {k: v.cpu().clone() for k, v in self.state_dict().items()}

    def _update_frozen_stream_weights_in_checkpoint(self, checkpoint: dict,
                                                     stream_early_stopping_state: dict) -> None:
        """
        Update checkpoint with frozen stream weights from their best epochs.

        This ensures that frozen streams stay at their best performance while
        the rest of the model is at the overall best epoch.
        """
        if stream_early_stopping_state['stream1']['frozen']:
            best_weights = stream_early_stopping_state['stream1']['best_weights']
            if best_weights is not None:
                for name, weight in best_weights.items():
                    checkpoint[name] = weight.clone()

        if stream_early_stopping_state['stream2']['frozen']:
            best_weights = stream_early_stopping_state['stream2']['best_weights']
            if best_weights is not None:
                for name, weight in best_weights.items():
                    checkpoint[name] = weight.clone()

    def _restore_checkpoint(self, checkpoint: dict, verbose: bool = True,
                           stream_early_stopping_state: Optional[dict] = None) -> None:
        """Restore model from checkpoint."""
        self.load_state_dict({k: v.to(self.device) for k, v in checkpoint.items()})

        if verbose:
            if stream_early_stopping_state and stream_early_stopping_state['enabled']:
                frozen_streams = []
                if stream_early_stopping_state['stream1']['frozen']:
                    frozen_streams.append("Stream1")
                if stream_early_stopping_state['stream2']['frozen']:
                    frozen_streams.append("Stream2")

                if frozen_streams:
                    print(f"🔄 Restored best model weights (preserved frozen {', '.join(frozen_streams)})")
                else:
                    print("🔄 Restored best model weights")
            else:
                print("🔄 Restored best model weights")

    def _should_stop_early(self, patience_counter: int, patience: int) -> bool:
        """Check if patience is exhausted."""
        return patience_counter > patience

    # ==================== Stream Early Stopping Helpers ====================

    def _update_best_full_model(self, stream_early_stopping_state: dict,
                                val_loss: float, val_acc: float,
                                monitor: str, min_delta: float, epoch: int) -> None:
        """
        Update best full model checkpoint in stream early stopping state.

        This tracks the best overall model performance for restoration when all streams freeze.
        """
        best_full = stream_early_stopping_state['best_full_model']

        # Get current metric based on monitor type
        current_metric = val_loss if monitor == 'val_loss' else val_acc

        # Check if we should update
        if self._should_update_checkpoint(current_metric, best_full['best_metric'], monitor, min_delta):
            best_full['best_metric'] = current_metric
            best_full['epoch'] = epoch
            # Save full model state (captures current model including any frozen streams)
            best_full['weights'] = self._save_checkpoint()

    def _check_stream_improvement(self, stream_state: dict, stream_metric: float,
                                  monitor: str, min_delta: float, epoch: int,
                                  stream_name: str) -> bool:
        """
        Check if a stream has improved and update its state accordingly.

        Returns:
            True if improvement detected, False otherwise
        """
        if self._should_update_checkpoint(stream_metric, stream_state['best_metric'], monitor, min_delta):
            # Improvement detected
            stream_state['best_metric'] = stream_metric
            stream_state['best_epoch'] = epoch
            stream_state['patience_counter'] = 0

            # Save best stream weights using state_dict
            model_state = self.state_dict()
            stream_state['best_weights'] = {
                name: weight.cpu().clone()
                for name, weight in model_state.items()
                if f'.{stream_name}_' in name
            }
            return True
        else:
            # No improvement
            stream_state['patience_counter'] += 1
            return False

    def _freeze_stream(self, stream_state: dict, stream_name: str,
                      epoch: int, monitor: str, verbose: bool) -> None:
        """
        Freeze a stream by restoring its best weights and setting requires_grad=False.
        """
        # Restore best weights before freezing
        if stream_state['best_weights'] is not None:
            for name, param in self.named_parameters():
                if f'.{stream_name}_' in name and name in stream_state['best_weights']:
                    param.data.copy_(stream_state['best_weights'][name].to(param.device))

        # Freeze stream
        stream_state['frozen'] = True
        stream_state['freeze_epoch'] = epoch

        # Set requires_grad = False for stream parameters
        # Integration weights remain trainable to allow rebalancing
        for name, param in self.named_parameters():
            if f'.{stream_name}_' in name:
                param.requires_grad = False

        if verbose:
            metric_str = f"{monitor}: {stream_state['best_metric']:.4f}"
            stream_label = "Stream1 (RGB)" if stream_name == "stream1" else "Stream2 (Depth)"
            print(f"❄️  {stream_label} frozen (no improvement for {stream_state['patience']} epochs, "
                  f"best {metric_str} at epoch {stream_state['best_epoch'] + 1})")

    def _restore_best_full_model(self, stream_early_stopping_state: dict,
                                 monitor: str, verbose: bool) -> None:
        """
        Restore best full model when all streams are frozen.

        Since we update best_full_model['weights'] whenever streams freeze (in li_net.py),
        this checkpoint already has the correct hybrid state with frozen streams at their
        best epochs. We just restore it directly without any additional preservation logic.
        """
        best_full = stream_early_stopping_state['best_full_model']

        if best_full['weights'] is not None:
            # Restore checkpoint (already has frozen streams at their best epochs)
            self.load_state_dict({k: v.to(self.device) for k, v in best_full['weights'].items()})

            if verbose:
                metric_str = f"{monitor}: {best_full['best_metric']:.4f}"

                # Build message showing which streams are frozen
                frozen_streams = []
                if stream_early_stopping_state['stream1']['frozen']:
                    frozen_streams.append(f"Stream1 at epoch {stream_early_stopping_state['stream1']['best_epoch'] + 1}")
                if stream_early_stopping_state['stream2']['frozen']:
                    frozen_streams.append(f"Stream2 at epoch {stream_early_stopping_state['stream2']['best_epoch'] + 1}")

                if frozen_streams:
                    frozen_info = ", ".join(frozen_streams)
                    print(f"🔄 Restored full model from epoch {best_full['epoch'] + 1} "
                          f"({metric_str}, preserved {frozen_info})")
                else:
                    print(f"🔄 Restored full model from epoch {best_full['epoch'] + 1} ({metric_str})")

    def _check_stream_early_stopping(self, stream_early_stopping_state: dict,
                                     stream_stats: dict, epoch: int, monitor: str,
                                     min_delta: float, verbose: bool,
                                     val_acc: float, val_loss: float) -> bool:
        """
        Check stream-specific early stopping and freeze streams when they plateau.

        This is the main orchestrator that:
        1. Updates best full model checkpoint
        2. Checks each stream for improvement
        3. Freezes streams when patience is exhausted
        4. Restores best model when all streams are frozen

        Returns:
            True if all streams are frozen, False otherwise
        """
        if not stream_early_stopping_state['enabled']:
            return False

        # Update best full model checkpoint
        self._update_best_full_model(stream_early_stopping_state, val_loss, val_acc,
                                     monitor, min_delta, epoch)

        # Check Stream1
        if not stream_early_stopping_state['stream1']['frozen']:
            # Get current metric
            if monitor == 'val_loss':
                stream1_metric = stream_stats.get('stream1_val_loss', float('inf'))
            else:
                stream1_metric = stream_stats.get('stream1_val_acc', 0.0)

            stream1_state = stream_early_stopping_state['stream1']

            # Check for improvement
            improved = self._check_stream_improvement(stream1_state, stream1_metric,
                                                      monitor, min_delta, epoch, 'stream1')

            # Freeze if patience exhausted
            if not improved and stream1_state['patience_counter'] > stream1_state['patience']:
                self._freeze_stream(stream1_state, 'stream1', epoch, monitor, verbose)

        # Check Stream2
        if not stream_early_stopping_state['stream2']['frozen']:
            # Get current metric
            if monitor == 'val_loss':
                stream2_metric = stream_stats.get('stream2_val_loss', float('inf'))
            else:
                stream2_metric = stream_stats.get('stream2_val_acc', 0.0)

            stream2_state = stream_early_stopping_state['stream2']

            # Check for improvement
            improved = self._check_stream_improvement(stream2_state, stream2_metric,
                                                      monitor, min_delta, epoch, 'stream2')

            # Freeze if patience exhausted
            if not improved and stream2_state['patience_counter'] > stream2_state['patience']:
                self._freeze_stream(stream2_state, 'stream2', epoch, monitor, verbose)

        # Check if all streams are frozen
        all_frozen = (stream_early_stopping_state['stream1']['frozen'] and
                     stream_early_stopping_state['stream2']['frozen'])

        # If all streams just became frozen, restore best full model
        if all_frozen and not stream_early_stopping_state['all_frozen']:
            self._restore_best_full_model(stream_early_stopping_state, monitor, verbose)

        stream_early_stopping_state['all_frozen'] = all_frozen

        return all_frozen

    # ==================== End Early Stopping Helpers ====================

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        callbacks: Optional[list] = None,
        verbose: bool = True,
        save_path: Optional[str] = None,
        early_stopping: bool = False,
        patience: int = 10,
        min_delta: float = 0.001,
        monitor: str = 'val_loss',  # 'val_loss' or 'val_accuracy' (for both main and stream early stopping)
        restore_best_weights: bool = True,
        gradient_accumulation_steps: int = 1,  # Gradient accumulation for larger effective batch size
        grad_clip_norm: Optional[float] = None,  # Gradient clipping max norm (None = disabled)
        clear_cache_per_epoch: bool = False,  # Clear CUDA cache after each epoch (only if experiencing OOM)
        stream_monitoring: bool = False,  # Enable stream-specific monitoring (LR, WD, acc per stream)
        stream_early_stopping: bool = False,  # Enable stream-specific early stopping (freezes streams when they plateau)
        stream1_patience: int = 10,  # Patience for Stream1 (RGB) before freezing
        stream2_patience: int = 10,  # Patience for Stream2 (Depth) before freezing
        stream_min_delta: float = 0.001  # Minimum improvement for stream early stopping
    ) -> dict:
        """
        Train the model with optional early stopping.

        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data (optional)
            epochs: Number of epochs to train
            callbacks: List of callbacks to apply during training
            verbose: Whether to print progress
            save_path: Path to save best model checkpoint
            early_stopping: Whether to enable early stopping
            patience: Number of epochs to wait for improvement before stopping
            min_delta: Minimum change to qualify as an improvement
            monitor: Metric to monitor for both main and stream early stopping ('val_loss' or 'val_accuracy').
                   Applied globally to main model early stopping and stream-specific early stopping.
            restore_best_weights: Whether to restore best weights when early stopping
            gradient_accumulation_steps: Number of steps to accumulate gradients before updating.
                                       Useful for simulating larger batch sizes (e.g., 4 steps = 4x effective batch size)
            grad_clip_norm: Maximum gradient norm for clipping (None to disable). Standard values: 1.0 or 5.0
            clear_cache_per_epoch: Whether to clear CUDA cache after each epoch (only enable if OOM issues)
            stream_monitoring: Enable stream-specific monitoring. Shows per-stream LR, WD, train/val acc.
                            Auxiliary classifiers (fc_stream1, fc_stream2) learn to classify stream features
                            but are gradient-isolated (use .detach()) so they DON'T affect stream weight training.
                            This provides accurate stream accuracy metrics without changing main training dynamics.
            stream_early_stopping: Enable stream-specific early stopping. When a stream plateaus, its
                                 parameters (.stream1_weight/.stream2_weight) are frozen while integration
                                 weights remain trainable, allowing the model to continue learning from the
                                 other stream. Training stops when both streams are frozen. Uses same monitor
                                 metric as main early stopping.
            stream1_patience: Number of epochs to wait before freezing Stream1 (RGB) if no improvement
            stream2_patience: Number of epochs to wait before freezing Stream2 (Depth) if no improvement
            stream_min_delta: Minimum improvement for stream metrics to count as progress

        Returns:
            Training history dictionary
        """
        if self.optimizer is None or self.criterion is None:
            raise ValueError("Model not compiled. Call compile() before fit().")

        callbacks = callbacks or []

        # Early stopping setup
        early_stopping_state = self._setup_early_stopping(
            early_stopping, val_loader, monitor, patience, min_delta, verbose
        )

        # Stream-specific early stopping setup (uses same monitor as main early stopping)
        stream_early_stopping_state = self._setup_stream_early_stopping(
            stream_early_stopping, monitor, stream1_patience, stream2_patience, stream_min_delta, verbose
        )

        # Warn if stream_early_stopping enabled without stream_monitoring
        if stream_early_stopping and not stream_monitoring and verbose:
            print("⚠️  Warning: stream_early_stopping=True requires stream_monitoring=True to work")
            print("   Stream early stopping will be disabled until stream_monitoring is enabled")

        # Create separate optimizer for auxiliary classifiers (if stream monitoring enabled)
        # This ensures auxiliary training doesn't affect main model's optimizer state
        aux_optimizer = None
        if stream_monitoring:
            aux_params = [
                self.fc_stream1.weight, self.fc_stream1.bias,
                self.fc_stream2.weight, self.fc_stream2.bias
            ]
            # Use same optimizer type and ALL hyperparameters as main optimizer
            # This ensures auxiliary classifiers train identically to main model
            main_group = self.optimizer.param_groups[0]

            if isinstance(self.optimizer, torch.optim.Adam):
                # Copy all Adam hyperparameters from main optimizer
                aux_optimizer = torch.optim.Adam(
                    aux_params,
                    lr=main_group['lr'],
                    betas=main_group.get('betas', (0.9, 0.999)),
                    eps=main_group.get('eps', 1e-8),
                    weight_decay=main_group.get('weight_decay', 0),
                    amsgrad=main_group.get('amsgrad', False)
                )
            elif isinstance(self.optimizer, torch.optim.AdamW):
                # Copy all AdamW hyperparameters from main optimizer
                aux_optimizer = torch.optim.AdamW(
                    aux_params,
                    lr=main_group['lr'],
                    betas=main_group.get('betas', (0.9, 0.999)),
                    eps=main_group.get('eps', 1e-8),
                    weight_decay=main_group.get('weight_decay', 0),
                    amsgrad=main_group.get('amsgrad', False)
                )
            elif isinstance(self.optimizer, torch.optim.SGD):
                # Copy all SGD hyperparameters from main optimizer
                aux_optimizer = torch.optim.SGD(
                    aux_params,
                    lr=main_group['lr'],
                    momentum=main_group.get('momentum', 0),
                    dampening=main_group.get('dampening', 0),
                    weight_decay=main_group.get('weight_decay', 0),
                    nesterov=main_group.get('nesterov', False)
                )
            elif isinstance(self.optimizer, torch.optim.RMSprop):
                # Copy all RMSprop hyperparameters from main optimizer
                aux_optimizer = torch.optim.RMSprop(
                    aux_params,
                    lr=main_group['lr'],
                    alpha=main_group.get('alpha', 0.99),
                    eps=main_group.get('eps', 1e-8),
                    weight_decay=main_group.get('weight_decay', 0),
                    momentum=main_group.get('momentum', 0)
                )
            else:
                # Fallback: use Adam with main optimizer's learning rate and weight_decay
                aux_optimizer = torch.optim.Adam(
                    aux_params,
                    lr=main_group['lr'],
                    weight_decay=main_group.get('weight_decay', 0)
                )

        # Legacy best model saving (preserve existing functionality)
        best_val_acc = 0.0

        # Initialize training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'learning_rates': [],
            # Stream-specific metrics (populated if stream_monitoring=True)
            'stream1_train_acc': [],
            'stream1_val_acc': [],
            'stream2_train_acc': [],
            'stream2_val_acc': [],
            'stream1_lr': [],
            'stream2_lr': [],
            # Stream freezing events (populated if stream_early_stopping=True)
            'stream1_frozen_epoch': None,
            'stream2_frozen_epoch': None,
            'streams_frozen': []  # List of (epoch, stream_name) tuples
        }

        # Scheduler is already set by compile(), no need to create it here

        for epoch in range(epochs):
            # Calculate total steps for this epoch (training + validation)
            total_steps = len(train_loader)
            if val_loader:
                total_steps += len(val_loader)
            
            # Create progress bar for the entire epoch
            pbar = create_progress_bar(verbose, epoch, epochs, total_steps)
            
            # Training phase - use helper method
            avg_train_loss, train_accuracy, stream1_train_acc, stream2_train_acc = self._train_epoch(
                train_loader, history, pbar, gradient_accumulation_steps, grad_clip_norm, clear_cache_per_epoch,
                stream_monitoring=stream_monitoring, aux_optimizer=aux_optimizer
            )

            # Validation phase
            val_loss = 0.0
            val_acc = 0.0
            stream1_val_acc = 0.0
            stream2_val_acc = 0.0
            stream1_val_loss = 0.0
            stream2_val_loss = 0.0

            if val_loader:
                val_loss, val_acc, stream1_val_acc, stream2_val_acc, stream1_val_loss, stream2_val_loss = self._validate(
                    val_loader, pbar=pbar, stream_monitoring=stream_monitoring
                )
                
                # Legacy save best model (preserve existing functionality)
                if save_path and val_acc > best_val_acc:
                    best_val_acc = val_acc
                    save_checkpoint(
                        model_state_dict=self.state_dict(),
                        optimizer_state_dict=self.optimizer.state_dict() if self.optimizer else None,
                        scheduler_state_dict=self.scheduler.state_dict() if self.scheduler else None,
                        path=save_path,
                        history=history
                    )
                
                # Check for early stopping
                if early_stopping_state['enabled']:

                    current_metric = val_loss if monitor == 'val_loss' else val_acc

                    # Check if we should update checkpoint
                    if self._should_update_checkpoint(current_metric, early_stopping_state['best_metric'],
                                                        monitor, min_delta):
                        # New best! Save checkpoint
                        early_stopping_state['best_metric'] = current_metric
                        early_stopping_state['best_epoch'] = epoch
                        early_stopping_state['patience_counter'] = 0

                        if restore_best_weights:
                            # Save checkpoint
                            early_stopping_state['best_weights'] = self._save_checkpoint()

                            # Update frozen stream weights in checkpoint
                            if stream_early_stopping_state['enabled']:
                                self._update_frozen_stream_weights_in_checkpoint(
                                    early_stopping_state['best_weights'],
                                    stream_early_stopping_state
                                )

                        if verbose and pbar is None:
                            print(f"✅ New best {monitor}: {current_metric:.4f}")
                    else:
                        # No improvement
                        early_stopping_state['patience_counter'] += 1

                        # Check if we should stop
                        if self._should_stop_early(early_stopping_state['patience_counter'],
                                                   early_stopping_state['patience']):
                            if verbose:
                                print(f"🛑 Early stopping triggered after {epoch + 1} epochs")
                                print(f"   Best {monitor}: {early_stopping_state['best_metric']:.4f} at epoch {early_stopping_state['best_epoch'] + 1}")

                            # Restore best weights
                            if restore_best_weights and early_stopping_state['best_weights'] is not None:
                                self._restore_checkpoint(early_stopping_state['best_weights'], verbose,
                                                        stream_early_stopping_state)
                            break
            
            # Step epoch-based schedulers at epoch end
            # Skip OneCycleLR (steps per batch) and schedulers with _step_per_batch=True
            if self.scheduler is not None:
                is_batch_scheduler = (isinstance(self.scheduler, OneCycleLR) or
                                     getattr(self.scheduler, '_step_per_batch', False))
                if not is_batch_scheduler:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        # ReduceLROnPlateau needs a metric to step
                        if self.scheduler.mode == 'max':
                            # Mode 'max': monitor accuracy (higher is better)
                            metric = val_acc if val_loader else train_accuracy
                        else:
                            # Mode 'min': monitor loss (lower is better)
                            metric = val_loss if val_loader else avg_train_loss
                        self.scheduler.step(metric)
                    else:
                        # All other epoch-based schedulers (CosineAnnealingLR, CosineAnnealingWarmRestarts, StepLR, etc.)
                        # step per epoch without arguments
                        self.scheduler.step()
            current_lr = self.optimizer.param_groups[-1]['lr']  # Base LR is last group (shared params)
            
            # Update history and finalize progress bar
            update_history(history, avg_train_loss, train_accuracy, val_loss, val_acc, current_lr, bool(val_loader))
            finalize_progress_bar(
                pbar, avg_train_loss, train_accuracy, val_loader,
                val_loss, val_acc, early_stopping_state, current_lr
            )

            # Stream-specific monitoring (print immediately after progress bar, on same line continuation)
            stream_stats = {}
            if stream_monitoring:
                # Always save stream accuracies to history (computed during training/validation)
                history['stream1_train_acc'].append(stream1_train_acc)
                history['stream1_val_acc'].append(stream1_val_acc)
                history['stream2_train_acc'].append(stream2_train_acc)
                history['stream2_val_acc'].append(stream2_val_acc)

                # Print detailed metrics only if using stream-specific parameter groups
                if len(self.optimizer.param_groups) >= 3:
                    stream_stats = self._print_stream_monitoring(
                        stream1_train_acc=stream1_train_acc,
                        stream1_val_acc=stream1_val_acc,
                        stream2_train_acc=stream2_train_acc,
                        stream2_val_acc=stream2_val_acc,
                        stream1_val_loss=stream1_val_loss,
                        stream2_val_loss=stream2_val_loss
                    )
                    # Save learning rates (only available with stream-specific param groups)
                    if stream_stats:
                        history['stream1_lr'].append(stream_stats['stream1_lr'])
                        history['stream2_lr'].append(stream_stats['stream2_lr'])
                else:
                    # Create minimal stream_stats for compatibility
                    stream_stats = {
                        'stream1_train_acc': stream1_train_acc,
                        'stream1_val_acc': stream1_val_acc,
                        'stream1_val_loss': stream1_val_loss,
                        'stream2_train_acc': stream2_train_acc,
                        'stream2_val_acc': stream2_val_acc,
                        'stream2_val_loss': stream2_val_loss
                    }

            # Stream-specific early stopping (freeze streams when they plateau)
            if stream_early_stopping_state['enabled'] and stream_stats:
                # Track previous frozen states
                prev_stream1_frozen = stream_early_stopping_state['stream1']['frozen']
                prev_stream2_frozen = stream_early_stopping_state['stream2']['frozen']

                # Check for stream early stopping using our new method
                all_streams_frozen = self._check_stream_early_stopping(
                    stream_early_stopping_state=stream_early_stopping_state,
                    stream_stats=stream_stats,
                    epoch=epoch,
                    monitor=monitor,
                    min_delta=stream_min_delta,
                    verbose=verbose,
                    val_acc=val_acc,
                    val_loss=val_loss
                )

                # Record freezing events and update checkpoints
                if not prev_stream1_frozen and stream_early_stopping_state['stream1']['frozen']:
                    history['stream1_frozen_epoch'] = epoch + 1
                    history['streams_frozen'].append((epoch + 1, 'Stream1'))

                    # Update main early stopping checkpoint with frozen Stream1 weights
                    if early_stopping_state['enabled'] and restore_best_weights:
                        # Create checkpoint if it doesn't exist yet
                        if early_stopping_state.get('best_weights') is None:
                            early_stopping_state['best_weights'] = self._save_checkpoint()

                        # Update with frozen stream weights
                        self._update_frozen_stream_weights_in_checkpoint(
                            early_stopping_state['best_weights'],
                            stream_early_stopping_state
                        )

                    # Also update stream ES checkpoint (used when all streams freeze)
                    if stream_early_stopping_state['best_full_model']['weights'] is not None:
                        self._update_frozen_stream_weights_in_checkpoint(
                            stream_early_stopping_state['best_full_model']['weights'],
                            stream_early_stopping_state
                        )

                if not prev_stream2_frozen and stream_early_stopping_state['stream2']['frozen']:
                    history['stream2_frozen_epoch'] = epoch + 1
                    history['streams_frozen'].append((epoch + 1, 'Stream2'))

                    # Update main early stopping checkpoint with frozen Stream2 weights
                    if early_stopping_state['enabled'] and restore_best_weights:
                        # Create checkpoint if it doesn't exist yet
                        if early_stopping_state.get('best_weights') is None:
                            early_stopping_state['best_weights'] = self._save_checkpoint()

                        # Update with frozen stream weights
                        self._update_frozen_stream_weights_in_checkpoint(
                            early_stopping_state['best_weights'],
                            stream_early_stopping_state
                        )

                    # Also update stream ES checkpoint (used when all streams freeze)
                    if stream_early_stopping_state['best_full_model']['weights'] is not None:
                        self._update_frozen_stream_weights_in_checkpoint(
                            stream_early_stopping_state['best_full_model']['weights'],
                            stream_early_stopping_state
                        )

                # Stop training if all streams are frozen
                if all_streams_frozen:
                    if verbose:
                        print("🛑 All streams frozen - stopping training")
                    break

            # Call callbacks
            for callback in callbacks:
                callback.on_epoch_end(epoch, {
                    'train_loss': avg_train_loss,
                    'train_accuracy': train_accuracy,
                    'val_loss': val_loss,
                    'val_accuracy': val_acc
                })
        
        # Final early stopping summary
        if early_stopping_state['enabled'] and val_loader is not None:
            stopped_early = early_stopping_state['patience_counter'] > early_stopping_state['patience']
            history['early_stopping'] = {
                'stopped_early': stopped_early,
                'best_epoch': early_stopping_state['best_epoch'] + 1,
                'best_metric': early_stopping_state['best_metric'],
                'monitor': monitor,  # Use parameter instead of state
                'patience': early_stopping_state['patience'],
                'min_delta': min_delta  # Use parameter instead of state
            }

        # Stream early stopping summary
        if stream_early_stopping_state['enabled']:
            # Use monitor parameter (monitor is shared between main and stream early stopping)
            history['stream_early_stopping'] = {
                'monitor': monitor,
                'stream1_frozen': stream_early_stopping_state['stream1']['frozen'],
                'stream1_frozen_epoch': history['stream1_frozen_epoch'],
                'stream1_best_metric': stream_early_stopping_state['stream1']['best_metric'],
                'stream2_frozen': stream_early_stopping_state['stream2']['frozen'],
                'stream2_frozen_epoch': history['stream2_frozen_epoch'],
                'stream2_best_metric': stream_early_stopping_state['stream2']['best_metric'],
                'all_frozen': stream_early_stopping_state['all_frozen']
            }

            if verbose:
                print("\n❄️  Stream Early Stopping Summary:")
                print(f"   Monitor: {monitor}")
                if stream_early_stopping_state['stream1']['frozen']:
                    print(f"   Stream1: Frozen at epoch {history['stream1_frozen_epoch']} "
                          f"(best {monitor}: {stream_early_stopping_state['stream1']['best_metric']:.4f})")
                else:
                    print(f"   Stream1: Not frozen (final {monitor}: {stream_early_stopping_state['stream1']['best_metric']:.4f})")

                if stream_early_stopping_state['stream2']['frozen']:
                    print(f"   Stream2: Frozen at epoch {history['stream2_frozen_epoch']} "
                          f"(best {monitor}: {stream_early_stopping_state['stream2']['best_metric']:.4f})")
                else:
                    print(f"   Stream2: Not frozen (final {monitor}: {stream_early_stopping_state['stream2']['best_metric']:.4f})")

        return history

    def evaluate(self, data_loader: DataLoader, stream_monitoring: bool = True) -> dict:
        """
        Evaluate the model on the given data.

        Args:
            data_loader: DataLoader containing dual-channel input data and targets
            stream_monitoring: Whether to calculate stream-specific metrics (default: True)

        Returns:
            Dictionary containing evaluation metrics
        """
        if self.criterion is None:
            raise ValueError("Model not compiled. Call compile() before evaluate().")

        loss, accuracy, stream1_val_acc, stream2_val_acc, stream1_val_loss, stream2_val_loss = self._validate(
            data_loader, stream_monitoring=stream_monitoring
        )

        return {
            'loss': loss,
            'accuracy': accuracy,
            'stream1_accuracy': stream1_val_acc,
            'stream2_accuracy': stream2_val_acc,
            'stream1_loss': stream1_val_loss,
            'stream2_loss': stream2_val_loss
        }
    def predict(self, data_loader: DataLoader) -> torch.Tensor:
        """
        Generate predictions for the input data.

        Args:
            data_loader: DataLoader containing dual-channel input data

        Returns:
            Tensor of predicted classes
        """
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.to(self.device)

        self.eval()
        all_predictions = []

        with torch.no_grad():
            for stream1_batch, stream2_batch, targets in data_loader:
                # GPU optimization: non-blocking transfer for dual-channel data
                stream1_batch = stream1_batch.to(self.device, non_blocking=True)
                stream2_batch = stream2_batch.to(self.device, non_blocking=True)

                outputs = self(stream1_batch, stream2_batch)
                _, predictions = torch.max(outputs, 1)
                all_predictions.append(predictions.cpu())

        return torch.cat(all_predictions, dim=0)
    
    def predict_proba(self, data_loader: DataLoader) -> np.ndarray:
        """
        Generate probability predictions for the input data.

        Args:
            data_loader: DataLoader containing dual-channel input data

        Returns:
            Numpy array of predicted probabilities with shape (n_samples, n_classes)
        """
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.to(self.device)

        self.eval()
        all_probabilities = []

        with torch.no_grad():
            for stream1_batch, stream2_batch, targets in data_loader:
                # GPU optimization: non-blocking transfer for dual-channel data
                stream1_batch = stream1_batch.to(self.device, non_blocking=True)
                stream2_batch = stream2_batch.to(self.device, non_blocking=True)

                outputs = self(stream1_batch, stream2_batch)
                # Apply softmax to get probabilities
                probabilities = torch.softmax(outputs, dim=1)
                all_probabilities.append(probabilities.cpu().numpy())

        return np.concatenate(all_probabilities, axis=0)
    
    
    def _train_epoch(self, train_loader: DataLoader, history: dict, pbar: Optional['TqdmType'] = None,
                     gradient_accumulation_steps: int = 1, grad_clip_norm: Optional[float] = None,
                     clear_cache_per_epoch: bool = False, stream_monitoring: bool = False,
                     aux_optimizer: Optional[torch.optim.Optimizer] = None) -> tuple:
        """
        Train the model for one epoch with GPU optimizations and gradient accumulation.

        Args:
            train_loader: DataLoader for training data
            history: Training history dictionary
            pbar: Optional progress bar to update
            gradient_accumulation_steps: Number of steps to accumulate gradients before updating
            grad_clip_norm: Maximum gradient norm for clipping (None to disable)
            clear_cache_per_epoch: Whether to clear CUDA cache after epoch
            stream_monitoring: Whether to track stream-specific metrics during training.
                            Auxiliary classifiers are trained with full gradients (gradient-isolated)
                            to provide accurate monitoring without affecting main model.

        Returns:
            Tuple of (average_train_loss, train_accuracy, stream1_train_acc, stream2_train_acc)
            If stream_monitoring=False, stream accuracies will be 0.0
        """
        self.train()
        train_loss = 0.0
        train_batches = 0
        train_correct = 0
        train_total = 0

        # Stream-specific tracking (if monitoring enabled)
        stream1_train_correct = 0
        stream2_train_correct = 0
        stream_train_total = 0
        
        # Ensure gradient_accumulation_steps is at least 1 to avoid division by zero
        gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        
        # OPTIMIZATION 1: Progress bar update frequency - major performance improvement
        update_frequency = max(1, len(train_loader) // 50)  # Update only 50 times per epoch
        
        for batch_idx, (stream1_batch, stream2_batch, targets) in enumerate(train_loader):
            # GPU optimization: non-blocking transfer for dual-channel data
            stream1_batch = stream1_batch.to(self.device, non_blocking=True)
            stream2_batch = stream2_batch.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            # OPTIMIZATION 5: Gradient accumulation - zero gradients only when starting accumulation
            if batch_idx % gradient_accumulation_steps == 0:
                self.optimizer.zero_grad()
            
            if self.use_amp:
                # Use automatic mixed precision
                with autocast(device_type=self.device.type):
                    outputs = self(stream1_batch, stream2_batch)
                    loss = self.criterion(outputs, targets)
                    # Scale loss for gradient accumulation
                    if gradient_accumulation_steps > 1:
                        loss = loss / gradient_accumulation_steps
                
                # Scale loss and backward pass
                self.scaler.scale(loss).backward()
                
                # OPTIMIZATION 5: Only step optimizer every accumulation_steps
                if (batch_idx + 1) % gradient_accumulation_steps == 0 or batch_idx == len(train_loader) - 1:
                    # Gradient clipping (if enabled)
                    if grad_clip_norm is not None:
                        self.scaler.unscale_(self.optimizer)
                        clip_grad_norm_(self.parameters(), grad_clip_norm)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    # Step scheduler after optimizer update
                    # OneCycleLR always steps per batch
                    # CosineAnnealingWarmRestarts can optionally step per batch if step_per_batch=True
                    if self.scheduler is not None:
                        should_step = (isinstance(self.scheduler, OneCycleLR) or
                                      getattr(self.scheduler, '_step_per_batch', False))
                        if should_step:
                            self.scheduler.step()
                            history['learning_rates'].append(self.optimizer.param_groups[-1]['lr'])
            else:
                # Standard precision training
                outputs = self(stream1_batch, stream2_batch)
                loss = self.criterion(outputs, targets)
                # Scale loss for gradient accumulation
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps
                loss.backward()

                # OPTIMIZATION 5: Only step optimizer every accumulation_steps
                if (batch_idx + 1) % gradient_accumulation_steps == 0 or batch_idx == len(train_loader) - 1:
                    # Gradient clipping (if enabled)
                    if grad_clip_norm is not None:
                        clip_grad_norm_(self.parameters(), grad_clip_norm)

                    # Update weights
                    self.optimizer.step()

                    # Unified scheduler stepping logic
                    # OneCycleLR always steps per batch
                    # CosineAnnealingWarmRestarts can optionally step per batch if step_per_batch=True
                    if self.scheduler is not None:
                        should_step = (isinstance(self.scheduler, OneCycleLR) or
                                      getattr(self.scheduler, '_step_per_batch', False))
                        if should_step:
                            self.scheduler.step()
                            history['learning_rates'].append(self.optimizer.param_groups[-1]['lr'])

            # Calculate training metrics (use original loss for metrics, not scaled)
            original_loss = loss.item() * gradient_accumulation_steps if gradient_accumulation_steps > 1 else loss.item()
            train_loss += original_loss
            train_batches += 1

            # Stream-specific monitoring with auxiliary classifiers (gradient-isolated)
            # This happens AFTER main optimizer.step() so auxiliary classifiers can learn independently
            # Note: Full gradients used (no scaling) for accurate monitoring
            if stream_monitoring:
                # === TRAINING PHASE: Train auxiliary classifiers ===
                # Forward through stream pathways
                stream1_features = self._forward_stream1_pathway(stream1_batch)
                stream2_features = self._forward_stream2_pathway(stream2_batch)

                # DETACH features - stops gradient flow to stream weights!
                stream1_features_detached = stream1_features.detach()
                stream2_features_detached = stream2_features.detach()

                # Classify with auxiliary classifiers (only they get gradients)
                stream1_outputs = self.fc_stream1(stream1_features_detached)
                stream2_outputs = self.fc_stream2(stream2_features_detached)

                # Compute auxiliary losses (full gradient, no scaling)
                stream1_aux_loss = self.criterion(stream1_outputs, targets)
                stream2_aux_loss = self.criterion(stream2_outputs, targets)

                # Backward pass for auxiliary classifiers only (detached features ensure no gradient to streams)
                # Each stream's classifier learns independently with full gradients
                # CRITICAL: Use separate aux_optimizer to avoid affecting main optimizer's internal state
                if self.use_amp:
                    self.scaler.scale(stream1_aux_loss).backward()
                    self.scaler.scale(stream2_aux_loss).backward()
                    # Update auxiliary classifiers using separate optimizer
                    self.scaler.step(aux_optimizer)
                    self.scaler.update()
                else:
                    stream1_aux_loss.backward()
                    stream2_aux_loss.backward()
                    # Update auxiliary classifiers using separate optimizer
                    aux_optimizer.step()

                # Zero gradients for auxiliary optimizer
                aux_optimizer.zero_grad()

                # === ACCURACY MEASUREMENT PHASE: Calculate stream accuracies ===
                with torch.no_grad():
                    was_training = self.training
                    self.eval()  # Disable dropout, use BN running stats

                    stream1_features = self._forward_stream1_pathway(stream1_batch)
                    stream2_features = self._forward_stream2_pathway(stream2_batch)

                    stream1_outputs = self.fc_stream1(stream1_features)
                    stream2_outputs = self.fc_stream2(stream2_features)

                    stream1_pred = stream1_outputs.argmax(1)
                    stream2_pred = stream2_outputs.argmax(1)

                    stream1_train_correct += (stream1_pred == targets).sum().item()
                    stream2_train_correct += (stream2_pred == targets).sum().item()
                    stream_train_total += targets.size(0)

                    self.train(was_training)  # Restore training mode

            # Calculate training accuracy
            with torch.no_grad():
                _, predicted = torch.max(outputs, 1)
                train_total += targets.size(0)
                train_correct += (predicted == targets).sum().item()

            # OPTIMIZATION 1: Update progress bar much less frequently - MAJOR SPEEDUP
            if pbar is not None and (batch_idx % update_frequency == 0 or batch_idx == len(train_loader) - 1):
                # Get base LR (last param group if using stream-specific, otherwise first)
                current_lr = self.optimizer.param_groups[-1]['lr']  # Base LR is last group (shared params)
                current_train_loss = train_loss / train_batches
                current_train_acc = train_correct / train_total
                
                # Update progress bar with training metrics (validation metrics will be updated later)
                postfix = {
                    'train_loss': f'{current_train_loss:.4f}',
                    'train_acc': f'{current_train_acc:.4f}'
                }
                
                # Keep existing validation metrics if they exist
                current_postfix = getattr(pbar, 'postfix', {})
                if isinstance(current_postfix, dict):
                    for key in ['val_loss', 'val_acc']:
                        if key in current_postfix:
                            postfix[key] = current_postfix[key]
                
                # Add lr at the end
                postfix['lr'] = f'{current_lr:.6f}'

                pbar.set_postfix(postfix)

                # Simplified progress bar update - tqdm handles remaining steps automatically
                updates_needed = min(update_frequency, len(train_loader) - pbar.n)
                if updates_needed > 0:
                    pbar.update(updates_needed)
        
        avg_train_loss = train_loss / train_batches
        train_accuracy = train_correct / train_total

        # Calculate stream-specific accuracies
        stream1_train_acc = stream1_train_correct / max(stream_train_total, 1) if stream_monitoring else 0.0
        stream2_train_acc = stream2_train_correct / max(stream_train_total, 1) if stream_monitoring else 0.0

        # Optional: Clear CUDA cache at end of epoch (only if experiencing OOM issues)
        # if clear_cache_per_epoch and self.device.type == 'cuda':
        #     torch.cuda.empty_cache()

        return avg_train_loss, train_accuracy, stream1_train_acc, stream2_train_acc
    
    def _validate(self, data_loader: DataLoader,
                  pbar: Optional['TqdmType'] = None, stream_monitoring: bool = True) -> tuple:
        """
        Validate the model on the given data with GPU optimizations and reduced progress updates.

        Args:
            data_loader: DataLoader containing dual-channel input data and targets
            pbar: Optional progress bar to update during validation
            stream_monitoring: Whether to track stream-specific metrics during validation

        Returns:
            Tuple of (loss, accuracy, stream1_val_acc, stream2_val_acc, stream1_val_loss, stream2_val_loss)
            If stream_monitoring=False, stream accuracies and losses will be 0.0
        """

        self.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        # Stream-specific tracking (if monitoring enabled)
        stream1_val_correct = 0
        stream2_val_correct = 0
        stream1_val_loss = 0.0
        stream2_val_loss = 0.0
        stream_val_total = 0
        
        # OPTIMIZATION 1: Progress bar update frequency for validation - major performance improvement
        update_frequency = max(1, len(data_loader) // 25)  # Update only 25 times during validation
        
        with torch.no_grad():
            for batch_idx, (stream1_batch, stream2_batch, targets) in enumerate(data_loader):
                # GPU optimization: non-blocking transfer for dual-channel data
                stream1_batch = stream1_batch.to(self.device, non_blocking=True)
                stream2_batch = stream2_batch.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                # Use AMP for validation if enabled
                if self.use_amp:
                    with autocast(device_type=self.device.type):
                        outputs = self(stream1_batch, stream2_batch)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self(stream1_batch, stream2_batch)
                    loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

                # Stream-specific monitoring with auxiliary classifiers
                # Model is already in eval() mode, so dropout is disabled
                if stream_monitoring:
                    # Forward through stream pathways
                    stream1_features = self._forward_stream1_pathway(stream1_batch)
                    stream2_features = self._forward_stream2_pathway(stream2_batch)

                    # Classify with auxiliary classifiers
                    stream1_outputs = self.fc_stream1(stream1_features)
                    stream2_outputs = self.fc_stream2(stream2_features)

                    # Calculate stream losses
                    stream1_loss = self.criterion(stream1_outputs, targets)
                    stream2_loss = self.criterion(stream2_outputs, targets)
                    stream1_val_loss += stream1_loss.item() * targets.size(0)
                    stream2_val_loss += stream2_loss.item() * targets.size(0)

                    stream1_pred = stream1_outputs.argmax(1)
                    stream2_pred = stream2_outputs.argmax(1)

                    stream1_val_correct += (stream1_pred == targets).sum().item()
                    stream2_val_correct += (stream2_pred == targets).sum().item()
                    stream_val_total += targets.size(0)

                # OPTIMIZATION 1: Update progress bar much less frequently during validation
                if pbar is not None and (batch_idx % update_frequency == 0 or batch_idx == len(data_loader) - 1):
                    current_val_loss = total_loss / (batch_idx + 1)
                    current_val_acc = correct / total
                    # Get base LR (last param group if using stream-specific, otherwise first)
                    current_lr = self.optimizer.param_groups[-1]['lr']  # Base LR is last group (shared params)
                    
                    # Get existing postfix and update with validation metrics
                    current_postfix = getattr(pbar, 'postfix', {})
                    if isinstance(current_postfix, dict):
                        postfix = current_postfix.copy()
                    else:
                        postfix = {}
                    
                    # Update validation metrics while preserving training metrics
                    postfix.update({
                        'val_loss': f'{current_val_loss:.4f}',
                        'val_acc': f'{current_val_acc:.4f}'
                    })
                    
                    # Add lr at the end
                    postfix['lr'] = f'{current_lr:.6f}'

                    pbar.set_postfix(postfix)

                    # Simplified progress bar update - tqdm handles remaining steps automatically
                    updates_needed = min(update_frequency, len(data_loader) - pbar.n)
                    if updates_needed > 0:
                        pbar.update(updates_needed)
        
        avg_loss = total_loss / len(data_loader)
        accuracy = correct / total

        # Calculate stream-specific accuracies and losses
        stream1_val_acc = stream1_val_correct / max(stream_val_total, 1) if stream_monitoring else 0.0
        stream2_val_acc = stream2_val_correct / max(stream_val_total, 1) if stream_monitoring else 0.0
        avg_stream1_val_loss = stream1_val_loss / max(stream_val_total, 1) if stream_monitoring else 0.0
        avg_stream2_val_loss = stream2_val_loss / max(stream_val_total, 1) if stream_monitoring else 0.0

        return avg_loss, accuracy, stream1_val_acc, stream2_val_acc, avg_stream1_val_loss, avg_stream2_val_loss

    def _print_stream_monitoring(self, stream1_train_acc: float, stream1_val_acc: float,
                                 stream2_train_acc: float, stream2_val_acc: float,
                                 stream1_val_loss: float = 0.0, stream2_val_loss: float = 0.0) -> dict:
        """
        Print stream-specific monitoring metrics (computed during main training loop).

        Args:
            stream1_train_acc: Stream1 training accuracy (from full epoch)
            stream1_val_acc: Stream1 validation accuracy (from full epoch)
            stream2_train_acc: Stream2 training accuracy (from full epoch)
            stream2_val_acc: Stream2 validation accuracy (from full epoch)
            stream1_val_loss: Stream1 validation loss (from full epoch)
            stream2_val_loss: Stream2 validation loss (from full epoch)

        Returns:
            Dictionary containing stream-specific metrics for history tracking
        """
        # Get parameter groups info
        param_groups = self.optimizer.param_groups

        # Build single-line monitoring output for 2 input streams
        if len(param_groups) >= 3:
            # Stream1 (RGB)
            stream1_lr = param_groups[0]['lr']
            stream1_wd = param_groups[0]['weight_decay']

            # Stream2 (Depth)
            stream2_lr = param_groups[1]['lr']
            stream2_wd = param_groups[1]['weight_decay']

            # Display only stream1 and stream2 (full model metrics shown in progress bar above)
            print(f"  Stream1: V_loss:{stream1_val_loss:.4f}, T_acc:{stream1_train_acc:.4f}, V_acc:{stream1_val_acc:.4f}, LR:{stream1_lr:.2e} | "
                  f"Stream2: V_loss:{stream2_val_loss:.4f}, T_acc:{stream2_train_acc:.4f}, V_acc:{stream2_val_acc:.4f}, LR:{stream2_lr:.2e}")

            # Return stream stats for history tracking
            return {
                'stream1_train_acc': stream1_train_acc,
                'stream1_val_acc': stream1_val_acc,
                'stream1_val_loss': stream1_val_loss,
                'stream2_train_acc': stream2_train_acc,
                'stream2_val_acc': stream2_val_acc,
                'stream2_val_loss': stream2_val_loss,
                'stream1_lr': stream1_lr,
                'stream1_wd': stream1_wd,
                'stream2_lr': stream2_lr,
                'stream2_wd': stream2_wd
            }

        return {}

    @property
    def fusion_strategy(self) -> str:
        """
        The type of fusion used in the model.

        For LINet, integration happens inside LIConv2d neurons (not separate fusion layer).

        Returns:
            A string representing the fusion type.
        """
        return "linear_integration"
    
    def _forward_stream1_pathway(self, stream1_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the stream1 pathway only.

        Args:
            stream1_input: The stream1 input tensor.

        Returns:
            The stream1 pathway output tensor (flattened features).
        """
        # Process through stream1 pathway of multi-channel layers
        stream1_x = stream1_input

        # Initial convolution (stream1 pathway only)
        stream1_x = self.conv1.forward_stream1(stream1_x)
        stream1_x = self.bn1.forward_stream1(stream1_x)
        stream1_x = self.relu.forward_stream1(stream1_x)

        # Max pooling (stream1 pathway only)
        stream1_x = self.maxpool.forward_stream1(stream1_x)

        # ResNet layers (stream1 pathway only)
        stream1_x = self.layer1.forward_stream1(stream1_x)
        stream1_x = self.layer2.forward_stream1(stream1_x)
        stream1_x = self.layer3.forward_stream1(stream1_x)
        stream1_x = self.layer4.forward_stream1(stream1_x)

        # Global average pooling (stream1 pathway only)
        # Note: avgpool is standard nn.AdaptiveAvgPool2d, just call it directly
        stream1_x = self.avgpool(stream1_x)

        # Flatten
        stream1_x = torch.flatten(stream1_x, 1)

        return stream1_x
    
    def _forward_stream2_pathway(self, stream2_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the stream2 pathway only.

        Args:
            stream2_input: The stream2 input tensor.

        Returns:
            The stream2 pathway output tensor (flattened features).
        """
        # Process through stream2 pathway of multi-channel layers
        stream2_x = stream2_input

        # Initial convolution (stream2 pathway only)
        stream2_x = self.conv1.forward_stream2(stream2_x)
        stream2_x = self.bn1.forward_stream2(stream2_x)
        stream2_x = self.relu.forward_stream2(stream2_x)

        # Max pooling (stream2 pathway only)
        stream2_x = self.maxpool.forward_stream2(stream2_x)

        # ResNet layers (stream2 pathway only)
        stream2_x = self.layer1.forward_stream2(stream2_x)
        stream2_x = self.layer2.forward_stream2(stream2_x)
        stream2_x = self.layer3.forward_stream2(stream2_x)
        stream2_x = self.layer4.forward_stream2(stream2_x)

        # Global average pooling (stream2 pathway only)
        # Note: avgpool is standard nn.AdaptiveAvgPool2d, just call it directly
        stream2_x = self.avgpool(stream2_x)

        # Flatten
        stream2_x = torch.flatten(stream2_x, 1)

        return stream2_x

    def _forward_integrated_pathway(self, stream1_input: torch.Tensor, stream2_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the integrated pathway only (without stream1/stream2).

        This extracts only the integrated stream features by running the full forward pass
        and returning only the integrated stream output.

        Args:
            stream1_input: The stream1 input tensor (needed for integration).
            stream2_input: The stream2 input tensor (needed for integration).

        Returns:
            The integrated pathway output tensor (flattened features).
        """
        # Initial convolution (creates integrated stream from stream1 + stream2)
        s1, s2, ic = self.conv1(stream1_input, stream2_input, None)
        s1, s2, ic = self.bn1(s1, s2, ic)
        s1, s2, ic = self.relu(s1, s2, ic)

        # Max pooling (all 3 streams)
        s1, s2, ic = self.maxpool(s1, s2, ic)

        # ResNet layers (integration happens inside LIConv2d neurons)
        s1, s2, ic = self.layer1(s1, s2, ic)
        s1, s2, ic = self.layer2(s1, s2, ic)
        s1, s2, ic = self.layer3(s1, s2, ic)
        s1, s2, ic = self.layer4(s1, s2, ic)

        # Global average pooling for integrated stream
        integrated_pooled = self.avgpool(ic)

        # Flatten integrated features
        integrated_features = torch.flatten(integrated_pooled, 1)

        return integrated_features

    def analyze_pathways(self, data_loader: DataLoader) -> dict:
        """
        Analyze the contribution and performance of individual pathways.

        Uses auxiliary classifiers (fc_stream1, fc_stream2) for stream1/stream2 accuracy.
        Uses main classifier (fc) for integrated pathway and full model accuracy.

        Args:
            data_loader: DataLoader containing dual-channel input data

        Returns:
            Dictionary containing pathway analysis results for all 3 streams:
            - accuracy: stream1_only, stream2_only, integrated_only, full_model
            - loss: for each pathway
            - feature_norms: mean/std for each pathway's feature activations
        """
        if self.criterion is None:
            raise ValueError("Model not compiled. Call compile() before analyze_pathways().")

        self.eval()

        # Initialize metrics for all 3 streams + full model
        full_model_correct = 0
        stream1_only_correct = 0
        stream2_only_correct = 0
        integrated_only_correct = 0
        total_samples = 0

        full_model_loss = 0.0
        stream1_only_loss = 0.0
        stream2_only_loss = 0.0
        integrated_only_loss = 0.0

        stream1_feature_norms = []
        stream2_feature_norms = []
        integrated_feature_norms = []

        with torch.no_grad():
            for stream1_batch, stream2_batch, targets_batch in data_loader:
                # Move to device
                stream1_batch = stream1_batch.to(self.device, non_blocking=True)
                stream2_batch = stream2_batch.to(self.device, non_blocking=True)
                targets_batch = targets_batch.to(self.device, non_blocking=True)

                batch_size_actual = stream1_batch.size(0)
                total_samples += batch_size_actual

                # Full model prediction (uses integrated stream)
                full_outputs = self(stream1_batch, stream2_batch)
                full_loss = self.criterion(full_outputs, targets_batch)
                full_model_loss += full_loss.item() * batch_size_actual

                _, full_predicted = torch.max(full_outputs, 1)
                full_model_correct += (full_predicted == targets_batch).sum().item()

                # Stream1 pathway only - USE AUXILIARY CLASSIFIER
                stream1_features = self._forward_stream1_pathway(stream1_batch)
                stream1_feature_norms.append(torch.norm(stream1_features, dim=1).cpu())
                stream1_outputs = self.fc_stream1(stream1_features)  # Auxiliary classifier!
                stream1_loss = self.criterion(stream1_outputs, targets_batch)
                stream1_only_loss += stream1_loss.item() * batch_size_actual

                _, stream1_predicted = torch.max(stream1_outputs, 1)
                stream1_only_correct += (stream1_predicted == targets_batch).sum().item()

                # Stream2 pathway only - USE AUXILIARY CLASSIFIER
                stream2_features = self._forward_stream2_pathway(stream2_batch)
                stream2_feature_norms.append(torch.norm(stream2_features, dim=1).cpu())
                stream2_outputs = self.fc_stream2(stream2_features)  # Auxiliary classifier!
                stream2_loss = self.criterion(stream2_outputs, targets_batch)
                stream2_only_loss += stream2_loss.item() * batch_size_actual

                _, stream2_predicted = torch.max(stream2_outputs, 1)
                stream2_only_correct += (stream2_predicted == targets_batch).sum().item()

                # Integrated pathway only (runs full forward but uses only integrated features)
                integrated_features = self._forward_integrated_pathway(stream1_batch, stream2_batch)
                integrated_feature_norms.append(torch.norm(integrated_features, dim=1).cpu())
                integrated_outputs = self.fc(integrated_features)
                integrated_loss = self.criterion(integrated_outputs, targets_batch)
                integrated_only_loss += integrated_loss.item() * batch_size_actual

                _, integrated_predicted = torch.max(integrated_outputs, 1)
                integrated_only_correct += (integrated_predicted == targets_batch).sum().item()

        # Calculate metrics
        full_accuracy = full_model_correct / total_samples
        stream1_accuracy = stream1_only_correct / total_samples
        stream2_accuracy = stream2_only_correct / total_samples
        integrated_accuracy = integrated_only_correct / total_samples

        avg_full_loss = full_model_loss / total_samples
        avg_stream1_loss = stream1_only_loss / total_samples
        avg_stream2_loss = stream2_only_loss / total_samples
        avg_integrated_loss = integrated_only_loss / total_samples

        # Feature analysis
        stream1_norms = torch.cat(stream1_feature_norms, dim=0)
        stream2_norms = torch.cat(stream2_feature_norms, dim=0)
        integrated_norms = torch.cat(integrated_feature_norms, dim=0)

        return {
            'accuracy': {
                'full_model': full_accuracy,
                'stream1_only': stream1_accuracy,
                'stream2_only': stream2_accuracy,
                'integrated_only': integrated_accuracy,
                'stream1_contribution': stream1_accuracy / full_accuracy if full_accuracy > 0 else 0,
                'stream2_contribution': stream2_accuracy / full_accuracy if full_accuracy > 0 else 0,
                'integrated_contribution': integrated_accuracy / full_accuracy if full_accuracy > 0 else 0
            },
            'loss': {
                'full_model': avg_full_loss,
                'stream1_only': avg_stream1_loss,
                'stream2_only': avg_stream2_loss,
                'integrated_only': avg_integrated_loss
            },
            'feature_norms': {
                'stream1_mean': stream1_norms.mean().item(),
                'stream1_std': stream1_norms.std().item(),
                'stream2_mean': stream2_norms.mean().item(),
                'stream2_std': stream2_norms.std().item(),
                'integrated_mean': integrated_norms.mean().item(),
                'integrated_std': integrated_norms.std().item(),
                'stream1_to_stream2_ratio': (stream1_norms.mean() / stream2_norms.mean()).item() if stream2_norms.mean() > 0 else float('inf'),
                'integrated_to_stream1_ratio': (integrated_norms.mean() / stream1_norms.mean()).item() if stream1_norms.mean() > 0 else float('inf')
            },
            'samples_analyzed': total_samples
        }
    
    def analyze_pathway_weights(self) -> dict:
        """
        Analyze the weight distributions and magnitudes across all 3 pathways.

        Returns:
            Dictionary containing weight analysis for stream1, stream2, and integrated pathways
        """
        stream1_weights = {}
        stream2_weights = {}
        integrated_weights = {}
        integration_weights = {}

        # Analyze LI layers - look for modules with stream1_weight, stream2_weight, and integrated_weight
        for name, module in self.named_modules():
            if hasattr(module, 'stream1_weight') and hasattr(module, 'stream2_weight') and hasattr(module, 'integrated_weight'):
                # LIConv2d modules
                stream1_weight = module.stream1_weight
                stream2_weight = module.stream2_weight
                integrated_weight = module.integrated_weight

                stream1_weights[name] = {
                    'mean': stream1_weight.mean().item(),
                    'std': stream1_weight.std().item(),
                    'norm': torch.norm(stream1_weight).item(),
                    'shape': list(stream1_weight.shape)
                }

                stream2_weights[name] = {
                    'mean': stream2_weight.mean().item(),
                    'std': stream2_weight.std().item(),
                    'norm': torch.norm(stream2_weight).item(),
                    'shape': list(stream2_weight.shape)
                }

                integrated_weights[name] = {
                    'mean': integrated_weight.mean().item(),
                    'std': integrated_weight.std().item(),
                    'norm': torch.norm(integrated_weight).item(),
                    'shape': list(integrated_weight.shape)
                }

                # Integration weights (from stream1 and stream2 to integrated)
                if hasattr(module, 'integration_from_stream1') and hasattr(module, 'integration_from_stream2'):
                    int_from_s1 = module.integration_from_stream1
                    int_from_s2 = module.integration_from_stream2

                    integration_weights[name] = {
                        'from_stream1': {
                            'mean': int_from_s1.mean().item(),
                            'std': int_from_s1.std().item(),
                            'norm': torch.norm(int_from_s1).item(),
                            'shape': list(int_from_s1.shape)
                        },
                        'from_stream2': {
                            'mean': int_from_s2.mean().item(),
                            'std': int_from_s2.std().item(),
                            'norm': torch.norm(int_from_s2).item(),
                            'shape': list(int_from_s2.shape)
                        }
                    }

        # Calculate overall statistics
        stream1_norms = [w['norm'] for w in stream1_weights.values()]
        stream2_norms = [w['norm'] for w in stream2_weights.values()]
        integrated_norms = [w['norm'] for w in integrated_weights.values()]

        return {
            'stream1_pathway': {
                'layer_weights': stream1_weights,
                'total_norm': sum(stream1_norms),
                'mean_norm': sum(stream1_norms) / len(stream1_norms) if stream1_norms else 0,
                'num_layers': len(stream1_weights)
            },
            'stream2_pathway': {
                'layer_weights': stream2_weights,
                'total_norm': sum(stream2_norms),
                'mean_norm': sum(stream2_norms) / len(stream2_norms) if stream2_norms else 0,
                'num_layers': len(stream2_weights)
            },
            'integrated_pathway': {
                'layer_weights': integrated_weights,
                'total_norm': sum(integrated_norms),
                'mean_norm': sum(integrated_norms) / len(integrated_norms) if integrated_norms else 0,
                'num_layers': len(integrated_weights)
            },
            'integration_weights': integration_weights,
            'ratio_analysis': {
                'stream1_to_stream2_norm_ratio': (sum(stream1_norms) / sum(stream2_norms)) if stream2_norms else float('inf'),
                'integrated_to_stream1_norm_ratio': (sum(integrated_norms) / sum(stream1_norms)) if stream1_norms else float('inf'),
                'layer_ratios': {
                    name: {
                        's1_to_s2': stream1_weights[name]['norm'] / stream2_weights[name]['norm']
                        if name in stream2_weights and stream2_weights[name]['norm'] > 0 else float('inf'),
                        'int_to_s1': integrated_weights[name]['norm'] / stream1_weights[name]['norm']
                        if stream1_weights[name]['norm'] > 0 else float('inf')
                    }
                    for name in stream1_weights.keys()
                    if name in stream2_weights and name in integrated_weights
                }
            }
        }
    
    def get_pathway_importance(self,
                              data_loader: DataLoader,
                              method: str = 'gradient') -> dict:
        """
        Calculate pathway importance using different methods.

        Args:
            data_loader: DataLoader containing dual-channel input data
            method: Method for importance calculation ('gradient', 'ablation', 'feature_norm')

        Returns:
            Dictionary containing pathway importance scores
        """
        if method == 'gradient':
            return self._calculate_gradient_importance(data_loader)
        elif method == 'ablation':
            return self.calculate_stream_contributions(data_loader)
        elif method == 'feature_norm':
            return self.calculate_feature_norm_importance(data_loader)
        else:
            raise ValueError(f"Unknown importance method: {method}. Choose from 'gradient', 'ablation', 'feature_norm'")
    
    def _calculate_gradient_importance(self, data_loader):
        """
        Calculate importance based on gradient magnitudes.

        Note: For LINet, the integrated stream is created FROM stream1 and stream2,
        so we measure input gradients for stream1 and stream2 only.
        The integrated stream's importance is implicit in how gradients flow back
        through the integration weights to the input streams.
        """
        if self.criterion is None:
            raise ValueError("Model not compiled. Call compile() before calculating importance.")

        # Save current training state
        was_training = self.training
        self.train()  # Enable gradients

        stream1_gradients = []
        stream2_gradients = []

        for stream1_batch, stream2_batch, targets_batch in data_loader:
            stream1_batch = stream1_batch.to(self.device, non_blocking=True)
            stream2_batch = stream2_batch.to(self.device, non_blocking=True)
            targets_batch = targets_batch.to(self.device, non_blocking=True)

            # Require gradients for inputs
            stream1_batch.requires_grad_(True)
            stream2_batch.requires_grad_(True)

            # Forward pass (creates integrated stream internally)
            outputs = self(stream1_batch, stream2_batch)
            loss = self.criterion(outputs, targets_batch)

            # Backward pass
            loss.backward()

            # Calculate gradient norms - flatten and then compute norm
            stream1_grad_norm = torch.norm(stream1_batch.grad.flatten(1), dim=1).mean().item()
            stream2_grad_norm = torch.norm(stream2_batch.grad.flatten(1), dim=1).mean().item()

            stream1_gradients.append(stream1_grad_norm)
            stream2_gradients.append(stream2_grad_norm)

            # Clear gradients
            self.zero_grad()
            stream1_batch.grad = None
            stream2_batch.grad = None

        # Restore original training state
        self.train(was_training)

        avg_stream1_grad = sum(stream1_gradients) / len(stream1_gradients)
        avg_stream2_grad = sum(stream2_gradients) / len(stream2_gradients)
        total_grad = avg_stream1_grad + avg_stream2_grad

        return {
            'method': 'gradient',
            'stream1_importance': avg_stream1_grad / total_grad if total_grad > 0 else 0.5,
            'stream2_importance': avg_stream2_grad / total_grad if total_grad > 0 else 0.5,
            'raw_gradients': {
                'stream1_avg': avg_stream1_grad,
                'stream2_avg': avg_stream2_grad
            },
            'note': 'Integrated stream importance is implicit in how gradients flow through integration weights'
        }

    def calculate_stream_contributions_to_integration(self, data_loader: DataLoader = None):
        """
        Calculate how much each input stream contributes to the integrated stream.

        This method measures architectural contribution by analyzing integration weight magnitudes:
        ||integration_from_stream1|| vs ||integration_from_stream2||

        This directly answers: "How much does the model architecture favor each stream?"

        Simple, fast, and interpretable:
        - No data required (analyzes learned weights)
        - Stable (doesn't vary with data sampling)
        - Clear meaning: "The model uses X% from stream1, Y% from stream2"

        Args:
            data_loader: Not used, kept for backward compatibility

        Returns:
            Dictionary containing:
            - stream1_contribution: float (0-1) - proportion from stream1 based on integration weights
            - stream2_contribution: float (0-1) - proportion from stream2 based on integration weights
            - raw_norms: dict with actual weight norms for each stream
        """
        # Analyze integration weight magnitudes across all LIConv2d layers
        stream1_weight_norm = 0.0
        stream2_weight_norm = 0.0

        for name, param in self.named_parameters():
            if 'integration_from_stream1' in name:
                stream1_weight_norm += torch.norm(param).item()
            elif 'integration_from_stream2' in name:
                stream2_weight_norm += torch.norm(param).item()

        total_weight_norm = stream1_weight_norm + stream2_weight_norm

        return {
            'method': 'integration_weights',
            'stream1_contribution': stream1_weight_norm / total_weight_norm if total_weight_norm > 0 else 0.5,
            'stream2_contribution': stream2_weight_norm / total_weight_norm if total_weight_norm > 0 else 0.5,
            'raw_norms': {
                'stream1_integration_weights': stream1_weight_norm,
                'stream2_integration_weights': stream2_weight_norm,
                'total': total_weight_norm
            },
            'interpretation': {
                'stream1_percentage': f"{100 * stream1_weight_norm / total_weight_norm:.1f}%" if total_weight_norm > 0 else "50.0%",
                'stream2_percentage': f"{100 * stream2_weight_norm / total_weight_norm:.1f}%" if total_weight_norm > 0 else "50.0%"
            },
            'note': 'Measures architectural contribution based on integration weight magnitudes. '
                   'Reflects how the trained model structurally combines the two input streams.'
        }

    def calculate_stream_contributions(self, data_loader: DataLoader):
        """
        DEPRECATED: Calculate hypothetical classification ability of each stream.

        WARNING: This method is MISLEADING for LINet because stream1/stream2 don't directly classify.
        They only feed into the integrated stream through integration weights.

        This method measures: "How well could each stream classify IF it had its own classifier?"
        But in LINet, streams DON'T have classifiers - only the integrated stream does.

        **Use calculate_stream_contributions_to_integration() instead** for meaningful analysis
        of how much each input stream contributes to the integrated stream's features.

        Args:
            data_loader: DataLoader containing dual-channel input data

        Returns:
            Dictionary containing hypothetical stream classification abilities (misleading for LINet)
        """
        import warnings
        warnings.warn(
            "calculate_stream_contributions() is deprecated and misleading for LINet. "
            "It measures hypothetical classification ability, but streams don't directly classify. "
            "Use calculate_stream_contributions_to_integration() instead for meaningful contribution analysis.",
            DeprecationWarning,
            stacklevel=2
        )
        # Use the analyze_pathways method for ablation analysis
        pathway_analysis = self.analyze_pathways(data_loader)

        full_accuracy = pathway_analysis['accuracy']['full_model']
        stream1_accuracy = pathway_analysis['accuracy']['stream1_only']
        stream2_accuracy = pathway_analysis['accuracy']['stream2_only']
        integrated_accuracy = pathway_analysis['accuracy']['integrated_only']

        # Calculate contribution scores
        # The integrated stream should perform best (or equal to full model) since that's what we use for prediction
        total_acc = stream1_accuracy + stream2_accuracy + integrated_accuracy

        return {
            'method': 'ablation',
            'stream1_importance': stream1_accuracy / total_acc if total_acc > 0 else 0.33,
            'stream2_importance': stream2_accuracy / total_acc if total_acc > 0 else 0.33,
            'integrated_importance': integrated_accuracy / total_acc if total_acc > 0 else 0.34,
            'individual_accuracies': {
                'full_model': full_accuracy,
                'stream1_only': stream1_accuracy,
                'stream2_only': stream2_accuracy,
                'integrated_only': integrated_accuracy
            },
            'relative_to_full_model': {
                'stream1_ratio': stream1_accuracy / full_accuracy if full_accuracy > 0 else 0,
                'stream2_ratio': stream2_accuracy / full_accuracy if full_accuracy > 0 else 0,
                'integrated_ratio': integrated_accuracy / full_accuracy if full_accuracy > 0 else 1.0
            },
            'note': 'Integrated stream should perform best as it combines information from both input streams'
        }
    
    def calculate_feature_norm_importance(self, data_loader):
        """Calculate importance based on feature norm magnitudes for all 3 streams."""
        # Use the analyze_pathways method for feature analysis
        pathway_analysis = self.analyze_pathways(data_loader)

        stream1_norm = pathway_analysis['feature_norms']['stream1_mean']
        stream2_norm = pathway_analysis['feature_norms']['stream2_mean']
        integrated_norm = pathway_analysis['feature_norms']['integrated_mean']
        total_norm = stream1_norm + stream2_norm + integrated_norm

        return {
            'method': 'feature_norm',
            'stream1_importance': stream1_norm / total_norm if total_norm > 0 else 0.33,
            'stream2_importance': stream2_norm / total_norm if total_norm > 0 else 0.33,
            'integrated_importance': integrated_norm / total_norm if total_norm > 0 else 0.34,
            'feature_norms': {
                'stream1_mean': stream1_norm,
                'stream2_mean': stream2_norm,
                'integrated_mean': integrated_norm,
                'stream1_to_stream2_ratio': stream1_norm / stream2_norm if stream2_norm > 0 else float('inf'),
                'integrated_to_stream1_ratio': integrated_norm / stream1_norm if stream1_norm > 0 else float('inf')
            }
        } 

# Factory functions for common LINet architectures
def li_resnet18(num_classes: int = 1000, **kwargs) -> LINet:
    """Create a Linear Integration ResNet-18 model."""
    return LINet(
        LIBasicBlock,
        [2, 2, 2, 2],
        num_classes=num_classes,
        **kwargs
    )

def li_resnet34(num_classes: int = 1000, **kwargs) -> LINet:
    """Create a Linear Integration ResNet-34 model."""
    return LINet(
        LIBasicBlock,
        [3, 4, 6, 3],
        num_classes=num_classes,
        **kwargs
    )


def li_resnet50(num_classes: int = 1000, **kwargs) -> LINet:
    """Create a Linear Integration ResNet-50 model."""
    return LINet(
        LIBottleneck,
        [3, 4, 6, 3],
        num_classes=num_classes,
        **kwargs
    )


def li_resnet101(num_classes: int = 1000, **kwargs) -> LINet:
    """Create a Linear Integration ResNet-101 model."""
    return LINet(
        LIBottleneck,
        [3, 4, 23, 3],
        num_classes=num_classes,
        **kwargs
    )


def li_resnet152(num_classes: int = 1000, **kwargs) -> LINet:
    """Create a Linear Integration ResNet-152 model."""
    return LINet(
        LIBottleneck,
        [3, 8, 36, 3],
        num_classes=num_classes,
        **kwargs
    )