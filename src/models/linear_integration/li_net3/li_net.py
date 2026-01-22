"""
Linear Integration ResNet (LINet) implementations.

Supports N input streams with a learned integrated pathway.
Designed for flexibility - works with any number of streams (N=1, 2, 3, ...).
"""

import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from sched import scheduler
from typing import Any, Callable, Optional, Union, TYPE_CHECKING
from torch import Tensor
from torch.amp import autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts

from src.models.abstracts.abstract_model import BaseModel
from src.training.schedulers import setup_scheduler
from src.training.modality_dropout import get_modality_dropout_prob, generate_per_sample_blanked_mask
from src.models.common import (
    save_checkpoint,
    create_progress_bar,
    finalize_progress_bar,
    update_history
)

# Smart tqdm import - detect environment
if TYPE_CHECKING:
    from tqdm import tqdm as TqdmType

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
from .conv import LIConv2d, LIBatchNorm2d
from .blocks import LIBasicBlock, LIBottleneck
from .container import LISequential, LIReLU
from .pooling import LIMaxPool2d, LIAdaptiveAvgPool2d
# Note: LINet doesn't use fusion - it uses the integrated stream directly


class LINet(BaseModel):
    """
    Linear Integration ResNet (LINet) implementation.

    Supports N input streams with a learned integrated pathway:
    - N independent stream pathways (e.g., RGB, Depth, Orthogonal, etc.)
    - Integrated: Learned fusion pathway (combines stream outputs during convolution)

    Uses unified LIConv2d neurons where integration happens INSIDE the neuron.
    Final predictions come from the integrated stream (no separate fusion layer needed).

    Designed for flexibility - works with any number of streams (N=1, 2, 3, ...):
    - N=2: [RGB, Depth] (default for backward compatibility)
    - N=3: [RGB, Depth, Orthogonal]
    - N=4+: Arbitrary modalities
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
        stream_input_channels: list[int] = [3, 1],  # Default: [RGB=3, Depth=1] for backward compatibility
        dropout_p: float = 0.0,  # Dropout probability (0.0 = no dropout, 0.5 = 50% dropout)
        **kwargs
    ) -> None:
        # Store LINet-specific parameters BEFORE calling super().__init__
        # because _build_network() (called by super()) needs these attributes
        self.stream_input_channels = stream_input_channels
        self.num_streams = len(stream_input_channels)
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
        # Initialize stream inplanes tracking for N streams
        # All streams start at 64 channels (standard ResNet convention)
        self.stream_inplanes = [64] * self.num_streams
        self.integrated_inplanes = 64

        # Network architecture - exactly like ResNet but with N-stream LI components
        # First conv: no integrated input yet (integrated starts after first block)
        self.conv1 = LIConv2d(
            self.stream_input_channels,  # list[int] of input channels for each stream
            self.stream_inplanes,  # list[int] of output channels for each stream (all 64)
            0,  # integrated_in_channels (doesn't exist yet)
            self.integrated_inplanes,  # integrated_out_channels (64)
            kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = self._norm_layer(self.stream_inplanes, self.integrated_inplanes)
        self.relu = LIReLU(inplace=True)
        self.maxpool = LIMaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers - equal scaling for all N streams
        # Integrated pathway follows standard ResNet channel progression: 64, 128, 256, 512
        self.layer1 = self._make_layer(block, [64] * self.num_streams, 64, layers[0])
        self.layer2 = self._make_layer(block, [128] * self.num_streams, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, [256] * self.num_streams, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, [512] * self.num_streams, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        # Adaptive average pooling for integrated stream only
        # (Stream pooling happens in LIMaxPool2d layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Add dropout for regularization (configurable, critical for small datasets)
        self.dropout = nn.Dropout(p=self.dropout_p) if self.dropout_p > 0.0 else nn.Identity()

        # Single classifier for integrated stream features
        feature_dim = 512 * block.expansion
        self.fc = nn.Linear(512 * block.expansion, self.num_classes)

        # Auxiliary classifiers for stream monitoring (gradient-isolated)
        # These learn to classify from stream features but DON'T affect stream training via .detach()
        # Only used when stream_monitoring=True
        self.fc_streams = nn.ModuleList([
            nn.Linear(feature_dim, self.num_classes)
            for _ in range(self.num_streams)
        ])
    
    def _initialize_weights(self, zero_init_residual: bool):
        """Initialize network weights for N-stream architecture."""
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
                if isinstance(m, LIBottleneck):
                    # Zero-init all stream BN weights
                    for stream_weight in m.bn3.stream_weights:
                        nn.init.constant_(stream_weight, 0)
                    # Zero-init integrated stream's last BN weight
                    nn.init.constant_(m.bn3.integrated_weight, 0)
                elif isinstance(m, LIBasicBlock):
                    # Zero-init all stream BN weights
                    for stream_weight in m.bn2.stream_weights:
                        nn.init.constant_(stream_weight, 0)
                    # Zero-init integrated stream's last BN weight
                    nn.init.constant_(m.bn2.integrated_weight, 0)

    def _make_layer(
        self,
        block: type[Union[LIBasicBlock, LIBottleneck]],
        stream_planes: list[int],  # Channel counts for N streams
        integrated_planes: int,  # Channel count for integrated pathway
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> LISequential:
        """
        Create a layer composed of multiple residual blocks with N-stream support.
        Fully compliant with ResNet implementation but using LI blocks with N-stream channel tracking.
        """
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        # Check if downsampling is needed (exactly like original ResNet)
        if stride != 1 or self.stream_inplanes != [p * block.expansion for p in stream_planes] or self.integrated_inplanes != integrated_planes * block.expansion:
            downsample = LISequential(
                LIConv2d(
                    self.stream_inplanes,
                    [p * block.expansion for p in stream_planes],
                    self.integrated_inplanes,
                    integrated_planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                norm_layer([p * block.expansion for p in stream_planes], integrated_planes * block.expansion)
            )

        layers = []
        # First block with potential downsampling
        layers.append(
            block(
                self.stream_inplanes,
                stream_planes,
                self.integrated_inplanes,
                integrated_planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer
            )
        )

        # Update inplanes (exactly like original ResNet)
        self.stream_inplanes = [p * block.expansion for p in stream_planes]
        self.integrated_inplanes = integrated_planes * block.expansion

        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.stream_inplanes,
                    stream_planes,
                    self.integrated_inplanes,
                    integrated_planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return LISequential(*layers)
    
    def forward(
        self,
        stream_inputs: list[Tensor],
        blanked_mask: Optional[dict[int, Tensor]] = None
    ) -> Tensor:
        """
        Forward pass through the Linear Integration ResNet.

        Args:
            stream_inputs: List of input tensors, one per stream [batch_size, channels, height, width]
                          For N=2: [RGB, Depth]
                          For N=3: [RGB, Depth, Orthogonal]
            blanked_mask: Optional per-sample blanking mask for modality dropout.
                         dict[stream_idx] -> bool tensor [batch_size] where True = blanked.
                         When a stream is blanked for a sample, its output is zeroed.

        Returns:
            Classification logits [batch_size, num_classes] from integrated stream
        """
        # Initial convolution (creates integrated stream from all input streams)
        stream_outputs, integrated = self.conv1(stream_inputs, None, blanked_mask)  # integrated_input=None for first layer
        stream_outputs, integrated = self.bn1(stream_outputs, integrated, blanked_mask)
        stream_outputs, integrated = self.relu(stream_outputs, integrated, blanked_mask)

        # Max pooling (all N streams + integrated)
        stream_outputs, integrated = self.maxpool(stream_outputs, integrated, blanked_mask)

        # ResNet layers (all N streams, integration happens inside LIConv2d neurons!)
        stream_outputs, integrated = self.layer1(stream_outputs, integrated, blanked_mask)
        stream_outputs, integrated = self.layer2(stream_outputs, integrated, blanked_mask)
        stream_outputs, integrated = self.layer3(stream_outputs, integrated, blanked_mask)
        stream_outputs, integrated = self.layer4(stream_outputs, integrated, blanked_mask)

        # Global average pooling for integrated stream only
        # (We only use integrated stream for final prediction)
        integrated_pooled = self.avgpool(integrated)

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
                                     stream_patience: Union[int, list[int]],
                                     min_delta: float, verbose: bool) -> dict:
        """
        Initialize stream-specific early stopping state for N streams.

        Args:
            stream_patience: Either a single int (same patience for all streams) or
                           list[int] with patience value for each stream

        Returns a dictionary with per-stream state and a full model checkpoint
        for when all streams freeze.
        """
        if not enabled:
            return {'enabled': False}

        # Validate monitor
        if monitor not in ('val_loss', 'val_accuracy'):
            raise ValueError(f"Invalid monitor: {monitor}. Use 'val_loss' or 'val_accuracy'.")

        # Convert stream_patience to list if needed
        if isinstance(stream_patience, int):
            patience_list = [stream_patience] * self.num_streams
        else:
            patience_list = stream_patience
            if len(patience_list) != self.num_streams:
                raise ValueError(f"stream_patience list length ({len(patience_list)}) must match num_streams ({self.num_streams})")

        # Initialize best metric based on monitor type
        best_metric = float('inf') if monitor == 'val_loss' else 0.0

        if verbose:
            print(f"❄️  Stream-specific early stopping enabled:")
            print(f"   Monitor: {monitor}")
            patience_str = ", ".join([f"Stream{i} patience: {p}" for i, p in enumerate(patience_list)])
            print(f"   {patience_str}")
            print(f"   Min delta: {min_delta}")

        # Create state for each stream dynamically
        state = {
            'enabled': True,
            'streams': [],
            'all_frozen': False,
            'best_full_model': {
                'best_metric': best_metric,
                'epoch': 0,
                'weights': None
            }
        }

        # Initialize state for each stream
        for i in range(self.num_streams):
            state['streams'].append({
                'best_metric': best_metric,
                'patience': patience_list[i],
                'patience_counter': 0,
                'best_epoch': 0,
                'frozen': False,
                'best_weights': None
            })

        return state

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
        for i, stream_state in enumerate(stream_early_stopping_state['streams']):
            if stream_state['frozen']:
                best_weights = stream_state['best_weights']
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
                for i, stream_state in enumerate(stream_early_stopping_state['streams']):
                    if stream_state['frozen']:
                        frozen_streams.append(f"Stream{i}")

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
                                  stream_index: int) -> bool:
        """
        Check if a stream has improved and update its state accordingly.

        Args:
            stream_index: Index of the stream (0, 1, 2, ...)

        Returns:
            True if improvement detected, False otherwise
        """
        if self._should_update_checkpoint(stream_metric, stream_state['best_metric'], monitor, min_delta):
            # Improvement detected
            stream_state['best_metric'] = stream_metric
            stream_state['best_epoch'] = epoch
            stream_state['patience_counter'] = 0

            # Save best stream weights using state_dict
            # Match parameters with .stream_weights.{stream_index}. pattern
            model_state = self.state_dict()
            stream_state['best_weights'] = {
                name: weight.cpu().clone()
                for name, weight in model_state.items()
                if f'.stream_weights.{stream_index}.' in name
            }
            return True
        else:
            # No improvement
            stream_state['patience_counter'] += 1
            return False

    def _freeze_stream(self, stream_state: dict, stream_index: int,
                      epoch: int, monitor: str, verbose: bool) -> None:
        """
        Freeze a stream by restoring its best weights and setting requires_grad=False.

        Args:
            stream_index: Index of the stream to freeze (0, 1, 2, ...)
        """
        # Restore best weights before freezing
        if stream_state['best_weights'] is not None:
            for name, param in self.named_parameters():
                # Restore stream_weights and stream_biases (but not integration_from_streams)
                if (f'.stream_weights.{stream_index}' in name or f'.stream_biases.{stream_index}' in name) \
                   and name in stream_state['best_weights']:
                    param.data.copy_(stream_state['best_weights'][name].to(param.device))

        # Freeze stream
        stream_state['frozen'] = True
        stream_state['freeze_epoch'] = epoch

        # Set requires_grad = False for stream parameters
        # Integration weights remain trainable to allow rebalancing
        for name, param in self.named_parameters():
            if f'.stream_weights.{stream_index}' in name or f'.stream_biases.{stream_index}' in name:
                param.requires_grad = False

        if verbose:
            metric_str = f"{monitor}: {stream_state['best_metric']:.4f}"
            print(f"❄️  Stream{stream_index} frozen (no improvement for {stream_state['patience']} epochs, "
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
                for i, stream_state in enumerate(stream_early_stopping_state['streams']):
                    if stream_state['frozen']:
                        frozen_streams.append(f"Stream{i} at epoch {stream_state['best_epoch'] + 1}")

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

        # Check each stream
        for i, stream_state in enumerate(stream_early_stopping_state['streams']):
            if not stream_state['frozen']:
                # Get current metric for this stream
                if monitor == 'val_loss':
                    stream_metric = stream_stats.get(f'stream_{i}_val_loss', float('inf'))
                else:
                    stream_metric = stream_stats.get(f'stream_{i}_val_acc', 0.0)

                # Check for improvement
                improved = self._check_stream_improvement(stream_state, stream_metric,
                                                          monitor, min_delta, epoch, i)

                # Freeze if patience exhausted
                if not improved and stream_state['patience_counter'] > stream_state['patience']:
                    self._freeze_stream(stream_state, i, epoch, monitor, verbose)

        # Check if all streams are frozen
        all_frozen = all(stream_state['frozen'] for stream_state in stream_early_stopping_state['streams'])

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
        stream_patience: Union[int, list[int]] = 10,  # Patience per stream (int for all, or list[int] per stream)
        stream_min_delta: float = 0.001,  # Minimum improvement for stream early stopping
        # Modality dropout parameters
        modality_dropout: bool = False,  # Enable per-sample modality dropout
        modality_dropout_start: int = 0,  # Epoch to start modality dropout
        modality_dropout_ramp: int = 20,  # Epochs to ramp dropout from 0 to final rate
        modality_dropout_rate: float = 0.2,  # Final dropout rate (prob of blanking ANY stream per sample)
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
                            Auxiliary classifiers (fc_streams[i] for i in 0..N-1) learn to classify stream features
                            but are gradient-isolated (use .detach()) so they DON'T affect stream weight training.
                            This provides accurate stream accuracy metrics without changing main training dynamics.
            stream_early_stopping: Enable stream-specific early stopping. When a stream plateaus, its
                                 parameters (.stream_weights.{i}.) are frozen while integration
                                 weights remain trainable, allowing the model to continue learning from
                                 other streams. Training stops when all streams are frozen. Uses same monitor
                                 metric as main early stopping.
            stream_patience: Patience for each stream before freezing. Can be:
                           - int: same patience for all streams
                           - list[int]: individual patience per stream
            stream_min_delta: Minimum improvement for stream metrics to count as progress
            modality_dropout: Enable per-sample modality dropout. Each sample can have one
                            stream blanked (zeroed out) to train the model to handle missing streams.
            modality_dropout_start: Epoch to start modality dropout (default: 0)
            modality_dropout_ramp: Number of epochs to ramp dropout from 0% to final rate (default: 20)
            modality_dropout_rate: Final dropout probability (per-sample probability of blanking ANY stream)

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
            stream_early_stopping, monitor, stream_patience, stream_min_delta, verbose
        )

        # Warn if stream_early_stopping enabled without stream_monitoring
        if stream_early_stopping and not stream_monitoring and verbose:
            print("⚠️  Warning: stream_early_stopping=True requires stream_monitoring=True to work")
            print("   Stream early stopping will be disabled until stream_monitoring is enabled")

        # Create separate optimizer for auxiliary classifiers (if stream monitoring enabled)
        # This ensures auxiliary training doesn't affect main model's optimizer state
        aux_optimizer = None
        if stream_monitoring:
            # Collect all auxiliary classifier parameters from fc_streams ModuleList
            aux_params = []
            for fc_stream in self.fc_streams:
                aux_params.extend([fc_stream.weight, fc_stream.bias])

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
            # Stream freezing events (populated if stream_early_stopping=True)
            'streams_frozen': []  # List of (epoch, stream_index) tuples
        }

        # Add stream-specific metrics dynamically (if stream_monitoring=True)
        if stream_monitoring:
            for i in range(self.num_streams):
                history[f'stream_{i}_train_acc'] = []
                history[f'stream_{i}_val_acc'] = []
                history[f'stream_{i}_lr'] = []
                history[f'stream_{i}_frozen_epoch'] = None

        # Add modality dropout tracking (if enabled)
        if modality_dropout:
            history['modality_dropout_prob'] = []

        # Track when modality dropout first becomes active (for logging)
        modality_dropout_started = False

        # Scheduler is already set by compile(), no need to create it here

        for epoch in range(epochs):
            # Compute modality dropout probability for this epoch
            modality_dropout_prob = 0.0
            if modality_dropout:
                modality_dropout_prob = get_modality_dropout_prob(
                    epoch=epoch,
                    start_epoch=modality_dropout_start,
                    ramp_epochs=modality_dropout_ramp,
                    final_rate=modality_dropout_rate
                )
                history['modality_dropout_prob'].append(modality_dropout_prob)

                # Log when modality dropout first becomes active
                if modality_dropout_prob > 0 and not modality_dropout_started:
                    modality_dropout_started = True
                    if verbose:
                        print(f"\n🎲 Modality dropout activated at epoch {epoch} (prob={modality_dropout_prob:.1%})")
                        print(f"   Schedule: ramp over {modality_dropout_ramp} epochs to {modality_dropout_rate:.0%}")

                # Log dropout probability periodically (every 10 epochs during ramp)
                elif modality_dropout_prob > 0 and epoch % 10 == 0 and verbose:
                    ramp_end_epoch = modality_dropout_start + modality_dropout_ramp
                    if epoch < ramp_end_epoch:
                        print(f"🎲 Modality dropout: {modality_dropout_prob:.1%} (ramping)")
                    elif epoch == ramp_end_epoch:
                        print(f"🎲 Modality dropout: {modality_dropout_prob:.1%} (reached final rate)")

            # Calculate total steps for this epoch (training + validation)
            total_steps = len(train_loader)
            if val_loader:
                total_steps += len(val_loader)

            # Create progress bar for the entire epoch
            pbar = create_progress_bar(verbose, epoch, epochs, total_steps)

            # Training phase - use helper method
            avg_train_loss, train_accuracy, stream_train_accs = self._train_epoch(
                train_loader, history, pbar, gradient_accumulation_steps, grad_clip_norm, clear_cache_per_epoch,
                stream_monitoring=stream_monitoring, aux_optimizer=aux_optimizer,
                modality_dropout_prob=modality_dropout_prob
            )

            # Validation phase
            val_loss = 0.0
            val_acc = 0.0
            stream_val_accs = [0.0] * self.num_streams
            stream_val_losses = [0.0] * self.num_streams

            if val_loader:
                val_loss, val_acc, stream_val_accs, stream_val_losses = self._validate(
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
                # Always save stream accuracies to history (computed during training/validation) - N streams
                for i in range(self.num_streams):
                    history[f'stream_{i}_train_acc'].append(stream_train_accs[i])
                    history[f'stream_{i}_val_acc'].append(stream_val_accs[i])

                # Print detailed metrics only if using stream-specific parameter groups
                if len(self.optimizer.param_groups) >= self.num_streams + 1:
                    stream_stats = self._print_stream_monitoring(
                        stream_train_accs=stream_train_accs,
                        stream_val_accs=stream_val_accs,
                        stream_val_losses=stream_val_losses
                    )
                    # Save learning rates (only available with stream-specific param groups)
                    if stream_stats:
                        for i in range(self.num_streams):
                            history[f'stream_{i}_lr'].append(stream_stats[f'stream_{i}_lr'])
                else:
                    # Create minimal stream_stats for compatibility
                    stream_stats = {}
                    for i in range(self.num_streams):
                        stream_stats[f'stream_{i}_train_acc'] = stream_train_accs[i]
                        stream_stats[f'stream_{i}_val_acc'] = stream_val_accs[i]
                        stream_stats[f'stream_{i}_val_loss'] = stream_val_losses[i]

            # Stream-specific early stopping (freeze streams when they plateau)
            if stream_early_stopping_state['enabled'] and stream_stats:
                # Track previous frozen states for all N streams
                prev_frozen_states = [
                    stream_early_stopping_state['streams'][i]['frozen']
                    for i in range(self.num_streams)
                ]

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

                # Record freezing events and update checkpoints for each stream
                for i in range(self.num_streams):
                    if not prev_frozen_states[i] and stream_early_stopping_state['streams'][i]['frozen']:
                        history[f'stream_{i}_frozen_epoch'] = epoch + 1
                        history['streams_frozen'].append((epoch + 1, f'Stream_{i}'))

                        # Update main early stopping checkpoint with frozen stream weights
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
                'all_frozen': stream_early_stopping_state['all_frozen']
            }

            # Add per-stream data dynamically
            for i in range(self.num_streams):
                stream_key = f'stream_{i}'
                history['stream_early_stopping'][f'{stream_key}_frozen'] = stream_early_stopping_state['streams'][i]['frozen']
                history['stream_early_stopping'][f'{stream_key}_frozen_epoch'] = history[f'{stream_key}_frozen_epoch']
                history['stream_early_stopping'][f'{stream_key}_best_metric'] = stream_early_stopping_state['streams'][i]['best_metric']

            if verbose:
                print("\n❄️  Stream Early Stopping Summary:")
                print(f"   Monitor: {monitor}")
                for i in range(self.num_streams):
                    stream_key = f'stream_{i}'
                    if stream_early_stopping_state['streams'][i]['frozen']:
                        print(f"   Stream_{i}: Frozen at epoch {history[f'{stream_key}_frozen_epoch']} "
                              f"(best {monitor}: {stream_early_stopping_state['streams'][i]['best_metric']:.4f})")
                    else:
                        print(f"   Stream_{i}: Not frozen (final {monitor}: {stream_early_stopping_state['streams'][i]['best_metric']:.4f})")

        return history

    def evaluate(
        self,
        data_loader: DataLoader,
        stream_monitoring: bool = True,
        blanked_streams: Optional[set[int]] = None
    ) -> dict[str, float]:
        """
        Evaluate the model on the given data.

        DataLoader format: Tuple of (stream1, stream2, ..., streamN, labels)
        Each stream is a tensor of shape [B, C, H, W]

        Args:
            data_loader: DataLoader containing N-stream input data and targets
            stream_monitoring: Whether to calculate stream-specific metrics (default: True)
            blanked_streams: Optional set of stream indices to blank for ALL samples.
                           Use this to test single-stream robustness:
                           - {0}: blank stream 0 (e.g., RGB), test with stream 1 only
                           - {1}: blank stream 1 (e.g., Depth), test with stream 0 only

        Returns:
            Dictionary containing evaluation metrics (e.g., accuracy, loss, stream accuracies)
        """
        if self.criterion is None:
            raise ValueError("Model not compiled. Call compile() before evaluate().")

        loss, accuracy, stream_val_accs, stream_val_losses = self._validate(
            data_loader, stream_monitoring=stream_monitoring, blanked_streams=blanked_streams
        )

        # Build result dictionary with N streams
        result = {
            'loss': loss,
            'accuracy': accuracy,
        }

        # Add stream-specific metrics
        for i in range(self.num_streams):
            result[f'stream{i}_accuracy'] = stream_val_accs[i]
            result[f'stream{i}_loss'] = stream_val_losses[i]

        return result
    def predict(self, data_loader: DataLoader) -> np.ndarray:
        """
        Generate predictions for the input data.

        DataLoader format: Tuple of (stream1, stream2, ..., streamN, labels)
        Each stream is a tensor of shape [B, C, H, W]

        Args:
            data_loader: DataLoader containing N-stream input data

        Returns:
            Predicted class labels as numpy array
        """
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.to(self.device)

        self.eval()
        all_predictions = []

        with torch.no_grad():
            for batch_data in data_loader:
                # Unpack N streams + labels from tuple format
                # DataLoader returns: (stream1, stream2, ..., streamN, labels)
                *stream_batches, _ = batch_data

                # GPU optimization: non-blocking transfer for N-stream data
                stream_batches = [stream.to(self.device, non_blocking=True) for stream in stream_batches]

                outputs = self(stream_batches)
                _, predictions = torch.max(outputs, 1)
                all_predictions.append(predictions.cpu().numpy())

        return np.concatenate(all_predictions, axis=0)
    
    def predict_proba(self, data_loader: DataLoader) -> np.ndarray:
        """
        Generate probability predictions for the input data.

        DataLoader format: Tuple of (stream1, stream2, ..., streamN, labels)
        Each stream is a tensor of shape [B, C, H, W]

        Args:
            data_loader: DataLoader containing N-stream input data

        Returns:
            Predicted probabilities as numpy array [num_samples, num_classes]
        """
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.to(self.device)

        self.eval()
        all_probabilities = []

        with torch.no_grad():
            for batch_data in data_loader:
                # Unpack N streams + labels from tuple format
                # DataLoader returns: (stream1, stream2, ..., streamN, labels)
                *stream_batches, _ = batch_data

                # GPU optimization: non-blocking transfer for N-stream data
                stream_batches = [stream.to(self.device, non_blocking=True) for stream in stream_batches]

                outputs = self(stream_batches)
                # Apply softmax to get probabilities
                probabilities = torch.softmax(outputs, dim=1)
                all_probabilities.append(probabilities.cpu().numpy())

        return np.concatenate(all_probabilities, axis=0)
    
    
    def _train_epoch(self, train_loader: DataLoader, history: dict, pbar: Optional['TqdmType'] = None,
                     gradient_accumulation_steps: int = 1, grad_clip_norm: Optional[float] = None,
                     clear_cache_per_epoch: bool = False, stream_monitoring: bool = False,
                     aux_optimizer: Optional[torch.optim.Optimizer] = None,
                     modality_dropout_prob: float = 0.0) -> tuple:
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
            modality_dropout_prob: Per-sample probability of blanking a stream (0.0 = disabled)

        Returns:
            Tuple of (average_train_loss, train_accuracy, stream_accs: list[float])
            If stream_monitoring=False, stream_accs will be list of 0.0 values
        """
        self.train()
        train_loss = 0.0
        train_batches = 0
        train_correct = 0
        train_total = 0

        # Stream-specific tracking (if monitoring enabled) - N streams
        # Per-stream active sample counters (accounts for modality dropout)
        stream_train_correct = [0] * self.num_streams
        stream_train_active = [0] * self.num_streams

        # Modality dropout statistics tracking (epoch-level accumulators)
        dropout_stats = {
            'total_samples': 0,
            'total_blanked': 0,
            'blanked_per_stream': [0] * self.num_streams
        } if modality_dropout_prob > 0 else None

        # Ensure gradient_accumulation_steps is at least 1 to avoid division by zero
        gradient_accumulation_steps = max(1, gradient_accumulation_steps)

        # OPTIMIZATION 1: Progress bar update frequency - major performance improvement
        update_frequency = max(1, len(train_loader) // 50)  # Update only 50 times per epoch

        for batch_idx, batch_data in enumerate(train_loader):
            # Unpack N streams + targets from tuple format
            # DataLoader returns: (stream1, stream2, ..., streamN, labels)
            *stream_batches, targets = batch_data

            # GPU optimization: non-blocking transfer for all streams
            stream_batches = [batch.to(self.device, non_blocking=True) for batch in stream_batches]
            targets = targets.to(self.device, non_blocking=True)

            # Generate per-sample blanked mask for modality dropout
            blanked_mask = None
            if modality_dropout_prob > 0:
                blanked_mask = generate_per_sample_blanked_mask(
                    batch_size=targets.shape[0],
                    num_streams=self.num_streams,
                    dropout_prob=modality_dropout_prob,
                    device=self.device
                )

                # Accumulate mask statistics for this batch
                batch_size = targets.shape[0]
                dropout_stats['total_samples'] += batch_size
                if blanked_mask is not None:
                    for i in range(self.num_streams):
                        blanked_count = blanked_mask[i].sum().item()
                        dropout_stats['blanked_per_stream'][i] += blanked_count
                        dropout_stats['total_blanked'] += blanked_count

                # Periodic logging (same frequency as progress bar updates)
                if pbar is not None and batch_idx % update_frequency == 0 and batch_idx > 0:
                    pct_blanked = 100 * dropout_stats['total_blanked'] / max(dropout_stats['total_samples'], 1)
                    stream_pcts = [100 * dropout_stats['blanked_per_stream'][i] / max(dropout_stats['total_samples'], 1)
                                   for i in range(self.num_streams)]
                    stream_str = ", ".join([f"s{i}:{p:.1f}%" for i, p in enumerate(stream_pcts)])
                    # Use carriage return to update in place (like tqdm)
                    print(f"\r  🎲 Dropout: {pct_blanked:.1f}% blanked ({stream_str})", end="", flush=True)

            # OPTIMIZATION 5: Gradient accumulation - zero gradients only when starting accumulation
            if batch_idx % gradient_accumulation_steps == 0:
                self.optimizer.zero_grad()

            if self.use_amp:
                # Use automatic mixed precision
                with autocast(device_type=self.device.type):
                    outputs = self(stream_batches, blanked_mask=blanked_mask)
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
                outputs = self(stream_batches, blanked_mask=blanked_mask)
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
                # === STREAM MONITORING: Pure observation without side effects ===
                # CRITICAL: Use eval mode during stream pathway forwards to prevent
                # any modification to BN running stats. Monitoring should be pure
                # observation - it must not affect the main model's training dynamics.
                was_training = self.training
                self.eval()  # Prevent BN stats updates during monitoring

                # Forward through stream pathways and compute auxiliary losses
                # Note: When modality dropout is active, we skip blanked samples for each stream
                aux_losses = []
                for i in range(self.num_streams):
                    # Get mask for this stream (if dropout active)
                    stream_blanked = blanked_mask.get(i) if blanked_mask else None

                    # Forward through this stream's pathway (no BN stats updates in eval mode)
                    stream_features = self._forward_stream_pathway(i, stream_batches[i])

                    # DETACH features - stops gradient flow to stream weights!
                    stream_features_detached = stream_features.detach()

                    # Classify with auxiliary classifier (only it gets gradients)
                    stream_outputs = self.fc_streams[i](stream_features_detached)

                    # Compute auxiliary loss ONLY for non-blanked samples
                    if stream_blanked is not None and stream_blanked.any():
                        # Get indices of active (non-blanked) samples
                        active_idx = (~stream_blanked).nonzero(as_tuple=True)[0]
                        if len(active_idx) > 0:
                            aux_loss = self.criterion(stream_outputs[active_idx], targets[active_idx])
                            aux_losses.append(aux_loss)
                        # If all samples blanked for this stream, skip this stream's aux loss
                    else:
                        # No blanking - use all samples
                        aux_loss = self.criterion(stream_outputs, targets)
                        aux_losses.append(aux_loss)

                # Restore training mode for auxiliary classifier backward pass
                # (fc_streams don't have BN, so this is safe)
                self.train(was_training)

                # Backward pass for auxiliary classifiers only (detached features ensure no gradient to streams)
                # Each stream's classifier learns independently with full gradients
                # CRITICAL: Use separate aux_optimizer to avoid affecting main optimizer's internal state
                if aux_losses:  # Only if we have any losses to backprop
                    if self.use_amp:
                        for aux_loss in aux_losses:
                            self.scaler.scale(aux_loss).backward()
                        # Update auxiliary classifiers using separate optimizer
                        self.scaler.step(aux_optimizer)
                        self.scaler.update()
                    else:
                        for aux_loss in aux_losses:
                            aux_loss.backward()
                        # Update auxiliary classifiers using separate optimizer
                        aux_optimizer.step()

                    # Zero gradients for auxiliary optimizer
                    aux_optimizer.zero_grad()

                # === ACCURACY MEASUREMENT PHASE: Calculate stream accuracies ===
                # Only count non-blanked samples for each stream's accuracy
                with torch.no_grad():
                    self.eval()  # Disable dropout, use BN running stats

                    for i in range(self.num_streams):
                        stream_features = self._forward_stream_pathway(i, stream_batches[i])
                        stream_outputs = self.fc_streams[i](stream_features)
                        stream_pred = stream_outputs.argmax(1)

                        # Only count non-blanked samples for accuracy
                        stream_blanked = blanked_mask.get(i) if blanked_mask else None
                        if stream_blanked is not None and stream_blanked.any():
                            active_idx = (~stream_blanked).nonzero(as_tuple=True)[0]
                            if len(active_idx) > 0:
                                stream_train_correct[i] += (stream_pred[active_idx] == targets[active_idx]).sum().item()
                                stream_train_active[i] += len(active_idx)
                        else:
                            stream_train_correct[i] += (stream_pred == targets).sum().item()
                            stream_train_active[i] += targets.size(0)

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
                
                # Add lr at the end (scientific notation for small LRs)
                postfix['lr'] = f'{current_lr:.2e}'

                pbar.set_postfix(postfix)

                # Simplified progress bar update - tqdm handles remaining steps automatically
                updates_needed = min(update_frequency, len(train_loader) - pbar.n)
                if updates_needed > 0:
                    pbar.update(updates_needed)
        
        avg_train_loss = train_loss / train_batches
        train_accuracy = train_correct / train_total

        # Calculate stream-specific accuracies (N streams)
        # Use per-stream active sample counts (accounts for modality dropout)
        stream_accs = [
            stream_train_correct[i] / max(stream_train_active[i], 1) if stream_monitoring else 0.0
            for i in range(self.num_streams)
        ]

        # End-of-epoch modality dropout summary
        if dropout_stats is not None and dropout_stats['total_samples'] > 0:
            total_samples = dropout_stats['total_samples']
            total_blanked = dropout_stats['total_blanked']
            pct_blanked = 100 * total_blanked / total_samples
            stream_pcts = [100 * dropout_stats['blanked_per_stream'][i] / total_samples
                          for i in range(self.num_streams)]
            stream_str = ", ".join([f"stream{i}: {p:.1f}%" for i, p in enumerate(stream_pcts)])

            # Clear the periodic update line and print summary
            print(f"\r  🎲 Epoch dropout summary: {total_blanked}/{total_samples} samples blanked ({pct_blanked:.1f}%)")
            print(f"     Per-stream: {stream_str}")

            # Store in history for analysis
            if '_dropout_epoch_stats' not in history:
                history['_dropout_epoch_stats'] = []
            history['_dropout_epoch_stats'].append({
                'total_samples': total_samples,
                'total_blanked': total_blanked,
                'pct_blanked': pct_blanked,
                'blanked_per_stream': dropout_stats['blanked_per_stream'].copy(),
                'pct_per_stream': stream_pcts
            })

        # Optional: Clear CUDA cache at end of epoch (only if experiencing OOM issues)
        # if clear_cache_per_epoch and self.device.type == 'cuda':
        #     torch.cuda.empty_cache()

        return avg_train_loss, train_accuracy, stream_accs
    
    def _validate(self, data_loader: DataLoader,
                  pbar: Optional['TqdmType'] = None, stream_monitoring: bool = True,
                  blanked_streams: Optional[set[int]] = None) -> tuple:
        """
        Validate the model on the given data with GPU optimizations and reduced progress updates.

        Args:
            data_loader: DataLoader containing N-stream input data and targets
            pbar: Optional progress bar to update during validation
            stream_monitoring: Whether to track stream-specific metrics during validation
            blanked_streams: Optional set of stream indices to blank for ALL samples.
                           Used for single-stream robustness evaluation.

        Returns:
            Tuple of (loss, accuracy, stream_val_accs: list[float], stream_val_losses: list[float])
            If stream_monitoring=False, stream accuracies and losses will be lists of 0.0
        """

        self.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        # Stream-specific tracking (if monitoring enabled) - N streams
        stream_val_correct = [0] * self.num_streams
        stream_val_loss = [0.0] * self.num_streams
        stream_val_total = 0

        # OPTIMIZATION 1: Progress bar update frequency for validation - major performance improvement
        update_frequency = max(1, len(data_loader) // 25)  # Update only 25 times during validation

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(data_loader):
                # Unpack N streams + targets from tuple format
                # DataLoader returns: (stream1, stream2, ..., streamN, labels)
                *stream_batches, targets = batch_data

                # GPU optimization: non-blocking transfer for all streams
                stream_batches = [batch.to(self.device, non_blocking=True) for batch in stream_batches]
                targets = targets.to(self.device, non_blocking=True)

                # Convert blanked_streams set to blanked_mask for entire batch
                blanked_mask = None
                if blanked_streams:
                    batch_size = stream_batches[0].shape[0]
                    blanked_mask = {
                        i: torch.ones(batch_size, dtype=torch.bool, device=self.device)
                        if i in blanked_streams else
                        torch.zeros(batch_size, dtype=torch.bool, device=self.device)
                        for i in range(self.num_streams)
                    }

                # Use AMP for validation if enabled
                if self.use_amp:
                    with autocast(device_type=self.device.type):
                        outputs = self(stream_batches, blanked_mask=blanked_mask)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self(stream_batches, blanked_mask=blanked_mask)
                    loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

                # Stream-specific monitoring with auxiliary classifiers
                # Model is already in eval() mode, so dropout is disabled
                if stream_monitoring:
                    # Forward through stream pathways and compute metrics for all streams
                    for i in range(self.num_streams):
                        stream_features = self._forward_stream_pathway(i, stream_batches[i])

                        # Classify with auxiliary classifier
                        stream_outputs = self.fc_streams[i](stream_features)

                        # Calculate stream loss
                        loss_i = self.criterion(stream_outputs, targets)
                        stream_val_loss[i] += loss_i.item() * targets.size(0)

                        # Calculate stream accuracy
                        stream_pred = stream_outputs.argmax(1)
                        stream_val_correct[i] += (stream_pred == targets).sum().item()

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

        # Calculate stream-specific accuracies and losses (N streams)
        stream_val_accs = [
            stream_val_correct[i] / max(stream_val_total, 1) if stream_monitoring else 0.0
            for i in range(self.num_streams)
        ]
        avg_stream_val_losses = [
            stream_val_loss[i] / max(stream_val_total, 1) if stream_monitoring else 0.0
            for i in range(self.num_streams)
        ]

        return avg_loss, accuracy, stream_val_accs, avg_stream_val_losses

    def _print_stream_monitoring(self, stream_train_accs: list[float], stream_val_accs: list[float],
                                 stream_val_losses: list[float]) -> dict:
        """
        Print stream-specific monitoring metrics (computed during main training loop).

        Args:
            stream_train_accs: List of training accuracies for each stream
            stream_val_accs: List of validation accuracies for each stream
            stream_val_losses: List of validation losses for each stream

        Returns:
            Dictionary containing stream-specific metrics for history tracking
        """
        # Get parameter groups info
        param_groups = self.optimizer.param_groups

        # Build monitoring output for N input streams
        # If using stream-specific parameter groups, we have N stream groups + 1 shared group
        if len(param_groups) >= self.num_streams + 1:
            # Build output strings for each stream
            stream_strs = []
            stream_stats = {}

            for i in range(self.num_streams):
                stream_lr = param_groups[i]['lr']
                stream_wd = param_groups[i]['weight_decay']

                stream_str = (f"Stream_{i}: "
                             f"T_acc:{stream_train_accs[i]:.4f}, V_acc:{stream_val_accs[i]:.4f}, "
                             f"LR:{stream_lr:.2e}")
                stream_strs.append(stream_str)

                # Add to stats dict
                stream_stats[f'stream_{i}_train_acc'] = stream_train_accs[i]
                stream_stats[f'stream_{i}_val_acc'] = stream_val_accs[i]
                stream_stats[f'stream_{i}_val_loss'] = stream_val_losses[i]
                stream_stats[f'stream_{i}_lr'] = stream_lr
                stream_stats[f'stream_{i}_wd'] = stream_wd

            # Print all streams on one line, separated by " | "
            print("  " + " | ".join(stream_strs))

            return stream_stats

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
    
    def _forward_stream_pathway(self, stream_idx: int, stream_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through a single stream pathway.

        Uses dedicated forward_stream() methods on each layer to process ONLY the
        specified stream without corrupting BN running stats for other streams.

        Args:
            stream_idx: Index of the stream to forward through (0-indexed)
            stream_input: The stream input tensor [batch_size, channels, height, width]

        Returns:
            The stream pathway output tensor (flattened features) [batch_size, feature_dim]
        """
        # Use forward_stream() methods that process ONLY this stream
        # This avoids corrupting BN running stats for other streams
        stream_x = self.conv1.forward_stream(stream_idx, stream_input)
        stream_x = self.bn1.forward_stream(stream_idx, stream_x)
        stream_x = self.relu.forward_stream(stream_idx, stream_x)
        stream_x = self.maxpool.forward_stream(stream_idx, stream_x)

        # ResNet layers (each layer has forward_stream that processes all blocks)
        stream_x = self.layer1.forward_stream(stream_idx, stream_x)
        stream_x = self.layer2.forward_stream(stream_idx, stream_x)
        stream_x = self.layer3.forward_stream(stream_idx, stream_x)
        stream_x = self.layer4.forward_stream(stream_idx, stream_x)

        # Global average pooling (standard nn.AdaptiveAvgPool2d)
        stream_x = self.avgpool(stream_x)

        # Flatten
        stream_x = torch.flatten(stream_x, 1)

        return stream_x

    def _forward_integrated_pathway(self, stream_inputs: list[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the integrated pathway only (without individual streams).

        This extracts only the integrated stream features by running the full forward pass
        and returning only the integrated stream output.

        Args:
            stream_inputs: List of input tensors, one per stream (needed for integration).

        Returns:
            The integrated pathway output tensor (flattened features).
        """
        # Initial convolution (creates integrated stream from N streams)
        stream_outputs, integrated = self.conv1(stream_inputs, None)
        stream_outputs, integrated = self.bn1(stream_outputs, integrated)
        stream_outputs, integrated = self.relu(stream_outputs, integrated)

        # Max pooling (all N streams + integrated)
        stream_outputs, integrated = self.maxpool(stream_outputs, integrated)

        # ResNet layers (integration happens inside LIConv2d neurons)
        stream_outputs, integrated = self.layer1(stream_outputs, integrated)
        stream_outputs, integrated = self.layer2(stream_outputs, integrated)
        stream_outputs, integrated = self.layer3(stream_outputs, integrated)
        stream_outputs, integrated = self.layer4(stream_outputs, integrated)

        # Global average pooling for integrated stream
        integrated_pooled = self.avgpool(integrated)

        # Flatten integrated features
        integrated_features = torch.flatten(integrated_pooled, 1)

        return integrated_features

    def analyze_pathways(self, data_loader: DataLoader) -> dict:
        """
        Analyze the contribution and performance of individual pathways.

        Uses auxiliary classifiers (fc_streams) for per-stream accuracy.
        Uses main classifier (fc) for integrated pathway and full model accuracy.

        Args:
            data_loader: DataLoader containing N-stream input data

        Returns:
            Dictionary containing pathway analysis results for all N streams + integrated:
            - accuracy: stream{i}_only for i in 0..N-1, integrated_only, full_model
            - loss: for each pathway
            - feature_norms: mean/std for each pathway's feature activations
        """
        if self.criterion is None:
            raise ValueError("Model not compiled. Call compile() before analyze_pathways().")

        self.eval()

        # Initialize metrics for N streams + integrated + full model
        full_model_correct = 0
        stream_only_correct = [0] * self.num_streams
        integrated_only_correct = 0
        total_samples = 0

        full_model_loss = 0.0
        stream_only_losses = [0.0] * self.num_streams
        integrated_only_loss = 0.0

        stream_feature_norms = [[] for _ in range(self.num_streams)]
        integrated_feature_norms = []

        with torch.no_grad():
            for batch_data in data_loader:
                # Unpack N streams + targets from tuple format
                # DataLoader returns: (stream1, stream2, ..., streamN, labels)
                *stream_batches, targets_batch = batch_data

                # Move to device
                stream_batches = [stream.to(self.device, non_blocking=True) for stream in stream_batches]
                targets_batch = targets_batch.to(self.device, non_blocking=True)

                batch_size_actual = stream_batches[0].size(0)
                total_samples += batch_size_actual

                # Full model prediction (uses integrated stream)
                full_outputs = self(stream_batches)
                full_loss = self.criterion(full_outputs, targets_batch)
                full_model_loss += full_loss.item() * batch_size_actual

                _, full_predicted = torch.max(full_outputs, 1)
                full_model_correct += (full_predicted == targets_batch).sum().item()

                # Stream pathway analysis - USE AUXILIARY CLASSIFIERS
                for i in range(self.num_streams):
                    stream_features = self._forward_stream_pathway(i, stream_batches[i])
                    stream_feature_norms[i].append(torch.norm(stream_features, dim=1).cpu())
                    stream_outputs = self.fc_streams[i](stream_features)  # Auxiliary classifier!
                    stream_loss = self.criterion(stream_outputs, targets_batch)
                    stream_only_losses[i] += stream_loss.item() * batch_size_actual

                    _, stream_predicted = torch.max(stream_outputs, 1)
                    stream_only_correct[i] += (stream_predicted == targets_batch).sum().item()

                # Integrated pathway only (runs full forward but uses only integrated features)
                integrated_features = self._forward_integrated_pathway(stream_batches)
                integrated_feature_norms.append(torch.norm(integrated_features, dim=1).cpu())
                integrated_outputs = self.fc(integrated_features)
                integrated_loss = self.criterion(integrated_outputs, targets_batch)
                integrated_only_loss += integrated_loss.item() * batch_size_actual

                _, integrated_predicted = torch.max(integrated_outputs, 1)
                integrated_only_correct += (integrated_predicted == targets_batch).sum().item()

        # Calculate metrics
        full_accuracy = full_model_correct / total_samples
        stream_accuracies = [correct / total_samples for correct in stream_only_correct]
        integrated_accuracy = integrated_only_correct / total_samples

        avg_full_loss = full_model_loss / total_samples
        avg_stream_losses = [loss / total_samples for loss in stream_only_losses]
        avg_integrated_loss = integrated_only_loss / total_samples

        # Feature analysis
        stream_norms = [torch.cat(norms, dim=0) for norms in stream_feature_norms]
        integrated_norms = torch.cat(integrated_feature_norms, dim=0)

        # Build accuracy dictionary
        accuracy_dict = {'full_model': full_accuracy}
        for i in range(self.num_streams):
            accuracy_dict[f'stream{i}_only'] = stream_accuracies[i]
            accuracy_dict[f'stream{i}_contribution'] = stream_accuracies[i] / full_accuracy if full_accuracy > 0 else 0
        accuracy_dict['integrated_only'] = integrated_accuracy
        accuracy_dict['integrated_contribution'] = integrated_accuracy / full_accuracy if full_accuracy > 0 else 0

        # Build loss dictionary
        loss_dict = {'full_model': avg_full_loss}
        for i in range(self.num_streams):
            loss_dict[f'stream{i}_only'] = avg_stream_losses[i]
            # Contribution = ratio of stream loss to full model loss (lower is better)
            loss_dict[f'stream{i}_contribution'] = avg_stream_losses[i] / avg_full_loss if avg_full_loss > 0 else 0
        loss_dict['integrated_only'] = avg_integrated_loss
        loss_dict['integrated_contribution'] = avg_integrated_loss / avg_full_loss if avg_full_loss > 0 else 0

        # Build feature norms dictionary
        feature_norms_dict = {}
        for i in range(self.num_streams):
            feature_norms_dict[f'stream{i}_mean'] = stream_norms[i].mean().item()
            feature_norms_dict[f'stream{i}_std'] = stream_norms[i].std().item()
        feature_norms_dict['integrated_mean'] = integrated_norms.mean().item()
        feature_norms_dict['integrated_std'] = integrated_norms.std().item()

        return {
            'accuracy': accuracy_dict,
            'loss': loss_dict,
            'feature_norms': feature_norms_dict,
            'samples_analyzed': total_samples
        }
    
    def analyze_pathway_weights(self) -> dict:
        """
        Analyze the weight distributions and magnitudes across all N stream pathways + integrated pathway.

        Returns:
            Dictionary containing weight analysis for all stream pathways and integrated pathway
        """
        # Initialize dictionaries for N streams + integrated
        stream_weights = [{} for _ in range(self.num_streams)]
        integrated_weights = {}
        integration_weights = {}

        # Analyze LI layers - look for modules with stream_weights and integrated_weight
        for name, module in self.named_modules():
            if hasattr(module, 'stream_weights') and hasattr(module, 'integrated_weight'):
                # LIConv2d modules
                integrated_weight = module.integrated_weight

                # Analyze each stream's weights
                for i in range(self.num_streams):
                    stream_weight = module.stream_weights[i]
                    stream_weights[i][name] = {
                        'mean': stream_weight.mean().item(),
                        'std': stream_weight.std().item(),
                        'norm': torch.norm(stream_weight).item(),
                        'shape': list(stream_weight.shape)
                    }

                integrated_weights[name] = {
                    'mean': integrated_weight.mean().item(),
                    'std': integrated_weight.std().item(),
                    'norm': torch.norm(integrated_weight).item(),
                    'shape': list(integrated_weight.shape)
                }

                # Integration weights (from all streams to integrated)
                if hasattr(module, 'integration_from_streams'):
                    integration_weights[name] = {}
                    for i in range(self.num_streams):
                        int_from_stream = module.integration_from_streams[i]
                        integration_weights[name][f'from_stream{i}'] = {
                            'mean': int_from_stream.mean().item(),
                            'std': int_from_stream.std().item(),
                            'norm': torch.norm(int_from_stream).item(),
                            'shape': list(int_from_stream.shape)
                        }

        # Calculate overall statistics for each stream
        stream_norms = [[w['norm'] for w in stream_weights[i].values()] for i in range(self.num_streams)]
        integrated_norms = [w['norm'] for w in integrated_weights.values()]

        # Build result dictionary
        result = {}

        # Add stream pathway information
        for i in range(self.num_streams):
            result[f'stream{i}_pathway'] = {
                'layer_weights': stream_weights[i],
                'total_norm': sum(stream_norms[i]),
                'mean_norm': sum(stream_norms[i]) / len(stream_norms[i]) if stream_norms[i] else 0,
                'num_layers': len(stream_weights[i])
            }

        # Add integrated pathway information
        result['integrated_pathway'] = {
            'layer_weights': integrated_weights,
            'total_norm': sum(integrated_norms),
            'mean_norm': sum(integrated_norms) / len(integrated_norms) if integrated_norms else 0,
            'num_layers': len(integrated_weights)
        }

        # Add integration weights
        result['integration_weights'] = integration_weights

        # Ratio analysis - compare streams pairwise and integrated to first stream
        ratio_analysis = {}

        # Stream-to-stream ratios (comparing to stream 0 as baseline)
        if stream_norms[0]:
            for i in range(1, self.num_streams):
                ratio_analysis[f'stream{i}_to_stream0_norm_ratio'] = (
                    sum(stream_norms[i]) / sum(stream_norms[0])
                ) if stream_norms[i] else float('inf')

        # Integrated to stream0 ratio
        if stream_norms[0]:
            ratio_analysis['integrated_to_stream0_norm_ratio'] = (
                sum(integrated_norms) / sum(stream_norms[0])
            ) if stream_norms[0] else float('inf')

        # Per-layer ratios
        layer_ratios = {}
        for name in stream_weights[0].keys():
            layer_ratios[name] = {}

            # Compare each stream to stream 0
            for i in range(1, self.num_streams):
                if name in stream_weights[i]:
                    layer_ratios[name][f's{i}_to_s0'] = (
                        stream_weights[i][name]['norm'] / stream_weights[0][name]['norm']
                    ) if stream_weights[0][name]['norm'] > 0 else float('inf')

            # Compare integrated to stream 0
            if name in integrated_weights:
                layer_ratios[name]['int_to_s0'] = (
                    integrated_weights[name]['norm'] / stream_weights[0][name]['norm']
                ) if stream_weights[0][name]['norm'] > 0 else float('inf')

        ratio_analysis['layer_ratios'] = layer_ratios
        result['ratio_analysis'] = ratio_analysis

        return result
    
    def get_pathway_importance(self,
                              data_loader: DataLoader,
                              method: str = 'gradient') -> dict:
        """
        Calculate pathway importance using different methods.

        Args:
            data_loader: DataLoader containing N-stream input data
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

        Note: For LINet, the integrated stream is created FROM the N input streams,
        so we measure input gradients for all N streams.
        The integrated stream's importance is implicit in how gradients flow back
        through the integration weights to the input streams.
        """
        if self.criterion is None:
            raise ValueError("Model not compiled. Call compile() before calculating importance.")

        # Save current training state
        was_training = self.training
        self.train()  # Enable gradients

        stream_gradients = [[] for _ in range(self.num_streams)]

        for batch_data in data_loader:
            # Unpack N streams + targets from tuple format
            # DataLoader returns: (stream1, stream2, ..., streamN, labels)
            *stream_batches, targets_batch = batch_data

            # Move to device
            stream_batches = [stream.to(self.device, non_blocking=True) for stream in stream_batches]
            targets_batch = targets_batch.to(self.device, non_blocking=True)

            # Require gradients for all inputs
            for stream_batch in stream_batches:
                stream_batch.requires_grad_(True)

            # Forward pass (creates integrated stream internally)
            outputs = self(stream_batches)
            loss = self.criterion(outputs, targets_batch)

            # Backward pass
            loss.backward()

            # Calculate gradient norms for each stream - flatten and then compute norm
            for i in range(self.num_streams):
                grad_norm = torch.norm(stream_batches[i].grad.flatten(1), dim=1).mean().item()
                stream_gradients[i].append(grad_norm)

            # Clear gradients
            self.zero_grad()
            for stream_batch in stream_batches:
                stream_batch.grad = None

        # Restore original training state
        self.train(was_training)

        # Calculate average gradients for each stream
        avg_stream_grads = [sum(grads) / len(grads) for grads in stream_gradients]
        total_grad = sum(avg_stream_grads)

        # Build result dictionary
        result = {
            'method': 'gradient',
            'raw_gradients': {}
        }

        # Add importance for each stream
        for i in range(self.num_streams):
            result[f'stream{i}_importance'] = (
                avg_stream_grads[i] / total_grad if total_grad > 0 else 1.0 / self.num_streams
            )
            result['raw_gradients'][f'stream{i}_avg'] = avg_stream_grads[i]

        result['note'] = 'Integrated stream importance is implicit in how gradients flow through integration weights'

        return result

    def calculate_stream_contributions_to_integration(self, data_loader: DataLoader = None):
        """
        Calculate how much each input stream contributes to the integrated stream.

        This method measures architectural contribution by analyzing integration weight magnitudes:
        ||integration_from_streams[i]|| for i in 0..N-1

        This directly answers: "How much does the model architecture favor each stream?"

        Simple, fast, and interpretable:
        - No data required (analyzes learned weights)
        - Stable (doesn't vary with data sampling)
        - Clear meaning: "The model uses X% from stream0, Y% from stream1, etc."

        Args:
            data_loader: Not used, kept for backward compatibility

        Returns:
            Dictionary containing:
            - stream{i}_contribution: float (0-1) - proportion from stream i based on integration weights
            - raw_norms: dict with actual weight norms for each stream
        """
        # Analyze integration weight magnitudes across all LIConv2d layers
        stream_weight_norms = [0.0] * self.num_streams

        for name, param in self.named_parameters():
            # Check if this is an integration weight parameter
            if 'integration_from_streams' in name:
                # Extract stream index from parameter name
                # Parameter names look like: "layer.integration_from_streams.0", "layer.integration_from_streams.1", etc.
                for i in range(self.num_streams):
                    if f'integration_from_streams.{i}' in name:
                        stream_weight_norms[i] += torch.norm(param).item()
                        break

        total_weight_norm = sum(stream_weight_norms)

        # Build result dictionary
        result = {
            'method': 'integration_weights',
            'raw_norms': {'total': total_weight_norm},
            'interpretation': {}
        }

        # Add contribution and interpretation for each stream
        for i in range(self.num_streams):
            result[f'stream{i}_contribution'] = (
                stream_weight_norms[i] / total_weight_norm if total_weight_norm > 0 else 1.0 / self.num_streams
            )
            result['raw_norms'][f'stream{i}_integration_weights'] = stream_weight_norms[i]
            result['interpretation'][f'stream{i}_percentage'] = (
                f"{100 * stream_weight_norms[i] / total_weight_norm:.1f}%" if total_weight_norm > 0
                else f"{100.0 / self.num_streams:.1f}%"
            )

        result['note'] = (
            'Measures architectural contribution based on integration weight magnitudes. '
            f'Reflects how the trained model structurally combines the {self.num_streams} input streams.'
        )

        return result

    def calculate_stream_contributions(self, data_loader: DataLoader):
        """
        DEPRECATED: Calculate hypothetical classification ability of each stream.

        WARNING: This method is MISLEADING for LINet because streams don't directly classify.
        They only feed into the integrated stream through integration weights.

        This method measures: "How well could each stream classify IF it had its own classifier?"
        But in LINet, streams DON'T have classifiers - only the integrated stream does.

        **Use calculate_stream_contributions_to_integration() instead** for meaningful analysis
        of how much each input stream contributes to the integrated stream's features.

        Args:
            data_loader: DataLoader containing N-stream input data

        Returns:
            Dictionary containing hypothetical stream classification abilities (misleading for LINet)
        """
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
        integrated_accuracy = pathway_analysis['accuracy']['integrated_only']

        # Gather stream accuracies
        stream_accuracies = []
        for i in range(self.num_streams):
            stream_accuracies.append(pathway_analysis['accuracy'][f'stream{i}_only'])

        # Calculate contribution scores
        # The integrated stream should perform best (or equal to full model) since that's what we use for prediction
        total_acc = sum(stream_accuracies) + integrated_accuracy

        # Build result dictionary
        result = {
            'method': 'ablation',
            'individual_accuracies': {'full_model': full_accuracy},
            'relative_to_full_model': {}
        }

        # Add importance for each stream
        for i in range(self.num_streams):
            result[f'stream{i}_importance'] = (
                stream_accuracies[i] / total_acc if total_acc > 0 else 1.0 / (self.num_streams + 1)
            )
            result['individual_accuracies'][f'stream{i}_only'] = stream_accuracies[i]
            result['relative_to_full_model'][f'stream{i}_ratio'] = (
                stream_accuracies[i] / full_accuracy if full_accuracy > 0 else 0
            )

        # Add integrated pathway information
        result['integrated_importance'] = (
            integrated_accuracy / total_acc if total_acc > 0 else 1.0 / (self.num_streams + 1)
        )
        result['individual_accuracies']['integrated_only'] = integrated_accuracy
        result['relative_to_full_model']['integrated_ratio'] = (
            integrated_accuracy / full_accuracy if full_accuracy > 0 else 1.0
        )

        result['note'] = f'Integrated stream should perform best as it combines information from all {self.num_streams} input streams'

        return result
    
    def calculate_feature_norm_importance(self, data_loader):
        """Calculate importance based on feature norm magnitudes for all N streams + integrated."""
        # Use the analyze_pathways method for feature analysis
        pathway_analysis = self.analyze_pathways(data_loader)

        # Gather stream norms
        stream_norms = []
        for i in range(self.num_streams):
            stream_norms.append(pathway_analysis['feature_norms'][f'stream{i}_mean'])

        integrated_norm = pathway_analysis['feature_norms']['integrated_mean']
        total_norm = sum(stream_norms) + integrated_norm

        # Build result dictionary
        result = {
            'method': 'feature_norm',
            'feature_norms': {
                'integrated_mean': integrated_norm
            }
        }

        # Add importance for each stream
        for i in range(self.num_streams):
            result[f'stream{i}_importance'] = (
                stream_norms[i] / total_norm if total_norm > 0 else 1.0 / (self.num_streams + 1)
            )
            result['feature_norms'][f'stream{i}_mean'] = stream_norms[i]

        # Add integrated importance
        result['integrated_importance'] = (
            integrated_norm / total_norm if total_norm > 0 else 1.0 / (self.num_streams + 1)
        )

        # Add ratios comparing to stream 0 as baseline
        if stream_norms[0] > 0:
            for i in range(1, self.num_streams):
                result['feature_norms'][f'stream{i}_to_stream0_ratio'] = (
                    stream_norms[i] / stream_norms[0]
                )
            result['feature_norms']['integrated_to_stream0_ratio'] = (
                integrated_norm / stream_norms[0]
            )

        return result 

# Factory functions for common LINet architectures
def li_resnet18(num_classes: int = 1000, stream_input_channels: list[int] = None, **kwargs) -> LINet:
    """
    Create a Linear Integration ResNet-18 model.

    Args:
        num_classes: Number of output classes
        stream_input_channels: List of input channels for each stream.
                              Default: [3, 1] for RGB + Depth (backward compatible)
                              For 3 streams: [3, 1, 1] for RGB + Depth + Orthogonal
        **kwargs: Additional arguments passed to LINet constructor

    Returns:
        LINet model instance

    Examples:
        # 2-stream (backward compatible - default)
        model = li_resnet18(num_classes=15)

        # 3-stream (RGB + Depth + Orthogonal)
        model = li_resnet18(num_classes=15, stream_input_channels=[3, 1, 1])
    """
    # Default to 2-stream for backward compatibility with existing code
    if stream_input_channels is None:
        stream_input_channels = [3, 1]  # RGB + Depth

    return LINet(
        LIBasicBlock,
        [2, 2, 2, 2],
        num_classes=num_classes,
        stream_input_channels=stream_input_channels,
        **kwargs
    )

def li_resnet34(num_classes: int = 1000, stream_input_channels: list[int] = None, **kwargs) -> LINet:
    """
    Create a Linear Integration ResNet-34 model.

    Args:
        num_classes: Number of output classes
        stream_input_channels: List of input channels for each stream.
                              Default: [3, 1] for RGB + Depth (backward compatible)
                              For 3 streams: [3, 1, 1] for RGB + Depth + Orthogonal
        **kwargs: Additional arguments passed to LINet constructor

    Returns:
        LINet model instance
    """
    # Default to 2-stream for backward compatibility with existing code
    if stream_input_channels is None:
        stream_input_channels = [3, 1]  # RGB + Depth

    return LINet(
        LIBasicBlock,
        [3, 4, 6, 3],
        num_classes=num_classes,
        stream_input_channels=stream_input_channels,
        **kwargs
    )


def li_resnet50(num_classes: int = 1000, stream_input_channels: list[int] = None, **kwargs) -> LINet:
    """
    Create a Linear Integration ResNet-50 model.

    Args:
        num_classes: Number of output classes
        stream_input_channels: List of input channels for each stream.
                              Default: [3, 1] for RGB + Depth (backward compatible)
                              For 3 streams: [3, 1, 1] for RGB + Depth + Orthogonal
        **kwargs: Additional arguments passed to LINet constructor

    Returns:
        LINet model instance
    """
    # Default to 2-stream for backward compatibility with existing code
    if stream_input_channels is None:
        stream_input_channels = [3, 1]  # RGB + Depth

    return LINet(
        LIBottleneck,
        [3, 4, 6, 3],
        num_classes=num_classes,
        stream_input_channels=stream_input_channels,
        **kwargs
    )


def li_resnet101(num_classes: int = 1000, stream_input_channels: list[int] = None, **kwargs) -> LINet:
    """
    Create a Linear Integration ResNet-101 model.

    Args:
        num_classes: Number of output classes
        stream_input_channels: List of input channels for each stream.
                              Default: [3, 1] for RGB + Depth (backward compatible)
                              For 3 streams: [3, 1, 1] for RGB + Depth + Orthogonal
        **kwargs: Additional arguments passed to LINet constructor

    Returns:
        LINet model instance
    """
    # Default to 2-stream for backward compatibility with existing code
    if stream_input_channels is None:
        stream_input_channels = [3, 1]  # RGB + Depth

    return LINet(
        LIBottleneck,
        [3, 4, 23, 3],
        num_classes=num_classes,
        stream_input_channels=stream_input_channels,
        **kwargs
    )


def li_resnet152(num_classes: int = 1000, stream_input_channels: list[int] = None, **kwargs) -> LINet:
    """
    Create a Linear Integration ResNet-152 model.

    Args:
        num_classes: Number of output classes
        stream_input_channels: List of input channels for each stream.
                              Default: [3, 1] for RGB + Depth (backward compatible)
                              For 3 streams: [3, 1, 1] for RGB + Depth + Orthogonal
        **kwargs: Additional arguments passed to LINet constructor

    Returns:
        LINet model instance
    """
    # Default to 2-stream for backward compatibility with existing code
    if stream_input_channels is None:
        stream_input_channels = [3, 1]  # RGB + Depth

    return LINet(
        LIBottleneck,
        [3, 8, 36, 3],
        num_classes=num_classes,
        stream_input_channels=stream_input_channels,
        **kwargs
    )