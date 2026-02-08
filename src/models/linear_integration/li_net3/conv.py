"""
Linear Integration convolution layers for N-stream processing.

This module provides N-stream convolution operations with learned integration
that maintain independent pathways with unified neuron architecture.
"""

import math
from click import group
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F, init
from torch.nn.parameter import Parameter
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair, _reverse_repeat_tuple
from typing import Optional, Union


class _LIConvNd(nn.Module):
    """
    Linear Integration convolution base class - follows PyTorch's _ConvNd pattern exactly.

    This is the base class for linear integration convolution layers, mirroring PyTorch's
    _ConvNd but adapted for N-stream processing with dynamic stream handling.
    """

    __constants__ = [
        "stride", "padding", "dilation", "groups", "padding_mode", "output_padding",
        "num_streams", "stream_in_channels", "stream_out_channels",
        "integrated_in_channels", "integrated_out_channels", "kernel_size",
    ]
    __annotations__ = {
        "stream_biases": Optional[nn.ParameterList],
        "integrated_bias": Optional[torch.Tensor],
    }
    
    def _conv_forward(
        self,
        stream_inputs: list[Tensor],
        integrated_input: Optional[Tensor],
        stream_weights: list[Tensor],
        integrated_weight: Optional[Tensor],
        integration_from_streams_weights: list[Tensor],
        stream_biases: list[Optional[Tensor]],
        integrated_bias: Optional[Tensor]
    ) -> tuple[list[Tensor], Tensor]:
        """Abstract method to be implemented by subclasses."""
        ...

    num_streams: int
    stream_in_channels: list[int]
    stream_out_channels: list[int]
    integrated_in_channels: int
    integrated_out_channels: int
    kernel_size: tuple[int, ...]
    stride: tuple[int, ...]
    padding: Union[str, tuple[int, ...]]
    dilation: tuple[int, ...]
    transposed: bool
    output_padding: tuple[int, ...]
    groups: int
    padding_mode: str
    stream_weights: nn.ParameterList
    integrated_weight: Tensor
    integration_from_streams: nn.ParameterList
    stream_biases: Optional[nn.ParameterList]
    integrated_bias: Optional[Tensor]
    _reversed_padding_repeated_twice: list[int]

    def __init__(
        self,
        stream_in_channels: list[int],
        stream_out_channels: list[int],
        integrated_in_channels: int,
        integrated_out_channels: int,
        kernel_size: tuple[int, ...],
        stride: tuple[int, ...],
        padding: Union[str, tuple[int, ...]],
        dilation: tuple[int, ...],
        transposed: bool,
        output_padding: tuple[int, ...],
        groups: int,
        bias: bool,
        padding_mode: str,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        # Validate parameters exactly like _ConvNd
        if groups <= 0:
            raise ValueError("groups must be a positive integer")

        # Validate stream channel counts
        num_streams = len(stream_in_channels)
        if len(stream_out_channels) != num_streams:
            raise ValueError(f"stream_in_channels and stream_out_channels must have same length, "
                           f"got {num_streams} and {len(stream_out_channels)}")

        for i, in_ch in enumerate(stream_in_channels):
            if in_ch % groups != 0:
                raise ValueError(f"stream_in_channels[{i}] must be divisible by groups")

        for i, out_ch in enumerate(stream_out_channels):
            if out_ch % groups != 0:
                raise ValueError(f"stream_out_channels[{i}] must be divisible by groups")

        # Allow integrated_in_channels=0 for first layer (no integrated input yet)
        if integrated_in_channels > 0 and integrated_in_channels % groups != 0:
            raise ValueError("integrated_in_channels must be divisible by groups")
        if integrated_out_channels % groups != 0:
            raise ValueError("integrated_out_channels must be divisible by groups")

        valid_padding_strings = {"same", "valid"}
        if isinstance(padding, str):
            if padding not in valid_padding_strings:
                raise ValueError(
                    f"Invalid padding string {padding!r}, should be one of {valid_padding_strings}"
                )
            if padding == "same" and any(s != 1 for s in stride):
                raise ValueError(
                    "padding='same' is not supported for strided convolutions"
                )

        valid_padding_modes = {"zeros", "reflect", "replicate", "circular"}
        if padding_mode not in valid_padding_modes:
            raise ValueError(
                f"padding_mode must be one of {valid_padding_modes}, but got padding_mode='{padding_mode}'"
            )

        # Store parameters
        self.num_streams = num_streams
        self.stream_in_channels = stream_in_channels
        self.stream_out_channels = stream_out_channels
        self.integrated_in_channels = integrated_in_channels
        self.integrated_out_channels = integrated_out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        
        # Handle padding mode exactly like _ConvNd
        # `_reversed_padding_repeated_twice` is the padding to be passed to
        # `F.pad` if needed (e.g., for non-zero padding types that are
        # implemented as two ops: padding + conv). `F.pad` accepts paddings in
        # reverse order than the dimension.
        if isinstance(self.padding, str):
            self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size)
            if padding == "same":
                for d, k, i in zip(
                    dilation, kernel_size, range(len(kernel_size) - 1, -1, -1)
                ):
                    total_padding = d * (k - 1)
                    left_pad = total_padding // 2
                    self._reversed_padding_repeated_twice[2 * i] = left_pad
                    self._reversed_padding_repeated_twice[2 * i + 1] = (
                        total_padding - left_pad
                    )
        else:
            # Use the same reverse repeat tuple function as PyTorch
            self._reversed_padding_repeated_twice = _reverse_repeat_tuple(
                self.padding, 2
            )
        
        # Create weight parameters for all N streams (full kernel_size)
        if transposed:
            self.stream_weights = nn.ParameterList([
                Parameter(torch.empty(
                    (stream_in_channels[i], stream_out_channels[i] // groups, *kernel_size),
                    **factory_kwargs,
                ))
                for i in range(num_streams)
            ])
            # Integrated weight is always 1x1 (channel-wise only)
            # Note: Can have shape (0, out_channels, 1, 1) when integrated_in_channels=0 (first layer)
            self.integrated_weight = Parameter(
                torch.empty(
                    (integrated_in_channels, integrated_out_channels // groups, 1, 1),
                    **factory_kwargs,
                )
            )
        else:
            self.stream_weights = nn.ParameterList([
                Parameter(torch.empty(
                    (stream_out_channels[i], stream_in_channels[i] // groups, *kernel_size),
                    **factory_kwargs,
                ))
                for i in range(num_streams)
            ])
            # Integrated weight is always 1x1 (channel-wise only)
            # Note: Can have shape (out_channels, 0, 1, 1) when integrated_in_channels=0 (first layer)
            self.integrated_weight = Parameter(
                torch.empty(
                    (integrated_out_channels, integrated_in_channels // groups, 1, 1),
                    **factory_kwargs,
                )
            )

        # Create bias parameters for all N streams
        if bias:
            self.stream_biases = nn.ParameterList([
                Parameter(torch.empty(stream_out_channels[i], **factory_kwargs))
                for i in range(num_streams)
            ])
            self.integrated_bias = Parameter(torch.empty(integrated_out_channels, **factory_kwargs))
        else:
            self.register_parameter("stream_biases", None)
            self.register_parameter("integrated_bias", None)

        # Create 1x1 integration weights for Linear Integration (channel-wise mixing)
        # These weights learn how to combine all stream outputs into the integrated stream
        self.integration_from_streams = nn.ParameterList([
            Parameter(torch.empty(
                (integrated_out_channels, stream_out_channels[i], 1, 1),
                **factory_kwargs,
            ))
            for i in range(num_streams)
        ])

        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        """Reset parameters exactly like _ConvNd."""
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573

        # Initialize all stream pathway weights
        for stream_weight in self.stream_weights:
            init.kaiming_uniform_(stream_weight, a=math.sqrt(5))

        if self.stream_biases is not None:
            for i, stream_bias in enumerate(self.stream_biases):
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.stream_weights[i])
                if fan_in != 0:
                    bound = 1 / math.sqrt(fan_in)
                    init.uniform_(stream_bias, -bound, bound)

        # Initialize integrated pathway weights
        init.kaiming_uniform_(self.integrated_weight, a=math.sqrt(5))
        if self.integrated_bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.integrated_weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.integrated_bias, -bound, bound)

        # Initialize integration weights (1x1 convolutions for stream mixing)
        for integration_weight in self.integration_from_streams:
            init.kaiming_uniform_(integration_weight, a=math.sqrt(5))
    
    def extra_repr(self):
        """String representation exactly like _ConvNd."""
        s = (
            f"num_streams={self.num_streams}, "
            f"stream_in_channels={self.stream_in_channels}, "
            f"stream_out_channels={self.stream_out_channels}, "
            f"integrated_in_channels={self.integrated_in_channels}, "
            f"integrated_out_channels={self.integrated_out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}"
        )
        if self.padding != (0,) * len(self.padding):
            s += f", padding={self.padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += f", dilation={self.dilation}"
        if self.output_padding != (0,) * len(self.output_padding):
            s += f", output_padding={self.output_padding}"
        if self.groups != 1:
            s += f", groups={self.groups}"
        if self.stream_biases is None:
            s += ", bias=False"
        if self.padding_mode != "zeros":
            s += f", padding_mode={self.padding_mode}"
        return s
    
    def __setstate__(self, state):
        """Handle backward compatibility like _ConvNd."""
        super().__setstate__(state)
        if not hasattr(self, "padding_mode"):
            self.padding_mode = "zeros"


class LIConv2d(_LIConvNd):
    """
    Linear Integration 2D Convolution layer - follows PyTorch's Conv2d pattern exactly.

    This layer processes N input streams and integrated stream with learned integration,
    maintaining full compatibility with PyTorch's Conv2d interface.
    """

    def __init__(
        self,
        stream_in_channels: list[int],
        stream_out_channels: list[int],
        integrated_in_channels: int,
        integrated_out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super().__init__(
            stream_in_channels,
            stream_out_channels,
            integrated_in_channels,
            integrated_out_channels,
            kernel_size_,
            stride_,
            padding_,
            dilation_,
            False,  # transposed (Conv2d is not transposed)
            _pair(0),  # output_padding (not used for Conv2d)
            groups,
            bias,
            padding_mode,
            **factory_kwargs,
        )
    
    def _conv_forward(
        self,
        stream_inputs: list[Tensor],
        integrated_input: Optional[Tensor],
        stream_weights: list[Tensor],
        integrated_weight: Optional[Tensor],
        integration_from_streams_weights: list[Tensor],
        stream_biases: list[Optional[Tensor]],
        integrated_bias: Optional[Tensor],
        blanked_mask: Optional[dict[int, Tensor]] = None
    ) -> tuple[list[Tensor], Tensor]:
        """
        Forward pass with biologically-inspired integration.

        Biological analogy:
        - Dendritic filtering: Conv operation (spatial processing without bias)
        - Stream outputs: Conv + bias (pathway-specific baseline potential)
        - Soma integration: Integrates RAW conv outputs (without stream biases)
        - Soma threshold: integrated_bias (neuron's firing threshold)

        This separates pathway-specific biases from the integration threshold.

        Args:
            blanked_mask: Optional per-sample blanking mask for modality dropout.
                         dict[stream_idx] -> bool tensor [batch_size] where True = blanked.
                         When a stream is blanked for a sample, its output is zeroed.
        """
        # Process all stream pathways
        # Compute RAW conv outputs (dendritic filtering, no bias)
        stream_outputs = []
        stream_outputs_raw = []  # Raw outputs for integration (without bias)

        for i, (stream_input, stream_weight, stream_bias) in enumerate(
            zip(stream_inputs, stream_weights, stream_biases)
        ):
            # Compute raw conv output (dendritic spatial filtering)
            if self.padding_mode != "zeros":
                stream_out_raw = F.conv2d(
                    F.pad(
                        stream_input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                    ),
                    stream_weight,
                    None,  # No bias for raw output
                    self.stride,
                    _pair(0),
                    self.dilation,
                    self.groups,
                )
            else:
                stream_out_raw = F.conv2d(
                    stream_input, stream_weight, None, self.stride, self.padding, self.dilation, self.groups
                )

            # Add bias for stream's own output (pathway-specific baseline)
            if stream_bias is not None:
                stream_out = stream_out_raw + stream_bias.view(1, -1, 1, 1)
            else:
                stream_out = stream_out_raw

            # === PER-SAMPLE MASKING: Zero blanked samples for modality dropout ===
            stream_blanked = blanked_mask.get(i) if blanked_mask else None
            if stream_blanked is not None and stream_blanked.any():
                # mask: [batch_size, 1, 1, 1] for broadcasting, 1.0 for active, 0.0 for blanked
                mask = (~stream_blanked).float().view(-1, 1, 1, 1)
                # OPTIMIZATION: Use in-place operations to avoid creating temporary tensors
                stream_out.mul_(mask)
                stream_out_raw.mul_(mask)  # Critical: zeros for integration

            stream_outputs.append(stream_out)  # Biased output for stream pathway
            stream_outputs_raw.append(stream_out_raw)  # Raw output for integration

        # ===== Process integrated pathway (Soma Integration) =====
        # Apply Linear Integration on RAW conv outputs (dendritic signals without bias)
        # integrated_out = W_prev·integrated_prev + Σ(Wi·stream_i_raw) + bias_integrated
        # This ensures only the soma has its own threshold bias, not redundant with stream biases

        # Process previous integrated (if exists) using 1x1 conv with stride matching
        # IMPORTANT: integrated_weight is 1x1, but stride must match main conv for spatial alignment
        if integrated_input is not None:
            integrated_from_prev = F.conv2d(
                integrated_input, integrated_weight, None,  # No bias here
                stride=self.stride, padding=0  # 1x1 conv, stride matches main conv
            )
        else:
            # First layer: no previous integrated stream
            integrated_from_prev = 0

        # Integration step: combine RAW stream outputs using 1x1 convs (soma integrates dendritic signals)
        # Key difference from Original: Use stream_outputs_raw (without bias) instead of stream_outputs (with bias)
        # Use sequential addition (not sum()) to match Original's floating-point behavior
        integrated_out = integrated_from_prev
        for stream_out_raw, integration_weight in zip(stream_outputs_raw, integration_from_streams_weights):
            integrated_contrib = F.conv2d(
                stream_out_raw, integration_weight, None,  # Integrate RAW dendritic signals
                stride=1, padding=0  # 1x1 conv, stride=1 (already spatially aligned)
            )
            integrated_out = integrated_out + integrated_contrib

        # Add integrated bias (soma's firing threshold)
        # This is the ONLY bias for integration, representing the membrane potential threshold
        if integrated_bias is not None:
            integrated_out = integrated_out + integrated_bias.view(1, -1, 1, 1)

        return stream_outputs, integrated_out
    
    def forward(
        self,
        stream_inputs: list[Tensor],
        integrated_input: Optional[Tensor] = None,
        blanked_mask: Optional[dict[int, Tensor]] = None
    ) -> tuple[list[Tensor], Tensor]:
        """Forward pass through all N convolution streams with Linear Integration.

        Args:
            stream_inputs: List of input tensors for each stream
            integrated_input: Optional integrated stream input (None for first layer)
            blanked_mask: Optional per-sample blanking mask for modality dropout.
                         dict[stream_idx] -> bool tensor [batch_size] where True = blanked.
        """
        # Convert ParameterList to list of tensors for _conv_forward
        stream_weights_list = list(self.stream_weights)
        integration_from_streams_weights_list = list(self.integration_from_streams)
        stream_biases_list = list(self.stream_biases) if self.stream_biases is not None else [None] * self.num_streams

        return self._conv_forward(
            stream_inputs,
            integrated_input,
            stream_weights_list,
            self.integrated_weight,
            integration_from_streams_weights_list,
            stream_biases_list,
            self.integrated_bias,
            blanked_mask
        )

    def forward_stream(self, stream_idx: int, stream_input: Tensor) -> Tensor:
        """
        Forward pass through a single stream pathway only.

        This method processes ONLY the specified stream without processing other streams
        or the integrated stream. Used for stream monitoring during training to avoid
        running forward passes with dummy/zero data for other streams.

        Args:
            stream_idx: Index of the stream to forward (0-indexed)
            stream_input: The input tensor for this stream

        Returns:
            The convolution output for this stream only (with bias if applicable)
        """
        stream_weight = self.stream_weights[stream_idx]
        stream_bias = self.stream_biases[stream_idx] if self.stream_biases is not None else None

        # Compute conv output with bias (matching main forward)
        if self.padding_mode != "zeros":
            return F.conv2d(
                F.pad(
                    stream_input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                stream_weight,
                stream_bias,
                self.stride,
                _pair(0),
                self.dilation,
                self.groups,
            )
        else:
            return F.conv2d(
                stream_input, stream_weight, stream_bias, self.stride, self.padding, self.dilation, self.groups
            )


class _LINormBase(nn.Module):
    """Common base for Linear Integration normalization - follows PyTorch's _NormBase pattern exactly."""

    _version = 2
    __constants__ = ["track_running_stats", "momentum", "eps", "num_streams", "stream_num_features", "integrated_num_features", "affine"]
    num_streams: int
    stream_num_features: list[int]
    integrated_num_features: int
    eps: float
    momentum: Optional[float]
    affine: bool
    track_running_stats: bool
    # WARNING: stream_weights, stream_biases purposely not defined here.
    # Following PyTorch's pattern from https://github.com/pytorch/pytorch/issues/39670

    # Type annotations for buffers that are always present
    integrated_running_mean: Optional[Tensor]
    integrated_running_var: Optional[Tensor]
    num_batches_tracked: Optional[Tensor]
    # NOTE: Per-stream buffers (stream{i}_running_mean, stream{i}_running_var) are registered
    # dynamically at runtime for i in range(num_streams), so they cannot be statically typed here.
    
    def __init__(
        self,
        stream_num_features: list[int],
        integrated_num_features: int,
        eps: float = 1e-5,
        momentum: Optional[float] = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_streams = len(stream_num_features)
        self.stream_num_features = stream_num_features
        self.integrated_num_features = integrated_num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        # Create parameters for all N stream pathways - exactly like _NormBase but for N streams
        if self.affine:
            self.stream_weights = nn.ParameterList([
                Parameter(torch.empty(num_features, **factory_kwargs))
                for num_features in stream_num_features
            ])
            self.stream_biases = nn.ParameterList([
                Parameter(torch.empty(num_features, **factory_kwargs))
                for num_features in stream_num_features
            ])
            self.integrated_weight = Parameter(torch.empty(integrated_num_features, **factory_kwargs))
            self.integrated_bias = Parameter(torch.empty(integrated_num_features, **factory_kwargs))
        else:
            self.register_parameter("stream_weights", None)
            self.register_parameter("stream_biases", None)
            self.register_parameter("integrated_weight", None)
            self.register_parameter("integrated_bias", None)

        # Create buffers for all N stream pathways
        if self.track_running_stats:
            # Register running stats for each stream
            for i, num_features in enumerate(stream_num_features):
                self.register_buffer(
                    f"stream{i}_running_mean", torch.zeros(num_features, **factory_kwargs)
                )
                self.register_buffer(
                    f"stream{i}_running_var", torch.ones(num_features, **factory_kwargs)
                )

            # Register integrated stream buffers
            self.register_buffer(
                "integrated_running_mean", torch.zeros(integrated_num_features, **factory_kwargs)
            )
            self.register_buffer(
                "integrated_running_var", torch.ones(integrated_num_features, **factory_kwargs)
            )

            # Register num_batches_tracked (shared across all streams)
            self.register_buffer(
                "num_batches_tracked",
                torch.tensor(
                    0,
                    dtype=torch.long,
                    **{k: v for k, v in factory_kwargs.items() if k != "dtype"},
                ),
            )
        else:
            # Register None buffers for each stream
            for i in range(self.num_streams):
                self.register_buffer(f"stream{i}_running_mean", None)
                self.register_buffer(f"stream{i}_running_var", None)

            self.register_buffer("integrated_running_mean", None)
            self.register_buffer("integrated_running_var", None)
            self.register_buffer("num_batches_tracked", None)

        self.reset_parameters()
    
    def reset_running_stats(self) -> None:
        """Reset running statistics for all N pathways - exactly like _NormBase but for N streams."""
        if self.track_running_stats:
            # running_mean/running_var/num_batches... are registered at runtime depending
            # if self.track_running_stats is on
            for i in range(self.num_streams):
                getattr(self, f"stream{i}_running_mean").zero_()  # type: ignore[union-attr]
                getattr(self, f"stream{i}_running_var").fill_(1)  # type: ignore[union-attr]

            self.integrated_running_mean.zero_()  # type: ignore[union-attr]
            self.integrated_running_var.fill_(1)  # type: ignore[union-attr]

            # Shared batch tracking
            self.num_batches_tracked.zero_()  # type: ignore[union-attr,operator]

    def reset_parameters(self) -> None:
        """Reset parameters for all N pathways - exactly like _NormBase but for N streams."""
        self.reset_running_stats()
        if self.affine:
            for stream_weight, stream_bias in zip(self.stream_weights, self.stream_biases):
                init.ones_(stream_weight)
                init.zeros_(stream_bias)
            init.ones_(self.integrated_weight)
            init.zeros_(self.integrated_bias)
    
    def _check_input_dim(self, input):
        """Check input dimensions - to be implemented by subclasses."""
        raise NotImplementedError
    
    def extra_repr(self):
        """String representation - updated for N streams."""
        return (
            f"num_streams={self.num_streams}, "
            f"stream_num_features={self.stream_num_features}, "
            f"integrated_num_features={self.integrated_num_features}, "
            f"eps={self.eps}, momentum={self.momentum}, affine={self.affine}, "
            f"track_running_stats={self.track_running_stats}"
        )
    
    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Handle state dict loading - exactly like _NormBase but for dual pathways."""
        version = local_metadata.get("version", None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            # this should have a default value of 0
            num_batches_tracked_key = prefix + "num_batches_tracked"
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = (
                    self.num_batches_tracked
                    if self.num_batches_tracked is not None
                    and self.num_batches_tracked.device != torch.device("meta")
                    else torch.tensor(0, dtype=torch.long)
                )

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class _LIBatchNorm(_LINormBase):
    """Linear Integration BatchNorm - follows PyTorch's _BatchNorm pattern exactly."""

    def __init__(
        self,
        stream_num_features: list[int],
        integrated_num_features: int,
        eps: float = 1e-5,
        momentum: Optional[float] = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            stream_num_features, integrated_num_features, eps, momentum, affine, track_running_stats, **factory_kwargs
        )
    
    def forward(
        self,
        stream_inputs: list[Tensor],
        integrated_input: Optional[Tensor] = None,
        blanked_mask: Optional[dict[int, Tensor]] = None
    ) -> tuple[list[Tensor], Tensor]:
        """
        Forward pass - implements the exact same algorithm as _BatchNorm.forward() for N streams.

        Args:
            stream_inputs: List of input tensors for each stream
            integrated_input: Optional integrated stream input
            blanked_mask: Optional per-sample blanking mask for modality dropout.
                         dict[stream_idx] -> bool tensor [batch_size] where True = blanked.
                         For blanked samples, BN is skipped and zeros are passed through.
        """
        # Check input dimensions for all inputs
        for stream_input in stream_inputs:
            self._check_input_dim(stream_input)
        if integrated_input is not None:
            self._check_input_dim(integrated_input)

        # Compute exponential_average_factor ONCE per forward pass (not per stream!)
        # This fixes a bug where num_batches_tracked was incremented N+1 times per batch
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked.add_(1)  # Increment ONCE per forward pass
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # Process all stream pathways using subset BN approach for modality dropout
        stream_outputs = []
        for i, stream_input in enumerate(stream_inputs):
            stream_blanked = blanked_mask.get(i) if blanked_mask else None

            if stream_blanked is None or not stream_blanked.any():
                # No blanking - standard BN on all samples
                stream_out = self._forward_single_pathway(
                    stream_input,
                    getattr(self, f"stream{i}_running_mean"),
                    getattr(self, f"stream{i}_running_var"),
                    self.stream_weights[i] if self.affine else None,
                    self.stream_biases[i] if self.affine else None,
                    exponential_average_factor,
                )
            elif stream_blanked.all():
                # All samples blanked - pass through zeros unchanged
                # (input is already zeros from conv masking)
                stream_out = stream_input
            else:
                # Partial blanking - subset BN on active samples only
                # OPTIMIZATION: More efficient indexing and scatter operations
                active_idx = torch.where(~stream_blanked)[0]

                # Process only active samples to maintain correct BN statistics
                active_output = self._forward_single_pathway(
                    stream_input[active_idx],
                    getattr(self, f"stream{i}_running_mean"),
                    getattr(self, f"stream{i}_running_var"),
                    self.stream_weights[i] if self.affine else None,
                    self.stream_biases[i] if self.affine else None,
                    exponential_average_factor,
                )

                # OPTIMIZATION: Use scatter operation with new_zeros for efficiency
                # new_zeros is slightly faster than zeros_like for large tensors
                stream_out = stream_input.new_zeros(stream_input.shape)
                stream_out.index_copy_(0, active_idx, active_output)

            stream_outputs.append(stream_out)

        # Process integrated pathway (no masking - always all samples)
        if integrated_input is not None:
            integrated_out = self._forward_single_pathway(
                integrated_input,
                self.integrated_running_mean,
                self.integrated_running_var,
                self.integrated_weight,
                self.integrated_bias,
                exponential_average_factor,
            )
        else:
            integrated_out = None

        return stream_outputs, integrated_out

    def _forward_single_pathway(
        self,
        input: Tensor,
        running_mean: Optional[Tensor],
        running_var: Optional[Tensor],
        weight: Optional[Tensor],
        bias: Optional[Tensor],
        exponential_average_factor: float,
    ) -> Tensor:
        """
        Forward pass for a single pathway - applies batch normalization.

        Note: exponential_average_factor is pre-computed in forward() to ensure
        num_batches_tracked is only incremented once per forward pass, not per stream.
        """
        # Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        # Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        if self.training:
            bn_training = True
        else:
            bn_training = (running_mean is None) and (running_var is None)

        # Buffers are only updated if they are to be tracked and we are in training mode.
        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            running_mean if not self.training or self.track_running_stats else None,
            running_var if not self.training or self.track_running_stats else None,
            weight,
            bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )


class LIBatchNorm2d(_LIBatchNorm):
    r"""Applies Linear Integration Batch Normalization over N 4D inputs.

    This layer applies Batch Normalization separately to N stream and integrated pathways,
    following PyTorch's BatchNorm2d pattern exactly but extended for N-stream processing.
    Each pathway operates independently with its own parameters and running statistics.

    The 4D inputs are mini-batches of 2D inputs with additional channel dimension.
    Method described in the paper `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__.

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over the mini-batches
    and :math:`\gamma` and :math:`\beta` are learnable parameter vectors of size `C`
    (where `C` is the input size). By default, the elements of :math:`\gamma` are set
    to 1 and the elements of :math:`\beta` are set to 0. At train time in the forward pass, the
    standard-deviation is calculated via the biased estimator, equivalent to
    ``torch.var(input, unbiased=False)``. However, the value stored in the moving average of the
    standard-deviation is calculated via the unbiased estimator, equivalent to
    ``torch.var(input, unbiased=True)``.

    Also by default, during training this layer keeps running estimates of its
    computed mean and variance, which are then used for normalization during
    evaluation. The running estimates are kept with a default :attr:`momentum`
    of 0.1.

    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and batch statistics are instead used during
    evaluation time as well.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    Because the Batch Normalization is done over the `C` dimension, computing statistics
    on `(N, H, W)` slices, it's common terminology to call this Spatial Batch Normalization.

    Args:
        stream_num_features: List of :math:`C` values from expected stream inputs of size :math:`(N, C, H, W)`
        integrated_num_features: :math:`C` from expected integrated input of size :math:`(N, C, H, W)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics, and initializes statistics
            buffers :attr:`running_mean` and :attr:`running_var` as ``None``.
            When these buffers are ``None``, this module always uses batch statistics.
            in both training and eval modes. Default: ``True``

    Shape:
        - Stream Inputs: List of :math:`(N, C_{i}, H, W)` where i = 0..N-1
        - Integrated Input: :math:`(N, C_{integrated}, H, W)`
        - Stream Outputs: List of :math:`(N, C_{i}, H, W)` (same shape as stream inputs)
        - Integrated Output: :math:`(N, C_{integrated}, H, W)` (same shape as integrated input)

    Examples::

        >>> # With Learnable Parameters for 3 streams
        >>> m = LIBatchNorm2d([64, 64, 64], 64)
        >>> # Without Learnable Parameters
        >>> m = LIBatchNorm2d([64, 64, 64], 64, affine=False)
        >>> stream_inputs = [torch.randn(20, 64, 35, 45) for _ in range(3)]
        >>> integrated_input = torch.randn(20, 64, 35, 45)
        >>> stream_outputs, integrated_output = m(stream_inputs, integrated_input)
    """
    
    def _check_input_dim(self, input):
        """Check input dimensions - same as BatchNorm2d."""
        if input.dim() != 4:
            raise ValueError(f"expected 4D input (got {input.dim()}D input)")

    def forward_stream(self, stream_idx: int, stream_input: Tensor) -> Tensor:
        """
        Forward pass through a single stream pathway only.

        This method processes ONLY the specified stream without affecting other streams'
        running statistics. Used for stream monitoring during training to avoid
        corrupting BN stats for other streams.

        Args:
            stream_idx: Index of the stream to forward (0-indexed)
            stream_input: The input tensor for this stream

        Returns:
            The batch-normalized output for this stream
        """
        self._check_input_dim(stream_input)

        # Compute exponential_average_factor for single-stream forward
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked.add_(1)
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        return self._forward_single_pathway(
            stream_input,
            getattr(self, f"stream{stream_idx}_running_mean"),
            getattr(self, f"stream{stream_idx}_running_var"),
            self.stream_weights[stream_idx] if self.affine else None,
            self.stream_biases[stream_idx] if self.affine else None,
            exponential_average_factor,
        )