"""
Linear Integration convolution layers for 3-stream processing.

This module provides 3-stream convolution operations with learned integration
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
    _ConvNd but adapted for 3-stream (stream1/stream2/integrated) processing.
    """
    
    __constants__ = [
        "stride", "padding", "dilation", "groups", "padding_mode", "output_padding",
        "stream1_in_channels", "stream2_in_channels", "integrated_in_channels",
        "stream1_out_channels", "stream2_out_channels", "integrated_out_channels", "kernel_size",
    ]
    __annotations__ = {
        "stream1_bias": Optional[torch.Tensor],
        "stream2_bias": Optional[torch.Tensor],
        "integrated_bias": Optional[torch.Tensor],
    }
    
    def _conv_forward(self, stream1_input: Tensor, stream2_input: Tensor, integrated_input: Tensor,
                     stream1_weight: Tensor, stream2_weight: Tensor, integrated_weight: Tensor,
                     stream1_bias: Optional[Tensor], stream2_bias: Optional[Tensor], integrated_bias: Optional[Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        """Abstract method to be implemented by subclasses."""
        ...

    stream1_in_channels: int
    stream2_in_channels: int
    integrated_in_channels: int
    stream1_out_channels: int
    stream2_out_channels: int
    integrated_out_channels: int
    kernel_size: tuple[int, ...]
    stride: tuple[int, ...]
    padding: Union[str, tuple[int, ...]]
    dilation: tuple[int, ...]
    transposed: bool
    output_padding: tuple[int, ...]
    groups: int
    padding_mode: str
    stream1_weight: Tensor
    stream2_weight: Tensor
    integrated_weight: Tensor
    stream1_bias: Optional[Tensor]
    stream2_bias: Optional[Tensor]
    integrated_bias: Optional[Tensor]
    _reversed_padding_repeated_twice: list[int]

    def __init__(
        self,
        stream1_in_channels: int,
        stream2_in_channels: int,
        integrated_in_channels: int,
        stream1_out_channels: int,
        stream2_out_channels: int,
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
        if stream1_in_channels % groups != 0:
            raise ValueError("stream1_in_channels must be divisible by groups")
        if stream2_in_channels % groups != 0:
            raise ValueError("stream2_in_channels must be divisible by groups")
        # Allow integrated_in_channels=0 for first layer (no integrated input yet)
        if integrated_in_channels > 0 and integrated_in_channels % groups != 0:
            raise ValueError("integrated_in_channels must be divisible by groups")
        if stream1_out_channels % groups != 0:
            raise ValueError("stream1_out_channels must be divisible by groups")
        if stream2_out_channels % groups != 0:
            raise ValueError("stream2_out_channels must be divisible by groups")
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
        self.stream1_in_channels = stream1_in_channels
        self.stream2_in_channels = stream2_in_channels
        self.integrated_in_channels = integrated_in_channels
        self.stream1_out_channels = stream1_out_channels
        self.stream2_out_channels = stream2_out_channels
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
        
        # Create weight parameters for stream1 and stream2 pathways (full kernel_size)
        if transposed:
            self.stream1_weight = Parameter(
                torch.empty(
                    (stream1_in_channels, stream1_out_channels // groups, *kernel_size),
                    **factory_kwargs,
                )
            )
            self.stream2_weight = Parameter(
                torch.empty(
                    (stream2_in_channels, stream2_out_channels // groups, *kernel_size),
                    **factory_kwargs,
                )
            )
            # Integrated weight is always 1x1 (channel-wise only)
            self.integrated_weight = Parameter(
                torch.empty(
                    (integrated_in_channels, integrated_out_channels // groups, 1, 1),
                    **factory_kwargs,
                )
            )
        else:
            self.stream1_weight = Parameter(
                torch.empty(
                    (stream1_out_channels, stream1_in_channels // groups, *kernel_size),
                    **factory_kwargs,
                )
            )
            self.stream2_weight = Parameter(
                torch.empty(
                    (stream2_out_channels, stream2_in_channels // groups, *kernel_size),
                    **factory_kwargs,
                )
            )
            # Integrated weight is always 1x1 (channel-wise only)
            self.integrated_weight = Parameter(
                torch.empty(
                    (integrated_out_channels, integrated_in_channels // groups, 1, 1),
                    **factory_kwargs,
                )
            )
        
        # Create bias parameters for all 3 pathways
        if bias:
            self.stream1_bias = Parameter(torch.empty(stream1_out_channels, **factory_kwargs))
            self.stream2_bias = Parameter(torch.empty(stream2_out_channels, **factory_kwargs))
            self.integrated_bias = Parameter(torch.empty(integrated_out_channels, **factory_kwargs))
        else:
            self.register_parameter("stream1_bias", None)
            self.register_parameter("stream2_bias", None)
            self.register_parameter("integrated_bias", None)

        # Create 1x1 integration weights for Linear Integration (channel-wise mixing)
        # These weights learn how to combine stream1 and stream2 outputs into the integrated stream
        self.integration_from_stream1 = Parameter(
            torch.empty(
                (integrated_out_channels, stream1_out_channels, 1, 1),
                **factory_kwargs,
            )
        )
        self.integration_from_stream2 = Parameter(
            torch.empty(
                (integrated_out_channels, stream2_out_channels, 1, 1),
                **factory_kwargs,
            )
        )

        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        """Reset parameters exactly like _ConvNd."""
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573

        # Initialize stream1 pathway weights
        init.kaiming_uniform_(self.stream1_weight, a=math.sqrt(5))
        if self.stream1_bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.stream1_weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.stream1_bias, -bound, bound)

        # Initialize stream2 pathway weights
        init.kaiming_uniform_(self.stream2_weight, a=math.sqrt(5))
        if self.stream2_bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.stream2_weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.stream2_bias, -bound, bound)

        # Initialize integrated pathway weights
        init.kaiming_uniform_(self.integrated_weight, a=math.sqrt(5))
        if self.integrated_bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.integrated_weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.integrated_bias, -bound, bound)

        # Initialize integration weights (1x1 convolutions for stream mixing)
        # CRITICAL: Use variance-preserving initialization for deep networks (ResNet50+)
        # The integrated stream combines 3 independent contributions:
        #   integrated_out = integrated_from_prev + integrated_from_s1 + integrated_from_s2
        # When summing N independent variables, output variance = N * input variance
        # To maintain unit variance, scale each contribution by 1/sqrt(N)
        # This prevents FP16 overflow in AMP and ensures stable training in deep networks
        init.kaiming_uniform_(self.integration_from_stream1, a=math.sqrt(5))
        self.integration_from_stream1.data /= math.sqrt(3.0)
        init.kaiming_uniform_(self.integration_from_stream2, a=math.sqrt(5))
        self.integration_from_stream2.data /= math.sqrt(3.0)
        # Also scale the integrated pathway weight for consistency
        init.kaiming_uniform_(self.integrated_weight, a=math.sqrt(5))
        self.integrated_weight.data /= math.sqrt(3.0)
    
    def extra_repr(self):
        """String representation exactly like _ConvNd."""
        s = (
            "stream1_in_channels={stream1_in_channels}, stream2_in_channels={stream2_in_channels}, "
            "integrated_in_channels={integrated_in_channels}, "
            "stream1_out_channels={stream1_out_channels}, stream2_out_channels={stream2_out_channels}, "
            "integrated_out_channels={integrated_out_channels}, "
            "kernel_size={kernel_size}, stride={stride}"
        )
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.output_padding != (0,) * len(self.output_padding):
            s += ", output_padding={output_padding}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.stream1_bias is None:
            s += ", bias=False"
        if self.padding_mode != "zeros":
            s += ", padding_mode={padding_mode}"
        return s.format(**self.__dict__)
    
    def __setstate__(self, state):
        """Handle backward compatibility like _ConvNd."""
        super().__setstate__(state)
        if not hasattr(self, "padding_mode"):
            self.padding_mode = "zeros"


class LIConv2d(_LIConvNd):
    """
    Linear Integration 2D Convolution layer - follows PyTorch's Conv2d pattern exactly.

    This layer processes stream1, stream2, and integrated streams with learned integration
    using 5 weight matrices, maintaining full compatibility with PyTorch's Conv2d interface.
    """
    
    def __init__(
        self,
        stream1_in_channels: int,
        stream2_in_channels: int,
        integrated_in_channels: int,
        stream1_out_channels: int,
        stream2_out_channels: int,
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
            stream1_in_channels,
            stream2_in_channels,
            integrated_in_channels,
            stream1_out_channels,
            stream2_out_channels,
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
    
    def _conv_forward(self, stream1_input: Tensor, stream2_input: Tensor, integrated_input: Optional[Tensor],
                     stream1_weight: Tensor, stream2_weight: Tensor, integrated_weight: Tensor,
                     stream1_bias: Optional[Tensor], stream2_bias: Optional[Tensor], integrated_bias: Optional[Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass implementation exactly like Conv2d._conv_forward."""
        # Process stream1 pathway
        if self.padding_mode != "zeros":
            stream1_out = F.conv2d(
                F.pad(
                    stream1_input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                stream1_weight,
                stream1_bias,
                self.stride,
                _pair(0),
                self.dilation,
                self.groups,
            )
        else:
            stream1_out = F.conv2d(
                stream1_input, stream1_weight, stream1_bias, self.stride, self.padding, self.dilation, self.groups
            )

        # Process stream2 pathway
        if self.padding_mode != "zeros":
            stream2_out = F.conv2d(
                F.pad(
                    stream2_input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                stream2_weight,
                stream2_bias,
                self.stride,
                _pair(0),
                self.dilation,
                self.groups,
            )
        else:
            stream2_out = F.conv2d(
                stream2_input, stream2_weight, stream2_bias, self.stride, self.padding, self.dilation, self.groups
            )

        # ===== Process integrated pathway =====
        # Apply Linear Integration: integrated_out = W3·integrated + W1·stream1 + W2·stream2 + bias

        # Process previous integrated (if exists) using 1x1 conv with stride matching
        # IMPORTANT: integrated_weight is 1x1, but stride must match main conv for spatial alignment
        # Note: We apply bias at the end (after summing all contributions) for consistency
        if integrated_input is not None:
            integrated_from_prev = F.conv2d(
                integrated_input, integrated_weight, None,  # No bias here
                stride=self.stride, padding=0  # 1x1 conv, stride matches main conv
            )
        else:
            # First layer: no previous integrated stream
            integrated_from_prev = 0

        # Integration step: combine stream outputs using 1x1 convs (channel-wise mixing)
        integrated_from_s1 = F.conv2d(
            stream1_out, self.integration_from_stream1, None,
            stride=1, padding=0  # 1x1 conv, stride=1 (already spatially aligned)
        )
        integrated_from_s2 = F.conv2d(
            stream2_out, self.integration_from_stream2, None,
            stride=1, padding=0  # 1x1 conv, stride=1 (already spatially aligned)
        )

        # Combine all contributions to create integrated output
        integrated_out = integrated_from_prev + integrated_from_s1 + integrated_from_s2

        # Add bias at the end (ensures bias is applied in both first layer and later layers)
        if integrated_bias is not None:
            integrated_out = integrated_out + integrated_bias.view(1, -1, 1, 1)

        return stream1_out, stream2_out, integrated_out
    
    def forward(self, stream1_input: Tensor, stream2_input: Tensor, integrated_input: Optional[Tensor] = None) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass through all 3 convolution streams with Linear Integration."""
        return self._conv_forward(
            stream1_input, stream2_input, integrated_input,
            self.stream1_weight, self.stream2_weight, self.integrated_weight,
            self.stream1_bias, self.stream2_bias, self.integrated_bias
        )
    
    def forward_stream1(self, stream1_input: Tensor) -> Tensor:
        """Forward pass through stream1 stream only."""
        if self.padding_mode != "zeros":
            return F.conv2d(
                F.pad(
                    stream1_input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                self.stream1_weight,
                self.stream1_bias,
                self.stride,
                _pair(0),
                self.dilation,
                self.groups,
            )
        return F.conv2d(
            stream1_input, self.stream1_weight, self.stream1_bias, self.stride, self.padding, self.dilation, self.groups
        )
    
    def forward_stream2(self, stream2_input: Tensor) -> Tensor:
        """Forward pass through stream2 stream only."""
        if self.padding_mode != "zeros":
            return F.conv2d(
                F.pad(
                    stream2_input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                self.stream2_weight,
                self.stream2_bias,
                self.stride,
                _pair(0),
                self.dilation,
                self.groups,
            )
        return F.conv2d(
            stream2_input, self.stream2_weight, self.stream2_bias, self.stride, self.padding, self.dilation, self.groups
        )
    
    def _forward_interleaved(self, stream1_input: Tensor, stream2_input: Tensor, integrated_input: Optional[Tensor] = None) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass using channel interleaving for balanced processing."""
        # Repeat brightness to match color channels for interleaving
        if self.stream2_in_channels < self.stream1_in_channels:
            # Repeat brightness channels to match color
            repeat_factor = self.stream1_in_channels // self.stream2_in_channels
            remainder = self.stream1_in_channels % self.stream2_in_channels
            
            brightness_repeated = stream2_input.repeat(1, repeat_factor, 1, 1)
            if remainder > 0:
                brightness_extra = stream2_input[:, :remainder, :, :]
                brightness_repeated = torch.cat([brightness_repeated, brightness_extra], dim=1)
        else:
            brightness_repeated = stream2_input
        
        # Interleave channels: [C1, B1, C2, B2, C3, B3, ...]
        channels = []
        for i in range(max(self.stream1_in_channels, self.stream2_in_channels)):
            if i < self.stream1_in_channels:
                channels.append(stream1_input[:, i:i+1, :, :])
            if i < brightness_repeated.size(1):
                channels.append(brightness_repeated[:, i:i+1, :, :])
        
        interleaved_input = torch.cat(channels, dim=1)
        
        # Create interleaved weights
        stream1_weight_expanded = self.stream1_weight.repeat_interleave(2, dim=1)
        stream2_weight_expanded = self.stream2_weight.repeat_interleave(2, dim=1)
        weight_interleaved = torch.cat([stream1_weight_expanded, stream2_weight_expanded], dim=0)
        
        # Apply convolution
        if self.padding_mode != "zeros":
            interleaved_input = F.pad(interleaved_input, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            out_combined = F.conv2d(
                interleaved_input, weight_interleaved, None,
                self.stride, _pair(0), self.dilation, groups=2
            )
        else:
            out_combined = F.conv2d(
                interleaved_input, weight_interleaved, None,
                self.stride, self.padding, self.dilation, groups=2
            )
        
        # De-interleave outputs
        stream1_out = out_combined[:, :self.stream1_out_channels, :, :]
        stream2_out = out_combined[:, self.stream1_out_channels:, :, :]

        # ===== Process integrated pathway =====
        # Apply Linear Integration: integrated_out = W3·integrated + W1·stream1 + W2·stream2 + bias

        # Process previous integrated (if exists) using 1x1 conv with stride matching
        # IMPORTANT: integrated_weight is 1x1, but stride must match main conv for spatial alignment
        # Note: We apply bias at the end (after summing all contributions) for consistency
        if integrated_input is not None:
            integrated_from_prev = F.conv2d(
                integrated_input, self.integrated_weight, None,  # No bias here
                stride=self.stride, padding=0  # 1x1 conv, stride matches main conv
            )
        else:
            # First layer: no previous integrated stream
            integrated_from_prev = 0

        # Integration step: combine stream outputs using 1x1 convs (channel-wise mixing)
        integrated_from_s1 = F.conv2d(
            stream1_out, self.integration_from_stream1, None,
            stride=1, padding=0  # 1x1 conv, stride=1 (already spatially aligned)
        )
        integrated_from_s2 = F.conv2d(
            stream2_out, self.integration_from_stream2, None,
            stride=1, padding=0  # 1x1 conv, stride=1 (already spatially aligned)
        )

        # Combine all contributions to create integrated output
        integrated_out = integrated_from_prev + integrated_from_s1 + integrated_from_s2

        # Add bias at the end (ensures bias is applied in both first layer and later layers)
        if self.integrated_bias is not None:
            integrated_out = integrated_out + self.integrated_bias.view(1, -1, 1, 1)

        return stream1_out, stream2_out, integrated_out
    


class _LINormBase(nn.Module):
    """Common base for Linear Integration normalization - follows PyTorch's _NormBase pattern exactly."""
    
    _version = 2
    __constants__ = ["track_running_stats", "momentum", "eps", "stream1_num_features", "stream2_num_features", "integrated_num_features", "affine"]
    stream1_num_features: int
    stream2_num_features: int
    integrated_num_features: int
    eps: float
    momentum: Optional[float]
    affine: bool
    track_running_stats: bool
    # WARNING: stream1_weight, stream2_weight, stream1_bias, stream2_bias purposely not defined here.
    # Following PyTorch's pattern from https://github.com/pytorch/pytorch/issues/39670
    
    def __init__(
        self,
        stream1_num_features: int,
        stream2_num_features: int,
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
        self.stream1_num_features = stream1_num_features
        self.stream2_num_features = stream2_num_features
        self.integrated_num_features = integrated_num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        # Create parameters for all 3 pathways - exactly like _NormBase but for 3 streams
        if self.affine:
            self.stream1_weight = Parameter(torch.empty(stream1_num_features, **factory_kwargs))
            self.stream1_bias = Parameter(torch.empty(stream1_num_features, **factory_kwargs))
            self.stream2_weight = Parameter(torch.empty(stream2_num_features, **factory_kwargs))
            self.stream2_bias = Parameter(torch.empty(stream2_num_features, **factory_kwargs))
            self.integrated_weight = Parameter(torch.empty(integrated_num_features, **factory_kwargs))
            self.integrated_bias = Parameter(torch.empty(integrated_num_features, **factory_kwargs))
        else:
            self.register_parameter("stream1_weight", None)
            self.register_parameter("stream1_bias", None)
            self.register_parameter("stream2_weight", None)
            self.register_parameter("stream2_bias", None)
            self.register_parameter("integrated_weight", None)
            self.register_parameter("integrated_bias", None)
        
        # Create buffers for all 3 pathways - stream alternating pattern like PyTorch
        if self.track_running_stats:
            self.register_buffer(
                "stream1_running_mean", torch.zeros(stream1_num_features, **factory_kwargs)
            )
            self.register_buffer(
                "stream2_running_mean", torch.zeros(stream2_num_features, **factory_kwargs)
            )
            self.register_buffer(
                "integrated_running_mean", torch.zeros(integrated_num_features, **factory_kwargs)
            )
            self.register_buffer(
                "stream1_running_var", torch.ones(stream1_num_features, **factory_kwargs)
            )
            self.register_buffer(
                "stream2_running_var", torch.ones(stream2_num_features, **factory_kwargs)
            )
            self.register_buffer(
                "integrated_running_var", torch.ones(integrated_num_features, **factory_kwargs)
            )
            self.register_buffer(
                "num_batches_tracked",
                torch.tensor(
                    0,
                    dtype=torch.long,
                    **{k: v for k, v in factory_kwargs.items() if k != "dtype"},
                ),
            )
            self.stream1_running_mean: Optional[Tensor]
            self.stream2_running_mean: Optional[Tensor]
            self.integrated_running_mean: Optional[Tensor]
            self.stream1_running_var: Optional[Tensor]
            self.stream2_running_var: Optional[Tensor]
            self.integrated_running_var: Optional[Tensor]
            self.num_batches_tracked: Optional[Tensor]
        else:
            self.register_buffer("stream1_running_mean", None)
            self.register_buffer("stream2_running_mean", None)
            self.register_buffer("integrated_running_mean", None)
            self.register_buffer("stream1_running_var", None)
            self.register_buffer("stream2_running_var", None)
            self.register_buffer("integrated_running_var", None)
            self.register_buffer("num_batches_tracked", None)
        
        self.reset_parameters()
    
    def reset_running_stats(self) -> None:
        """Reset running statistics for all 3 pathways - exactly like _NormBase but for 3 streams."""
        if self.track_running_stats:
            # running_mean/running_var/num_batches... are registered at runtime depending
            # if self.track_running_stats is on
            self.stream1_running_mean.zero_()  # type: ignore[union-attr]
            self.stream1_running_var.fill_(1)  # type: ignore[union-attr]

            self.stream2_running_mean.zero_()  # type: ignore[union-attr]
            self.stream2_running_var.fill_(1)  # type: ignore[union-attr]

            self.integrated_running_mean.zero_()  # type: ignore[union-attr]
            self.integrated_running_var.fill_(1)  # type: ignore[union-attr]

            # Shared batch tracking
            self.num_batches_tracked.zero_()  # type: ignore[union-attr,operator]

    def reset_parameters(self) -> None:
        """Reset parameters for all 3 pathways - exactly like _NormBase but for 3 streams."""
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.stream1_weight)
            init.zeros_(self.stream1_bias)
            init.ones_(self.stream2_weight)
            init.zeros_(self.stream2_bias)
            init.ones_(self.integrated_weight)
            init.zeros_(self.integrated_bias)
    
    def _check_input_dim(self, input):
        """Check input dimensions - to be implemented by subclasses."""
        raise NotImplementedError
    
    def extra_repr(self):
        """String representation - updated for 3 streams."""
        return (
            "stream1_num_features={stream1_num_features}, stream2_num_features={stream2_num_features}, "
            "integrated_num_features={integrated_num_features}, "
            "eps={eps}, momentum={momentum}, affine={affine}, "
            "track_running_stats={track_running_stats}".format(**self.__dict__)
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
        stream1_num_features: int,
        stream2_num_features: int,
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
            stream1_num_features, stream2_num_features, integrated_num_features, eps, momentum, affine, track_running_stats, **factory_kwargs
        )
    
    def forward(self, stream1_input: Tensor, stream2_input: Tensor, integrated_input: Optional[Tensor] = None) -> tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass - implements the exact same algorithm as _BatchNorm.forward() for 3 streams.
        """
        # Check input dimensions for all inputs
        self._check_input_dim(stream1_input)
        self._check_input_dim(stream2_input)
        if integrated_input is not None:
            self._check_input_dim(integrated_input)

        # Process stream1 pathway using the exact _BatchNorm algorithm
        stream1_out = self._forward_single_pathway(
            stream1_input,
            self.stream1_running_mean,
            self.stream1_running_var,
            self.stream1_weight,
            self.stream1_bias,
            self.num_batches_tracked,
        )

        # Process stream2 pathway using the exact _BatchNorm algorithm
        stream2_out = self._forward_single_pathway(
            stream2_input,
            self.stream2_running_mean,
            self.stream2_running_var,
            self.stream2_weight,
            self.stream2_bias,
            self.num_batches_tracked,
        )

        # Process integrated pathway using the exact _BatchNorm algorithm (if exists)
        if integrated_input is not None:
            integrated_out = self._forward_single_pathway(
                integrated_input,
                self.integrated_running_mean,
                self.integrated_running_var,
                self.integrated_weight,
                self.integrated_bias,
                self.num_batches_tracked,
            )
        else:
            integrated_out = None

        return stream1_out, stream2_out, integrated_out
    
    def forward_stream1(self, stream1_input: Tensor) -> Tensor:
        """Forward pass through stream1 pathway only."""
        self._check_input_dim(stream1_input)
        return self._forward_single_pathway(
            stream1_input,
            self.stream1_running_mean,
            self.stream1_running_var,
            self.stream1_weight,
            self.stream1_bias,
            self.num_batches_tracked,
        )
    
    def forward_stream2(self, stream2_input: Tensor) -> Tensor:
        """Forward pass through stream2 pathway only."""
        self._check_input_dim(stream2_input)
        return self._forward_single_pathway(
            stream2_input,
            self.stream2_running_mean,
            self.stream2_running_var,
            self.stream2_weight,
            self.stream2_bias,
            self.num_batches_tracked,
        )
    
    def _forward_single_pathway(
        self,
        input: Tensor,
        running_mean: Optional[Tensor],
        running_var: Optional[Tensor],
        weight: Optional[Tensor],
        bias: Optional[Tensor],
        num_batches_tracked: Optional[Tensor],
    ) -> Tensor:
        """
        Forward pass for a single pathway - exact copy of _BatchNorm.forward() algorithm
        """
        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if num_batches_tracked is not None:  # type: ignore[has-type]
                num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (running_mean is None) and (running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
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
    r"""Applies Linear Integration Batch Normalization over 3 4D inputs.

    This layer applies Batch Normalization separately to stream1, stream2, and integrated pathways,
    following PyTorch's BatchNorm2d pattern exactly but extended for 3-stream processing.
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
        stream1_num_features: :math:`C` from expected stream1 input of size :math:`(N, C, H, W)`
        stream2_num_features: :math:`C` from expected stream2 input of size :math:`(N, C, H, W)`
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
        - Stream1 Input: :math:`(N, C_{stream1}, H, W)`
        - Stream2 Input: :math:`(N, C_{stream2}, H, W)`
        - Stream1 Output: :math:`(N, C_{stream1}, H, W)` (same shape as stream1 input)
        - Stream2 Output: :math:`(N, C_{stream2}, H, W)` (same shape as stream2 input)

    Examples::

        >>> # With Learnable Parameters
        >>> m = LIBatchNorm2d(100, 50, 75)
        >>> # Without Learnable Parameters
        >>> m = LIBatchNorm2d(100, 50, 75, affine=False)
        >>> stream1_input = torch.randn(20, 100, 35, 45)
        >>> stream2_input = torch.randn(20, 50, 35, 45)
        >>> integrated_input = torch.randn(20, 75, 35, 45)
        >>> stream1_output, stream2_output, integrated_output = m(stream1_input, stream2_input, integrated_input)
    """
    
    def _check_input_dim(self, input):
        """Check input dimensions - same as BatchNorm2d."""
        if input.dim() != 4:
            raise ValueError(f"expected 4D input (got {input.dim()}D input)")