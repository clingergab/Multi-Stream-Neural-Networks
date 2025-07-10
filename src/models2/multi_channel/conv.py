"""
Multi-Channel convolution layers for separate color and brightness processing.
"""

import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F, init
from torch.nn.parameter import Parameter
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair
from typing import Optional, Union


class _MCConvNd(nn.Module):
    """
    Multi-Channel convolution base class - follows PyTorch's _ConvNd pattern exactly.
    
    This is the base class for multi-channel convolution layers, mirroring PyTorch's
    _ConvNd but adapted for dual-stream (color/brightness) processing.
    """
    
    __constants__ = [
        "stride", "padding", "dilation", "groups", "padding_mode",
        "color_in_channels", "brightness_in_channels", "color_out_channels", "brightness_out_channels", "kernel_size",
    ]
    __annotations__ = {
        "color_bias": Optional[torch.Tensor],
        "brightness_bias": Optional[torch.Tensor],
    }
    
    def _conv_forward(self, color_input: Tensor, brightness_input: Tensor, 
                     color_weight: Tensor, brightness_weight: Tensor,
                     color_bias: Optional[Tensor], brightness_bias: Optional[Tensor]) -> tuple[Tensor, Tensor]:
        """Abstract method to be implemented by subclasses."""
        ...
    
    color_in_channels: int
    brightness_in_channels: int
    color_out_channels: int
    brightness_out_channels: int
    kernel_size: tuple[int, ...]
    stride: tuple[int, ...]
    padding: Union[str, tuple[int, ...]]
    dilation: tuple[int, ...]
    groups: int
    padding_mode: str
    color_weight: Tensor
    brightness_weight: Tensor
    color_bias: Optional[Tensor]
    brightness_bias: Optional[Tensor]
    
    def __init__(
        self,
        color_in_channels: int,
        brightness_in_channels: int,
        color_out_channels: int,
        brightness_out_channels: int,
        kernel_size: tuple[int, ...],
        stride: tuple[int, ...],
        padding: Union[str, tuple[int, ...]],
        dilation: tuple[int, ...],
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
        if color_in_channels % groups != 0:
            raise ValueError("color_in_channels must be divisible by groups")
        if brightness_in_channels % groups != 0:
            raise ValueError("brightness_in_channels must be divisible by groups")
        if color_out_channels % groups != 0:
            raise ValueError("color_out_channels must be divisible by groups")
        if brightness_out_channels % groups != 0:
            raise ValueError("brightness_out_channels must be divisible by groups")
        
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
        self.color_in_channels = color_in_channels
        self.brightness_in_channels = brightness_in_channels
        self.color_out_channels = color_out_channels
        self.brightness_out_channels = brightness_out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        
        # Create weight parameters for both pathways
        self.color_weight = Parameter(
            torch.empty(
                (color_out_channels, color_in_channels // groups, *kernel_size),
                **factory_kwargs,
            )
        )
        self.brightness_weight = Parameter(
            torch.empty(
                (brightness_out_channels, brightness_in_channels // groups, *kernel_size),
                **factory_kwargs,
            )
        )
        
        # Create bias parameters for both pathways
        if bias:
            self.color_bias = Parameter(torch.empty(color_out_channels, **factory_kwargs))
            self.brightness_bias = Parameter(torch.empty(brightness_out_channels, **factory_kwargs))
        else:
            self.register_parameter("color_bias", None)
            self.register_parameter("brightness_bias", None)
        
        # Expose primary parameters for ResNet compatibility (point to color pathway)
        self.weight = self.color_weight
        self.bias = self.color_bias
        
        # Handle padding mode exactly like _ConvNd
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
            from torch.nn.modules.utils import _reverse_repeat_tuple
            self._reversed_padding_repeated_twice = _reverse_repeat_tuple(
                self.padding, 2
            )
        
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        """Reset parameters exactly like _ConvNd."""
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        
        # Initialize color pathway weights
        init.kaiming_uniform_(self.color_weight, a=math.sqrt(5))
        if self.color_bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.color_weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.color_bias, -bound, bound)
        
        # Initialize brightness pathway weights
        init.kaiming_uniform_(self.brightness_weight, a=math.sqrt(5))
        if self.brightness_bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.brightness_weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.brightness_bias, -bound, bound)
    
    def extra_repr(self):
        """String representation exactly like _ConvNd."""
        s = (
            "color_in_channels={color_in_channels}, brightness_in_channels={brightness_in_channels}, "
            "color_out_channels={color_out_channels}, brightness_out_channels={brightness_out_channels}, "
            "kernel_size={kernel_size}, stride={stride}"
        )
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.color_bias is None:
            s += ", bias=False"
        if self.padding_mode != "zeros":
            s += ", padding_mode={padding_mode}"
        return s.format(**self.__dict__)


class MCConv2d(_MCConvNd):
    """
    Multi-Channel 2D Convolution layer - follows PyTorch's Conv2d pattern exactly.
    
    This layer processes color and brightness streams separately using dual weight/bias
    parameters, maintaining full compatibility with PyTorch's Conv2d interface.
    """
    
    def __init__(
        self,
        color_in_channels: int,
        brightness_in_channels: int,
        color_out_channels: int,
        brightness_out_channels: int,
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
            color_in_channels,
            brightness_in_channels,
            color_out_channels,
            brightness_out_channels,
            kernel_size_,
            stride_,
            padding_,
            dilation_,
            groups,
            bias,
            padding_mode,
            **factory_kwargs,
        )
    
    def _conv_forward(self, color_input: Tensor, brightness_input: Tensor,
                     color_weight: Tensor, brightness_weight: Tensor,
                     color_bias: Optional[Tensor], brightness_bias: Optional[Tensor]) -> tuple[Tensor, Tensor]:
        """Forward pass implementation exactly like Conv2d._conv_forward."""
        # Process color pathway
        if self.padding_mode != "zeros":
            color_out = F.conv2d(
                F.pad(
                    color_input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                color_weight,
                color_bias,
                self.stride,
                _pair(0),
                self.dilation,
                self.groups,
            )
        else:
            color_out = F.conv2d(
                color_input, color_weight, color_bias, self.stride, self.padding, self.dilation, self.groups
            )
        
        # Process brightness pathway
        if self.padding_mode != "zeros":
            brightness_out = F.conv2d(
                F.pad(
                    brightness_input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                brightness_weight,
                brightness_bias,
                self.stride,
                _pair(0),
                self.dilation,
                self.groups,
            )
        else:
            brightness_out = F.conv2d(
                brightness_input, brightness_weight, brightness_bias, self.stride, self.padding, self.dilation, self.groups
            )
        
        return color_out, brightness_out
    
    def forward(self, color_input: Tensor, brightness_input: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass through both convolution streams."""
        return self._conv_forward(color_input, brightness_input, 
                                 self.color_weight, self.brightness_weight,
                                 self.color_bias, self.brightness_bias)
    
    def forward_color(self, color_input: Tensor) -> Tensor:
        """Forward pass through color stream only."""
        if self.padding_mode != "zeros":
            return F.conv2d(
                F.pad(
                    color_input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                self.color_weight,
                self.color_bias,
                self.stride,
                _pair(0),
                self.dilation,
                self.groups,
            )
        return F.conv2d(
            color_input, self.color_weight, self.color_bias, self.stride, self.padding, self.dilation, self.groups
        )
    
    def forward_brightness(self, brightness_input: Tensor) -> Tensor:
        """Forward pass through brightness stream only."""
        if self.padding_mode != "zeros":
            return F.conv2d(
                F.pad(
                    brightness_input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                self.brightness_weight,
                self.brightness_bias,
                self.stride,
                _pair(0),
                self.dilation,
                self.groups,
            )
        return F.conv2d(
            brightness_input, self.brightness_weight, self.brightness_bias, self.stride, self.padding, self.dilation, self.groups
        )


class _MCNormBase(nn.Module):
    """Common base for Multi-Channel normalization - follows PyTorch's _NormBase pattern exactly."""
    
    _version = 2
    __constants__ = ["track_running_stats", "momentum", "eps", "color_num_features", "brightness_num_features", "affine"]
    color_num_features: int
    brightness_num_features: int
    eps: float
    momentum: Optional[float]
    affine: bool
    track_running_stats: bool
    # WARNING: color_weight, brightness_weight, color_bias, brightness_bias purposely not defined here.
    # Following PyTorch's pattern from https://github.com/pytorch/pytorch/issues/39670
    
    def __init__(
        self,
        color_num_features: int,
        brightness_num_features: int,
        eps: float = 1e-5,
        momentum: Optional[float] = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.color_num_features = color_num_features
        self.brightness_num_features = brightness_num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        # Create parameters for both pathways - exactly like _NormBase but dual
        if self.affine:
            self.color_weight = Parameter(torch.empty(color_num_features, **factory_kwargs))
            self.color_bias = Parameter(torch.empty(color_num_features, **factory_kwargs))
            self.brightness_weight = Parameter(torch.empty(brightness_num_features, **factory_kwargs))
            self.brightness_bias = Parameter(torch.empty(brightness_num_features, **factory_kwargs))
        else:
            self.register_parameter("color_weight", None)
            self.register_parameter("color_bias", None)
            self.register_parameter("brightness_weight", None)
            self.register_parameter("brightness_bias", None)
        
        # Create buffers for both pathways - exactly like _NormBase but dual
        if self.track_running_stats:
            # Color pathway buffers
            self.register_buffer("color_running_mean", torch.zeros(color_num_features, **factory_kwargs))
            self.register_buffer("color_running_var", torch.ones(color_num_features, **factory_kwargs))
            self.color_running_mean: Optional[Tensor]
            self.color_running_var: Optional[Tensor]
            self.register_buffer("color_num_batches_tracked", torch.tensor(0, dtype=torch.long, **{k: v for k, v in factory_kwargs.items() if k != "dtype"}))
            self.color_num_batches_tracked: Optional[Tensor]
            
            # Brightness pathway buffers
            self.register_buffer("brightness_running_mean", torch.zeros(brightness_num_features, **factory_kwargs))
            self.register_buffer("brightness_running_var", torch.ones(brightness_num_features, **factory_kwargs))
            self.brightness_running_mean: Optional[Tensor]
            self.brightness_running_var: Optional[Tensor]
            self.register_buffer("brightness_num_batches_tracked", torch.tensor(0, dtype=torch.long, **{k: v for k, v in factory_kwargs.items() if k != "dtype"}))
            self.brightness_num_batches_tracked: Optional[Tensor]
        else:
            self.register_buffer("color_running_mean", None)
            self.register_buffer("color_running_var", None)
            self.register_buffer("color_num_batches_tracked", None)
            self.register_buffer("brightness_running_mean", None)
            self.register_buffer("brightness_running_var", None)
            self.register_buffer("brightness_num_batches_tracked", None)
        
        # Expose primary parameters for ResNet compatibility (point to color pathway)
        # This follows the same pattern as PyTorch but gives us backward compatibility
        self.weight = self.color_weight if affine else None
        self.bias = self.color_bias if affine else None
        self.running_mean = self.color_running_mean if track_running_stats else None
        self.running_var = self.color_running_var if track_running_stats else None
        self.num_batches_tracked = self.color_num_batches_tracked if track_running_stats else None
        
        # For backward compatibility, expose num_features as color_num_features
        self.num_features = self.color_num_features
        
        self.reset_parameters()
    
    def reset_running_stats(self) -> None:
        """Reset running statistics for both pathways - exactly like _NormBase but dual."""
        if self.track_running_stats:
            # running_mean/running_var/num_batches... are registered at runtime depending
            # if self.track_running_stats is on
            self.color_running_mean.zero_()  # type: ignore[union-attr]
            self.color_running_var.fill_(1)  # type: ignore[union-attr]
            self.color_num_batches_tracked.zero_()  # type: ignore[union-attr,operator]
            
            self.brightness_running_mean.zero_()  # type: ignore[union-attr]
            self.brightness_running_var.fill_(1)  # type: ignore[union-attr]
            self.brightness_num_batches_tracked.zero_()  # type: ignore[union-attr,operator]
    
    def reset_parameters(self) -> None:
        """Reset parameters for both pathways - exactly like _NormBase but dual."""
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.color_weight)
            init.zeros_(self.color_bias)
            init.ones_(self.brightness_weight)
            init.zeros_(self.brightness_bias)
    
    def _check_input_dim(self, input):
        """Check input dimensions - to be implemented by subclasses."""
        raise NotImplementedError
    
    def extra_repr(self):
        """String representation - updated for dual channels."""
        return (
            "color_num_features={color_num_features}, brightness_num_features={brightness_num_features}, "
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
            # Handle both pathways
            for pathway in ["color", "brightness"]:
                num_batches_tracked_key = prefix + f"{pathway}_num_batches_tracked"
                if num_batches_tracked_key not in state_dict:
                    pathway_attr = getattr(self, f"{pathway}_num_batches_tracked")
                    state_dict[num_batches_tracked_key] = (
                        pathway_attr
                        if pathway_attr is not None
                        and pathway_attr.device != torch.device("meta")
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


class _MCBatchNorm(_MCNormBase):
    """Multi-Channel BatchNorm - follows PyTorch's _BatchNorm pattern exactly."""
    
    def __init__(
        self,
        color_num_features: int,
        brightness_num_features: int,
        eps: float = 1e-5,
        momentum: Optional[float] = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            color_num_features, brightness_num_features, eps, momentum, affine, track_running_stats, **factory_kwargs
        )
    
    def forward(self, color_input: Tensor, brightness_input: Tensor) -> tuple[Tensor, Tensor]:
        """
        Forward pass - implements the exact same algorithm as _BatchNorm.forward() for dual streams.
        """
        # Check input dimensions for both inputs
        self._check_input_dim(color_input)
        self._check_input_dim(brightness_input)
        
        # Process color pathway using the exact _BatchNorm algorithm
        color_out = self._forward_single_pathway(
            color_input,
            self.color_running_mean,
            self.color_running_var,
            self.color_weight,
            self.color_bias,
            self.color_num_batches_tracked,
        )
        
        # Process brightness pathway using the exact _BatchNorm algorithm
        brightness_out = self._forward_single_pathway(
            brightness_input,
            self.brightness_running_mean,
            self.brightness_running_var,
            self.brightness_weight,
            self.brightness_bias,
            self.brightness_num_batches_tracked,
        )
        
        return color_out, brightness_out
    
    def forward_color(self, color_input: Tensor) -> Tensor:
        """Forward pass through color pathway only."""
        self._check_input_dim(color_input)
        return self._forward_single_pathway(
            color_input,
            self.color_running_mean,
            self.color_running_var,
            self.color_weight,
            self.color_bias,
            self.color_num_batches_tracked,
        )
    
    def forward_brightness(self, brightness_input: Tensor) -> Tensor:
        """Forward pass through brightness pathway only."""
        self._check_input_dim(brightness_input)
        return self._forward_single_pathway(
            brightness_input,
            self.brightness_running_mean,
            self.brightness_running_var,
            self.brightness_weight,
            self.brightness_bias,
            self.brightness_num_batches_tracked,
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


class MCBatchNorm2d(_MCBatchNorm):
    """
    Multi-Channel 2D Batch Normalization - follows PyTorch's BatchNorm2d pattern exactly.
    
    This is the exact equivalent of nn.BatchNorm2d but for dual-stream processing.
    """
    
    def _check_input_dim(self, input):
        """Check input dimensions - same as BatchNorm2d."""
        if input.dim() != 4:
            raise ValueError(f"expected 4D input (got {input.dim()}D input)")
