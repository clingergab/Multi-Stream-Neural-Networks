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
from torch.nn.modules.utils import _pair, _reverse_repeat_tuple
from typing import Optional, Union


class _MCConvNd(nn.Module):
    """
    Multi-Channel convolution base class - follows PyTorch's _ConvNd pattern exactly.
    
    This is the base class for multi-channel convolution layers, mirroring PyTorch's
    _ConvNd but adapted for dual-stream (color/brightness) processing.
    """
    
    __constants__ = [
        "stride", "padding", "dilation", "groups", "padding_mode", "output_padding",
        "color_in_channels", "brightness_in_channels", 
        "color_out_channels", "brightness_out_channels", "kernel_size",
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
    transposed: bool
    output_padding: tuple[int, ...]
    groups: int
    padding_mode: str
    color_weight: Tensor
    brightness_weight: Tensor
    color_bias: Optional[Tensor]
    brightness_bias: Optional[Tensor]
    _reversed_padding_repeated_twice: list[int]
    
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
        
        # Create weight parameters for both pathways
        if transposed:
            self.color_weight = Parameter(
                torch.empty(
                    (color_in_channels, color_out_channels // groups, *kernel_size),
                    **factory_kwargs,
                )
            )
            self.brightness_weight = Parameter(
                torch.empty(
                    (brightness_in_channels, brightness_out_channels // groups, *kernel_size),
                    **factory_kwargs,
                )
            )
        else:
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
        if self.output_padding != (0,) * len(self.output_padding):
            s += ", output_padding={output_padding}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.color_bias is None:
            s += ", bias=False"
        if self.padding_mode != "zeros":
            s += ", padding_mode={padding_mode}"
        return s.format(**self.__dict__)
    
    def __setstate__(self, state):
        """Handle backward compatibility like _ConvNd."""
        super().__setstate__(state)
        if not hasattr(self, "padding_mode"):
            self.padding_mode = "zeros"


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
            False,  # transposed (Conv2d is not transposed)
            _pair(0),  # output_padding (not used for Conv2d)
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
        """Forward pass through both convolution streams using grouped convolution."""
        # Combine inputs along channel dimension
        input_combined = torch.cat([color_input, brightness_input], dim=1)
        # Combine weights along output channel dimension
        weight_combined = torch.cat([self.color_weight, self.brightness_weight], dim=0)
        # Combine biases if present
        bias_combined = None
        if self.color_bias is not None:
            bias_combined = torch.cat([self.color_bias, self.brightness_bias], dim=0)
        # Apply convolution with grouping
        # Determine effective groups = original groups * 2 (one per stream)
        effective_groups = self.groups * 2
        if self.padding_mode != "zeros":
            input_combined = F.pad(input_combined, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            out_combined = F.conv2d(
                input_combined,
                weight_combined,
                bias_combined,
                self.stride,
                _pair(0),
                self.dilation,
                effective_groups
            )
        else:
            out_combined = F.conv2d(
                input_combined,
                weight_combined,
                bias_combined,
                self.stride,
                self.padding,
                self.dilation,
                effective_groups
            )
        # Split combined output back into separate streams
        color_out, brightness_out = out_combined.chunk(2, dim=1)
        return color_out, brightness_out
    
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
        
        # Create buffers for both pathways - stream alternating pattern like PyTorch
        if self.track_running_stats:
            self.register_buffer(
                "color_running_mean", torch.zeros(color_num_features, **factory_kwargs)
            )
            self.register_buffer(
                "brightness_running_mean", torch.zeros(brightness_num_features, **factory_kwargs)
            )
            self.register_buffer(
                "color_running_var", torch.ones(color_num_features, **factory_kwargs)
            )
            self.register_buffer(
                "brightness_running_var", torch.ones(brightness_num_features, **factory_kwargs)
            )
            self.register_buffer(
                "num_batches_tracked",
                torch.tensor(
                    0,
                    dtype=torch.long,
                    **{k: v for k, v in factory_kwargs.items() if k != "dtype"},
                ),
            )
            self.color_running_mean: Optional[Tensor]
            self.brightness_running_mean: Optional[Tensor]
            self.color_running_var: Optional[Tensor]
            self.brightness_running_var: Optional[Tensor]
            self.num_batches_tracked: Optional[Tensor]
        else:
            self.register_buffer("color_running_mean", None)
            self.register_buffer("brightness_running_mean", None)
            self.register_buffer("color_running_var", None)
            self.register_buffer("brightness_running_var", None)
            self.register_buffer("num_batches_tracked", None)
        
        self.reset_parameters()
    
    def reset_running_stats(self) -> None:
        """Reset running statistics for both pathways - exactly like _NormBase but dual."""
        if self.track_running_stats:
            # running_mean/running_var/num_batches... are registered at runtime depending
            # if self.track_running_stats is on
            self.color_running_mean.zero_()  # type: ignore[union-attr]
            self.color_running_var.fill_(1)  # type: ignore[union-attr]
            
            self.brightness_running_mean.zero_()  # type: ignore[union-attr]
            self.brightness_running_var.fill_(1)  # type: ignore[union-attr]
            
            # Shared batch tracking
            self.num_batches_tracked.zero_()  # type: ignore[union-attr,operator]
    
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
            self.num_batches_tracked,
        )
        
        # Process brightness pathway using the exact _BatchNorm algorithm
        brightness_out = self._forward_single_pathway(
            brightness_input,
            self.brightness_running_mean,
            self.brightness_running_var,
            self.brightness_weight,
            self.brightness_bias,
            self.num_batches_tracked,
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
            self.num_batches_tracked,
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


class MCBatchNorm2d(_MCBatchNorm):
    r"""Applies Multi-Channel Batch Normalization over dual 4D inputs.

    This layer applies Batch Normalization separately to color and brightness pathways,
    following PyTorch's BatchNorm2d pattern exactly but extended for dual-stream processing.
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
        color_num_features: :math:`C` from expected color input of size :math:`(N, C, H, W)`
        brightness_num_features: :math:`C` from expected brightness input of size :math:`(N, C, H, W)`
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
        - Color Input: :math:`(N, C_{color}, H, W)`
        - Brightness Input: :math:`(N, C_{brightness}, H, W)`
        - Color Output: :math:`(N, C_{color}, H, W)` (same shape as color input)
        - Brightness Output: :math:`(N, C_{brightness}, H, W)` (same shape as brightness input)

    Examples::

        >>> # With Learnable Parameters
        >>> m = MCBatchNorm2d(100, 50)
        >>> # Without Learnable Parameters
        >>> m = MCBatchNorm2d(100, 50, affine=False)
        >>> color_input = torch.randn(20, 100, 35, 45)
        >>> brightness_input = torch.randn(20, 50, 35, 45)
        >>> color_output, brightness_output = m(color_input, brightness_input)
    """
    
    def _check_input_dim(self, input):
        """Check input dimensions - same as BatchNorm2d."""
        if input.dim() != 4:
            raise ValueError(f"expected 4D input (got {input.dim()}D input)")
