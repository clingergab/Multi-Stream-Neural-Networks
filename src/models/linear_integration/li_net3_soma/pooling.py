"""
Linear Integration pooling layers for N-stream processing.
"""

from typing import Optional
from torch import Tensor
from torch.nn import functional as F
from torch.nn.common_types import _size_2_t, _size_any_t, _size_2_opt_t, _size_any_opt_t
from torch.nn.modules.utils import _pair

from torch.nn import Module


__all__ = [
    "LIMaxPool2d",
    "LIAdaptiveAvgPool2d",
]


class _LIMaxPoolNd(Module):
    """
    Linear Integration MaxPool base class - follows PyTorch's _MaxPoolNd pattern exactly.

    This is the base class for N-stream max pooling layers, mirroring PyTorch's
    _MaxPoolNd but adapted for N-stream (streams + integrated) processing.
    """
    
    __constants__ = [
        "kernel_size",
        "stride", 
        "padding",
        "dilation",
        "return_indices",
        "ceil_mode",
    ]
    return_indices: bool
    ceil_mode: bool

    def __init__(
        self,
        kernel_size: _size_any_t,
        stride: Optional[_size_any_t] = None,
        padding: _size_any_t = 0,
        dilation: _size_any_t = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if (stride is not None) else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def extra_repr(self) -> str:
        return (
            "kernel_size={kernel_size}, stride={stride}, padding={padding}"
            ", dilation={dilation}, ceil_mode={ceil_mode}".format(**self.__dict__)
        )


class LIMaxPool2d(_LIMaxPoolNd):
    r"""Linear Integration 2D max pooling - follows PyTorch's MaxPool2d pattern exactly.

    Applies a 2D max pooling over N input signals (stream inputs and integrated stream)
    composed of several input planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, H, W)`,
    output :math:`(N, C, H_{out}, W_{out})` and :attr:`kernel_size` :math:`(kH, kW)`
    can be precisely described as:

    .. math::
        \begin{aligned}
            out(N_i, C_j, h, w) ={} & \max_{m=0, \ldots, kH-1} \max_{n=0, \ldots, kW-1} \\
                                    & \text{input}(N_i, C_j, \text{stride[0]} \times h + m,
                                                   \text{stride[1]} \times w + n)
        \end{aligned}

    This operation is applied independently to all N streams and the integrated stream.

    If :attr:`padding` is non-zero, then the input is implicitly padded with negative infinity on both sides
    for :attr:`padding` number of points. :attr:`dilation` controls the spacing between the kernel points.
    It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.

    Note:
        When ceil_mode=True, sliding windows are allowed to go off-bounds if they start within the left padding
        or the input. Sliding windows that would start in the right padded region are ignored.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    Args:
        kernel_size: the size of the window to take a max over
        stride: the stride of the window. Default value is :attr:`kernel_size`
        padding: Implicit negative infinity padding to be added on both sides
        dilation: a parameter that controls the stride of elements in the window
        return_indices: if ``True``, will return the max indices along with the outputs.
                        Useful for :class:`torch.nn.MaxUnpool2d` later
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})` for both stream1 and stream2
        - Output: :math:`(N, C, H_{out}, W_{out})` or :math:`(C, H_{out}, W_{out})` for both stream1 and stream2, where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in} + 2 * \text{padding[0]} - \text{dilation[0]}
                    \times (\text{kernel\_size[0]} - 1) - 1}{\text{stride[0]}} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in} + 2 * \text{padding[1]} - \text{dilation[1]}
                    \times (\text{kernel\_size[1]} - 1) - 1}{\text{stride[1]}} + 1\right\rfloor

    Examples::

        >>> # pool of square window of size=3, stride=2 for 3 streams
        >>> m = LIMaxPool2d(3, stride=2)
        >>> stream_inputs = [
        ...     torch.randn(20, 16, 50, 32),  # stream 0
        ...     torch.randn(20, 8, 50, 32),   # stream 1
        ...     torch.randn(20, 12, 50, 32)   # stream 2
        ... ]
        >>> integrated_input = torch.randn(20, 16, 50, 32)
        >>> stream_outputs, integrated_output = m(stream_inputs, integrated_input)

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    kernel_size: _size_2_t
    stride: _size_2_t
    padding: _size_2_t
    dilation: _size_2_t

    def forward(self, stream_inputs: list[Tensor], integrated_input: Optional[Tensor] = None) -> tuple[list[Tensor], Tensor]:
        """
        Forward pass through N-stream max pooling layers.

        Args:
            stream_inputs: List of stream input tensors [batch_size, channels, height, width]
            integrated_input: Integrated input tensor [batch_size, channels, height, width] or None

        Returns:
            Tuple of (stream_outputs, integrated_output) where stream_outputs is a list of tensors
        """
        # Apply max pooling to all stream inputs
        stream_outputs = []
        for stream_input in stream_inputs:
            stream_output = F.max_pool2d(
                stream_input,
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
                ceil_mode=self.ceil_mode,
                return_indices=self.return_indices,
            )
            stream_outputs.append(stream_output)

        # Apply max pooling to integrated (if exists)
        if integrated_input is not None:
            integrated_output = F.max_pool2d(
                integrated_input,
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
                ceil_mode=self.ceil_mode,
                return_indices=self.return_indices,
            )
        else:
            integrated_output = None

        return stream_outputs, integrated_output

    def forward_stream(self, stream_idx: int, stream_input: Tensor) -> Tensor:
        """Apply max pooling to a single stream pathway."""
        return F.max_pool2d(
            stream_input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            ceil_mode=self.ceil_mode,
            return_indices=self.return_indices,
        )


class _LIAdaptiveAvgPoolNd(Module):
    """
    Linear Integration Adaptive Average Pool base class - follows PyTorch's _AdaptiveAvgPoolNd pattern exactly.

    This is the base class for N-stream adaptive average pooling layers, mirroring PyTorch's
    _AdaptiveAvgPoolNd but adapted for N-stream (streams + integrated) processing.
    """
    
    __constants__ = ["output_size"]

    def __init__(self, output_size: _size_any_opt_t) -> None:
        super().__init__()
        self.output_size = output_size

    def extra_repr(self) -> str:
        return f"output_size={self.output_size}"


class LIAdaptiveAvgPool2d(_LIAdaptiveAvgPoolNd):
    r"""Linear Integration 2D adaptive average pooling - follows PyTorch's AdaptiveAvgPool2d pattern exactly.

    Applies a 2D adaptive average pooling over N input signals (stream inputs and integrated stream)
    composed of several input planes.

    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.

    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H.
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.

    Shape:
        - Input: List of :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})` for all streams.
        - Output: List of :math:`(N, C, S_{0}, S_{1})` or :math:`(C, S_{0}, S_{1})` for all streams, where
          :math:`S=\text{output\_size}`.

    Examples::

        >>> # target output size of 5x7 for 3 streams
        >>> m = LIAdaptiveAvgPool2d((5, 7))
        >>> stream_inputs = [torch.randn(1, 64, 8, 9) for _ in range(3)]
        >>> stream_outputs, _ = m(stream_inputs, None)
        >>> # target output size of 7x7 (square)
        >>> m = LIAdaptiveAvgPool2d(7)
        >>> stream_inputs = [torch.randn(1, 64, 10, 9) for _ in range(3)]
        >>> integrated_input = torch.randn(1, 64, 10, 9)
        >>> stream_outputs, integrated_output = m(stream_inputs, integrated_input)

    """

    output_size: _size_2_opt_t

    def forward(self, stream_inputs: list[Tensor], integrated_input: Optional[Tensor] = None) -> tuple[list[Tensor], Tensor]:
        """
        Forward pass through N-stream adaptive average pooling layers.

        Args:
            stream_inputs: List of stream input tensors [batch_size, channels, height, width]
            integrated_input: Integrated input tensor [batch_size, channels, height, width] or None

        Returns:
            Tuple of (stream_outputs, integrated_output) where stream_outputs is a list of tensors
        """
        # Apply adaptive average pooling to all stream inputs
        stream_outputs = [F.adaptive_avg_pool2d(stream_input, self.output_size) for stream_input in stream_inputs]

        # Apply adaptive average pooling to integrated (if exists)
        if integrated_input is not None:
            integrated_output = F.adaptive_avg_pool2d(integrated_input, self.output_size)
        else:
            integrated_output = None

        return stream_outputs, integrated_output
