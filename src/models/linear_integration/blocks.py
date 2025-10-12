"""
Linear Integration ResNet blocks for building LINet architectures.
"""

from typing import Callable, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
from .conv import LIConv2d, LIBatchNorm2d
from .container import LIReLU


def li_conv3x3(
    stream1_in_planes: int,
    stream2_in_planes: int,
    integrated_in_planes: int,
    stream1_out_planes: int,
    stream2_out_planes: int,
    integrated_out_planes: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1
) -> LIConv2d:
    """3x3 linear integration convolution with padding."""
    return LIConv2d(
        stream1_in_planes,
        stream2_in_planes,
        integrated_in_planes,
        stream1_out_planes,
        stream2_out_planes,
        integrated_out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        groups=groups,
        bias=False,
    )


def li_conv1x1(
    stream1_in_planes: int,
    stream2_in_planes: int,
    integrated_in_planes: int,
    stream1_out_planes: int,
    stream2_out_planes: int,
    integrated_out_planes: int,
    stride: int = 1
) -> LIConv2d:
    """1x1 linear integration convolution."""
    return LIConv2d(
        stream1_in_planes,
        stream2_in_planes,
        integrated_in_planes,
        stream1_out_planes,
        stream2_out_planes,
        integrated_out_planes,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class LIBasicBlock(nn.Module):
    """
    Linear Integration version of ResNet BasicBlock.

    Uses unified LIConv2d neurons that process 3 streams simultaneously.
    Integration happens INSIDE LIConv2d during convolution (not as separate step).
    """

    expansion: int = 1

    def __init__(
        self,
        stream1_inplanes: int,
        stream2_inplanes: int,
        integrated_inplanes: int,
        stream1_planes: int,
        stream2_planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = LIBatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("LIBasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in LIBasicBlock")

        # Store channel counts for output
        self.stream1_outplanes = stream1_planes * self.expansion
        self.stream2_outplanes = stream2_planes * self.expansion
        self.integrated_outplanes = stream1_planes * self.expansion  # Same as stream1

        # Use unified LI neurons instead of MC neurons
        # Integration happens inside LIConv2d!
        self.conv1 = li_conv3x3(
            stream1_inplanes, stream2_inplanes, integrated_inplanes,
            stream1_planes, stream2_planes, stream1_planes,
            stride=stride
        )
        self.bn1 = norm_layer(stream1_planes, stream2_planes, stream1_planes)
        self.relu = LIReLU(inplace=True)

        self.conv2 = li_conv3x3(
            stream1_planes, stream2_planes, stream1_planes,
            stream1_planes, stream2_planes, stream1_planes
        )
        self.bn2 = norm_layer(stream1_planes, stream2_planes, stream1_planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, stream1_input: Tensor, stream2_input: Tensor, integrated_input: Tensor = None) -> tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass - MUCH SIMPLER with unified LIConv2d neurons!

        The integration now happens INSIDE LIConv2d, so this block
        just needs to do: conv → bn → relu → residual (ResNet pattern)
        """
        # Save identities
        stream1_identity = stream1_input
        stream2_identity = stream2_input
        integrated_identity = integrated_input

        # First conv block (integration happens inside LIConv2d!)
        s1, s2, ic = self.conv1(stream1_input, stream2_input, integrated_input)
        s1, s2, ic = self.bn1(s1, s2, ic)
        s1, s2, ic = self.relu(s1, s2, ic)

        # Second conv block (integration happens inside LIConv2d!)
        s1, s2, ic = self.conv2(s1, s2, ic)
        s1, s2, ic = self.bn2(s1, s2, ic)

        # Apply downsampling to identities if needed (handles all 3 streams together)
        if self.downsample is not None:
            stream1_identity, stream2_identity, integrated_identity = self.downsample(
                stream1_identity, stream2_identity, integrated_identity
            )

        # Residual connections (exact ResNet pattern)
        s1 += stream1_identity
        s2 += stream2_identity
        if integrated_identity is not None:
            ic += integrated_identity

        # Final activation
        s1, s2, ic = self.relu(s1, s2, ic)

        return s1, s2, ic

    def forward_stream1(self, stream1_input: Tensor) -> Tensor:
        """Forward pass through stream1 pathway only."""
        stream1_identity = stream1_input

        stream1_out = self.conv1.forward_stream1(stream1_input)
        stream1_out = self.bn1.forward_stream1(stream1_out)
        stream1_out = self.relu.forward_stream1(stream1_out)

        stream1_out = self.conv2.forward_stream1(stream1_out)
        stream1_out = self.bn2.forward_stream1(stream1_out)

        if self.downsample is not None:
            stream1_identity = self.downsample.forward_stream1(stream1_identity)

        stream1_out += stream1_identity
        stream1_out = self.relu.forward_stream1(stream1_out)

        return stream1_out

    def forward_stream2(self, stream2_input: Tensor) -> Tensor:
        """Forward pass through stream2 pathway only."""
        stream2_identity = stream2_input

        stream2_out = self.conv1.forward_stream2(stream2_input)
        stream2_out = self.bn1.forward_stream2(stream2_out)
        stream2_out = self.relu.forward_stream2(stream2_out)

        stream2_out = self.conv2.forward_stream2(stream2_out)
        stream2_out = self.bn2.forward_stream2(stream2_out)

        if self.downsample is not None:
            stream2_identity = self.downsample.forward_stream2(stream2_identity)

        stream2_out += stream2_identity
        stream2_out = self.relu.forward_stream2(stream2_out)

        return stream2_out

class LIBottleneck(nn.Module):
    """
    Linear Integration version of ResNet Bottleneck block.

    Uses unified LIConv2d neurons that process 3 streams simultaneously.
    Integration happens INSIDE LIConv2d during convolution (not as separate step).

    Note: Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    while original implementation places the stride at the first 1x1 convolution(self.conv1)
    according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    This variant is also known as ResNet V1.5 and improves accuracy according to
    https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
    """

    expansion: int = 4
    
    def __init__(
        self,
        stream1_inplanes: int,
        stream2_inplanes: int,
        integrated_inplanes: int,
        stream1_planes: int,
        stream2_planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = LIBatchNorm2d

        # Calculate intermediate width for all pathways
        stream1_width = int(stream1_planes * (base_width / 64.0)) * groups
        stream2_width = int(stream2_planes * (base_width / 64.0)) * groups
        integrated_width = stream1_width  # Same as stream1

        # Store channel counts for output
        self.stream1_outplanes = stream1_planes * self.expansion
        self.stream2_outplanes = stream2_planes * self.expansion
        self.integrated_outplanes = stream1_planes * self.expansion

        # Use unified LI neurons - integration happens inside LIConv2d!
        self.conv1 = li_conv1x1(
            stream1_inplanes, stream2_inplanes, integrated_inplanes,
            stream1_width, stream2_width, integrated_width
        )
        self.bn1 = norm_layer(stream1_width, stream2_width, integrated_width)
        self.conv2 = li_conv3x3(
            stream1_width, stream2_width, integrated_width,
            stream1_width, stream2_width, integrated_width,
            stride, groups, dilation
        )
        self.bn2 = norm_layer(stream1_width, stream2_width, integrated_width)
        self.conv3 = li_conv1x1(
            stream1_width, stream2_width, integrated_width,
            self.stream1_outplanes, self.stream2_outplanes, self.integrated_outplanes
        )
        self.bn3 = norm_layer(self.stream1_outplanes, self.stream2_outplanes, self.integrated_outplanes)
        self.relu = LIReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, stream1_input: Tensor, stream2_input: Tensor, integrated_input: Tensor = None) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass with 3-stream processing."""
        # Save identities
        stream1_identity = stream1_input
        stream2_identity = stream2_input
        integrated_identity = integrated_input

        # Three conv blocks (integration happens inside LIConv2d!)
        s1, s2, ic = self.conv1(stream1_input, stream2_input, integrated_input)
        s1, s2, ic = self.bn1(s1, s2, ic)
        s1, s2, ic = self.relu(s1, s2, ic)

        s1, s2, ic = self.conv2(s1, s2, ic)
        s1, s2, ic = self.bn2(s1, s2, ic)
        s1, s2, ic = self.relu(s1, s2, ic)

        s1, s2, ic = self.conv3(s1, s2, ic)
        s1, s2, ic = self.bn3(s1, s2, ic)

        # Apply downsampling to identities if needed (handles all 3 streams together)
        if self.downsample is not None:
            stream1_identity, stream2_identity, integrated_identity = self.downsample(
                stream1_identity, stream2_identity, integrated_identity
            )

        # Residual connections
        s1 += stream1_identity
        s2 += stream2_identity
        if integrated_identity is not None:
            ic += integrated_identity

        # Final activation
        s1, s2, ic = self.relu(s1, s2, ic)

        return s1, s2, ic

    def forward_stream1(self, stream1_input: Tensor) -> Tensor:
        """Forward pass through stream1 pathway only."""
        stream1_identity = stream1_input
        
        stream1_out = self.conv1.forward_stream1(stream1_input)
        stream1_out = self.bn1.forward_stream1(stream1_out)
        stream1_out = self.relu.forward_stream1(stream1_out)
        
        stream1_out = self.conv2.forward_stream1(stream1_out)
        stream1_out = self.bn2.forward_stream1(stream1_out)
        stream1_out = self.relu.forward_stream1(stream1_out)
        
        stream1_out = self.conv3.forward_stream1(stream1_out)
        stream1_out = self.bn3.forward_stream1(stream1_out)
        
        if self.downsample is not None:
            stream1_identity = self.downsample.forward_stream1(stream1_identity)
        
        stream1_out += stream1_identity
        stream1_out = self.relu.forward_stream1(stream1_out)
        
        return stream1_out
    
    def forward_stream2(self, stream2_input: Tensor) -> Tensor:
        """Forward pass through stream2 pathway only."""
        stream2_identity = stream2_input
        
        stream2_out = self.conv1.forward_stream2(stream2_input)
        stream2_out = self.bn1.forward_stream2(stream2_out)
        stream2_out = self.relu.forward_stream2(stream2_out)
        
        stream2_out = self.conv2.forward_stream2(stream2_out)
        stream2_out = self.bn2.forward_stream2(stream2_out)
        stream2_out = self.relu.forward_stream2(stream2_out)
        
        stream2_out = self.conv3.forward_stream2(stream2_out)
        stream2_out = self.bn3.forward_stream2(stream2_out)
        
        if self.downsample is not None:
            stream2_identity = self.downsample.forward_stream2(stream2_identity)
        
        stream2_out += stream2_identity
        stream2_out = self.relu.forward_stream2(stream2_out)
        
        return stream2_out
