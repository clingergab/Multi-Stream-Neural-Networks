"""
Direct Mixing ResNet blocks for building DMNet architectures.
"""

from typing import Callable, Optional

import torch.nn as nn
from torch import Tensor
from .conv import DMConv2d, DMBatchNorm2d
from .container import DMReLU


def dm_conv3x3(
    stream_in_planes: list[int],
    stream_out_planes: list[int],
    integrated_in_planes: int,
    integrated_out_planes: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1
) -> DMConv2d:
    """3x3 direct mixing convolution with padding."""
    return DMConv2d(
        stream_in_planes,
        stream_out_planes,
        integrated_in_planes,
        integrated_out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        groups=groups,
        bias=False,
    )


def dm_conv1x1(
    stream_in_planes: list[int],
    stream_out_planes: list[int],
    integrated_in_planes: int,
    integrated_out_planes: int,
    stride: int = 1
) -> DMConv2d:
    """1x1 direct mixing convolution."""
    return DMConv2d(
        stream_in_planes,
        stream_out_planes,
        integrated_in_planes,
        integrated_out_planes,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class DMBasicBlock(nn.Module):
    """
    Direct Mixing version of ResNet BasicBlock.

    Uses unified DMConv2d neurons that process N streams simultaneously.
    Integration happens INSIDE DMConv2d during convolution using scalar mixing.
    """

    expansion: int = 1

    def __init__(
        self,
        stream_inplanes: list[int],
        stream_planes: list[int],
        integrated_inplanes: int,
        integrated_planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = DMBatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("DMBasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in DMBasicBlock")

        # Store channel counts for output
        self.num_streams = len(stream_inplanes)
        self.stream_outplanes = [planes * self.expansion for planes in stream_planes]
        self.integrated_outplanes = integrated_planes * self.expansion

        # Use unified DM neurons - integration happens inside DMConv2d!
        self.conv1 = dm_conv3x3(
            stream_inplanes,
            stream_planes,
            integrated_inplanes,
            integrated_planes,
            stride=stride
        )
        self.bn1 = norm_layer(stream_planes, integrated_planes)
        self.relu = DMReLU(inplace=True)

        self.conv2 = dm_conv3x3(
            stream_planes,
            stream_planes,
            integrated_planes,
            integrated_planes
        )
        self.bn2 = norm_layer(stream_planes, integrated_planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, stream_inputs: list[Tensor], integrated_input: Optional[Tensor] = None) -> tuple[list[Tensor], Tensor]:
        """
        Forward pass for Direct Mixing + Activation architecture.

        Key differences from direct_mixing:
        - Streams: conv → bn → relu (standard ResNet)
        - Integrated: conv → relu (NO BN, simpler processing)

        Integration happens INSIDE DMConv2d (pre-activation mixing),
        then ReLU is applied to integrated here (no BN).
        """
        # Save identities
        stream_identities = stream_inputs.copy()
        integrated_identity = integrated_input

        # First conv block
        # DMConv2d returns: streams (pre-activation), integrated (pre-activation)
        stream_outputs, integrated = self.conv1(stream_inputs, integrated_input)

        # Apply BN + ReLU to streams and integrated
        # DMBatchNorm skips BN on integrated (returns it as-is)
        # DMReLU applies ReLU to both streams and integrated
        stream_outputs, integrated = self.bn1(stream_outputs, integrated)
        stream_outputs, integrated = self.relu(stream_outputs, integrated)

        # Second conv block
        stream_outputs, integrated = self.conv2(stream_outputs, integrated)

        # Apply BN to streams and integrated (no final ReLU yet)
        # DMBatchNorm skips BN on integrated (returns it as-is)
        stream_outputs, integrated = self.bn2(stream_outputs, integrated)

        # Apply downsampling to identities if needed (handles all N streams together)
        if self.downsample is not None:
            stream_identities, integrated_identity = self.downsample(
                stream_identities, integrated_identity
            )

        # Residual connections (exact ResNet pattern)
        stream_outputs = [s + s_id for s, s_id in zip(stream_outputs, stream_identities)]
        if integrated_identity is not None:
            integrated = integrated + integrated_identity

        # Final activation
        stream_outputs, integrated = self.relu(stream_outputs, integrated)

        return stream_outputs, integrated

class DMBottleneck(nn.Module):
    """
    Direct Mixing version of ResNet Bottleneck block.

    Uses unified DMConv2d neurons that process N streams simultaneously.
    Integration happens INSIDE DMConv2d during convolution using scalar mixing.

    Note: Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    while original implementation places the stride at the first 1x1 convolution(self.conv1)
    according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    This variant is also known as ResNet V1.5 and improves accuracy according to
    https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
    """

    expansion: int = 4

    def __init__(
        self,
        stream_inplanes: list[int],
        stream_planes: list[int],
        integrated_inplanes: int,
        integrated_planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = DMBatchNorm2d

        # Calculate intermediate width for all pathways
        stream_widths = [int(planes * (base_width / 64.0)) * groups for planes in stream_planes]
        integrated_width = int(integrated_planes * (base_width / 64.0)) * groups

        # Store channel counts for output
        self.num_streams = len(stream_inplanes)
        self.stream_outplanes = [planes * self.expansion for planes in stream_planes]
        self.integrated_outplanes = integrated_planes * self.expansion

        # Use unified DM neurons - integration happens inside DMConv2d!
        self.conv1 = dm_conv1x1(
            stream_inplanes,
            stream_widths,
            integrated_inplanes,
            integrated_width
        )
        self.bn1 = norm_layer(stream_widths, integrated_width)
        self.conv2 = dm_conv3x3(
            stream_widths,
            stream_widths,
            integrated_width,
            integrated_width,
            stride, groups, dilation
        )
        self.bn2 = norm_layer(stream_widths, integrated_width)
        self.conv3 = dm_conv1x1(
            stream_widths,
            self.stream_outplanes,
            integrated_width,
            self.integrated_outplanes
        )
        self.bn3 = norm_layer(self.stream_outplanes, self.integrated_outplanes)
        self.relu = DMReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, stream_inputs: list[Tensor], integrated_input: Optional[Tensor] = None) -> tuple[list[Tensor], Tensor]:
        """Forward pass with N-stream processing."""
        # Save identities
        stream_identities = stream_inputs.copy()
        integrated_identity = integrated_input

        # Three conv blocks (integration happens inside DMConv2d!)
        stream_outputs, integrated = self.conv1(stream_inputs, integrated_input)
        stream_outputs, integrated = self.bn1(stream_outputs, integrated)
        stream_outputs, integrated = self.relu(stream_outputs, integrated)

        stream_outputs, integrated = self.conv2(stream_outputs, integrated)
        stream_outputs, integrated = self.bn2(stream_outputs, integrated)
        stream_outputs, integrated = self.relu(stream_outputs, integrated)

        stream_outputs, integrated = self.conv3(stream_outputs, integrated)
        stream_outputs, integrated = self.bn3(stream_outputs, integrated)

        # Apply downsampling to identities if needed (handles all N streams together)
        if self.downsample is not None:
            stream_identities, integrated_identity = self.downsample(
                stream_identities, integrated_identity
            )

        # Residual connections
        stream_outputs = [s + s_id for s, s_id in zip(stream_outputs, stream_identities)]
        if integrated_identity is not None:
            integrated = integrated + integrated_identity

        # Final activation
        stream_outputs, integrated = self.relu(stream_outputs, integrated)

        return stream_outputs, integrated
