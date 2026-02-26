"""
Linear Integration ResNet blocks for building LINet architectures.
"""

from typing import Callable, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from .conv import LIConv2d, LIBatchNorm2d
from .container import LIReLU


def li_conv3x3(
    stream_in_planes: list[int],
    stream_out_planes: list[int],
    integrated_in_planes: int,
    integrated_out_planes: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1
) -> LIConv2d:
    """3x3 linear integration convolution with padding."""
    return LIConv2d(
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


def li_conv1x1(
    stream_in_planes: list[int],
    stream_out_planes: list[int],
    integrated_in_planes: int,
    integrated_out_planes: int,
    stride: int = 1
) -> LIConv2d:
    """1x1 linear integration convolution."""
    return LIConv2d(
        stream_in_planes,
        stream_out_planes,
        integrated_in_planes,
        integrated_out_planes,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class LIBasicBlock(nn.Module):
    """
    Linear Integration version of ResNet BasicBlock.

    Uses unified LIConv2d neurons that process N streams simultaneously.
    Integration happens INSIDE LIConv2d during convolution (not as separate step).
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
            norm_layer = LIBatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("LIBasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in LIBasicBlock")

        # Store channel counts for output
        self.num_streams = len(stream_inplanes)
        self.stream_outplanes = [planes * self.expansion for planes in stream_planes]
        self.integrated_outplanes = integrated_planes * self.expansion

        # Use unified LI neurons - integration happens inside LIConv2d!
        self.conv1 = li_conv3x3(
            stream_inplanes,
            stream_planes,
            integrated_inplanes,
            integrated_planes,
            stride=stride
        )
        self.bn1 = norm_layer(stream_planes, integrated_planes)
        self.relu = LIReLU(inplace=True)

        self.conv2 = li_conv3x3(
            stream_planes,
            stream_planes,
            integrated_planes,
            integrated_planes
        )
        self.bn2 = norm_layer(stream_planes, integrated_planes)

        self.downsample = downsample
        self.stride = stride

    def forward(
        self,
        stream_inputs: list[Tensor],
        integrated_input: Optional[Tensor] = None,
        blanked_mask: Optional[dict[int, Tensor]] = None
    ) -> tuple[list[Tensor], Tensor]:
        """
        Forward pass - MUCH SIMPLER with unified LIConv2d neurons!

        The integration now happens INSIDE LIConv2d, so this block
        just needs to do: conv → bn → relu → residual (ResNet pattern)

        Args:
            stream_inputs: List of input tensors for each stream
            integrated_input: Optional integrated stream input
            blanked_mask: Optional per-sample blanking mask for modality dropout.
                         Propagated to all contained modules.
        """
        # Save identities (already zeros for blanked samples from previous layer)
        stream_identities = stream_inputs.copy()
        integrated_identity = integrated_input

        # First conv block: conv1 → bn1+relu (fused)
        stream_outputs, integrated = self.conv1(stream_inputs, integrated_input, blanked_mask)
        stream_outputs, integrated = self.bn1(stream_outputs, integrated, blanked_mask, apply_relu=True)

        # Second conv block: conv2 → bn2 (no relu — relu comes after residual)
        stream_outputs, integrated = self.conv2(stream_outputs, integrated, blanked_mask)
        stream_outputs, integrated = self.bn2(stream_outputs, integrated, blanked_mask)

        # Apply downsampling to identities if needed (handles all N streams together)
        # Downsample is an LISequential containing LIConv2d + LIBatchNorm2d - accepts blanked_mask
        if self.downsample is not None:
            stream_identities, integrated_identity = self.downsample(
                stream_identities, integrated_identity, blanked_mask
            )

        # Residual connections + final activation
        # Note: for blanked samples, both stream_outputs and stream_identities are zeros
        # so the result is 0 + 0 = 0 (blanked samples stay zeroed)
        stream_outputs = [F.relu(s + s_id, inplace=True) for s, s_id in zip(stream_outputs, stream_identities)]
        if integrated_identity is not None:
            integrated = F.relu(integrated + integrated_identity, inplace=True)

        return stream_outputs, integrated

    def forward_stream(self, stream_idx: int, stream_input: Tensor) -> Tensor:
        """
        Forward pass for a single stream through the BasicBlock.

        This processes only the specified stream without affecting other streams'
        BN running statistics. Used for stream monitoring.
        """
        identity = stream_input

        # First conv block
        out = self.conv1.forward_stream(stream_idx, stream_input)
        out = self.bn1.forward_stream(stream_idx, out)
        out = F.relu(out, inplace=True)

        # Second conv block
        out = self.conv2.forward_stream(stream_idx, out)
        out = self.bn2.forward_stream(stream_idx, out)

        # Apply downsampling to identity if needed
        if self.downsample is not None:
            identity = self.downsample.forward_stream(stream_idx, identity)

        # Residual connection + final activation
        out = F.relu(out + identity, inplace=True)

        return out


class LIBottleneck(nn.Module):
    """
    Linear Integration version of ResNet Bottleneck block.

    Uses unified LIConv2d neurons that process N streams simultaneously.
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
            norm_layer = LIBatchNorm2d

        # Calculate intermediate width for all pathways
        stream_widths = [int(planes * (base_width / 64.0)) * groups for planes in stream_planes]
        integrated_width = int(integrated_planes * (base_width / 64.0)) * groups

        # Store channel counts for output
        self.num_streams = len(stream_inplanes)
        self.stream_outplanes = [planes * self.expansion for planes in stream_planes]
        self.integrated_outplanes = integrated_planes * self.expansion

        # Use unified LI neurons - integration happens inside LIConv2d!
        self.conv1 = li_conv1x1(
            stream_inplanes,
            stream_widths,
            integrated_inplanes,
            integrated_width
        )
        self.bn1 = norm_layer(stream_widths, integrated_width)
        self.conv2 = li_conv3x3(
            stream_widths,
            stream_widths,
            integrated_width,
            integrated_width,
            stride, groups, dilation
        )
        self.bn2 = norm_layer(stream_widths, integrated_width)
        self.conv3 = li_conv1x1(
            stream_widths,
            self.stream_outplanes,
            integrated_width,
            self.integrated_outplanes
        )
        self.bn3 = norm_layer(self.stream_outplanes, self.integrated_outplanes)
        self.relu = LIReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    
    def forward(
        self,
        stream_inputs: list[Tensor],
        integrated_input: Optional[Tensor] = None,
        blanked_mask: Optional[dict[int, Tensor]] = None
    ) -> tuple[list[Tensor], Tensor]:
        """Forward pass with N-stream processing.

        Args:
            stream_inputs: List of input tensors for each stream
            integrated_input: Optional integrated stream input
            blanked_mask: Optional per-sample blanking mask for modality dropout.
                         Propagated to all contained modules.
        """
        # Save identities (already zeros for blanked samples from previous layer)
        stream_identities = stream_inputs.copy()
        integrated_identity = integrated_input

        # Three conv blocks: conv1 → bn1+relu, conv2 → bn2+relu, conv3 → bn3 (no relu)
        stream_outputs, integrated = self.conv1(stream_inputs, integrated_input, blanked_mask)
        stream_outputs, integrated = self.bn1(stream_outputs, integrated, blanked_mask, apply_relu=True)

        stream_outputs, integrated = self.conv2(stream_outputs, integrated, blanked_mask)
        stream_outputs, integrated = self.bn2(stream_outputs, integrated, blanked_mask, apply_relu=True)

        stream_outputs, integrated = self.conv3(stream_outputs, integrated, blanked_mask)
        stream_outputs, integrated = self.bn3(stream_outputs, integrated, blanked_mask)

        # Apply downsampling to identities if needed (handles all N streams together)
        # Downsample is an LISequential containing LIConv2d + LIBatchNorm2d - accepts blanked_mask
        if self.downsample is not None:
            stream_identities, integrated_identity = self.downsample(
                stream_identities, integrated_identity, blanked_mask
            )

        # Residual connections + final activation
        # Note: for blanked samples, both stream_outputs and stream_identities are zeros
        # so the result is 0 + 0 = 0 (blanked samples stay zeroed)
        stream_outputs = [F.relu(s + s_id, inplace=True) for s, s_id in zip(stream_outputs, stream_identities)]
        if integrated_identity is not None:
            integrated = F.relu(integrated + integrated_identity, inplace=True)

        return stream_outputs, integrated

    def forward_stream(self, stream_idx: int, stream_input: Tensor) -> Tensor:
        """
        Forward pass for a single stream through the Bottleneck block.

        This processes only the specified stream without affecting other streams'
        BN running statistics. Used for stream monitoring.
        """
        identity = stream_input

        # Three conv blocks
        out = self.conv1.forward_stream(stream_idx, stream_input)
        out = self.bn1.forward_stream(stream_idx, out)
        out = F.relu(out, inplace=True)

        out = self.conv2.forward_stream(stream_idx, out)
        out = self.bn2.forward_stream(stream_idx, out)
        out = F.relu(out, inplace=True)

        out = self.conv3.forward_stream(stream_idx, out)
        out = self.bn3.forward_stream(stream_idx, out)

        # Apply downsampling to identity if needed
        if self.downsample is not None:
            identity = self.downsample.forward_stream(stream_idx, identity)

        # Residual connection + final activation
        out = F.relu(out + identity, inplace=True)

        return out