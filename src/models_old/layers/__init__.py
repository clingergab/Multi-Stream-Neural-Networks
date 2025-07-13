"""
Neural network layers for Multi-Stream Neural Networks
"""

# Basic multi-channel layers
from .basic_layers import BasicMultiChannelLayer, BasicBlock, ResidualBlock

# Multi-channel convolutional layers
from .conv_layers import (
    MultiChannelConv2d, MultiChannelBatchNorm2d, MultiChannelActivation,
    MultiChannelDropout2d, MultiChannelAdaptiveAvgPool2d
)

# Multi-channel blocks - import directly from source modules
from .resnet_blocks import (
    MultiChannelResNetBasicBlock, MultiChannelResNetBottleneck,
    MultiChannelDownsample, MultiChannelSequential
)
from .blocks import MultiChannelConvBlock

# Integration layers
from .integration_layers import *

# Pathway layers (legacy)
from .pathway_layers import ColorPathway, BrightnessPathway, PathwayBlock

# Attention layers
from .attention_layers import *

__all__ = [
    # Basic multi-channel layers
    'BasicMultiChannelLayer',
    'BasicBlock',
    'ResidualBlock',
    
    # Multi-channel convolutional layers
    'MultiChannelConv2d',
    'MultiChannelBatchNorm2d', 
    'MultiChannelActivation',
    'MultiChannelDropout2d',
    'MultiChannelAdaptiveAvgPool2d',
    
    # Multi-channel blocks
    'MultiChannelResNetBasicBlock',
    'MultiChannelResNetBottleneck',
    'MultiChannelConvBlock',
    'MultiChannelDownsample',
    'MultiChannelSequential',
    
    # Integration layers
    'BaseIntegrationLayer',
    'BaseDirectMixingLayer', 
    'ScalarMixingLayer',
    'ChannelAdaptiveMixingLayer',
    'DynamicMixingLayer',
    'SpatialAdaptiveMixingLayer',
    
    # Pathway layers (legacy)
    'ColorPathway',
    'BrightnessPathway',
    'PathwayBlock'
]
