"""
Multi-Stream Neural Networks model implementations (new, cleaner version).

This package contains reorganized implementations of various neural network architectures
used in the Multi-Stream Neural Networks project.
"""

from .abstracts.abstract_model import BaseModel
from .core.resnet import ResNet, resnet18, resnet34, resnet50, resnet101, resnet152
from .multi_channel import (
    MCResNet, 
    mc_resnet18, 
    mc_resnet34, 
    mc_resnet50, 
    mc_resnet101, 
    mc_resnet152,
    MCSequential,
    MCConv2d, 
    MCBatchNorm2d,
    MCBasicBlock,
    MCBottleneck,
    mc_conv3x3,
    mc_conv1x1,
    MCMaxPool2d,
    MCAdaptiveAvgPool2d
)

__all__ = [
    "BaseModel",
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "MCResNet",
    "mc_resnet18",
    "mc_resnet34", 
    "mc_resnet50",
    "mc_resnet101",
    "mc_resnet152",
    "MCSequential",
    "MCConv2d", 
    "MCBatchNorm2d",
    "MCBasicBlock",
    "MCBottleneck",
    "mc_conv3x3",
    "mc_conv1x1",
    "MCMaxPool2d",
    "MCAdaptiveAvgPool2d"
]
