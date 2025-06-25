"""
Continuous integration models for multi-stream neural networks.
"""

from .concat_linear import ConcatLinearModel
from .attention_based import AttentionBasedModel  
from .neural_processing import NeuralProcessingModel
from .direct_mixing import (
    ScalarMixingModel,
    ChannelAdaptiveMixingModel,
    DynamicMixingModel,
    SpatialAdaptiveMixingModel
)

__all__ = [
    'ConcatLinearModel',
    'AttentionBasedModel',
    'NeuralProcessingModel', 
    'ScalarMixingModel',
    'ChannelAdaptiveMixingModel',
    'DynamicMixingModel',
    'SpatialAdaptiveMixingModel'
]