"""
Linear Integration Network (LINet) - 3-Stream SOMA Architecture

N-stream neural network with neuron-level integration through learned linear weights.
"""

from .li_net_soma import (
    LINet,
    li_resnet18,
    li_resnet34,
    li_resnet50,
    li_resnet101,
    li_resnet152,
)

__all__ = [
    'LINet',
    'li_resnet18',
    'li_resnet34',
    'li_resnet50',
    'li_resnet101',
    'li_resnet152',
]
