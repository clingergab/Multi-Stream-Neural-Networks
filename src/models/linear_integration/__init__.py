"""
Linear Integration module for multi-stream neural networks.

This module provides linear integration architectures that combine multiple input streams
(e.g., RGB, Depth, Orthogonal) through learned linear combinations at the neuron level.
"""

from .li_net import LINet as LINet2Stream

# Submodules are imported on-demand to avoid circular dependencies
# from .li_net3 import LINet as LINet3Stream
# from .li_net3_soma import LINet as LINet3StreamSoma

__all__ = [
    'LINet2Stream',
]
