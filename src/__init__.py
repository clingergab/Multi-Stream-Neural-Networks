"""
Multi-Stream Neural Networks (MSNNs)

A PyTorch implementation of biologically-inspired neural networks with separate
pathways for color and brightness processing.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import only what exists and is needed
try:
    from . import models
except ImportError:
    pass

try:
    from . import datasets
except ImportError:
    pass

try:
    from . import training
except ImportError:
    pass
