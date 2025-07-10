"""
Multi-Stream Neural Networks (MSNNs)

A PyTorch implementation of biologically-inspired neural networks with separate
pathways for color and brightness processing.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import main model implementations
try:
    from .models2 import (
        BaseModel, MCResNet, 
        mc_resnet18, mc_resnet34, mc_resnet50, mc_resnet101, mc_resnet152
    )
except ImportError:
    pass

# Import data utilities
try:
    from .data_utils import DualChannelDataset, create_dual_channel_dataloader
except ImportError:
    pass

# Import utilities
try:
    from .utils import ConfigManager, ModelRegistry
except ImportError:
    pass

# Import training components  
try:
    from . import training
except ImportError:
    pass
