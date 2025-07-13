"""Basic multi-channel model package."""

from .base_multi_channel_network import (
    BaseMultiChannelNetwork,
    base_multi_channel_small,
    base_multi_channel_medium, 
    base_multi_channel_large
)
from .multi_channel_resnet_network import (
    MultiChannelResNetNetwork,
    multi_channel_resnet18,
    multi_channel_resnet34,
    multi_channel_resnet50,
    multi_channel_resnet101,
    multi_channel_resnet152
)

__all__ = [
    # New modular architecture
    'BaseMultiChannelNetwork',
    'base_multi_channel_small',
    'base_multi_channel_medium', 
    'base_multi_channel_large',
    'MultiChannelResNetNetwork',
    'multi_channel_resnet18',
    'multi_channel_resnet34',
    'multi_channel_resnet50',
    'multi_channel_resnet101',
    'multi_channel_resnet152'
]
