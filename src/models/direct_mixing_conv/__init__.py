"""
Direct Mixing Convolution (DMNet) module.

This module provides direct mixing models where streams are combined through
learned scalar coefficients at the convolutional layer level.
"""

from .dm_net import DMNet, dm_net18, dm_net34, dm_net50, dm_net101

__all__ = [
    'DMNet',
    'dm_net18',
    'dm_net34',
    'dm_net50',
    'dm_net101',
]
