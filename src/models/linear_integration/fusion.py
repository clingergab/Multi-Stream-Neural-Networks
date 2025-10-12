"""
Linear Integration fusion module.

Note: LINet doesn't use a separate fusion layer like MCResNet does.
Instead, fusion happens INSIDE LIConv2d neurons during convolution.

The integrated stream IS the fused representation, created by combining
stream1 and stream2 outputs using learned 1x1 convolutions within LIConv2d.

This file exists for consistency with the multi_channel folder structure,
but doesn't define any fusion classes since they're not needed for LINet.
"""

# No fusion classes needed - integration happens in LIConv2d neurons!
