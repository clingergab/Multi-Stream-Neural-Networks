"""
Multi-channel convolutional layers for Multi-Stream Neural Networks.

This module implements convolutional layers that process color and brightness
inputs separately through dedicated weights and biases for each modality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiChannelConv2d(nn.Module):
    """
    Multi-channel 2D convolutional layer.
    
    Applies separate convolutions to color and brightness inputs with 
    independent weights and biases for each modality.
    
    Mathematical Formulation:
        color_output = conv2d(color_input, color_weights) + color_bias
        brightness_output = conv2d(brightness_input, brightness_weights) + brightness_bias
    """
    
    def __init__(self, color_in_channels=None, brightness_in_channels=None, out_channels=None, 
                 kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True, 
                 padding_mode='zeros', in_channels=None):
        super().__init__()
        
        # Handle backward compatibility
        if in_channels is not None:
            color_in_channels = in_channels
            brightness_in_channels = in_channels
        
        if color_in_channels is None or brightness_in_channels is None or out_channels is None:
            raise ValueError("Must provide color_in_channels, brightness_in_channels, and out_channels")
        
        # Store parameters
        self.color_in_channels = color_in_channels
        self.brightness_in_channels = brightness_in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size) 
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        self.padding_mode = padding_mode
        
        # Separate weights for each modality with different input channel sizes
        self.color_weight = nn.Parameter(torch.randn(
            out_channels, color_in_channels // groups, *self.kernel_size
        ))
        self.brightness_weight = nn.Parameter(torch.randn(
            out_channels, brightness_in_channels // groups, *self.kernel_size
        ))
        
        # Separate biases for each modality
        if bias:
            self.color_bias = nn.Parameter(torch.zeros(out_channels))
            self.brightness_bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('color_bias', None)
            self.register_parameter('brightness_bias', None)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization."""
        nn.init.kaiming_uniform_(self.color_weight, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.brightness_weight, a=0, mode='fan_in', nonlinearity='relu')
        
        if self.color_bias is not None:
            nn.init.zeros_(self.color_bias)
            nn.init.zeros_(self.brightness_bias)
    
    def forward(self, color_input, brightness_input):
        """
        Forward pass through multi-channel convolution.
        
        Args:
            color_input: Color input tensor (B, C, H, W)
            brightness_input: Brightness input tensor (B, C, H, W)
            
        Returns:
            Tuple of (color_output, brightness_output)
        """
        # Apply convolutions with respective weights
        color_output = F.conv2d(
            color_input, self.color_weight, self.color_bias,
            self.stride, self.padding, self.dilation, self.groups
        )
        
        brightness_output = F.conv2d(
            brightness_input, self.brightness_weight, self.brightness_bias,
            self.stride, self.padding, self.dilation, self.groups
        )
        
        return color_output, brightness_output
    
    def forward_color(self, color_input):
        """
        Forward pass through only the color pathway.
        
        Args:
            color_input: Color input tensor (B, C, H, W)
            
        Returns:
            Color output tensor
        """
        # Apply convolution with color weights
        color_output = F.conv2d(
            color_input, self.color_weight, self.color_bias,
            self.stride, self.padding, self.dilation, self.groups
        )
        
        return color_output
        
    def forward_brightness(self, brightness_input):
        """
        Forward pass through only the brightness pathway.
        
        Args:
            brightness_input: Brightness input tensor (B, C, H, W)
            
        Returns:
            Brightness output tensor
        """
        # Apply convolution with brightness weights
        brightness_output = F.conv2d(
            brightness_input, self.brightness_weight, self.brightness_bias,
            self.stride, self.padding, self.dilation, self.groups
        )
        
        return brightness_output


class MultiChannelBatchNorm2d(nn.Module):
    """
    Multi-channel 2D batch normalization.
    
    Applies separate batch normalization to color and brightness features
    with independent running statistics for each modality.
    """
    
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        
        # Separate batch norm for each modality
        self.color_bn = nn.BatchNorm2d(
            num_features, eps=eps, momentum=momentum, 
            affine=affine, track_running_stats=track_running_stats
        )
        self.brightness_bn = nn.BatchNorm2d(
            num_features, eps=eps, momentum=momentum,
            affine=affine, track_running_stats=track_running_stats
        )
    
    def forward(self, color_input, brightness_input):
        """
        Forward pass through multi-channel batch normalization.
        
        Args:
            color_input: Color input tensor (B, C, H, W)
            brightness_input: Brightness input tensor (B, C, H, W)
            
        Returns:
            Tuple of (color_output, brightness_output)
        """
        color_output = self.color_bn(color_input)
        brightness_output = self.brightness_bn(brightness_input)
        
        return color_output, brightness_output
    
    def forward_color(self, color_input):
        """
        Forward pass through batch normalization for the color pathway only.
        
        Args:
            color_input: Color input tensor (B, C, H, W)
            
        Returns:
            Normalized color output tensor
        """
        return self.color_bn(color_input)
    
    def forward_brightness(self, brightness_input):
        """
        Forward pass through batch normalization for the brightness pathway only.
        
        Args:
            brightness_input: Brightness input tensor (B, C, H, W)
            
        Returns:
            Normalized brightness output tensor
        """
        return self.brightness_bn(brightness_input)


class MultiChannelActivation(nn.Module):
    """
    Multi-channel activation function.
    
    Applies the same activation function to both color and brightness streams.
    """
    
    def __init__(self, activation='relu', inplace=False):
        super().__init__()
        
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=inplace)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1, inplace=inplace)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'silu':
            self.activation = nn.SiLU(inplace=inplace)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, color_input, brightness_input):
        """
        Forward pass through multi-channel activation.
        
        Args:
            color_input: Color input tensor
            brightness_input: Brightness input tensor
            
        Returns:
            Tuple of (color_output, brightness_output)
        """
        color_output = self.activation(color_input)
        brightness_output = self.activation(brightness_input)
        
        return color_output, brightness_output
    
    def forward_single(self, x):
        """
        Forward pass for a single stream.
        
        Args:
            x: Input tensor from either color or brightness stream
            
        Returns:
            Activated output tensor
        """
        return self.activation(x)


class MultiChannelDropout2d(nn.Module):
    """
    Multi-channel 2D dropout.
    
    Applies dropout independently to color and brightness features.
    """
    
    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        
        self.color_dropout = nn.Dropout2d(p=p, inplace=inplace)
        self.brightness_dropout = nn.Dropout2d(p=p, inplace=inplace)
    
    def forward(self, color_input, brightness_input):
        """
        Forward pass through multi-channel dropout.
        
        Args:
            color_input: Color input tensor
            brightness_input: Brightness input tensor
            
        Returns:
            Tuple of (color_output, brightness_output)
        """
        color_output = self.color_dropout(color_input)
        brightness_output = self.brightness_dropout(brightness_input)
        
        return color_output, brightness_output


class MultiChannelAdaptiveAvgPool2d(nn.Module):
    """
    Multi-channel adaptive average pooling.
    
    Applies adaptive average pooling to both color and brightness streams.
    """
    
    def __init__(self, output_size):
        super().__init__()
        
        self.pool = nn.AdaptiveAvgPool2d(output_size)
    
    def forward(self, color_input, brightness_input):
        """
        Forward pass through multi-channel adaptive average pooling.
        
        Args:
            color_input: Color input tensor
            brightness_input: Brightness input tensor
            
        Returns:
            Tuple of (color_output, brightness_output)
        """
        color_output = self.pool(color_input)
        brightness_output = self.pool(brightness_input)
        
        return color_output, brightness_output
    
    def forward_single(self, x):
        """
        Forward pass for a single stream.
        
        Args:
            x: Input tensor from either color or brightness stream
            
        Returns:
            Pooled output tensor
        """
        return self.pool(x)
