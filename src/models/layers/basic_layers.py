"""Basic building blocks and utility layers."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Basic convolutional block with BatchNorm and ReLU."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """Residual block for deeper networks."""
    
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class BasicMultiChannelLayer(nn.Module):
    """
    Basic Multi-Channel Layer implementation.
    
    Each neuron outputs separate values for each modality (color and brightness).
    Mathematical Formulation:
        y_color = f(Σ(wc_i * xc_i) + b_c)
        y_brightness = f(Σ(wb_i * xb_i) + b_b)
        output = [y_color, y_brightness]
    """
    
    def __init__(self, color_input_size=None, brightness_input_size=None, output_size=None, 
                 bias=True, activation='relu', input_size=None):
        """
        Initialize BasicMultiChannelLayer.
        
        Args:
            color_input_size: Input size for color stream
            brightness_input_size: Input size for brightness stream  
            output_size: Output size for both streams
            bias: Whether to use bias
            activation: Activation function
            input_size: For backward compatibility - if provided, both streams use same input size
        """
        super().__init__()
        
        # Handle backward compatibility
        if input_size is not None:
            color_input_size = input_size
            brightness_input_size = input_size
            
        if color_input_size is None or brightness_input_size is None or output_size is None:
            raise ValueError("Must provide color_input_size, brightness_input_size, and output_size")
        
        # Separate weights for each modality with different input sizes
        self.color_weights = nn.Parameter(torch.randn(output_size, color_input_size))
        self.brightness_weights = nn.Parameter(torch.randn(output_size, brightness_input_size))
        
        # Store input sizes for reference
        self.color_input_size = color_input_size
        self.brightness_input_size = brightness_input_size
        self.output_size = output_size
        
        # Separate biases for each modality
        if bias:
            self.color_bias = nn.Parameter(torch.zeros(output_size))
            self.brightness_bias = nn.Parameter(torch.zeros(output_size))
        else:
            self.register_parameter('color_bias', None)
            self.register_parameter('brightness_bias', None)
        
        # Activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'gelu':
            self.activation = F.gelu
        else:
            self.activation = lambda x: x  # Linear activation
            
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        nn.init.xavier_uniform_(self.color_weights)
        nn.init.xavier_uniform_(self.brightness_weights)
        
        if self.color_bias is not None:
            nn.init.zeros_(self.color_bias)
            nn.init.zeros_(self.brightness_bias)
    
    def forward(self, color_inputs, brightness_inputs):
        """
        Forward pass through multi-channel layer.
        
        Args:
            color_inputs: Color input tensor (batch_size, input_size)
            brightness_inputs: Brightness input tensor (batch_size, input_size)
            
        Returns:
            Tuple of (color_output, brightness_output)
        """
        # Linear transformations for each modality
        color_linear = F.linear(color_inputs, self.color_weights, self.color_bias)
        brightness_linear = F.linear(brightness_inputs, self.brightness_weights, self.brightness_bias)
        
        # Apply activation functions
        color_output = self.activation(color_linear)
        brightness_output = self.activation(brightness_linear)
        
        return color_output, brightness_output
    
    def to_device_optimized(self, device=None):
        """
        Move layer to device with optimizations.
        
        Args:
            device: Target device (auto-detected if None)
            
        Returns:
            Self for method chaining
        """
        if device is None:
            from ...utils.device_utils import get_device
            device = get_device()
        
        # Move to device
        self.to(device)
        
        # Apply device-specific optimizations
        if device.type == 'cuda':
            # Enable memory efficient attention if available
            if hasattr(torch.backends.cuda, 'enable_math_sdp'):
                torch.backends.cuda.enable_math_sdp(True)
        
        return self
