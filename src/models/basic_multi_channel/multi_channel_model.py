"""
Multi-channel model implementation.

This module implements the true multi-channel neural network as described in the documentation,
where each neuron has separate weights for color and brightness inputs but shares processing.
"""

import torch
import torch.nn as nn
from ..layers.conv_layers import MultiChannelConv2d, MultiChannelBatchNorm2d, MultiChannelActivation, MultiChannelAdaptiveAvgPool2d
from ..layers.resnet_blocks import MultiChannelResNetBasicBlock, MultiChannelResNetBottleneck
from ...utils.device_utils import get_device_manager


class MultiChannelNetwork(nn.Module):
    """
    Multi-Channel Neural Network implementation.
    
    This network follows the architecture described in the documentation where:
    - Each layer has separate weights for color and brightness inputs
    - Pathways remain separate throughout the network  
    - Only the final classifier combines the streams
    
    Mathematical Formulation per layer:
        y_color = f(Σ(wc_i * xc_i) + b_c)
        y_brightness = f(Σ(wb_i * xb_i) + b_b)
        output = [y_color, y_brightness]
    """
    
    def __init__(self, num_classes=10, input_channels=3, hidden_channels=64, 
                 num_blocks=[2, 2, 2, 2], block_type='basic', activation='relu'):
        super().__init__()
        
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        
        # Initial convolutional layer
        self.conv1 = MultiChannelConv2d(
            input_channels, hidden_channels, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = MultiChannelBatchNorm2d(hidden_channels)
        self.activation = MultiChannelActivation(activation, inplace=True)
        
        # Max pooling
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Choose block type
        if block_type == 'basic':
            block = MultiChannelResNetBasicBlock
            expansion = 1
        elif block_type == 'bottleneck':
            block = MultiChannelResNetBottleneck
            expansion = 4
        else:
            raise ValueError(f"Unsupported block type: {block_type}")
        
        # Create residual layers
        self.layer1 = self._make_layer(block, hidden_channels, hidden_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, hidden_channels * expansion, hidden_channels * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, hidden_channels * 2 * expansion, hidden_channels * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, hidden_channels * 4 * expansion, hidden_channels * 8, num_blocks[3], stride=2)
        
        # Global average pooling
        self.avgpool = MultiChannelAdaptiveAvgPool2d((1, 1))
        
        # Final classifier - combines both streams
        final_features = hidden_channels * 8 * expansion * 2  # *2 because we concatenate color and brightness
        self.classifier = nn.Linear(final_features, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, block, in_channels, out_channels, num_blocks, stride=1):
        """Create a layer with multiple blocks."""
        layers = []
        
        # First block might need downsampling
        downsample = None
        if stride != 1 or in_channels != out_channels * block.expansion:
            from ..layers.resnet_blocks import MultiChannelDownsample
            downsample = MultiChannelDownsample(in_channels, out_channels * block.expansion, stride)
        
        # First block
        layers.append(block(in_channels, out_channels, stride, downsample))
        
        # Remaining blocks
        in_channels = out_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(in_channels, out_channels))
        
        return nn.ModuleList(layers)
    
    def _forward_layer(self, layer, color_input, brightness_input):
        """Forward pass through a layer of blocks."""
        for block in layer:
            color_input, brightness_input = block(color_input, brightness_input)
        return color_input, brightness_input
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, MultiChannelConv2d)):
                nn.init.kaiming_normal_(module.weight if hasattr(module, 'weight') else module.color_weight, 
                                      mode='fan_out', nonlinearity='relu')
            elif isinstance(module, (nn.BatchNorm2d, MultiChannelBatchNorm2d)):
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.constant_(module.weight, 1)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, color_input, brightness_input):
        """
        Forward pass through the multi-channel network.
        
        Args:
            color_input: Color input tensor (B, C, H, W)
            brightness_input: Brightness input tensor (B, C, H, W) 
            
        Returns:
            Classification logits (B, num_classes)
        """
        # Initial convolution
        color_x, brightness_x = self.conv1(color_input, brightness_input)
        color_x, brightness_x = self.bn1(color_x, brightness_x)
        color_x, brightness_x = self.activation(color_x, brightness_x)
        
        # Max pooling (applied to both streams independently)
        color_x = self.maxpool(color_x)
        brightness_x = self.maxpool(brightness_x)
        
        # Residual layers
        color_x, brightness_x = self._forward_layer(self.layer1, color_x, brightness_x)
        color_x, brightness_x = self._forward_layer(self.layer2, color_x, brightness_x)
        color_x, brightness_x = self._forward_layer(self.layer3, color_x, brightness_x)
        color_x, brightness_x = self._forward_layer(self.layer4, color_x, brightness_x)
        
        # Global average pooling
        color_x, brightness_x = self.avgpool(color_x, brightness_x)
        
        # Flatten features
        color_x = torch.flatten(color_x, 1)
        brightness_x = torch.flatten(brightness_x, 1)
        
        # Combine streams for classification
        combined_features = torch.cat([color_x, brightness_x], dim=1)
        
        # Final classification
        output = self.classifier(combined_features)
        
        return output
    
    def to_device_optimized(self, device=None, enable_optimizations=True):
        """
        Move model to optimal device with performance optimizations.
        
        Args:
            device: Target device (auto-detected if None)
            enable_optimizations: Whether to enable device-specific optimizations
            
        Returns:
            Self for method chaining
        """
        if device is None:
            device_manager = get_device_manager()
            device = device_manager.device
        else:
            device_manager = get_device_manager()
            if isinstance(device, str):
                device = torch.device(device)
        
        # Move model to device
        self.to(device)
        
        if enable_optimizations:
            # Apply device-specific optimizations
            if device.type == 'cuda':
                # Enable cuDNN optimizations
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                print(f"   ⚡ CUDA optimizations enabled for {device}")
                
            elif device.type == 'mps':
                print(f"   ⚡ MPS optimizations enabled for {device}")
        
        return self
    
    def get_memory_usage(self):
        """Get model memory usage information."""
        device_manager = get_device_manager()
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Estimate memory usage (rough approximation)
        param_memory_mb = (total_params * 4) / (1024 * 1024)  # 4 bytes per float32 parameter
        
        result = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'estimated_param_memory_mb': param_memory_mb,
            'device': str(next(self.parameters()).device)
        }
        
        # Add GPU-specific memory info if available
        if device_manager.device.type == 'cuda':
            gpu_memory = device_manager.get_memory_info()
            result.update(gpu_memory)
        
        return result
    
    def enable_mixed_precision_if_available(self):
        """
        Enable mixed precision training if supported.
        
        Returns:
            bool: True if mixed precision was enabled
        """
        device_manager = get_device_manager()
        if device_manager.enable_mixed_precision():
            print("   ⚡ Mixed precision (AMP) enabled - using Tensor Cores")
            return True
        else:
            print("   ℹ️  Mixed precision not available on this device")
            return False


# Factory functions for common configurations
def multi_channel_18(num_classes=10, input_channels=3, **kwargs):
    """
    Create a Multi-Channel 18-layer network (ResNet-18 style).
    
    Automatically optimizes for the best available device (CUDA → MPS → CPU).
    
    Args:
        num_classes: Number of output classes
        input_channels: Number of input channels
        **kwargs: Additional arguments for MultiChannelNetwork
    
    Returns:
        Device-optimized MultiChannelNetwork
    """
    model = MultiChannelNetwork(
        num_classes=num_classes,
        input_channels=input_channels,
        hidden_channels=64,
        num_blocks=[2, 2, 2, 2],
        block_type='basic',
        **kwargs
    )
    
    # Always optimize for best available device
    model.to_device_optimized()
    print(f"📱 multi_channel_18 optimized for: {next(model.parameters()).device}")
    
    return model


def multi_channel_50(num_classes=10, input_channels=3, **kwargs):
    """
    Create a Multi-Channel 50-layer network (ResNet-50 style with bottleneck blocks).
    
    Automatically optimizes for the best available device (CUDA → MPS → CPU).
    
    Args:
        num_classes: Number of output classes
        input_channels: Number of input channels
        **kwargs: Additional arguments for MultiChannelNetwork
        
    Returns:
        Device-optimized MultiChannelNetwork
    """
    model = MultiChannelNetwork(
        num_classes=num_classes,
        input_channels=input_channels,
        hidden_channels=64,
        num_blocks=[3, 4, 6, 3],  # ResNet-50 style blocks
        block_type='bottleneck',  # Use bottleneck blocks for deeper network
        **kwargs
    )
    
    # Always optimize for best available device
    model.to_device_optimized()
    print(f"📱 multi_channel_50 optimized for: {next(model.parameters()).device}")
    
    # Enable mixed precision for larger model if available
    model.enable_mixed_precision_if_available()
    
    return model

