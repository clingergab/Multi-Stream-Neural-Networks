"""
Multi-Channel ResNet implementation following standard ResNet architecture.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Dict
from ..base import BaseMultiStreamModel
from ..layers.resnet_blocks import (
    MultiChannelResNetBasicBlock, 
    MultiChannelResNetBottleneck,
    MultiChannelDownsample,
    MultiChannelSequential
)
from ..layers.conv_layers import (
    MultiChannelConv2d, 
    MultiChannelBatchNorm2d, 
    MultiChannelActivation,
    MultiChannelAdaptiveAvgPool2d
)


class MultiChannelResNetNetwork(BaseMultiStreamModel):
    """
    Multi-Channel ResNet Network for image-based multi-stream data.
    
    Follows standard ResNet architecture but processes two streams 
    (color and brightness) separately with proper residual connections.
    
    Suitable for:
    - Image classification with RGB + brightness streams
    - Computer vision tasks requiring multi-modal processing
    - Spatial feature extraction with residual learning
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        color_input_channels: int = 3,
        brightness_input_channels: int = 1,
        num_blocks: List[int] = [2, 2, 2, 2],
        block_type: str = 'basic',
        activation: str = 'relu',
        input_channels: int = None,  # For backward compatibility
        **kwargs
    ):
        # Handle backward compatibility
        if input_channels is not None:
            color_input_channels = input_channels
            brightness_input_channels = input_channels
        
        # Initialize base class
        super().__init__(
            input_size=(max(color_input_channels, brightness_input_channels), 224, 224),
            hidden_size=64,  # Initial channel count
            num_classes=num_classes,
            **kwargs
        )
        
        self.color_input_channels = color_input_channels
        self.brightness_input_channels = brightness_input_channels
        self.num_blocks = num_blocks
        self.block_type = block_type
        self.activation = activation
        
        # Track current channel count for proper ResNet progression
        self.inplanes = 64
        
        # Choose block type
        if block_type == 'basic':
            self.block = MultiChannelResNetBasicBlock
        elif block_type == 'bottleneck':
            self.block = MultiChannelResNetBottleneck
        else:
            raise ValueError(f"Unsupported block type: {block_type}")
        
        # Build network
        self._build_network()
        
        # Initialize weights
        self._initialize_weights()
    
    def _build_network(self):
        """Build the ResNet network with optimized architecture for different input channels."""
        # Initial layers (stem) - OPTIMIZED: use different input channels!
        self.conv1 = MultiChannelConv2d(
            color_in_channels=self.color_input_channels,        # 3 for RGB
            brightness_in_channels=self.brightness_input_channels,  # 1 for brightness - efficient!
            out_channels=64, 
            kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = MultiChannelBatchNorm2d(64)
        self.activation_initial = MultiChannelActivation(self.activation, inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers with proper channel progression
        self.layer1 = self._make_layer(64, self.num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, self.num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, self.num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, self.num_blocks[3], stride=2)
        
        # Global average pooling and classifier
        self.avgpool = MultiChannelAdaptiveAvgPool2d((1, 1))
        
        # Final classifier - note we keep separate outputs for multi-channel consistency
        final_features = 512 * self.block.expansion
        self.color_classifier = nn.Linear(final_features, self.num_classes)
        self.brightness_classifier = nn.Linear(final_features, self.num_classes)
    
    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> MultiChannelSequential:
        """
        Create a ResNet layer with multiple blocks.
        
        Args:
            planes: Number of output channels for this layer
            blocks: Number of blocks in this layer
            stride: Stride for the first block (for downsampling)
            
        Returns:
            MultiChannelSequential containing all blocks
        """
        downsample = None
        
        # Create downsample if dimensions change
        if stride != 1 or self.inplanes != planes * self.block.expansion:
            downsample = MultiChannelDownsample(
                self.inplanes,
                planes * self.block.expansion,
                stride
            )
        
        layers = []
        
        # First block (might downsample)
        layers.append(self.block(
            self.inplanes, 
            planes, 
            stride, 
            downsample,
            self.activation
        ))
        
        # Update inplanes for subsequent blocks
        self.inplanes = planes * self.block.expansion
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(self.block(
                self.inplanes,
                planes,
                activation=self.activation
            ))
        
        return MultiChannelSequential(*layers)
    
    def _initialize_weights(self):
        """Initialize network weights following ResNet conventions."""
        for module in self.modules():
            if isinstance(module, MultiChannelConv2d):
                # Initialize both color and brightness conv weights
                nn.init.kaiming_normal_(module.color_weight, mode='fan_out', nonlinearity='relu')
                nn.init.kaiming_normal_(module.brightness_weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, MultiChannelBatchNorm2d):
                # Initialize both color and brightness batch norm
                nn.init.constant_(module.color_bn.weight, 1)
                nn.init.constant_(module.color_bn.bias, 0)
                nn.init.constant_(module.brightness_bn.weight, 1)
                nn.init.constant_(module.brightness_bn.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, color_input: torch.Tensor, brightness_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the ResNet network.
        
        Args:
            color_input: Color image tensor [batch_size, channels, height, width]
            brightness_input: Brightness image tensor [batch_size, channels, height, width]
            
        Returns:
            Tuple of (color_logits, brightness_logits) [batch_size, num_classes] each
        """
        # Initial layers
        color_x, brightness_x = self.conv1(color_input, brightness_input)
        color_x, brightness_x = self.bn1(color_x, brightness_x)
        color_x, brightness_x = self.activation_initial(color_x, brightness_x)
        
        # Apply maxpool to both streams
        color_x = self.maxpool(color_x)
        brightness_x = self.maxpool(brightness_x)
        
        # ResNet layers
        color_x, brightness_x = self.layer1(color_x, brightness_x)
        color_x, brightness_x = self.layer2(color_x, brightness_x)
        color_x, brightness_x = self.layer3(color_x, brightness_x)
        color_x, brightness_x = self.layer4(color_x, brightness_x)
        
        # Global average pooling
        color_x, brightness_x = self.avgpool(color_x, brightness_x)
        
        # Flatten features
        color_x = torch.flatten(color_x, 1)
        brightness_x = torch.flatten(brightness_x, 1)
        
        # Separate classification for each stream
        color_logits = self.color_classifier(color_x)
        brightness_logits = self.brightness_classifier(brightness_x)
        
        return color_logits, brightness_logits
    
    def forward_combined(self, color_input: torch.Tensor, brightness_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with combined output for standard classification.
        
        Args:
            color_input: Color image tensor
            brightness_input: Brightness image tensor
            
        Returns:
            Combined classification logits [batch_size, num_classes]
        """
        color_logits, brightness_logits = self.forward(color_input, brightness_input)
        return color_logits + brightness_logits
    
    def extract_features(self, color_input: torch.Tensor, brightness_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features before final classification."""
        # Process through all layers except classifier
        color_x, brightness_x = self.conv1(color_input, brightness_input)
        color_x, brightness_x = self.bn1(color_x, brightness_x)
        color_x, brightness_x = self.activation_initial(color_x, brightness_x)
        
        color_x = self.maxpool(color_x)
        brightness_x = self.maxpool(brightness_x)
        
        color_x, brightness_x = self.layer1(color_x, brightness_x)
        color_x, brightness_x = self.layer2(color_x, brightness_x)
        color_x, brightness_x = self.layer3(color_x, brightness_x)
        color_x, brightness_x = self.layer4(color_x, brightness_x)
        
        color_x, brightness_x = self.avgpool(color_x, brightness_x)
        
        color_x = torch.flatten(color_x, 1)
        brightness_x = torch.flatten(brightness_x, 1)
        
        return color_x, brightness_x
    
    def get_pathway_importance(self) -> Dict[str, float]:
        """Calculate pathway importance based on classifier weights."""
        with torch.no_grad():
            color_weight_norm = torch.norm(self.color_classifier.weight.data).item()
            brightness_weight_norm = torch.norm(self.brightness_classifier.weight.data).item()
            total_norm = color_weight_norm + brightness_weight_norm + 1e-8
            
            return {
                'color_pathway': color_weight_norm / total_norm,
                'brightness_pathway': brightness_weight_norm / total_norm,
                'pathway_ratio': color_weight_norm / (brightness_weight_norm + 1e-8)
            }


# Factory functions following ResNet naming conventions
def multi_channel_resnet18(num_classes: int = 10, **kwargs) -> MultiChannelResNetNetwork:
    """Create a Multi-Channel ResNet-18."""
    return MultiChannelResNetNetwork(
        num_classes=num_classes,
        num_blocks=[2, 2, 2, 2],
        block_type='basic',
        **kwargs
    )


def multi_channel_resnet34(num_classes: int = 10, **kwargs) -> MultiChannelResNetNetwork:
    """Create a Multi-Channel ResNet-34."""
    return MultiChannelResNetNetwork(
        num_classes=num_classes,
        num_blocks=[3, 4, 6, 3],
        block_type='basic',
        **kwargs
    )


def multi_channel_resnet50(num_classes: int = 10, **kwargs) -> MultiChannelResNetNetwork:
    """Create a Multi-Channel ResNet-50."""
    return MultiChannelResNetNetwork(
        num_classes=num_classes,
        num_blocks=[3, 4, 6, 3],
        block_type='bottleneck',
        **kwargs
    )


def multi_channel_resnet101(num_classes: int = 10, **kwargs) -> MultiChannelResNetNetwork:
    """Create a Multi-Channel ResNet-101."""
    return MultiChannelResNetNetwork(
        num_classes=num_classes,
        num_blocks=[3, 4, 23, 3],
        block_type='bottleneck',
        **kwargs
    )


def multi_channel_resnet152(num_classes: int = 10, **kwargs) -> MultiChannelResNetNetwork:
    """Create a Multi-Channel ResNet-152."""
    return MultiChannelResNetNetwork(
        num_classes=num_classes,
        num_blocks=[3, 8, 36, 3],
        block_type='bottleneck',
        **kwargs
    )


