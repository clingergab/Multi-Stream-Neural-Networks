"""
Direct mixing models for continuous integration.
Implements various approaches to mixing multi-channel inputs.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from ..base import BaseMultiStreamModel


class ScalarMixingModel(BaseMultiStreamModel):
    """Simple scalar mixing of channels."""
    
    def __init__(
        self,
        input_channels: int = 4,
        hidden_dim: int = 128,
        num_classes: int = 10,
        mixing_weight: float = 0.5,
        **kwargs
    ):
        super().__init__(
            input_size=(input_channels, 32, 32),
            hidden_size=hidden_dim,
            num_classes=num_classes,
            **kwargs
        )
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.mixing_weight = mixing_weight
        
        # Simple feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, channels, height, width)
        
        Returns:
            Output tensor of shape (batch, num_classes)
        """
        # Apply scalar mixing to channels
        if self.input_channels == 4:  # RGBL
            rgb = x[:, :3]  # RGB channels
            l = x[:, 3:4]   # L channel
            # Mix RGB and L with scalar weight
            mixed = self.mixing_weight * rgb.mean(dim=1, keepdim=True) + (1 - self.mixing_weight) * l
            x = torch.cat([rgb, mixed], dim=1)
        
        features = self.feature_extractor(x)
        return self.classifier(features)


class ChannelAdaptiveMixingModel(BaseMultiStreamModel):
    """Channel-adaptive mixing with learned weights."""
    
    def __init__(
        self,
        input_channels: int = 4,
        hidden_dim: int = 128,
        num_classes: int = 10,
        **kwargs
    ):
        super().__init__(
            input_size=(input_channels, 32, 32),
            hidden_size=hidden_dim,
            num_classes=num_classes,
            **kwargs
        )
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Channel mixing weights
        self.channel_weights = nn.Parameter(torch.ones(input_channels))
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, channels, height, width)
        
        Returns:
            Output tensor of shape (batch, num_classes)
        """
        # Apply learned channel weights
        weights = torch.softmax(self.channel_weights, dim=0)
        x = x * weights.view(1, -1, 1, 1)
        
        features = self.feature_extractor(x)
        return self.classifier(features)


class DynamicMixingModel(BaseMultiStreamModel):
    """Dynamic mixing with attention-based weighting."""
    
    def __init__(
        self,
        input_channels: int = 4,
        hidden_dim: int = 128,
        num_classes: int = 10,
        **kwargs
    ):
        super().__init__(
            input_size=(input_channels, 32, 32),
            hidden_size=hidden_dim,
            num_classes=num_classes,
            **kwargs
        )
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Attention mechanism for dynamic mixing
        self.attention = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim // 4, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 4, input_channels, 1),
            nn.Sigmoid()
        )
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim * 2, 3, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, channels, height, width)
        
        Returns:
            Output tensor of shape (batch, num_classes)
        """
        # Compute attention weights
        attention_weights = self.attention(x)
        
        # Apply dynamic mixing
        x = x * attention_weights
        
        features = self.feature_extractor(x)
        return self.classifier(features)


class SpatialAdaptiveMixingModel(BaseMultiStreamModel):
    """Spatial-adaptive mixing with position-aware weights."""
    
    def __init__(
        self,
        input_channels: int = 4,
        hidden_dim: int = 128,
        num_classes: int = 10,
        **kwargs
    ):
        super().__init__(
            input_size=(input_channels, 32, 32),
            hidden_size=hidden_dim,
            num_classes=num_classes,
            **kwargs
        )
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Spatial mixing network
        self.spatial_mixer = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 2, input_channels, 3, padding=1),
            nn.Sigmoid()
        )
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, channels, height, width)
        
        Returns:
            Output tensor of shape (batch, num_classes)
        """
        # Compute spatial mixing weights
        spatial_weights = self.spatial_mixer(x)
        
        # Apply spatial-adaptive mixing
        x = x * spatial_weights
        
        features = self.feature_extractor(x)
        return self.classifier(features)
