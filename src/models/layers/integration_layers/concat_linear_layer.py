"""
Concatenation + Linear Integration Layer

This module implements integration through concatenation followed by linear transformation.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from .base_integration import BaseIntegrationLayer


class ConcatLinearLayer(BaseIntegrationLayer):
    """
    Concatenation + Linear integration layer.
    
    Concatenates color and brightness features along channel dimension,
    then applies a linear transformation to integrate them.
    
    Mathematical formulation:
        F_concat = concat(F_color, F_brightness)
        F_integrated = W * F_concat + b
    """
    
    def __init__(
        self,
        input_channels: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        batch_norm: bool = True,
        activation: str = "relu"
    ):
        super().__init__(input_channels)
        
        self.hidden_dim = hidden_dim or input_channels
        self.dropout = dropout
        self.batch_norm = batch_norm
        
        # Concatenated input has 2x channels
        concat_channels = 2 * input_channels
        
        # Linear integration layers
        layers = []
        
        # First linear layer
        layers.append(nn.Conv2d(concat_channels, self.hidden_dim, 1))
        
        if batch_norm:
            layers.append(nn.BatchNorm2d(self.hidden_dim))
            
        # Activation
        if activation == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif activation == "gelu":
            layers.append(nn.GELU())
        elif activation == "leaky_relu":
            layers.append(nn.LeakyReLU(0.1, inplace=True))
            
        # Dropout
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
            
        # Output layer
        layers.append(nn.Conv2d(self.hidden_dim, input_channels, 1))
        
        self.integration_network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
        
    def forward(
        self, 
        color_features: torch.Tensor, 
        brightness_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through concatenation + linear integration.
        
        Args:
            color_features: Color pathway features (B, C, H, W)
            brightness_features: Brightness pathway features (B, C, H, W)
            
        Returns:
            Integrated features (B, C, H, W)
        """
        # Validate inputs
        self._validate_inputs(color_features, brightness_features)
        
        # Concatenate along channel dimension
        concatenated = torch.cat([color_features, brightness_features], dim=1)
        
        # Apply integration network
        integrated = self.integration_network(concatenated)
        
        return integrated
        
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.integration_network.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
                
    def get_integration_info(self) -> dict:
        """Get information about the integration layer."""
        return {
            'type': 'concat_linear',
            'input_channels': self.input_channels,
            'hidden_dim': self.hidden_dim,
            'dropout': self.dropout,
            'batch_norm': self.batch_norm,
            'num_parameters': sum(p.numel() for p in self.parameters())
        }


class AdaptiveConcatLinearLayer(ConcatLinearLayer):
    """
    Adaptive Concatenation + Linear layer with learned feature alignment.
    
    Includes feature alignment modules before concatenation to ensure
    optimal feature compatibility.
    """
    
    def __init__(
        self,
        input_channels: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        batch_norm: bool = True,
        activation: str = "relu",
        feature_alignment: bool = True
    ):
        super().__init__(input_channels, hidden_dim, dropout, batch_norm, activation)
        
        self.feature_alignment = feature_alignment
        
        if feature_alignment:
            # Feature alignment networks
            self.color_alignment = nn.Sequential(
                nn.Conv2d(input_channels, input_channels, 1),
                nn.BatchNorm2d(input_channels) if batch_norm else nn.Identity(),
                nn.ReLU(inplace=True)
            )
            
            self.brightness_alignment = nn.Sequential(
                nn.Conv2d(input_channels, input_channels, 1),
                nn.BatchNorm2d(input_channels) if batch_norm else nn.Identity(),
                nn.ReLU(inplace=True)
            )
            
    def forward(
        self, 
        color_features: torch.Tensor, 
        brightness_features: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass with feature alignment."""
        # Validate inputs
        self._validate_inputs(color_features, brightness_features)
        
        # Apply feature alignment if enabled
        if self.feature_alignment:
            color_aligned = self.color_alignment(color_features)
            brightness_aligned = self.brightness_alignment(brightness_features)
        else:
            color_aligned = color_features
            brightness_aligned = brightness_features
            
        # Concatenate aligned features
        concatenated = torch.cat([color_aligned, brightness_aligned], dim=1)
        
        # Apply integration network
        integrated = self.integration_network(concatenated)
        
        return integrated
