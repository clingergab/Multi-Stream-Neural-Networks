"""
Neural Processing Integration Layer

This module implements integration through deep neural processing of concatenated features.
"""

import torch
import torch.nn as nn
from typing import List, Optional
from .base_integration import BaseIntegrationLayer


class NeuralProcessingLayer(BaseIntegrationLayer):
    """
    Neural Processing integration layer.
    
    Uses a deep neural network to process concatenated pathway features,
    allowing for complex non-linear integration patterns.
    
    Mathematical formulation:
        F_concat = concat(F_color, F_brightness)
        F_integrated = MLP(F_concat)
    """
    
    def __init__(
        self,
        input_channels: int,
        hidden_dims: List[int] = None,
        activation: str = "relu",
        dropout: float = 0.2,
        batch_norm: bool = True,
        residual_connections: bool = True
    ):
        super().__init__(input_channels)
        
        self.hidden_dims = hidden_dims or [512, 256, 128]
        self.activation = activation
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual_connections = residual_connections
        
        # Build neural processing network
        self.processing_network = self._build_network()
        
        # Initialize weights
        self._initialize_weights()
        
    def _build_network(self) -> nn.Module:
        """Build the neural processing network."""
        # Input has 2x channels from concatenation
        concat_channels = 2 * self.input_channels
        
        layers = []
        in_channels = concat_channels
        
        # Hidden layers
        for i, hidden_dim in enumerate(self.hidden_dims):
            # Convolutional layer
            layers.append(nn.Conv2d(in_channels, hidden_dim, 3, padding=1))
            
            # Batch normalization
            if self.batch_norm:
                layers.append(nn.BatchNorm2d(hidden_dim))
                
            # Activation
            if self.activation == "relu":
                layers.append(nn.ReLU(inplace=True))
            elif self.activation == "gelu":
                layers.append(nn.GELU())
            elif self.activation == "leaky_relu":
                layers.append(nn.LeakyReLU(0.1, inplace=True))
                
            # Dropout
            if self.dropout > 0:
                layers.append(nn.Dropout2d(self.dropout))
                
            in_channels = hidden_dim
            
        # Output layer to match input channels
        layers.append(nn.Conv2d(in_channels, self.input_channels, 1))
        
        return nn.Sequential(*layers)
        
    def forward(
        self, 
        color_features: torch.Tensor, 
        brightness_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through neural processing integration.
        
        Args:
            color_features: Color pathway features (B, C, H, W)
            brightness_features: Brightness pathway features (B, C, H, W)
            
        Returns:
            Integrated features (B, C, H, W)
        """
        # Validate inputs
        self._validate_inputs(color_features, brightness_features)
        
        # Concatenate features
        concatenated = torch.cat([color_features, brightness_features], dim=1)
        
        # Process through neural network
        processed = self.processing_network(concatenated)
        
        # Apply residual connection if enabled
        if self.residual_connections:
            # Use average of input pathways as residual
            residual = (color_features + brightness_features) / 2
            processed = processed + residual
            
        return processed
        
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.processing_network.modules():
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
            'type': 'neural_processing',
            'input_channels': self.input_channels,
            'hidden_dims': self.hidden_dims,
            'activation': self.activation,
            'dropout': self.dropout,
            'batch_norm': self.batch_norm,
            'residual_connections': self.residual_connections,
            'num_parameters': sum(p.numel() for p in self.parameters())
        }


class AttentiveNeuralProcessingLayer(NeuralProcessingLayer):
    """
    Neural Processing layer with attention mechanism.
    
    Includes attention weights to focus on important features
    during the neural processing stage.
    """
    
    def __init__(
        self,
        input_channels: int,
        hidden_dims: List[int] = None,
        activation: str = "relu",
        dropout: float = 0.2,
        batch_norm: bool = True,
        residual_connections: bool = True,
        attention_heads: int = 8
    ):
        super().__init__(
            input_channels, hidden_dims, activation, 
            dropout, batch_norm, residual_connections
        )
        
        self.attention_heads = attention_heads
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=self.input_channels,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization for attention
        self.attention_norm = nn.LayerNorm(self.input_channels)
        
    def forward(
        self, 
        color_features: torch.Tensor, 
        brightness_features: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass with attention mechanism."""
        # Validate inputs
        self._validate_inputs(color_features, brightness_features)
        
        B, C, H, W = color_features.shape
        
        # Reshape for attention (B, H*W, C)
        color_flat = color_features.view(B, C, H*W).transpose(1, 2)
        brightness_flat = brightness_features.view(B, C, H*W).transpose(1, 2)
        
        # Apply attention between pathways
        attended_color, _ = self.attention(
            color_flat, brightness_flat, brightness_flat
        )
        attended_brightness, _ = self.attention(
            brightness_flat, color_flat, color_flat
        )
        
        # Layer normalization
        attended_color = self.attention_norm(attended_color)
        attended_brightness = self.attention_norm(attended_brightness)
        
        # Reshape back to spatial format
        attended_color = attended_color.transpose(1, 2).view(B, C, H, W)
        attended_brightness = attended_brightness.transpose(1, 2).view(B, C, H, W)
        
        # Concatenate attended features
        concatenated = torch.cat([attended_color, attended_brightness], dim=1)
        
        # Process through neural network
        processed = self.processing_network(concatenated)
        
        # Apply residual connection if enabled
        if self.residual_connections:
            residual = (color_features + brightness_features) / 2
            processed = processed + residual
            
        return processed
