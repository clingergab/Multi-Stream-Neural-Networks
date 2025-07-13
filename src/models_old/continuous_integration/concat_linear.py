"""Concatenation + Linear integration model for multi-stream neural networks."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
from ..base import BaseMultiStreamModel


class ConcatLinearModel(BaseMultiStreamModel):
    """
    Multi-stream model using concatenation followed by linear transformation.
    
    This model processes color and brightness streams separately, then concatenates
    the features and applies linear transformations for integration.
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 10,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        batch_norm: bool = True,
        activation: str = "relu",
        concat_dim: int = 1,
        feature_alignment: bool = True,
        **kwargs
    ):
        super().__init__(
            num_classes=num_classes,
            **kwargs
        )
        
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.activation = activation
        self.concat_dim = concat_dim
        self.feature_alignment = feature_alignment
        
        # Activation function
        if activation == "relu":
            self.act_fn = nn.ReLU()
        elif activation == "gelu":
            self.act_fn = nn.GELU()
        elif activation == "swish":
            self.act_fn = nn.SiLU()
        else:
            self.act_fn = nn.ReLU()
        
        # Color stream pathway (RGB)
        self.color_backbone = self._build_backbone(input_channels)
        
        # Brightness stream pathway (single channel)
        self.brightness_backbone = self._build_backbone(1)
        
        # Feature alignment if needed
        if feature_alignment:
            self.color_align = nn.Linear(512, hidden_dim // 2)
            self.brightness_align = nn.Linear(512, hidden_dim // 2)
        
        # Concatenation + Linear integration
        concat_features = hidden_dim if feature_alignment else 1024
        
        self.integration_layers = nn.Sequential()
        
        # First linear layer after concatenation
        self.integration_layers.add_module("linear1", nn.Linear(concat_features, hidden_dim))
        
        if batch_norm:
            self.integration_layers.add_module("bn1", nn.BatchNorm1d(hidden_dim))
        
        self.integration_layers.add_module("act1", self.act_fn)
        
        if dropout > 0:
            self.integration_layers.add_module("dropout1", nn.Dropout(dropout))
        
        # Second linear layer
        self.integration_layers.add_module("linear2", nn.Linear(hidden_dim, hidden_dim // 2))
        
        if batch_norm:
            self.integration_layers.add_module("bn2", nn.BatchNorm1d(hidden_dim // 2))
        
        self.integration_layers.add_module("act2", self.act_fn)
        
        if dropout > 0:
            self.integration_layers.add_module("dropout2", nn.Dropout(dropout))
        
        # Final classifier
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _build_backbone(self, in_channels: int) -> nn.Module:
        """Build a simple CNN backbone for feature extraction."""
        return nn.Sequential(
            # Conv Block 1
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            self.act_fn,
            nn.MaxPool2d(2),
            
            # Conv Block 2  
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            self.act_fn,
            nn.MaxPool2d(2),
            
            # Conv Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            self.act_fn,
            nn.MaxPool2d(2),
            
            # Conv Block 4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            self.act_fn,
            nn.AdaptiveAvgPool2d((1, 1)),
            
            # Flatten
            nn.Flatten(),
            nn.Linear(256, 512)
        )
    
    def forward(self, color_input: torch.Tensor, brightness_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the concatenation + linear model.
        
        Args:
            color_input: RGB input tensor [B, 3, H, W]
            brightness_input: Brightness input tensor [B, 1, H, W] or [B, 3, H, W]
            
        Returns:
            Classification logits [B, num_classes]
        """
        # Extract features from each stream
        color_features = self.color_backbone(color_input)  # [B, 512]
        
        # Handle brightness input (convert to single channel if needed)
        if brightness_input.size(1) == 3:
            # Convert RGB brightness to single channel (take mean)
            brightness_input = brightness_input.mean(dim=1, keepdim=True)
        
        brightness_features = self.brightness_backbone(brightness_input)  # [B, 512]
        
        # Feature alignment
        if self.feature_alignment:
            color_features = self.color_align(color_features)      # [B, hidden_dim//2]
            brightness_features = self.brightness_align(brightness_features)  # [B, hidden_dim//2]
        
        # Concatenate features
        if self.concat_dim == 1:
            concatenated = torch.cat([color_features, brightness_features], dim=1)
        else:
            concatenated = torch.stack([color_features, brightness_features], dim=1)
            concatenated = concatenated.view(concatenated.size(0), -1)
        
        # Apply integration layers
        integrated_features = self.integration_layers(concatenated)
        
        # Final classification
        logits = self.classifier(integrated_features)
        
        return logits
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def get_pathway_features(self, color_input: torch.Tensor, brightness_input: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract features from individual pathways for analysis.
        
        Returns:
            Dictionary with pathway features
        """
        with torch.no_grad():
            color_features = self.color_backbone(color_input)
            
            if brightness_input.size(1) == 3:
                brightness_input = brightness_input.mean(dim=1, keepdim=True)
            brightness_features = self.brightness_backbone(brightness_input)
            
            if self.feature_alignment:
                color_features = self.color_align(color_features)
                brightness_features = self.brightness_align(brightness_features)
            
            return {
                'color_features': color_features,
                'brightness_features': brightness_features,
                'concatenated': torch.cat([color_features, brightness_features], dim=1)
            }
    
    def compute_pathway_importance(self, color_input: torch.Tensor, brightness_input: torch.Tensor) -> Dict[str, float]:
        """
        Compute the relative importance of each pathway.
        
        Returns:
            Dictionary with importance scores
        """
        features = self.get_pathway_features(color_input, brightness_input)
        
        color_norm = torch.norm(features['color_features'], dim=1).mean().item()
        brightness_norm = torch.norm(features['brightness_features'], dim=1).mean().item()
        
        total_norm = color_norm + brightness_norm
        
        return {
            'color_importance': color_norm / total_norm if total_norm > 0 else 0.5,
            'brightness_importance': brightness_norm / total_norm if total_norm > 0 else 0.5
        }
