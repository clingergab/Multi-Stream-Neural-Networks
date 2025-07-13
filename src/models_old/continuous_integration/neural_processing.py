"""Neural processing integration model for multi-stream neural networks."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
from ..base import BaseMultiStreamModel


class NeuralProcessingModel(BaseMultiStreamModel):
    """
    Multi-stream model using neural processing for feature integration.
    
    This model processes multiple input channels and uses learnable neural
    networks to process and integrate the features.
    """
    
    def __init__(
        self,
        input_channels: int = 4,
        num_classes: int = 10,
        hidden_dim: int = 512,
        processing_layers: int = 3,
        dropout: float = 0.1,
        activation: str = "relu",
        **kwargs
    ):
        super().__init__(
            num_classes=num_classes,
            **kwargs
        )
        
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.processing_layers = processing_layers
        self.dropout = dropout
        self.activation = activation
        
        # Activation function
        if activation == "relu":
            self.act_fn = nn.ReLU()
        elif activation == "gelu":
            self.act_fn = nn.GELU()
        elif activation == "swish":
            self.act_fn = nn.SiLU()
        else:
            self.act_fn = nn.ReLU()
        
        # Feature extraction backbone
        self.backbone = self._build_backbone()
        
        # Neural processing layers
        self.processing = self._build_processing_layers()
        
        # Final classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        self._initialize_weights()
    
    def _build_backbone(self) -> nn.Module:
        """Build the feature extraction backbone."""
        return nn.Sequential(
            nn.Conv2d(self.input_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            self.act_fn,
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            self.act_fn,
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            self.act_fn,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, self.hidden_dim)
        )
    
    def _build_processing_layers(self) -> nn.Module:
        """Build the neural processing layers."""
        layers = []
        
        for i in range(self.processing_layers):
            layers.extend([
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim),
                self.act_fn,
                nn.Dropout(self.dropout)
            ])
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
        
        Returns:
            Classification logits of shape (batch, num_classes)
        """
        # Feature extraction
        features = self.backbone(x)
        
        # Neural processing
        processed_features = self.processing(features)
        
        # Classification
        logits = self.classifier(processed_features)
        
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
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def get_pathway_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get intermediate features from different pathways.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
        
        Returns:
            Dictionary with pathway features
        """
        # Feature extraction
        features = self.backbone(x)
        
        # Processed features
        processed_features = self.processing(features)
        
        return {
            'raw_features': features,
            'processed_features': processed_features
        }
