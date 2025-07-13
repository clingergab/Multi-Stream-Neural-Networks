"""Attention-based integration model for multi-stream neural networks."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
from ..base import BaseMultiStreamModel


class AttentionBasedModel(BaseMultiStreamModel):
    """
    Multi-stream model using attention mechanisms for feature integration.
    
    This model processes color and brightness streams separately, then uses
    attention mechanisms to integrate features from both pathways.
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 10,
        hidden_dim: int = 512,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_type: str = "multihead",
        **kwargs
    ):
        super().__init__(
            num_classes=num_classes,
            **kwargs
        )
        
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.attention_type = attention_type
        
        # Color stream pathway
        self.color_backbone = self._build_backbone(input_channels)
        
        # Brightness stream pathway
        self.brightness_backbone = self._build_backbone(1)
        
        # Feature projection to same dimension
        self.color_proj = nn.Linear(512, hidden_dim)
        self.brightness_proj = nn.Linear(512, hidden_dim)
        
        # Attention mechanism
        if attention_type == "multihead":
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_attention_heads,
                dropout=dropout,
                batch_first=True
            )
        elif attention_type == "cross":
            self.cross_attention = CrossAttention(hidden_dim, num_attention_heads, dropout)
        
        # Output layers
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.dropout_layer = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        self._initialize_weights()
    
    def _build_backbone(self, in_channels: int) -> nn.Module:
        """Build a simple CNN backbone for feature extraction."""
        return nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            
            nn.Flatten(),
            nn.Linear(256, 512)
        )
    
    def forward(self, color_input: torch.Tensor, brightness_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the attention-based model.
        
        Args:
            color_input: RGB input tensor [B, 3, H, W]
            brightness_input: Brightness input tensor [B, 1, H, W] or [B, 3, H, W]
            
        Returns:
            Classification logits [B, num_classes]
        """
        # Extract features from each stream
        color_features = self.color_backbone(color_input)  # [B, 512]
        
        # Handle brightness input
        if brightness_input.size(1) == 3:
            brightness_input = brightness_input.mean(dim=1, keepdim=True)
        brightness_features = self.brightness_backbone(brightness_input)  # [B, 512]
        
        # Project to same dimension
        color_proj = self.color_proj(color_features)      # [B, hidden_dim]
        brightness_proj = self.brightness_proj(brightness_features)  # [B, hidden_dim]
        
        # Prepare for attention (add sequence dimension)
        features = torch.stack([color_proj, brightness_proj], dim=1)  # [B, 2, hidden_dim]
        
        # Apply attention
        if self.attention_type == "multihead":
            attended_features, attention_weights = self.attention(
                features, features, features
            )  # [B, 2, hidden_dim]
            
            # Aggregate attended features (mean over sequence)
            integrated_features = attended_features.mean(dim=1)  # [B, hidden_dim]
            
        elif self.attention_type == "cross":
            integrated_features = self.cross_attention(color_proj, brightness_proj)
        
        # Apply output layers
        integrated_features = self.output_norm(integrated_features)
        integrated_features = self.dropout_layer(integrated_features)
        
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


class CrossAttention(nn.Module):
    """Cross-attention mechanism between two feature sets."""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, query_features: torch.Tensor, key_value_features: torch.Tensor) -> torch.Tensor:
        """
        Apply cross-attention between query and key-value features.
        
        Args:
            query_features: Query features [B, hidden_dim]
            key_value_features: Key-Value features [B, hidden_dim]
            
        Returns:
            Attended features [B, hidden_dim]
        """
        batch_size = query_features.size(0)
        
        # Linear transformations
        Q = self.q_linear(query_features).view(batch_size, self.num_heads, self.head_dim)
        K = self.k_linear(key_value_features).view(batch_size, self.num_heads, self.head_dim)
        V = self.v_linear(key_value_features).view(batch_size, self.num_heads, self.head_dim)
        
        # Attention computation
        scores = torch.einsum('bhd,bhd->bh', Q, K) / (self.head_dim ** 0.5)  # [B, num_heads]
        attention_weights = F.softmax(scores, dim=-1)  # [B, num_heads]
        
        # Apply attention to values
        attended = torch.einsum('bh,bhd->bhd', attention_weights, V)  # [B, num_heads, head_dim]
        attended = attended.view(batch_size, -1)  # [B, hidden_dim]
        
        # Output linear transformation
        output = self.out_linear(attended)
        
        # Residual connection with query
        return output + query_features
