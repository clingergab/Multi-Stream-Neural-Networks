"""
Attention Layers for Multi-Stream Neural Networks

This module implements various attention mechanisms for pathway integration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class MultiHeadPathwayAttention(nn.Module):
    """
    Multi-head attention mechanism for pathway integration.
    
    Computes attention weights between color and brightness pathways
    to determine optimal mixing strategies.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.dropout_layer = nn.Dropout(dropout)
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.)
            nn.init.constant_(self.k_proj.bias, 0.)
            nn.init.constant_(self.v_proj.bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
            
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of multi-head attention.
        
        Args:
            query: Query tensor (B, L, D)
            key: Key tensor (B, S, D)
            value: Value tensor (B, S, D)
            attn_mask: Attention mask (L, S)
            
        Returns:
            Tuple of (attended_output, attention_weights)
        """
        B, L, D = query.shape
        S = key.size(1)
        
        # Linear projections
        q = self.q_proj(query)  # (B, L, D)
        k = self.k_proj(key)    # (B, S, D)
        v = self.v_proj(value)  # (B, S, D)
        
        # Reshape for multi-head attention
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, D_h)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, S, D_h)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, S, D_h)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, L, S)
        
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)  # (B, H, L, S)
        attn_weights = self.dropout_layer(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # (B, H, L, D_h)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)  # (B, L, D)
        
        # Final projection
        output = self.out_proj(attn_output)
        
        return output, attn_weights.mean(dim=1)  # Average attention weights across heads


class SpatialAttentionModule(nn.Module):
    """
    Spatial attention module for feature maps.
    
    Computes spatial attention weights to focus on important regions
    in the feature maps during pathway integration.
    """
    
    def __init__(
        self,
        in_channels: int,
        reduction_ratio: int = 16
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio
        
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.channel_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of spatial attention.
        
        Args:
            x: Input feature map (B, C, H, W)
            
        Returns:
            Attended feature map (B, C, H, W)
        """
        # Channel attention
        avg_out = self.channel_attention(self.avg_pool(x))
        max_out = self.channel_attention(self.max_pool(x))
        channel_att = torch.sigmoid(avg_out + max_out)
        
        x = x * channel_att
        
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))
        
        x = x * spatial_att
        
        return x


class CrossPathwayAttention(nn.Module):
    """
    Cross-pathway attention mechanism.
    
    Allows each pathway to attend to features from the other pathway,
    enabling adaptive information exchange.
    """
    
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.channels = channels
        self.num_heads = num_heads
        
        # Attention for color pathway attending to brightness
        self.color_to_brightness_attention = MultiHeadPathwayAttention(
            channels, num_heads, dropout
        )
        
        # Attention for brightness pathway attending to color
        self.brightness_to_color_attention = MultiHeadPathwayAttention(
            channels, num_heads, dropout
        )
        
        # Layer normalization
        self.color_norm = nn.LayerNorm(channels)
        self.brightness_norm = nn.LayerNorm(channels)
        
        # Feed-forward networks
        self.color_ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 4, channels),
            nn.Dropout(dropout)
        )
        
        self.brightness_ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 4, channels),
            nn.Dropout(dropout)
        )
        
    def forward(
        self,
        color_features: torch.Tensor,
        brightness_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of cross-pathway attention.
        
        Args:
            color_features: Color pathway features (B, C, H, W)
            brightness_features: Brightness pathway features (B, C, H, W)
            
        Returns:
            Tuple of attended (color_features, brightness_features)
        """
        B, C, H, W = color_features.shape
        
        # Reshape for attention (B, H*W, C)
        color_flat = color_features.view(B, C, H*W).transpose(1, 2)
        brightness_flat = brightness_features.view(B, C, H*W).transpose(1, 2)
        
        # Cross-pathway attention
        color_attended, color_attn_weights = self.color_to_brightness_attention(
            color_flat, brightness_flat, brightness_flat
        )
        brightness_attended, brightness_attn_weights = self.brightness_to_color_attention(
            brightness_flat, color_flat, color_flat
        )
        
        # Residual connections and layer normalization
        color_attended = self.color_norm(color_flat + color_attended)
        brightness_attended = self.brightness_norm(brightness_flat + brightness_attended)
        
        # Feed-forward networks
        color_ffn_out = self.color_ffn(color_attended)
        brightness_ffn_out = self.brightness_ffn(brightness_attended)
        
        # Final residual connections
        color_output = self.color_norm(color_attended + color_ffn_out)
        brightness_output = self.brightness_norm(brightness_attended + brightness_ffn_out)
        
        # Reshape back to spatial format
        color_output = color_output.transpose(1, 2).view(B, C, H, W)
        brightness_output = brightness_output.transpose(1, 2).view(B, C, H, W)
        
        return color_output, brightness_output


class PathwayGatingAttention(nn.Module):
    """
    Pathway gating attention mechanism.
    
    Learns to gate information flow between pathways based on
    the current input characteristics.
    """
    
    def __init__(
        self,
        channels: int,
        gate_activation: str = "sigmoid"
    ):
        super().__init__()
        
        self.channels = channels
        self.gate_activation = gate_activation
        
        # Gate computation network
        self.gate_network = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2 * channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 2, 1)  # Output 2 gates
        )
        
    def forward(
        self,
        color_features: torch.Tensor,
        brightness_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of pathway gating attention.
        
        Args:
            color_features: Color pathway features (B, C, H, W)
            brightness_features: Brightness pathway features (B, C, H, W)
            
        Returns:
            Tuple of gated (color_features, brightness_features)
        """
        # Concatenate features for gate computation
        combined = torch.cat([color_features, brightness_features], dim=1)
        
        # Compute gates
        gates = self.gate_network(combined)  # (B, 2, 1, 1)
        
        if self.gate_activation == "sigmoid":
            gates = torch.sigmoid(gates)
        elif self.gate_activation == "softmax":
            gates = F.softmax(gates, dim=1)
        elif self.gate_activation == "tanh":
            gates = torch.tanh(gates)
            
        # Split gates
        color_gate = gates[:, 0:1]      # (B, 1, 1, 1)
        brightness_gate = gates[:, 1:2] # (B, 1, 1, 1)
        
        # Apply gates
        gated_color = color_features * color_gate
        gated_brightness = brightness_features * brightness_gate
        
        return gated_color, gated_brightness
