"""
Fusion strategies for multi-stream neural networks.

This module provides different strategies for fusing features from multiple streams
(e.g., RGB and depth) in a modular, plug-and-play manner.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple


class BaseFusion(nn.Module, ABC):
    """
    Abstract base class for fusion strategies.

    All fusion strategies must implement the forward method with a consistent interface.
    """

    def __init__(self, feature_dim: int):
        """
        Initialize the fusion module.

        Args:
            feature_dim: Dimension of features from each stream
        """
        super().__init__()
        self.feature_dim = feature_dim

    @abstractmethod
    def forward(self, stream1_features: torch.Tensor, stream2_features: torch.Tensor) -> torch.Tensor:
        """
        Fuse features from two streams.

        Args:
            stream1_features: Features from first stream [batch_size, feature_dim]
            stream2_features: Features from second stream [batch_size, feature_dim]

        Returns:
            Fused features tensor
        """
        pass

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Return the output dimension of fused features."""
        pass


class ConcatFusion(BaseFusion):
    """
    Simple concatenation fusion (baseline).

    Concatenates features from both streams along the feature dimension.
    This is the default/baseline fusion strategy.
    """

    def __init__(self, feature_dim: int):
        """
        Initialize concatenation fusion.

        Args:
            feature_dim: Dimension of features from each stream
        """
        super().__init__(feature_dim)

    def forward(self, stream1_features: torch.Tensor, stream2_features: torch.Tensor) -> torch.Tensor:
        """
        Concatenate features from both streams.

        Args:
            stream1_features: Features from first stream [batch_size, feature_dim]
            stream2_features: Features from second stream [batch_size, feature_dim]

        Returns:
            Concatenated features [batch_size, 2 * feature_dim]
        """
        return torch.cat([stream1_features, stream2_features], dim=1)

    @property
    def output_dim(self) -> int:
        """Output dimension is 2x feature_dim due to concatenation."""
        return self.feature_dim * 2


class WeightedFusion(BaseFusion):
    """
    Learned weighted fusion with normalized weights.

    Learns relative importance weights for each stream using softmax normalization.
    This ensures weights sum to 1 and prevents scale drift while allowing the model
    to emphasize one stream over another based on their relative contributions.
    """

    def __init__(self, feature_dim: int):
        """
        Initialize weighted fusion.

        Args:
            feature_dim: Dimension of features from each stream
        """
        super().__init__(feature_dim)

        # Learnable logits for softmax weighting (initialized to 0 = equal weights)
        # After softmax: [0.5, 0.5] - both streams contribute equally at start
        self.logits = nn.Parameter(torch.zeros(2))

    def forward(self, stream1_features: torch.Tensor, stream2_features: torch.Tensor) -> torch.Tensor:
        """
        Apply learned normalized weights and concatenate.

        Args:
            stream1_features: Features from first stream [batch_size, feature_dim]
            stream2_features: Features from second stream [batch_size, feature_dim]

        Returns:
            Weighted and concatenated features [batch_size, 2 * feature_dim]
        """
        # Compute normalized weights using softmax (sum to 1)
        weights = torch.nn.functional.softmax(self.logits, dim=0)
        weight1, weight2 = weights[0], weights[1]

        # Scale by 2 to preserve feature magnitude
        # (since weights sum to 1, scaling by 2 keeps average contribution at 1.0)
        weighted_stream1 = stream1_features * (2 * weight1)
        weighted_stream2 = stream2_features * (2 * weight2)

        # Concatenate weighted features
        return torch.cat([weighted_stream1, weighted_stream2], dim=1)

    @property
    def output_dim(self) -> int:
        """Output dimension is 2x feature_dim due to concatenation."""
        return self.feature_dim * 2


class GatedFusion(BaseFusion):
    """
    Gated fusion with learned adaptive weighting.

    Uses a small MLP to compute sample-adaptive gate weights for each stream.
    This allows the model to dynamically adjust which stream to emphasize
    based on the input features themselves.
    """

    def __init__(self, feature_dim: int):
        """
        Initialize gated fusion.

        Args:
            feature_dim: Dimension of features from each stream
        """
        super().__init__(feature_dim)

        # Gate network: takes concatenated features, outputs 2 weights
        self.gate_network = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 2),
            nn.Softmax(dim=1)  # Normalize weights to sum to 1
        )

    def forward(self, stream1_features: torch.Tensor, stream2_features: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive gates and apply to features.

        Args:
            stream1_features: Features from first stream [batch_size, feature_dim]
            stream2_features: Features from second stream [batch_size, feature_dim]

        Returns:
            Gated and concatenated features [batch_size, 2 * feature_dim]
        """
        # Concatenate features for gate computation
        combined = torch.cat([stream1_features, stream2_features], dim=1)

        # Compute gate weights [batch_size, 2]
        gate_weights = self.gate_network(combined)

        # Apply gates to features
        # gate_weights[:, 0:1] has shape [batch_size, 1]
        # Broadcasting will apply it across feature_dim
        gated_stream1 = stream1_features * gate_weights[:, 0:1]
        gated_stream2 = stream2_features * gate_weights[:, 1:2]

        # Concatenate gated features
        return torch.cat([gated_stream1, gated_stream2], dim=1)

    @property
    def output_dim(self) -> int:
        """Output dimension is 2x feature_dim due to concatenation."""
        return self.feature_dim * 2


# Factory function for easy fusion creation
def create_fusion(fusion_type: str, feature_dim: int) -> BaseFusion:
    """
    Factory function to create fusion modules.

    Args:
        fusion_type: Type of fusion ('concat', 'weighted', 'gated')
        feature_dim: Dimension of features from each stream

    Returns:
        Fusion module instance

    Raises:
        ValueError: If fusion_type is not recognized
    """
    fusion_types = {
        'concat': ConcatFusion,
        'weighted': WeightedFusion,
        'gated': GatedFusion
    }

    if fusion_type not in fusion_types:
        raise ValueError(
            f"Unknown fusion type: {fusion_type}. "
            f"Supported types: {list(fusion_types.keys())}"
        )

    return fusion_types[fusion_type](feature_dim)
