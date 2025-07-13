"""
Base integration layer for all Multi-Stream integration approaches
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any


class BaseIntegrationLayer(nn.Module, ABC):
    """
    Abstract base class for all integration layers in Multi-Stream networks.
    
    Defines the common interface and functionality for integrating color and
    brightness pathways.
    """
    
    def __init__(
        self,
        color_size: int,
        brightness_size: int,
        hidden_size: int,
        **kwargs
    ):
        super().__init__()
        self.color_size = color_size
        self.brightness_size = brightness_size
        self.hidden_size = hidden_size
        
        # Common pathway layers
        self.color_weights = nn.Linear(color_size, hidden_size)
        self.brightness_weights = nn.Linear(brightness_size, hidden_size)
        
        # Initialize pathway weights
        self._initialize_pathway_weights()
        
    def _initialize_pathway_weights(self):
        """Initialize pathway weights using Xavier/He initialization."""
        # Xavier initialization for color pathway
        nn.init.xavier_uniform_(self.color_weights.weight)
        nn.init.zeros_(self.color_weights.bias)
        
        # Xavier initialization for brightness pathway
        nn.init.xavier_uniform_(self.brightness_weights.weight)
        nn.init.zeros_(self.brightness_weights.bias)
    
    @abstractmethod
    def forward(
        self,
        color_input: torch.Tensor,
        brightness_input: torch.Tensor,
        prev_integrated: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the integration layer.
        
        Args:
            color_input: Color pathway input
            brightness_input: Brightness pathway input
            prev_integrated: Previous layer's integrated features (optional)
            
        Returns:
            Tuple of (color_output, brightness_output, integrated_output)
        """
        pass
    
    def get_pathway_outputs(
        self,
        color_input: torch.Tensor,
        brightness_input: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get individual pathway outputs before integration.
        
        Args:
            color_input: Color pathway input
            brightness_input: Brightness pathway input
            
        Returns:
            Tuple of (color_output, brightness_output)
        """
        color_output = torch.relu(self.color_weights(color_input))
        brightness_output = torch.relu(self.brightness_weights(brightness_input))
        
        return color_output, brightness_output
    
    def get_integration_weights(self) -> Dict[str, torch.Tensor]:
        """
        Get current integration weights for analysis.
        
        Returns:
            Dictionary mapping weight names to tensors
        """
        # Default implementation - subclasses should override
        return {}
    
    def compute_pathway_importance(
        self,
        color_input: torch.Tensor,
        brightness_input: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute relative importance of pathways for given inputs.
        
        Args:
            color_input: Color pathway input
            brightness_input: Brightness pathway input
            
        Returns:
            Dictionary with pathway importance scores
        """
        # Default implementation - subclasses should override
        return {'color': 0.5, 'brightness': 0.5}
    
    def extra_repr(self) -> str:
        """Extra representation for debugging."""
        return f'color_size={self.color_size}, brightness_size={self.brightness_size}, hidden_size={self.hidden_size}'
