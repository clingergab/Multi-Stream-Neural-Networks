"""
Direct mixing integration layers - all variants
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict

from .base_integration import BaseIntegrationLayer


class BaseDirectMixingLayer(BaseIntegrationLayer):
    """
    Base class for all direct mixing integration layers.
    
    Implements the common functionality for direct mixing approaches
    where integration follows: I_l = α·C_l + β·B_l + γ·I_{l-1}
    """
    
    def __init__(
        self,
        color_size: int,
        brightness_size: int,
        hidden_size: int,
        alpha_init: float = 1.0,
        beta_init: float = 1.0,
        gamma_init: float = 0.2,
        **kwargs
    ):
        super().__init__(color_size, brightness_size, hidden_size, **kwargs)
        self.alpha_init = alpha_init
        self.beta_init = beta_init
        self.gamma_init = gamma_init
        
    def _apply_regularization(
        self,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        gamma: Optional[torch.Tensor] = None,
        min_val: float = 0.01
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Apply clamping regularization to prevent pathway collapse."""
        alpha_reg = torch.clamp(alpha, min=min_val)
        beta_reg = torch.clamp(beta, min=min_val)
        gamma_reg = torch.clamp(gamma, min=min_val) if gamma is not None else None
        
        return alpha_reg, beta_reg, gamma_reg


class ScalarMixingLayer(BaseDirectMixingLayer):
    """
    Scalar direct mixing layer using global α, β, γ parameters.
    
    Uses single scalar values for mixing across all neurons in the layer.
    Most interpretable but least flexible variant.
    """
    
    def __init__(
        self,
        color_size: int,
        brightness_size: int,
        hidden_size: int,
        alpha_init: float = 1.0,
        beta_init: float = 1.0,
        gamma_init: float = 0.2,
        **kwargs
    ):
        super().__init__(color_size, brightness_size, hidden_size, alpha_init, beta_init, gamma_init, **kwargs)
        
        # Learnable scalar mixing weights
        self.alpha = nn.Parameter(torch.normal(alpha_init, 0.1, (1,)))
        self.beta = nn.Parameter(torch.normal(beta_init, 0.1, (1,)))
        self.gamma = nn.Parameter(torch.normal(gamma_init, 0.02, (1,)))
    
    def forward(
        self,
        color_input: torch.Tensor,
        brightness_input: torch.Tensor,
        prev_integrated: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with scalar mixing."""
        # Process pathways independently
        color_output, brightness_output = self.get_pathway_outputs(color_input, brightness_input)
        
        # Apply regularization
        alpha_reg, beta_reg, gamma_reg = self._apply_regularization(self.alpha, self.beta, self.gamma)
        
        # Direct mixing
        if prev_integrated is not None:
            integrated_output = (alpha_reg * color_output + 
                               beta_reg * brightness_output + 
                               gamma_reg * prev_integrated)
        else:
            integrated_output = alpha_reg * color_output + beta_reg * brightness_output
        
        return color_output, brightness_output, integrated_output
    
    def get_integration_weights(self) -> Dict[str, torch.Tensor]:
        """Get current scalar mixing weights."""
        return {
            'alpha': self.alpha.clone(),
            'beta': self.beta.clone(),
            'gamma': self.gamma.clone()
        }
    
    def compute_pathway_importance(
        self,
        color_input: torch.Tensor,
        brightness_input: torch.Tensor
    ) -> Dict[str, float]:
        """Compute pathway importance based on mixing weights."""
        alpha_val = torch.abs(self.alpha).item()
        beta_val = torch.abs(self.beta).item()
        total = alpha_val + beta_val
        
        return {
            'color': alpha_val / total if total > 0 else 0.5,
            'brightness': beta_val / total if total > 0 else 0.5
        }


class ChannelAdaptiveMixingLayer(BaseDirectMixingLayer):
    """
    Channel-wise adaptive mixing layer.
    
    Each channel/neuron has its own α, β, γ parameters, allowing
    different features to have different pathway preferences.
    """
    
    def __init__(
        self,
        color_size: int,
        brightness_size: int,
        hidden_size: int,
        alpha_init: float = 1.0,
        beta_init: float = 1.0,
        gamma_init: float = 0.2,
        **kwargs
    ):
        super().__init__(color_size, brightness_size, hidden_size, alpha_init, beta_init, gamma_init, **kwargs)
        
        # Channel-wise mixing weights (vectors, not scalars)
        self.alpha = nn.Parameter(torch.ones(hidden_size) * alpha_init)
        self.beta = nn.Parameter(torch.ones(hidden_size) * beta_init)
        self.gamma = nn.Parameter(torch.ones(hidden_size) * gamma_init)
    
    def forward(
        self,
        color_input: torch.Tensor,
        brightness_input: torch.Tensor,
        prev_integrated: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with channel-wise mixing."""
        # Process pathways independently
        color_output, brightness_output = self.get_pathway_outputs(color_input, brightness_input)
        
        # Apply regularization
        alpha_reg, beta_reg, gamma_reg = self._apply_regularization(self.alpha, self.beta, self.gamma)
        
        # Channel-wise mixing
        if prev_integrated is not None:
            integrated_output = (alpha_reg * color_output + 
                               beta_reg * brightness_output + 
                               gamma_reg * prev_integrated)
        else:
            integrated_output = alpha_reg * color_output + beta_reg * brightness_output
        
        return color_output, brightness_output, integrated_output
    
    def get_integration_weights(self) -> Dict[str, torch.Tensor]:
        """Get current channel-wise mixing weights."""
        return {
            'alpha': self.alpha.clone(),
            'beta': self.beta.clone(),
            'gamma': self.gamma.clone()
        }
    
    def compute_pathway_importance(
        self,
        color_input: torch.Tensor,
        brightness_input: torch.Tensor
    ) -> Dict[str, float]:
        """Compute average pathway importance across channels."""
        alpha_mean = torch.abs(self.alpha).mean().item()
        beta_mean = torch.abs(self.beta).mean().item()
        total = alpha_mean + beta_mean
        
        return {
            'color': alpha_mean / total if total > 0 else 0.5,
            'brightness': beta_mean / total if total > 0 else 0.5
        }


class DynamicMixingLayer(BaseDirectMixingLayer):
    """
    Dynamic input-dependent mixing layer.
    
    Generates mixing weights dynamically based on current input features
    using a small neural network weight generator.
    """
    
    def __init__(
        self,
        color_size: int,
        brightness_size: int,
        hidden_size: int,
        **kwargs
    ):
        super().__init__(color_size, brightness_size, hidden_size, **kwargs)
        
        # Dynamic weight generator network
        self.weight_generator = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 3),  # Outputs α, β, γ
            nn.Softmax(dim=1)  # Ensures weights sum to 1
        )
        
        # Initialize weight generator
        self._initialize_weight_generator()
    
    def _initialize_weight_generator(self):
        """Initialize the weight generator network."""
        for module in self.weight_generator.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        color_input: torch.Tensor,
        brightness_input: torch.Tensor,
        prev_integrated: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with dynamic mixing."""
        # Process pathways independently
        color_output, brightness_output = self.get_pathway_outputs(color_input, brightness_input)
        
        # Generate dynamic weights based on current features
        if prev_integrated is not None:
            features = torch.cat([color_output, brightness_output, prev_integrated], dim=1)
            weights = self.weight_generator(features)
            alpha = weights[:, 0:1]
            beta = weights[:, 1:2]
            gamma = weights[:, 2:3]
            
            integrated_output = (alpha * color_output + 
                               beta * brightness_output + 
                               gamma * prev_integrated)
        else:
            features = torch.cat([color_output, brightness_output], dim=1)
            weights = self.weight_generator(features)
            alpha = weights[:, 0:1]
            beta = weights[:, 1:2]
            
            integrated_output = alpha * color_output + beta * brightness_output
        
        return color_output, brightness_output, integrated_output
    
    def get_integration_weights(self) -> Dict[str, torch.Tensor]:
        """Get weight generator parameters (not dynamic weights)."""
        return {param_name: param.clone() for param_name, param in self.weight_generator.named_parameters()}
    
    def compute_pathway_importance(
        self,
        color_input: torch.Tensor,
        brightness_input: torch.Tensor
    ) -> Dict[str, float]:
        """Compute pathway importance for given inputs."""
        with torch.no_grad():
            color_output, brightness_output = self.get_pathway_outputs(color_input, brightness_input)
            features = torch.cat([color_output, brightness_output], dim=1)
            weights = self.weight_generator(features)
            
            alpha_mean = weights[:, 0].mean().item()
            beta_mean = weights[:, 1].mean().item()
            
            return {
                'color': alpha_mean,
                'brightness': beta_mean
            }


class SpatialAdaptiveMixingLayer(BaseDirectMixingLayer):
    """
    Spatial adaptive mixing layer for convolutional architectures.
    
    Different spatial locations have different mixing parameters,
    creating spatial attention maps over pathways.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        height: int,
        width: int,
        alpha_init: float = 1.0,
        beta_init: float = 1.0,
        gamma_init: float = 0.2,
        **kwargs
    ):
        # Note: For spatial layers, we use different parameter names
        super().__init__(in_channels, 1, out_channels, alpha_init, beta_init, gamma_init, **kwargs)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.height = height
        self.width = width
        
        # Convolutional pathways (preserve spatial structure)
        self.color_conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.brightness_conv = nn.Conv2d(1, out_channels, 3, padding=1)
        
        # Spatial integration weights
        self.alpha = nn.Parameter(torch.ones(1, 1, height, width) * alpha_init)
        self.beta = nn.Parameter(torch.ones(1, 1, height, width) * beta_init)
        self.gamma = nn.Parameter(torch.ones(1, 1, height, width) * gamma_init)
        
        # Initialize conv layers
        self._initialize_conv_layers()
    
    def _initialize_conv_layers(self):
        """Initialize convolutional layers."""
        nn.init.xavier_uniform_(self.color_conv.weight)
        nn.init.zeros_(self.color_conv.bias)
        nn.init.xavier_uniform_(self.brightness_conv.weight)
        nn.init.zeros_(self.brightness_conv.bias)
    
    def forward(
        self,
        color_input: torch.Tensor,
        brightness_input: torch.Tensor,
        prev_integrated: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with spatial mixing."""
        # Process pathways (maintaining spatial dimensions)
        color_output = F.relu(self.color_conv(color_input))
        brightness_output = F.relu(self.brightness_conv(brightness_input))
        
        # Apply regularization
        alpha_reg, beta_reg, gamma_reg = self._apply_regularization(self.alpha, self.beta, self.gamma)
        
        # Spatial mixing
        if prev_integrated is not None:
            integrated_output = (alpha_reg * color_output + 
                               beta_reg * brightness_output + 
                               gamma_reg * prev_integrated)
        else:
            integrated_output = alpha_reg * color_output + beta_reg * brightness_output
        
        return color_output, brightness_output, integrated_output
    
    def get_integration_weights(self) -> Dict[str, torch.Tensor]:
        """Get current spatial mixing weights."""
        return {
            'alpha': self.alpha.clone(),
            'beta': self.beta.clone(),
            'gamma': self.gamma.clone()
        }
    
    def compute_pathway_importance(
        self,
        color_input: torch.Tensor,
        brightness_input: torch.Tensor
    ) -> Dict[str, float]:
        """Compute average pathway importance across spatial locations."""
        alpha_mean = torch.abs(self.alpha).mean().item()
        beta_mean = torch.abs(self.beta).mean().item()
        total = alpha_mean + beta_mean
        
        return {
            'color': alpha_mean / total if total > 0 else 0.5,
            'brightness': beta_mean / total if total > 0 else 0.5
        }
