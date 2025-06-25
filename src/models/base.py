"""
Base classes for Multi-Stream Neural Networks
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any, List


class BaseMultiStreamModel(nn.Module, ABC):
    """
    Base class for all Multi-Stream Neural Network models.
    
    Provides common functionality and interface for MSNN architectures.
    """
    
    def __init__(
        self,
        input_size: Tuple[int, ...],
        hidden_size: int,
        num_classes: int,
        **kwargs
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        # Track pathway gradients for analysis
        self.pathway_gradients = {}
        
    @abstractmethod
    def forward(self, color_input: torch.Tensor, brightness_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the multi-stream network.
        
        Args:
            color_input: Color/RGB input tensor
            brightness_input: Brightness/luminance input tensor
            
        Returns:
            Tuple of (color_output, brightness_output)
        """
        pass
    
    def extract_color_brightness(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract color (RGB) and brightness (L) channels from input.
        
        Args:
            x: Input tensor of shape (B, 4, H, W) containing RGB+L channels
            
        Returns:
            Tuple of (color_channels, brightness_channel)
        """
        if x.size(1) != 4:
            raise ValueError(f"Expected 4 channels (RGB+L), got {x.size(1)}")
            
        color_channels = x[:, :3, :, :]  # RGB channels
        brightness_channel = x[:, 3:4, :, :]  # Luminance channel
        
        return color_channels, brightness_channel
    
    def register_pathway_hooks(self):
        """Register hooks to monitor gradient flow through pathways."""
        def make_hook(name):
            def hook(grad):
                self.pathway_gradients[name] = grad.clone()
                return grad
            return hook
            
        # Register hooks for pathway analysis
        # Subclasses should override to register specific pathway hooks
        pass
    
    def get_pathway_importance(self) -> Dict[str, float]:
        """
        Calculate relative importance of different pathways.
        
        Returns:
            Dictionary mapping pathway names to importance scores
        """
        # Default implementation - subclasses should override
        return {}
    
    def get_model_stats(self) -> Dict[str, Any]:
        """
        Get model statistics including parameter counts and pathway information.
        
        Returns:
            Dictionary containing model statistics
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_classes': self.num_classes,
            'model_type': self.__class__.__name__
        }


class IntegrationMixin:
    """
    Mixin providing common integration functionality for multi-stream models.
    
    This mixin provides utility methods for pathway integration, gradient monitoring,
    and parameter management that can be shared across different integration strategies.
    """
    
    def register_integration_hooks(self):
        """Register hooks to monitor gradients and activations during training."""
        self.integration_hooks = {}
        
        # Register hooks for integration parameters
        for name, param in self.named_parameters():
            if any(key in name.lower() for key in ['alpha', 'beta', 'gamma', 'mixing']):
                hook = param.register_hook(
                    lambda grad, param_name=name: self._track_integration_gradient(param_name, grad)
                )
                self.integration_hooks[name] = hook
    
    def _track_integration_gradient(self, param_name: str, grad: torch.Tensor):
        """Track gradients for integration parameters."""
        if not hasattr(self, 'integration_gradients'):
            self.integration_gradients = {}
        
        self.integration_gradients[param_name] = {
            'norm': grad.norm().item(),
            'mean': grad.mean().item(),
            'std': grad.std().item(),
            'max': grad.max().item(),
            'min': grad.min().item()
        }
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get statistics about integration parameters and their gradients."""
        stats = {}
        
        # Parameter statistics
        for name, param in self.named_parameters():
            if any(key in name.lower() for key in ['alpha', 'beta', 'gamma', 'mixing']):
                stats[f'{name}_stats'] = {
                    'mean': param.data.mean().item(),
                    'std': param.data.std().item(),
                    'min': param.data.min().item(),
                    'max': param.data.max().item(),
                    'norm': param.data.norm().item()
                }
        
        # Gradient statistics (if available)
        if hasattr(self, 'integration_gradients'):
            stats['gradients'] = self.integration_gradients.copy()
            
        return stats
    
    def apply_integration_constraints(self):
        """Apply constraints to integration parameters (e.g., non-negativity, normalization)."""
        for name, param in self.named_parameters():
            with torch.no_grad():
                if 'alpha' in name.lower() or 'beta' in name.lower():
                    # Ensure non-negativity for mixing weights
                    param.data = torch.clamp(param.data, min=0.0)
                elif 'gamma' in name.lower():
                    # Clamp interaction term to reasonable range
                    param.data = torch.clamp(param.data, min=-1.0, max=1.0)
    
    def normalize_pathway_weights(self, alpha: torch.Tensor, beta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Normalize pathway weights so they sum to 1.
        
        Args:
            alpha: Color pathway weights
            beta: Brightness pathway weights
            
        Returns:
            Normalized (alpha, beta) weights
        """
        total = alpha + beta + 1e-8  # Small epsilon for numerical stability
        return alpha / total, beta / total
    
    def compute_pathway_importance(self, alpha: torch.Tensor, beta: torch.Tensor) -> Dict[str, float]:
        """
        Compute relative importance of each pathway.
        
        Args:
            alpha: Color pathway weights
            beta: Brightness pathway weights
            
        Returns:
            Dictionary with pathway importance metrics
        """
        alpha_mean = alpha.abs().mean().item()
        beta_mean = beta.abs().mean().item()
        total = alpha_mean + beta_mean + 1e-8
        
        return {
            'color_importance': alpha_mean / total,
            'brightness_importance': beta_mean / total,
            'importance_ratio': alpha_mean / (beta_mean + 1e-8)
        }
