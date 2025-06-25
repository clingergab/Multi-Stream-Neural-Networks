"""Mixing weight initialization strategies."""

import torch
import torch.nn as nn
import math
from typing import Dict, Optional, Union


def init_scalar_weights(alpha_init: float = 1.0, 
                       beta_init: float = 1.0, 
                       gamma_init: float = 0.2) -> Dict[str, torch.Tensor]:
    """Initialize scalar mixing weights."""
    return {
        'alpha': torch.tensor(alpha_init, dtype=torch.float32),
        'beta': torch.tensor(beta_init, dtype=torch.float32),
        'gamma': torch.tensor(gamma_init, dtype=torch.float32)
    }


def init_channel_weights(num_channels: int,
                        alpha_init: float = 1.0,
                        beta_init: float = 1.0, 
                        gamma_init: float = 0.2,
                        variance: float = 0.1) -> Dict[str, torch.Tensor]:
    """Initialize channel-wise mixing weights."""
    return {
        'alpha': torch.normal(alpha_init, variance, (num_channels,)),
        'beta': torch.normal(beta_init, variance, (num_channels,)),
        'gamma': torch.normal(gamma_init, variance * 0.5, (num_channels,))
    }


def init_spatial_weights(height: int, width: int,
                        alpha_init: float = 1.0,
                        beta_init: float = 1.0,
                        gamma_init: float = 0.2,
                        spatial_variance: float = 0.05) -> Dict[str, torch.Tensor]:
    """Initialize spatial mixing weights."""
    return {
        'alpha': torch.normal(alpha_init, spatial_variance, (1, height, width)),
        'beta': torch.normal(beta_init, spatial_variance, (1, height, width)),
        'gamma': torch.normal(gamma_init, spatial_variance * 0.5, (1, height, width))
    }


def init_dynamic_weights(input_dim: int, 
                        output_dim: int,
                        init_type: str = 'xavier') -> Dict[str, nn.Module]:
    """Initialize dynamic weight generation networks."""
    
    def create_weight_network():
        """Create a simple network for weight generation."""
        return nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, output_dim),
            nn.Sigmoid()
        )
    
    networks = {
        'alpha_net': create_weight_network(),
        'beta_net': create_weight_network(), 
        'gamma_net': create_weight_network()
    }
    
    # Initialize network weights
    for net in networks.values():
        for layer in net:
            if isinstance(layer, nn.Linear):
                if init_type == 'xavier':
                    nn.init.xavier_uniform_(layer.weight)
                elif init_type == 'he':
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                elif init_type == 'normal':
                    nn.init.normal_(layer.weight, 0, 0.02)
                
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    return networks


def init_balanced_mixing_weights(num_pathways: int = 2,
                                balance_factor: float = 1.0) -> Dict[str, torch.Tensor]:
    """Initialize mixing weights for balanced pathway contribution."""
    base_weight = balance_factor / num_pathways
    
    weights = {}
    for i in range(num_pathways):
        weights[f'pathway_{i}_weight'] = torch.tensor(base_weight)
    
    # Interaction term (smaller)
    weights['interaction'] = torch.tensor(base_weight * 0.2)
    
    return weights


def apply_mixing_weight_constraints(weights: Dict[str, torch.Tensor],
                                   min_val: float = 0.01,
                                   max_val: float = 10.0,
                                   normalize: bool = False) -> Dict[str, torch.Tensor]:
    """Apply constraints to mixing weights."""
    constrained = {}
    
    for name, weight in weights.items():
        # Clamp to range
        constrained_weight = torch.clamp(weight, min_val, max_val)
        constrained[name] = constrained_weight
    
    # Normalize alpha and beta to sum to 1 if requested
    if normalize and 'alpha' in constrained and 'beta' in constrained:
        alpha, beta = constrained['alpha'], constrained['beta']
        total = alpha + beta + 1e-8  # Avoid division by zero
        constrained['alpha'] = alpha / total
        constrained['beta'] = beta / total
    
    return constrained


def get_initialization_strategy(strategy: str, **kwargs) -> Dict[str, torch.Tensor]:
    """Get weights using specified initialization strategy."""
    
    strategies = {
        'balanced': lambda: init_scalar_weights(1.0, 1.0, 0.2),
        'color_biased': lambda: init_scalar_weights(1.5, 0.5, 0.1),
        'brightness_biased': lambda: init_scalar_weights(0.5, 1.5, 0.1),
        'minimal_interaction': lambda: init_scalar_weights(1.0, 1.0, 0.05),
        'strong_interaction': lambda: init_scalar_weights(1.0, 1.0, 0.5),
        'random': lambda: init_scalar_weights(
            torch.rand(1).item() * 2,
            torch.rand(1).item() * 2, 
            torch.rand(1).item() * 0.5
        )
    }
    
    if strategy not in strategies:
        raise ValueError(f"Unknown strategy: {strategy}. Available: {list(strategies.keys())}")
    
    return strategies[strategy]()


def initialize_mixing_layer(layer: nn.Module, 
                           strategy: str = 'balanced',
                           **kwargs):
    """Initialize a mixing layer with specified strategy."""
    
    weights = get_initialization_strategy(strategy, **kwargs)
    
    # Apply weights to layer parameters
    for name, param in layer.named_parameters():
        param_name = name.split('.')[-1]  # Get parameter name without module prefix
        
        if param_name in weights:
            with torch.no_grad():
                if param.shape == weights[param_name].shape:
                    param.copy_(weights[param_name])
                else:
                    # Handle shape mismatch (e.g., broadcasting)
                    param.fill_(weights[param_name].mean().item())


def analyze_initialization_impact(model: nn.Module, 
                                 strategies: list,
                                 test_input: torch.Tensor) -> Dict[str, float]:
    """Analyze the impact of different initialization strategies."""
    results = {}
    
    original_state = model.state_dict()
    
    for strategy in strategies:
        # Reset model
        model.load_state_dict(original_state)
        
        # Apply initialization strategy
        for name, module in model.named_modules():
            if 'mixing' in name.lower():
                initialize_mixing_layer(module, strategy)
        
        # Test forward pass
        with torch.no_grad():
            output = model(test_input)
            results[strategy] = {
                'output_std': output.std().item(),
                'output_mean': output.mean().item(),
                'output_range': (output.max() - output.min()).item()
            }
    
    # Restore original state
    model.load_state_dict(original_state)
    
    return results
