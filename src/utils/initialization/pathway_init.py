"""Pathway initialization utilities."""

import torch
import torch.nn as nn
from typing import Dict, List, Optional


def init_pathway_weights(layer: nn.Module, 
                        initialization: str = 'xavier',
                        pathway_name: str = 'pathway') -> None:
    """Initialize weights for a specific pathway."""
    
    for name, param in layer.named_parameters():
        if 'weight' in name:
            if initialization == 'xavier':
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
            elif initialization == 'he':
                if len(param.shape) >= 2:
                    nn.init.kaiming_uniform_(param, nonlinearity='relu')
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
            elif initialization == 'normal':
                nn.init.normal_(param, 0, 0.02)
            elif initialization == 'orthogonal':
                if len(param.shape) >= 2:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
        
        elif 'bias' in name and param is not None:
            nn.init.zeros_(param)


def init_balanced_pathways(pathways: Dict[str, nn.Module],
                          initialization: str = 'xavier',
                          balance_factor: float = 1.0) -> None:
    """Initialize multiple pathways with balanced weights."""
    
    for pathway_name, pathway_module in pathways.items():
        init_pathway_weights(pathway_module, initialization, pathway_name)
        
        # Apply balance factor
        if balance_factor != 1.0:
            with torch.no_grad():
                for param in pathway_module.parameters():
                    param.mul_(balance_factor)


def xavier_pathway_init(pathways: List[nn.Module]) -> None:
    """Initialize pathways using Xavier initialization."""
    for pathway in pathways:
        init_pathway_weights(pathway, 'xavier')


def he_pathway_init(pathways: List[nn.Module]) -> None:
    """Initialize pathways using He initialization."""
    for pathway in pathways:
        init_pathway_weights(pathway, 'he')


def orthogonal_pathway_init(pathways: List[nn.Module]) -> None:
    """Initialize pathways using orthogonal initialization."""
    for pathway in pathways:
        init_pathway_weights(pathway, 'orthogonal')


def asymmetric_pathway_init(color_pathway: nn.Module,
                           brightness_pathway: nn.Module,
                           color_scale: float = 1.0,
                           brightness_scale: float = 1.0) -> None:
    """Initialize pathways with different scales for asymmetric learning."""
    
    # Initialize both pathways
    init_pathway_weights(color_pathway, 'xavier')
    init_pathway_weights(brightness_pathway, 'xavier')
    
    # Scale weights differently
    with torch.no_grad():
        for param in color_pathway.parameters():
            param.mul_(color_scale)
        
        for param in brightness_pathway.parameters():
            param.mul_(brightness_scale)


def progressive_pathway_init(pathways: List[nn.Module],
                           layer_scaling: bool = True) -> None:
    """Initialize pathways with progressive scaling through layers."""
    
    for pathway in pathways:
        layer_idx = 0
        
        for name, module in pathway.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # Standard initialization
                init_pathway_weights(module, 'xavier')
                
                # Apply progressive scaling
                if layer_scaling:
                    scale_factor = 1.0 / (layer_idx + 1) ** 0.5
                    with torch.no_grad():
                        for param in module.parameters():
                            param.mul_(scale_factor)
                
                layer_idx += 1


def pathway_diversity_init(pathways: List[nn.Module],
                          diversity_factor: float = 0.1) -> None:
    """Initialize pathways to encourage diversity."""
    
    # First, initialize all pathways normally
    for pathway in pathways:
        init_pathway_weights(pathway, 'xavier')
    
    # Add diversity by modifying each pathway differently
    for i, pathway in enumerate(pathways):
        with torch.no_grad():
            for param in pathway.parameters():
                # Add pathway-specific noise
                noise = torch.randn_like(param) * diversity_factor
                param.add_(noise)
                
                # Apply pathway-specific scaling
                pathway_scale = 1.0 + (i * 0.1 - 0.05)  # Small variations
                param.mul_(pathway_scale)


def analyze_pathway_initialization(pathways: Dict[str, nn.Module]) -> Dict[str, Dict[str, float]]:
    """Analyze the initialization of pathways."""
    
    analysis = {}
    
    for pathway_name, pathway in pathways.items():
        pathway_stats = {
            'weight_mean': 0.0,
            'weight_std': 0.0,
            'weight_min': float('inf'),
            'weight_max': float('-inf'),
            'num_parameters': 0
        }
        
        all_weights = []
        
        for param in pathway.parameters():
            if param.requires_grad:
                weights = param.data.flatten()
                all_weights.append(weights)
                pathway_stats['num_parameters'] += weights.numel()
        
        if all_weights:
            all_weights = torch.cat(all_weights)
            pathway_stats['weight_mean'] = all_weights.mean().item()
            pathway_stats['weight_std'] = all_weights.std().item()
            pathway_stats['weight_min'] = all_weights.min().item()
            pathway_stats['weight_max'] = all_weights.max().item()
        
        analysis[pathway_name] = pathway_stats
    
    return analysis


def check_pathway_balance(pathways: Dict[str, nn.Module]) -> Dict[str, float]:
    """Check if pathways are balanced in terms of parameter magnitudes."""
    
    pathway_norms = {}
    
    for pathway_name, pathway in pathways.items():
        total_norm = 0.0
        
        for param in pathway.parameters():
            if param.requires_grad:
                total_norm += param.data.norm().item() ** 2
        
        pathway_norms[pathway_name] = total_norm ** 0.5
    
    # Compute balance ratios
    if len(pathway_norms) >= 2:
        values = list(pathway_norms.values())
        max_norm = max(values)
        balance_ratios = {name: norm / max_norm for name, norm in pathway_norms.items()}
        return balance_ratios
    
    return pathway_norms


def rebalance_pathways(pathways: Dict[str, nn.Module],
                      target_balance: Optional[Dict[str, float]] = None) -> None:
    """Rebalance pathway weights to achieve target balance."""
    
    current_balance = check_pathway_balance(pathways)
    
    if target_balance is None:
        # Default: equal balance
        target_balance = {name: 1.0 for name in pathways.keys()}
    
    with torch.no_grad():
        for pathway_name, pathway in pathways.items():
            if pathway_name in target_balance and pathway_name in current_balance:
                current_ratio = current_balance[pathway_name]
                target_ratio = target_balance[pathway_name]
                
                if current_ratio > 0:
                    scale_factor = target_ratio / current_ratio
                    
                    for param in pathway.parameters():
                        param.mul_(scale_factor)
