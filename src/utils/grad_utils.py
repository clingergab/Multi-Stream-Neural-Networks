"""
Patch to avoid torch.nn.utils.clip_grad_norm_ function that's causing recursion errors.
Import and use these functions in your model instead.
"""

import torch
from typing import Iterable

def safe_clip_grad_value(parameters: Iterable[torch.Tensor], clip_value: float) -> None:
    """
    A safe implementation of gradient value clipping that avoids using torch.nn.utils.clip_grad_value_
    to prevent recursion errors with monkey patching.
    
    Args:
        parameters: an iterable of Tensors or a single Tensor that will have gradients normalized
        clip_value: maximum allowed value of the gradients
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    
    for p in parameters:
        if p.grad is not None:
            p.grad.data.clamp_(min=-clip_value, max=clip_value)

def safe_clip_grad_norm(parameters: Iterable[torch.Tensor], max_norm: float, norm_type: float = 2.0) -> float:
    """
    A safe implementation of gradient norm clipping that avoids using torch.nn.utils.clip_grad_norm_
    to prevent recursion errors with monkey patching.
    
    Args:
        parameters: an iterable of Tensors or a single Tensor that will have gradients normalized
        max_norm: max norm of the gradients
        norm_type: type of the used p-norm. Can be 'inf' for infinity norm.
    
    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    
    # Filter parameters with gradients
    parameters = [p for p in parameters if p.grad is not None]
    if len(parameters) == 0:
        return torch.tensor(0.0)
    
    # Calculate norm
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.data, norm_type) for p in parameters]), norm_type)
    
    # Apply scaling if needed
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p in parameters:
            p.grad.data.mul_(clip_coef)
    
    return total_norm
