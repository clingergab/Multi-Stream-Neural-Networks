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
    to prevent recursion errors with monkey patching. Also handles NaN/Inf gradients gracefully.
    
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
    
    # Get device for consistent tensor operations
    device = parameters[0].grad.device
    
    # Calculate norm with NaN/Inf handling
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        try:
            total_norm = torch.norm(torch.stack([torch.norm(p.grad.data, norm_type).to(device) for p in parameters]), norm_type)
        except RuntimeError:
            # Fallback if tensor stacking fails
            total_norm = torch.tensor(0.0, device=device)
    
    # Handle NaN/Inf values in gradients
    if torch.isnan(total_norm) or torch.isinf(total_norm):
        for p in parameters:
            # Set gradients to zero if they contain NaN/Inf
            if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                p.grad.detach().zero_()
        return torch.tensor(0.0, device=device)
    
    # Apply scaling if needed
    if max_norm > 0 and total_norm > max_norm:
        clip_coef = max_norm / (total_norm + 1e-6)
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)
    
    return total_norm
