"""Optimizers and learning rate schedulers."""

import torch
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau


def create_optimizer(model, optimizer_type='adam', lr=0.001, weight_decay=1e-4, **kwargs):
    """Create optimizer based on configuration."""
    
    if optimizer_type == 'adam':
        return Adam(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    elif optimizer_type == 'adamw':
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    elif optimizer_type == 'sgd':
        momentum = kwargs.get('momentum', 0.9)
        return SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def create_scheduler(optimizer, scheduler_type='step', **kwargs):
    """Create learning rate scheduler."""
    
    if scheduler_type == 'step':
        step_size = kwargs.get('step_size', 30)
        gamma = kwargs.get('gamma', 0.1)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == 'cosine':
        T_max = kwargs.get('T_max', 100)
        return CosineAnnealingLR(optimizer, T_max=T_max)
    elif scheduler_type == 'plateau':
        return ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")