"""
Optimizer utilities for multi-stream neural networks.

This module provides convenience functions for creating optimizers and parameter groups,
especially useful for multi-stream models that benefit from stream-specific learning rates.
"""

import torch
from torch.optim import Adam, SGD, AdamW, RMSprop
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau


def create_stream_optimizer(
    model,
    optimizer_type: str = 'adamw',
    stream1_lr: float = 2e-4,
    stream2_lr: float = 7e-4,
    shared_lr: float = 5e-4,
    stream1_weight_decay: float = 1e-4,
    stream2_weight_decay: float = 2e-4,
    shared_weight_decay: float = 1.5e-4,
    **optimizer_kwargs
):
    """
    Create optimizer with stream-specific learning rates for multi-stream models.

    This is a convenience function that combines parameter group creation and optimizer
    instantiation in one step. It's designed for multi-stream models (MCResNet, LINet)
    where different streams benefit from different learning rates.

    Args:
        model: The model instance (must have get_stream_parameter_groups method)
        optimizer_type: Type of optimizer ('adam', 'adamw', 'sgd', 'rmsprop')
        stream1_lr: Learning rate for stream1 (RGB/color) parameters
        stream2_lr: Learning rate for stream2 (depth/brightness) parameters
        shared_lr: Learning rate for shared/fusion/integrated parameters
        stream1_weight_decay: Weight decay for stream1 parameters
        stream2_weight_decay: Weight decay for stream2 parameters
        shared_weight_decay: Weight decay for shared parameters
        **optimizer_kwargs: Additional optimizer-specific arguments
                           (e.g., betas, eps, momentum, nesterov)

    Returns:
        torch.optim.Optimizer with 3 parameter groups

    Example:
        >>> from src.models.multi_channel.mc_resnet import mc_resnet50
        >>> from src.training.optimizers import create_stream_optimizer
        >>> from src.training.schedulers import setup_scheduler
        >>>
        >>> model = mc_resnet50(num_classes=15)
        >>>
        >>> # Create optimizer with stream-specific LRs (one function call!)
        >>> optimizer = create_stream_optimizer(
        ...     model,
        ...     optimizer_type='adamw',
        ...     stream1_lr=2e-4,
        ...     stream2_lr=7e-4,
        ...     shared_lr=5e-4,
        ...     stream1_weight_decay=1e-4,
        ...     stream2_weight_decay=2e-4,
        ...     shared_weight_decay=1.5e-4
        ... )
        >>>
        >>> # Create scheduler
        >>> scheduler = setup_scheduler(optimizer, 'decaying_cosine', epochs=80, train_loader_len=40)
        >>>
        >>> # Compile and train
        >>> model.compile(optimizer=optimizer, scheduler=scheduler, loss='cross_entropy')
        >>> model.fit(train_loader, val_loader, epochs=80, stream_monitoring=True)
    """
    # Get stream-specific parameter groups from model
    if not hasattr(model, 'get_stream_parameter_groups'):
        raise AttributeError(
            f"Model {type(model).__name__} does not have get_stream_parameter_groups() method. "
            "This function only works with multi-stream models (MCResNet, LINet)."
        )

    param_groups = model.get_stream_parameter_groups(
        stream1_lr=stream1_lr,
        stream2_lr=stream2_lr,
        shared_lr=shared_lr,
        stream1_weight_decay=stream1_weight_decay,
        stream2_weight_decay=stream2_weight_decay,
        shared_weight_decay=shared_weight_decay
    )

    # Create optimizer with parameter groups
    optimizer_type = optimizer_type.lower()

    if optimizer_type == 'adam':
        return Adam(param_groups, **optimizer_kwargs)
    elif optimizer_type == 'adamw':
        return AdamW(param_groups, **optimizer_kwargs)
    elif optimizer_type == 'sgd':
        # Set sensible defaults for SGD
        if 'momentum' not in optimizer_kwargs:
            optimizer_kwargs['momentum'] = 0.9
        if 'nesterov' not in optimizer_kwargs:
            optimizer_kwargs['nesterov'] = True
        return SGD(param_groups, **optimizer_kwargs)
    elif optimizer_type == 'rmsprop':
        if 'momentum' not in optimizer_kwargs:
            optimizer_kwargs['momentum'] = 0.9
        return RMSprop(param_groups, **optimizer_kwargs)
    else:
        raise ValueError(
            f"Unknown optimizer type: {optimizer_type}. "
            f"Supported: 'adam', 'adamw', 'sgd', 'rmsprop'"
        )


def create_optimizer(
    model,
    optimizer_type: str = 'adamw',
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    **optimizer_kwargs
):
    """
    Create optimizer with single learning rate for all parameters.

    This is a simple convenience function for standard training where all parameters
    use the same learning rate. For multi-stream models with stream-specific learning
    rates, use create_stream_optimizer() instead.

    Args:
        model: The model instance
        optimizer_type: Type of optimizer ('adam', 'adamw', 'sgd', 'rmsprop')
        lr: Learning rate for all parameters
        weight_decay: Weight decay for all parameters
        **optimizer_kwargs: Additional optimizer-specific arguments
                           (e.g., betas, eps, momentum, nesterov)

    Returns:
        torch.optim.Optimizer with single parameter group

    Example:
        >>> from src.models.multi_channel.mc_resnet import mc_resnet50
        >>> from src.training.optimizers import create_optimizer
        >>> from src.training.schedulers import setup_scheduler
        >>>
        >>> model = mc_resnet50(num_classes=15)
        >>>
        >>> # Create simple optimizer (one learning rate for all params)
        >>> optimizer = create_optimizer(model, optimizer_type='adamw', lr=1e-3, weight_decay=1e-4)
        >>>
        >>> # Create scheduler
        >>> scheduler = setup_scheduler(optimizer, 'cosine', epochs=80, train_loader_len=40)
        >>>
        >>> # Compile and train
        >>> model.compile(optimizer=optimizer, scheduler=scheduler, loss='cross_entropy')
        >>> model.fit(train_loader, val_loader, epochs=80)
    """
    optimizer_type = optimizer_type.lower()

    if optimizer_type == 'adam':
        return Adam(model.parameters(), lr=lr, weight_decay=weight_decay, **optimizer_kwargs)
    elif optimizer_type == 'adamw':
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, **optimizer_kwargs)
    elif optimizer_type == 'sgd':
        # Set sensible defaults for SGD
        if 'momentum' not in optimizer_kwargs:
            optimizer_kwargs['momentum'] = 0.9
        if 'nesterov' not in optimizer_kwargs:
            optimizer_kwargs['nesterov'] = True
        return SGD(model.parameters(), lr=lr, weight_decay=weight_decay, **optimizer_kwargs)
    elif optimizer_type == 'rmsprop':
        if 'momentum' not in optimizer_kwargs:
            optimizer_kwargs['momentum'] = 0.9
        return RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay, **optimizer_kwargs)
    else:
        raise ValueError(
            f"Unknown optimizer type: {optimizer_type}. "
            f"Supported: 'adam', 'adamw', 'sgd', 'rmsprop'"
        )


def create_scheduler(optimizer, scheduler_type='step', **kwargs):
    """
    Create learning rate scheduler (legacy function, use setup_scheduler instead).

    This function is deprecated in favor of setup_scheduler() in schedulers.py,
    which provides more scheduler options including decaying variants.

    For new code, import and use:
        from src.training.schedulers import setup_scheduler

    Args:
        optimizer: The optimizer instance
        scheduler_type: Type of scheduler ('step', 'cosine', 'plateau')
        **kwargs: Scheduler-specific arguments

    Returns:
        Learning rate scheduler instance
    """
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