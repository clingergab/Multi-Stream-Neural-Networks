"""
Custom learning rate schedulers for multi-stream neural networks.

This module provides custom schedulers that extend PyTorch's built-in schedulers
with additional functionality useful for multi-stream learning.
"""

import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    StepLR,
    ReduceLROnPlateau
)


def setup_scheduler(optimizer, scheduler_type: str, epochs: int, train_loader_len: int, **scheduler_kwargs):
    """
    Set up and return the learning rate scheduler based on the scheduler type.

    Args:
        optimizer: The optimizer instance
        scheduler_type: Type of scheduler ('cosine', 'cosine_restarts', 'decaying_cosine_restarts',
                        'onecycle', 'step', 'plateau', or None)
        epochs: Number of training epochs
        train_loader_len: Length of the training data loader
        **scheduler_kwargs: Additional arguments for the scheduler
            - For 'cosine': t_max (epochs), eta_min (min LR)
            - For 'cosine_restarts': t_0 (cycle length in epochs), t_mult (cycle multiplier),
              eta_min (min LR), step_per_batch (bool, default=False - whether to step per batch instead of per epoch)
            - For 'decaying_cosine_restarts': t_0 (cycle length in epochs), t_mult (cycle multiplier),
              eta_min (min LR), restart_decay (default=0.8), step_per_batch (bool, default=False)
            - For 'onecycle': max_lr, pct_start, anneal_strategy, div_factor, final_div_factor
            - For 'step': step_size (in epochs), gamma
            - For 'plateau': patience (epochs), factor, mode, min_lr, threshold, cooldown, eps

    Returns:
        Scheduler instance or None if no scheduler requested
    """
    if not scheduler_type:
        return None
        
    if scheduler_type == 'cosine':
        # T_max is in epochs (scheduler steps per epoch, not per batch)
        t_max = scheduler_kwargs.get('t_max', epochs)
        eta_min = scheduler_kwargs.get('eta_min', 0)
        return CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
    elif scheduler_type == 'cosine_restarts':
        # CosineAnnealingWarmRestarts - multiple cycles with warm restarts
        # User always specifies t_0 in EPOCHS (intuitive)
        t_0_epochs = scheduler_kwargs.get('t_0', 20)  # First cycle length in epochs (user-facing)
        t_mult = scheduler_kwargs.get('t_mult', 1)  # Cycle length multiplier (1 = equal cycles)
        eta_min = scheduler_kwargs.get('eta_min', 1e-7)  # Minimum learning rate
        step_per_batch = scheduler_kwargs.get('step_per_batch', False)  # Step per batch or per epoch

        if step_per_batch:
            # Convert epochs to batches for per-batch stepping
            # Example: t_0=20 epochs × 40 batches/epoch = 800 batches per cycle
            t_0 = t_0_epochs * train_loader_len
        else:
            # Keep in epochs for per-epoch stepping (default)
            t_0 = t_0_epochs

        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=t_0, T_mult=t_mult, eta_min=eta_min)
        # Add metadata so mc_resnet.py knows whether to step per batch or per epoch
        scheduler._step_per_batch = step_per_batch
        return scheduler
    elif scheduler_type == 'decaying_cosine_restarts':
        # DecayingCosineAnnealingWarmRestarts - warm restarts with decaying peak LR
        # User always specifies t_0 in EPOCHS (intuitive)
        t_0_epochs = scheduler_kwargs.get('t_0', 20)  # First cycle length in epochs (user-facing)
        t_mult = scheduler_kwargs.get('t_mult', 1)  # Cycle length multiplier (1 = equal cycles)
        eta_min = scheduler_kwargs.get('eta_min', 1e-7)  # Minimum learning rate
        restart_decay = scheduler_kwargs.get('restart_decay', 0.8)  # Decay factor (0.8 = 80% of previous peak)
        step_per_batch = scheduler_kwargs.get('step_per_batch', False)  # Step per batch or per epoch

        if step_per_batch:
            # Convert epochs to batches for per-batch stepping
            # Example: t_0=20 epochs × 40 batches/epoch = 800 batches per cycle
            t_0 = t_0_epochs * train_loader_len
        else:
            # Keep in epochs for per-epoch stepping (default)
            t_0 = t_0_epochs

        scheduler = DecayingCosineAnnealingWarmRestarts(
            optimizer, T_0=t_0, T_mult=t_mult, eta_min=eta_min, restart_decay=restart_decay
        )
        # Add metadata so mc_resnet.py knows whether to step per batch or per epoch
        scheduler._step_per_batch = step_per_batch
        return scheduler
    elif scheduler_type == 'onecycle':
        # For OneCycleLR, we need total number of steps (epochs * steps_per_epoch)
        steps_per_epoch = scheduler_kwargs.get('steps_per_epoch', train_loader_len)
        # Handle both single scalar max_lr and per-group max_lr for multi-stream models
        # Default: 10x the initial LR for each parameter group
        default_max_lr = [pg['lr'] * 10 for pg in optimizer.param_groups]
        max_lr = scheduler_kwargs.get('max_lr', default_max_lr)
        pct_start = scheduler_kwargs.get('pct_start', 0.3)
        anneal_strategy = scheduler_kwargs.get('anneal_strategy', 'cos')
        div_factor = scheduler_kwargs.get('div_factor', 25.0)
        final_div_factor = scheduler_kwargs.get('final_div_factor', 1e4)

        # Create the OneCycleLR scheduler with calculated total_steps
        return OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=epochs * steps_per_epoch,
            pct_start=pct_start,
            anneal_strategy=anneal_strategy,
            div_factor=div_factor,
            final_div_factor=final_div_factor
        )
    elif scheduler_type == 'step':
        step_size = scheduler_kwargs.get('step_size', 30)
        gamma = scheduler_kwargs.get('gamma', 0.1)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == 'plateau':
        # Use scheduler_patience if provided, otherwise fall back to patience from scheduler_kwargs
        scheduler_patience = scheduler_kwargs.get('scheduler_patience', scheduler_kwargs.get('patience', 5))
        factor = scheduler_kwargs.get('factor', 0.5)
        mode = scheduler_kwargs.get('mode', 'min')  # 'min' for loss (default), 'max' for accuracy
        min_lr = scheduler_kwargs.get('min_lr', 1e-7)  # Minimum learning rate
        threshold = scheduler_kwargs.get('threshold', 1e-4)  # Minimum change to qualify as improvement
        cooldown = scheduler_kwargs.get('cooldown', 0)  # Cooldown period after LR reduction
        eps = scheduler_kwargs.get('eps', 1e-8)  # Minimal decay applied to lr
        return ReduceLROnPlateau(
            optimizer, mode=mode, patience=scheduler_patience, factor=factor, min_lr=min_lr, threshold=threshold, cooldown=cooldown, eps=eps
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

def update_scheduler(scheduler, val_loss: float) -> None:
    """
    Update the learning rate scheduler.

    Args:
        scheduler: The scheduler instance
        val_loss: Validation loss for plateau scheduler
    """
    if scheduler is not None:
        # Skip OneCycleLR as it's updated after each batch
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        elif not isinstance(scheduler, OneCycleLR):
            # CosineAnnealingWarmRestarts and other schedulers step per epoch
            scheduler.step()


class DecayingCosineAnnealingWarmRestarts:
    """
    CosineAnnealingWarmRestarts with decaying restart peaks.

    Each time the scheduler restarts (warm restart), the peak learning rate
    is multiplied by `restart_decay`, creating a gradually decreasing envelope
    over the cosine annealing cycles.

    This provides the benefits of warm restarts (escaping local minima) while
    gradually reducing the learning rate for better convergence.

    Pattern:
        Cycle 1: high = start_lr,                  low = eta_min
        Cycle 2: high = start_lr * restart_decay,  low = eta_min
        Cycle 3: high = start_lr * restart_decay², low = eta_min
        ...

    Example with restart_decay=0.8:
        Cycle 1: 0.100 → 0.001
        Cycle 2: 0.080 → 0.001
        Cycle 3: 0.064 → 0.001
        Cycle 4: 0.051 → 0.001

    Args:
        optimizer: Wrapped optimizer
        T_0: Number of iterations for the first restart (in epochs or batches)
        T_mult: Factor to increase T_i after each restart. Default: 1
        eta_min: Minimum learning rate. Default: 0
        restart_decay: Multiply peak LR by this after each restart. Default: 1.0 (no decay)
        last_epoch: The index of last epoch. Default: -1

    Example:
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        >>> scheduler = DecayingCosineAnnealingWarmRestarts(
        ...     optimizer,
        ...     T_0=10,
        ...     T_mult=2,
        ...     restart_decay=0.8,
        ...     eta_min=1e-6
        ... )
        >>> for epoch in range(100):
        ...     train(...)
        ...     scheduler.step()

    Note:
        - restart_decay=1.0 behaves identically to CosineAnnealingWarmRestarts
        - restart_decay=0.8 reduces peak by 20% each restart (recommended)
        - restart_decay=0.5 reduces peak by 50% each restart (aggressive)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0,
        restart_decay: float = 1.0,
        last_epoch: int = -1
    ):
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(f"optimizer must be an Optimizer, got {type(optimizer)}")
        if T_0 <= 0:
            raise ValueError(f"T_0 must be positive, got {T_0}")
        if T_mult < 1:
            raise ValueError(f"T_mult must be >= 1, got {T_mult}")
        if not 0.0 <= restart_decay <= 1.0:
            raise ValueError(f"restart_decay must be in [0, 1], got {restart_decay}")

        self.optimizer = optimizer
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.restart_decay = restart_decay
        self.last_epoch = last_epoch

        # Store original base learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

        # Track current peak LRs (will decay over time)
        self.current_peak_lrs = self.base_lrs.copy()

        # Create underlying CosineAnnealingWarmRestarts scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min, last_epoch=last_epoch
        )

        # Track for restart detection
        self.prev_lrs = [group['lr'] for group in optimizer.param_groups]
        self.restart_count = 0

    def step(self, epoch=None):
        """
        Step the scheduler.

        This should be called once per epoch (or once per batch if using batch-level scheduling).

        Args:
            epoch: Current epoch number (optional, for manual epoch setting)
        """
        # Get current LRs before stepping
        current_lrs = [group['lr'] for group in self.optimizer.param_groups]

        # Step the underlying scheduler
        self.scheduler.step(epoch)

        # Get new LRs after stepping
        new_lrs = [group['lr'] for group in self.optimizer.param_groups]

        # Detect restart: LR increased significantly (warm restart occurred)
        # Use threshold to detect restart (new LR > 1.1 * old LR indicates jump)
        restart_detected = any(
            new_lr > old_lr * 1.1
            for new_lr, old_lr in zip(new_lrs, current_lrs)
        )

        if restart_detected:
            self.restart_count += 1

            # Decay the peak LRs for the next cycle
            self.current_peak_lrs = [
                peak_lr * self.restart_decay
                for peak_lr in self.current_peak_lrs
            ]

            # Set optimizer to new peak LRs (start of new cycle)
            for param_group, new_peak_lr in zip(self.optimizer.param_groups, self.current_peak_lrs):
                param_group['lr'] = new_peak_lr
                param_group['initial_lr'] = new_peak_lr  # Update base LR for scheduler

            # Recreate scheduler with new peak LRs
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.T_0,
                T_mult=self.T_mult,
                eta_min=self.eta_min,
                last_epoch=-1  # Reset to start of new cycle
            )

        self.prev_lrs = new_lrs
        self.last_epoch += 1

    def get_last_lr(self):
        """
        Return last computed learning rate by current scheduler.

        Returns:
            List of learning rates for each parameter group
        """
        return [group['lr'] for group in self.optimizer.param_groups]

    def state_dict(self):
        """
        Returns the state of the scheduler as a dict.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer. Useful for checkpointing.

        Returns:
            Dictionary containing scheduler state
        """
        return {
            'T_0': self.T_0,
            'T_mult': self.T_mult,
            'eta_min': self.eta_min,
            'restart_decay': self.restart_decay,
            'base_lrs': self.base_lrs,
            'current_peak_lrs': self.current_peak_lrs,
            'restart_count': self.restart_count,
            'last_epoch': self.last_epoch,
            'scheduler_state': self.scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        """
        Loads the scheduler state.

        Args:
            state_dict: Scheduler state dict from state_dict()
        """
        self.T_0 = state_dict['T_0']
        self.T_mult = state_dict['T_mult']
        self.eta_min = state_dict['eta_min']
        self.restart_decay = state_dict['restart_decay']
        self.base_lrs = state_dict['base_lrs']
        self.current_peak_lrs = state_dict['current_peak_lrs']
        self.restart_count = state_dict['restart_count']
        self.last_epoch = state_dict['last_epoch']
        self.scheduler.load_state_dict(state_dict['scheduler_state'])

    def __repr__(self):
        """String representation of the scheduler."""
        return (
            f"{self.__class__.__name__}("
            f"T_0={self.T_0}, "
            f"T_mult={self.T_mult}, "
            f"eta_min={self.eta_min}, "
            f"restart_decay={self.restart_decay}, "
            f"restarts={self.restart_count})"
        )
