"""
Custom learning rate schedulers for multi-stream neural networks.

This module provides custom schedulers that extend PyTorch's built-in schedulers
with additional functionality useful for multi-stream learning.
"""

import math
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
                        'quadratic_inout', 'cubic_inout', 'onecycle', 'step', 'plateau', or None)
        epochs: Number of training epochs
        train_loader_len: Length of the training data loader
        **scheduler_kwargs: Additional arguments for the scheduler
            - For 'cosine': t_max (epochs), eta_min (min LR)
            - For 'cosine_restarts': t_0 (cycle length in epochs), t_mult (cycle multiplier),
              eta_min (min LR), step_per_batch (bool, default=False - whether to step per batch instead of per epoch)
            - For 'decaying_cosine_restarts': t_0 (cycle length in epochs), t_mult (cycle multiplier),
              eta_min (min LR), restart_decay (default=0.8), step_per_batch (bool, default=False)
            - For 'quadratic_inout': t_max (epochs), eta_min (min LR)
            - For 'cubic_inout': t_max (epochs), eta_min (min LR)
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
    elif scheduler_type == 'quadratic_inout':
        # Quadratic InOut easing - smooth S-curve with quadratic acceleration/deceleration
        t_max = scheduler_kwargs.get('t_max', epochs)
        eta_min = scheduler_kwargs.get('eta_min', 0)
        return QuadraticInOutLR(optimizer, T_max=t_max, eta_min=eta_min)
    elif scheduler_type == 'cubic_inout':
        # Cubic InOut easing - very smooth S-curve with cubic acceleration/deceleration
        t_max = scheduler_kwargs.get('t_max', epochs)
        eta_min = scheduler_kwargs.get('eta_min', 0)
        return CubicInOutLR(optimizer, T_max=t_max, eta_min=eta_min)
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
            # CosineAnnealingWarmRestarts, QuadraticInOutLR, CubicInOutLR,
            # and other schedulers step per epoch
            scheduler.step()


class DecayingCosineAnnealingWarmRestarts:
    """
    CosineAnnealingWarmRestarts with decaying restart peaks and support for fractional T_mult.

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
        T_mult: Factor to increase T_i after each restart. Can be fractional (e.g., 1.5). Default: 1
        eta_min: Minimum learning rate. Default: 0
        restart_decay: Multiply peak LR by this after each restart. Default: 1.0 (no decay)
        last_epoch: The index of last epoch. Default: -1

    Example:
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        >>> scheduler = DecayingCosineAnnealingWarmRestarts(
        ...     optimizer,
        ...     T_0=10,
        ...     T_mult=1.5,  # Fractional growth!
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
        - T_mult can be fractional (e.g., 1.5) - cycle lengths are floored to integers
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_0: int,
        T_mult: float = 1,
        eta_min: float = 0,
        restart_decay: float = 1.0,
        last_epoch: int = -1
    ):
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(f"optimizer must be an Optimizer, got {type(optimizer)}")
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError(f"T_0 must be a positive integer, got {T_0}")
        if not isinstance(T_mult, (int, float)) or T_mult < 1.0:
            raise ValueError(f"T_mult must be a number >= 1.0, got {T_mult}")
        if not 0.0 <= restart_decay <= 1.0:
            raise ValueError(f"restart_decay must be in [0, 1], got {restart_decay}")

        self.optimizer = optimizer
        self.T_0 = T_0
        self.T_mult = float(T_mult)  # Ensure it's a float
        self.eta_min = eta_min
        self.restart_decay = restart_decay
        self.last_epoch = last_epoch

        # Store original base learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

        # Track decay multiplier (starts at 1.0, multiplied by restart_decay each restart)
        self.current_multiplier = 1.0

        # Current cycle tracking (integers) - follows PyTorch's pattern
        self.T_i = T_0  # Current cycle length (integer)
        # PyTorch starts T_cur at last_epoch, but since we don't call super().__init__
        # which would call step() for last_epoch >= 0, we handle it directly:
        # When last_epoch = -1 (default), the first step() will increment to 0
        # When last_epoch >= 0, we've already done that many epochs
        self.T_cur = last_epoch if last_epoch >= 0 else -1

        # Track restart count (our addition for decay tracking)
        self.restart_count = 0

        # Store last learning rates
        self._last_lr = None

        # Mimic PyTorch's LRScheduler base class behavior:
        # The base class calls step(0) during __init__ when last_epoch == -1
        # This initializes T_cur and last_epoch to 0
        if last_epoch == -1:
            self.last_epoch = -1  # Will be set to 0 by step()
            self.step(0)

    def get_lr(self):
        """
        Compute learning rate using cosine annealing formula.

        Returns:
            List of learning rates for each parameter group
        """
        lrs = []
        for base_lr in self.base_lrs:
            # Current peak is base_lr * current_multiplier (accounts for decay)
            current_peak = base_lr * self.current_multiplier

            # Clamp current_peak to be at least eta_min
            # This prevents LR from going below eta_min when peak decays too much
            current_peak = max(current_peak, self.eta_min)

            # Cosine annealing: eta_min + (peak - eta_min) * (1 + cos(π * T_cur / T_i)) / 2
            lr = self.eta_min + (current_peak - self.eta_min) * \
                 (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
            lrs.append(lr)

        return lrs

    def step(self, epoch=None):
        """
        Step the scheduler.

        Follows PyTorch's CosineAnnealingWarmRestarts logic with additions for:
        - Fractional T_mult support (floored to integers for cycle lengths)
        - Peak LR decay via restart_decay parameter

        Args:
            epoch: Current epoch number (optional, for manual epoch setting)
        """
        # Handle first call - matches PyTorch's pattern
        if epoch is None and self.last_epoch < 0:
            epoch = 0

        if epoch is None:
            # Auto-increment mode (standard usage)
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1

            # Check for restart - matches PyTorch's pattern
            if self.T_cur >= self.T_i:
                # Our addition: apply decay when restarting
                self.restart_count += 1
                self.current_multiplier *= self.restart_decay

                # Reset position and grow cycle length
                self.T_cur = self.T_cur - self.T_i
                # Our modification: floor fractional T_mult to integer
                self.T_i = int(math.floor(self.T_i * self.T_mult))
        else:
            # Manual epoch setting (advanced usage) - matches PyTorch's pattern
            if epoch < 0:
                raise ValueError(f"Expected non-negative epoch, but got {epoch}")

            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    # For fractional T_mult, use logarithm to find which cycle we're in
                    # Note: This is an approximation for fractional T_mult
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))

                    # Calculate T_cur based on cycle number
                    # Sum of geometric series: T_0 * (T_mult^n - 1) / (T_mult - 1)
                    cycle_start = self.T_0 * (self.T_mult**n - 1) / (self.T_mult - 1)
                    self.T_cur = epoch - int(cycle_start)

                    # Calculate current cycle length (floored)
                    self.T_i = int(math.floor(self.T_0 * (self.T_mult ** n)))

                    # Update decay multiplier based on cycle number
                    self.restart_count = n
                    self.current_multiplier = self.restart_decay ** n
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                self.restart_count = 0
                self.current_multiplier = 1.0

        self.last_epoch = int(math.floor(epoch))

        # Update optimizer with new LRs - matches PyTorch's pattern
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def get_last_lr(self):
        """
        Return last computed learning rate by current scheduler.

        Returns:
            List of learning rates for each parameter group
        """
        return self._last_lr

    @property
    def current_peak_lrs(self):
        """
        Return current peak learning rates (for backward compatibility).

        Returns:
            List of current peak LRs for each parameter group
        """
        return [base_lr * self.current_multiplier for base_lr in self.base_lrs]

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
            'current_multiplier': self.current_multiplier,
            'restart_count': self.restart_count,
            'last_epoch': self.last_epoch,
            'T_i': self.T_i,
            'T_cur': self.T_cur,
            '_last_lr': self._last_lr,
        }

    def load_state_dict(self, state_dict):
        """
        Loads the scheduler state.

        Args:
            state_dict: Scheduler state dict from state_dict()
        """
        self.T_0 = state_dict['T_0']
        self.T_mult = float(state_dict['T_mult'])  # Ensure float
        self.eta_min = state_dict['eta_min']
        self.restart_decay = state_dict['restart_decay']
        self.base_lrs = state_dict['base_lrs']

        # Handle both old and new state dict formats
        if 'current_multiplier' in state_dict:
            self.current_multiplier = state_dict['current_multiplier']
        elif 'current_peak_lrs' in state_dict:
            # Old format: convert to multiplier
            self.current_multiplier = state_dict['current_peak_lrs'][0] / self.base_lrs[0] if self.base_lrs[0] > 0 else 1.0
        else:
            self.current_multiplier = 1.0

        self.restart_count = state_dict['restart_count']
        self.last_epoch = state_dict['last_epoch']

        # Load cycle state if available (new format)
        if 'T_i' in state_dict and 'T_cur' in state_dict:
            self.T_i = state_dict['T_i']
            self.T_cur = state_dict['T_cur']
        else:
            # Old format: reconstruct from restart_count
            self.T_i = self.T_0
            for _ in range(self.restart_count):
                self.T_i = int(math.floor(self.T_i * self.T_mult))
            self.T_cur = 0

        # Load last LR if available
        if '_last_lr' in state_dict:
            self._last_lr = state_dict['_last_lr']
        else:
            self._last_lr = self.base_lrs.copy()

        # Apply the current LR to the optimizer's param_groups
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = self._last_lr[i]

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


class QuadraticInOutLR:
    """
    Learning rate scheduler using quadratic (t²) easing with InOut pattern.

    Based on Robert Penner's easing functions. Accelerates in the first half,
    decelerates in the second half, creating a smooth S-curve.

    The quadratic InOut easing function provides:
    - Gentle start (slow acceleration)
    - Aggressive middle (faster LR change)
    - Gentle end (slow deceleration)

    More aggressive in the middle compared to CosineAnnealingLR, but gentler
    at the start and end compared to linear decay.

    Formula:
        t = current_epoch / T_max  (normalized time in [0, 1])

        # Inverted easeInOutQuad (1 - easing_value for decay)
        if t < 0.5:
            factor = 1 - 2 * t²
        else:
            factor = 2 * (t - 1)²

        lr = eta_min + (base_lr - eta_min) * factor

    Args:
        optimizer: Wrapped optimizer
        T_max: Maximum number of iterations (total epochs)
        eta_min: Minimum learning rate. Default: 0
        last_epoch: The index of last epoch. Default: -1

    Example:
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        >>> scheduler = QuadraticInOutLR(optimizer, T_max=100, eta_min=1e-5)
        >>> for epoch in range(100):
        ...     train(...)
        ...     scheduler.step()
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_max: int,
        eta_min: float = 0,
        last_epoch: int = -1
    ):
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(f"optimizer must be an Optimizer, got {type(optimizer)}")
        if T_max <= 0 or not isinstance(T_max, int):
            raise ValueError(f"T_max must be a positive integer, got {T_max}")
        if eta_min < 0:
            raise ValueError(f"eta_min must be non-negative, got {eta_min}")

        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.last_epoch = last_epoch

        # Store base learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

        # Store last computed LR
        self._last_lr = None

        # Initialize (mimic PyTorch's pattern)
        if last_epoch == -1:
            self.last_epoch = -1
            self.step(0)

    def get_lr(self):
        """
        Compute learning rate using quadratic InOut easing.

        Returns:
            List of learning rates for each parameter group
        """
        if self.last_epoch == 0:
            # First epoch: return base LRs
            return self.base_lrs
        elif self.last_epoch >= self.T_max:
            # After T_max: return eta_min
            return [self.eta_min for _ in self.base_lrs]

        # Normalized time: t in [0, 1]
        t = self.last_epoch / self.T_max

        # Quadratic InOut easing (inverted for decay: 1 → 0)
        # Standard easeInOutQuad but inverted (1 - easing_value)
        if t < 0.5:
            # First half: gentle decay
            easing = 2 * t * t
            factor = 1 - easing
        else:
            # Second half: aggressive decay
            easing = -2 * (t - 1) * (t - 1) + 1
            factor = 1 - easing

        # Apply easing to each parameter group
        lrs = []
        for base_lr in self.base_lrs:
            lr = self.eta_min + (base_lr - self.eta_min) * factor
            lrs.append(lr)

        return lrs

    def step(self, epoch=None):
        """
        Step the scheduler.

        Args:
            epoch: Current epoch number (optional)
        """
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch

        # Update optimizer with new LRs
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def get_last_lr(self):
        """Return last computed learning rate."""
        return self._last_lr

    def state_dict(self):
        """Returns the state of the scheduler as a dict."""
        return {
            'T_max': self.T_max,
            'eta_min': self.eta_min,
            'base_lrs': self.base_lrs,
            'last_epoch': self.last_epoch,
            '_last_lr': self._last_lr,
        }

    def load_state_dict(self, state_dict):
        """Loads the scheduler state."""
        self.T_max = state_dict['T_max']
        self.eta_min = state_dict['eta_min']
        self.base_lrs = state_dict['base_lrs']
        self.last_epoch = state_dict['last_epoch']
        self._last_lr = state_dict.get('_last_lr', self.base_lrs.copy())

        # Apply current LR to optimizer
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = self._last_lr[i]

    def __repr__(self):
        """String representation of the scheduler."""
        return (
            f"{self.__class__.__name__}("
            f"T_max={self.T_max}, "
            f"eta_min={self.eta_min}, "
            f"last_epoch={self.last_epoch})"
        )


class CubicInOutLR:
    """
    Learning rate scheduler using cubic (t³) easing with InOut pattern.

    Based on Robert Penner's easing functions. Accelerates in the first half,
    decelerates in the second half, creating a smooth S-curve.

    The cubic InOut easing function provides:
    - Very gentle start (very slow acceleration)
    - Very aggressive middle (rapid LR change)
    - Very gentle end (very slow deceleration)

    More aggressive in the middle than QuadraticInOutLR, with even gentler
    transitions at the start and end.

    Formula:
        t = current_epoch / T_max  (normalized time in [0, 1])

        # Inverted easeInOutCubic (1 - easing_value for decay)
        if t < 0.5:
            factor = 1 - 4 * t³
        else:
            factor = -4 * (t - 1)³

        lr = eta_min + (base_lr - eta_min) * factor

    Args:
        optimizer: Wrapped optimizer
        T_max: Maximum number of iterations (total epochs)
        eta_min: Minimum learning rate. Default: 0
        last_epoch: The index of last epoch. Default: -1

    Example:
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        >>> scheduler = CubicInOutLR(optimizer, T_max=100, eta_min=1e-5)
        >>> for epoch in range(100):
        ...     train(...)
        ...     scheduler.step()
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_max: int,
        eta_min: float = 0,
        last_epoch: int = -1
    ):
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(f"optimizer must be an Optimizer, got {type(optimizer)}")
        if T_max <= 0 or not isinstance(T_max, int):
            raise ValueError(f"T_max must be a positive integer, got {T_max}")
        if eta_min < 0:
            raise ValueError(f"eta_min must be non-negative, got {eta_min}")

        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.last_epoch = last_epoch

        # Store base learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

        # Store last computed LR
        self._last_lr = None

        # Initialize (mimic PyTorch's pattern)
        if last_epoch == -1:
            self.last_epoch = -1
            self.step(0)

    def get_lr(self):
        """
        Compute learning rate using cubic InOut easing.

        Returns:
            List of learning rates for each parameter group
        """
        if self.last_epoch == 0:
            # First epoch: return base LRs
            return self.base_lrs
        elif self.last_epoch >= self.T_max:
            # After T_max: return eta_min
            return [self.eta_min for _ in self.base_lrs]

        # Normalized time: t in [0, 1]
        t = self.last_epoch / self.T_max

        # Cubic InOut easing (inverted for decay: 1 → 0)
        # Standard easeInOutCubic but inverted (1 - easing_value)
        if t < 0.5:
            # First half: very gentle decay
            easing = 4 * t * t * t
            factor = 1 - easing
        else:
            # Second half: very aggressive decay
            easing = 4 * (t - 1) * (t - 1) * (t - 1) + 1
            factor = 1 - easing

        # Apply easing to each parameter group
        lrs = []
        for base_lr in self.base_lrs:
            lr = self.eta_min + (base_lr - self.eta_min) * factor
            lrs.append(lr)

        return lrs

    def step(self, epoch=None):
        """
        Step the scheduler.

        Args:
            epoch: Current epoch number (optional)
        """
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch

        # Update optimizer with new LRs
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def get_last_lr(self):
        """Return last computed learning rate."""
        return self._last_lr

    def state_dict(self):
        """Returns the state of the scheduler as a dict."""
        return {
            'T_max': self.T_max,
            'eta_min': self.eta_min,
            'base_lrs': self.base_lrs,
            'last_epoch': self.last_epoch,
            '_last_lr': self._last_lr,
        }

    def load_state_dict(self, state_dict):
        """Loads the scheduler state."""
        self.T_max = state_dict['T_max']
        self.eta_min = state_dict['eta_min']
        self.base_lrs = state_dict['base_lrs']
        self.last_epoch = state_dict['last_epoch']
        self._last_lr = state_dict.get('_last_lr', self.base_lrs.copy())

        # Apply current LR to optimizer
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = self._last_lr[i]

    def __repr__(self):
        """String representation of the scheduler."""
        return (
            f"{self.__class__.__name__}("
            f"T_max={self.T_max}, "
            f"eta_min={self.eta_min}, "
            f"last_epoch={self.last_epoch})"
        )
