"""
Custom learning rate schedulers for multi-stream neural networks.

This module provides custom schedulers that extend PyTorch's built-in schedulers
with additional functionality useful for multi-stream learning, including warmup
support.
"""

import math
import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    StepLR,
    ReduceLROnPlateau,
    SequentialLR,
    LinearLR
)


class PerGroupSchedulerWrapper:
    """
    Universal wrapper that creates independent scheduler instances for each parameter group.

    This is the MATHEMATICALLY CORRECT way to implement per-group scheduler parameters
    (eta_min, min_lr, T_max, etc.) because each parameter group follows its own
    independent decay curve, not a clamped global curve.

    Key benefits:
    - Each group follows its OWN cosine/decay curve with its OWN eta_min
    - Preserves ALL scheduler behaviors (restarts, warm restarts, etc.)
    - Works with ANY PyTorch scheduler (built-in or custom)
    - No clamping artifacts - smooth curves for all groups

    Args:
        optimizer: Optimizer with multiple parameter groups
        scheduler_class: Scheduler class to instantiate (e.g., CosineAnnealingLR)
        scheduler_kwargs_list: List of scheduler kwargs, one dict per param group
            OR single dict to use for all groups (with scalar eta_min/min_lr)

    Example (per-group eta_min):
        >>> optimizer = AdamW([
        ...     {'params': model.stream1.parameters(), 'lr': 3e-4},
        ...     {'params': model.stream2.parameters(), 'lr': 1.2e-3},
        ...     {'params': model.shared.parameters(), 'lr': 1.5e-4},
        ... ])
        >>> scheduler = PerGroupSchedulerWrapper(
        ...     optimizer,
        ...     CosineAnnealingLR,
        ...     [
        ...         {'T_max': 80, 'eta_min': 1e-7},  # Stream1
        ...         {'T_max': 80, 'eta_min': 1e-6},  # Stream2
        ...         {'T_max': 80, 'eta_min': 5e-7},  # Shared
        ...     ]
        ... )

    Example (same scheduler for all groups):
        >>> scheduler = PerGroupSchedulerWrapper(
        ...     optimizer,
        ...     CosineAnnealingLR,
        ...     {'T_max': 80, 'eta_min': 1e-6}  # Same for all groups
        ... )
    """

    def __init__(self, optimizer, scheduler_class, scheduler_kwargs_list):
        self.optimizer = optimizer
        self.scheduler_class = scheduler_class

        # If single dict provided, replicate for all groups
        if isinstance(scheduler_kwargs_list, dict):
            scheduler_kwargs_list = [scheduler_kwargs_list.copy() for _ in optimizer.param_groups]

        if len(scheduler_kwargs_list) != len(optimizer.param_groups):
            raise ValueError(
                f"scheduler_kwargs_list length ({len(scheduler_kwargs_list)}) must match "
                f"number of parameter groups ({len(optimizer.param_groups)})"
            )

        self.scheduler_kwargs_list = scheduler_kwargs_list
        self.schedulers = []

        # Create one scheduler per parameter group
        # IMPORTANT: We pass the actual param_group dict to each single-group optimizer.
        # This creates a SHARED REFERENCE - when the scheduler updates the LR in its
        # optimizer's param_group, it automatically updates the main optimizer's param_group.
        # This is intentional and correct behavior.
        for i, (param_group, sched_kwargs) in enumerate(zip(optimizer.param_groups, scheduler_kwargs_list)):
            # Create single-param-group optimizer
            # Use the same optimizer class as the original
            single_group_opt = type(optimizer)([param_group])

            # Create scheduler for this group
            scheduler = scheduler_class(single_group_opt, **sched_kwargs)
            self.schedulers.append(scheduler)

        # Track last epoch for compatibility
        self.last_epoch = -1
        self._last_lr = None

    def step(self, *args, **kwargs):
        """
        Step all per-group schedulers and update optimizer LRs.

        Note: Because param_groups are shared by reference between the single-group
        optimizers and the main optimizer, when each scheduler updates its optimizer's
        param_group['lr'], it automatically updates the main optimizer's param_group['lr'].
        The explicit update loop below is kept for clarity and to ensure _last_lr tracking.
        """
        # Step each scheduler (this updates LRs in shared param_groups)
        for scheduler in self.schedulers:
            scheduler.step(*args, **kwargs)

        # Update main optimizer LRs (technically redundant due to shared references,
        # but kept for explicitness and future-proofing)
        for param_group, scheduler in zip(self.optimizer.param_groups, self.schedulers):
            param_group['lr'] = scheduler.get_last_lr()[0]

        # Update tracking
        self.last_epoch = self.schedulers[0].last_epoch
        self._last_lr = [scheduler.get_last_lr()[0] for scheduler in self.schedulers]

    def get_last_lr(self):
        """Get last learning rates from all schedulers."""
        if self._last_lr is None:
            return [scheduler.get_last_lr()[0] for scheduler in self.schedulers]
        return self._last_lr

    def state_dict(self):
        """Return state dict containing all per-group scheduler states."""
        return {
            'scheduler_class': self.scheduler_class.__name__,
            'scheduler_kwargs_list': self.scheduler_kwargs_list,
            'scheduler_states': [sched.state_dict() for sched in self.schedulers],
            'last_epoch': self.last_epoch,
            '_last_lr': self._last_lr,
        }

    def load_state_dict(self, state_dict):
        """Load state dict for all per-group schedulers."""
        self.last_epoch = state_dict.get('last_epoch', -1)
        self._last_lr = state_dict.get('_last_lr', None)

        # Load each scheduler's state
        scheduler_states = state_dict.get('scheduler_states', [])
        if len(scheduler_states) == len(self.schedulers):
            for scheduler, sched_state in zip(self.schedulers, scheduler_states):
                scheduler.load_state_dict(sched_state)

    def __getattr__(self, name):
        """
        Delegate attribute access to first scheduler for compatibility.

        Note: Some attributes may differ between schedulers (e.g., eta_min).
        This returns the first scheduler's value for compatibility with code
        that expects a single scheduler.
        """
        if name in ['optimizer', 'scheduler_class', 'scheduler_kwargs_list',
                    'schedulers', 'last_epoch', '_last_lr']:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        return getattr(self.schedulers[0], name)

    def __repr__(self):
        return (
            f"PerGroupSchedulerWrapper("
            f"{self.scheduler_class.__name__}, "
            f"num_groups={len(self.schedulers)})"
        )


def setup_scheduler(optimizer, scheduler_type: str, epochs: int, train_loader_len: int, **scheduler_kwargs):
    """
    Set up and return the learning rate scheduler based on the scheduler type.

    Supports warmup for all scheduler types via the 'warmup_epochs' parameter.

    Args:
        optimizer: The optimizer instance
        scheduler_type: Type of scheduler ('cosine', 'decaying_cosine', 'cosine_restarts',
                        'decaying_cosine_restarts', 'quadratic_inout', 'cubic_inout',
                        'onecycle', 'step', 'plateau', or None)
        epochs: Number of training epochs
        train_loader_len: Length of the training data loader
        **scheduler_kwargs: Additional arguments for the scheduler
            - warmup_epochs (int): Number of warmup epochs (0 = no warmup). Default: 0
            - warmup_start_factor (float): Initial LR multiplier during warmup. Default: 0.1
            - For ALL schedulers: eta_min or min_lr can be float OR list for per-group minimum LRs
                Example: eta_min=[1e-7, 1e-6, 5e-7] for [stream1, stream2, shared]
            - For 'cosine': t_max (epochs), eta_min (min LR)
            - For 'decaying_cosine': t_max (epochs), eta_min (min LR), max_factor (default=1.0), min_factor (default=1.0)
            - For 'cosine_restarts': t_0 (cycle length in epochs), t_mult (cycle multiplier),
              eta_min (min LR), step_per_batch (bool, default=False)
            - For 'decaying_cosine_restarts': t_0 (cycle length in epochs), t_mult (cycle multiplier),
              eta_min (min LR), restart_decay (default=0.8), step_per_batch (bool, default=False)
            - For 'quadratic_inout': t_max (epochs), eta_min (min LR)
            - For 'cubic_inout': t_max (epochs), eta_min (min LR)
            - For 'onecycle': max_lr, pct_start, anneal_strategy, div_factor, final_div_factor
            - For 'step': step_size (in epochs), gamma
            - For 'plateau': patience (epochs), factor, mode, min_lr, threshold, cooldown, eps

    Returns:
        Scheduler instance or None if no scheduler requested

    Example with warmup:
        >>> # 5 epochs of warmup, then cosine annealing
        >>> scheduler = setup_scheduler(
        ...     optimizer, 'cosine', epochs=80, train_loader_len=40,
        ...     warmup_epochs=5, warmup_start_factor=0.1, t_max=80
        ... )
    """
    if not scheduler_type:
        return None

    # Extract warmup parameters
    warmup_epochs = scheduler_kwargs.pop('warmup_epochs', 0)
    warmup_start_factor = scheduler_kwargs.pop('warmup_start_factor', 0.1)

    # Detect if per-group scheduler parameters were requested
    # Check for eta_min, min_lr, or any other parameter as list
    use_per_group = False
    per_group_params = []

    # Check common per-group parameters
    for param_name in ['eta_min', 'min_lr', 't_max', 't_0']:
        param_value = scheduler_kwargs.get(param_name)
        if isinstance(param_value, (list, tuple)):
            use_per_group = True
            per_group_params.append(param_name)
            break

    if use_per_group:
        # Build per-group scheduler kwargs
        num_groups = len(optimizer.param_groups)

        # Validate that all list/tuple parameters have correct length
        for key, value in scheduler_kwargs.items():
            if isinstance(value, (list, tuple)):
                if len(value) != num_groups:
                    raise ValueError(
                        f"Parameter '{key}' is a list with {len(value)} values, "
                        f"but optimizer has {num_groups} parameter groups. "
                        f"List parameters must have length equal to number of param groups."
                    )

        scheduler_kwargs_list = []

        for i in range(num_groups):
            group_kwargs = {}
            for key, value in scheduler_kwargs.items():
                if isinstance(value, (list, tuple)):
                    # Per-group parameter - use group-specific value
                    group_kwargs[key] = value[i]
                else:
                    # Scalar parameter - use same value for all groups
                    group_kwargs[key] = value
            scheduler_kwargs_list.append(group_kwargs)

        # Map scheduler_type to scheduler class
        scheduler_class_map = {
            'cosine': CosineAnnealingLR,
            'cosine_restarts': CosineAnnealingWarmRestarts,
            'decaying_cosine': DecayingCosineAnnealingLR,
            'decaying_cosine_restarts': DecayingCosineAnnealingWarmRestarts,
            'quadratic_inout': QuadraticInOutLR,
            'cubic_inout': CubicInOutLR,
            'plateau': ReduceLROnPlateau,
            'step': StepLR,
        }

        scheduler_class = scheduler_class_map.get(scheduler_type)
        if scheduler_class is None:
            raise ValueError(f"Per-group scheduling not supported for scheduler_type='{scheduler_type}'")

        # Map user-facing parameter names to actual PyTorch parameter names
        # Our API uses lowercase (t_max, t_0) but PyTorch uses uppercase (T_max, T_0)
        param_name_mapping = {
            't_max': 'T_max',
            't_0': 'T_0',
            't_mult': 'T_mult',
        }

        # Apply parameter name mapping to all group kwargs
        mapped_kwargs_list = []
        for group_kwargs in scheduler_kwargs_list:
            mapped_kwargs = {}
            for key, value in group_kwargs.items():
                # Convert parameter name if there's a mapping
                mapped_key = param_name_mapping.get(key, key)
                mapped_kwargs[mapped_key] = value
            mapped_kwargs_list.append(mapped_kwargs)

        # Create per-group wrapper
        main_scheduler = PerGroupSchedulerWrapper(optimizer, scheduler_class, mapped_kwargs_list)

    else:
        # Standard single-scheduler path (no per-group parameters)
        # Create the main scheduler based on type
        if scheduler_type == 'cosine':
            # T_max is in epochs (scheduler steps per epoch, not per batch)
            t_max = scheduler_kwargs.get('t_max', epochs)
            eta_min = scheduler_kwargs.get('eta_min', 0)
            main_scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
        elif scheduler_type == 'decaying_cosine':
            # DecayingCosineAnnealingLR - cosine annealing with dual decay (max and min)
            t_max = scheduler_kwargs.get('t_max', epochs)
            eta_min = scheduler_kwargs.get('eta_min', 0)
            max_factor = scheduler_kwargs.get('max_factor', 1.0)
            min_factor = scheduler_kwargs.get('min_factor', 1.0)
            main_scheduler = DecayingCosineAnnealingLR(
                optimizer,
                T_max=t_max,
                eta_min=eta_min,
                max_factor=max_factor,
                min_factor=min_factor
            )
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

            main_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=t_0, T_mult=t_mult, eta_min=eta_min)
            # Add metadata so mc_resnet.py knows whether to step per batch or per epoch
            main_scheduler._step_per_batch = step_per_batch
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

            main_scheduler = DecayingCosineAnnealingWarmRestarts(
                optimizer, T_0=t_0, T_mult=t_mult, eta_min=eta_min, restart_decay=restart_decay
            )
            # Add metadata so mc_resnet.py knows whether to step per batch or per epoch
            main_scheduler._step_per_batch = step_per_batch
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
            main_scheduler = OneCycleLR(
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
            main_scheduler = QuadraticInOutLR(optimizer, T_max=t_max, eta_min=eta_min)
        elif scheduler_type == 'cubic_inout':
            # Cubic InOut easing - very smooth S-curve with cubic acceleration/deceleration
            t_max = scheduler_kwargs.get('t_max', epochs)
            eta_min = scheduler_kwargs.get('eta_min', 0)
            main_scheduler = CubicInOutLR(optimizer, T_max=t_max, eta_min=eta_min)
        elif scheduler_type == 'step':
            step_size = scheduler_kwargs.get('step_size', 30)
            gamma = scheduler_kwargs.get('gamma', 0.1)
            main_scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_type == 'plateau':
            # Use scheduler_patience if provided, otherwise fall back to patience from scheduler_kwargs
            scheduler_patience = scheduler_kwargs.get('scheduler_patience', scheduler_kwargs.get('patience', 5))
            factor = scheduler_kwargs.get('factor', 0.5)
            mode = scheduler_kwargs.get('mode', 'min')  # 'min' for loss (default), 'max' for accuracy
            min_lr = scheduler_kwargs.get('min_lr', 1e-7)  # Minimum learning rate
            threshold = scheduler_kwargs.get('threshold', 1e-4)  # Minimum change to qualify as improvement
            cooldown = scheduler_kwargs.get('cooldown', 0)  # Cooldown period after LR reduction
            eps = scheduler_kwargs.get('eps', 1e-8)  # Minimal decay applied to lr
            main_scheduler = ReduceLROnPlateau(
                optimizer, mode=mode, patience=scheduler_patience, factor=factor, min_lr=min_lr, threshold=threshold, cooldown=cooldown, eps=eps
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    # Apply warmup if requested
    if warmup_epochs > 0:
        # Create warmup scheduler using LinearLR
        # LinearLR interpolates from start_factor to 1.0 over total_iters
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=warmup_start_factor,
            end_factor=1.0,
            total_iters=warmup_epochs
        )

        # Use SequentialLR to chain warmup + main scheduler
        # Milestones specify when to switch schedulers (in epochs)
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs]
        )
    else:
        # No warmup, use main scheduler directly
        scheduler = main_scheduler

    return scheduler

def update_scheduler(scheduler, val_loss: float) -> None:
    """
    Update the learning rate scheduler.

    Args:
        scheduler: The scheduler instance (can be PerGroupSchedulerWrapper)
        val_loss: Validation loss for plateau scheduler
    """
    if scheduler is not None:
        # Skip OneCycleLR as it's updated after each batch
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        elif isinstance(scheduler, PerGroupSchedulerWrapper):
            # PerGroupSchedulerWrapper - check if it contains plateau schedulers
            # The wrapper's step() forwards args to all individual schedulers
            if scheduler.scheduler_class == torch.optim.lr_scheduler.ReduceLROnPlateau:
                scheduler.step(val_loss)
            else:
                scheduler.step()
        elif not isinstance(scheduler, OneCycleLR):
            # CosineAnnealingWarmRestarts, QuadraticInOutLR, CubicInOutLR,
            # and other schedulers step per epoch
            scheduler.step()


class StreamSpecificCosineAnnealingLR(torch.optim.lr_scheduler.LRScheduler):
    """
    CosineAnnealingLR with per-parameter-group eta_min support.

    Allows different minimum learning rates for different streams in multi-stream models.
    This is critical for multi-stream architectures where different streams may need
    different minimum LRs (e.g., RGB stream needs lower eta_min than depth stream).

    Uses the same cosine annealing formula as PyTorch's CosineAnnealingLR, but applies
    different eta_min values to each parameter group.

    Args:
        optimizer: Optimizer with one or more parameter groups
        T_max: Maximum number of iterations (epochs)
        eta_min: Minimum LR - can be:
            - float: Same eta_min for all parameter groups (backward compatible)
            - list/tuple: Different eta_min for each parameter group
                Example: [1e-7, 1e-6, 5e-7] for [stream1, stream2, shared]
        last_epoch: The index of last epoch. Default: -1

    Example (single eta_min):
        >>> scheduler = StreamSpecificCosineAnnealingLR(
        ...     optimizer, T_max=80, eta_min=1e-6
        ... )

    Example (per-stream eta_min):
        >>> # optimizer has 3 param groups: [stream1, stream2, shared]
        >>> scheduler = StreamSpecificCosineAnnealingLR(
        ...     optimizer,
        ...     T_max=80,
        ...     eta_min=[1e-7, 1e-6, 5e-7]  # Different floor for each stream
        ... )

    Note:
        - If eta_min is a list, length must match number of parameter groups
        - Works seamlessly with SequentialLR for warmup support
        - Falls back to standard behavior when eta_min is a scalar
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_max: int,
        eta_min = 0,  # Can be float or list/tuple
        last_epoch: int = -1
    ):
        if T_max <= 0 or not isinstance(T_max, int):
            raise ValueError(f"T_max must be a positive integer, got {T_max}")

        self.T_max = T_max

        # Handle eta_min as either scalar or per-group list
        if isinstance(eta_min, (list, tuple)):
            if len(eta_min) != len(optimizer.param_groups):
                raise ValueError(
                    f"eta_min list length ({len(eta_min)}) must match number of "
                    f"parameter groups ({len(optimizer.param_groups)})"
                )
            self.eta_min_list = list(eta_min)
            self.eta_min_is_list = True
        else:
            # Scalar: create list with same value for all groups
            self.eta_min_list = [eta_min] * len(optimizer.param_groups)
            self.eta_min_is_list = False

        # For backward compatibility, store scalar eta_min if provided as scalar
        self.eta_min = eta_min if not isinstance(eta_min, (list, tuple)) else eta_min[0]

        # Call parent __init__ which will call step(0) if last_epoch == -1
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        Compute learning rate using cosine annealing formula with per-group eta_min.

        Formula (per parameter group):
            lr_i = eta_min_i + (base_lr_i - eta_min_i) * (1 + cos(π * epoch / T_max)) / 2

        Returns:
            List of learning rates for each parameter group
        """
        if self.last_epoch == 0:
            # First epoch: return base LRs
            return [group['lr'] for group in self.optimizer.param_groups]
        elif self.last_epoch == self.T_max:
            # At T_max: return eta_min for each group
            return self.eta_min_list
        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            # Restart point (for compatibility with closed form)
            return [
                group['lr'] + (base_lr - eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2
                for base_lr, eta_min, group in zip(
                    self.base_lrs, self.eta_min_list, self.optimizer.param_groups
                )
            ]

        # Standard cosine annealing step
        return [
            (1 + math.cos(math.pi * self.last_epoch / self.T_max)) /
            (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max)) *
            (group['lr'] - eta_min) + eta_min
            for eta_min, group in zip(self.eta_min_list, self.optimizer.param_groups)
        ]

    def _get_closed_form_lr(self):
        """Closed form LR computation (for direct epoch setting)."""
        return [
            eta_min + (base_lr - eta_min) *
            (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
            for base_lr, eta_min in zip(self.base_lrs, self.eta_min_list)
        ]

    def state_dict(self):
        """Returns the state of the scheduler as a dict."""
        state = super().state_dict()
        state['T_max'] = self.T_max
        state['eta_min_list'] = self.eta_min_list
        state['eta_min_is_list'] = self.eta_min_is_list
        state['eta_min'] = self.eta_min
        return state

    def load_state_dict(self, state_dict):
        """Loads the scheduler state."""
        # Extract our custom state before calling parent (use .get() for safety)
        self.T_max = state_dict.pop('T_max', self.T_max)  # Keep current if not in state_dict
        self.eta_min_list = state_dict.pop('eta_min_list', self.eta_min_list)
        self.eta_min_is_list = state_dict.pop('eta_min_is_list', False)
        self.eta_min = state_dict.pop('eta_min', self.eta_min_list[0] if self.eta_min_list else 0)

        # Call parent load_state_dict
        super().load_state_dict(state_dict)

    def __repr__(self):
        """String representation of the scheduler."""
        if self.eta_min_is_list:
            eta_min_str = f"eta_min={self.eta_min_list}"
        else:
            eta_min_str = f"eta_min={self.eta_min}"

        return (
            f"{self.__class__.__name__}("
            f"T_max={self.T_max}, "
            f"{eta_min_str})"
        )


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


class DecayingCosineAnnealingLR(torch.optim.lr_scheduler.LRScheduler):
    """
    CosineAnnealingLR with decaying restart peaks.

    Uses the EXACT same math as PyTorch's CosineAnnealingLR, but applies decay
    multipliers to BOTH base_lrs (max) and eta_min (min) after each T_max cycle.

    This creates a decaying envelope for the cosine oscillations, where both the
    peaks and valleys decrease over time. Allows fine control over the narrowing
    or widening of LR oscillations across training.

    Pattern:
        Cycle 1: high = base_lr,                  low = eta_min
        Cycle 2: high = base_lr * max_factor,     low = eta_min * min_factor
        Cycle 3: high = base_lr * max_factor²,    low = eta_min * min_factor²
        ...

    Example with max_factor=0.8, min_factor=0.9, T_max=20, base_lr=0.1, eta_min=1e-3:
        Epochs 0-20:   0.100 → 0.001 (cosine)
        Epochs 21-40:  0.080 → 0.0009 (cosine, both ends decayed)
        Epochs 41-60:  0.064 → 0.00081 (cosine, continues decaying)
        Epochs 61-80:  0.051 → 0.00073 (cosine, overall downward trend)

    Oscillation behavior:
        - If max_factor == min_factor: Maintains constant oscillation ratio
        - If max_factor < min_factor: Oscillations narrow over time (more stable)
        - If max_factor > min_factor: Oscillations widen over time (more exploration)

    Args:
        optimizer: Wrapped optimizer
        T_max: Number of epochs for each cycle (constant cycle length)
        eta_min: Initial minimum learning rate. Default: 0
        max_factor: Multiply peak LR by this after each cycle. Default: 1.0 (no decay)
        min_factor: Multiply eta_min by this after each cycle. Default: None (uses max_factor)
        last_epoch: The index of last epoch. Default: -1

    Example 1 (Constant ratio - recommended):
        >>> scheduler = DecayingCosineAnnealingLR(
        ...     optimizer, T_max=20, max_factor=0.85, eta_min=1e-5
        ... )
        # Both max and min decay by 15% each cycle, maintaining oscillation amplitude ratio

    Example 2 (Narrowing oscillations):
        >>> scheduler = DecayingCosineAnnealingLR(
        ...     optimizer, T_max=20, max_factor=0.8, min_factor=0.9, eta_min=1e-5
        ... )
        # Max decays faster (20%) than min (10%), oscillations narrow for stability

    Example 3 (Widening oscillations):
        >>> scheduler = DecayingCosineAnnealingLR(
        ...     optimizer, T_max=20, max_factor=0.9, min_factor=0.8, eta_min=1e-5
        ... )
        # Max decays slower (10%) than min (20%), oscillations widen for exploration

    Note:
        - If min_factor not specified, defaults to max_factor (constant ratio)
        - max_factor=1.0 and min_factor=1.0 behaves like CosineAnnealingLR
        - Very aggressive factors (e.g., 0.5) can cause LRs to become too small quickly
        - T_max is constant (unlike CosineAnnealingWarmRestarts with T_mult)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_max: int,
        eta_min: float = 0,
        max_factor = 1.0,  # Can be float or callable
        min_factor = 1.0,  # Can be float or callable
        last_epoch: int = -1
    ):
        if T_max <= 0 or not isinstance(T_max, int):
            raise ValueError(f"T_max must be a positive integer, got {T_max}")

        self.T_max = T_max
        self.eta_min = eta_min
        self.eta_min_initial = eta_min  # Store initial eta_min for tracking

        # Handle max_factor (can be constant or callable)
        if callable(max_factor):
            self.max_factor_fn = max_factor
            self.max_factor = None  # Will be computed dynamically
        else:
            if not 0.0 <= max_factor <= 1.0:
                raise ValueError(f"max_factor must be in [0, 1], got {max_factor}")
            self.max_factor = max_factor
            self.max_factor_fn = None

        # Handle min_factor (can be constant or callable)
        if callable(min_factor):
            self.min_factor_fn = min_factor
            self.min_factor = None  # Will be computed dynamically
        else:
            if not 0.0 <= min_factor <= 1.0:
                raise ValueError(f"min_factor must be in [0, 1], got {min_factor}")
            self.min_factor = min_factor
            self.min_factor_fn = None

        # Track which cycle we're in (0-indexed)
        self.cycle_count = 0

        # Call parent __init__ which will call step(0) if last_epoch == -1
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Retrieve the learning rate of each parameter group."""

        # Apply decay at multiples of T_max
        if self.last_epoch > 0 and self.last_epoch % self.T_max == 0:
            cycle_position = self.last_epoch // self.T_max

            # At valleys (odd multiples: T_max, 3*T_max, 5*T_max...): decay base_lrs (max LR)
            if cycle_position % 2 == 1:
                if self.max_factor_fn:
                    # Callable: apply function to current base_lrs
                    self.base_lrs = [
                        max(self.max_factor_fn(base_lr), self.eta_min)
                        for base_lr in self.base_lrs
                    ]
                else:
                    # Constant: multiply current base_lrs by factor
                    self.base_lrs = [
                        max(base_lr * self.max_factor, self.eta_min)
                        for base_lr in self.base_lrs
                    ]
                self.cycle_count += 1

            # At peaks (even multiples: 2*T_max, 4*T_max, 6*T_max...): decay eta_min (min LR)
            elif cycle_position % 2 == 0:
                if self.min_factor_fn:
                    # Callable: apply function to current eta_min
                    self.eta_min = self.min_factor_fn(self.eta_min)
                else:
                    # Constant: multiply current eta_min by factor
                    self.eta_min = self.eta_min * self.min_factor

        if self.last_epoch == 0:
            return [group["lr"] for group in self.optimizer.param_groups]
        elif self._step_count == 1 and self.last_epoch > 0:
            return [
                self.eta_min
                + (base_lr - self.eta_min)
                * (1 + math.cos((self.last_epoch) * math.pi / self.T_max))
                / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        return [
            (1 + math.cos(math.pi * self.last_epoch / self.T_max))
            / (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max))
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self):
        return [
            self.eta_min
            + (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * self.last_epoch / self.T_max))
            / 2
            for base_lr in self.base_lrs
        ]
    

    @property
    def current_peak_lrs(self):
        """
        Return current peak learning rates (after decay).

        Returns:
            List of current peak LRs for each parameter group
        """
        return self.base_lrs.copy()

    def state_dict(self):
        """
        Returns the state of the scheduler as a dict.

        Returns:
            Dictionary containing scheduler state
        """
        state = super().state_dict()
        state['max_factor'] = self.max_factor
        state['min_factor'] = self.min_factor
        state['eta_min'] = self.eta_min
        state['eta_min_initial'] = self.eta_min_initial
        state['cycle_count'] = self.cycle_count
        return state

    def load_state_dict(self, state_dict):
        """
        Loads the scheduler state.

        Args:
            state_dict: Scheduler state dict from state_dict()
        """
        # Extract our custom state before calling parent
        self.max_factor = state_dict.get('max_factor', 1.0)
        self.min_factor = state_dict.get('min_factor', 1.0)
        self.eta_min = state_dict.get('eta_min', 0)
        self.eta_min_initial = state_dict.get('eta_min_initial', 0)
        self.cycle_count = state_dict.get('cycle_count', 0)

        # Remove our custom keys so parent doesn't complain
        state_dict_copy = state_dict.copy()
        state_dict_copy.pop('max_factor', None)
        state_dict_copy.pop('min_factor', None)
        state_dict_copy.pop('eta_min', None)
        state_dict_copy.pop('eta_min_initial', None)
        state_dict_copy.pop('cycle_count', None)

        # Call parent load_state_dict
        super().load_state_dict(state_dict_copy)

    def __repr__(self):
        """String representation of the scheduler."""
        return (
            f"{self.__class__.__name__}("
            f"T_max={self.T_max}, "
            f"eta_min={self.eta_min}, "
            f"max_factor={self.max_factor}, "
            f"min_factor={self.min_factor}, "
            f"cycles={self.cycle_count})"
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
