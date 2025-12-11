"""
Seed utilities for reproducible experiments.

Provides functions to set random seeds for PyTorch, NumPy, and Python's random module
to ensure deterministic behavior across runs.
"""

import random
import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True):
    """
    Set random seed for reproducible experiments.

    Args:
        seed: Random seed value (integer)
        deterministic: If True, forces PyTorch to use deterministic algorithms
                      (slower but fully reproducible). If False, uses faster
                      non-deterministic algorithms (still reproducible across runs
                      with same seed, but not bitwise identical).

    Notes:
        - With deterministic=True: Fully reproducible but slower (~10-30% slower)
        - With deterministic=False: Reproducible across runs but may have minor
          floating-point differences due to non-deterministic algorithms

        For research papers, use deterministic=True for final experiments.
        For hyperparameter search, deterministic=False is faster.

    Example:
        >>> from src.utils.seed import set_seed
        >>> set_seed(42)  # All experiments will be reproducible
        >>> model = dm_net18(...)  # Same initialization every time
        >>> # Training will produce identical results
    """
    # Python random module
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    if deterministic:
        # Force deterministic algorithms (slower but fully reproducible)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Use deterministic algorithms where available (PyTorch 1.8+)
        try:
            torch.use_deterministic_algorithms(True)
        except AttributeError:
            # Older PyTorch versions don't have this function
            pass
    else:
        # Allow non-deterministic algorithms (faster)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True  # Auto-tune kernels for speed


def get_worker_seed(worker_id: int, base_seed: int) -> int:
    """
    Generate unique seed for DataLoader worker.

    Ensures each DataLoader worker has a different but reproducible seed
    based on the base seed and worker ID.

    Args:
        worker_id: DataLoader worker ID (0 to num_workers-1)
        base_seed: Base random seed

    Returns:
        Unique seed for this worker

    Example:
        >>> def worker_init_fn(worker_id):
        ...     worker_seed = get_worker_seed(worker_id, 42)
        ...     np.random.seed(worker_seed)
        ...     random.seed(worker_seed)
        >>>
        >>> train_loader = DataLoader(
        ...     dataset, batch_size=32, num_workers=4,
        ...     worker_init_fn=worker_init_fn
        ... )
    """
    return base_seed + worker_id


def make_reproducible_dataloader(base_seed: int):
    """
    Create a worker_init_fn for reproducible DataLoader.

    Args:
        base_seed: Base random seed

    Returns:
        worker_init_fn function to pass to DataLoader

    Example:
        >>> from src.utils.seed import set_seed, make_reproducible_dataloader
        >>>
        >>> set_seed(42)
        >>> worker_init_fn = make_reproducible_dataloader(42)
        >>>
        >>> train_loader = DataLoader(
        ...     dataset, batch_size=32, num_workers=4, shuffle=True,
        ...     worker_init_fn=worker_init_fn,
        ...     generator=torch.Generator().manual_seed(42)
        ... )
    """
    def worker_init_fn(worker_id):
        worker_seed = get_worker_seed(worker_id, base_seed)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return worker_init_fn
