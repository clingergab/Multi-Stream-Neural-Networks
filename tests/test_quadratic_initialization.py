"""Test QuadraticInOutLR initialization and stepping behavior."""

import torch
import torch.nn as nn
from src.training.schedulers import QuadraticInOutLR


def test_quadratic_init_and_step():
    """Test that QuadraticInOutLR starts at base_lr and steps correctly."""
    # Create a simple model and optimizer
    model = nn.Linear(10, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # Create scheduler with T_max=50
    scheduler = QuadraticInOutLR(optimizer, T_max=50, eta_min=1e-6)

    print("\n=== QuadraticInOutLR Initialization Test ===")
    print(f"After initialization:")
    print(f"  scheduler.last_epoch = {scheduler.last_epoch}")
    print(f"  optimizer LR = {optimizer.param_groups[0]['lr']:.6f}")
    print(f"  scheduler.get_last_lr() = {scheduler.get_last_lr()}")

    # Manually calculate expected LR for epoch 0
    print(f"\nExpected LR at epoch 0: {scheduler.base_lrs[0]:.6f}")

    # Now simulate first training epoch
    # In training loop, after epoch 0 completes, we call scheduler.step()
    print(f"\n--- Calling scheduler.step() (simulating end of epoch 0) ---")
    scheduler.step()
    print(f"After first step():")
    print(f"  scheduler.last_epoch = {scheduler.last_epoch}")
    print(f"  optimizer LR = {optimizer.param_groups[0]['lr']:.6f}")

    # Calculate expected LR for epoch 1
    t = 1 / 50
    if t < 0.5:
        easing = 2 * t * t
        factor = 1 - easing
    else:
        easing = -2 * (t - 1) * (t - 1) + 1
        factor = 1 - easing
    expected_lr_epoch1 = 1e-6 + (0.1 - 1e-6) * factor
    print(f"Expected LR at epoch 1: {expected_lr_epoch1:.6f}")

    # Continue stepping and check a few more epochs
    print(f"\n--- Learning rate progression ---")
    lrs = [scheduler.base_lrs[0]]  # Start with base LR
    for epoch in range(1, 11):
        scheduler.step()
        lrs.append(optimizer.param_groups[0]['lr'])
        print(f"Epoch {epoch:2d}: LR = {optimizer.param_groups[0]['lr']:.8f}, last_epoch = {scheduler.last_epoch}")

    # Check if we're skipping epoch 0
    print(f"\n=== Issue Check ===")
    print(f"Did we start at base_lr (0.1)? {abs(lrs[0] - 0.1) < 1e-9}")
    print(f"After first step, are we at epoch 1 LR? {abs(lrs[1] - expected_lr_epoch1) < 1e-9}")


if __name__ == "__main__":
    test_quadratic_init_and_step()
