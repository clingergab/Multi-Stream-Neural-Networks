"""Compare PyTorch CosineAnnealingLR initialization with our QuadraticInOutLR."""

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.training.schedulers import QuadraticInOutLR


def test_pytorch_cosine_init():
    """Test how PyTorch CosineAnnealingLR initializes and steps."""
    model = nn.Linear(10, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

    print("\n=== PyTorch CosineAnnealingLR ===")
    print(f"After initialization:")
    print(f"  scheduler.last_epoch = {scheduler.last_epoch}")
    print(f"  optimizer LR = {optimizer.param_groups[0]['lr']:.6f}")
    print(f"  scheduler.get_last_lr() = {scheduler.get_last_lr()}")

    print(f"\n--- Stepping through epochs ---")
    for epoch in range(5):
        print(f"Epoch {epoch}: LR = {optimizer.param_groups[0]['lr']:.8f}, last_epoch = {scheduler.last_epoch}")
        # Train here...
        scheduler.step()

    print(f"\nAfter 5 steps: last_epoch = {scheduler.last_epoch}")


def test_quadratic_init():
    """Test how our QuadraticInOutLR initializes and steps."""
    model = nn.Linear(10, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = QuadraticInOutLR(optimizer, T_max=50, eta_min=1e-6)

    print("\n\n=== Our QuadraticInOutLR ===")
    print(f"After initialization:")
    print(f"  scheduler.last_epoch = {scheduler.last_epoch}")
    print(f"  optimizer LR = {optimizer.param_groups[0]['lr']:.6f}")
    print(f"  scheduler.get_last_lr() = {scheduler.get_last_lr()}")

    print(f"\n--- Stepping through epochs ---")
    for epoch in range(5):
        print(f"Epoch {epoch}: LR = {optimizer.param_groups[0]['lr']:.8f}, last_epoch = {scheduler.last_epoch}")
        # Train here...
        scheduler.step()

    print(f"\nAfter 5 steps: last_epoch = {scheduler.last_epoch}")


if __name__ == "__main__":
    test_pytorch_cosine_init()
    test_quadratic_init()
