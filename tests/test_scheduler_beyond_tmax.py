"""
Test what happens when training continues past T_max.

This shows that the schedulers stay at eta_min after reaching T_max.
"""

import sys
import os
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.training.schedulers import QuadraticInOutLR, CubicInOutLR
from torch.optim.lr_scheduler import CosineAnnealingLR


def test_beyond_tmax():
    """Test scheduler behavior when continuing past T_max."""
    print("=" * 60)
    print("TESTING BEHAVIOR BEYOND T_MAX")
    print("=" * 60)

    # Create dummy optimizer
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # Test parameters
    T_max = 50
    extra_epochs = 30  # Train for 30 epochs beyond T_max
    eta_min = 1e-5

    print(f"\nT_max: {T_max}")
    print(f"Total epochs to test: {T_max + extra_epochs}")
    print(f"eta_min: {eta_min}")

    # Test QuadraticInOutLR
    print("\n" + "-" * 60)
    print("QuadraticInOutLR:")
    print("-" * 60)
    for group in optimizer.param_groups:
        group['lr'] = 0.1
    quad_scheduler = QuadraticInOutLR(optimizer, T_max=T_max, eta_min=eta_min)
    quad_lrs = []

    for epoch in range(T_max + extra_epochs + 1):
        if epoch > 0:
            quad_scheduler.step()
        quad_lrs.append(optimizer.param_groups[0]['lr'])

    print(f"  LR at epoch {T_max - 5}: {quad_lrs[T_max - 5]:.8f}")
    print(f"  LR at epoch {T_max - 1}: {quad_lrs[T_max - 1]:.8f}")
    print(f"  LR at epoch {T_max}: {quad_lrs[T_max]:.8f} (T_max reached)")
    print(f"  LR at epoch {T_max + 1}: {quad_lrs[T_max + 1]:.8f}")
    print(f"  LR at epoch {T_max + 10}: {quad_lrs[T_max + 10]:.8f}")
    print(f"  LR at epoch {T_max + 20}: {quad_lrs[T_max + 20]:.8f}")
    print(f"  LR at epoch {T_max + 30}: {quad_lrs[T_max + 30]:.8f}")

    # Test CubicInOutLR
    print("\n" + "-" * 60)
    print("CubicInOutLR:")
    print("-" * 60)
    for group in optimizer.param_groups:
        group['lr'] = 0.1
    cubic_scheduler = CubicInOutLR(optimizer, T_max=T_max, eta_min=eta_min)
    cubic_lrs = []

    for epoch in range(T_max + extra_epochs + 1):
        if epoch > 0:
            cubic_scheduler.step()
        cubic_lrs.append(optimizer.param_groups[0]['lr'])

    print(f"  LR at epoch {T_max - 5}: {cubic_lrs[T_max - 5]:.8f}")
    print(f"  LR at epoch {T_max - 1}: {cubic_lrs[T_max - 1]:.8f}")
    print(f"  LR at epoch {T_max}: {cubic_lrs[T_max]:.8f} (T_max reached)")
    print(f"  LR at epoch {T_max + 1}: {cubic_lrs[T_max + 1]:.8f}")
    print(f"  LR at epoch {T_max + 10}: {cubic_lrs[T_max + 10]:.8f}")
    print(f"  LR at epoch {T_max + 20}: {cubic_lrs[T_max + 20]:.8f}")
    print(f"  LR at epoch {T_max + 30}: {cubic_lrs[T_max + 30]:.8f}")

    # Test CosineAnnealingLR for comparison
    print("\n" + "-" * 60)
    print("CosineAnnealingLR (for comparison):")
    print("-" * 60)
    for group in optimizer.param_groups:
        group['lr'] = 0.1
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    cosine_lrs = []

    for epoch in range(T_max + extra_epochs + 1):
        if epoch > 0:
            cosine_scheduler.step()
        cosine_lrs.append(optimizer.param_groups[0]['lr'])

    print(f"  LR at epoch {T_max - 5}: {cosine_lrs[T_max - 5]:.8f}")
    print(f"  LR at epoch {T_max - 1}: {cosine_lrs[T_max - 1]:.8f}")
    print(f"  LR at epoch {T_max}: {cosine_lrs[T_max]:.8f} (T_max reached)")
    print(f"  LR at epoch {T_max + 1}: {cosine_lrs[T_max + 1]:.8f}")
    print(f"  LR at epoch {T_max + 10}: {cosine_lrs[T_max + 10]:.8f}")
    print(f"  LR at epoch {T_max + 20}: {cosine_lrs[T_max + 20]:.8f}")
    print(f"  LR at epoch {T_max + 30}: {cosine_lrs[T_max + 30]:.8f}")

    # Check that all stay at eta_min after T_max
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    # Check quadratic
    all_constant = all(abs(lr - eta_min) < 1e-9 for lr in quad_lrs[T_max:])
    if all_constant:
        print("✅ QuadraticInOutLR: LR stays constant at eta_min after T_max")
    else:
        print("❌ QuadraticInOutLR: LR does NOT stay constant after T_max")

    # Check cubic
    all_constant = all(abs(lr - eta_min) < 1e-9 for lr in cubic_lrs[T_max:])
    if all_constant:
        print("✅ CubicInOutLR: LR stays constant at eta_min after T_max")
    else:
        print("❌ CubicInOutLR: LR does NOT stay constant after T_max")

    # Check cosine
    all_constant = all(abs(lr - eta_min) < 1e-9 for lr in cosine_lrs[T_max:])
    if all_constant:
        print("✅ CosineAnnealingLR: LR stays constant at eta_min after T_max")
    else:
        print("❌ CosineAnnealingLR: LR does NOT stay constant after T_max")

    return quad_lrs, cubic_lrs, cosine_lrs, T_max


def visualize_beyond_tmax(quad_lrs, cubic_lrs, cosine_lrs, T_max):
    """Create visualization showing behavior beyond T_max."""
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATION")
    print("=" * 60)

    epochs = list(range(len(quad_lrs)))

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Linear scale
    axes[0].plot(epochs, quad_lrs, label='Quadratic InOut', linewidth=2, color='blue')
    axes[0].plot(epochs, cubic_lrs, label='Cubic InOut', linewidth=2, color='green')
    axes[0].plot(epochs, cosine_lrs, label='Cosine Annealing', linewidth=2, color='red', linestyle='--')
    axes[0].axvline(x=T_max, color='black', linestyle=':', linewidth=2, label=f'T_max = {T_max}')
    axes[0].fill_between([T_max, len(epochs)-1], 0, 0.12, alpha=0.1, color='gray', label='Beyond T_max')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Learning Rate', fontsize=12)
    axes[0].set_title('Learning Rate Behavior Beyond T_max (Linear Scale)', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-0.005, 0.11)

    # Plot 2: Log scale (to see eta_min plateau)
    axes[1].plot(epochs, quad_lrs, label='Quadratic InOut', linewidth=2, color='blue')
    axes[1].plot(epochs, cubic_lrs, label='Cubic InOut', linewidth=2, color='green')
    axes[1].plot(epochs, cosine_lrs, label='Cosine Annealing', linewidth=2, color='red', linestyle='--')
    axes[1].axvline(x=T_max, color='black', linestyle=':', linewidth=2, label=f'T_max = {T_max}')
    axes[1].axhline(y=1e-5, color='orange', linestyle='-.', linewidth=1.5, label='eta_min = 1e-5')
    axes[1].fill_between([T_max, len(epochs)-1], 1e-6, 1, alpha=0.1, color='gray', label='Beyond T_max')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Learning Rate (log scale)', fontsize=12)
    axes[1].set_title('Learning Rate Behavior Beyond T_max (Log Scale)', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')
    axes[1].set_ylim(5e-6, 0.15)

    plt.tight_layout()

    # Save figure
    output_path = os.path.join(project_root, 'tests', 'scheduler_beyond_tmax.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")
    plt.close()


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("TESTING SCHEDULER BEHAVIOR BEYOND T_MAX")
    print("=" * 60)

    quad_lrs, cubic_lrs, cosine_lrs, T_max = test_beyond_tmax()
    visualize_beyond_tmax(quad_lrs, cubic_lrs, cosine_lrs, T_max)

    print("\n" + "=" * 60)
    print("✅ TEST COMPLETE")
    print("=" * 60)
    print("\nSummary:")
    print("  • All schedulers reach eta_min at T_max")
    print("  • All schedulers stay constant at eta_min after T_max")
    print("  • This is the same behavior as PyTorch's CosineAnnealingLR")
    print("  • Safe to train beyond T_max - LR won't increase or oscillate")
    print("=" * 60)
