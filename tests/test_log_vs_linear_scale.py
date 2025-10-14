"""Demonstrate how log-scale y-axis makes QuadraticInOut look like Cosine."""

import sys
sys.path.insert(0, '/Users/gclinger/Documents/projects/Multi-Stream-Neural-Networks')

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.training.schedulers import QuadraticInOutLR
import matplotlib.pyplot as plt


def demonstrate_log_scale_effect():
    """Show how log scale makes the curves look identical."""
    T_max = 50
    eta_min = 1e-6
    base_lr = 0.1

    # Create schedulers
    model1 = nn.Linear(10, 10)
    optimizer1 = torch.optim.SGD(model1.parameters(), lr=base_lr)
    quad_scheduler = QuadraticInOutLR(optimizer1, T_max=T_max, eta_min=eta_min)

    model2 = nn.Linear(10, 10)
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=base_lr)
    cosine_scheduler = CosineAnnealingLR(optimizer2, T_max=T_max, eta_min=eta_min)

    # Collect LR values
    quad_lrs = []
    cosine_lrs = []

    for epoch in range(T_max + 1):
        quad_lrs.append(optimizer1.param_groups[0]['lr'])
        cosine_lrs.append(optimizer2.param_groups[0]['lr'])
        quad_scheduler.step()
        cosine_scheduler.step()

    # Create comparison: Linear vs Log scale
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Linear scale (differences visible)
    axes[0].plot(quad_lrs, label='QuadraticInOutLR', linewidth=2.5, color='blue')
    axes[0].plot(cosine_lrs, label='CosineAnnealingLR', linewidth=2, linestyle='--', alpha=0.8, color='orange')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Learning Rate', fontsize=12)
    axes[0].set_title('Linear Scale (Differences Visible)', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Highlight differences with annotations
    axes[0].annotate('Subtle differences\nvisible here',
                     xy=(15, quad_lrs[15]), xytext=(25, 0.07),
                     arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                     fontsize=10, color='red')

    # Plot 2: Log scale (differences hidden)
    axes[1].plot(quad_lrs, label='QuadraticInOutLR', linewidth=2.5, color='blue')
    axes[1].plot(cosine_lrs, label='CosineAnnealingLR', linewidth=2, linestyle='--', alpha=0.8, color='orange')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Learning Rate (log scale)', fontsize=12)
    axes[1].set_title('Log Scale (Curves Look Identical)', fontsize=14, fontweight='bold')
    axes[1].set_yscale('log')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3, which='both')

    # Highlight that they look identical
    axes[1].annotate('Curves appear\nidentical on log scale!',
                     xy=(15, quad_lrs[15]), xytext=(25, 0.02),
                     arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
                     fontsize=10, color='green')

    plt.tight_layout()
    plt.savefig('tests/log_vs_linear_scale_effect.png', dpi=150)
    print("Saved comparison to tests/log_vs_linear_scale_effect.png")

    # Demonstrate with actual differences
    print("\n=== Why Log Scale Hides the Differences ===")
    print("\nAt various epochs:")
    print(f"{'Epoch':<8} {'QuadInOut':<12} {'Cosine':<12} {'Abs Diff':<12} {'% Diff':<10}")
    print("-" * 60)

    for epoch in [0, 10, 20, 30, 40, 49]:
        diff = abs(quad_lrs[epoch] - cosine_lrs[epoch])
        pct_diff = (diff / quad_lrs[epoch]) * 100 if quad_lrs[epoch] > 0 else 0
        print(f"{epoch:<8} {quad_lrs[epoch]:<12.6f} {cosine_lrs[epoch]:<12.6f} {diff:<12.6f} {pct_diff:<10.2f}%")

    print("\nKey insight:")
    print("- On LINEAR scale: Absolute differences of ~0.001-0.003 are visible in the middle")
    print("- On LOG scale: These small absolute differences get compressed and become invisible")
    print("- Log scale emphasizes relative (multiplicative) changes, not absolute differences")
    print("- Since both curves decay from 0.1 → 1e-6 similarly, they look identical on log scale")


if __name__ == "__main__":
    demonstrate_log_scale_effect()
