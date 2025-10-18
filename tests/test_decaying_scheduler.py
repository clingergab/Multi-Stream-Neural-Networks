"""
Test DecayingCosineAnnealingLR scheduler.
Verify that it creates smooth cosine curves with decaying peaks.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from src.training.schedulers import DecayingCosineAnnealingLR


def test_decaying_cosine_annealing():
    """
    Test that DecayingCosineAnnealingLR creates smooth cosine curves with decaying peaks.

    Expected behavior with T_max=30, restart_decay=0.8:
    - Epochs 0-30: DOWN from base_lr to eta_min (smooth cosine)
    - Epochs 30-60: UP from eta_min to base_lr*0.8 (smooth cosine, decayed peak)
    - Epochs 60-90: DOWN from base_lr*0.8 to eta_min (smooth cosine)
    - Epochs 90-120: UP from eta_min to base_lr*0.64 (smooth cosine, decayed peak)
    """
    print("\n" + "="*80)
    print("DECAYING COSINE ANNEALING LR TEST")
    print("="*80)

    # Create dummy model and optimizer
    model = nn.Linear(10, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # Create scheduler
    scheduler = DecayingCosineAnnealingLR(
        optimizer,
        T_max=30,
        eta_min=0.001,
        restart_decay=0.5
    )

    # Record learning rates
    lrs = []
    epochs = 150

    for epoch in range(epochs):
        lrs.append(optimizer.param_groups[0]['lr'])
        optimizer.step()
        scheduler.step()

    # Print key checkpoints
    print(f"\nKey Learning Rate Checkpoints:")
    print(f"  Epoch 0 (start):        {lrs[0]:.6f}")
    print(f"  Epoch 30 (1st bottom):  {lrs[30]:.6f}")
    print(f"  Epoch 60 (1st peak):    {lrs[60]:.6f} (expected: {0.1 * 0.8:.6f})")
    print(f"  Epoch 90 (2nd bottom):  {lrs[90]:.6f}")
    print(f"  Epoch 120 (2nd peak):   {lrs[120]:.6f} (expected: {0.1 * 0.8 * 0.8:.6f})")

    # Verify decay at correct positions
    tolerance = 1e-5

    # At epoch 60, should be at decayed peak (0.1 * 0.8 = 0.08)
    expected_60 = 0.1 * 0.8
    if abs(lrs[60] - expected_60) < tolerance:
        print(f"\n  ✅ Epoch 60: Correct decayed peak ({lrs[60]:.6f} ≈ {expected_60:.6f})")
    else:
        print(f"\n  ❌ Epoch 60: Incorrect peak ({lrs[60]:.6f}, expected {expected_60:.6f})")

    # At epoch 120, should be at second decayed peak (0.1 * 0.8 * 0.8 = 0.064)
    expected_120 = 0.1 * 0.8 * 0.8
    if abs(lrs[120] - expected_120) < tolerance:
        print(f"  ✅ Epoch 120: Correct decayed peak ({lrs[120]:.6f} ≈ {expected_120:.6f})")
    else:
        print(f"  ❌ Epoch 120: Incorrect peak ({lrs[120]:.6f}, expected {expected_120:.6f})")

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Full curve
    axes[0].plot(range(epochs), lrs, linewidth=2, color='blue', marker='o', markersize=3, markevery=10)
    axes[0].axhline(y=0.001, color='red', linestyle='--', linewidth=1, alpha=0.5, label='eta_min')
    axes[0].axvline(x=30, color='green', linestyle='--', linewidth=1, alpha=0.5, label='T_max boundaries')
    axes[0].axvline(x=60, color='green', linestyle='--', linewidth=1, alpha=0.5)
    axes[0].axvline(x=90, color='green', linestyle='--', linewidth=1, alpha=0.5)
    axes[0].axvline(x=120, color='green', linestyle='--', linewidth=1, alpha=0.5)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Learning Rate', fontsize=12)
    axes[0].set_title('DecayingCosineAnnealingLR (T_max=30, restart_decay=0.8)', fontsize=14, fontweight='bold')
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    # Plot 2: Zoomed in on first 90 epochs
    axes[1].plot(range(90), lrs[:90], linewidth=2, color='blue', marker='o', markersize=4, markevery=5)
    axes[1].axhline(y=0.001, color='red', linestyle='--', linewidth=1, alpha=0.5, label='eta_min')
    axes[1].axvline(x=30, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Decay points')
    axes[1].axvline(x=60, color='orange', linestyle='--', linewidth=1, alpha=0.5)

    # Annotate key points
    axes[1].annotate('Start: 0.1', xy=(0, lrs[0]), xytext=(5, lrs[0]+0.01),
                     arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                     fontsize=10, fontweight='bold')
    axes[1].annotate(f'Bottom: {lrs[30]:.4f}', xy=(30, lrs[30]), xytext=(30, lrs[30]-0.015),
                     arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                     fontsize=10, fontweight='bold')
    axes[1].annotate(f'Peak: {lrs[60]:.4f}', xy=(60, lrs[60]), xytext=(55, lrs[60]+0.01),
                     arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                     fontsize=10, fontweight='bold')

    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Learning Rate', fontsize=12)
    axes[1].set_title('First 90 Epochs (Detailed View)', fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('tests/test_decaying_scheduler.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: tests/test_decaying_scheduler.png")
    print("="*80 + "\n")


if __name__ == '__main__':
    test_decaying_cosine_annealing()
