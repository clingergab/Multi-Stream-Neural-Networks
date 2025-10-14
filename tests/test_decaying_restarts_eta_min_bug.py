"""
Test for eta_min bug in DecayingCosineAnnealingWarmRestarts.

This test checks if stream-specific LRs can go below eta_min when
using restart_decay with multiple parameter groups.
"""

import sys
import os
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.training.schedulers import DecayingCosineAnnealingWarmRestarts


class DummyMultiStreamModel(nn.Module):
    """Dummy model simulating a 3-stream architecture."""
    def __init__(self):
        super().__init__()
        self.stream1 = nn.Linear(10, 5)
        self.stream2 = nn.Linear(10, 5)
        self.fusion = nn.Linear(10, 5)


def test_eta_min_bug():
    """Test if LRs can go below eta_min with restart_decay."""
    print("=" * 60)
    print("TESTING ETA_MIN BUG IN DECAYING COSINE WARM RESTARTS")
    print("=" * 60)

    # Create dummy multi-stream model
    model = DummyMultiStreamModel()

    # Create optimizer with different base LRs
    # Use a LOW base LR for stream1 to expose the bug
    optimizer = torch.optim.Adam([
        {'params': model.stream1.parameters(), 'lr': 1e-5},   # Very low base LR
        {'params': model.stream2.parameters(), 'lr': 5e-4},
        {'params': model.fusion.parameters(), 'lr': 1e-3}
    ])

    print("\nParameter Groups:")
    print(f"  Stream1 (RGB):  base_lr = {optimizer.param_groups[0]['lr']:.6f}")
    print(f"  Stream2 (Depth): base_lr = {optimizer.param_groups[1]['lr']:.6f}")
    print(f"  Fusion:         base_lr = {optimizer.param_groups[2]['lr']:.6f}")

    # Scheduler with aggressive restart_decay
    T_0 = 10
    T_mult = 1
    eta_min = 1e-6
    restart_decay = 0.5  # Aggressive decay - each restart halves the peak
    total_epochs = 50

    print(f"\nScheduler parameters:")
    print(f"  T_0: {T_0}")
    print(f"  T_mult: {T_mult}")
    print(f"  eta_min: {eta_min}")
    print(f"  restart_decay: {restart_decay} (aggressive!)")
    print(f"  Total epochs: {total_epochs}")

    scheduler = DecayingCosineAnnealingWarmRestarts(
        optimizer,
        T_0=T_0,
        T_mult=T_mult,
        eta_min=eta_min,
        restart_decay=restart_decay
    )

    # Track LRs
    stream1_lrs = []
    stream2_lrs = []
    fusion_lrs = []

    for epoch in range(total_epochs + 1):
        if epoch > 0:
            scheduler.step()
        stream1_lrs.append(optimizer.param_groups[0]['lr'])
        stream2_lrs.append(optimizer.param_groups[1]['lr'])
        fusion_lrs.append(optimizer.param_groups[2]['lr'])

    print("\n" + "=" * 60)
    print("CHECKING FOR LRS BELOW ETA_MIN")
    print("=" * 60)

    # Check Stream1 (the one with low base_lr)
    print(f"\nStream1 (base_lr = 1e-5, eta_min = {eta_min}):")
    min_lr_stream1 = min(stream1_lrs)
    print(f"  Minimum LR reached: {min_lr_stream1:.8f}")
    print(f"  eta_min:            {eta_min:.8f}")

    if min_lr_stream1 < eta_min:
        print(f"  ❌ BUG FOUND! LR went {(eta_min - min_lr_stream1):.8f} below eta_min")

        # Find which epochs went below
        below_epochs = [i for i, lr in enumerate(stream1_lrs) if lr < eta_min]
        print(f"  Epochs below eta_min: {below_epochs[:10]}...")  # Show first 10

        # Show cycle info
        print(f"\n  Cycle analysis:")
        print(f"    After restart 1 (epoch 10): peak = {1e-5 * 0.5:.8f}")
        print(f"    After restart 2 (epoch 20): peak = {1e-5 * 0.5**2:.8f}")
        print(f"    After restart 3 (epoch 30): peak = {1e-5 * 0.5**3:.8f}")
        print(f"    After restart 4 (epoch 40): peak = {1e-5 * 0.5**4:.8f}")
        print(f"    → Peak becomes lower than eta_min = {eta_min:.8f}!")
    else:
        print(f"  ✅ OK: LR never went below eta_min")

    # Check Stream2
    print(f"\nStream2 (base_lr = 5e-4, eta_min = {eta_min}):")
    min_lr_stream2 = min(stream2_lrs)
    print(f"  Minimum LR reached: {min_lr_stream2:.8f}")
    if min_lr_stream2 < eta_min:
        print(f"  ❌ BUG FOUND! LR went below eta_min")
    else:
        print(f"  ✅ OK: LR never went below eta_min")

    # Check Fusion
    print(f"\nFusion (base_lr = 1e-3, eta_min = {eta_min}):")
    min_lr_fusion = min(fusion_lrs)
    print(f"  Minimum LR reached: {min_lr_fusion:.8f}")
    if min_lr_fusion < eta_min:
        print(f"  ❌ BUG FOUND! LR went below eta_min")
    else:
        print(f"  ✅ OK: LR never went below eta_min")

    # Visualization
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATION")
    print("=" * 60)

    epochs = list(range(len(stream1_lrs)))

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Linear scale
    axes[0].plot(epochs, stream1_lrs, label='Stream1 (RGB) - LOW base_lr', linewidth=2, color='blue')
    axes[0].plot(epochs, stream2_lrs, label='Stream2 (Depth)', linewidth=2, color='orange')
    axes[0].plot(epochs, fusion_lrs, label='Fusion', linewidth=2, color='green')
    axes[0].axhline(y=eta_min, color='red', linestyle='--', linewidth=2, label=f'eta_min = {eta_min}')

    # Mark restarts
    for restart in [10, 20, 30, 40]:
        if restart <= total_epochs:
            axes[0].axvline(x=restart, color='gray', linestyle=':', alpha=0.5)

    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Learning Rate', fontsize=12)
    axes[0].set_title('DecayingCosineAnnealingWarmRestarts - Potential eta_min Bug', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Log scale
    axes[1].plot(epochs, stream1_lrs, label='Stream1 (RGB) - LOW base_lr', linewidth=2, color='blue')
    axes[1].plot(epochs, stream2_lrs, label='Stream2 (Depth)', linewidth=2, color='orange')
    axes[1].plot(epochs, fusion_lrs, label='Fusion', linewidth=2, color='green')
    axes[1].axhline(y=eta_min, color='red', linestyle='--', linewidth=2, label=f'eta_min = {eta_min}')

    # Mark restarts
    for restart in [10, 20, 30, 40]:
        if restart <= total_epochs:
            axes[1].axvline(x=restart, color='gray', linestyle=':', alpha=0.5)

    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Learning Rate (log scale)', fontsize=12)
    axes[1].set_title('Log Scale View - Shows Stream1 Dropping Below eta_min', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')

    plt.tight_layout()
    output_path = os.path.join(project_root, 'tests', 'decaying_restarts_eta_min_bug.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")
    plt.close()

    return stream1_lrs, stream2_lrs, fusion_lrs, min_lr_stream1 < eta_min


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("DECAYING COSINE WARM RESTARTS - ETA_MIN BUG TEST")
    print("=" * 60)

    stream1_lrs, stream2_lrs, fusion_lrs, bug_found = test_eta_min_bug()

    print("\n" + "=" * 60)
    if bug_found:
        print("❌ BUG CONFIRMED!")
        print("=" * 60)
        print("\nThe bug occurs when:")
        print("  1. Using restart_decay < 1.0")
        print("  2. A stream has low base_lr")
        print("  3. After multiple restarts, peak LR < eta_min")
        print("  4. Cosine formula then produces LR < eta_min")
        print("\nFix needed: Clamp current_peak to be at least eta_min")
        print("  current_peak = max(base_lr * current_multiplier, eta_min)")
    else:
        print("✅ NO BUG FOUND (but may need different parameters)")
        print("=" * 60)
    print("=" * 60)
