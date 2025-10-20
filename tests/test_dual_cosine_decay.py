"""
Test DecayingCosineAnnealingLR with cosine transform for both min and max factors.

This creates narrowing oscillations where both peaks and valleys decay smoothly
following cosine curves, with max_factor decaying 90% as fast as min_factor.
"""

import math
import torch
import matplotlib.pyplot as plt
from src.training.schedulers import DecayingCosineAnnealingLR


def make_cosine_decay(start_lr, min_lr, num_decay_events):
    """
    Create a cosine decay function that transforms current LR through a cosine curve.

    Args:
        start_lr: Initial LR value (e.g., 1e-4)
        min_lr: Final LR value (e.g., 1e-7)
        num_decay_events: Number of decay events (e.g., 3 for epochs 20, 40, 60)

    Returns:
        Function that takes current LR and returns next LR
    """
    A = (start_lr - min_lr) / 2  # Amplitude
    B = (start_lr + min_lr) / 2  # Offset
    step = math.pi / num_decay_events  # Step size on cosine curve

    def decay_fn(x):
        # Normalize to [-1, 1]
        normalized = (x - B) / A
        normalized = max(-1, min(1, normalized))  # Clamp for arccos

        # Find current position and step forward
        theta = math.acos(normalized)
        next_theta = min(theta + step, math.pi)

        # Calculate new value
        return A * math.cos(next_theta) + B

    return decay_fn


def make_gentler_cosine_decay(start_lr, min_lr, num_decay_events, gentleness=0.9):
    """
    Create a gentler cosine decay (90% of the original decay).

    This makes the decay less aggressive, useful for max_factor to create
    narrowing oscillations.
    """
    # Calculate what the "gentle end" would be
    # If original goes start_lr -> min_lr
    # Gentle version goes start_lr -> (start_lr - (start_lr - min_lr) * gentleness)
    gentle_min_lr = start_lr - (start_lr - min_lr) * gentleness

    A = (start_lr - gentle_min_lr) / 2
    B = (start_lr + gentle_min_lr) / 2
    step = math.pi / num_decay_events

    def decay_fn(x):
        normalized = (x - B) / A
        normalized = max(-1, min(1, normalized))
        theta = math.acos(normalized)
        next_theta = min(theta + step, math.pi)
        return A * math.cos(next_theta) + B

    return decay_fn


def test_dual_cosine_decay():
    """Test with both min_factor and max_factor using cosine transforms."""
    print("="*80)
    print("Dual Cosine Decay Test: Narrowing Oscillations")
    print("="*80)

    # Training configuration
    T_max = 10
    total_epochs = 60
    num_decay_events = total_epochs // (2 * T_max)  # 3 decay events

    print(f"\nConfiguration:")
    print(f"  T_max: {T_max}")
    print(f"  Total epochs: {total_epochs}")
    print(f"  Decay events: {num_decay_events} (at epochs 20, 40, 60)")

    # LR range configuration
    base_lr = 1e-4
    eta_min_start = 9.3e-5  # Start at 93% of base_lr
    eta_min_end = 1e-7

    print(f"\nLR Configuration:")
    print(f"  Base LR (initial): {base_lr:.6e}")
    print(f"  eta_min (start): {eta_min_start:.6e}")
    print(f"  eta_min (target end): {eta_min_end:.6e}")
    print(f"  Initial oscillation width: {(base_lr - eta_min_start):.6e}")

    # Create decay functions
    min_factor_fn = make_cosine_decay(eta_min_start, eta_min_end, num_decay_events)
    max_factor_fn = make_gentler_cosine_decay(base_lr, base_lr * 0.1, num_decay_events, gentleness=0.9)

    print(f"\nDecay Functions:")
    print(f"  min_factor: Cosine decay from {eta_min_start:.6e} to {eta_min_end:.6e}")
    print(f"  max_factor: 90% gentler cosine decay (creates narrowing oscillations)")

    # Create model and optimizer
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr)

    # Create scheduler
    scheduler = DecayingCosineAnnealingLR(
        optimizer,
        T_max=T_max,
        eta_min=eta_min_start,
        min_factor=min_factor_fn,
        max_factor=max_factor_fn,
    )

    # Track LR values
    lrs = []
    eta_mins = []
    base_lrs = []

    # Also create a regular cosine schedule for comparison
    from torch.optim.lr_scheduler import CosineAnnealingLR
    model_regular = torch.nn.Linear(10, 10)
    optimizer_regular = torch.optim.SGD(model_regular.parameters(), lr=base_lr)
    scheduler_regular = CosineAnnealingLR(optimizer_regular, T_max=total_epochs, eta_min=eta_min_end)
    regular_lrs = []

    print(f"\nRunning simulation for {total_epochs} epochs...")
    print("="*80)

    for epoch in range(total_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        current_eta_min = scheduler.eta_min
        current_base_lr = scheduler.base_lrs[0]

        lrs.append(current_lr)
        eta_mins.append(current_eta_min)
        base_lrs.append(current_base_lr)

        # Track regular cosine
        regular_lrs.append(optimizer_regular.param_groups[0]['lr'])
        optimizer_regular.step()
        scheduler_regular.step()

        # Print key epochs
        if epoch in [0, 10, 20, 30, 40, 50, 59]:
            oscillation_width = current_base_lr - current_eta_min
            print(f"Epoch {epoch:2d}: LR={current_lr:.6e}, "
                  f"eta_min={current_eta_min:.6e}, "
                  f"base_lr={current_base_lr:.6e}, "
                  f"oscillation={oscillation_width:.6e}")

        # Simulate training step
        optimizer.step()
        scheduler.step()

    print("="*80)

    # Calculate oscillation widths over time
    oscillation_widths = [base_lrs[i] - eta_mins[i] for i in range(len(lrs))]

    print(f"\nOscillation Analysis:")
    print(f"  Initial width: {oscillation_widths[0]:.6e}")
    print(f"  Width at epoch 20: {oscillation_widths[20]:.6e}")
    print(f"  Width at epoch 40: {oscillation_widths[40]:.6e}")
    print(f"  Final width: {oscillation_widths[-1]:.6e}")
    if oscillation_widths[0] > 0:
        print(f"  Narrowing ratio: {oscillation_widths[-1] / oscillation_widths[0]:.2%}")
    else:
        print(f"  Narrowing ratio: N/A (initial width is zero)")

    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Full LR schedule with envelope
    ax1.plot(lrs, label='Dual-Decay LR (oscillating)', linewidth=2, color='blue')
    ax1.plot(regular_lrs, label='Regular Cosine LR', linewidth=2, color='orange', linestyle=':')
    ax1.plot(eta_mins, label='eta_min (valley floor)', linestyle='--', linewidth=2, color='red')
    ax1.plot(base_lrs, label='base_lr (peak ceiling)', linestyle='--', linewidth=2, color='green')
    ax1.fill_between(range(len(lrs)), eta_mins, base_lrs, alpha=0.2, color='gray', label='Oscillation envelope')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Learning Rate')
    ax1.set_title('Dual Cosine Decay vs Regular Cosine: Narrowing Oscillations')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Zoom on eta_min decay vs regular cosine
    ax2.plot(eta_mins, label='Dual-Decay eta_min (valley)', linewidth=2, color='red')
    ax2.plot(regular_lrs, label='Regular Cosine LR', linewidth=2, color='orange', linestyle=':')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Valley Floor vs Regular Cosine Decay')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Zoom on base_lr decay
    ax3.plot(base_lrs, label='base_lr', linewidth=2, color='green')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('base_lr')
    ax3.set_title('Peak Ceiling Decay (max_factor with 90% gentler cosine)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Oscillation width over time
    ax4.plot(oscillation_widths, label='Oscillation Width', linewidth=2, color='purple')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Width (base_lr - eta_min)')
    ax4.set_title('Oscillation Width (Narrowing Pattern)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('tests/test_dual_cosine_decay.png', dpi=150)
    plt.close()

    print(f"\n✅ Test complete! Saved visualization to: tests/test_dual_cosine_decay.png")

    return lrs, eta_mins, base_lrs


if __name__ == '__main__':
    print("\nTesting Dual Cosine Decay with Narrowing Oscillations...")
    print("Both min_factor and max_factor use cosine transforms!")
    print("max_factor is 90% gentler to create narrowing oscillations.\n")

    test_dual_cosine_decay()

    print("\n" + "="*80)
    print("DUAL COSINE DECAY TEST COMPLETE!")
    print("="*80)
