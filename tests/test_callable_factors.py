"""
Test DecayingCosineAnnealingLR with callable decay factor functions.

Functions receive only the current value (x) and return the new value.
This allows for sophisticated transforms like shifted cosine functions!
"""

import math
import torch
import matplotlib.pyplot as plt
from src.training.schedulers import DecayingCosineAnnealingLR


def test_cosine_transform():
    """Test scheduler with cosine transform function."""
    print("\n" + "="*80)
    print("TEST 1: Cosine Transform (Shifted Cosine)")
    print("="*80)

    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=8e-5)

    # Cosine transform: maps x through cosine curve
    # For eta_min starting at 7.5e-5, decaying to ~1e-7 over 3 steps
    # Range: [1e-7, 7.5e-5] -> A = (7.5e-5 - 1e-7)/2, B = (7.5e-5 + 1e-7)/2

    eta_min_start = 7.5e-5
    eta_min_end = 1e-7

    A = (eta_min_start - eta_min_end) / 2  # Amplitude
    B = (eta_min_start + eta_min_end) / 2  # Offset
    step = math.pi / 3  # 3 decay events -> divide curve into 3 steps

    def cosine_transform(x):
        """Transform x through cosine curve, advancing by one step."""
        # Normalize to [-1, 1]
        normalized = (x - B) / A
        normalized = max(-1, min(1, normalized))  # Clamp for arccos

        # Find current position and step forward
        theta = math.acos(normalized)
        next_theta = min(theta + step, math.pi)

        # Calculate new value
        next_normalized = math.cos(next_theta)
        return A * next_normalized + B

    print(f"Cosine parameters: A={A:.6e}, B={B:.6e}, step={step:.4f}")

    scheduler = DecayingCosineAnnealingLR(
        optimizer,
        T_max=10,
        eta_min=eta_min_start,
        min_factor=cosine_transform,
        max_factor=0.7,
    )

    lrs = []
    eta_mins = []

    for epoch in range(60):
        lrs.append(optimizer.param_groups[0]['lr'])
        eta_mins.append(scheduler.eta_min)

        if epoch in [0, 10, 20, 30, 40, 50]:
            print(f"Epoch {epoch}: LR={optimizer.param_groups[0]['lr']:.6e}, eta_min={scheduler.eta_min:.6e}")

        optimizer.step()
        scheduler.step()

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(lrs, label='Learning Rate', linewidth=2)
    ax1.plot(eta_mins, label='eta_min (valley floor)', linestyle='--', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Learning Rate')
    ax1.set_title('Cosine Transform: eta_min follows pure cosine curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(eta_mins, label='eta_min', color='orange', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('eta_min')
    ax2.set_title('eta_min Decay Pattern (cosine curve)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('tests/test_cosine_transform.png', dpi=150)
    plt.close()

    print(f"\n✅ Cosine transform test complete! Saved to: tests/test_cosine_transform.png")


def test_simple_multiply():
    """Test scheduler with simple multiplicative decay."""
    print("\n" + "="*80)
    print("TEST 2: Simple Multiplicative Decay")
    print("="*80)

    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=8e-5)

    # Simple: multiply by constant each time
    scheduler = DecayingCosineAnnealingLR(
        optimizer,
        T_max=10,
        eta_min=7.5e-5,
        min_factor=lambda x: x * 0.5,  # Halve each cycle
        max_factor=0.7,
    )

    lrs = []
    eta_mins = []

    for epoch in range(60):
        lrs.append(optimizer.param_groups[0]['lr'])
        eta_mins.append(scheduler.eta_min)

        if epoch in [0, 10, 20, 30, 40, 50]:
            print(f"Epoch {epoch}: LR={optimizer.param_groups[0]['lr']:.6e}, eta_min={scheduler.eta_min:.6e}")

        optimizer.step()
        scheduler.step()

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(lrs, label='Learning Rate', linewidth=2)
    ax1.plot(eta_mins, label='eta_min (valley floor)', linestyle='--', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Learning Rate')
    ax1.set_title('Simple Decay: eta_min = eta_min * 0.5 each cycle')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(eta_mins, label='eta_min', color='orange', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('eta_min')
    ax2.set_title('eta_min Decay Pattern (constant halving)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('tests/test_simple_multiply.png', dpi=150)
    plt.close()

    print(f"\n✅ Simple multiply test complete! Saved to: tests/test_simple_multiply.png")


def test_power_decay():
    """Test scheduler with power law decay."""
    print("\n" + "="*80)
    print("TEST 3: Power Law Decay")
    print("="*80)

    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=8e-5)

    # Power law: x^(3/4) - gentler than sqrt, more aggressive than linear
    scheduler = DecayingCosineAnnealingLR(
        optimizer,
        T_max=10,
        eta_min=7.5e-5,
        min_factor=lambda x: x ** 0.75,
        max_factor=0.7,
    )

    lrs = []
    eta_mins = []

    for epoch in range(60):
        lrs.append(optimizer.param_groups[0]['lr'])
        eta_mins.append(scheduler.eta_min)

        if epoch in [0, 10, 20, 30, 40, 50]:
            print(f"Epoch {epoch}: LR={optimizer.param_groups[0]['lr']:.6e}, eta_min={scheduler.eta_min:.6e}")

        optimizer.step()
        scheduler.step()

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(lrs, label='Learning Rate', linewidth=2)
    ax1.plot(eta_mins, label='eta_min (valley floor)', linestyle='--', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Learning Rate')
    ax1.set_title('Power Law Decay: eta_min = eta_min^0.75 each cycle')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(eta_mins, label='eta_min', color='orange', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('eta_min')
    ax2.set_title('eta_min Decay Pattern (power law)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('tests/test_power_decay.png', dpi=150)
    plt.close()

    print(f"\n✅ Power law test complete! Saved to: tests/test_power_decay.png")


if __name__ == '__main__':
    print("Testing DecayingCosineAnnealingLR with one-parameter callable factors...")
    print("Functions receive only current value (x) and return new value!")
    print("="*80)

    test_cosine_transform()
    test_simple_multiply()
    test_power_decay()

    print("\n" + "="*80)
    print("ALL TESTS COMPLETE!")
    print("="*80)
    print("\nGenerated visualizations:")
    print("  - tests/test_cosine_transform.png")
    print("  - tests/test_simple_multiply.png")
    print("  - tests/test_power_decay.png")
