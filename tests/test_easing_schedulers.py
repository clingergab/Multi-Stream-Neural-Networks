"""
Test and visualize QuadraticInOutLR and CubicInOutLR schedulers.

This test script compares the new easing-based schedulers with
CosineAnnealingLR to show their different behaviors.
"""

import sys
import os
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.training.schedulers import QuadraticInOutLR, CubicInOutLR
from torch.optim.lr_scheduler import CosineAnnealingLR


def test_scheduler_behavior():
    """Test that schedulers properly decrease learning rate over time."""
    print("=" * 60)
    print("TESTING SCHEDULER BEHAVIOR")
    print("=" * 60)

    # Create dummy optimizer
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # Test parameters
    T_max = 100
    eta_min = 1e-5

    # Create schedulers
    quadratic_scheduler = QuadraticInOutLR(optimizer, T_max=T_max, eta_min=eta_min)

    # Reset optimizer
    for group in optimizer.param_groups:
        group['lr'] = 0.1
    cubic_scheduler = CubicInOutLR(optimizer, T_max=T_max, eta_min=eta_min)

    # Reset optimizer
    for group in optimizer.param_groups:
        group['lr'] = 0.1
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

    print("\n✓ All schedulers created successfully")

    # Test QuadraticInOutLR
    print("\nTesting QuadraticInOutLR...")
    for group in optimizer.param_groups:
        group['lr'] = 0.1
    quadratic_lrs = []
    for epoch in range(T_max + 1):
        if epoch > 0:
            quadratic_scheduler.step()
        quadratic_lrs.append(optimizer.param_groups[0]['lr'])

    print(f"  Initial LR: {quadratic_lrs[0]:.6f}")
    print(f"  LR at epoch {T_max//2}: {quadratic_lrs[T_max//2]:.6f}")
    print(f"  Final LR: {quadratic_lrs[-1]:.6f}")

    # Test CubicInOutLR
    print("\nTesting CubicInOutLR...")
    for group in optimizer.param_groups:
        group['lr'] = 0.1
    cubic_lrs = []
    for epoch in range(T_max + 1):
        if epoch > 0:
            cubic_scheduler.step()
        cubic_lrs.append(optimizer.param_groups[0]['lr'])

    print(f"  Initial LR: {cubic_lrs[0]:.6f}")
    print(f"  LR at epoch {T_max//2}: {cubic_lrs[T_max//2]:.6f}")
    print(f"  Final LR: {cubic_lrs[-1]:.6f}")

    # Test CosineAnnealingLR
    print("\nTesting CosineAnnealingLR...")
    for group in optimizer.param_groups:
        group['lr'] = 0.1
    cosine_lrs = []
    for epoch in range(T_max + 1):
        if epoch > 0:
            cosine_scheduler.step()
        cosine_lrs.append(optimizer.param_groups[0]['lr'])

    print(f"  Initial LR: {cosine_lrs[0]:.6f}")
    print(f"  LR at epoch {T_max//2}: {cosine_lrs[T_max//2]:.6f}")
    print(f"  Final LR: {cosine_lrs[-1]:.6f}")

    # Validate schedulers reach eta_min
    assert abs(quadratic_lrs[-1] - eta_min) < 1e-6, "QuadraticInOutLR should reach eta_min"
    assert abs(cubic_lrs[-1] - eta_min) < 1e-6, "CubicInOutLR should reach eta_min"
    assert abs(cosine_lrs[-1] - eta_min) < 1e-6, "CosineAnnealingLR should reach eta_min"

    print("\n✓ All schedulers correctly reach eta_min")

    # Validate schedulers start at base_lr
    assert abs(quadratic_lrs[0] - 0.1) < 1e-6, "QuadraticInOutLR should start at base_lr"
    assert abs(cubic_lrs[0] - 0.1) < 1e-6, "CubicInOutLR should start at base_lr"
    assert abs(cosine_lrs[0] - 0.1) < 1e-6, "CosineAnnealingLR should start at base_lr"

    print("✓ All schedulers correctly start at base_lr")

    return quadratic_lrs, cubic_lrs, cosine_lrs


def test_state_dict():
    """Test state_dict save/load functionality."""
    print("\n" + "=" * 60)
    print("TESTING STATE DICT SAVE/LOAD")
    print("=" * 60)

    # Create dummy optimizer
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # Test QuadraticInOutLR
    print("\nTesting QuadraticInOutLR state dict...")
    scheduler1 = QuadraticInOutLR(optimizer, T_max=100, eta_min=1e-5)

    # Step a few times
    for _ in range(10):
        scheduler1.step()

    lr_before = optimizer.param_groups[0]['lr']

    # Save and load state
    state = scheduler1.state_dict()

    # Create new scheduler and load state
    for group in optimizer.param_groups:
        group['lr'] = 0.1  # Reset
    scheduler2 = QuadraticInOutLR(optimizer, T_max=100, eta_min=1e-5)
    scheduler2.load_state_dict(state)

    lr_after = optimizer.param_groups[0]['lr']

    print(f"  LR before save: {lr_before:.6f}")
    print(f"  LR after load: {lr_after:.6f}")
    assert abs(lr_before - lr_after) < 1e-9, "QuadraticInOutLR state dict mismatch"
    print("✓ QuadraticInOutLR state dict works correctly")

    # Test CubicInOutLR
    print("\nTesting CubicInOutLR state dict...")
    for group in optimizer.param_groups:
        group['lr'] = 0.1  # Reset
    scheduler1 = CubicInOutLR(optimizer, T_max=100, eta_min=1e-5)

    # Step a few times
    for _ in range(10):
        scheduler1.step()

    lr_before = optimizer.param_groups[0]['lr']

    # Save and load state
    state = scheduler1.state_dict()

    # Create new scheduler and load state
    for group in optimizer.param_groups:
        group['lr'] = 0.1  # Reset
    scheduler2 = CubicInOutLR(optimizer, T_max=100, eta_min=1e-5)
    scheduler2.load_state_dict(state)

    lr_after = optimizer.param_groups[0]['lr']

    print(f"  LR before save: {lr_before:.6f}")
    print(f"  LR after load: {lr_after:.6f}")
    assert abs(lr_before - lr_after) < 1e-9, "CubicInOutLR state dict mismatch"
    print("✓ CubicInOutLR state dict works correctly")


def visualize_schedulers(quadratic_lrs, cubic_lrs, cosine_lrs):
    """Create visualization comparing all three schedulers."""
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATION")
    print("=" * 60)

    epochs = list(range(len(quadratic_lrs)))

    plt.figure(figsize=(14, 10))

    # Plot 1: All schedulers together
    plt.subplot(2, 2, 1)
    plt.plot(epochs, quadratic_lrs, label='Quadratic InOut', linewidth=2, color='blue')
    plt.plot(epochs, cubic_lrs, label='Cubic InOut', linewidth=2, color='green')
    plt.plot(epochs, cosine_lrs, label='Cosine Annealing', linewidth=2, color='red', linestyle='--')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('Learning Rate Schedulers Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    # Plot 2: Quadratic vs Cosine (zoomed)
    plt.subplot(2, 2, 2)
    plt.plot(epochs, quadratic_lrs, label='Quadratic InOut', linewidth=2, color='blue')
    plt.plot(epochs, cosine_lrs, label='Cosine Annealing', linewidth=2, color='red', linestyle='--')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('Quadratic InOut vs Cosine', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # Plot 3: Cubic vs Cosine (zoomed)
    plt.subplot(2, 2, 3)
    plt.plot(epochs, cubic_lrs, label='Cubic InOut', linewidth=2, color='green')
    plt.plot(epochs, cosine_lrs, label='Cosine Annealing', linewidth=2, color='red', linestyle='--')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('Cubic InOut vs Cosine', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # Plot 4: Rate of change (derivative approximation)
    plt.subplot(2, 2, 4)
    quadratic_rate = [abs(quadratic_lrs[i+1] - quadratic_lrs[i]) for i in range(len(quadratic_lrs)-1)]
    cubic_rate = [abs(cubic_lrs[i+1] - cubic_lrs[i]) for i in range(len(cubic_lrs)-1)]
    cosine_rate = [abs(cosine_lrs[i+1] - cosine_lrs[i]) for i in range(len(cosine_lrs)-1)]

    plt.plot(epochs[:-1], quadratic_rate, label='Quadratic InOut', linewidth=2, color='blue')
    plt.plot(epochs[:-1], cubic_rate, label='Cubic InOut', linewidth=2, color='green')
    plt.plot(epochs[:-1], cosine_rate, label='Cosine Annealing', linewidth=2, color='red', linestyle='--')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('|ΔLR|', fontsize=12)
    plt.title('Rate of LR Change', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    plt.tight_layout()

    # Save figure
    output_path = os.path.join(project_root, 'tests', 'easing_schedulers_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")
    plt.close()


def print_scheduler_comparison():
    """Print a detailed comparison of scheduler characteristics."""
    print("\n" + "=" * 60)
    print("SCHEDULER CHARACTERISTICS COMPARISON")
    print("=" * 60)

    print("\n📊 CosineAnnealingLR (Baseline):")
    print("  • Smooth, symmetric cosine curve")
    print("  • Moderate rate of change throughout")
    print("  • Well-studied and widely used")
    print("  • Good for most training scenarios")

    print("\n📊 QuadraticInOutLR:")
    print("  • Quadratic (t²) easing function")
    print("  • Gentle start (slow acceleration)")
    print("  • More aggressive in middle than cosine")
    print("  • Gentle end (slow deceleration)")
    print("  • Best for: When you want more aggressive LR decay in the middle")

    print("\n📊 CubicInOutLR:")
    print("  • Cubic (t³) easing function")
    print("  • Very gentle start (very slow acceleration)")
    print("  • Much more aggressive in middle than cosine")
    print("  • Very gentle end (very slow deceleration)")
    print("  • Best for: When you want very smooth start/end with rapid middle decay")

    print("\n💡 When to use each:")
    print("  • Cosine: Default choice, balanced behavior")
    print("  • Quadratic: Need faster LR decay in mid-training")
    print("  • Cubic: Need very stable start/end, rapid middle transition")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("EASING SCHEDULERS TEST SUITE")
    print("=" * 60)

    # Run tests
    quadratic_lrs, cubic_lrs, cosine_lrs = test_scheduler_behavior()
    test_state_dict()

    # Print comparison
    print_scheduler_comparison()

    # Create visualization
    visualize_schedulers(quadratic_lrs, cubic_lrs, cosine_lrs)

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
