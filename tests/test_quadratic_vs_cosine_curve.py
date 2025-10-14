"""Compare QuadraticInOutLR curve with CosineAnnealingLR curve."""

import sys
sys.path.insert(0, '/Users/gclinger/Documents/projects/Multi-Stream-Neural-Networks')

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.training.schedulers import QuadraticInOutLR
import matplotlib.pyplot as plt


def generate_lr_curves():
    """Generate and plot LR curves for comparison."""
    # Setup
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

    for epoch in range(T_max + 10):  # Go beyond T_max to test behavior
        quad_lrs.append(optimizer1.param_groups[0]['lr'])
        cosine_lrs.append(optimizer2.param_groups[0]['lr'])
        quad_scheduler.step()
        cosine_scheduler.step()

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(quad_lrs, label='QuadraticInOutLR', linewidth=2)
    plt.plot(cosine_lrs, label='CosineAnnealingLR', linewidth=2, linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title(f'Learning Rate Comparison (T_max={T_max}, eta_min={eta_min})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axvline(x=T_max, color='r', linestyle=':', label=f'T_max={T_max}')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('tests/quadratic_vs_cosine_comparison.png', dpi=150)
    print(f"Plot saved to tests/quadratic_vs_cosine_comparison.png")

    # Print some values for comparison
    print("\n=== LR values at key epochs ===")
    print(f"{'Epoch':<8} {'QuadraticInOut':<15} {'CosineAnnealing':<15} {'Difference':<15}")
    print("-" * 60)
    for epoch in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 49, 50, 55]:
        if epoch < len(quad_lrs):
            diff = abs(quad_lrs[epoch] - cosine_lrs[epoch])
            print(f"{epoch:<8} {quad_lrs[epoch]:<15.8f} {cosine_lrs[epoch]:<15.8f} {diff:<15.8f}")

    # Check if they're similar
    print("\n=== Visual similarity check ===")
    differences = [abs(q - c) for q, c in zip(quad_lrs[:T_max], cosine_lrs[:T_max])]
    avg_diff = sum(differences) / len(differences)
    max_diff = max(differences)
    print(f"Average difference: {avg_diff:.8f}")
    print(f"Maximum difference: {max_diff:.8f}")
    print(f"Max difference at epoch: {differences.index(max_diff)}")


if __name__ == "__main__":
    generate_lr_curves()
