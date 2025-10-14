"""Compare QuadraticInOutLR curve with CosineAnnealingLR on LINEAR scale."""

import sys
sys.path.insert(0, '/Users/gclinger/Documents/projects/Multi-Stream-Neural-Networks')

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.training.schedulers import QuadraticInOutLR
import matplotlib.pyplot as plt
import numpy as np


def generate_linear_comparison():
    """Generate and plot LR curves on LINEAR scale for better visual comparison."""
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

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Full curve (linear scale)
    axes[0, 0].plot(quad_lrs, label='QuadraticInOutLR', linewidth=2)
    axes[0, 0].plot(cosine_lrs, label='CosineAnnealingLR', linewidth=2, linestyle='--', alpha=0.8)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Learning Rate')
    axes[0, 0].set_title('Full Curve (Linear Scale)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Difference between curves
    differences = [abs(q - c) for q, c in zip(quad_lrs, cosine_lrs)]
    axes[0, 1].plot(differences, color='red', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Absolute Difference')
    axes[0, 1].set_title('Absolute Difference Between Curves')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: First derivative (rate of change) - shows S-curve shape
    quad_diff = np.diff(quad_lrs)
    cosine_diff = np.diff(cosine_lrs)
    epochs_diff = range(len(quad_diff))
    axes[1, 0].plot(epochs_diff, quad_diff, label='QuadraticInOut (derivative)', linewidth=2)
    axes[1, 0].plot(epochs_diff, cosine_diff, label='CosineAnnealing (derivative)', linewidth=2, linestyle='--', alpha=0.8)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Rate of Change (dLR/dEpoch)')
    axes[1, 0].set_title('First Derivative (Rate of LR Decay)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Normalized curves (both start at 1, end at 0)
    quad_normalized = [(lr - eta_min) / (base_lr - eta_min) for lr in quad_lrs]
    cosine_normalized = [(lr - eta_min) / (base_lr - eta_min) for lr in cosine_lrs]
    axes[1, 1].plot(quad_normalized, label='QuadraticInOut (normalized)', linewidth=2)
    axes[1, 1].plot(cosine_normalized, label='CosineAnnealing (normalized)', linewidth=2, linestyle='--', alpha=0.8)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Normalized LR (1 → 0)')
    axes[1, 1].set_title('Normalized Curves (Shape Comparison)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('tests/linear_scale_comparison.png', dpi=150)
    print(f"Linear scale comparison saved to tests/linear_scale_comparison.png")

    # Print statistics
    print("\n=== Curve Statistics ===")
    print(f"Average absolute difference: {np.mean(differences):.8f}")
    print(f"Maximum absolute difference: {np.max(differences):.8f} at epoch {np.argmax(differences)}")
    print(f"Minimum absolute difference: {np.min(differences):.8f} at epoch {np.argmin(differences)}")

    # Check derivative patterns (S-curve should have different derivative pattern)
    print("\n=== Derivative Analysis ===")
    quad_max_change = np.argmin(quad_diff)
    cosine_max_change = np.argmin(cosine_diff)
    print(f"QuadraticInOut: Maximum decay rate at epoch {quad_max_change}")
    print(f"CosineAnnealing: Maximum decay rate at epoch {cosine_max_change}")


if __name__ == "__main__":
    generate_linear_comparison()
