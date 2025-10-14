"""Show what the notebook LR plot will look like with linear scale."""

import sys
sys.path.insert(0, '/Users/gclinger/Documents/projects/Multi-Stream-Neural-Networks')

import torch
import torch.nn as nn
from src.training.schedulers import QuadraticInOutLR
import matplotlib.pyplot as plt


def show_notebook_lr_plot():
    """Demonstrate what the notebook LR plot will look like now."""
    T_max = 50
    eta_min = 1e-6
    base_lr = 0.1

    # Create scheduler
    model = nn.Linear(10, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr)
    scheduler = QuadraticInOutLR(optimizer, T_max=T_max, eta_min=eta_min)

    # Collect LR values
    lrs = []
    for epoch in range(T_max + 1):
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()

    # Create the plot as it will appear in the notebook
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    ax.plot(lrs, linewidth=2, color='green', label='QuadraticInOutLR')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule (Linear Scale)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    # Linear scale (no set_yscale('log'))

    plt.tight_layout()
    plt.savefig('tests/notebook_lr_plot_linear_scale.png', dpi=150)
    print("Saved notebook LR plot preview to: tests/notebook_lr_plot_linear_scale.png")

    # Print what the user will see
    print("\n✅ Notebook Updated!")
    print("\nWhat changed:")
    print("  BEFORE: axes[2].set_yscale('log')  → Log scale (curves look identical)")
    print("  AFTER:  # axes[2].set_yscale('log')  → Linear scale (shows S-curve shape)")

    print("\nNow your learning rate plot will:")
    print("  ✓ Show the actual quadratic S-curve decay shape")
    print("  ✓ Make differences between schedulers visible")
    print("  ✓ Display absolute LR values clearly")

    print("\nThe plot will look like the saved image above!")
    print("The S-curve will be subtle but visible - gentle start, faster middle, gentle end.")


if __name__ == "__main__":
    show_notebook_lr_plot()
