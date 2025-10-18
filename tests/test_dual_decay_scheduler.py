"""
Test the enhanced DecayingCosineAnnealingLR with dual decay (max and min).
Visualize different decay strategies.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import matplotlib.pyplot as plt
from src.training.schedulers import DecayingCosineAnnealingLR


def test_dual_decay_strategies():
    """Test and visualize different dual decay strategies."""

    # Create dummy optimizer
    model = torch.nn.Linear(10, 10)

    strategies = [
        {
            'name': 'Proposed: Narrowing (0.5, 0.8)',
            'params': {'max_factor': 0.5, 'min_factor': 0.8},
            'color': 'blue'
        },
        {
            'name': 'Alternative: Constant (0.65, 0.65)',
            'params': {'max_factor': 0.65, 'min_factor': 0.65},
            'color': 'green'
        },
        {
            'name': 'Conservative: (0.7, 0.8)',
            'params': {'max_factor': 0.7, 'min_factor': 0.8},
            'color': 'orange'
        }
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, strategy in enumerate(strategies):
        optimizer = torch.optim.SGD(model.parameters(), lr=5e-5)  # Match your actual base LR

        scheduler = DecayingCosineAnnealingLR(
            optimizer,
            T_max=20,
            eta_min=8e-6,  # Match your proposed eta_min
            **strategy['params']
        )

        lrs = []
        epochs = 100

        for epoch in range(epochs):
            lrs.append(optimizer.param_groups[0]['lr'])

            # Debug: print eta_min at key epochs
            if epoch in [0, 20, 40, 60, 80] and idx == 0:  # Only for first strategy
                print(f"Epoch {epoch}: eta_min={scheduler.eta_min:.6e}, LR={optimizer.param_groups[0]['lr']:.6e}")

            optimizer.step()
            scheduler.step()

        # Plot
        ax = axes[idx]
        ax.plot(lrs, linewidth=2, color=strategy['color'], alpha=0.8, marker='o', markersize=3, markevery=10)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title(strategy['name'], fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3)
        # Use linear scale with auto range for actual LR values
        ax.set_ylim(0, 6e-5)
        # Format y-axis with scientific notation
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

        # Add vertical lines at cycle boundaries
        for cycle in range(1, 6):
            ax.axvline(x=cycle*20, color='red', linestyle='--', alpha=0.3, linewidth=1)

        # Annotate decay values
        decay_text = f"Max factor: {strategy['params']['max_factor']}\n"
        min_factor = strategy['params']['min_factor']
        if min_factor is None:
            min_factor = strategy['params']['max_factor']
        decay_text += f"Min factor: {min_factor}"
        ax.text(0.02, 0.98, decay_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('DecayingCosineAnnealingLR with Dual Decay Strategies',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save and show
    save_path = 'tests/test_dual_decay_scheduler.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved to: {save_path}")

    # Display the plot
    plt.show()

    # Print numerical comparison
    print("\n" + "="*80)
    print("NUMERICAL COMPARISON AT KEY EPOCHS")
    print("="*80)

    for strategy in strategies:
        optimizer = torch.optim.SGD(model.parameters(), lr=5e-5)
        scheduler = DecayingCosineAnnealingLR(
            optimizer, T_max=20, eta_min=8e-6, **strategy['params']
        )

        print(f"\n{strategy['name']}:")
        print(f"  Epoch 0:   {optimizer.param_groups[0]['lr']:.6e}")

        for epoch in range(100):
            optimizer.step()
            scheduler.step()
            if epoch + 1 in [20, 40, 60, 80]:
                print(f"  Epoch {epoch+1:2d}:  {optimizer.param_groups[0]['lr']:.6e}")


if __name__ == '__main__':
    test_dual_decay_strategies()
