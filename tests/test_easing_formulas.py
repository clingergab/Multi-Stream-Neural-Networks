"""Test and visualize easing formulas to verify correctness."""

import numpy as np
import matplotlib.pyplot as plt


def quadratic_in_out_easing(t):
    """
    Standard Robert Penner's easeInOutQuad.
    Output goes from 0 to 1 as t goes from 0 to 1.
    """
    if t < 0.5:
        return 2 * t * t
    else:
        return -2 * (t - 1) * (t - 1) + 1


def cosine_easing(t):
    """
    Cosine easing from 0 to 1.
    """
    return (1 - np.cos(t * np.pi)) / 2


def quadratic_in_out_decay(t):
    """
    Our current implementation: inverted for decay (1 → 0).
    """
    if t < 0.5:
        easing = 2 * t * t
        return 1 - easing
    else:
        easing = -2 * (t - 1) * (t - 1) + 1
        return 1 - easing


def cosine_decay(t):
    """
    Cosine annealing decay (1 → 0).
    """
    return (1 + np.cos(t * np.pi)) / 2


# Generate values
t_values = np.linspace(0, 1, 100)

# Standard easing (0 → 1)
quad_easing = np.array([quadratic_in_out_easing(t) for t in t_values])
cos_easing = np.array([cosine_easing(t) for t in t_values])

# Decay versions (1 → 0)
quad_decay = np.array([quadratic_in_out_decay(t) for t in t_values])
cos_decay = np.array([cosine_decay(t) for t in t_values])

# Create plots
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Plot 1: Standard easing (0 → 1)
axes[0, 0].plot(t_values, quad_easing, label='Quadratic InOut', linewidth=2)
axes[0, 0].plot(t_values, cos_easing, label='Cosine', linewidth=2, linestyle='--', alpha=0.7)
axes[0, 0].set_xlabel('t')
axes[0, 0].set_ylabel('Easing Value')
axes[0, 0].set_title('Standard Easing (0 → 1)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Decay versions (1 → 0)
axes[0, 1].plot(t_values, quad_decay, label='Quadratic InOut Decay', linewidth=2)
axes[0, 1].plot(t_values, cos_decay, label='Cosine Decay', linewidth=2, linestyle='--', alpha=0.7)
axes[0, 1].set_xlabel('t')
axes[0, 1].set_ylabel('Decay Factor')
axes[0, 1].set_title('Decay Versions (1 → 0)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Difference
diff_easing = quad_easing - cos_easing
diff_decay = quad_decay - cos_decay
axes[0, 2].plot(t_values, diff_easing, label='Easing Difference', linewidth=2)
axes[0, 2].plot(t_values, diff_decay, label='Decay Difference', linewidth=2, linestyle='--')
axes[0, 2].set_xlabel('t')
axes[0, 2].set_ylabel('Difference')
axes[0, 2].set_title('Quadratic - Cosine Difference')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)
axes[0, 2].axhline(y=0, color='k', linestyle=':', alpha=0.3)

# Plot 4: First derivative (easing)
quad_easing_deriv = np.gradient(quad_easing, t_values)
cos_easing_deriv = np.gradient(cos_easing, t_values)
axes[1, 0].plot(t_values, quad_easing_deriv, label='Quadratic InOut', linewidth=2)
axes[1, 0].plot(t_values, cos_easing_deriv, label='Cosine', linewidth=2, linestyle='--', alpha=0.7)
axes[1, 0].set_xlabel('t')
axes[1, 0].set_ylabel('Rate of Change')
axes[1, 0].set_title('First Derivative (Easing)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 5: First derivative (decay)
quad_decay_deriv = np.gradient(quad_decay, t_values)
cos_decay_deriv = np.gradient(cos_decay, t_values)
axes[1, 1].plot(t_values, quad_decay_deriv, label='Quadratic InOut Decay', linewidth=2)
axes[1, 1].plot(t_values, cos_decay_deriv, label='Cosine Decay', linewidth=2, linestyle='--', alpha=0.7)
axes[1, 1].set_xlabel('t')
axes[1, 1].set_ylabel('Rate of Change')
axes[1, 1].set_title('First Derivative (Decay)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Plot 6: Second derivative (shows acceleration/deceleration)
quad_easing_deriv2 = np.gradient(quad_easing_deriv, t_values)
cos_easing_deriv2 = np.gradient(cos_easing_deriv, t_values)
axes[1, 2].plot(t_values, quad_easing_deriv2, label='Quadratic InOut', linewidth=2)
axes[1, 2].plot(t_values, cos_easing_deriv2, label='Cosine', linewidth=2, linestyle='--', alpha=0.7)
axes[1, 2].set_xlabel('t')
axes[1, 2].set_ylabel('Acceleration')
axes[1, 2].set_title('Second Derivative (Acceleration)')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)
axes[1, 2].axhline(y=0, color='k', linestyle=':', alpha=0.3)

plt.tight_layout()
plt.savefig('tests/easing_formula_analysis.png', dpi=150)
print("Easing formula analysis saved to tests/easing_formula_analysis.png")

# Print statistics
print("\n=== Easing Value Comparison (key points) ===")
print(f"{'t':<6} {'Quad':<12} {'Cosine':<12} {'Difference':<12}")
print("-" * 45)
for t in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
    quad_val = quadratic_in_out_easing(t)
    cos_val = cosine_easing(t)
    print(f"{t:<6.2f} {quad_val:<12.6f} {cos_val:<12.6f} {quad_val - cos_val:<12.6f}")

print("\n=== Maximum Difference ===")
max_diff_idx = np.argmax(np.abs(diff_easing))
print(f"At t = {t_values[max_diff_idx]:.4f}")
print(f"Quadratic: {quad_easing[max_diff_idx]:.6f}")
print(f"Cosine: {cos_easing[max_diff_idx]:.6f}")
print(f"Difference: {diff_easing[max_diff_idx]:.6f}")
