"""
Visual explanation of Method 4: Normalized Local Orthogonal Stream

This script creates a simple diagram showing what we're computing.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

print("="*80)
print("SIMPLIFIED 3D VISUALIZATION (using RGB only, ignoring D for visualization)")
print("="*80)
print("\nImagine we have a 5×5 pixel window around a center pixel.")
print("Each pixel has [R, G, B, D] values (4D point).")
print("We fit a 3D hyperplane through these 25 points in 4D space.")
print("The orthogonal direction tells us how much the center pixel")
print("deviates from this local hyperplane.\n")

# Create synthetic data
np.random.seed(42)

# SCENARIO 1: Smooth surface (all pixels similar)
print("\n" + "="*80)
print("SCENARIO 1: Smooth Surface (Low Orthogonal Value)")
print("="*80)

# All points lie roughly on a plane in RGB space
plane_normal = np.array([0, 0, 1])
points_smooth = []
for i in range(25):
    # Points lie on a tilted plane with small noise
    x = np.random.uniform(0.4, 0.6)
    y = np.random.uniform(0.4, 0.6)
    z = 0.5 + 0.1*x + 0.1*y + np.random.normal(0, 0.01)  # Small noise
    points_smooth.append([x, y, z])

points_smooth = np.array(points_smooth)
center_smooth = np.array([0.5, 0.5, 0.55])  # Center pixel fits well

# Standardize
mean_smooth = points_smooth.mean(axis=0)
std_smooth = points_smooth.std(axis=0) + 1e-8
points_smooth_norm = (points_smooth - mean_smooth) / std_smooth
center_smooth_norm = (center_smooth - mean_smooth) / std_smooth

# SVD
U, S, Vt = np.linalg.svd(points_smooth_norm, full_matrices=False)
orth_vec_smooth = Vt[2, :]  # 3rd component for 3D
orth_value_smooth = np.dot(center_smooth_norm, orth_vec_smooth)

print(f"Singular values: {S}")
print(f"Orthogonal vector: {orth_vec_smooth}")
print(f"Orthogonal value: {orth_value_smooth:.4f}")
print(f"→ Small orthogonal value = pixel fits the local surface well")


# SCENARIO 2: Edge (center pixel different from neighbors)
print("\n" + "="*80)
print("SCENARIO 2: Edge/Boundary (High Orthogonal Value)")
print("="*80)

# Half points from surface A, half from surface B
points_edge = []
for i in range(13):
    # Surface A (dark wall)
    x = np.random.uniform(0.2, 0.4)
    y = np.random.uniform(0.2, 0.4)
    z = 0.3 + np.random.normal(0, 0.01)
    points_edge.append([x, y, z])

for i in range(12):
    # Surface B (bright table)
    x = np.random.uniform(0.6, 0.8)
    y = np.random.uniform(0.6, 0.8)
    z = 0.7 + np.random.normal(0, 0.01)
    points_edge.append([x, y, z])

points_edge = np.array(points_edge)
center_edge = np.array([0.5, 0.5, 0.9])  # Center pixel is outlier!

# Standardize
mean_edge = points_edge.mean(axis=0)
std_edge = points_edge.std(axis=0) + 1e-8
points_edge_norm = (points_edge - mean_edge) / std_edge
center_edge_norm = (center_edge - mean_edge) / std_edge

# SVD
U, S, Vt = np.linalg.svd(points_edge_norm, full_matrices=False)
orth_vec_edge = Vt[2, :]
orth_value_edge = np.dot(center_edge_norm, orth_vec_edge)

print(f"Singular values: {S}")
print(f"Orthogonal vector: {orth_vec_edge}")
print(f"Orthogonal value: {orth_value_edge:.4f}")
print(f"→ Large orthogonal value = pixel deviates from local surface")


# Create visualization
fig = plt.figure(figsize=(16, 7))

# Plot 1: Smooth surface
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(points_smooth[:, 0], points_smooth[:, 1], points_smooth[:, 2],
           c='blue', marker='o', s=50, alpha=0.6, label='Neighborhood pixels')
ax1.scatter(center_smooth[0], center_smooth[1], center_smooth[2],
           c='red', marker='*', s=300, label='Center pixel')

# Draw plane (approximation)
xx, yy = np.meshgrid(np.linspace(0.3, 0.7, 10), np.linspace(0.3, 0.7, 10))
zz = mean_smooth[2] + 0.1*(xx - mean_smooth[0]) + 0.1*(yy - mean_smooth[1])
ax1.plot_surface(xx, yy, zz, alpha=0.2, color='cyan')

ax1.set_xlabel('R (Red)')
ax1.set_ylabel('G (Green)')
ax1.set_zlabel('B (Blue)')
ax1.set_title(f'Smooth Surface\nOrthogonal Value = {orth_value_smooth:.4f}\n(LOW = fits well)')
ax1.legend()
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_zlim(0, 1)

# Plot 2: Edge
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(points_edge[:13, 0], points_edge[:13, 1], points_edge[:13, 2],
           c='darkblue', marker='o', s=50, alpha=0.6, label='Surface A')
ax2.scatter(points_edge[13:, 0], points_edge[13:, 1], points_edge[13:, 2],
           c='orange', marker='o', s=50, alpha=0.6, label='Surface B')
ax2.scatter(center_edge[0], center_edge[1], center_edge[2],
           c='red', marker='*', s=300, label='Center pixel (outlier)')

# Best-fit plane tries to fit both surfaces
xx, yy = np.meshgrid(np.linspace(0.1, 0.9, 10), np.linspace(0.1, 0.9, 10))
zz = mean_edge[2] * np.ones_like(xx)
ax2.plot_surface(xx, yy, zz, alpha=0.2, color='cyan')

ax2.set_xlabel('R (Red)')
ax2.set_ylabel('G (Green)')
ax2.set_zlabel('B (Blue)')
ax2.set_title(f'Edge/Boundary\nOrthogonal Value = {orth_value_edge:.4f}\n(HIGH = outlier)')
ax2.legend()
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_zlim(0, 1)

plt.tight_layout()
plt.savefig('tests/method4_concept_visualization.png', dpi=100, bbox_inches='tight')
print("\n✓ Saved visualization: tests/method4_concept_visualization.png")
plt.close()

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("""
For each pixel:
1. Extract 5×5 neighborhood (25 pixels)
2. Each pixel is a point in 4D RGBD space
3. Fit a 3D hyperplane through these 25 points (using SVD)
4. Find the orthogonal direction (4th principal component)
5. Project the center pixel onto this orthogonal direction
6. Result = how much center pixel deviates from local RGBD hyperplane

High orthogonal value → Edge, boundary, structure change
Low orthogonal value  → Smooth surface, consistent RGBD relationship

This creates a new channel O that captures local RGB-D structure breaks!
""")
print("="*80)
