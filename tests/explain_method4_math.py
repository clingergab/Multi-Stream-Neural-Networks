"""
Detailed explanation of Method 4: Normalized Local Orthogonal Stream

MATHEMATICAL BREAKDOWN:
=======================

For each pixel at position (i, j):

STEP 1: Extract local neighborhood
-----------------------------------
- Window size: 5×5 = 25 pixels around (i, j)
- For each of the 25 pixels, get [R, G, B, D] values
- Result: X matrix of shape (25, 4)

Example for one pixel:
    X = [[r1, g1, b1, d1],    <- neighbor pixel 1
         [r2, g2, b2, d2],    <- neighbor pixel 2
         ...
         [r25, g25, b25, d25]] <- neighbor pixel 25


STEP 2: Standardize each RGBD dimension
----------------------------------------
For each column (dimension) of X:
    X_mean[dim] = mean of all 25 values for that dimension
    X_std[dim] = std of all 25 values for that dimension

    X_normalized[:, dim] = (X[:, dim] - X_mean[dim]) / X_std[dim]

Why? This puts all 4 dimensions on equal footing:
- RGB might vary 0.1-0.9 (range = 0.8)
- Depth might vary 0.3-0.4 (range = 0.1)
- After normalization, both have mean=0, std=1

Example:
    Before normalization:
    X = [[0.5, 0.6, 0.7, 0.3],
         [0.5, 0.6, 0.7, 0.3],
         ...]

    X_mean = [0.5, 0.6, 0.7, 0.3]
    X_std = [0.1, 0.1, 0.1, 0.05]

    After normalization:
    X_normalized = [[0, 0, 0, 0],
                    [0, 0, 0, 0],
                    ...]
    (All relative to local neighborhood)


STEP 3: SVD to find 4D hyperplane
----------------------------------
SVD decomposes X_normalized into:
    X_normalized = U @ S @ Vt

Where:
    U: (25, 4) - Left singular vectors
    S: (4,) - Singular values [s1, s2, s3, s4] (sorted descending)
    Vt: (4, 4) - Right singular vectors (rows are directions in RGBD space)

The 4 rows of Vt represent orthogonal directions in RGBD space:
    Vt[0] = 1st principal component (most variance)
    Vt[1] = 2nd principal component
    Vt[2] = 3rd principal component
    Vt[3] = 4th principal component (LEAST variance = orthogonal to hyperplane)

The hyperplane is spanned by the first 3 vectors: Vt[0], Vt[1], Vt[2]
The orthogonal direction is: Vt[3]


STEP 4: Project center pixel onto orthogonal direction
-------------------------------------------------------
Get the center pixel's RGBD values:
    center_rgbd = [r_center, g_center, b_center, d_center]

Normalize using the same mean/std from the neighborhood:
    center_normalized = (center_rgbd - X_mean) / X_std

Project onto the orthogonal vector:
    orthogonal_value = dot(center_normalized, Vt[3])

This is a SCALAR value representing:
    "How much does the center pixel deviate from the local RGBD hyperplane?"


STEP 5: Store result
--------------------
    O[i, j] = orthogonal_value


WHAT DOES THIS CAPTURE?
========================

High absolute orthogonal values mean:
    - The center pixel's RGBD relationship is DIFFERENT from its neighbors
    - Could be an edge, material boundary, depth discontinuity, etc.

Low orthogonal values mean:
    - The center pixel fits well on the local RGBD hyperplane
    - Smooth surface with consistent RGB-D relationship


EXAMPLE SCENARIOS:
==================

1. Smooth wall (low orthogonal value):
   - All 25 pixels have similar R, G, B, D
   - Hyperplane fits well
   - Center pixel projects near zero onto orthogonal direction

2. Edge between table and floor (high orthogonal value):
   - Left half of window: brown table at depth=0.5
   - Right half of window: gray floor at depth=0.8
   - Center pixel is ON the edge
   - Hyperplane tries to fit BOTH surfaces
   - Center pixel deviates significantly → high orthogonal value

3. Shadow boundary (medium orthogonal value):
   - RGB changes but depth stays constant
   - Hyperplane captures this pattern
   - Pixels in shadow have different RGB but same D
   - Orthogonal value captures this RGB-D relationship break


KEY INSIGHT:
============
By standardizing each dimension, we ensure that:
- Small variations in depth are weighted equally with RGB
- The orthogonal direction captures RELATIVE deviations
- Edges and boundaries create high orthogonal values
- Smooth regions create low orthogonal values

This gives us a per-pixel "local RGBD structure deviation" measure!
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import torch
from src.data_utils.sunrgbd_dataset import SUNRGBDDataset
import matplotlib.pyplot as plt

print(__doc__)

# ============================================================================
# CONCRETE EXAMPLE: Demonstrate on one pixel
# ============================================================================
print("\n" + "="*80)
print("CONCRETE EXAMPLE: One pixel from real data")
print("="*80)

train_dataset = SUNRGBDDataset(split='train')
rgb, depth, label = train_dataset[0]

# Denormalize
rgb_denorm = rgb * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
depth_denorm = depth * 0.2197 + 0.5027

window_size = 5
pad = window_size // 2

rgb_padded = torch.nn.functional.pad(rgb_denorm, (pad, pad, pad, pad), mode='reflect')
depth_padded = torch.nn.functional.pad(depth_denorm, (pad, pad, pad, pad), mode='reflect')

rgb_np = rgb_padded.numpy()
depth_np = depth_padded.numpy()

# Pick a pixel in the middle of the image
i, j = 112, 112

print(f"\nAnalyzing pixel at position ({i}, {j})...")

# Extract 5×5 neighborhood
r_patch = rgb_np[0, i:i+window_size, j:j+window_size].flatten()
g_patch = rgb_np[1, i:i+window_size, j:j+window_size].flatten()
b_patch = rgb_np[2, i:i+window_size, j:j+window_size].flatten()
d_patch = depth_np[0, i:i+window_size, j:j+window_size].flatten()

X = np.stack([r_patch, g_patch, b_patch, d_patch], axis=1)

print(f"\nSTEP 1: Extract local neighborhood")
print(f"  X.shape = {X.shape}  (25 pixels × 4 dimensions)")
print(f"\n  First 5 pixels:")
print(f"    {'R':<10} {'G':<10} {'B':<10} {'D':<10}")
for k in range(5):
    print(f"    {X[k,0]:<10.4f} {X[k,1]:<10.4f} {X[k,2]:<10.4f} {X[k,3]:<10.4f}")

# Compute mean and std
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)

print(f"\nSTEP 2: Compute statistics")
print(f"  X_mean = {X_mean}")
print(f"  X_std  = {X_std}")

# Standardize
X_normalized = (X - X_mean) / (X_std + 1e-8)

print(f"\nSTEP 2b: Standardize")
print(f"  X_normalized.shape = {X_normalized.shape}")
print(f"  X_normalized mean = {X_normalized.mean(axis=0)}  (should be ~0)")
print(f"  X_normalized std  = {X_normalized.std(axis=0)}   (should be ~1)")

# SVD
U, S, Vt = np.linalg.svd(X_normalized, full_matrices=False)

print(f"\nSTEP 3: SVD decomposition")
print(f"  U.shape  = {U.shape}   (25 × 4)")
print(f"  S.shape  = {S.shape}   (4 singular values)")
print(f"  Vt.shape = {Vt.shape}  (4 × 4)")
print(f"\n  Singular values: {S}")
print(f"  (Larger = more variance in that direction)")

orth_vector = Vt[3, :]

print(f"\nSTEP 4: Extract orthogonal vector")
print(f"  orth_vector = Vt[3, :] = {orth_vector}")
print(f"  (This is the 4th principal component = orthogonal to hyperplane)")

# Get center pixel
center_r = rgb_denorm[0, i, j].item()
center_g = rgb_denorm[1, i, j].item()
center_b = rgb_denorm[2, i, j].item()
center_d = depth_denorm[0, i, j].item()
center_rgbd = np.array([center_r, center_g, center_b, center_d])

print(f"\nSTEP 5: Get center pixel")
print(f"  center_rgbd = {center_rgbd}")

# Normalize center pixel
center_normalized = (center_rgbd - X_mean) / (X_std + 1e-8)

print(f"\nSTEP 6: Normalize center pixel")
print(f"  center_normalized = {center_normalized}")

# Project
orthogonal_value = np.dot(center_normalized, orth_vector)

print(f"\nSTEP 7: Project onto orthogonal vector")
print(f"  orthogonal_value = dot(center_normalized, orth_vector)")
print(f"  orthogonal_value = {orthogonal_value:.6f}")

print(f"\n{'='*80}")
print(f"RESULT: O[{i}, {j}] = {orthogonal_value:.6f}")
print(f"{'='*80}")

if abs(orthogonal_value) > 0.1:
    print(f"\n✅ HIGH orthogonal value → pixel deviates from local RGBD hyperplane")
    print(f"   (Likely an edge, boundary, or structure change)")
elif abs(orthogonal_value) > 0.01:
    print(f"\n⚠️  MEDIUM orthogonal value → some deviation")
else:
    print(f"\n✅ LOW orthogonal value → pixel fits local RGBD hyperplane well")
    print(f"   (Likely a smooth surface)")

print("\n" + "="*80)
