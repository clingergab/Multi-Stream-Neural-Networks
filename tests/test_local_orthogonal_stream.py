"""
Test local per-pixel orthogonal stream extraction.

For each pixel:
1. Extract local neighborhood (e.g., 5×5 window)
2. Fit 3D hyperplane to neighborhood's RGBD points
3. Extract orthogonal vector for that local hyperplane
4. Project the center pixel onto its orthogonal vector
5. Result: [5, H, W] = [R, G, B, D, O]

This captures local RGB-D structure variations, edges, and material boundaries.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import torch
from src.data_utils.sunrgbd_dataset import SUNRGBDDataset
import matplotlib.pyplot as plt
import time

print("="*80)
print("LOCAL PER-PIXEL ORTHOGONAL STREAM EXTRACTION")
print("="*80)
print("\nGoal: Create [R, G, B, D, O] representation where O is computed")
print("      from each pixel's local RGB-D neighborhood hyperplane\n")

# Load dataset
train_dataset = SUNRGBDDataset(train=True)
print(f"Loaded SUN RGB-D train set: {len(train_dataset)} samples\n")


def extract_local_orthogonal_stream(rgb, depth, window_size=5):
    """
    Extract local per-pixel orthogonal stream.

    Args:
        rgb: (3, H, W) RGB tensor (normalized)
        depth: (1, H, W) Depth tensor (normalized)
        window_size: Size of local neighborhood window (odd number)

    Returns:
        orth_stream: (1, H, W) orthogonal projection values (per-pixel)
        rgbdo: (5, H, W) full representation [R, G, B, D, O]
    """
    # Denormalize to [0, 1] range
    rgb_denorm = rgb * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    depth_denorm = depth * 0.2197 + 0.5027

    H, W = rgb.shape[1], rgb.shape[2]
    pad = window_size // 2

    # Initialize orthogonal stream
    orth_stream = np.zeros((H, W), dtype=np.float32)

    # Pad the images for boundary handling
    rgb_padded = torch.nn.functional.pad(rgb_denorm, (pad, pad, pad, pad), mode='reflect')
    depth_padded = torch.nn.functional.pad(depth_denorm, (pad, pad, pad, pad), mode='reflect')

    # Convert to numpy
    rgb_np = rgb_padded.numpy()
    depth_np = depth_padded.numpy()

    # Process each pixel
    for i in range(H):
        for j in range(W):
            # Extract neighborhood (window_size × window_size)
            r_patch = rgb_np[0, i:i+window_size, j:j+window_size].flatten()
            g_patch = rgb_np[1, i:i+window_size, j:j+window_size].flatten()
            b_patch = rgb_np[2, i:i+window_size, j:j+window_size].flatten()
            d_patch = depth_np[0, i:i+window_size, j:j+window_size].flatten()

            # Stack into (window_size^2, 4) matrix
            X = np.stack([r_patch, g_patch, b_patch, d_patch], axis=1)

            # Check if neighborhood has enough variance
            if X.std() < 1e-6:
                orth_stream[i, j] = 0.0
                continue

            # Center the data
            X_mean = X.mean(axis=0)
            X_centered = X - X_mean

            # SVD to find hyperplane
            try:
                U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

                # Orthogonal vector (4th singular vector)
                orth_vector = Vt[3, :]  # Shape: (4,)

                # Get center pixel's RGBD values (denormalized)
                center_r = rgb_denorm[0, i, j].item()
                center_g = rgb_denorm[1, i, j].item()
                center_b = rgb_denorm[2, i, j].item()
                center_d = depth_denorm[0, i, j].item()

                center_rgbd = np.array([center_r, center_g, center_b, center_d])

                # Center using neighborhood mean
                center_rgbd_centered = center_rgbd - X_mean

                # Project center pixel onto orthogonal vector
                orth_stream[i, j] = np.dot(center_rgbd_centered, orth_vector)

            except np.linalg.LinAlgError:
                # SVD failed (degenerate case)
                orth_stream[i, j] = 0.0

    # Convert to tensor
    orth_stream_tensor = torch.from_numpy(orth_stream).float().unsqueeze(0)  # (1, H, W)

    # Create full [R, G, B, D, O] representation
    rgbdo = torch.cat([rgb, depth, orth_stream_tensor], dim=0)  # (5, H, W)

    return orth_stream_tensor, rgbdo


# ============================================================================
# TEST 1: Extract local orthogonal stream for sample images
# ============================================================================
print("="*80)
print("TEST 1: Extract Local Orthogonal Stream")
print("="*80)

print("\nTesting on 1 image first (to check performance)...")

rgb, depth, label = train_dataset[0]

start_time = time.time()
orth_stream, rgbdo = extract_local_orthogonal_stream(rgb, depth, window_size=5)
elapsed = time.time() - start_time

print(f"\n✓ Extracted local orthogonal stream")
print(f"  Time: {elapsed:.2f}s per image")
print(f"  Output shape: {rgbdo.shape} (should be [5, H, W])")
print(f"  Orthogonal stream stats:")
print(f"    Mean: {orth_stream.mean().item():.6f}")
print(f"    Std:  {orth_stream.std().item():.6f}")
print(f"    Min:  {orth_stream.min().item():.6f}")
print(f"    Max:  {orth_stream.max().item():.6f}")

# Estimate time for full dataset
estimated_time = elapsed * len(train_dataset) / 3600
print(f"\n  Estimated time for full dataset: {estimated_time:.2f} hours")
print(f"  (This could be optimized with vectorization or caching)")

# ============================================================================
# TEST 2: Compare local vs global orthogonal streams
# ============================================================================
print("\n" + "="*80)
print("TEST 2: Compare Local vs Global Orthogonal Streams")
print("="*80)

# Global orthogonal stream (from previous test)
def extract_global_orthogonal_stream(rgb, depth):
    """Extract global per-image orthogonal stream."""
    rgb_denorm = rgb * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    depth_denorm = depth * 0.2197 + 0.5027

    H, W = rgb.shape[1], rgb.shape[2]
    rgbd = np.stack([
        rgb_denorm[0].flatten().numpy(),
        rgb_denorm[1].flatten().numpy(),
        rgb_denorm[2].flatten().numpy(),
        depth_denorm[0].flatten().numpy()
    ], axis=1)

    rgbd_centered = rgbd - rgbd.mean(axis=0)
    U, S, Vt = np.linalg.svd(rgbd_centered, full_matrices=False)
    orth_vector = Vt[3, :]

    orth_values = rgbd_centered @ orth_vector
    orth_stream = torch.from_numpy(orth_values.reshape(H, W)).float().unsqueeze(0)

    return orth_stream, orth_vector

print("\nComparing on 3 images...")

fig, axes = plt.subplots(3, 5, figsize=(20, 12))

for idx in range(3):
    rgb, depth, label = train_dataset[idx]

    # Extract both versions
    orth_local, rgbdo = extract_local_orthogonal_stream(rgb, depth, window_size=5)
    orth_global, orth_vector_global = extract_global_orthogonal_stream(rgb, depth)

    # Denormalize for visualization
    rgb_vis = rgb * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    rgb_vis = torch.clamp(rgb_vis, 0, 1).permute(1, 2, 0).numpy()

    depth_vis = depth.squeeze().numpy()
    orth_local_vis = orth_local.squeeze().numpy()
    orth_global_vis = orth_global.squeeze().numpy()

    # Plot RGB
    axes[idx, 0].imshow(rgb_vis)
    axes[idx, 0].set_title(f'Image {idx+1}: RGB')
    axes[idx, 0].axis('off')

    # Plot Depth
    axes[idx, 1].imshow(depth_vis, cmap='viridis')
    axes[idx, 1].set_title('Depth')
    axes[idx, 1].axis('off')

    # Plot Local Orthogonal
    im_local = axes[idx, 2].imshow(orth_local_vis, cmap='RdBu_r')
    axes[idx, 2].set_title(f'Local Orthogonal\nstd={orth_local_vis.std():.3f}')
    axes[idx, 2].axis('off')
    plt.colorbar(im_local, ax=axes[idx, 2], fraction=0.046)

    # Plot Global Orthogonal
    im_global = axes[idx, 3].imshow(orth_global_vis, cmap='RdBu_r')
    axes[idx, 3].set_title(f'Global Orthogonal\nstd={orth_global_vis.std():.3f}')
    axes[idx, 3].axis('off')
    plt.colorbar(im_global, ax=axes[idx, 3], fraction=0.046)

    # Plot difference (Local - Global)
    diff = orth_local_vis - orth_global_vis
    im_diff = axes[idx, 4].imshow(diff, cmap='RdBu_r')
    axes[idx, 4].set_title(f'Difference (L-G)\nstd={diff.std():.3f}')
    axes[idx, 4].axis('off')
    plt.colorbar(im_diff, ax=axes[idx, 4], fraction=0.046)

plt.tight_layout()
plt.savefig('tests/local_vs_global_orthogonal.png', dpi=100, bbox_inches='tight')
print("\n✓ Saved visualization: tests/local_vs_global_orthogonal.png")
plt.close()

# ============================================================================
# TEST 3: Analyze local orthogonal stream properties
# ============================================================================
print("\n" + "="*80)
print("TEST 3: Analyze Local Orthogonal Stream Properties")
print("="*80)

print("\nAnalyzing 10 images...")

local_stats = []
global_stats = []

for i in range(10):
    rgb, depth, label = train_dataset[i]

    orth_local, _ = extract_local_orthogonal_stream(rgb, depth, window_size=5)
    orth_global, _ = extract_global_orthogonal_stream(rgb, depth)

    local_stats.append({
        'std': orth_local.std().item(),
        'range': (orth_local.max() - orth_local.min()).item()
    })

    global_stats.append({
        'std': orth_global.std().item(),
        'range': (orth_global.max() - orth_global.min()).item()
    })

print("\n" + "-"*80)
print("LOCAL ORTHOGONAL STREAM:")
print("-"*80)
print(f"Mean std:   {np.mean([s['std'] for s in local_stats]):.6f}")
print(f"Mean range: {np.mean([s['range'] for s in local_stats]):.6f}")

print("\n" + "-"*80)
print("GLOBAL ORTHOGONAL STREAM:")
print("-"*80)
print(f"Mean std:   {np.mean([s['std'] for s in global_stats]):.6f}")
print(f"Mean range: {np.mean([s['range'] for s in global_stats]):.6f}")

print("\n" + "-"*80)
print("COMPARISON:")
print("-"*80)
std_ratio = np.mean([s['std'] for s in local_stats]) / np.mean([s['std'] for s in global_stats])
range_ratio = np.mean([s['range'] for s in local_stats]) / np.mean([s['range'] for s in global_stats])
print(f"Std ratio (local/global):   {std_ratio:.2f}x")
print(f"Range ratio (local/global): {range_ratio:.2f}x")

if std_ratio > 1.5:
    print("\n✅ Local orthogonal stream has SIGNIFICANTLY more variance")
    print("   (captures richer spatial information)")
elif std_ratio > 1.1:
    print("\n✅ Local orthogonal stream has moderately more variance")
else:
    print("\n⚠️ Local and global orthogonal streams have similar variance")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("FINAL SUMMARY: Local Per-Pixel Orthogonal Stream")
print("="*80)

print(f"\n1. COMPUTATIONAL COST:")
print(f"   Time per image: {elapsed:.2f}s")
print(f"   Estimated full dataset: {estimated_time:.2f} hours")
print(f"   {'⚠️ Slow - consider caching or GPU acceleration' if elapsed > 1.0 else '✅ Acceptable'}")

print(f"\n2. INFORMATION CONTENT:")
print(f"   Variance ratio (local/global): {std_ratio:.2f}x")
print(f"   {'✅ Local captures more information' if std_ratio > 1.2 else '⚠️ Similar to global'}")

print(f"\n3. OUTPUT FORMAT:")
print(f"   [R, G, B, D, O] with shape [5, H, W]")
print(f"   O represents local orthogonal projection per pixel")

print(f"\n4. NEXT STEPS:")
if elapsed < 1.0 and std_ratio > 1.2:
    print(f"   ✅ Local orthogonal stream is ready to use!")
    print(f"   → Integrate into SUNRGBDDataset")
    print(f"   → Update LINet to accept 3 streams")
elif elapsed > 1.0:
    print(f"   ⚠️ Optimize performance first:")
    print(f"   → Vectorize SVD computation")
    print(f"   → Pre-compute and cache orthogonal streams")
    print(f"   → Use GPU acceleration")
else:
    print(f"   ⚠️ Local orthogonal stream doesn't add much over global")
    print(f"   → Consider using global instead")

print("\n" + "="*80)
