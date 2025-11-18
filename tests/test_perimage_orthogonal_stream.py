"""
Test per-image orthogonal stream extraction.

For each image:
1. Fit 3D hyperplane to that image's RGB-D pixel distribution
2. Extract the orthogonal vector (normal to the hyperplane)
3. Project all pixels onto this orthogonal vector to create Stream 3
4. Stream 3 becomes a single-channel image (like depth)

Result:
- Stream 1: RGB (3 channels)
- Stream 2: Depth (1 channel)
- Stream 3: Orthogonal projection (1 channel)
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import torch
from src.data_utils.sunrgbd_dataset import SUNRGBDDataset
import matplotlib.pyplot as plt

print("="*80)
print("PER-IMAGE ORTHOGONAL STREAM EXTRACTION")
print("="*80)
print("\nGoal: Create a 3rd stream by projecting each image's pixels onto")
print("      the orthogonal vector of that image's RGB-D hyperplane\n")

# Load dataset
train_dataset = SUNRGBDDataset(train=True)
print(f"Loaded SUN RGB-D train set: {len(train_dataset)} samples\n")

def extract_orthogonal_stream(rgb, depth):
    """
    Extract orthogonal stream for a single image.

    Args:
        rgb: (3, H, W) RGB tensor (normalized)
        depth: (1, H, W) Depth tensor (normalized)

    Returns:
        orth_stream: (1, H, W) orthogonal projection values
        orth_vector: (4,) orthogonal vector [R, G, B, D]
        fit_quality: float, percentage of variance explained by 3D plane
    """
    # Denormalize to original [0, 1] range
    rgb_denorm = rgb * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    depth_denorm = depth * 0.2197 + 0.5027  # Denormalize depth to [0, 1] range

    # Flatten to get all pixels
    H, W = rgb.shape[1], rgb.shape[2]
    r = rgb_denorm[0].flatten().numpy()
    g = rgb_denorm[1].flatten().numpy()
    b = rgb_denorm[2].flatten().numpy()
    d = depth_denorm[0].flatten().numpy()

    # Stack into (N_pixels, 4) matrix
    X = np.stack([r, g, b, d], axis=1)

    # Center data
    X_mean = X.mean(axis=0)
    X_centered = X - X_mean

    # SVD to find hyperplane
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # Orthogonal vector (4th singular vector - smallest variance)
    orth_vector = Vt[3, :]  # Shape: (4,)

    # Fit quality
    fit_quality = 1 - (S[3]**2 / np.sum(S**2))

    # Project all pixels onto orthogonal vector
    # orth_stream[i] = dot(X_centered[i], orth_vector)
    orth_values = X_centered @ orth_vector  # Shape: (N_pixels,)

    # Reshape back to image shape
    orth_stream = orth_values.reshape(H, W)

    # Convert to tensor
    orth_stream = torch.from_numpy(orth_stream).float().unsqueeze(0)  # (1, H, W)

    return orth_stream, orth_vector, fit_quality


# ============================================================================
# TEST 1: Extract orthogonal stream for sample images
# ============================================================================
print("="*80)
print("TEST 1: Extract Orthogonal Stream for Sample Images")
print("="*80)

num_test_images = 10
print(f"\nExtracting orthogonal streams for {num_test_images} images...\n")

fit_qualities = []
orth_vectors = []
orth_stream_stats = []

for i in range(num_test_images):
    rgb, depth, label = train_dataset[i]

    orth_stream, orth_vector, fit_quality = extract_orthogonal_stream(rgb, depth)

    fit_qualities.append(fit_quality)
    orth_vectors.append(orth_vector)

    # Compute statistics
    orth_mean = orth_stream.mean().item()
    orth_std = orth_stream.std().item()
    orth_min = orth_stream.min().item()
    orth_max = orth_stream.max().item()
    orth_stream_stats.append({
        'mean': orth_mean,
        'std': orth_std,
        'min': orth_min,
        'max': orth_max,
        'range': orth_max - orth_min
    })

    print(f"Image {i+1}:")
    print(f"  Fit quality:     {fit_quality*100:.2f}%")
    print(f"  Orth vector:     [{orth_vector[0]:+.3f}*R, {orth_vector[1]:+.3f}*G, {orth_vector[2]:+.3f}*B, {orth_vector[3]:+.3f}*D]")
    print(f"  Orth stream:     mean={orth_mean:.3f}, std={orth_std:.3f}, range=[{orth_min:.3f}, {orth_max:.3f}]")
    print()

print("-"*80)
print("SUMMARY STATISTICS:")
print("-"*80)
print(f"Fit quality:         {np.mean(fit_qualities)*100:.2f}% ± {np.std(fit_qualities)*100:.2f}%")
print(f"Orth stream range:   {np.mean([s['range'] for s in orth_stream_stats]):.3f} ± {np.std([s['range'] for s in orth_stream_stats]):.3f}")
print(f"Orth stream std:     {np.mean([s['std'] for s in orth_stream_stats]):.3f} ± {np.std([s['std'] for s in orth_stream_stats]):.3f}")

# Check orthogonal vector consistency
orth_vectors = np.array(orth_vectors)
# Compute pairwise angles
angles = []
for i in range(len(orth_vectors)):
    for j in range(i+1, len(orth_vectors)):
        # Use abs(dot) to account for sign ambiguity
        cos_angle = np.abs(np.dot(orth_vectors[i], orth_vectors[j]))
        cos_angle = np.clip(cos_angle, -1, 1)  # Numerical stability
        angle = np.arccos(cos_angle) * 180 / np.pi
        angles.append(angle)

print(f"\nOrthogonal vector consistency:")
print(f"  Mean angle between images: {np.mean(angles):.1f}° ± {np.std(angles):.1f}°")
print(f"  Range: [{np.min(angles):.1f}°, {np.max(angles):.1f}°]")

if np.mean(angles) < 30:
    print("  ✅ Vectors are CONSISTENT across images")
else:
    print("  ⚠️ Vectors vary significantly across images (scene-dependent)")

# ============================================================================
# TEST 2: Visualize orthogonal streams
# ============================================================================
print("\n" + "="*80)
print("TEST 2: Visualize Orthogonal Streams")
print("="*80)

fig, axes = plt.subplots(4, 4, figsize=(16, 16))

for idx in range(4):
    rgb, depth, label = train_dataset[idx]
    orth_stream, orth_vector, fit_quality = extract_orthogonal_stream(rgb, depth)

    # Denormalize for visualization
    rgb_vis = rgb * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    rgb_vis = torch.clamp(rgb_vis, 0, 1)
    rgb_vis = rgb_vis.permute(1, 2, 0).numpy()

    depth_vis = depth.squeeze().numpy()
    orth_vis = orth_stream.squeeze().numpy()

    # Plot RGB
    axes[idx, 0].imshow(rgb_vis)
    axes[idx, 0].set_title(f'Image {idx+1}: RGB')
    axes[idx, 0].axis('off')

    # Plot Depth
    axes[idx, 1].imshow(depth_vis, cmap='viridis')
    axes[idx, 1].set_title('Depth (Stream 2)')
    axes[idx, 1].axis('off')

    # Plot Orthogonal Stream
    im = axes[idx, 2].imshow(orth_vis, cmap='RdBu_r')
    axes[idx, 2].set_title(f'Orthogonal (Stream 3)\nFit: {fit_quality*100:.1f}%')
    axes[idx, 2].axis('off')
    plt.colorbar(im, ax=axes[idx, 2], fraction=0.046)

    # Plot histogram of orthogonal values
    axes[idx, 3].hist(orth_vis.flatten(), bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    axes[idx, 3].set_title(f'Orth Distribution\nstd={orth_vis.std():.2f}')
    axes[idx, 3].set_xlabel('Orthogonal Value')
    axes[idx, 3].set_ylabel('Frequency')
    axes[idx, 3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('tests/perimage_orthogonal_streams.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved visualization: tests/perimage_orthogonal_streams.png")

# ============================================================================
# TEST 3: Independence test (correlation with RGB and Depth)
# ============================================================================
print("\n" + "="*80)
print("TEST 3: Independence Test")
print("="*80)
print("\nTesting if orthogonal stream is independent of RGB and Depth...\n")

correlations_with_rgb = []
correlations_with_depth = []

for i in range(20):
    rgb, depth, label = train_dataset[i]
    orth_stream, orth_vector, fit_quality = extract_orthogonal_stream(rgb, depth)

    # Flatten
    rgb_r = rgb[0].flatten().numpy()
    rgb_g = rgb[1].flatten().numpy()
    rgb_b = rgb[2].flatten().numpy()
    depth_flat = depth.flatten().numpy()
    orth_flat = orth_stream.flatten().numpy()

    # Compute correlations (use mean of RGB channels)
    corr_r = abs(np.corrcoef(rgb_r, orth_flat)[0, 1])
    corr_g = abs(np.corrcoef(rgb_g, orth_flat)[0, 1])
    corr_b = abs(np.corrcoef(rgb_b, orth_flat)[0, 1])
    corr_rgb = (corr_r + corr_g + corr_b) / 3
    corr_depth = abs(np.corrcoef(depth_flat, orth_flat)[0, 1])

    correlations_with_rgb.append(abs(corr_rgb))
    correlations_with_depth.append(abs(corr_depth))

print(f"Correlation with RGB channels:")
print(f"  Mean: {np.mean(correlations_with_rgb):.4f} ± {np.std(correlations_with_rgb):.4f}")
print(f"  Range: [{np.min(correlations_with_rgb):.4f}, {np.max(correlations_with_rgb):.4f}]")

print(f"\nCorrelation with Depth:")
print(f"  Mean: {np.mean(correlations_with_depth):.4f} ± {np.std(correlations_with_depth):.4f}")
print(f"  Range: [{np.min(correlations_with_depth):.4f}, {np.max(correlations_with_depth):.4f}]")

if np.mean(correlations_with_rgb) < 0.3 and np.mean(correlations_with_depth) < 0.3:
    print("\n✅ Orthogonal stream is INDEPENDENT (low correlation)")
else:
    print("\n⚠️ Orthogonal stream shows some correlation with inputs")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("FINAL SUMMARY: Per-Image Orthogonal Stream")
print("="*80)

print(f"\n1. HYPERPLANE FIT QUALITY:")
print(f"   Mean: {np.mean(fit_qualities)*100:.2f}% ± {np.std(fit_qualities)*100:.2f}%")
print(f"   {'✅ Excellent fit' if np.mean(fit_qualities) > 0.99 else '⚠️ Moderate fit'}")

print(f"\n2. ORTHOGONAL VECTOR CONSISTENCY:")
print(f"   Angular deviation: {np.mean(angles):.1f}° ± {np.std(angles):.1f}°")
print(f"   {'✅ Consistent across images' if np.mean(angles) < 30 else '⚠️ Scene-dependent'}")

print(f"\n3. STREAM 3 CHARACTERISTICS:")
print(f"   Mean std deviation: {np.mean([s['std'] for s in orth_stream_stats]):.3f}")
print(f"   Mean range: {np.mean([s['range'] for s in orth_stream_stats]):.3f}")
print(f"   {'✅ Has meaningful variance' if np.mean([s['std'] for s in orth_stream_stats]) > 0.1 else '⚠️ Low variance'}")

print(f"\n4. INDEPENDENCE:")
print(f"   Correlation with RGB: {np.mean(correlations_with_rgb):.4f}")
print(f"   Correlation with Depth: {np.mean(correlations_with_depth):.4f}")
print(f"   {'✅ Independent' if np.mean(correlations_with_rgb) < 0.3 else '⚠️ Correlated'}")

print(f"\n{'✅ READY TO USE AS 3RD STREAM!' if (np.mean(fit_qualities) > 0.99 and np.mean([s['std'] for s in orth_stream_stats]) > 0.1) else '⚠️ May need refinement'}")

print("\n" + "="*80)
print("Next step: Implement orthogonal stream extraction in dataset class")
print("="*80)
