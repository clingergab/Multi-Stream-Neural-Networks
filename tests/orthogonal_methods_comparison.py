"""
Compare different methods for computing local orthogonal streams.

We'll test different approaches to see which captures the most information:
1. Current method: SVD on centered RGBD, project onto 4th singular vector
2. Residual method: Distance from local hyperplane
3. Weighted method: Weight by singular values
4. Normalized method: Normalize RGBD dimensions before SVD
5. Gradient method: Local RGBD gradient magnitude
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import torch
from src.data_utils.sunrgbd_dataset import SUNRGBDDataset
import matplotlib.pyplot as plt

print("="*80)
print("COMPARING ORTHOGONAL STREAM EXTRACTION METHODS")
print("="*80)

# Load dataset
train_dataset = SUNRGBDDataset(train=True)
print(f"\nLoaded SUN RGB-D train set: {len(train_dataset)} samples\n")


def method1_current(rgb, depth, window_size=5):
    """Current method: SVD projection onto 4th singular vector."""
    rgb_denorm = rgb * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    depth_denorm = depth * 0.2197 + 0.5027

    H, W = rgb.shape[1], rgb.shape[2]
    pad = window_size // 2
    orth_stream = np.zeros((H, W), dtype=np.float32)

    rgb_padded = torch.nn.functional.pad(rgb_denorm, (pad, pad, pad, pad), mode='reflect')
    depth_padded = torch.nn.functional.pad(depth_denorm, (pad, pad, pad, pad), mode='reflect')

    rgb_np = rgb_padded.numpy()
    depth_np = depth_padded.numpy()

    for i in range(H):
        for j in range(W):
            r_patch = rgb_np[0, i:i+window_size, j:j+window_size].flatten()
            g_patch = rgb_np[1, i:i+window_size, j:j+window_size].flatten()
            b_patch = rgb_np[2, i:i+window_size, j:j+window_size].flatten()
            d_patch = depth_np[0, i:i+window_size, j:j+window_size].flatten()

            X = np.stack([r_patch, g_patch, b_patch, d_patch], axis=1)

            if X.std() < 1e-6:
                orth_stream[i, j] = 0.0
                continue

            X_mean = X.mean(axis=0)
            X_centered = X - X_mean

            try:
                U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
                orth_vector = Vt[3, :]

                center_r = rgb_denorm[0, i, j].item()
                center_g = rgb_denorm[1, i, j].item()
                center_b = rgb_denorm[2, i, j].item()
                center_d = depth_denorm[0, i, j].item()
                center_rgbd = np.array([center_r, center_g, center_b, center_d])
                center_rgbd_centered = center_rgbd - X_mean

                orth_stream[i, j] = np.dot(center_rgbd_centered, orth_vector)
            except np.linalg.LinAlgError:
                orth_stream[i, j] = 0.0

    return torch.from_numpy(orth_stream).float().unsqueeze(0)


def method2_residual(rgb, depth, window_size=5):
    """Residual method: Distance from fitted hyperplane."""
    rgb_denorm = rgb * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    depth_denorm = depth * 0.2197 + 0.5027

    H, W = rgb.shape[1], rgb.shape[2]
    pad = window_size // 2
    orth_stream = np.zeros((H, W), dtype=np.float32)

    rgb_padded = torch.nn.functional.pad(rgb_denorm, (pad, pad, pad, pad), mode='reflect')
    depth_padded = torch.nn.functional.pad(depth_denorm, (pad, pad, pad, pad), mode='reflect')

    rgb_np = rgb_padded.numpy()
    depth_np = depth_padded.numpy()

    for i in range(H):
        for j in range(W):
            r_patch = rgb_np[0, i:i+window_size, j:j+window_size].flatten()
            g_patch = rgb_np[1, i:i+window_size, j:j+window_size].flatten()
            b_patch = rgb_np[2, i:i+window_size, j:j+window_size].flatten()
            d_patch = depth_np[0, i:i+window_size, j:j+window_size].flatten()

            X = np.stack([r_patch, g_patch, b_patch, d_patch], axis=1)

            if X.std() < 1e-6:
                orth_stream[i, j] = 0.0
                continue

            X_mean = X.mean(axis=0)
            X_centered = X - X_mean

            try:
                U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

                # Use first 3 singular vectors to reconstruct (the hyperplane)
                X_reconstructed = U[:, :3] @ np.diag(S[:3]) @ Vt[:3, :]

                # Center pixel
                center_r = rgb_denorm[0, i, j].item()
                center_g = rgb_denorm[1, i, j].item()
                center_b = rgb_denorm[2, i, j].item()
                center_d = depth_denorm[0, i, j].item()
                center_rgbd = np.array([center_r, center_g, center_b, center_d])
                center_rgbd_centered = center_rgbd - X_mean

                # Find distance to hyperplane (residual)
                center_reconstructed = Vt[:3, :].T @ (Vt[:3, :] @ center_rgbd_centered)
                residual = center_rgbd_centered - center_reconstructed
                orth_stream[i, j] = np.linalg.norm(residual)

            except np.linalg.LinAlgError:
                orth_stream[i, j] = 0.0

    return torch.from_numpy(orth_stream).float().unsqueeze(0)


def method3_weighted(rgb, depth, window_size=5):
    """Weighted method: Weight projection by smallest singular value."""
    rgb_denorm = rgb * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    depth_denorm = depth * 0.2197 + 0.5027

    H, W = rgb.shape[1], rgb.shape[2]
    pad = window_size // 2
    orth_stream = np.zeros((H, W), dtype=np.float32)

    rgb_padded = torch.nn.functional.pad(rgb_denorm, (pad, pad, pad, pad), mode='reflect')
    depth_padded = torch.nn.functional.pad(depth_denorm, (pad, pad, pad, pad), mode='reflect')

    rgb_np = rgb_padded.numpy()
    depth_np = depth_padded.numpy()

    for i in range(H):
        for j in range(W):
            r_patch = rgb_np[0, i:i+window_size, j:j+window_size].flatten()
            g_patch = rgb_np[1, i:i+window_size, j:j+window_size].flatten()
            b_patch = rgb_np[2, i:i+window_size, j:j+window_size].flatten()
            d_patch = depth_np[0, i:i+window_size, j:j+window_size].flatten()

            X = np.stack([r_patch, g_patch, b_patch, d_patch], axis=1)

            if X.std() < 1e-6:
                orth_stream[i, j] = 0.0
                continue

            X_mean = X.mean(axis=0)
            X_centered = X - X_mean

            try:
                U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
                orth_vector = Vt[3, :]

                center_r = rgb_denorm[0, i, j].item()
                center_g = rgb_denorm[1, i, j].item()
                center_b = rgb_denorm[2, i, j].item()
                center_d = depth_denorm[0, i, j].item()
                center_rgbd = np.array([center_r, center_g, center_b, center_d])
                center_rgbd_centered = center_rgbd - X_mean

                # Weight by 4th singular value (planarity measure)
                # Higher S[3] = less planar = more orthogonal variation
                orth_stream[i, j] = np.dot(center_rgbd_centered, orth_vector) * S[3]

            except np.linalg.LinAlgError:
                orth_stream[i, j] = 0.0

    return torch.from_numpy(orth_stream).float().unsqueeze(0)


def method4_normalized(rgb, depth, window_size=5):
    """Normalized method: Standardize RGBD dimensions before SVD."""
    rgb_denorm = rgb * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    depth_denorm = depth * 0.2197 + 0.5027

    H, W = rgb.shape[1], rgb.shape[2]
    pad = window_size // 2
    orth_stream = np.zeros((H, W), dtype=np.float32)

    rgb_padded = torch.nn.functional.pad(rgb_denorm, (pad, pad, pad, pad), mode='reflect')
    depth_padded = torch.nn.functional.pad(depth_denorm, (pad, pad, pad, pad), mode='reflect')

    rgb_np = rgb_padded.numpy()
    depth_np = depth_padded.numpy()

    for i in range(H):
        for j in range(W):
            r_patch = rgb_np[0, i:i+window_size, j:j+window_size].flatten()
            g_patch = rgb_np[1, i:i+window_size, j:j+window_size].flatten()
            b_patch = rgb_np[2, i:i+window_size, j:j+window_size].flatten()
            d_patch = depth_np[0, i:i+window_size, j:j+window_size].flatten()

            X = np.stack([r_patch, g_patch, b_patch, d_patch], axis=1)

            if X.std() < 1e-6:
                orth_stream[i, j] = 0.0
                continue

            # Standardize each dimension (mean=0, std=1)
            X_mean = X.mean(axis=0)
            X_std = X.std(axis=0) + 1e-8
            X_normalized = (X - X_mean) / X_std

            try:
                U, S, Vt = np.linalg.svd(X_normalized, full_matrices=False)
                orth_vector = Vt[3, :]

                center_r = rgb_denorm[0, i, j].item()
                center_g = rgb_denorm[1, i, j].item()
                center_b = rgb_denorm[2, i, j].item()
                center_d = depth_denorm[0, i, j].item()
                center_rgbd = np.array([center_r, center_g, center_b, center_d])
                center_rgbd_normalized = (center_rgbd - X_mean) / X_std

                orth_stream[i, j] = np.dot(center_rgbd_normalized, orth_vector)

            except np.linalg.LinAlgError:
                orth_stream[i, j] = 0.0

    return torch.from_numpy(orth_stream).float().unsqueeze(0)


def method5_gradient(rgb, depth, window_size=5):
    """Gradient method: Local RGBD gradient magnitude."""
    rgb_denorm = rgb * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    depth_denorm = depth * 0.2197 + 0.5027

    H, W = rgb.shape[1], rgb.shape[2]
    pad = window_size // 2
    orth_stream = np.zeros((H, W), dtype=np.float32)

    rgb_padded = torch.nn.functional.pad(rgb_denorm, (pad, pad, pad, pad), mode='reflect')
    depth_padded = torch.nn.functional.pad(depth_denorm, (pad, pad, pad, pad), mode='reflect')

    rgb_np = rgb_padded.numpy()
    depth_np = depth_padded.numpy()

    for i in range(H):
        for j in range(W):
            r_patch = rgb_np[0, i:i+window_size, j:j+window_size]
            g_patch = rgb_np[1, i:i+window_size, j:j+window_size]
            b_patch = rgb_np[2, i:i+window_size, j:j+window_size]
            d_patch = depth_np[0, i:i+window_size, j:j+window_size]

            # Compute variance in the window
            r_var = r_patch.std()
            g_var = g_patch.std()
            b_var = b_patch.std()
            d_var = d_patch.std()

            # Combined gradient magnitude
            orth_stream[i, j] = np.sqrt(r_var**2 + g_var**2 + b_var**2 + d_var**2)

    return torch.from_numpy(orth_stream).float().unsqueeze(0)


# ============================================================================
# TEST: Compare all methods
# ============================================================================
print("="*80)
print("TESTING ALL METHODS")
print("="*80)

methods = {
    'Method 1: Current (SVD projection)': method1_current,
    'Method 2: Residual (distance to plane)': method2_residual,
    'Method 3: Weighted (S[3] * projection)': method3_weighted,
    'Method 4: Normalized (standardized SVD)': method4_normalized,
    'Method 5: Gradient (local variance)': method5_gradient,
}

print("\nTesting on 3 images...\n")

fig, axes = plt.subplots(3, 7, figsize=(28, 12))

for img_idx in range(3):
    rgb, depth, label = train_dataset[img_idx]

    # Denormalize for visualization
    rgb_vis = rgb * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    rgb_vis = torch.clamp(rgb_vis, 0, 1).permute(1, 2, 0).numpy()
    depth_vis = depth.squeeze().numpy()

    # Plot RGB and Depth
    axes[img_idx, 0].imshow(rgb_vis)
    axes[img_idx, 0].set_title(f'Image {img_idx+1}: RGB')
    axes[img_idx, 0].axis('off')

    axes[img_idx, 1].imshow(depth_vis, cmap='viridis')
    axes[img_idx, 1].set_title('Depth')
    axes[img_idx, 1].axis('off')

    # Test each method
    col_idx = 2
    for method_name, method_func in methods.items():
        print(f"  Image {img_idx+1}: {method_name}...")
        orth = method_func(rgb, depth, window_size=5)
        orth_vis = orth.squeeze().numpy()

        im = axes[img_idx, col_idx].imshow(orth_vis, cmap='RdBu_r')
        title = method_name.split(':')[0] + f'\nstd={orth_vis.std():.4f}'
        axes[img_idx, col_idx].set_title(title, fontsize=9)
        axes[img_idx, col_idx].axis('off')
        plt.colorbar(im, ax=axes[img_idx, col_idx], fraction=0.046)

        col_idx += 1

plt.tight_layout()
plt.savefig('tests/orthogonal_methods_comparison.png', dpi=100, bbox_inches='tight')
print("\n✓ Saved visualization: tests/orthogonal_methods_comparison.png")
plt.close()

# ============================================================================
# QUANTITATIVE COMPARISON
# ============================================================================
print("\n" + "="*80)
print("QUANTITATIVE COMPARISON")
print("="*80)

print("\nAnalyzing 10 images...\n")

stats = {name: [] for name in methods.keys()}

for i in range(10):
    rgb, depth, label = train_dataset[i]

    for method_name, method_func in methods.items():
        orth = method_func(rgb, depth, window_size=5)
        stats[method_name].append({
            'std': orth.std().item(),
            'mean': orth.mean().item(),
            'range': (orth.max() - orth.min()).item()
        })

print("-"*80)
print(f"{'Method':<40} {'Mean Std':<12} {'Mean Range':<12}")
print("-"*80)

for method_name in methods.keys():
    mean_std = np.mean([s['std'] for s in stats[method_name]])
    mean_range = np.mean([s['range'] for s in stats[method_name]])
    print(f"{method_name:<40} {mean_std:<12.6f} {mean_range:<12.6f}")

print("-"*80)

# Find best method
best_method = max(stats.keys(), key=lambda k: np.mean([s['std'] for s in stats[k]]))
print(f"\n✅ WINNER: {best_method}")
print(f"   (Highest variance = most information captured)")

print("\n" + "="*80)
