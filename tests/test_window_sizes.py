"""
Test different window sizes for local orthogonal stream extraction.

Compare:
- 3×3 window (9 points)
- 5×5 window (25 points)
- 7×7 window (49 points)

Metrics:
- Computation time
- Variance captured (information content)
- Visual quality
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import torch
from src.data_utils.sunrgbd_dataset import SUNRGBDDataset
import matplotlib.pyplot as plt
import time

print("="*80)
print("WINDOW SIZE COMPARISON FOR LOCAL ORTHOGONAL STREAM")
print("="*80)

train_dataset = SUNRGBDDataset(split='train')
print(f"\nLoaded SUN RGB-D train set: {len(train_dataset)} samples\n")


def extract_orthogonal_stream(rgb, depth, window_size=5):
    """Extract normalized local orthogonal stream with configurable window size."""
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


# Test different window sizes
window_sizes = [3, 5, 7]

print("="*80)
print("TEST 1: Computation Time")
print("="*80)

rgb, depth, label = train_dataset[0]

for ws in window_sizes:
    print(f"\nWindow size: {ws}×{ws} ({ws*ws} points)")

    start_time = time.time()
    orth_stream = extract_orthogonal_stream(rgb, depth, window_size=ws)
    elapsed = time.time() - start_time

    estimated_hours = elapsed * len(train_dataset) / 3600

    print(f"  Time per image: {elapsed:.2f}s")
    print(f"  Estimated full dataset: {estimated_hours:.2f} hours")
    print(f"  Speedup vs 5×5: {1.73/elapsed:.2f}x" if ws != 5 else "  (baseline)")


# Test information content
print("\n" + "="*80)
print("TEST 2: Information Content (Variance)")
print("="*80)

stats = {ws: [] for ws in window_sizes}

print("\nAnalyzing 10 images...\n")

for i in range(10):
    rgb, depth, label = train_dataset[i]

    for ws in window_sizes:
        orth = extract_orthogonal_stream(rgb, depth, window_size=ws)
        stats[ws].append({
            'std': orth.std().item(),
            'range': (orth.max() - orth.min()).item()
        })

print("-"*80)
print(f"{'Window Size':<15} {'Mean Std':<15} {'Mean Range':<15} {'vs 5×5':<10}")
print("-"*80)

baseline_std = np.mean([s['std'] for s in stats[5]])

for ws in window_sizes:
    mean_std = np.mean([s['std'] for s in stats[ws]])
    mean_range = np.mean([s['range'] for s in stats[ws]])
    ratio = mean_std / baseline_std

    print(f"{ws}×{ws} ({ws*ws} pts)   {mean_std:<15.6f} {mean_range:<15.6f} {ratio:.2f}x")


# Visual comparison
print("\n" + "="*80)
print("TEST 3: Visual Quality")
print("="*80)

print("\nGenerating comparison for 3 images...\n")

fig, axes = plt.subplots(3, 5, figsize=(20, 12))

for img_idx in range(3):
    rgb, depth, label = train_dataset[img_idx]

    # RGB
    rgb_vis = rgb * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    rgb_vis = torch.clamp(rgb_vis, 0, 1).permute(1, 2, 0).numpy()

    axes[img_idx, 0].imshow(rgb_vis)
    axes[img_idx, 0].set_title(f'Image {img_idx+1}: RGB')
    axes[img_idx, 0].axis('off')

    # Depth
    depth_vis = depth.squeeze().numpy()
    axes[img_idx, 1].imshow(depth_vis, cmap='viridis')
    axes[img_idx, 1].set_title('Depth')
    axes[img_idx, 1].axis('off')

    # Different window sizes
    for col_idx, ws in enumerate(window_sizes):
        print(f"  Image {img_idx+1}, window {ws}×{ws}...")
        orth = extract_orthogonal_stream(rgb, depth, window_size=ws)
        orth_vis = orth.squeeze().numpy()

        im = axes[img_idx, col_idx+2].imshow(orth_vis, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[img_idx, col_idx+2].set_title(f'{ws}×{ws} window\nstd={orth_vis.std():.3f}')
        axes[img_idx, col_idx+2].axis('off')
        plt.colorbar(im, ax=axes[img_idx, col_idx+2], fraction=0.046)

plt.tight_layout()
plt.savefig('tests/window_size_comparison.png', dpi=100, bbox_inches='tight')
print("\n✓ Saved visualization: tests/window_size_comparison.png")
plt.close()


# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print("\nRecommendations:\n")

for ws in window_sizes:
    mean_std = np.mean([s['std'] for s in stats[ws]])

    print(f"{ws}×{ws} window ({ws*ws} points):")

    if ws == 3:
        print(f"  Pros: Fastest (good for real-time)")
        print(f"  Cons: May be noisy, only {ws*ws} points for 4D hyperplane")
        print(f"  Use case: Real-time applications, fine detail needed")
    elif ws == 5:
        print(f"  Pros: Good balance of speed and stability")
        print(f"  Cons: Moderate computation time (~2s/image)")
        print(f"  Use case: RECOMMENDED - good all-around choice")
    elif ws == 7:
        print(f"  Pros: Most stable hyperplane fitting")
        print(f"  Cons: Slowest, may over-smooth edges")
        print(f"  Use case: When quality > speed, very noisy data")

    print()

print("="*80)
