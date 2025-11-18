#!/usr/bin/env python3
"""
Viewer for orthogonal stream PNG files with colormap applied.

Usage:
    python view_orthogonal.py <image_index>
    python view_orthogonal.py 0
    python view_orthogonal.py 42

Or view multiple samples:
    python view_orthogonal.py --gallery
"""

import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_orthogonal(idx, split='train'):
    """Load orthogonal stream from preprocessed file."""
    # Load normalization parameters
    with open('data/sunrgbd_15/orth_normalization.txt', 'r') as f:
        vmin = float(f.readline().strip())
        vmax = float(f.readline().strip())

    # Load 16-bit PNG
    orth_path = f'data/sunrgbd_15/{split}/orth/{idx:05d}.png'
    orth_uint16 = np.array(Image.open(orth_path), dtype=np.uint16)

    # Denormalize to actual values
    orth_values = (orth_uint16.astype(np.float32) / 65535.0) * (vmax - vmin) + vmin

    return orth_values

def load_rgb(idx, split='train'):
    """Load RGB image."""
    rgb_path = f'data/sunrgbd_15/{split}/rgb/{idx:05d}.png'
    rgb = np.array(Image.open(rgb_path))
    return rgb

def load_depth(idx, split='train'):
    """Load depth image."""
    depth_path = f'data/sunrgbd_15/{split}/depth/{idx:05d}.png'
    depth = np.array(Image.open(depth_path))
    return depth

def view_single(idx, split='train'):
    """View a single sample with all modalities."""
    # Load all modalities
    rgb = load_rgb(idx, split)
    depth = load_depth(idx, split)
    orth = load_orthogonal(idx, split)

    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # RGB
    axes[0].imshow(rgb)
    axes[0].set_title(f'RGB (Sample {idx})')
    axes[0].axis('off')

    # Depth
    im1 = axes[1].imshow(depth, cmap='viridis')
    axes[1].set_title('Depth')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    # Orthogonal (auto-scale like test images)
    im2 = axes[2].imshow(orth, cmap='RdBu_r')
    axes[2].set_title(f'Orthogonal (auto-scale)\nstd={orth.std():.4f}')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    # Orthogonal (enhanced contrast)
    mean, std = orth.mean(), orth.std()
    im3 = axes[3].imshow(orth, cmap='RdBu_r', vmin=mean-2*std, vmax=mean+2*std)
    axes[3].set_title(f'Orthogonal (enhanced)\n±2σ range')
    axes[3].axis('off')
    plt.colorbar(im3, ax=axes[3], fraction=0.046)

    plt.suptitle(f'{split.upper()} Set - Sample {idx:05d}', fontsize=16)
    plt.tight_layout()
    plt.show()

def view_gallery(split='train', num_samples=9):
    """View a gallery of samples."""
    rows = 3
    cols = 4

    fig, axes = plt.subplots(rows, cols, figsize=(20, 15))

    for i in range(rows):
        idx = i * 100  # Sample every 100 images

        # Load data
        rgb = load_rgb(idx, split)
        depth = load_depth(idx, split)
        orth = load_orthogonal(idx, split)

        # RGB
        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(f'Sample {idx}: RGB')
        axes[i, 0].axis('off')

        # Depth
        im1 = axes[i, 1].imshow(depth, cmap='viridis')
        axes[i, 1].set_title('Depth')
        axes[i, 1].axis('off')

        # Orthogonal (auto-scale)
        im2 = axes[i, 2].imshow(orth, cmap='RdBu_r')
        axes[i, 2].set_title(f'Orthogonal\nstd={orth.std():.4f}')
        axes[i, 2].axis('off')
        plt.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)

        # Orthogonal (enhanced)
        mean, std = orth.mean(), orth.std()
        im3 = axes[i, 3].imshow(orth, cmap='RdBu_r', vmin=mean-2*std, vmax=mean+2*std)
        axes[i, 3].set_title('Enhanced')
        axes[i, 3].axis('off')
        plt.colorbar(im3, ax=axes[i, 3], fraction=0.046, pad=0.04)

    plt.suptitle(f'{split.upper()} Set - Gallery View', fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage:")
        print("  View single sample:  python view_orthogonal.py <index>")
        print("  View gallery:        python view_orthogonal.py --gallery")
        print("\nExamples:")
        print("  python view_orthogonal.py 0")
        print("  python view_orthogonal.py 42")
        print("  python view_orthogonal.py --gallery")
        sys.exit(1)

    if sys.argv[1] == '--gallery':
        split = 'val' if '--val' in sys.argv else 'train'
        view_gallery(split=split)
    else:
        idx = int(sys.argv[1])
        split = 'val' if '--val' in sys.argv else 'train'
        view_single(idx, split=split)
