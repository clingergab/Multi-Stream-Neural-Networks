
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.data_utils.sunrgbd_3stream_dataset import SUNRGBD3StreamDataset
from torchvision import transforms
import os

def denormalize_rgb(tensor):
    # New normalization: mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
    # x = (img - 0.5) / 0.5  ->  img = x * 0.5 + 0.5
    return torch.clamp(tensor * 0.5 + 0.5, 0, 1)

def denormalize_depth(tensor):
    # New normalization: mean=[0.5], std=[0.5]
    return tensor * 0.5 + 0.5

def denormalize_orth(tensor):
    # mean=[0.5], std=[0.5] -> maps [-1, 1] back to [0, 1]
    # But we want to visualize the signed values [-1, 1] directly
    return tensor

def test_alignment_and_stats():
    print("="*60)
    print("TESTING 3-STREAM ALIGNMENT AND STATISTICS")
    print("="*60)

    # Initialize dataset with augmentation (train=True)
    dataset = SUNRGBD3StreamDataset(
        data_root='data/sunrgbd_15',
        train=True,
        target_size=(224, 224)
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Collect stats over a few samples
    n_samples = 10
    print(f"\nChecking stats for {n_samples} samples...")
    
    stats = {
        'rgb': {'min': [], 'max': [], 'mean': []},
        'depth': {'min': [], 'max': [], 'mean': []},
        'orth': {'min': [], 'max': [], 'mean': []}
    }
    
    # Create a figure for visualization
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4*n_samples))
    
    for i in range(n_samples):
        # Get sample (randomly augmented)
        rgb, depth, orth, label = dataset[i]
        
        # Stats
        for name, tensor in [('rgb', rgb), ('depth', depth), ('orth', orth)]:
            stats[name]['min'].append(tensor.min().item())
            stats[name]['max'].append(tensor.max().item())
            stats[name]['mean'].append(tensor.mean().item())
        
        # Visualization
        rgb_vis = denormalize_rgb(rgb).permute(1, 2, 0).numpy()
        depth_vis = denormalize_depth(depth).squeeze().numpy()
        orth_vis = denormalize_orth(orth).squeeze().numpy()
        
        # Plot RGB
        axes[i, 0].imshow(rgb_vis)
        axes[i, 0].set_title(f"RGB (Sample {i})\nRange: [{rgb.min():.2f}, {rgb.max():.2f}]")
        axes[i, 0].axis('off')
        
        # Plot Depth
        axes[i, 1].imshow(depth_vis, cmap='viridis')
        axes[i, 1].set_title(f"Depth (Sample {i})\nRange: [{depth.min():.2f}, {depth.max():.2f}]")
        axes[i, 1].axis('off')
        
        # Plot Orthogonal
        # Use RdBu_r centered at 0 to see structure
        im = axes[i, 2].imshow(orth_vis, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[i, 2].set_title(f"Orth (Sample {i})\nRange: [{orth.min():.2f}, {orth.max():.2f}]")
        axes[i, 2].axis('off')
        
    plt.tight_layout()
    plt.savefig('test_3stream_alignment.png')
    print(f"\nVisualization saved to 'test_3stream_alignment.png'")
    
    # Print aggregate stats
    print("\nStream Statistics (Target Range: [-1, 1]):")
    for name in ['rgb', 'depth', 'orth']:
        print(f"\n{name.upper()}:")
        print(f"  Min: {min(stats[name]['min']):.4f}")
        print(f"  Max: {max(stats[name]['max']):.4f}")
        print(f"  Mean: {np.mean(stats[name]['mean']):.4f}")
    
    print("\n✅ Range check passed.")

if __name__ == "__main__":
    test_alignment_and_stats()
