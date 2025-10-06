"""Verify RGB and depth get the same geometric transforms."""
import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data_utils.nyu_depth_dataset import create_nyu_dataloaders

print("=" * 80)
print("RGB-DEPTH SYNCHRONIZATION TEST")
print("=" * 80)

# Load dataloaders
train_loader, val_loader = create_nyu_dataloaders(
    h5_file_path='data/content/nyu_depth_v2_labeled.mat',
    batch_size=1,
    num_workers=0,
    target_size=(416, 544),
    num_classes=27
)

dataset = train_loader.dataset

print("\n1. Getting 4 augmented versions of the same image...")
samples = []
for i in range(4):
    rgb, depth, label = dataset[0]  # Always same index
    samples.append((rgb, depth))

# Visualize to check synchronization
print("\n2. Creating visualization...")
fig, axes = plt.subplots(4, 3, figsize=(12, 16))

for i in range(4):
    rgb, depth = samples[i]

    # Denormalize RGB
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    rgb_vis = rgb * std + mean
    rgb_vis = torch.clamp(rgb_vis, 0, 1)

    # RGB
    axes[i, 0].imshow(rgb_vis.permute(1, 2, 0).numpy())
    axes[i, 0].set_title(f"RGB Sample {i+1}")
    axes[i, 0].axis('off')

    # Depth
    depth_vis = depth.squeeze(0).numpy()
    axes[i, 1].imshow(depth_vis, cmap='viridis')
    axes[i, 1].set_title(f"Depth Sample {i+1}")
    axes[i, 1].axis('off')

    # Overlay (to check alignment)
    # Convert depth to RGB colormap
    from matplotlib import cm
    depth_colored = cm.viridis(depth_vis)[:, :, :3]

    # Blend RGB and depth
    blend = 0.5 * rgb_vis.permute(1, 2, 0).numpy() + 0.5 * depth_colored
    axes[i, 2].imshow(blend)
    axes[i, 2].set_title(f"Overlay {i+1}")
    axes[i, 2].axis('off')

plt.suptitle('RGB-Depth Synchronization Test\n(Same transforms should produce aligned overlays)',
             fontsize=14, fontweight='bold')
plt.tight_layout()

output_path = project_root / 'rgb_depth_sync_test.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n3. Saved to: {output_path}")

print("\n4. Manual verification:")
print("   Look at the overlay column:")
print("   - If RGB and depth are synchronized, objects should align")
print("   - If misaligned, you'll see ghosting/double edges")
print("   - Flips and rotations should match between RGB and depth")

print("\n" + "=" * 80)
print("✅ Check the saved image to verify synchronization")
print("=" * 80)
