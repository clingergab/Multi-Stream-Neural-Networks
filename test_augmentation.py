"""Test that data augmentation works correctly and stays synchronized."""
import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data_utils.nyu_depth_dataset import create_nyu_dataloaders

print("=" * 80)
print("DATA AUGMENTATION TEST")
print("=" * 80)

# Load dataloaders
print("\n1. Loading dataloaders...")
train_loader, val_loader = create_nyu_dataloaders(
    h5_file_path='data/content/nyu_depth_v2_labeled.mat',
    batch_size=4,
    num_workers=0,
    target_size=(416, 544),
    num_classes=27
)

# Get same sample multiple times to see augmentation variation
print("\n2. Testing augmentation synchronization...")
print("   Fetching the same sample 3 times to check if augmentation varies...")

dataset = train_loader.dataset
rgb_samples = []
depth_samples = []

# Get index 0 three times
for i in range(3):
    rgb, depth, label = dataset[0]  # Always index 0
    rgb_samples.append(rgb)
    depth_samples.append(depth)
    print(f"   Sample {i+1}: RGB shape={rgb.shape}, Depth shape={depth.shape}, Label={label}")

# Check if augmentation is working (samples should differ)
print("\n3. Checking augmentation variation...")
for i in range(1, 3):
    rgb_diff = torch.abs(rgb_samples[0] - rgb_samples[i]).mean().item()
    depth_diff = torch.abs(depth_samples[0] - depth_samples[i]).mean().item()
    print(f"   Sample 0 vs {i}:")
    print(f"     RGB mean diff: {rgb_diff:.4f}")
    print(f"     Depth mean diff: {depth_diff:.4f}")

if rgb_diff > 0.01:
    print("   ✅ Augmentation is varying RGB images")
else:
    print("   ⚠️  RGB images are identical - augmentation might not be working")

if depth_diff > 0.01:
    print("   ✅ Augmentation is varying depth images")
else:
    print("   ⚠️  Depth images are identical - augmentation might not be working")

# Visual test: Show augmented samples
print("\n4. Creating visualization...")
fig, axes = plt.subplots(3, 6, figsize=(18, 9))

for i in range(3):
    # Denormalize RGB for visualization
    rgb = rgb_samples[i]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    rgb_vis = rgb * std + mean
    rgb_vis = torch.clamp(rgb_vis, 0, 1)

    # RGB
    axes[i, 0].imshow(rgb_vis.permute(1, 2, 0).numpy())
    axes[i, 0].set_title(f"RGB Sample {i+1}")
    axes[i, 0].axis('off')

    # Depth
    depth_vis = depth_samples[i].squeeze(0).numpy()
    axes[i, 1].imshow(depth_vis, cmap='viridis')
    axes[i, 1].set_title(f"Depth Sample {i+1}")
    axes[i, 1].axis('off')

    # RGB - Channel 0 (Red)
    axes[i, 2].imshow(rgb_vis[0].numpy(), cmap='gray')
    axes[i, 2].set_title(f"R Channel {i+1}")
    axes[i, 2].axis('off')

    # RGB - Channel 1 (Green)
    axes[i, 3].imshow(rgb_vis[1].numpy(), cmap='gray')
    axes[i, 3].set_title(f"G Channel {i+1}")
    axes[i, 3].axis('off')

    # RGB - Channel 2 (Blue)
    axes[i, 4].imshow(rgb_vis[2].numpy(), cmap='gray')
    axes[i, 4].set_title(f"B Channel {i+1}")
    axes[i, 4].axis('off')

    # Depth (repeated for symmetry)
    axes[i, 5].imshow(depth_vis, cmap='gray')
    axes[i, 5].set_title(f"Depth {i+1}")
    axes[i, 5].axis('off')

plt.suptitle('Augmentation Test: Same Image Fetched 3 Times (Should Vary)', fontsize=16, fontweight='bold')
plt.tight_layout()

output_path = project_root / 'augmentation_test.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n5. Visualization saved to: {output_path}")

# Test batch loading
print("\n6. Testing batch loading...")
rgb_batch, depth_batch, label_batch = next(iter(train_loader))
print(f"   Batch RGB shape: {rgb_batch.shape}")
print(f"   Batch depth shape: {depth_batch.shape}")
print(f"   Batch labels: {label_batch.tolist()}")

# Check for NaN or Inf
if torch.isnan(rgb_batch).any() or torch.isinf(rgb_batch).any():
    print("   ❌ RGB batch contains NaN or Inf!")
else:
    print("   ✅ RGB batch is clean")

if torch.isnan(depth_batch).any() or torch.isinf(depth_batch).any():
    print("   ❌ Depth batch contains NaN or Inf!")
else:
    print("   ✅ Depth batch is clean")

print("\n" + "=" * 80)
print("✅ Augmentation test complete!")
print("=" * 80)
