"""Find problematic depth images that cause NaN values."""

import numpy as np
from PIL import Image
import os

data_root = 'data/sunrgbd_15'
split = 'train'
depth_dir = os.path.join(data_root, split, 'depth')

print("Scanning for problematic depth images...")
problematic = []

# Check first 100 images
for idx in range(100):
    depth_path = os.path.join(depth_dir, f'{idx:05d}.png')
    if os.path.exists(depth_path):
        depth = Image.open(depth_path)
        depth_arr = np.array(depth, dtype=np.float32)

        if depth_arr.max() == 0:
            problematic.append((idx, 'all zeros', depth_arr.min(), depth_arr.max()))
        elif np.isnan(depth_arr).any():
            problematic.append((idx, 'contains NaN', depth_arr.min(), depth_arr.max()))
        elif depth_arr.max() < 1e-6:
            problematic.append((idx, 'near zero', depth_arr.min(), depth_arr.max()))

if problematic:
    print(f"\nFound {len(problematic)} problematic images:")
    for idx, issue, min_val, max_val in problematic:
        print(f"  Image {idx:05d}: {issue}, range=[{min_val}, {max_val}]")
else:
    print("\nNo problematic images found in first 100 samples")
