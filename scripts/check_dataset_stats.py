
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch

def check_stats(data_root='data/sunrgbd_15', split='train'):
    print(f"Checking statistics for {split} set in {data_root}...")
    
    rgb_dir = os.path.join(data_root, split, 'rgb')
    depth_dir = os.path.join(data_root, split, 'depth')
    orth_dir = os.path.join(data_root, split, 'orth')
    
    # Get list of files (assuming filenames match across directories)
    files = sorted(os.listdir(rgb_dir))
    # Filter for images
    files = [f for f in files if f.endswith('.png') or f.endswith('.jpg')]
    
    print(f"Found {len(files)} samples.")
    
    # Initialize stats
    rgb_min, rgb_max = float('inf'), float('-inf')
    depth_min, depth_max = float('inf'), float('-inf')
    orth_min, orth_max = float('inf'), float('-inf')
    
    # For mean/std calculation (running sum)
    # RGB
    rgb_sum = np.zeros(3)
    rgb_sq_sum = np.zeros(3)
    rgb_pixel_count = 0
    
    # Depth
    depth_sum = 0
    depth_sq_sum = 0
    depth_pixel_count = 0
    
    # Orth
    orth_sum = 0
    orth_sq_sum = 0
    orth_pixel_count = 0
    
    for f in tqdm(files):
        # --- RGB ---
        rgb_path = os.path.join(rgb_dir, f)
        rgb = Image.open(rgb_path).convert('RGB')
        rgb_arr = np.array(rgb) # Raw [0, 255]
        
        rgb_min = min(rgb_min, rgb_arr.min())
        rgb_max = max(rgb_max, rgb_arr.max())
        
        rgb_sum += rgb_arr.sum(axis=(0, 1))
        rgb_sq_sum += (rgb_arr.astype(np.float64) ** 2).sum(axis=(0, 1))
        rgb_pixel_count += rgb_arr.shape[0] * rgb_arr.shape[1]
        
        # --- Depth ---
        depth_path = os.path.join(depth_dir, f)
        depth = Image.open(depth_path)
        # Check raw depth values before any processing
        depth_arr = np.array(depth) # Raw values
        
        depth_min = min(depth_min, depth_arr.min())
        depth_max = max(depth_max, depth_arr.max())
        
        depth_sum += depth_arr.sum()
        depth_sq_sum += (depth_arr.astype(np.float64) ** 2).sum()
        depth_pixel_count += depth_arr.size
        
        # --- Orthogonal ---
        orth_path = os.path.join(orth_dir, f)
        orth = Image.open(orth_path)
        # Check raw orth values
        orth_arr = np.array(orth) # Raw values
        
        orth_min = min(orth_min, orth_arr.min())
        orth_max = max(orth_max, orth_arr.max())
        
        orth_sum += orth_arr.sum()
        orth_sq_sum += (orth_arr.astype(np.float64) ** 2).sum()
        orth_pixel_count += orth_arr.size

    # Calculate final stats
    print("\n--- RAW Statistics (Before Normalization) ---")
    
    # RGB
    rgb_mean = rgb_sum / rgb_pixel_count
    rgb_std = np.sqrt(rgb_sq_sum / rgb_pixel_count - rgb_mean ** 2)
    print(f"RGB (Raw):")
    print(f"  Min: {rgb_min}, Max: {rgb_max}")
    print(f"  Mean: {rgb_mean}, Std: {rgb_std}")
    
    # Depth
    depth_mean = depth_sum / depth_pixel_count
    depth_std = np.sqrt(depth_sq_sum / depth_pixel_count - depth_mean ** 2)
    print(f"Depth (Raw):")
    print(f"  Min: {depth_min}, Max: {depth_max}")
    print(f"  Mean: {depth_mean:.4f}, Std: {depth_std:.4f}")
    
    # Orth
    orth_mean = orth_sum / orth_pixel_count
    orth_std = np.sqrt(orth_sq_sum / orth_pixel_count - orth_mean ** 2)
    print(f"Orthogonal (Raw):")
    print(f"  Min: {orth_min}, Max: {orth_max}")
    print(f"  Mean: {orth_mean:.4f}, Std: {orth_std:.4f}")
    
    # Suggest scaling for Orth
    print("\n--- Analysis ---")
    if orth_max > 255:
        print(f"Orthogonal is > 8-bit. Max value: {orth_max}")
    else:
        print(f"Orthogonal fits in 8-bit. Max value: {orth_max}")

    if depth_max > 255:
        print(f"Depth is > 8-bit. Max value: {depth_max}")
    else:
        print(f"Depth fits in 8-bit. Max value: {depth_max}")


if __name__ == "__main__":
    check_stats()
