
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def check_orth_stats(data_root):
    orth_dir = os.path.join(data_root, 'train', 'orth')
    if not os.path.exists(orth_dir):
        print(f"Orth dir not found: {orth_dir}")
        return

    files = list(os.listdir(orth_dir))[:100] # Check first 100 files
    
    min_val = float('inf')
    max_val = float('-inf')
    pixel_sum = 0
    count = 0
    
    print(f"Checking stats for {len(files)} orthogonal images (16-bit check)...")
    
    for f in tqdm(files):
        path = os.path.join(orth_dir, f)
        try:
            img = Image.open(path)
            img_np = np.array(img).astype(np.float32)
            
            min_val = min(min_val, img_np.min())
            max_val = max(max_val, img_np.max())
            pixel_sum += img_np.sum()
            count += img_np.size
        except Exception as e:
            print(f"Error reading {f}: {e}")

    mean = pixel_sum / count
    
    print(f"Orthogonal Stream Stats (Train) - Raw Values:")
    print(f"  Min:  {min_val}")
    print(f"  Max:  {max_val}")
    print(f"  Mean: {mean:.4f}")
    
    # Check what happens with the dataset loader's normalization logic
    print("\nSimulating Dataset Loader Normalization (per-image max norm):")
    simulated_means = []
    for f in files[:10]:
        path = os.path.join(orth_dir, f)
        img = Image.open(path)
        img_np = np.array(img).astype(np.float32)
        if img_np.max() > 0:
            norm_img = (img_np / img_np.max() * 255).astype(np.uint8)
            simulated_means.append(norm_img.mean())
    
    print(f"  Avg Mean after loader norm (0-255): {np.mean(simulated_means):.4f}")
    print(f"  Expected Mean (if centered): ~127.5")

if __name__ == "__main__":
    check_orth_stats("data/sunrgbd_15")
