
import torch
from src.data_utils.sunrgbd_3stream_dataset import SUNRGBD3StreamDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

def check_stats():
    # Initialize dataset
    # Note: We use train=True to get the training distribution
    dataset = SUNRGBD3StreamDataset(train=True)
    loader = DataLoader(dataset, batch_size=32, num_workers=0, shuffle=False)
    
    print("Computing Orthogonal stream statistics (on current [-1, 1] normalization)...")
    
    all_values = []
    
    # Check first 20 batches (~320 images) to get a good estimate
    for i, batch in enumerate(tqdm(loader)):
        orth = batch[2] # [B, 1, H, W]
        
        # Subsample pixels to avoid massive memory usage
        # Take every 10th pixel
        vals = orth[:, :, ::10, ::10].flatten()
        all_values.append(vals.numpy())
        
        if i >= 20:
            break
            
    all_values = np.concatenate(all_values)
    
    mean = np.mean(all_values)
    std = np.std(all_values)
    min_val = np.min(all_values)
    max_val = np.max(all_values)
    
    print(f"\nStatistics of current Orthogonal stream:")
    print(f"  Mean: {mean:.4f}")
    print(f"  Std:  {std:.4f}")
    print(f"  Min:  {min_val:.4f}")
    print(f"  Max:  {max_val:.4f}")
    
    print(f"\nTo standardize (Mean=0, Std=1), we should divide by {std:.4f}")

if __name__ == "__main__":
    check_stats()
