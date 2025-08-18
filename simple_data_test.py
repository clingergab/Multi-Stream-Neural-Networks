#!/usr/bin/env python3
"""
Simple data pipeline benchmark - no model complexity.
Just test data loading speed.
"""

import torch
import time
import os
from src.data_utils.streaming_dual_channel_dataset import create_imagenet_train_val_dataloaders

def test_data_loading():
    """Test data loading speed only."""
    print("Testing data loading speed...")
    
    # ImageNet paths
    train_folder = "/Users/gclinger/Documents/projects/Multi-Stream-Neural-Networks/data/ImageNet-1K/train_images_0"
    val_folder = "/Users/gclinger/Documents/projects/Multi-Stream-Neural-Networks/data/ImageNet-1K/val_images"
    truth_file = "/Users/gclinger/Documents/projects/Multi-Stream-Neural-Networks/data/ImageNet-1K/ILSVRC2012_validation_ground_truth.txt"
    
    # Check if files exist
    if not os.path.exists(train_folder):
        print(f"Train folder not found: {train_folder}")
        return
    
    # Create dataloaders with different configurations
    configs = [
        {"batch_size": 32, "num_workers": 0, "name": "Single threaded"},
        {"batch_size": 32, "num_workers": 4, "name": "4 workers"},
        {"batch_size": 32, "num_workers": 8, "name": "8 workers"},
        {"batch_size": 64, "num_workers": 8, "name": "Batch 64, 8 workers"},
    ]
    
    for config in configs:
        print(f"\n--- Testing: {config['name']} ---")
        
        try:
            train_loader, _ = create_imagenet_train_val_dataloaders(
                train_folders=train_folder,
                val_folder=val_folder,
                truth_file=truth_file,
                batch_size=config["batch_size"],
                num_workers=config["num_workers"],
                pin_memory=True,
                persistent_workers=config["num_workers"] > 0
            )
            
            # Time first few batches
            start_time = time.time()
            batch_count = 0
            max_batches = 10
            
            for batch in train_loader:
                rgb_tensor, labels = batch
                batch_count += 1
                
                if batch_count >= max_batches:
                    break
            
            end_time = time.time()
            total_time = end_time - start_time
            time_per_batch = total_time / batch_count
            
            print(f"  Batches processed: {batch_count}")
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Time per batch: {time_per_batch:.3f}s")
            print(f"  Batches per minute: {60 / time_per_batch:.1f}")
            
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    test_data_loading()
