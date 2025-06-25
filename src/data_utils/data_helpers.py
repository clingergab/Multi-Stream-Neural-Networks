"""Data utility functions and helpers."""

import os
import torch
from typing import Dict, Any, Tuple


def calculate_dataset_stats(dataloader) -> Dict[str, Any]:
    """Calculate mean and std for dataset normalization."""
    color_sum = torch.zeros(3)
    brightness_sum = torch.zeros(1)
    color_sq_sum = torch.zeros(3)
    brightness_sq_sum = torch.zeros(1)
    num_samples = 0
    
    for batch in dataloader:
        color_data = batch['color']
        brightness_data = batch['brightness']
        
        batch_samples = color_data.size(0)
        num_samples += batch_samples
        
        # Sum across batch, height, width
        color_sum += color_data.sum(dim=[0, 2, 3])
        brightness_sum += brightness_data.sum(dim=[0, 2, 3])
        color_sq_sum += (color_data ** 2).sum(dim=[0, 2, 3])
        brightness_sq_sum += (brightness_data ** 2).sum(dim=[0, 2, 3])
    
    # Calculate total pixels
    total_pixels = num_samples * dataloader.dataset[0]['color'].shape[-1] * dataloader.dataset[0]['color'].shape[-2]
    
    # Calculate means and stds
    color_mean = color_sum / total_pixels
    brightness_mean = brightness_sum / total_pixels
    color_std = torch.sqrt(color_sq_sum / total_pixels - color_mean ** 2)
    brightness_std = torch.sqrt(brightness_sq_sum / total_pixels - brightness_mean ** 2)
    
    return {
        'color_mean': color_mean.tolist(),
        'color_std': color_std.tolist(),
        'brightness_mean': brightness_mean.tolist(),
        'brightness_std': brightness_std.tolist(),
        'num_samples': num_samples
    }


def get_class_weights(dataset) -> torch.Tensor:
    """Calculate class weights for imbalanced datasets."""
    class_counts = {}
    
    for i in range(len(dataset)):
        item = dataset[i]
        target = item['target']
        if isinstance(target, torch.Tensor):
            target = target.item()
        
        class_counts[target] = class_counts.get(target, 0) + 1
    
    num_classes = len(class_counts)
    total_samples = sum(class_counts.values())
    
    # Calculate inverse frequency weights
    weights = torch.zeros(num_classes)
    for class_idx, count in class_counts.items():
        weights[class_idx] = total_samples / (num_classes * count)
    
    return weights


def create_data_splits(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, 
                      random_seed=42) -> Tuple[torch.utils.data.Dataset, ...]:
    """Split dataset into train/val/test sets."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"
    
    torch.manual_seed(random_seed)
    dataset_size = len(dataset)
    indices = torch.randperm(dataset_size).tolist()
    
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    return train_dataset, val_dataset, test_dataset


def save_dataset_info(dataset_stats: Dict[str, Any], save_path: str):
    """Save dataset statistics to file."""
    import json
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(dataset_stats, f, indent=2)


def load_dataset_info(info_path: str) -> Dict[str, Any]:
    """Load dataset statistics from file."""
    import json
    
    with open(info_path, 'r') as f:
        return json.load(f)
