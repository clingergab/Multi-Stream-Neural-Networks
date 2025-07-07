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
    
    if len(class_counts) == 0:
        return torch.zeros(0)
    
    # Get the maximum class index to determine tensor size
    max_class_idx = max(class_counts.keys())
    num_classes = len(class_counts)
    total_samples = sum(class_counts.values())
    
    # Create weights tensor with size based on max class index + 1
    weights = torch.zeros(max_class_idx + 1)
    
    # Calculate inverse frequency weights only for existing classes
    for class_idx, count in class_counts.items():
        weights[class_idx] = total_samples / (num_classes * count)
    
    return weights
