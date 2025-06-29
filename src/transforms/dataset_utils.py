"""Dataset utilities for multi-stream processing."""

import torch
from tqdm import tqdm
from typing import Tuple, Any

from .rgb_to_rgbl import RGBtoRGBL


def process_dataset_to_streams(dataset: Any, batch_size: int = 1000, desc: str = "Processing") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert RGB dataset to RGB + Brightness streams efficiently.
    
    This function processes a PyTorch dataset in batches to manage memory usage,
    applying the RGB to RGB+L transform to create separate color and brightness streams.
    
    Args:
        dataset: Dataset with RGB images (PyTorch dataset format)
        batch_size: Size of batches for memory-efficient processing
        desc: Description for progress bar
        
    Returns:
        Tuple of (rgb_stream, brightness_stream, labels_tensor)
            - rgb_stream: Tensor containing all RGB data [N, 3, H, W]
            - brightness_stream: Tensor containing all brightness data [N, 1, H, W]
            - labels_tensor: Tensor containing all labels [N]
            
    Note:
        This function processes the entire dataset upfront. For very large datasets,
        consider using the `create_dataloader_with_streams` function instead, which
        performs the RGB→L transformation on-the-fly during training, significantly
        reducing memory requirements.
        
    TODO:
        For future development, consider replacing all usages of this function with
        the more memory-efficient `create_dataloader_with_streams` approach for better
        scalability with large datasets.
    """
    rgb_tensors = []
    brightness_tensors = []
    labels = []
    
    # Initialize RGB to brightness transform
    rgb_to_rgbl = RGBtoRGBL()
    
    # Process in batches to manage memory
    for i in tqdm(range(0, len(dataset), batch_size), desc=desc):
        batch_end = min(i + batch_size, len(dataset))
        batch_data = []
        batch_labels = []
        
        # Collect batch data
        for j in range(i, batch_end):
            data, label = dataset[j]
            batch_data.append(data)
            batch_labels.append(label)
        
        # Convert to tensor batch
        batch_tensor = torch.stack(batch_data)
        
        # Apply RGB to RGB+L transform
        rgb_batch, brightness_batch = rgb_to_rgbl(batch_tensor)
        
        rgb_tensors.append(rgb_batch)
        brightness_tensors.append(brightness_batch)
        labels.extend(batch_labels)
    
    # Concatenate all batches
    rgb_stream = torch.cat(rgb_tensors, dim=0)
    brightness_stream = torch.cat(brightness_tensors, dim=0)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    return rgb_stream, brightness_stream, labels_tensor


def collate_with_streams(batch):
    """
    Custom collate function that extracts RGB images and applies RGB to RGB+L transform.
    
    Args:
        batch: A list of tuples (data, label)
        
    Returns:
        Tuple of (rgb_tensor, brightness_tensor, labels_tensor)
    """
    from .rgb_to_rgbl import RGBtoRGBL
    
    # Initialize RGB to brightness transform
    rgb_to_rgbl = RGBtoRGBL()
    
    data = [item[0] for item in batch]  # Extract images
    labels = [item[1] for item in batch]  # Extract labels
    
    # Stack images into a batch tensor
    data_tensor = torch.stack(data)
    
    # Apply RGB to RGB+L transform
    rgb_tensor, brightness_tensor = rgb_to_rgbl(data_tensor)
    
    # Convert labels to tensor
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    return rgb_tensor, brightness_tensor, labels_tensor


def create_dataloader_with_streams(dataset: Any, batch_size: int = 32, shuffle: bool = True, 
                                num_workers: int = 4, pin_memory: bool = True):
    """
    Create a DataLoader that converts RGB data to RGB+Brightness streams on-the-fly.
    
    This is a more memory-efficient approach for large datasets compared to pre-processing
    the entire dataset. The transformation happens during training as batches are loaded.
    
    Args:
        dataset: PyTorch dataset with RGB images
        batch_size: Batch size for the DataLoader
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        DataLoader that yields (rgb_batch, brightness_batch, labels) tuples
    
    Note:
        This is the recommended approach for large datasets as it performs
        the conversion on-the-fly rather than storing all data in memory.
    """
    from torch.utils.data import DataLoader
    
    # Create DataLoader with custom collate function
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_with_streams
    )
    
    return loader
