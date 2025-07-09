"""
Dual-channel dataset implementation for multi-stream neural networks.

This module provides a clean, idiomatic PyTorch Dataset implementation that handles
dual-stream data (RGB + brightness) with consistent augmentation across both channels.
"""

from pyparsing import col
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
import random
from typing import Optional, Tuple, Callable, Any
from .rgb_to_rgbl import RGBtoRGBL


class DualChannelDataset(Dataset):
    """
    A PyTorch Dataset that provides dual-stream data (RGB + brightness) with consistent augmentation.
    
    This is the recommended approach for multi-stream data loading, replacing custom collate functions
    and complex dataloader logic with a simple, standard PyTorch Dataset pattern.
    
    Features:
    - Consistent augmentation across RGB and brightness channels using shared random seeds
    - Standard PyTorch Dataset interface - works with any DataLoader
    - Optional brightness conversion from RGB (if brightness_data not provided)
    - Configurable augmentation pipeline
    - Memory efficient - applies transforms on-the-fly
    """
    
    def __init__(
        self,
        rgb_data: torch.Tensor,
        labels: torch.Tensor,
        brightness_data: Optional[torch.Tensor] = None,
        transform: Optional[Callable] = None
    ):
        """
        Initialize the dual-channel dataset.
        
        Args:
            rgb_data: RGB image tensor of shape [N, C, H, W]
            labels: Label tensor of shape [N]
            brightness_data: Optional brightness tensor of shape [N, 1, H, W].
                           If None, computed from RGB on-the-fly.
            transform: Transform applied to both RGB and brightness channels with 
                      guaranteed synchronization using the same random seed
        """
        self.rgb_data = rgb_data
        self.labels = labels
        self.transform = transform
        self.brightness_data = brightness_data if brightness_data is not None else None
        
        # Validate shapes
        if len(self.rgb_data) != len(self.labels):
            raise ValueError(f"RGB data length {len(self.rgb_data)} != labels length {len(self.labels)}")
        
        if brightness_data is not None and len(brightness_data) != len(self.labels):
            raise ValueError(f"Brightness data length {len(brightness_data)} != labels length {len(self.labels)}")
    
    def __len__(self) -> int:
        return len(self.rgb_data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Tuple of (rgb_tensor, brightness_tensor, label)
        """
        rgb = self.rgb_data[idx]
        label = self.labels[idx]
        
        # Get or compute brightness
        if self.brightness_data is not None:
            brightness = self.brightness_data[idx]
        else:
            # Create converter instance once and reuse
            if not hasattr(self, '_rgb_converter'):
                self._rgb_converter = RGBtoRGBL()
            # Compute brightness from RGB on-the-fly
            brightness = self._rgb_converter.get_brightness(rgb)
        
        # Apply transforms with consistent random seed for synchronized augmentation
        if self.transform:
            # Set random seed for consistent augmentation
            seed = random.randint(0, 2**32 - 1)
            
            # Apply to RGB
            torch.manual_seed(seed)
            random.seed(seed)
            rgb = self.transform(rgb)
            
            # Apply to brightness with same seed for guaranteed synchronization
            torch.manual_seed(seed)
            random.seed(seed)
            brightness = self.transform(brightness)
        
        return rgb, brightness, label


# Convenience function for easy setup
def create_dual_channel_dataloaders(
    train_rgb: torch.Tensor,
    train_brightness: torch.Tensor,
    train_labels: torch.Tensor,
    val_rgb: torch.Tensor,
    val_brightness: torch.Tensor,
    val_labels: torch.Tensor,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    batch_size: int = 32,
    num_workers: int = 2,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int = 2
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Convenience function to create train and validation dataloaders.
    
    Args:
        train_rgb: Training RGB data
        train_brightness: Training brightness data
        train_labels: Training labels
        val_rgb: Validation RGB data  
        val_brightness: Validation brightness data
        val_labels: Validation labels
        train_transform: Transform to apply to training data (both RGB and brightness)
        val_transform: Transform to apply to validation data (both RGB and brightness)
        batch_size: Batch size for training (validation will use batch_size*2)
        num_workers: Number of parallel data loading workers (CPU cores)
        pin_memory: Whether to pin memory for faster CPU→GPU transfer
        persistent_workers: Whether to keep workers alive between epochs
        prefetch_factor: Number of batches to prefetch per worker
    
    Returns:
        Tuple of (train_dataloader, val_dataloader). Validation dataloader uses 2x batch size.
    """
    from torch.utils.data import DataLoader
    
    # Create datasets
    train_dataset = DualChannelDataset(
        train_rgb, train_labels, train_brightness,
        transform=train_transform
    )
    
    val_dataset = DualChannelDataset(
        val_rgb, val_labels, val_brightness,
        transform=val_transform
    )
    
    # Create dataloaders with GPU optimizations
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size * 2,  # Use larger batch size for validation (no gradients needed)
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    
    return train_loader, val_loader

# Convenience function for easy setup
def create_dual_channel_dataloader(
    rgb_data: torch.Tensor,
    brightness_data: torch.Tensor,
    labels: torch.Tensor,
    transform: Optional[Callable] = None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 2,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int = 2
) -> torch.utils.data.DataLoader:
    """
    Convenience function to create a single dual-channel dataloader.
    
    Args:
        rgb_data: RGB input data
        brightness_data: Brightness input data
        labels: Target labels
        transform: Transform to apply to data (both RGB and brightness)
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        num_workers: Number of parallel data loading workers (CPU cores)
        pin_memory: Whether to pin memory for faster CPU→GPU transfer
        persistent_workers: Whether to keep workers alive between epochs
        prefetch_factor: Number of batches to prefetch per worker
    
    Returns:
        DataLoader for dual-channel data
    """
    from torch.utils.data import DataLoader
    
    # Create dataset
    dataset = DualChannelDataset(
        rgb_data, labels, brightness_data,
        transform=transform
    )
    
    # Create dataloader with GPU optimizations
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers, 
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    
    return dataloader