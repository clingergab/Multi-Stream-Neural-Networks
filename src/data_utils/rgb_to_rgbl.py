"""RGB to RGB+L (brightness) transforms."""

import torch
from torchvision import transforms
from tqdm import tqdm
from typing import Tuple, Union, Any
from torch.utils.data import Dataset


class RGBtoRGBL:
    """Transform RGB image into separate RGB and brightness streams."""
    
    def __init__(self):
        # Standard RGB to luminance weights
        self.rgb_weights = torch.tensor([0.299, 0.587, 0.114])
    
    def __call__(self, rgb_tensor):
        """
        Convert RGB tensor to separate RGB and brightness streams.
        
        Args:
            rgb_tensor: RGB tensor of shape [C, H, W] or [B, C, H, W] where C=3
            
        Returns:
            tuple: (rgb_tensor, brightness_tensor)
                - rgb_tensor: Original RGB tensor [C, H, W] or [B, C, H, W]
                - brightness_tensor: Luminance tensor [1, H, W] or [B, 1, H, W]
        """
        # Handle both single images and batches
        if rgb_tensor.dim() == 3:
            # Single image: [C, H, W]
            if rgb_tensor.shape[0] != 3:
                raise ValueError(f"Expected 3 channels, got {rgb_tensor.shape[0]}")
            
            # Move weights to same device as input
            weights = self.rgb_weights.to(rgb_tensor.device)
            # Calculate luminance channel
            brightness = torch.sum(rgb_tensor * weights.view(-1, 1, 1), dim=0, keepdim=True)
            
        elif rgb_tensor.dim() == 4:
            # Batch: [B, C, H, W]
            if rgb_tensor.shape[1] != 3:
                raise ValueError(f"Expected 3 channels, got {rgb_tensor.shape[1]}")
            
            # Move weights to same device as input
            weights = self.rgb_weights.to(rgb_tensor.device)
            # Calculate luminance channel for batch
            brightness = torch.sum(rgb_tensor * weights.view(1, -1, 1, 1), dim=1, keepdim=True)
            
        else:
            raise ValueError(f"Expected tensor with 3 or 4 dimensions, got {rgb_tensor.dim()}")
        
        # Return both streams
        return rgb_tensor, brightness

    def get_brightness(self, rgb_tensor):
        """
        Extract only the brightness/luminance channel from RGB tensor.
        
        More efficient than __call__ when you only need brightness and already have RGB data.
        
        Args:
            rgb_tensor: RGB tensor of shape [C, H, W] or [B, C, H, W] where C=3
            
        Returns:
            brightness_tensor: Luminance tensor [1, H, W] or [B, 1, H, W]
        """
        # Handle both single images and batches
        if rgb_tensor.dim() == 3:
            # Single image: [C, H, W]
            if rgb_tensor.shape[0] != 3:
                raise ValueError(f"Expected 3 channels, got {rgb_tensor.shape[0]}")
            
            # Move weights to same device as input
            weights = self.rgb_weights.to(rgb_tensor.device)
            # Calculate luminance channel
            brightness = torch.sum(rgb_tensor * weights.view(-1, 1, 1), dim=0, keepdim=True)
            
        elif rgb_tensor.dim() == 4:
            # Batch: [B, C, H, W]
            if rgb_tensor.shape[1] != 3:
                raise ValueError(f"Expected 3 channels, got {rgb_tensor.shape[1]}")
            
            # Move weights to same device as input
            weights = self.rgb_weights.to(rgb_tensor.device)
            # Calculate luminance channel for batch
            brightness = torch.sum(rgb_tensor * weights.view(1, -1, 1, 1), dim=1, keepdim=True)
            
        else:
            raise ValueError(f"Expected tensor with 3 or 4 dimensions, got {rgb_tensor.dim()}")
            
        return brightness

    def get_rgbl(self, rgb_tensor):
        """
        Get combined RGBL tensor.
        
        Args:
            rgb_tensor: RGB tensor of shape [C, H, W] or [B, C, H, W] where C=3
            
        Returns:
            RGBL tensor of shape [4, H, W] or [B, 4, H, W]
        """
        rgb, brightness = self.__call__(rgb_tensor)
        return torch.cat([rgb, brightness], dim=-3)  # Concatenate along channel dimension


# Convenience function for creating transform pipelines
def create_rgbl_transform():
    """Create a transform pipeline for RGB to streams conversion."""
    return transforms.Compose([
        transforms.ToTensor(),
        RGBtoRGBL()
    ])


def collate_with_streams(batch):
    """
    Custom collate function that extracts RGB images and applies RGB to RGB+L transform.
    
    This function is designed to be used with PyTorch DataLoaders as a collate_fn.
    It handles both PIL Images and Tensors, converting them to dual-stream format.
    
    Args:
        batch: A list of tuples (data, label) from dataset
        
    Returns:
        Tuple of (rgb_tensor, brightness_tensor, labels_tensor)
            - rgb_tensor: RGB data [B, 3, H, W]
            - brightness_tensor: Brightness data [B, 1, H, W]  
            - labels_tensor: Labels [B]
            
    Example:
        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(dataset, collate_fn=collate_with_streams)
        >>> for rgb_batch, brightness_batch, labels in loader:
        ...     # Process dual-stream data
    """
    
    # Initialize RGB to brightness transform
    rgb_to_rgbl = RGBtoRGBL()
    
    data = [item[0] for item in batch]  # Extract images
    labels = [item[1] for item in batch]  # Extract labels
    
    # Convert PIL Images to tensors if needed
    to_tensor = transforms.ToTensor()
    tensor_data = []
    for item in data:
        if isinstance(item, torch.Tensor):
            tensor_data.append(item)
        else:
            # Assume PIL Image or similar, convert to tensor
            tensor_data.append(to_tensor(item))
    
    # Stack images into a batch tensor
    data_tensor = torch.stack(tensor_data)
    
    # Apply RGB to RGB+L transform
    rgb_tensor, brightness_tensor = rgb_to_rgbl(data_tensor)
    
    # Convert labels to tensor
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    return rgb_tensor, brightness_tensor, labels_tensor


def process_dataset_to_streams(dataset: Union[Dataset, Any], batch_size: int = 1000, desc: str = "Processing") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        This function processes the entire dataset upfront, which provides fast access
        during training but requires more memory. For very large datasets, consider using
        the `AugmentedMultiStreamDataset` with on-the-fly processing instead.
        
    Performance Note:
        For optimal memory usage with large datasets, consider using the high-level
        `create_augmented_dataloaders` function which provides on-the-fly processing
        and built-in augmentation capabilities.
    """
    rgb_tensors = []
    brightness_tensors = []
    labels = []
    
    # Validate inputs
    if len(dataset) == 0:
        raise ValueError("Dataset is empty")
    
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")
    
    # Initialize RGB to brightness transform
    rgb_to_rgbl = RGBtoRGBL()
    
    # Process in batches to manage memory
    try:
        for i in tqdm(range(0, len(dataset), batch_size), desc=desc):
            batch_end = min(i + batch_size, len(dataset))
            batch_data = []
            batch_labels = []
            
            # Collect batch data
            for j in range(i, batch_end):
                try:
                    data, label = dataset[j]
                    batch_data.append(data)
                    batch_labels.append(label)
                except Exception as e:
                    raise RuntimeError(f"Error accessing dataset item {j}: {e}")
            
            # Convert to tensor batch
            try:
                batch_tensor = torch.stack(batch_data)
            except Exception as e:
                raise RuntimeError(f"Error stacking batch data: {e}. Ensure all images have the same shape.")
            
            # Apply RGB to RGB+L transform
            try:
                rgb_batch, brightness_batch = rgb_to_rgbl(batch_tensor)
            except Exception as e:
                raise RuntimeError(f"Error applying RGB to RGBL transform: {e}")
            
            rgb_tensors.append(rgb_batch)
            brightness_tensors.append(brightness_batch)
            labels.extend(batch_labels)
            
    except Exception as e:
        raise RuntimeError(f"Error processing dataset: {e}")
    
    # Concatenate all batches
    try:
        rgb_stream = torch.cat(rgb_tensors, dim=0)
        brightness_stream = torch.cat(brightness_tensors, dim=0)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
    except Exception as e:
        raise RuntimeError(f"Error concatenating results: {e}")
    
    return rgb_stream, brightness_stream, labels_tensor
