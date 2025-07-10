import torch
import random
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from torchvision import transforms
from typing import Optional, Callable, Tuple
from PIL import Image

class StreamingDualChannelDataset(Dataset):
    """
    A PyTorch Dataset that provides dual-stream data (RGB + brightness) with consistent augmentation,
    streaming from ImageNet without requiring full dataset download.
    
    Features:
    - Streams ImageNet data on-demand using Hugging Face datasets
    - Consistent augmentation across RGB and brightness channels using shared random seeds
    - Standard PyTorch Dataset interface - works with any DataLoader
    - Brightness conversion from RGB on-the-fly
    - Memory efficient - no pre-loading required
    """
    
    def __init__(
        self,
        split: str = "train",
        transform: Optional[Callable] = None,
        streaming: bool = True,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the streaming dual-channel dataset.
        
        Args:
            split: Dataset split ("train", "validation", "test")
            transform: Transform applied to both RGB and brightness channels with 
                      guaranteed synchronization using the same random seed
            streaming: Whether to use streaming mode (recommended for ImageNet)
            cache_dir: Directory to cache downloaded data
        """
        self.transform = transform
        self.split = split
        
        # Load ImageNet dataset with streaming
        print(f"Loading ImageNet {split} split with streaming={streaming}...")
        self.dataset = load_dataset(
            "imagenet-1k", 
            split=split, 
            streaming=streaming,
            cache_dir=cache_dir
        )
        
        # For streaming datasets, we need to handle iteration differently
        self.streaming = streaming
        if streaming:
            # Convert to iterable for consistent access
            self._dataset_iter = iter(self.dataset)
            self._cache = {}  # Simple cache for accessed items
        
        # Create brightness converter
        if not hasattr(self, '_rgb_converter'):
            # Assuming you have this class - replace with your actual implementation
            self._rgb_converter = RGBtoRGBL()
    
    def __len__(self) -> int:
        """
        Return dataset length. For streaming datasets, this is approximate.
        """
        if self.streaming:
            # ImageNet-1k approximate sizes
            sizes = {
                "train": 1281167,
                "validation": 50000,
                "test": 100000
            }
            return sizes.get(self.split, 1000000)  # fallback estimate
        else:
            return len(self.dataset)
    
    def _get_sample(self, idx: int) -> dict:
        """
        Get a sample from the dataset, handling both streaming and non-streaming modes.
        """
        if self.streaming:
            # For streaming, we need to handle random access differently
            if idx in self._cache:
                return self._cache[idx]
            
            # This is a limitation of streaming - we can't truly do random access
            # In practice, you'd typically iterate sequentially or use a different approach
            # For now, we'll iterate through the dataset
            try:
                # Reset iterator if needed (not ideal, but works for demo)
                if not hasattr(self, '_current_idx') or self._current_idx > idx:
                    self._dataset_iter = iter(self.dataset)
                    self._current_idx = 0
                
                # Iterate to the desired index
                while self._current_idx <= idx:
                    sample = next(self._dataset_iter)
                    if self._current_idx == idx:
                        self._cache[idx] = sample
                        return sample
                    self._current_idx += 1
                    
            except StopIteration:
                raise IndexError(f"Index {idx} out of range")
        else:
            return self.dataset[idx]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Tuple of (rgb_tensor, brightness_tensor, label)
        """
        # Get sample from dataset
        sample = self._get_sample(idx)
        
        # Extract image and label
        image = sample['image']  # PIL Image
        label = torch.tensor(sample['label'], dtype=torch.long)
        
        # Convert PIL image to tensor
        if isinstance(image, Image.Image):
            rgb = transforms.ToTensor()(image)
        else:
            rgb = image  # assume already tensor
        
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
    
def create_streaming_dual_channel_train_val_dataloaders(
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    cache_dir: Optional[str] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders for streaming ImageNet.
    
    Args:
        train_transform: Transform to apply to training data
        val_transform: Transform to apply to validation data  
        batch_size: Batch size for training
        num_workers: Number of parallel data loading workers
        pin_memory: Whether to pin memory for faster CPU→GPU transfer
        persistent_workers: Whether to keep workers alive between epochs
        prefetch_factor: Number of batches to prefetch per worker
        cache_dir: Directory to cache downloaded data
    
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    
    # Create streaming datasets
    train_dataset = StreamingDualChannelDataset(
        split="train",
        transform=train_transform,
        streaming=True,
        cache_dir=cache_dir
    )
    
    val_dataset = StreamingDualChannelDataset(
        split="validation",
        transform=val_transform,
        streaming=True,
        cache_dir=cache_dir
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Note: shuffle with streaming has limitations
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  # Larger batch for validation
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    
    return train_loader, val_loader

def create_streaming_dual_channel_test_dataloader(
    test_transform: Optional[Callable] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    cache_dir: Optional[str] = None
) -> DataLoader:
    """
    Create test dataloader for streaming ImageNet.

    Args:
        test_transform: Transform to apply to test data
        batch_size: Batch size for training
        num_workers: Number of parallel data loading workers
        pin_memory: Whether to pin memory for faster CPU→GPU transfer
        persistent_workers: Whether to keep workers alive between epochs
        prefetch_factor: Number of batches to prefetch per worker
        cache_dir: Directory to cache downloaded data
    
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    
    # Create streaming datasets
    test_dataset = StreamingDualChannelDataset(
        split="test",
        transform=test_transform,
        streaming=True,
        cache_dir=cache_dir
    )
    

    
    # Create dataloaders
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )

    return test_loader