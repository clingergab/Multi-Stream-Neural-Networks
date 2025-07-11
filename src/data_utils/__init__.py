"""Data utilities package - consolidated from transforms and data_utils."""

from torch.utils.data import DataLoader

# Modern dual-channel dataset (recommended approach)
from .dual_channel_dataset import (
    DualChannelDataset,
    create_dual_channel_dataloaders,
    create_dual_channel_dataloader
)

# Streaming dual-channel dataset for ImageNet (large datasets)
from .streaming_dual_channel_dataset import (
    StreamingDualChannelDataset,
    create_imagenet_dual_channel_train_val_dataloaders,
    create_imagenet_dual_channel_test_dataloader,
    create_default_imagenet_transforms
)

# Collate function for multi-stream data (legacy support)
from .rgb_to_rgbl import (
    collate_with_streams
)

# Data helpers and utilities
from .data_helpers import (
    calculate_dataset_stats, 
    get_class_weights
)

# Dataset utilities and CIFAR-100 loading
from .dataset_utils import (
    load_cifar100_data,
    CIFAR100_FINE_LABELS
)

# Transform utilities and dataset processing
from .rgb_to_rgbl import (
    RGBtoRGBL, 
    create_rgbl_transform,
    process_dataset_to_streams
)

# Export all imports
__all__ = [
    # Modern dual-channel dataset
    'DualChannelDataset',
    'create_dual_channel_dataloaders',
    'create_dual_channel_dataloader',

    # Streaming dual-channel dataset for ImageNet
    'StreamingDualChannelDataset',
    'create_imagenet_dual_channel_train_val_dataloaders',
    'create_imagenet_dual_channel_test_dataloader', 
    'create_default_imagenet_transforms',

    # Legacy support
    'collate_with_streams',
    
    # Data helpers
    'calculate_dataset_stats',
    'get_class_weights',
    
    # Dataset utilities and CIFAR-100 loading
    'load_cifar100_data',
    'CIFAR100_FINE_LABELS',
    
    # Transform utilities and dataset processing
    'process_dataset_to_streams',
    'RGBtoRGBL',
    'create_rgbl_transform'
]
