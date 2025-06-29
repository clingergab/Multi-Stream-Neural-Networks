"""Transforms package."""

from .rgb_to_rgbl import RGBtoRGBL, AdaptiveBrightnessExtraction, create_rgbl_transform
from .spatial_transforms import SpatialConsistentTransform, RandomCropBoth, RandomFlipBoth, create_spatial_transforms
from .augmentation import BaseAugmentation, CIFAR100Augmentation, ImageNetAugmentation, MixUp
from .augmentation import AugmentedMultiStreamDataset, create_augmented_dataloaders, create_test_dataloader
from .augmentation import ColorJitter, BrightnessNoise, MultiStreamAugmentation
from .dataset_utils import process_dataset_to_streams, create_dataloader_with_streams, collate_with_streams

__all__ = [
    # Core transforms
    'RGBtoRGBL',
    'AdaptiveBrightnessExtraction',
    'create_rgbl_transform',
    'SpatialConsistentTransform',
    'RandomCropBoth',
    'RandomFlipBoth', 
    'create_spatial_transforms',
    
    # Augmentation - New API
    'BaseAugmentation',
    'CIFAR100Augmentation',
    'ImageNetAugmentation',
    'AugmentedMultiStreamDataset',
    'create_augmented_dataloaders',
    'create_test_dataloader',
    
    # Dataset utilities
    'process_dataset_to_streams',
    'create_dataloader_with_streams',
    'collate_with_streams',
    
    # Deprecated - For backward compatibility
    'ColorJitter',
    'BrightnessNoise',
    'MultiStreamAugmentation',
    'MixUp'
]
