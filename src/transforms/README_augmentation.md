# Data Augmentation for Multi-Stream Neural Networks

## Module Organization

All augmentation functionality is now consolidated in a single module:

- `augmentation.py`: Main implementation with dataset-specific augmentations
- `rgb_to_rgbl.py`: RGB to brightness conversion
- `spatial_transforms.py`: Spatial transformations

## Key Components

### Modern API

- `BaseAugmentation`: Base class for all augmentations with common functionality
- `CIFAR100Augmentation`: Specialized augmentation for CIFAR-100 dataset
- `ImageNetAugmentation`: Specialized augmentation for ImageNet dataset
- `MixUp`: Implementation of MixUp augmentation for multi-stream data
- `AugmentedMultiStreamDataset`: Dataset class that applies augmentations on-the-fly
- Helper functions: `create_augmented_dataloaders`, `create_test_dataloader`

### Deprecated Classes (For Backward Compatibility)

The following classes are maintained for backward compatibility but will be removed in a future release:

- `ColorJitter`: Use `BaseAugmentation` with `color_jitter_strength` parameter instead
- `BrightnessNoise`: Use `BaseAugmentation` with `gaussian_noise_std` parameter instead
- `MultiStreamAugmentation`: Use `BaseAugmentation` or dataset-specific classes instead

## Usage

```python
# Import from the correct location
from src.transforms.augmentation import (
    CIFAR100Augmentation,
    ImageNetAugmentation,
    create_augmented_dataloaders
)

# Create augmentation with dataset-specific defaults
augmentation = CIFAR100Augmentation(
    horizontal_flip_prob=0.5,
    rotation_degrees=10.0,
    color_jitter_strength=0.3
)

# Create dataloaders with augmentation
train_loader, val_loader = create_augmented_dataloaders(
    train_color, train_brightness, train_labels,
    val_color, val_brightness, val_labels,
    batch_size=64,
    dataset="cifar100"
)
```
