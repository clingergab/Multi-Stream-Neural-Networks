# Multi-Stream Neural Networks

A PyTorch implementation of Multi-Stream Neural Networks for processing color and brightness data with separate pathways, inspired by biological visual processing.

## Features

- **Multi-Channel Architecture**: Separate processing streams for color (RGB) and brightness data
- **ResNet & Dense Models**: Both convolutional (ResNet-based) and dense network architectures
- **Efficient Channel Handling**: Optimized for different input channels (3 for color, 1 for brightness)
- **Dataset-Agnostic Augmentation**: Comprehensive augmentation pipeline for CIFAR-100, ImageNet and custom datasets
- **Modular Design**: Easy to extend and customize
- **Comprehensive Testing**: End-to-end tests with real datasets (MNIST, CIFAR-100)

## Quick Start

```python
from src.models.builders.model_factory import create_model
from src.transforms.rgb_to_rgbl import RGBtoRGBL
from src.transforms.augmentation import create_augmented_dataloaders

# Create a dense multi-channel model
dense_model = create_model(
    'base_multi_channel',
    color_input_size=28*28*3,
    brightness_input_size=28*28,
    hidden_sizes=[128, 64, 32],
    num_classes=10
)

# Create a ResNet multi-channel model
resnet_model = create_model(
    'multi_channel_resnet18',
    num_classes=10,
    color_input_channels=3,
    brightness_input_channels=1
)

# Transform RGB data to RGB + Brightness
transform = RGBtoRGBL()
color_data, brightness_data = transform(rgb_tensor)

# Create augmented dataloaders
train_loader, val_loader = create_augmented_dataloaders(
    train_color, train_brightness, train_labels,
    val_color, val_brightness, val_labels,
    batch_size=64,
    dataset="cifar100",
    augmentation_config={'horizontal_flip_prob': 0.5},
    mixup_alpha=0.2  # Enable MixUp augmentation
)

# Forward pass
outputs = model.forward_combined(color_data, brightness_data)
```

## CIFAR-100 Data Loading

The project includes robust CIFAR-100 data loading utilities that work directly with pickle files:

```python
from src.data_utils.dataset_utils import get_cifar100_datasets, CIFAR100_FINE_LABELS

# Load CIFAR-100 datasets
train_dataset, test_dataset, class_names = get_cifar100_datasets()

# Access raw data for processing
train_data = train_dataset.data    # [50000, 3, 32, 32]
train_labels = train_dataset.labels # [50000]

# No torchvision naming conventions required!
# Works directly with data/cifar-100/ pickle files
```

### Key Features
- **Direct pickle loading**: No torchvision folder naming requirements
- **Multiple return formats**: Tensors, numpy arrays, or dataset objects
- **Simple dataset wrapper**: PyTorch-compatible for training loops
- **Full integration**: Works seamlessly with RGBtoRGBL processor

## Installation

```bash
git clone https://github.com/your-username/Multi-Stream-Neural-Networks.git
cd Multi-Stream-Neural-Networks
pip install -e .
```

## Project Structure

```
Multi-Stream-Neural-Networks/
├── src/                          # Source code
│   ├── models/                   # Model architectures
│   │   ├── basic_multi_channel/  # Core multi-channel models
│   │   ├── layers/              # Custom layers and blocks
│   │   └── builders/            # Model factory and registry
│   ├── transforms/              # Data transforms (RGBtoRGBL)
│   └── utils/                   # Utilities and registry
├── tests/                       # All test suites
│   ├── end_to_end/             # End-to-end integration tests
│   └── archived_tests/         # Legacy tests
├── scripts/                     # Utility scripts
│   ├── analysis/               # Model analysis tools
│   ├── train_multi_channel.py  # Training script
│   └── download_datasets.py    # Dataset preparation
├── verification/               # Verification scripts
│   ├── verify_corrections.py   # Comprehensive verification
│   ├── final_verification.py   # Final claim verification
│   └── check_optimizer_calc.py # Optimizer memory analysis
├── docs/                       # Documentation
│   └── comparisons.md          # Architecture comparison
├── examples/                   # Usage examples
├── experiments/               # Experiment results
├── notebooks/                 # Jupyter notebooks
├── configs/                   # Configuration files
├── archive/                   # Archived files (temp/redundant)
├── DESIGN.md                  # Architecture design document
├── VERIFICATION_SUMMARY.md    # Verification report
└── README.md                  # This file
```

## Usage Examples

See the `examples/` directory and `notebooks/` for detailed usage examples and tutorials.

## Testing

Run the comprehensive end-to-end tests:

```bash
python tests/end_to_end/test_refactored_models_e2e.py
```

## Data Augmentation

The project includes a comprehensive data augmentation pipeline for multi-stream neural networks:

```python
from src.transforms.augmentation import (
    CIFAR100Augmentation,
    ImageNetAugmentation, 
    MixUp,
    create_augmented_dataloaders
)

# Create augmentation with dataset-specific defaults
cifar_augmentation = CIFAR100Augmentation(
    horizontal_flip_prob=0.5,
    rotation_degrees=10.0,
    color_jitter_strength=0.3
)

# Create dataloaders with augmentation
train_loader, val_loader = create_augmented_dataloaders(
    train_color, train_brightness, train_labels,
    val_color, val_brightness, val_labels,
    batch_size=64,
    dataset="cifar100",  # Options: "cifar100", "imagenet", "custom"
    augmentation_config={'cutout_prob': 0.3, 'cutout_size': 8},
    mixup_alpha=0.2  # Enable MixUp augmentation
)
```

Key features:
- Dataset-specific augmentations (CIFAR-100, ImageNet, extensible)
- Consistent transformations across color and brightness streams
- Advanced techniques (MixUp, cutout, color jitter)
- Memory-efficient batch processing

See `src/utils/README_augmentation.md` for detailed documentation.

## Models

### BaseMultiChannelNetwork
Dense network optimized for different input sizes:
- Color input: Flattened RGB data (e.g., 28×28×3 = 2352)
- Brightness input: Flattened brightness data (e.g., 28×28 = 784)

### MultiChannelResNetNetwork  
ResNet-based network optimized for different input channels:
- Color input: 3-channel RGB images
- Brightness input: 1-channel brightness images

## Performance

The models show different strengths based on dataset characteristics:
- **Dense models**: Excel on small datasets, parameter-efficient
- **ResNet models**: Excel on large datasets, powerful feature extraction

See `docs/reports/` for detailed performance analysis and architectural insights.

## Contributing

Contributions are welcome! Please see the documentation in `docs/` for development guidelines.

## License

This project is licensed under the MIT License - see the LICENSE file for details.