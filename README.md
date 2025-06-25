# Multi-Stream Neural Networks

A PyTorch implementation of Multi-Stream Neural Networks for processing color and brightness data with separate pathways, inspired by biological visual processing.

## Features

- **Multi-Channel Architecture**: Separate processing streams for color (RGB) and brightness data
- **ResNet & Dense Models**: Both convolutional (ResNet-based) and dense network architectures
- **Efficient Channel Handling**: Optimized for different input channels (3 for color, 1 for brightness)
- **Modular Design**: Easy to extend and customize
- **Comprehensive Testing**: End-to-end tests with real datasets (MNIST, CIFAR-100)

## Quick Start

```python
from src.models.builders.model_factory import create_model
from src.transforms.rgb_to_rgbl import RGBtoRGBL

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

# Forward pass
outputs = model.forward_combined(color_data, brightness_data)
```

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