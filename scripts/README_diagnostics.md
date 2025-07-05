# Comprehensive Model Diagnostics

This directory contains comprehensive diagnostic tools for analyzing multi-channel neural network training performance on CIFAR-100 dataset.

## Overview

The comprehensive diagnostics system performs full training runs with extensive monitoring to identify why models don't train well and perform poorly. It analyzes:

- **Full-size models**: `base_multi_channel_large` and `multi_channel_resnet50`
- **Complete dataset**: Full CIFAR-100 training and validation sets with augmentation
- **Extended training**: 20 epochs with early stopping (patience=5)
- **Comprehensive metrics**: Gradient flow, weight magnitudes, pathway balance, dead neurons, etc.

## Files

### Main Scripts

- `comprehensive_model_diagnostics.py` - Main diagnostic system
- `run_comprehensive_diagnostics.py` - Usage example and runner script

### Utilities

- `../src/utils/early_stopping.py` - Early stopping implementation

## Usage

### Basic Usage

```bash
# Run with default settings
python scripts/comprehensive_model_diagnostics.py

# Run with custom settings
python scripts/comprehensive_model_diagnostics.py \
    --data-dir data/cifar-100 \
    --output-dir results/diagnostics \
    --batch-size 32 \
    --epochs 20 \
    --early-stopping-patience 5
```

### Using the Example Script

```bash
python scripts/run_comprehensive_diagnostics.py
```

### Programmatic Usage

```python
from scripts.comprehensive_model_diagnostics import ComprehensiveModelDiagnostics

# Create diagnostics instance
diagnostics = ComprehensiveModelDiagnostics(
    output_dir="results/diagnostics",
    device="auto"
)

# Run comprehensive analysis
diagnostics.run_comprehensive_diagnostics(
    data_dir="data/cifar-100",
    batch_size=32,
    epochs=20,
    early_stopping_patience=5
)
```

## Models Analyzed

### 1. base_multi_channel_large
- **Architecture**: Dense multi-channel network with large hidden layers
- **Input**: Flattened CIFAR-100 images (RGB: 3072, Brightness: 1024)
- **Features**: 4 hidden layers [1024, 512, 256, 128]
- **Fusion**: Shared classifier with feature concatenation

### 2. multi_channel_resnet50
- **Architecture**: ResNet-50 with dual-stream processing
- **Input**: CIFAR-100 images (RGB: 3×32×32, Brightness: 1×32×32)
- **Features**: Full ResNet-50 architecture with multi-channel processing
- **Fusion**: Shared classifier with feature concatenation

## Data Processing

### CIFAR-100 Dataset
- **Training**: ~45,000 samples (with 10% validation split)
- **Validation**: ~5,000 samples
- **Classes**: 100 classes
- **Augmentation**: Enabled (random crops, flips, etc.)

### RGB to RGBL Transformation
- **RGB Stream**: Original 3-channel color information
- **Brightness Stream**: Luminance channel (L) from RGB→LAB conversion
- **Purpose**: Dual-stream processing for color and brightness pathways

## Output Files

The diagnostic system generates comprehensive output files:

### Training Analysis
- `*_training_curves_*.png` - Loss, accuracy, learning rate curves
- `*_gradient_flow_*.png` - Gradient flow analysis
- `*_weight_magnitudes_*.png` - Weight magnitude analysis
- `model_comparison_*.png` - Side-by-side model comparison

### Data Files
- `*_architecture_analysis_*.json` - Detailed architecture analysis
- `comprehensive_diagnostics_*.json` - Complete training results
- `model_comparison_*.json` - Model comparison metrics

### Reports
- `diagnostic_report_*.md` - Comprehensive diagnostic report with findings and recommendations

## Key Features

### Training Diagnostics
- **Real-time monitoring**: Gradient norms, weight norms, pathway balance
- **Early stopping**: Automatic stopping when validation loss stops improving
- **Best model saving**: Saves best performing model weights
- **Comprehensive metrics**: Training/validation loss and accuracy

### Architecture Analysis
- **Parameter counting**: Total and trainable parameters by layer type
- **Weight initialization**: Statistical analysis of initial weights
- **Layer analysis**: Detailed breakdown of network architecture

### Gradient Analysis
- **Gradient flow**: Visualization of gradient magnitudes through layers
- **Dead neuron detection**: Identification of non-learning neurons
- **Pathway balance**: Analysis of multi-stream pathway contributions

### Model Comparison
- **Side-by-side comparison**: Training curves, final accuracies
- **Statistical analysis**: Best vs. final performance
- **Architectural differences**: Parameter counts and layer structures

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size: `--batch-size 16`
   - Use CPU: `--device cpu`

2. **Data Not Found**
   - Ensure CIFAR-100 data exists in `data/cifar-100/`
   - Check data directory path: `--data-dir /path/to/cifar-100`

3. **Import Errors**
   - Run from project root directory
   - Ensure all dependencies are installed

### Performance Tips

- **GPU Training**: Use CUDA for faster training
- **Batch Size**: Adjust based on available memory
- **Early Stopping**: Reduce patience for faster runs
- **Epochs**: Increase for thorough analysis

## Expected Results

### Performance Indicators
- **Validation Accuracy**: Should improve over training
- **Training/Validation Gap**: Indicates overfitting if large
- **Gradient Flow**: Should show learning in early layers
- **Pathway Balance**: Should show balanced contribution from both streams

### Common Issues to Identify
1. **Vanishing Gradients**: Gradients become very small in early layers
2. **Exploding Gradients**: Gradients become very large
3. **Dead Neurons**: Neurons that never activate
4. **Pathway Imbalance**: One stream dominates the other
5. **Overfitting**: Large gap between training and validation accuracy

## Next Steps

Based on diagnostic results, consider:

1. **Hyperparameter Tuning**: Adjust learning rate, weight decay, dropout
2. **Architecture Modifications**: Change layer sizes, add normalization
3. **Data Augmentation**: Modify augmentation strategies
4. **Training Strategies**: Implement curriculum learning, advanced optimizers
5. **Regularization**: Add batch normalization, dropout, weight decay

## Contributing

To extend the diagnostics system:

1. Add new diagnostic functions to `ComprehensiveModelDiagnostics`
2. Update the report generation to include new metrics
3. Add visualization functions for new analyses
4. Update documentation with new features
