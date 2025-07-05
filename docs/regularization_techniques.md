# Regularization Techniques for Multi-Stream Neural Networks

This document outlines techniques to combat overfitting in MultiChannel ResNet models that show high training accuracy (>90%) but low validation accuracy (<50%).

## Problem

The MultiChannel ResNet model is overfitting during training:
- After ~10 epochs, training accuracy exceeds 90%
- Validation accuracy plateaus below 50%
- Increasing patience and epochs doesn't improve validation performance

## Solution Approaches

### 1. Data Augmentation

Advanced data augmentation is critical for improving generalization:

- **Geometric Transformations**:
  - Random horizontal flips
  - Small rotations (±15 degrees)
  - Random crops with padding

- **Advanced Augmentation**:
  - **Cutout**: Randomly mask out square regions of the input image
  - **Mixup**: Blend pairs of images and their labels
  - **CutMix**: Replace sections of images with patches from other images

- **Dual-Stream Specific**:
  - Apply consistent transformations to both RGB and brightness streams
  - Custom `DualStreamAugmentation` class in `src/utils/augmentation.py`

### 2. Architectural Regularization

Modify the model architecture to reduce overfitting:

- **Dropout**: Add dropout layers (try 0.3-0.5 rate) after dense layers and between conv blocks
- **Batch Normalization**: Ensure all layers use batch normalization
- **Weight Standardization**: Standardize weights in convolutional layers
- **Stochastic Depth**: Randomly drop layers during training (similar to dropout but for entire layers)
- **Reduced Model Capacity**: Consider using a smaller model if the current one has too many parameters

### 3. Training Strategies

Optimize the training process:

- **Reduce Batch Size**: Smaller batches (32-64) can help escape sharp minima
- **Learning Rate Schedule**: Use cosine annealing with warm restarts
- **Weight Decay**: Increase L2 regularization (0.0001-0.001)
- **Early Stopping**: Stop training when validation accuracy plateaus
- **Gradient Clipping**: Limit gradient magnitude to prevent extreme updates
- **Label Smoothing**: Replace one-hot labels with soft targets (0.1 smoothing factor)

### 4. Ensemble Techniques

Use model averaging to improve generalization:

- **Exponential Moving Average (EMA)**: Keep a moving average of model weights
- **Stochastic Weight Averaging (SWA)**: Average weights from different points in training
- **Snapshot Ensembles**: Save models at various points and ensemble their predictions

### 5. Data Preprocessing

Improve data quality:

- **Normalization**: Ensure proper normalization of both RGB and brightness streams
- **Class Balancing**: Check for class imbalance and address if needed
- **Feature Engineering**: Consider additional features or transformations

## Implementation

We've implemented these techniques in:

1. **New Training Script**: `scripts/train_with_regularization.py`
2. **Augmentation Module**: `src/utils/augmentation.py`
3. **Configuration File**: `configs/model_configs/direct_mixing/regularized_scalar.yaml`

## Usage

Run the training script with regularization enabled:

```bash
python scripts/train_with_regularization.py --augment --dropout 0.3 --mixup 0.2 --cutmix 1.0
```

Or use the VSCode task with the regularized configuration:

```bash
python scripts/train.py --config configs/model_configs/direct_mixing/regularized_scalar.yaml
```

## Monitoring and Analysis

To understand overfitting behavior better:

- Plot training vs. validation accuracy curves
- Examine feature importance in dual-stream models
- Try different regularization strengths
- Visualize model activations and feature maps

## References

- He, K., et al. (2016). Deep Residual Learning for Image Recognition
- Zhang, H., et al. (2017). mixup: Beyond Empirical Risk Minimization
- DeVries, T., & Taylor, G. W. (2017). Improved Regularization of Convolutional Neural Networks with Cutout
- Yun, S., et al. (2019). CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features
