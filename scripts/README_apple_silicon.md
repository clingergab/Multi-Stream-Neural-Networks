# Apple Silicon (MPS) Optimization Guide

This guide provides specific optimizations and troubleshooting for running the comprehensive model diagnostics on Apple Silicon Macs (M1/M2/M3).

## 🍎 Apple Silicon Benefits

### Performance Advantages
- **Unified Memory**: Shared between CPU and GPU for efficient data transfer
- **Energy Efficiency**: Lower power consumption compared to discrete GPUs
- **Metal Performance Shaders**: Native GPU acceleration for PyTorch

### Automatic Detection
The diagnostics script automatically detects and uses MPS when available:
```python
# Automatic device selection priority:
# 1. MPS (Apple Silicon)
# 2. CUDA (NVIDIA)
# 3. CPU (fallback)
```

## ⚙️ MPS-Specific Optimizations

### 1. Data Loading
```bash
# Optimized settings for MPS
--batch-size 16  # Start with smaller batch size
--device auto    # Automatic MPS detection
```

### 2. Memory Management
- **Smaller Batch Sizes**: Start with batch_size=16, increase gradually
- **Monitor Memory**: Use Activity Monitor to watch memory usage
- **Memory Clearing**: The script automatically handles MPS memory management

### 3. DataLoader Settings
```python
# Automatically applied for MPS:
num_workers = 0      # MPS works better with single-threaded loading
pin_memory = False   # Not beneficial for MPS
```

## 🚀 Running on Apple Silicon

### Basic Usage
```bash
# Automatic MPS detection
python scripts/comprehensive_model_diagnostics.py \
    --data-dir data/cifar-100 \
    --batch-size 16

# Force MPS usage
python scripts/comprehensive_model_diagnostics.py \
    --device mps \
    --batch-size 16
```

### Memory-Constrained Systems
```bash
# For 8GB unified memory
python scripts/comprehensive_model_diagnostics.py \
    --batch-size 8 \
    --epochs 10

# For 16GB+ unified memory
python scripts/comprehensive_model_diagnostics.py \
    --batch-size 32 \
    --epochs 20
```

## 🔧 Troubleshooting

### Memory Issues
```bash
# Symptoms: "out of memory" errors
# Solutions:
--batch-size 8      # Reduce batch size
--epochs 10         # Reduce training duration
```

### Slow Performance
```bash
# Check if MPS is being used
# Look for: "🍎 Apple Silicon MPS GPU detected"

# If seeing CPU usage instead:
pip install --upgrade torch torchvision  # Update PyTorch
```

### Operation Not Supported
```bash
# Some operations may fall back to CPU
# Set environment variable:
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Then run diagnostics
python scripts/comprehensive_model_diagnostics.py
```

## 📊 Expected Performance

### Training Speed
- **M1 (8-core GPU)**: ~2-3x faster than CPU
- **M1 Pro (16-core GPU)**: ~4-5x faster than CPU  
- **M1 Max (32-core GPU)**: ~6-8x faster than CPU
- **M2/M3 variants**: Similar or better performance

### Memory Usage
- **Base Model**: ~2-4GB unified memory
- **ResNet50**: ~6-8GB unified memory
- **Both Models**: ~8-12GB unified memory

## 🎯 Optimization Tips

### 1. Batch Size Tuning
```python
# Start conservative and increase
batch_sizes = [8, 16, 24, 32]
# Monitor memory usage and find sweet spot
```

### 2. Model Size Considerations
```python
# For memory-constrained systems, consider:
reduce_architecture=True  # Smaller ResNet variant
dropout=0.5              # Higher dropout to reduce overfitting
```

### 3. Data Preprocessing
```python
# Keep data on CPU until needed
# MPS handles CPU->GPU transfer efficiently
# due to unified memory architecture
```

## 🔍 Device Information

The diagnostic report includes Apple Silicon specific information:
- MPS availability and usage
- PyTorch version compatibility
- Memory optimization recommendations
- Performance benchmarks

## 📈 Performance Monitoring

### Activity Monitor
- **GPU History**: Monitor Metal GPU usage
- **Memory**: Watch unified memory consumption
- **Energy**: Track power efficiency

### Console Output
```
🍎 Apple Silicon MPS GPU detected - using Metal Performance Shaders
🔧 Diagnostics initialized - Device: mps
🍎 Applying MPS optimizations...
```

## ⚠️ Known Limitations

### Operations Not Supported on MPS
Some PyTorch operations may fall back to CPU:
- Certain advanced indexing operations
- Some sparse tensor operations
- Specific gradient operations

### Workarounds
1. **Automatic Fallback**: Enable with `PYTORCH_ENABLE_MPS_FALLBACK=1`
2. **CPU Computation**: Critical operations automatically handled
3. **Mixed Mode**: CPU fallback with GPU acceleration where possible

## 🆕 Future Improvements

### PyTorch Updates
- Broader MPS operation support
- Performance optimizations
- Memory efficiency improvements

### Model Optimizations
- MPS-specific layer implementations
- Apple Neural Engine integration
- Optimized data pipeline for unified memory

## 📝 Example Output

```
🚀 Starting comprehensive model diagnostics...
🍎 Apple Silicon MPS GPU detected - using Metal Performance Shaders
🔧 Diagnostics initialized - Device: mps
📊 Setting up CIFAR-100 data loaders...
✅ Data loaders created:
   Training samples: 45000
   Validation samples: 5000
   Batch size: 16
   Augmentation: Enabled
🏗️ Creating full-size models...
🍎 Applying MPS optimizations...
✅ Created base_multi_channel_large - Parameters: 8,847,300
🍎 Applying MPS optimizations...  
✅ Created multi_channel_resnet50 - Parameters: 25,557,032
```

This comprehensive Apple Silicon support ensures optimal performance on your Mac while providing fallbacks for compatibility.
