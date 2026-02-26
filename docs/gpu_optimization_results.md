# GPU Optimization Results for LINet3

## Problem

LINet3 processes N=3 independent streams (RGB, Depth, Orthogonal) through a ResNet-18 backbone. Each per-stream operation (conv, BN, ReLU, pooling) runs sequentially in a Python loop, launching ~315 separate small GPU kernels per forward pass. The hypothesis was that the GPU is underutilized because each kernel is too small to saturate compute.

## What We Tried

### 1. Custom Triton Kernels (Reverted)

Wrote custom Triton kernels to process all N streams in a single launch:

- **Multi-stream integration kernel** (1x1 conv accumulation): Replaced the sequential loop of N `F.conv2d` calls with a single Triton matmul kernel that reads all stream inputs and weights in one launch.
- **Fused BN+ReLU kernel**: Combined batch normalization stats reduction + normalize + ReLU into one kernel per layer instead of separate BN and ReLU calls per stream.
- **Elementwise kernels**: Multi-stream residual add and ReLU backward.

**Result: Drastically slower than cuDNN.** The Triton kernels could not compete with cuDNN's heavily optimized tensor core paths, even though they eliminated kernel launch overhead. cuDNN's per-kernel efficiency advantage far outweighed the launch overhead savings. The Triton kernels were removed entirely.

**Lesson learned**: Triton `tl.dot` cannot match cuDNN's spatial convolution performance. Triton is better suited for custom fusions that cuDNN doesn't provide natively, not for reimplementing operations cuDNN already handles well.

### 2. Grouped cuDNN Convolution (Reverted)

For layers 1-4 where all streams have identical channel counts (`[64]*3`, `[128]*3`, etc.), packed the N=3 stream inputs and weights into a single `F.conv2d(groups=3)` call to reduce 3 sequential cuDNN launches to 1.

**Implementation**: Used `torch.cat` to pack inputs along the channel dimension and weights along dim=0, then called `F.conv2d` with `groups=N`, then split the output back into per-stream tensors.

**Result: 10% slower than baseline (98.3ms vs 88.7ms).** Profiling revealed the bottleneck:

| Operation | Self CUDA Time | Cause |
|-----------|---------------|-------|
| `aten::copy_` | 73.6ms (22.99%) | `torch.cat` packing inputs/weights |
| `aten::contiguous`/`aten::clone` | 58.9ms | Format conversion from cat |
| `nhwcSliceCKernel` | 23.3ms | Splitting grouped output back |

The packing/unpacking overhead (~97ms total GPU time) far exceeded the launch overhead it was supposed to eliminate. The grouped conv itself was faster (30.6ms vs 38.3ms), but nowhere near enough to compensate.

**Why pre-allocated buffers couldn't help**: During training, `requires_grad=True` on weights and inputs means in-place writes into pre-allocated buffers would break the autograd graph. The `torch.cat` path (which allocates new tensors) is mandatory for training.

### 3. channels_last Memory Format (Kept)

Converted model weights and inputs from NCHW (PyTorch default) to NHWC (`torch.channels_last`) format.

**Result: 1.35x speedup (65.6ms vs 88.7ms).** The entire 23ms savings came from eliminating internal cuDNN format conversion kernels:

| Kernel | Baseline CUDA Time | channels_last CUDA Time |
|--------|-------------------|------------------------|
| `nchwToNhwcKernel` | 23.6ms | 0ms |
| `nhwcToNchwKernel` | 9.9ms | 0ms |
| **Total reformatting** | **33.5ms** | **0ms** |

cuDNN internally operates in NHWC. When inputs are NCHW, every conv call reformats on the fly. With channels_last, the data is already in the format cuDNN wants.

### 4. BN+ReLU Fusion (Kept)

Added an `apply_relu` parameter to `LIBatchNorm2d._forward_single_pathway()` so ReLU is applied immediately after batch norm within the same method call, eliminating the separate `LIReLU` module invocation in the forward path.

**Impact**: Minor — saves Python-level overhead of the separate ReLU module call. The actual CUDA kernels are still separate (cuDNN BN + vectorized ReLU), but the Python dispatch is simplified.

## Final Configuration

Only two optimizations survived profiling:

### channels_last (NHWC) format

**Where**: `li_net.py` — model weights converted in `__init__()`, inputs converted in `forward()`

```python
# In LIResNet.__init__():
if _conv_module.USE_CHANNELS_LAST:
    self.to(memory_format=torch.channels_last)

# In LIResNet.forward():
if _conv_module.USE_CHANNELS_LAST:
    stream_inputs = [s.contiguous(memory_format=torch.channels_last) for s in stream_inputs]
```

Controlled by `USE_CHANNELS_LAST` flag in `conv.py` (default: `True`).

### BN+ReLU fusion

**Where**: `conv.py` `LIBatchNorm2d._forward_single_pathway()` accepts `apply_relu` parameter. Called with `apply_relu=True` from `blocks.py` (`BasicBlock`, `Bottleneck`) and `li_net.py` (after conv1).

## Profiling Numbers (A100, batch=16, 416x544, AMP)

| Configuration | Wall-clock (fwd+bwd) | vs Baseline |
|--------------|---------------------|-------------|
| Baseline (NCHW, sequential) | 88.7ms | -- |
| channels_last only (NHWC, sequential) | 65.6ms | 1.35x faster |
| channels_last + grouped conv | 98.3ms | 0.90x (slower) |

## Key Takeaways

1. **Memory layout matters more than kernel count.** A single format change (NCHW to NHWC) delivered 1.35x speedup by eliminating redundant data reformatting. Meanwhile, reducing kernel launches from 315 to ~73 via Triton kernels made things slower.

2. **cuDNN is hard to beat.** Custom Triton kernels for convolution and batch normalization were consistently slower than cuDNN's implementations, even when they theoretically reduced total kernel launches.

3. **`torch.cat` is expensive during training.** Grouped convolution requires packing inputs, which means `torch.cat` (new allocation + copy) because autograd prevents in-place buffer writes. The copy overhead dominated any launch-overhead savings.

4. **Profile before optimizing.** The original hypothesis (GPU underutilized due to small kernels) was partially wrong — the real bottleneck was NCHW-to-NHWC reformatting inside cuDNN, not kernel launch overhead.
