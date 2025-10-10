# Custom Dual-Stream Convolution Kernel Roadmap

**Goal:** Learn GPU programming by implementing dual-stream convolution in both Numba CUDA and Triton, then compare performance.

**Timeline:** 2-3 weeks (learning + implementation + optimization)

---

## 📚 Prerequisites

### Required Knowledge
- [x] Python programming
- [x] PyTorch basics
- [x] Understanding of convolution operation
- [ ] Basic CUDA concepts (threads, blocks, grids)
- [ ] Memory hierarchy (global, shared, registers)

### Environment Setup
```bash
# Install dependencies
pip install numba
pip install triton
pip install torch torchvision

# Verify CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
python -c "from numba import cuda; print(cuda.is_available())"
```

---

## Part 1: Numba CUDA Implementation

**Total Time:** ~1 week (5-7 days)

### Phase 1: Learning Numba CUDA Basics (Days 1-2)

#### Day 1: Tutorial & Simple Kernels (4-6 hours)

**Resources:**
- [ ] Read: [Numba CUDA Programming Guide](https://numba.readthedocs.io/en/stable/cuda/index.html)
- [ ] Read: [KDNuggets Tutorial](https://www.kdnuggets.com/writing-your-first-gpu-kernel-in-python-with-numba-and-cuda)
- [ ] Watch: Numba CUDA tutorial videos (YouTube)

**Hands-on Exercises:**

1. **Hello World Kernel**
```python
from numba import cuda
import numpy as np

@cuda.jit
def hello_kernel(arr):
    idx = cuda.grid(1)
    if idx < arr.size:
        arr[idx] = idx
```

2. **Vector Addition** (From tutorial)
```python
@cuda.jit
def vector_add(a, b, c):
    idx = cuda.grid(1)
    if idx < len(a):
        c[idx] = a[idx] + b[idx]
```

3. **Vector Multiplication**
```python
@cuda.jit
def vector_mul(a, b, c):
    idx = cuda.grid(1)
    if idx < len(a):
        c[idx] = a[idx] * b[idx]
```

**Key Concepts to Understand:**
- Thread indexing: `cuda.threadIdx.x`, `cuda.blockIdx.x`, `cuda.blockDim.x`
- Grid calculation: `cuda.grid(1)` = `threadIdx.x + blockIdx.x * blockDim.x`
- Kernel launch syntax: `kernel[blocks_per_grid, threads_per_block](args)`
- Bounds checking: Always check `idx < array_size`

**Deliverable:** Working vector operations, understand thread model

---

#### Day 2: 2D Kernels & Matrix Operations (4-6 hours)

**Exercises:**

1. **Matrix Addition**
```python
@cuda.jit
def matrix_add(A, B, C):
    row = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    col = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y

    if row < A.shape[0] and col < A.shape[1]:
        C[row, col] = A[row, col] + B[row, col]
```

2. **Transpose**
```python
@cuda.jit
def transpose(A, B):
    row = cuda.grid(1) // A.shape[1]
    col = cuda.grid(1) % A.shape[1]

    if row < A.shape[0] and col < A.shape[1]:
        B[col, row] = A[row, col]
```

3. **2D Grid Calculation**
```python
# Understanding 2D thread layout
threads_per_block = (16, 16)  # 16x16 = 256 threads
blocks_x = (width + threads_per_block[0] - 1) // threads_per_block[0]
blocks_y = (height + threads_per_block[1] - 1) // threads_per_block[1]
blocks_per_grid = (blocks_x, blocks_y)
```

**Key Concepts:**
- 2D thread indexing
- Row-major vs column-major layout
- Grid/block configuration for 2D data

**Deliverable:** Working 2D kernels, understand 2D indexing

---

### Phase 2: Implement Naive Convolution (Days 3-4)

#### Day 3: Single-Stream Convolution (6-8 hours)

**Step 1: Understand Convolution Math**
```
Output[b, oc, oh, ow] = Σ Σ Σ Input[b, ic, oh*s+kh-p, ow*s+kw-p] * Weight[oc, ic, kh, kw]
                        ic kh kw

Where:
  b  = batch index
  oc = output channel
  oh = output height
  ow = output width
  ic = input channel
  kh = kernel height
  kw = kernel width
  s  = stride
  p  = padding
```

**Step 2: Implement Naive Conv2d**
```python
@cuda.jit
def conv2d_naive(input, weight, bias, output,
                 batch, in_channels, out_channels,
                 in_h, in_w, out_h, out_w,
                 kernel_size, stride, padding):
    """
    Naive 2D convolution kernel.
    Each thread computes ONE output pixel for ONE output channel.
    """
    # Calculate output position
    b = cuda.blockIdx.x
    oh = cuda.threadIdx.x + cuda.blockIdx.y * cuda.blockDim.x
    ow = cuda.threadIdx.y + cuda.blockIdx.z * cuda.blockDim.y

    # Bounds check
    if b >= batch or oh >= out_h or ow >= out_w:
        return

    # Compute for each output channel
    for oc in range(out_channels):
        sum_val = 0.0

        # Convolve over input channels and kernel
        for ic in range(in_channels):
            for kh in range(kernel_size):
                for kw in range(kernel_size):
                    # Calculate input position
                    ih = oh * stride + kh - padding
                    iw = ow * stride + kw - padding

                    # Bounds check for input
                    if 0 <= ih < in_h and 0 <= iw < in_w:
                        inp_val = input[b, ic, ih, iw]
                        weight_val = weight[oc, ic, kh, kw]
                        sum_val += inp_val * weight_val

        # Add bias and store
        if bias is not None:
            sum_val += bias[oc]

        output[b, oc, oh, ow] = sum_val
```

**Step 3: PyTorch Wrapper**
```python
def conv2d_numba(input, weight, bias=None, stride=1, padding=0):
    """PyTorch wrapper for Numba CUDA convolution"""
    # Extract dimensions
    batch, in_c, in_h, in_w = input.shape
    out_c, _, k_h, k_w = weight.shape

    # Calculate output size
    out_h = (in_h + 2*padding - k_h) // stride + 1
    out_w = (in_w + 2*padding - k_w) // stride + 1

    # Allocate output
    output = torch.zeros(batch, out_c, out_h, out_w,
                        device=input.device, dtype=input.dtype)

    # Configure grid/blocks
    threads_per_block = (16, 16)
    blocks_x = batch
    blocks_y = (out_h + 15) // 16
    blocks_z = (out_w + 15) // 16
    blocks_per_grid = (blocks_x, blocks_y, blocks_z)

    # Launch kernel
    conv2d_naive[blocks_per_grid, threads_per_block](
        input, weight, bias, output,
        batch, in_c, out_c, in_h, in_w, out_h, out_w,
        k_h, stride, padding
    )

    cuda.synchronize()
    return output
```

**Step 4: Testing & Validation**
```python
def test_conv2d_numba():
    # Create test data
    x = torch.randn(2, 3, 32, 32, device='cuda')
    w = torch.randn(16, 3, 3, 3, device='cuda')
    b = torch.randn(16, device='cuda')

    # PyTorch reference
    out_pytorch = F.conv2d(x, w, b, stride=1, padding=1)

    # Numba implementation
    out_numba = conv2d_numba(x, w, b, stride=1, padding=1)

    # Compare
    diff = (out_pytorch - out_numba).abs().max()
    print(f"Max difference: {diff.item():.6f}")
    assert diff < 1e-4, "Results don't match!"
```

**Deliverable:** Working single-stream convolution that matches PyTorch output

---

#### Day 4: Dual-Stream Convolution (6-8 hours)

**Modify kernel to process TWO streams in same thread**

```python
@cuda.jit
def dual_conv2d_naive(
    # Stream 1
    stream1_input, stream1_weight, stream1_bias, stream1_output,
    # Stream 2
    stream2_input, stream2_weight, stream2_bias, stream2_output,
    # Dimensions
    batch, s1_in_c, s2_in_c, out_c,
    in_h, in_w, out_h, out_w,
    kernel_size, stride, padding
):
    """
    Dual-stream convolution kernel.
    Each thread computes output for BOTH streams.
    """
    # Calculate output position (shared for both streams)
    b = cuda.blockIdx.x
    oh = cuda.threadIdx.x + cuda.blockIdx.y * cuda.blockDim.x
    ow = cuda.threadIdx.y + cuda.blockIdx.z * cuda.blockDim.y

    if b >= batch or oh >= out_h or ow >= out_w:
        return

    # ========== STREAM 1 ==========
    for oc in range(out_c):
        sum_val = 0.0
        for ic in range(s1_in_c):
            for kh in range(kernel_size):
                for kw in range(kernel_size):
                    ih = oh * stride + kh - padding
                    iw = ow * stride + kw - padding

                    if 0 <= ih < in_h and 0 <= iw < in_w:
                        sum_val += stream1_input[b, ic, ih, iw] * \
                                   stream1_weight[oc, ic, kh, kw]

        if stream1_bias is not None:
            sum_val += stream1_bias[oc]
        stream1_output[b, oc, oh, ow] = sum_val

    # ========== STREAM 2 (in SAME thread!) ==========
    for oc in range(out_c):
        sum_val = 0.0
        for ic in range(s2_in_c):
            for kh in range(kernel_size):
                for kw in range(kernel_size):
                    ih = oh * stride + kh - padding
                    iw = ow * stride + kw - padding

                    if 0 <= ih < in_h and 0 <= iw < in_w:
                        sum_val += stream2_input[b, ic, ih, iw] * \
                                   stream2_weight[oc, ic, kh, kw]

        if stream2_bias is not None:
            sum_val += stream2_bias[oc]
        stream2_output[b, oc, oh, ow] = sum_val
```

**Testing:**
```python
def test_dual_conv2d():
    # RGB stream
    rgb = torch.randn(256, 3, 416, 544, device='cuda')
    rgb_w = torch.randn(64, 3, 3, 3, device='cuda')

    # Depth stream
    depth = torch.randn(256, 1, 416, 544, device='cuda')
    depth_w = torch.randn(64, 1, 3, 3, device='cuda')

    # Reference (sequential)
    rgb_out_ref = F.conv2d(rgb, rgb_w, stride=1, padding=1)
    depth_out_ref = F.conv2d(depth, depth_w, stride=1, padding=1)

    # Dual-stream kernel
    rgb_out, depth_out = dual_conv2d_numba(
        rgb, rgb_w, None, depth, depth_w, None,
        stride=1, padding=1
    )

    # Validate
    assert (rgb_out - rgb_out_ref).abs().max() < 1e-4
    assert (depth_out - depth_out_ref).abs().max() < 1e-4
    print("✅ Dual-stream kernel works correctly!")
```

**Deliverable:** Working dual-stream convolution, validated against PyTorch

---

### Phase 3: Optimization (Days 5-6)

#### Day 5: Shared Memory Optimization (6-8 hours)

**Problem with naive implementation:**
- Each thread loads from global memory (slow)
- Same data loaded multiple times by different threads
- Memory bandwidth bottleneck

**Solution: Shared memory**
```python
@cuda.jit
def dual_conv2d_shared(
    stream1_input, stream1_weight, stream1_bias, stream1_output,
    stream2_input, stream2_weight, stream2_bias, stream2_output,
    batch, s1_in_c, s2_in_c, out_c,
    in_h, in_w, out_h, out_w,
    kernel_size, stride, padding
):
    # Allocate shared memory for input tiles
    TILE_SIZE = 16
    s1_tile = cuda.shared.array(shape=(TILE_SIZE+2, TILE_SIZE+2), dtype=float32)
    s2_tile = cuda.shared.array(shape=(TILE_SIZE+2, TILE_SIZE+2), dtype=float32)

    # Thread position within block
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    # Global output position
    b = cuda.blockIdx.x
    oh = tx + cuda.blockIdx.y * cuda.blockDim.x
    ow = ty + cuda.blockIdx.z * cuda.blockDim.y

    if b >= batch or oh >= out_h or ow >= out_w:
        return

    # For each input channel
    for ic in range(max(s1_in_c, s2_in_c)):
        # Cooperatively load tile into shared memory
        ih_base = cuda.blockIdx.y * TILE_SIZE - padding
        iw_base = cuda.blockIdx.z * TILE_SIZE - padding

        # Load stream1 tile (if channel exists)
        if ic < s1_in_c:
            ih = ih_base + tx
            iw = iw_base + ty
            if 0 <= ih < in_h and 0 <= iw < in_w:
                s1_tile[tx, ty] = stream1_input[b, ic, ih, iw]
            else:
                s1_tile[tx, ty] = 0.0

        # Load stream2 tile (if channel exists)
        if ic < s2_in_c:
            ih = ih_base + tx
            iw = iw_base + ty
            if 0 <= ih < in_h and 0 <= iw < in_w:
                s2_tile[tx, ty] = stream2_input[b, ic, ih, iw]
            else:
                s2_tile[tx, ty] = 0.0

        # Synchronize to ensure tile is loaded
        cuda.syncthreads()

        # Compute convolution using shared memory
        # ... (use s1_tile and s2_tile instead of global memory)

        # Synchronize before loading next tile
        cuda.syncthreads()
```

**Expected speedup:** 2-3x over naive implementation

**Deliverable:** Optimized kernel with shared memory

---

#### Day 6: Additional Optimizations (4-6 hours)

**Optimization 1: Loop Unrolling**
```python
# Instead of:
for kh in range(3):
    for kw in range(3):
        # ...

# Manually unroll:
sum_val += input[...][kh=0, kw=0] * weight[...][0, 0]
sum_val += input[...][kh=0, kw=1] * weight[...][0, 1]
sum_val += input[...][kh=0, kw=2] * weight[...][0, 2]
# ... (9 operations total for 3x3 kernel)
```

**Optimization 2: Register Blocking**
- Keep frequently used values in registers
- Reduce shared memory accesses

**Optimization 3: Occupancy Tuning**
- Experiment with different block sizes
- Balance shared memory usage vs occupancy

**Deliverable:** Final optimized Numba CUDA kernel

---

### Phase 4: Integration & Benchmarking (Day 7)

#### Integration with MCConv2d

```python
# In src/models/multi_channel/conv.py

class MCConv2d(_MCConvNd):
    def __init__(self, ...):
        super().__init__(...)
        self.use_numba_kernel = False  # Enable manually

    def _conv_forward(self, stream1_input, stream2_input,
                     stream1_weight, stream2_weight,
                     stream1_bias, stream2_bias):

        if self.use_numba_kernel and stream1_input.is_cuda:
            # Use Numba kernel
            from src.kernels.numba_conv import dual_conv2d_numba
            return dual_conv2d_numba(
                stream1_input, stream1_weight, stream1_bias,
                stream2_input, stream2_weight, stream2_bias,
                stride=self.stride[0], padding=self.padding[0]
            )
        else:
            # Sequential PyTorch
            stream1_out = F.conv2d(stream1_input, self.stream1_weight, ...)
            stream2_out = F.conv2d(stream2_input, self.stream2_weight, ...)
            return stream1_out, stream2_out
```

#### Benchmarking Script

```python
# scripts/benchmark_numba_kernel.py

import torch
import time
from src.models.multi_channel.mc_resnet import mc_resnet18

def benchmark_kernel(batch_size, num_iterations=100):
    # Create model
    model = mc_resnet18(num_classes=15).cuda()

    # Test data
    rgb = torch.randn(batch_size, 3, 416, 544, device='cuda')
    depth = torch.randn(batch_size, 1, 416, 544, device='cuda')

    # Warmup
    for _ in range(10):
        _ = model(rgb, depth)

    # Benchmark sequential
    model.use_numba_kernel = False
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iterations):
        _ = model(rgb, depth)
    torch.cuda.synchronize()
    time_sequential = time.time() - start

    # Benchmark Numba
    model.use_numba_kernel = True
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iterations):
        _ = model(rgb, depth)
    torch.cuda.synchronize()
    time_numba = time.time() - start

    print(f"Batch Size: {batch_size}")
    print(f"  Sequential: {time_sequential:.4f}s")
    print(f"  Numba:      {time_numba:.4f}s")
    print(f"  Speedup:    {time_sequential/time_numba:.2f}x")
    print()

# Test different batch sizes
for bs in [32, 64, 128, 256, 512]:
    benchmark_kernel(bs)
```

**Deliverable:**
- Integrated kernel in MCConv2d
- Benchmark results for different batch sizes
- Comparison with sequential implementation

---

## Part 2: Triton Implementation

**Total Time:** ~1 week (5-7 days)

### Phase 1: Learning Triton Basics (Days 8-9)

#### Day 8: Triton Fundamentals (4-6 hours)

**Resources:**
- [ ] Read: [Triton Tutorial](https://triton-lang.org/main/getting-started/tutorials/index.html)
- [ ] Read: Vector Addition tutorial
- [ ] Watch: Triton overview videos

**Key Differences from Numba:**
- **Block programming model** (not thread-level)
- **Automatic tiling** and optimization
- **Pointer arithmetic** for memory access
- **Masking** for bounds checking

**Hello World:**
```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    # Program ID (which block are we?)
    pid = tl.program_id(axis=0)

    # Block start offset
    block_start = pid * BLOCK_SIZE

    # Offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Mask for bounds checking
    mask = offsets < n_elements

    # Load data (masked)
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Compute
    output = x + y

    # Store (masked)
    tl.store(output_ptr + offsets, output, mask=mask)
```

**Mental Model Shift:**
```
Numba:  "I am ONE thread, what do I compute?"
Triton: "I am ONE block, what does my block compute?"
```

**Deliverable:** Working vector operations in Triton

---

#### Day 9: Triton Matrix Operations (4-6 hours)

**Matrix Multiplication Example:**
```python
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Offsets
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Pointers
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator += tl.dot(a, b)

        # Advance pointers
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Store
    c_ptrs = c_ptr + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
    tl.store(c_ptrs, accumulator)
```

**Key Concepts:**
- Pointer arithmetic
- Stride handling
- Block-level operations
- `tl.dot()` for efficient matrix multiply

**Deliverable:** Working matrix operations in Triton

---

### Phase 2: Implement Triton Convolution (Days 10-12)

#### Day 10-11: Single-Stream Convolution (8-12 hours)

**Convolution in Triton:**
```python
@triton.jit
def conv2d_triton_kernel(
    # Pointers
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    # Dimensions
    batch, in_channels, out_channels,
    in_h, in_w, out_h, out_w,
    kernel_size, stride, padding,
    # Strides (for memory layout)
    input_batch_stride, input_channel_stride, input_h_stride, input_w_stride,
    weight_oc_stride, weight_ic_stride, weight_h_stride, weight_w_stride,
    output_batch_stride, output_channel_stride, output_h_stride, output_w_stride,
    # Block sizes
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Program IDs
    pid_batch = tl.program_id(axis=0)
    pid_oc = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)

    # Output tile offsets
    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_w = tl.arange(0, BLOCK_SIZE_W)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)

    # Loop over input channels
    for ic in range(in_channels):
        # Loop over kernel
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                # Calculate input positions
                ih = offs_h[:, None] * stride + kh - padding
                iw = offs_w[None, :] * stride + kw - padding

                # Bounds mask
                mask = (ih >= 0) & (ih < in_h) & (iw >= 0) & (iw < in_w)

                # Input pointers
                input_ptrs = input_ptr + \
                    pid_batch * input_batch_stride + \
                    ic * input_channel_stride + \
                    ih * input_h_stride + \
                    iw * input_w_stride

                # Load input (masked)
                inp = tl.load(input_ptrs, mask=mask, other=0.0)

                # Weight pointer
                weight_ptr_kh_kw = weight_ptr + \
                    pid_oc * weight_oc_stride + \
                    ic * weight_ic_stride + \
                    kh * weight_h_stride + \
                    kw * weight_w_stride

                # Load weight (scalar)
                w = tl.load(weight_ptr_kh_kw)

                # Accumulate
                accumulator += inp * w

    # Add bias
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + pid_oc)
        accumulator += bias_val

    # Store output
    output_ptrs = output_ptr + \
        pid_batch * output_batch_stride + \
        pid_oc * output_channel_stride + \
        offs_h[:, None] * output_h_stride + \
        offs_w[None, :] * output_w_stride

    # Bounds mask for output
    out_mask = (offs_h[:, None] < out_h) & (offs_w[None, :] < out_w)
    tl.store(output_ptrs, accumulator, mask=out_mask)
```

**Challenges:**
- Getting pointer arithmetic right
- Handling strides correctly
- Masking for bounds checking
- Block size tuning

**Deliverable:** Working Triton single-stream convolution

---

#### Day 12: Dual-Stream Triton Convolution (6-8 hours)

**Approach:** Two separate kernels launched in sequence (Triton doesn't easily support dual computation in one kernel)

```python
def dual_conv2d_triton(
    stream1_input, stream1_weight, stream1_bias,
    stream2_input, stream2_weight, stream2_bias,
    stride=1, padding=0
):
    # Launch stream1 kernel
    stream1_output = conv2d_triton(
        stream1_input, stream1_weight, stream1_bias,
        stride, padding
    )

    # Launch stream2 kernel
    stream2_output = conv2d_triton(
        stream2_input, stream2_weight, stream2_bias,
        stride, padding
    )

    return stream1_output, stream2_output
```

**Note:** Unlike Numba, Triton's strength is in automatic optimization, not dual-processing per thread.

**Deliverable:** Working dual-stream Triton convolution

---

### Phase 3: Auto-tuning (Day 13)

**Triton's Killer Feature: Auto-tuning**

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_H': 16, 'BLOCK_SIZE_W': 16}, num_warps=4),
        triton.Config({'BLOCK_SIZE_H': 32, 'BLOCK_SIZE_W': 32}, num_warps=8),
        triton.Config({'BLOCK_SIZE_H': 64, 'BLOCK_SIZE_W': 16}, num_warps=8),
        triton.Config({'BLOCK_SIZE_H': 16, 'BLOCK_SIZE_W': 64}, num_warps=8),
    ],
    key=['batch', 'in_channels', 'out_channels', 'in_h', 'in_w'],
)
@triton.jit
def conv2d_triton_kernel(...):
    # ... kernel code ...
```

**How it works:**
- Triton automatically tests all configs
- Picks fastest for given input dimensions
- Caches result for future runs

**Deliverable:** Auto-tuned Triton kernel

---

### Phase 4: Integration & Benchmarking (Day 14)

**Same as Numba - integrate into MCConv2d with flag**

```python
self.use_triton_kernel = False  # Enable manually
```

---

## Part 3: Comparison & Analysis

**Total Time:** 1-2 days (Days 15-16)

### Comprehensive Benchmarking

Create `scripts/compare_kernels.py`:

```python
import torch
import time
import pandas as pd
from src.models.multi_channel.mc_resnet import mc_resnet18

def benchmark_all_implementations(batch_sizes=[32, 64, 128, 256, 512]):
    results = []

    for bs in batch_sizes:
        print(f"\nBatch Size: {bs}")
        print("=" * 60)

        # Test data
        rgb = torch.randn(bs, 3, 416, 544, device='cuda')
        depth = torch.randn(bs, 1, 416, 544, device='cuda')

        # Test each implementation
        implementations = [
            ('Sequential (PyTorch)', False, False),
            ('Numba CUDA', True, False),
            ('Triton', False, True),
        ]

        for name, use_numba, use_triton in implementations:
            model = mc_resnet18(num_classes=15).cuda()
            model.use_numba_kernel = use_numba
            model.use_triton_kernel = use_triton

            # Warmup
            for _ in range(10):
                _ = model(rgb, depth)

            # Benchmark
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(100):
                out = model(rgb, depth)
            torch.cuda.synchronize()
            elapsed = time.time() - start

            # Verify correctness
            with torch.no_grad():
                model_ref = mc_resnet18(num_classes=15).cuda()
                model_ref.load_state_dict(model.state_dict())
                out_ref = model_ref(rgb, depth)
                diff = (out - out_ref).abs().max().item()

            results.append({
                'Batch Size': bs,
                'Implementation': name,
                'Time (s)': elapsed,
                'Time per batch (ms)': elapsed / 100 * 1000,
                'Throughput (samples/s)': bs * 100 / elapsed,
                'Max Error': diff,
            })

            print(f"  {name:20s}: {elapsed:.4f}s ({elapsed/100*1000:.2f}ms/batch)")

    # Create DataFrame
    df = pd.DataFrame(results)

    # Calculate speedups
    for bs in batch_sizes:
        df_bs = df[df['Batch Size'] == bs]
        baseline = df_bs[df_bs['Implementation'] == 'Sequential (PyTorch)']['Time (s)'].values[0]
        df.loc[df['Batch Size'] == bs, 'Speedup'] = baseline / df_bs['Time (s)']

    return df

# Run benchmarks
df = benchmark_all_implementations()

# Save results
df.to_csv('kernel_comparison_results.csv', index=False)

# Print summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(df.pivot_table(
    values='Speedup',
    index='Batch Size',
    columns='Implementation'
))
```

### Analysis Questions

1. **Performance:**
   - Which implementation is fastest at each batch size?
   - At what batch size does each kernel become beneficial?
   - What's the maximum speedup achieved?

2. **Scaling:**
   - How does each implementation scale with batch size?
   - Does speedup increase with larger batches?

3. **Trade-offs:**
   - Implementation complexity vs performance gain
   - Learning curve vs final speedup
   - Maintainability vs optimization

4. **Lessons Learned:**
   - What GPU concepts did you learn?
   - Which approach felt more intuitive?
   - What would you do differently?

---

## Expected Outcomes

### Numba CUDA
- **Speedup:** 1.3-1.5x (naive), 1.5-1.7x (optimized)
- **Pros:**
  - Easy to learn (Python syntax)
  - Direct control over threads
  - Good for understanding GPU fundamentals
- **Cons:**
  - Slower than cuDNN
  - Manual optimization required
  - No automatic tuning

### Triton
- **Speedup:** 1.4-1.6x (with auto-tuning)
- **Pros:**
  - Automatic optimization
  - Clean abstraction
  - Modern approach
- **Cons:**
  - Different mental model (blocks vs threads)
  - Less direct control
  - Harder to debug

### Overall Learning
- Understanding of GPU memory hierarchy
- Knowledge of parallel programming patterns
- Experience with kernel optimization
- Comparison of different GPU programming approaches

---

## Deliverables Checklist

- [ ] Numba CUDA naive implementation
- [ ] Numba CUDA optimized implementation
- [ ] Triton basic implementation
- [ ] Triton auto-tuned implementation
- [ ] Integration into MCConv2d
- [ ] Comprehensive benchmarks
- [ ] Comparison analysis
- [ ] Documentation of learnings

---

## Resources

### Numba CUDA
- Official docs: https://numba.readthedocs.io/en/stable/cuda/index.html
- KDNuggets tutorial: https://www.kdnuggets.com/writing-your-first-gpu-kernel-in-python-with-numba-and-cuda
- CUDA Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/

### Triton
- Official tutorial: https://triton-lang.org/main/getting-started/tutorials/index.html
- GitHub examples: https://github.com/openai/triton/tree/main/python/tutorials
- PyTorch blog: https://pytorch.org/blog/introduction-to-quantization-on-pytorch/

### General GPU Programming
- CUDA C++ Best Practices: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- Understanding GPU Memory: https://developer.nvidia.com/blog/cuda-pro-tip-understanding-memory-hierarchy/

---

## Next Steps After Completion

1. **Write blog post** about the experience
2. **Share results** on GitHub/Reddit
3. **Consider optimization further** if needed
4. **Apply learnings** to other operations (pooling, normalization)
5. **Contribute** to Numba/Triton communities

---

**Good luck! This will be a great learning journey! 🚀**
