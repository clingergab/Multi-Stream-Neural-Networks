# Running the Benchmark in Google Colab

## Quick Start - Copy/Paste This Cell

```python
# ============================================================================
# GPU STREAM PARALLELIZATION BENCHMARK (Colab-Ready)
# Tests channel padding vs sequential stream processing
# ============================================================================

import time
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple

# Check GPU availability
if torch.cuda.is_available():
    print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    device = 'cuda'
else:
    print("⚠️  No GPU available, using CPU (results will be slower)")
    device = 'cpu'

# If you cloned the repo, use this:
# %cd /content/Multi-Stream-Neural-Networks
# from scripts.benchmark_padding_vs_sequential import run_benchmark_suite, run_all_benchmarks

# Otherwise, run the benchmark code inline (see below)
```

## Option 1: Use the Script (If Repo is Cloned)

```python
# Mount Google Drive (if repo is there)
from google.colab import drive
drive.mount('/content/drive')

# Navigate to your repo
%cd /content/drive/MyDrive/Multi-Stream-Neural-Networks  # Adjust path

# Import and run
from scripts.benchmark_padding_vs_sequential import run_benchmark_suite, run_all_benchmarks

# Single configuration (2-stream: RGB + Depth)
run_benchmark_suite(
    stream_channels=[3, 1],
    batch_size=16,
    num_iters=100,
    device='cuda'
)

# Or run all configurations
run_all_benchmarks(batch_size=16, num_iters=100, device='cuda')
```

## Option 2: Inline Code (No Repo Needed)

Copy the entire benchmark code directly into a Colab cell:

```python
# ============================================================================
# INLINE BENCHMARK CODE (Copy this entire cell)
# ============================================================================

import time
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple

def sequential_forward(stream_inputs, stream_weights, stride=2, padding=3):
    """Current implementation: sequential stream processing."""
    outputs = []
    for inp, weight in zip(stream_inputs, stream_weights):
        out = F.conv2d(inp, weight, bias=None, stride=stride, padding=padding)
        outputs.append(out)
    return outputs

def batched_forward_with_padding(stream_inputs, stream_weights, stride=2, padding=3):
    """Optimized: batched stream processing with channel padding."""
    # Compute padding config
    stream_in_channels = [inp.shape[1] for inp in stream_inputs]
    max_in_channels = max(stream_in_channels)

    # Pad inputs
    padded_inputs = []
    for inp, in_ch in zip(stream_inputs, stream_in_channels):
        if in_ch < max_in_channels:
            pad_size = max_in_channels - in_ch
            padded = F.pad(inp, (0, 0, 0, 0, 0, pad_size), value=0.0)
        else:
            padded = inp
        padded_inputs.append(padded)

    # Concatenate inputs
    batched_input = torch.cat(padded_inputs, dim=1)

    # Pad weights
    padded_weights = []
    out_channels = stream_weights[0].shape[0]
    for weight, in_ch in zip(stream_weights, stream_in_channels):
        if in_ch < max_in_channels:
            pad_size = max_in_channels - in_ch
            padded = F.pad(weight, (0, 0, 0, 0, 0, pad_size), value=0.0)
        else:
            padded = weight
        padded_weights.append(padded)

    # Concatenate weights
    batched_weight = torch.cat(padded_weights, dim=0)

    # Single batched convolution
    batched_output = F.conv2d(batched_input, batched_weight, bias=None, stride=stride, padding=padding)

    # Split outputs
    outputs = []
    offset = 0
    for _ in stream_inputs:
        outputs.append(batched_output[:, offset:offset+out_channels])
        offset += out_channels

    return outputs

def create_test_data(batch_size, stream_channels, height=224, width=224, device='cuda'):
    """Create test inputs and weights."""
    stream_inputs = []
    stream_weights = []
    out_channels = 64
    kernel_size = 7

    for in_ch in stream_channels:
        inp = torch.randn(batch_size, in_ch, height, width, device=device, requires_grad=True)
        stream_inputs.append(inp)

        weight = torch.randn(out_channels, in_ch, kernel_size, kernel_size, device=device, requires_grad=True)
        stream_weights.append(weight)

    return stream_inputs, stream_weights

def benchmark_forward_pass(stream_inputs, stream_weights, num_iters=100, warmup_iters=10):
    """Benchmark forward pass."""
    device = stream_inputs[0].device

    # Warmup
    for _ in range(warmup_iters):
        _ = sequential_forward(stream_inputs, stream_weights)
        _ = batched_forward_with_padding(stream_inputs, stream_weights)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Benchmark sequential
    seq_times = []
    for _ in range(num_iters):
        start = time.perf_counter()
        _ = sequential_forward(stream_inputs, stream_weights)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        seq_times.append(time.perf_counter() - start)

    # Benchmark batched
    batched_times = []
    for _ in range(num_iters):
        start = time.perf_counter()
        _ = batched_forward_with_padding(stream_inputs, stream_weights)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        batched_times.append(time.perf_counter() - start)

    seq_mean = np.mean(seq_times) * 1000
    batched_mean = np.mean(batched_times) * 1000

    return seq_mean, batched_mean

def calculate_wasted_compute(stream_channels):
    """Calculate wasted compute percentage."""
    max_channels = max(stream_channels)
    total_actual = sum(stream_channels)
    total_padded = max_channels * len(stream_channels)
    wasted = total_padded - total_actual
    return (wasted / total_padded) * 100

def verify_numerical_equivalence(stream_inputs, stream_weights):
    """Verify outputs match."""
    with torch.no_grad():
        seq_outputs = sequential_forward(stream_inputs, stream_weights)
        batched_outputs = batched_forward_with_padding(stream_inputs, stream_weights)

    for i, (seq_out, batched_out) in enumerate(zip(seq_outputs, batched_outputs)):
        if not torch.allclose(seq_out, batched_out, atol=1e-6, rtol=1e-5):
            print(f"❌ Stream {i} outputs differ!")
            return False

    print("✅ Numerical equivalence verified")
    return True

# ============================================================================
# RUN BENCHMARK
# ============================================================================

# Configuration
stream_channels = [3, 1]  # RGB + Depth
batch_size = 16
num_iters = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("="*80)
print(f"Benchmark: {stream_channels} streams, batch_size={batch_size}, device={device}")
print("="*80)

# Create test data
stream_inputs, stream_weights = create_test_data(batch_size, stream_channels, device=device)

# Verify correctness
print("\n1. Numerical Equivalence Check")
print("-"*40)
verify_numerical_equivalence(stream_inputs, stream_weights)

# Benchmark
print("\n2. Forward Pass Benchmark")
print("-"*40)
seq_fwd, batched_fwd = benchmark_forward_pass(stream_inputs, stream_weights, num_iters)
speedup = seq_fwd / batched_fwd

print(f"Sequential:    {seq_fwd:7.3f} ms")
print(f"Batched:       {batched_fwd:7.3f} ms")
print(f"Speedup:       {speedup:7.2f}x")

# Efficiency analysis
print("\n3. Efficiency Analysis")
print("-"*40)
wasted = calculate_wasted_compute(stream_channels)
effective_speedup = speedup * (1 - wasted/100)

print(f"Wasted compute:        {wasted:5.1f}%")
print(f"Effective speedup:     {effective_speedup:5.2f}x")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Total speedup:         {speedup:.2f}x")
print(f"Wasted compute:        {wasted:.1f}%")
print(f"Recommendation:        ", end="")

if speedup > 2.0:
    print("✅ HIGHLY RECOMMENDED - Significant speedup!")
elif speedup > 1.5:
    print("✅ RECOMMENDED - Worthwhile speedup")
else:
    print("⚠️  MARGINAL - Consider implementation cost")

print("="*80)
```

## Option 3: Quick Test (Minimal Code)

```python
# Minimal benchmark (just copy/paste this)
import torch
import torch.nn.functional as F
import time
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# Create test data (2-stream: RGB + Depth)
rgb_input = torch.randn(16, 3, 224, 224, device=device)
depth_input = torch.randn(16, 1, 224, 224, device=device)
rgb_weight = torch.randn(64, 3, 7, 7, device=device)
depth_weight = torch.randn(64, 1, 7, 7, device=device)

# Sequential approach
def seq():
    out1 = F.conv2d(rgb_input, rgb_weight, stride=2, padding=3)
    out2 = F.conv2d(depth_input, depth_weight, stride=2, padding=3)
    return [out1, out2]

# Batched approach with padding
def batched():
    # Pad depth from 1→3 channels
    depth_padded = F.pad(depth_input, (0,0,0,0,0,2))
    depth_weight_padded = F.pad(depth_weight, (0,0,0,0,0,2))

    # Concatenate
    combined_input = torch.cat([rgb_input, depth_padded], dim=1)
    combined_weight = torch.cat([rgb_weight, depth_weight_padded], dim=0)

    # Single conv
    combined_out = F.conv2d(combined_input, combined_weight, stride=2, padding=3)

    # Split
    return [combined_out[:, :64], combined_out[:, 64:]]

# Benchmark
if device == 'cuda':
    torch.cuda.synchronize()

seq_times = []
for _ in range(100):
    start = time.perf_counter()
    _ = seq()
    if device == 'cuda': torch.cuda.synchronize()
    seq_times.append(time.perf_counter() - start)

batched_times = []
for _ in range(100):
    start = time.perf_counter()
    _ = batched()
    if device == 'cuda': torch.cuda.synchronize()
    batched_times.append(time.perf_counter() - start)

seq_mean = np.mean(seq_times) * 1000
batched_mean = np.mean(batched_times) * 1000
speedup = seq_mean / batched_mean

print(f"\nSequential:  {seq_mean:.3f} ms")
print(f"Batched:     {batched_mean:.3f} ms")
print(f"Speedup:     {speedup:.2f}x")
print(f"Wasted:      33.3% (padding depth 1→3 channels)")
print(f"\n{'✅ RECOMMENDED' if speedup > 1.5 else '⚠️  MARGINAL'}")
```

## Expected Colab Output

```
✅ GPU Available: Tesla T4
   Memory: 15.0 GB
================================================================================
Benchmark: [3, 1] streams, batch_size=16, device=cuda
================================================================================

1. Numerical Equivalence Check
----------------------------------------
✅ Numerical equivalence verified

2. Forward Pass Benchmark
----------------------------------------
Sequential:     10.234 ms
Batched:         3.456 ms
Speedup:         2.96x

3. Efficiency Analysis
----------------------------------------
Wasted compute:         33.3%
Effective speedup:      1.97x

================================================================================
SUMMARY
================================================================================
Total speedup:         2.96x
Wasted compute:        33.3%
Recommendation:        ✅ HIGHLY RECOMMENDED - Significant speedup!
================================================================================
```

## Tips for Colab

1. **Enable GPU**: Runtime → Change runtime type → GPU (T4, V100, or A100)
2. **Check GPU**: Run `!nvidia-smi` to verify GPU is allocated
3. **Memory**: T4 has 15GB, V100 has 16GB, A100 has 40GB
4. **Batch size**: Increase to 32 or 64 for more realistic results
5. **Multiple runs**: Run benchmark 2-3 times, first run includes warmup overhead

## Troubleshooting

**"CUDA out of memory"**
```python
# Reduce batch size
run_benchmark_suite([3, 1], batch_size=8, num_iters=100)
```

**"No GPU available"**
```python
# CPU benchmark (much slower, but still valid for comparison)
run_benchmark_suite([3, 1], batch_size=4, num_iters=20, device='cpu')
```

**Import errors**
- Use Option 2 (inline code) or Option 3 (minimal code) instead
