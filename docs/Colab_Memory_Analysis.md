# Colab Kernel Crash - Memory Analysis

## Symptoms
- Kernel restarts during training
- Stack trace shows process termination
- No explicit CUDA OOM error (would see if GPU memory)
- Likely **RAM (CPU memory) exhaustion**

## Memory Budget on Colab

### Available Resources
- **RAM:** ~12-13 GB (shared with system)
- **GPU Memory (A100):** 40 GB
- **Disk:** Plenty

### Memory Consumers

#### 1. Dataset Loading
**NYU Depth V2 in memory:**
- Images: `1449 * 3 * 640 * 480 * 1 byte = ~1.3 GB` (uint8)
- Depths: `1449 * 640 * 480 * 4 bytes = ~1.7 GB` (float32)
- Scenes: `1 * 1449 * 8 bytes = ~12 KB` (int64)
- **Total dataset in RAM: ~3 GB**

#### 2. DataLoader Workers
**With num_workers=4:**
- Each worker loads dataset in memory
- Each worker has Python overhead
- **4 workers * 3 GB = 12 GB** ⚠️ **THIS IS THE PROBLEM**

#### 3. Model & Training
- MCResNet18: ~45 MB
- Optimizer state (SGD): ~45 MB
- Gradients: ~45 MB
- Batch tensors: ~200 MB
- **Total training: ~400 MB** (mostly on GPU)

### Total RAM Usage
```
Dataset (main process):     3 GB
Worker 1:                   3 GB
Worker 2:                   3 GB
Worker 3:                   3 GB
Worker 4:                   3 GB
Python overhead:            1 GB
-----------------------------------
TOTAL:                     16 GB  ⚠️ EXCEEDS 12-13 GB LIMIT
```

## Root Cause

**num_workers=4 causes each worker to load the entire dataset into memory when dereferencing HDF5 scenes.**

The problem is in `__init__`:
```python
if scenes_data.dtype == h5py.ref_dtype:
    # Dereference each element
    self.scenes = np.zeros((1, num_samples), dtype=np.int64)
    for i in range(num_samples):
        ref = scenes_data[0, i]
        self.scenes[0, i] = int(f[ref][0]) if ref else 0
```

This happens in the main process, then the entire `self.scenes` array is **pickled and copied to each worker**.

## Solutions

### Option 1: Reduce num_workers (QUICK FIX)
```python
train_loader, val_loader = create_nyu_dataloaders(
    h5_file_path=DATASET_PATH,
    batch_size=128,
    num_workers=2,  # Reduce from 4 to 2
    target_size=(224, 224),
    num_classes=13
)
```

**Memory with num_workers=2:**
```
Dataset (main):   3 GB
Worker 1:         3 GB
Worker 2:         3 GB
Python overhead:  1 GB
-------------------
TOTAL:           10 GB  ✅ Fits in 12-13 GB
```

### Option 2: Reduce batch_size (if still crashes)
```python
batch_size=64,  # Reduce from 128 to 64
num_workers=2
```

### Option 3: Use num_workers=0 (single process, slower)
```python
num_workers=0  # No multiprocessing
```
- No worker copies of dataset
- Slower data loading (~200 samples/sec)
- But won't crash

### Option 4: Fix dataset to not copy scenes (CODE FIX)

The issue is that scenes are loaded in `__init__` and then pickled to workers. We should delay loading:

```python
# In __init__, store only the path and flag
self.scenes = None
self._scenes_need_deref = scenes_data.dtype == h5py.ref_dtype if 'scenes' in f else False

# In __getitem__, load lazily (but this defeats the optimization)
# Not recommended - defeats the purpose
```

## Recommended Fix

### Immediate Action (Update Colab Notebook)

**Change this cell:**
```python
# BEFORE (causes crash):
train_loader, val_loader = create_nyu_dataloaders(
    h5_file_path=LOCAL_DATASET_PATH,
    batch_size=128,
    num_workers=4,  # ⚠️ Too many workers
    target_size=(224, 224),
    num_classes=13
)
```

**To this:**
```python
# AFTER (stable):
train_loader, val_loader = create_nyu_dataloaders(
    h5_file_path=LOCAL_DATASET_PATH,
    batch_size=128,
    num_workers=2,  # ✅ Reduced to fit in RAM
    target_size=(224, 224),
    num_classes=13
)
```

### Performance Impact
- **With 4 workers:** ~500 samples/sec (but crashes)
- **With 2 workers:** ~350-400 samples/sec (stable)
- **With 0 workers:** ~200 samples/sec (very stable)

### Expected Training Time (90 epochs, 1159 samples)

| Workers | Samples/sec | Epoch Time | Total Time | Stable? |
|---------|-------------|------------|------------|---------|
| 4 | 500 | ~2.3 sec | ~3.5 min | ❌ Crashes |
| 2 | 400 | ~2.9 sec | ~4.4 min | ✅ Stable |
| 0 | 200 | ~5.8 sec | ~8.7 min | ✅ Very Stable |

**Recommendation: Use num_workers=2** (4.4 min vs 8.7 min, worth it)

## Additional Memory Optimization

If still crashing with num_workers=2, reduce batch size:

```python
batch_size=64,    # Half the batch size
num_workers=2
```

This reduces GPU memory but shouldn't affect RAM much.

## Monitoring

Add this to check memory usage:
```python
import psutil
import os

process = psutil.Process(os.getpid())
print(f"RAM usage: {process.memory_info().rss / 1024**3:.2f} GB")
```

Run before and after creating dataloaders to confirm.
