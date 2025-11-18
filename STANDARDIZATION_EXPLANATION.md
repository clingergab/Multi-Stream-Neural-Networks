# Why Test Visualizations Look Different from Preprocessed Images

## TL;DR

The test visualization images look colorful and show strong patterns because:
1. **Local orthogonal uses standardization** (dividing by std before SVD)
2. **Matplotlib auto-scales** the colormap to each image's data range
3. **Preprocessed files use non-standardized** global orthogonal (mathematically cleaner)

This is **NOT a bug** - it's a deliberate difference in approach.

---

## The Three Differences

### 1. Standardization vs Non-Standardization

#### Test Images (`tests/global_vs_local_comparison.png`)

**Local Orthogonal** (standardized):
```python
X = np.stack([r, g, b, d], axis=1)
X_mean = X.mean(axis=0)
X_std = X.std(axis=0) + 1e-8
X_standardized = (X - X_mean) / X_std  # STANDARDIZATION!

U, S, Vt = np.linalg.svd(X_standardized, full_matrices=False)
```

Result: std = 0.0992, 0.1532, 0.2637 (strong patterns)

**Global Orthogonal** (non-standardized):
```python
X = np.stack([r, g, b, d], axis=1)
X_centered = X - X.mean(axis=0)  # Just centering, NO standardization

U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
```

Result: std = 0.0033, 0.0503, 0.0446 (weaker patterns)

#### Preprocessed Files

Use **non-standardized** version (same as global in test, but this is correct).

### 2. Matplotlib Auto-Scaling

Test visualizations use:
```python
imshow(orth_values, cmap='RdBu_r')  # No vmin/vmax → auto-scale!
```

Matplotlib automatically scales the colormap to the data range:
- Min value → Deepest blue
- Max value → Deepest red
- Makes even small variations visible with full color range

When I created the "gray appearance explained" visualization, I used fixed ranges:
```python
imshow(orth_values, cmap='RdBu_r', vmin=-0.3, vmax=0.3)  # Fixed range
```

This is more representative of the actual magnitude of values.

### 3. Data Format

**Test images**: Matplotlib plots (RGBA, uint8) with applied colormaps
**Preprocessed files**: 16-bit grayscale data (uint16) without colormaps

When you open a 16-bit PNG in an image viewer, it maps [0, 65535] to [black, white], and since values cluster around the midpoint (32767), they appear gray.

---

## Mathematical Explanation

### Non-Standardized (Current Implementation)

Finds the orthogonal direction in **natural RGBD space**:
- RGB values: [0, 1] range
- Depth values: [0, 1] range
- Natural scales preserved
- RGB and Depth contribute based on their actual variance

**Pros**:
- Mathematically cleaner
- Respects natural scales
- Standard PCA approach

**Cons**:
- If one dimension has much larger variance, it dominates
- Smaller orthogonal values (lower std)

### Standardized (Like Local Method)

Finds the orthogonal direction in **standardized space**:
- Each dimension scaled to mean=0, std=1
- All dimensions given equal weight
- RGB and Depth contribute equally regardless of variance

**Pros**:
- Equal importance to all dimensions
- Stronger patterns (higher std)
- More visual variation

**Cons**:
- Loses information about natural scales
- Depth artificially amplified if it has lower variance
- Less standard approach

---

## Visual Comparison

See `tests/standardized_vs_nonstandardized_comparison.png` for side-by-side comparison showing:
1. RGB (original)
2. Depth (original)
3. Non-standardized orthogonal (current implementation)
4. Standardized orthogonal (like local method)

Both use auto-scaled colormaps like the test images.

---

## Which Should You Use?

### Option A: Keep Non-Standardized (RECOMMENDED) ✅

**Use case**: Standard mathematical approach
```python
# Training
train_dataset = SUNRGBDDataset(train=True, compute_orth_online=True)

# This will use non-standardized version (current implementation)
```

**Advantages**:
- Mathematically standard
- Natural scales preserved
- Consistent with most PCA literature

**Disadvantages**:
- Smaller variance (may be harder for network to learn)
- Weaker visual patterns

### Option B: Switch to Standardized

**Use case**: Want stronger patterns like local method
```python
# Would require updating SUNRGBDDataset.extract_orthogonal_stream()
# to add standardization step before SVD
```

**Advantages**:
- Stronger patterns (higher std = 0.15-0.20)
- Equal weight to all dimensions
- More visual variation

**Disadvantages**:
- Less standard mathematically
- Artificially amplifies lower-variance dimensions

---

## My Recommendation

**Keep the non-standardized version** for these reasons:

1. **Mathematically cleaner**: Standard PCA approach
2. **Natural scales**: RGB and Depth contribute based on actual variance
3. **Consistent**: Matches standard implementations
4. **Neural network can learn**: The network will learn to weight the orthogonal stream appropriately

The visual difference you see in the test images is primarily due to:
- Local using standardization (you chose global anyway)
- Auto-scaled colormaps making small variations visible

Your preprocessed files are **correct** - they just look gray when viewed directly because:
- 16-bit data format
- Values cluster around midpoint
- No colormap applied

When the neural network processes them, it sees the full range of values with proper spatial patterns.

---

## If You Want to Try Standardized Version

If you want to experiment with standardization, I can update the `extract_orthogonal_stream()` method in SUNRGBDDataset to add this option:

```python
@staticmethod
def extract_orthogonal_stream(rgb_tensor, depth_tensor, standardize=False):
    # ... denormalize ...

    X = np.stack([r, g, b, d], axis=1)
    X_mean = X.mean(axis=0)
    X_centered = X - X_mean

    if standardize:
        X_std = X_centered.std(axis=0) + 1e-8
        X_centered = X_centered / X_std  # Standardize

    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    # ... rest of computation ...
```

Then you could compare both versions during training to see which performs better.

---

## Summary Table

| Aspect | Test Visualizations | Preprocessed Files |
|--------|-------------------|-------------------|
| Local Method | Standardized | N/A (we chose global) |
| Global Method | Non-standardized | Non-standardized ✅ |
| Colormap | Auto-scaled | None (raw data) |
| Format | RGBA uint8 plot | 16-bit grayscale data |
| Visual Appearance | Colorful patterns | Gray (when viewed directly) |
| Data Correctness | ✅ Correct | ✅ Correct |
| Std (typical) | Local: 0.15-0.26, Global: 0.003-0.05 | 0.05-0.10 (normalized) |

Both are correct - just different visualization and computation approaches!
