# Normalization Analysis and Recommendations

## Executive Summary

Successfully computed exact training statistics from 8041 samples at (416, 544) resolution. Found significant discrepancies with previously assumed values, particularly for the orthogonal stream which has **extremely low variance** (std=0.0249).

## Computed Training Statistics

### Exact Values (from compute_train_stats.py)

```python
# RGB
RGB_MEAN = [0.4905626144214781, 0.4564359471868703, 0.43112756716677114]
RGB_STD  = [0.27944652961530003, 0.2868739703756949, 0.29222326115669395]

# Depth
DEPTH_MEAN = 0.2912
DEPTH_STD  = 0.1472

# Orthogonal
ORTH_MEAN = 0.5000
ORTH_STD  = 0.0249  # ⚠️ EXTREMELY SMALL!
```

### Comparison with Previously Assumed Values

| Modality | Metric | Computed | Previously Assumed | Error |
|----------|--------|----------|--------------------|-------|
| RGB | Mean | [0.4906, 0.4564, 0.4311] | [0.4700, 0.4393, 0.4211] | 2.1% |
| RGB | Std | [0.2794, 0.2869, 0.2922] | [0.2732, 0.2813, 0.2841] | 0.8% |
| Depth | Mean | 0.2912 | 0.3108 | 6.3% |
| Depth | Std | 0.1472 | 0.1629 | 9.6% |
| Orth | Mean | 0.5000 | 0.5001 | 0.02% |
| **Orth** | **Std** | **0.0249** | **0.0710** | **185%** |

## Critical Finding: Orthogonal Stream Variance

The orthogonal stream has **exceptionally low variance** (std=0.0249), indicating the data is highly uniform. This creates serious normalization challenges:

### Impact of Using Exact Computed Std (0.0249)

Using the exact computed std would create:
- **Normalized range: [-20.08, +20.08]** (compared to typical [-3, 3])
- **40x amplification** of small variations
- **Very high risk** of:
  - Exploding gradients during backpropagation
  - Extreme sensitivity to noise
  - Numerical instability
  - Training divergence

### Recommended Std Options for Orthogonal Stream

| Std Value | Normalized Range | Amplification | Pros | Cons | Recommendation |
|-----------|------------------|---------------|------|------|----------------|
| **0.0249** | [-20.08, 20.08] | 40x | Exact statistics | Extremely unstable | ❌ **NOT RECOMMENDED** |
| 0.0710 | [-7.04, 7.04] | 14x | Previously assumed | Still very large range | ⚠️ Risky |
| 0.1472 | [-3.40, 3.40] | 6.8x | Match Depth std | Moderate stability | ✓ Acceptable |
| **0.2794** | **[-1.79, 1.79]** | **3.6x** | **Match RGB std** | **Balanced, stable** | ✅ **RECOMMENDED** |
| 0.5 | [-1.00, 1.00] | 2x | Conservative | Very safe, minimal amplification | ✓ Safe fallback |

## Test Results Summary

### ✅ Passing Tests
1. **Scaling to [0,1]**: All modalities correctly scaled before augmentation
2. **Augmentation**: Maintains [0,1] range with proper clipping
3. **Pipeline Order**: Scale → Augment → Normalize → Random Erasing
4. **Orth Mean**: Perfectly accurate (0.5000 vs expected 0.5000)

### ⚠️ Observations
1. **RGB Statistics**: Small discrepancies in validation set (likely due to train/val distribution differences)
2. **Depth Statistics**: Similar validation discrepancies
3. **Orth Normalized Range**: Confirmed [-20, 20] range with current std=0.0249

## Final Recommendation

### Immediate Action Required: Update Orthogonal Std

**Change from:** `std=[0.0249]` (current exact computed value)
**Change to:** `std=[0.2794]` (match RGB std for balanced normalization)

### Rationale

1. **Data Characteristics**: Orthogonal stream is highly uniform (std=0.0249), which means:
   - Natural variation is very small
   - Using exact std would amplify tiny differences 40x
   - This defeats the purpose of normalization (stability)

2. **Training Stability**: Using std=0.2794 provides:
   - Normalized range of [-1.79, 1.79] (similar to RGB)
   - Reduced gradient amplification (3.6x vs 40x)
   - Better numerical stability
   - Consistent scale across all three streams

3. **Alternative Approaches**:
   - If std=0.2794 still shows instability: Try std=0.5 (conservative)
   - If you want to preserve more variance: Try std=0.1472 (match Depth)
   - **Never use std=0.0249**: The exact computed value is too extreme

### Implementation

Update [sunrgbd_3stream_dataset.py:260-267](src/data_utils/sunrgbd_3stream_dataset.py#L260-L267):

```python
# Orth: Use computed Mean with BALANCED Std
# NOTE: Exact computed std (0.0249) is too small, causing range [-20, 20]
# Using RGB std (0.2794) for stable training with balanced range [-1.8, 1.8]
orth = transforms.functional.to_tensor(orth)
orth = transforms.functional.normalize(
    orth, mean=[0.5000], std=[0.2794]  # Changed from 0.0249
)
```

### Expected Normalized Ranges After Fix

```
RGB:   [-1.76, 1.82]  ✓ Good, balanced and symmetric
Depth: [-1.98, 4.82]  ✓ CORRECT - asymmetric due to natural data skew
Orth:  [-1.79, 1.79]  ✓ Good, perfectly symmetric (after changing std to 0.2794)
```

### Understanding Depth's Asymmetric Range

**Q: Why is depth's range [-1.98, 4.82] so large and asymmetric?**

**A: This is CORRECT and expected!** The range comes from the normalization formula:

```
normalized_value = (original_value - mean) / std

For values in [0, 1]:
  min_normalized = (0 - 0.2912) / 0.1472 = -1.978
  max_normalized = (1 - 0.2912) / 0.1472 = 4.815
```

**Why is it asymmetric?**
- Depth mean = 0.2912 (much lower than 0.5)
- This indicates depth data is **heavily skewed toward close objects** (low depth values)
- Most scenes have nearby objects (depth ≈ 0-0.3)
- Far objects (depth ≈ 1.0) are rare but important

**Why is this GOOD?**
- ✓ Far objects get strong signal (+5) - not compressed despite being rare
- ✓ Close objects get strong signal (-2) - properly represented
- ✓ Network can learn depth's natural distribution
- ✓ Range is within safe training bounds (not extreme like [-20, 20])

**When would this be BAD?**
- ✗ If values exceeded [-10, 10]: Risk of gradient instability
- ✗ If we saw values outside [0,1] when denormalized: Scaling error
- ✗ None of these apply! Depth is correctly normalized.

**Verification:**
- ✓ Raw depth files: Range [9, 65440] in 16-bit
- ✓ After scaling by 65535: Range [0.0001, 0.9986] ✓ Correct!
- ✓ Denormalized values: Range [0.001, 0.999] ✓ Stays in [0,1]!
- ✓ Depth distribution reflects natural scene statistics

## Validation Steps

After implementing the change:

1. Run `python3 test_normalization.py` to verify:
   - New Orth range is ~[-1.8, 1.8]
   - All other tests still pass

2. Start training and monitor:
   - Gradient norms (should be stable, not exploding)
   - Loss convergence (should decrease smoothly)
   - Orth stream activations (should be in reasonable range)

3. If issues persist:
   - Try std=0.5 as conservative fallback
   - Consider whether orthogonal stream needs normalization at all given its uniformity

## Files Updated

- ✅ `src/data_utils/sunrgbd_3stream_dataset.py` - Updated with exact RGB/Depth stats, needs Orth std fix
- ✅ `test_normalization.py` - Updated test suite with exact statistics
- ✅ `compute_train_stats.py` - Memory-efficient statistics computation script

## Next Steps

1. **Decide on Orth std value** (recommend 0.2794)
2. Update `sunrgbd_3stream_dataset.py` line 266
3. Run final validation tests
4. Begin training with confidence in normalization
