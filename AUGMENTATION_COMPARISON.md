# Data Augmentation Comparison: 2-Stream vs 3-Stream

## Summary

Updated the 3-stream dataset to have **consistent augmentation** with the 2-stream dataset. The key change: **Orth now receives the same appearance augmentation as Depth**.

## Augmentation Pipeline Comparison

### Geometric Augmentations (Applied to ALL modalities)

| Augmentation | 2-Stream (RGB+Depth) | 3-Stream (RGB+Depth+Orth) | Notes |
|--------------|---------------------|---------------------------|-------|
| Horizontal Flip | 50% both | 50% all three | ✓ Synchronized |
| Random Resized Crop | 50% both | 50% all three | ✓ Synchronized, scale=(0.9, 1.0) |
| Resize (no crop) | 50% both | 50% all three | ✓ Alternative to crop |

### RGB-Only Augmentations

| Augmentation | 2-Stream | 3-Stream | Parameters |
|--------------|----------|----------|------------|
| Color Jitter | 43% | 43% | brightness=0.37, contrast=0.37, sat=0.37, hue=0.11 |
| Gaussian Blur | 25% | 25% | kernel=[3,5,7], sigma=[0.1, 1.7] |
| Grayscale | 17% | 17% | num_output_channels=3 |

### Depth & Orth Appearance Augmentations

| Augmentation | 2-Stream (Depth) | 3-Stream (Depth) | 3-Stream (Orth) |
|--------------|------------------|------------------|-----------------|
| **Probability** | 50% | 50% | 50% (same block as depth) |
| **Brightness** | ±25% on [0,255] | ±25% on [0,1] | ±25% on [0,1] |
| **Contrast** | ±25% on [0,255] | ±25% on [0,1] | ±25% on [0,1] |
| **Gaussian Noise** | σ=15 on [0,255] | σ=0.06 on [0,1] | σ=0.06 on [0,1] |
| **Applied Together** | ✓ Yes | ✓ Yes | ✓ Yes |

**Key Change**: Orth now receives appearance augmentation! Previously it was skipped.

### Post-Normalization Random Erasing

| Augmentation | 2-Stream | 3-Stream | Parameters |
|--------------|----------|----------|------------|
| RGB Erasing | 17% | 17% | scale=(0.02, 0.10), ratio=(0.5, 2.0) |
| Depth Erasing | 10% | 10% | scale=(0.02, 0.1), ratio=(0.5, 2.0) |
| Orth Erasing | N/A | 10% | scale=(0.02, 0.1), ratio=(0.5, 2.0) |

## Detailed Augmentation Flow

### 2-Stream Dataset (RGB + Depth)

```
1. Load RGB (uint8 [0,255]) and Depth (uint16 → converted to uint8 [0,255])

IF TRAINING:
  2. Random Horizontal Flip (50%) - synchronized
  3. Random Resized Crop OR Resize (50% each) - synchronized
  4. RGB Color Jitter (43%)
  5. RGB Gaussian Blur (25%)
  6. RGB Grayscale (17%)
  7. Depth Appearance Aug (50%):
     - Brightness ±25%
     - Contrast ±25%
     - Gaussian Noise σ=15
     [All on [0,255] range]

8. Convert to tensor + normalize:
   - RGB: to_tensor [0,1] → ImageNet norm
   - Depth: to_tensor [0,1] → custom norm (mean=0.5027, std=0.2197)

IF TRAINING:
  9. RGB Random Erasing (17%)
  10. Depth Random Erasing (10%)
```

### 3-Stream Dataset (RGB + Depth + Orth) - UPDATED

```
1. Load RGB (uint8 [0,255]), Depth (uint16 → [0,1]), Orth (uint16 → [0,1])

IF TRAINING:
  2. Random Horizontal Flip (50%) - synchronized to ALL three
  3. Random Resized Crop OR Resize (50% each) - synchronized to ALL three
  4. RGB Color Jitter (43%)
  5. RGB Gaussian Blur (25%)
  6. RGB Grayscale (17%)
  7. Depth & Orth Appearance Aug (50%) - NEW: Applied to BOTH:
     - Brightness ±25% (same factor for both)
     - Contrast ±25% (same factor for both)
     - Gaussian Noise σ=0.06 (independent)
     [All on [0,1] range]

8. Convert to tensor + normalize:
   - RGB: to_tensor [0,1] → normalize(mean=0.5, std=0.5) → [-1,1]
   - Depth: to_tensor [0,1] → normalize(mean=0.5, std=0.5) → [-1,1]
   - Orth: to_tensor [0,1] → normalize(mean=0.5, std=0.5) → [-1,1]

IF TRAINING:
  9. RGB Random Erasing (17%)
  10. Depth Random Erasing (10%)
  11. Orth Random Erasing (10%)
```

## Key Differences Explained

### 1. Data Range During Augmentation

**2-Stream:**
- Operates on [0, 255] range for both RGB and Depth
- Depth converted from uint16 to uint8 before augmentation

**3-Stream:**
- RGB still in PIL RGB mode [0, 255] for color aug
- Depth and Orth in [0, 1] float range (mode F)
- Brightness/contrast/noise scaled accordingly:
  - `σ=15` on [0,255] ≈ `σ=0.06` on [0,1]` (~6%)

### 2. Normalization Strategy

**2-Stream:**
- RGB: ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Depth: Custom normalization (mean=[0.5027], std=[0.2197])
- Final ranges: approximately [-2.1, 2.6] for RGB, [-2.3, 2.3] for Depth

**3-Stream:**
- ALL modalities: Consistent normalization (mean=0.5, std=0.5)
- Final range: [-1, 1] for all three modalities
- Simpler and more consistent across streams

### 3. Orth Augmentation (NEW)

**Before:** Orth skipped appearance augmentation (comment: "preserve geometric meaning")

**After:** Orth receives same augmentation as Depth:
- Same brightness/contrast factors within each sample
- Independent noise for variety
- Rationale: If Depth benefits from appearance aug, so should Orth (both are geometric)

## Augmentation Synchronization

### Always Synchronized (Same Random Seed/Decision):
1. Horizontal flip - ALL modalities
2. Crop parameters - ALL modalities
3. Brightness & Contrast factors - Depth & Orth (within single augmentation block)

### Independent (Different Random Seeds):
1. RGB color jitter
2. RGB gaussian blur
3. RGB grayscale
4. Gaussian noise for Depth
5. Gaussian noise for Orth
6. Random erasing for each modality

## Verification Results

Tested with 50 training samples with full augmentation:

```
RGB:   global range [-1.000, 1.000] ✓
Depth: global range [-1.000, 1.000] ✓
Orth:  global range [-1.000, 1.000] ✓

✓ No NaN values
✓ All augmentations working correctly
✓ Orth now augmented consistently with Depth
```

## Implementation Notes

### Brightness & Contrast on [0, 1] Range

When data is in [0, 1]:
```python
# Contrast centered at 0.5
img = (img - 0.5) * contrast_factor + 0.5

# Brightness as multiplication
img = img * brightness_factor

# Clip to valid range
img = np.clip(img, 0.0, 1.0)
```

### Noise Scaling

Equivalent noise levels:
- `σ=15` on [0, 255] → 15/255 ≈ 0.059 ≈ **0.06** on [0, 1]
- Represents ~6% noise relative to data range

## Conclusion

✓ 3-stream augmentation now consistent with 2-stream
✓ Orth receives same treatment as Depth
✓ All geometric augmentations synchronized across modalities
✓ Appearance augmentations applied independently where appropriate
✓ Consistent normalization to [-1, 1] for all modalities
