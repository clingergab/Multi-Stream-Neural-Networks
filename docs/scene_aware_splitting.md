# Scene-Aware Train/Val Splitting

## Problem

Both SUN RGB-D and NYU Depth V2 contain multiple images from the same physical scene (room/location). A random train/val split during HPO puts images from the same room into both sets, causing data leakage that inflates validation metrics.

## Datasets Affected

### SUN RGB-D (19-category)

| Metric | Before (random) | After (SGKF) |
|--------|-----------------|---------------|
| Training samples | 4,845 | 4,845 |
| Unique scenes | 3,531 | 3,531 |
| Train/Val split | 3,876 / 969 | 3,898 / 947 |
| Leaked scene groups | **84** | **0** |
| Val samples from leaked scenes | **286/969 (29.5%)** | **0 (0%)** |
| All classes in val | Yes (19/19) | Yes (19/19) |

**Source of leakage**: The `xtion/sun3ddata/` source has 207 scene directories contributing 1,563 samples (~6.0 images per scene). Largest: `home_ac` (87 images from one house).

**Split parameters**: `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=152)`
- Seed 152 chosen because seed 42 puts 0 `lab` samples in val (lab has only 40 training samples across 11 scene groups)
- Scene groups extracted by `extract_scene_group()` in `scripts/preprocess_sunrgbd_19.py`

### NYU Depth V2 (10-category)

| Metric | Before (random) | After (SGKF) |
|--------|-----------------|---------------|
| Training samples | 795 | 795 |
| Unique scenes | 249 | 249 |
| Train/Val split | 636 / 159 | 637 / 158 |
| Leaked scene groups | **105** | **0** |
| Val samples from leaked scenes | **144/159 (90.6%)** | **0 (0%)** |
| All classes in val | Yes (10/10) | Yes (10/10) |

**Source of leakage**: NYU Depth V2 has 795 training images from only 249 unique scenes (avg 3.2 per scene). 97.5% of training samples come from multi-image scenes. The largest scene (`furniture_store_0001`) has 16 images.

**Split parameters**: `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=60)`
- Seed 60 chosen for closest match to 20% val ratio (val=158 vs target 159) with all 10 classes represented

## Implementation

### Preprocessing

Both preprocessing scripts now save `scene_groups.json` and `sample_paths.json` alongside tensors:

```
data/{dataset}_traintest/train/
  rgb_tensors.pt
  depth_tensors.pt
  labels.txt
  scene_groups.json    # scene ID per tensor index
  sample_paths.json    # original path/index for traceability
```

### HPO Notebooks

All HPO notebooks use the "Pass the Blueprint" pattern:
1. Split computed **once** outside the trainable function
2. Indices passed via `tune.with_parameters(train_indices=..., val_indices=...)`
3. `StratifiedGroupKFold` ensures both group integrity and class stratification

### Files Modified

- `scripts/preprocess_sunrgbd_19.py` - added `extract_scene_group()`, saves scene metadata
- `scripts/preprocess_nyu_depth_v2.py` - returns scene names, saves scene metadata
- `scripts/analyze_scene_groups.py` - standalone SUN RGB-D leakage analysis tool
- 8 SUN RGB-D notebooks + 2 NYU notebooks - scene-aware splitting + Blueprint pattern
