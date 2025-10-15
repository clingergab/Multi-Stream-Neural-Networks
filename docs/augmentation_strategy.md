# SUN RGB-D Augmentation Strategy

**Updated**: 2025-10-14
**Status**: ✅ Implemented (Conservative & Balanced)

## 🎯 Goal

Add **moderate augmentation** to combat 28% train/val gap (overfitting) without over-augmenting.

## 📊 Augmentation Pipeline

### 1. **Synchronized Augmentation** (Both RGB + Depth)
Preserves geometric alignment between modalities.

| Augmentation | Probability | Params | Rationale |
|-------------|-------------|---------|-----------|
| **Random Horizontal Flip** | 50% | - | Common indoor transformation |
| **Random Resized Crop** | 100% | scale=(0.8, 1.0)<br>ratio=(0.95, 1.05) | **CRITICAL** - Forces spatial invariance<br>Conservative scale (80-100%) |

### 2. **RGB-Only Augmentation** (Independent)
Changes appearance without affecting geometry.

| Augmentation | Probability | Params | Rationale |
|-------------|-------------|---------|-----------|
| **Color Jitter** | 50% | brightness=0.2<br>contrast=0.2<br>saturation=0.2<br>hue=0.05 | Moderate lighting variations<br>Conservative values (±20%) |
| **Random Grayscale** | 5% | - | Rare - tests color robustness<br>Forces learning beyond color |
| **Random Erasing** | 10% | scale=(0.02, 0.1)<br>ratio=(0.5, 2.0) | Small occlusions (2-10% of image)<br>Post-normalization |

### 3. **Depth-Only Augmentation** (Independent)
Simulates sensor artifacts.

| Augmentation | Probability | Params | Rationale |
|-------------|-------------|---------|-----------|
| **Gaussian Noise** | 20% | mean=0, std=3 | Light sensor noise<br>Conservative std=3 |
| **Random Erasing** | 5% | scale=(0.02, 0.1) | Simulates missing depth readings<br>Applied 50% of RGB erasing time |

## 🔬 Design Principles

### ✅ Conservative by Design
- **Scale**: 0.8-1.0 (not aggressive like 0.5-1.0)
- **Color Jitter**: ±20% (not ±50%)
- **Probabilities**: 50% or less for most augmentations
- **Erasing**: 2-10% patches (not 10-33%)

### ✅ Synchronized Where Necessary
```python
# CORRECT: Same crop for both modalities
i, j, h, w = get_crop_params(rgb)
rgb_cropped = crop(rgb, i, j, h, w)
depth_cropped = crop(depth, i, j, h, w)  # Same i,j,h,w
```

### ✅ Independent Where Beneficial
```python
# CORRECT: Different appearance augmentations
rgb = color_jitter(rgb)    # RGB gets color changes
depth = add_noise(depth)   # Depth gets sensor noise
```

## 📈 Expected Impact

### Before (Weak Augmentation):
```
Only horizontal flip (50%)
Train acc: 95.2% (memorizing)
Val acc: 67.0%
Gap: 28.2% 🔴 (severe overfitting)
Stream1: 49.5%, Stream2: 49.9% (weak features)
```

### After (Balanced Augmentation):
```
Horizontal flip + Crop + ColorJitter + Noise + Erasing
Train acc: 78-82% (learning patterns, not memorizing)
Val acc: 70-73%
Gap: 8-12% ✅ (healthy)
Stream1: 55-58%, Stream2: 52-55% (stronger features)
```

## 🎛️ Augmentation Strength Levels

### Current Implementation: **LEVEL 2 (Moderate)**

| Level | Scale | Color Jitter | Noise | Erasing | Use Case |
|-------|-------|--------------|-------|---------|----------|
| Level 1 (Weak) | (0.9, 1.0) | brightness=0.1 | std=2 | 5% | Initial testing |
| **Level 2 (Moderate)** | **(0.8, 1.0)** | **brightness=0.2** | **std=3** | **10%** | **Current (Balanced)** |
| Level 3 (Strong) | (0.7, 1.0) | brightness=0.3 | std=5 | 20% | If still overfitting |
| Level 4 (Aggressive) | (0.5, 1.0) | brightness=0.4 | std=8 | 30% | Only if desperate |

**Recommendation**: Start with Level 2, move to Level 3 only if train/val gap > 15%.

## 🧪 Testing the Augmentation

Add this to your notebook to visualize augmented samples:

```python
import matplotlib.pyplot as plt

# Get a batch from training set
train_loader, _ = get_sunrgbd_dataloaders(batch_size=8)
rgb_batch, depth_batch, labels = next(iter(train_loader))

# Visualize 4 samples
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for i in range(4):
    # Denormalize RGB
    rgb = rgb_batch[i].cpu()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    rgb = rgb * std + mean
    rgb = torch.clamp(rgb, 0, 1)

    # Denormalize Depth
    depth = depth_batch[i].cpu()
    depth = depth * 0.2197 + 0.5027

    axes[0, i].imshow(rgb.permute(1, 2, 0))
    axes[0, i].set_title(f"RGB Aug - Class {labels[i]}", fontsize=10)
    axes[0, i].axis('off')

    axes[1, i].imshow(depth.squeeze(), cmap='viridis')
    axes[1, i].set_title(f"Depth Aug - Class {labels[i]}", fontsize=10)
    axes[1, i].axis('off')

plt.suptitle("Augmented Training Samples", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

**What to look for:**
- ✅ RGB images should have varied crops, slight color changes
- ✅ Depth images should have same crops as RGB
- ✅ Some depth images may have slight noise (barely visible)
- ❌ Crops should NOT be misaligned between RGB and Depth

## 📝 Implementation Details

### File Modified:
- [`src/data_utils/sunrgbd_dataset.py`](../src/data_utils/sunrgbd_dataset.py) - Lines 143-214

### Key Changes:
1. **Removed reliance on `self.rgb_transform`** for augmentation
2. **All augmentation now in `__getitem__`** for fine-grained control
3. **Synchronized transforms use same crop parameters**
4. **Independent transforms applied separately to RGB/Depth**

### Validation Behavior:
```python
if not self.train:
    # Validation: ONLY resize, NO augmentation
    rgb = resize(rgb)
    depth = resize(depth)
```

This ensures validation set remains consistent across epochs for fair comparison.

## 🔄 Reverting to Weak Augmentation

If augmentation proves too strong (train acc < 70%), edit lines 152-154:

```python
# Make crop less aggressive
i, j, h, w = transforms.RandomResizedCrop.get_params(
    rgb, scale=(0.9, 1.0),  # ← Change from (0.8, 1.0)
    ratio=(0.95, 1.05)
)
```

Or reduce color jitter at line 161:

```python
# Make color changes gentler
color_jitter = transforms.ColorJitter(
    brightness=0.1,  # ← Change from 0.2
    contrast=0.1,
    saturation=0.1,
    hue=0.02
)
```

## ✅ Next Steps

1. **Test augmentation visually** (use code snippet above)
2. **Run 20 epochs** with new augmentation
3. **Monitor train accuracy**:
   - If train_acc = 75-80% → Augmentation is working perfectly ✅
   - If train_acc < 70% → Too aggressive, reduce to Level 1
   - If train_acc > 85% → Too weak, increase to Level 3
4. **Check val accuracy improves** (target: 70-73%)
5. **Verify train/val gap reduces** (target: < 15%)

## 🎯 Success Criteria

| Metric | Before | Target | Status |
|--------|--------|--------|--------|
| Train Accuracy | 95.2% | 78-82% | Pending |
| Val Accuracy | 67.0% | 70-73% | Pending |
| Train/Val Gap | 28.2% | 8-12% | Pending |
| Stream1 Val Acc | 49.5% | 55-58% | Pending |
| Stream2 Val Acc | 49.9% | 52-55% | Pending |

**Overall Goal**: Reduce overfitting while improving generalization!
