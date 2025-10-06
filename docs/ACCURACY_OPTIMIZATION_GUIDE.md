# Accuracy Optimization Guide for MCResNet on NYU Depth V2

## Current Status
- **GPU RAM Usage:** 6.1 / 80.0 GB (only 7.6% utilized!)
- **Training Speed:** ~20 seconds/epoch (very fast)
- **Model:** ResNet18
- **Batch Size:** 128

## Recommended Optimizations (Priority Order)

### 🎯 Priority 1: Use Deeper Architecture (Biggest Accuracy Gain)

**Current:** ResNet18 (11M parameters)
**Recommended:** ResNet50 (25M parameters)

```python
MODEL_CONFIG = {
    'architecture': 'resnet50',  # ← CHANGE THIS
    'num_classes': 27,
    'stream1_channels': 3,
    'stream2_channels': 1,
    'device': 'cuda',
    'use_amp': True
}
```

**Expected Impact:**
- ✅ Accuracy gain: +3-7%
- ⚠️ GPU RAM increase: 6.1 GB → ~12 GB (still only 15% of A100!)
- ⚠️ Training time: 20s/epoch → ~45s/epoch (still very reasonable)

---

### 🎯 Priority 2: Increase Batch Size (Better Gradient Estimates)

**Current:** 128
**Recommended:** 256 or 512

```python
DATASET_CONFIG = {
    'dataset_path': '/content/nyu_depth_v2_labeled.mat',
    'batch_size': 256,  # ← DOUBLE IT (or try 512)
    'num_workers': 2,
    'target_size': (224, 224),
    'num_classes': 27
}
```

**Expected Impact:**
- ✅ More stable gradients → better convergence
- ✅ Accuracy gain: +1-3%
- ✅ Faster training (fewer optimizer steps)
- ⚠️ May need to adjust learning rate: `lr = 0.1 * (batch_size / 128)`

**If using batch_size=256:**
```python
TRAINING_CONFIG = {
    'optimizer': 'sgd',
    'learning_rate': 0.2,  # ← 0.1 * (256/128) = 0.2
    'weight_decay': 1e-4,
    'momentum': 0.9,
    'loss': 'cross_entropy',
    'scheduler': 'cosine'
}
```

**If using batch_size=512:**
```python
TRAINING_CONFIG = {
    'optimizer': 'sgd',
    'learning_rate': 0.4,  # ← 0.1 * (512/128) = 0.4
    'weight_decay': 1e-4,
    'momentum': 0.9,
    'loss': 'cross_entropy',
    'scheduler': 'cosine'
}
```

---

### 🎯 Priority 3: Increase Image Resolution (More Detail)

**Current:** 224×224
**Recommended:** 384×384 or 448×448

```python
DATASET_CONFIG = {
    'dataset_path': '/content/nyu_depth_v2_labeled.mat',
    'batch_size': 128,  # ← Keep at 128 or reduce to 64 for larger images
    'num_workers': 2,
    'target_size': (384, 384),  # ← INCREASE THIS
    'num_classes': 27
}
```

**Expected Impact:**
- ✅ More spatial detail → better scene recognition
- ✅ Accuracy gain: +2-5%
- ⚠️ GPU RAM increase: ~2-4x (but you have tons of headroom!)
- ⚠️ Training time increase: ~2-3x per epoch
- ⚠️ Reduce batch size if needed (try 64 with 384×384)

---

### 🎯 Priority 4: Add Data Augmentation (Regularization)

Currently, the transforms are basic. Add stronger augmentation:

**Edit:** `src/data_utils/nyu_depth_dataset.py` around line 230

```python
def get_nyu_transforms(train: bool = True):
    """Get transforms for NYU Depth V2 dataset."""
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),  # ← ADD
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # ← ADD
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # ← ADD (replace Resize)
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # Validation: just normalize
        return transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
```

**Expected Impact:**
- ✅ Better generalization
- ✅ Accuracy gain: +1-3%
- ⚠️ Slower training per epoch (~10% slower)

---

### 🎯 Priority 5: Train Longer with Better Scheduler

**Current:** 90 epochs, cosine annealing

**Option A: Longer Training**
```python
TRAIN_CONFIG = {
    'epochs': 150,  # ← INCREASE
    'grad_clip_norm': 5.0,
    'early_stopping': True,
    'patience': 20,  # ← Increase patience
    'min_delta': 0.001,
    'monitor': 'val_accuracy',
    'restore_best_weights': True,
    'save_path': f"{checkpoint_dir}/best_model.pt"
}
```

**Option B: Warmup + Cosine (Better for large batch sizes)**

You'll need to add warmup support, or use this trick:
```python
# Train for 5 epochs with low LR (warmup)
# Then restart with full LR and cosine annealing
```

---

### 🎯 Priority 6: Use Label Smoothing (Regularization)

Add label smoothing to CrossEntropyLoss to prevent overconfidence:

**Edit:** `src/models/abstracts/abstract_model.py` around line 290

```python
# In compile method, replace:
self.criterion = nn.CrossEntropyLoss()

# With:
self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # ← ADD label_smoothing
```

**Expected Impact:**
- ✅ Better calibration
- ✅ Accuracy gain: +0.5-2%
- ✅ No performance cost

---

### 🎯 Priority 7: Mixup or CutMix (Advanced Augmentation)

This requires code changes but can give significant gains. Not recommended for first iteration.

---

## 🚀 Recommended Configuration for Maximum Accuracy

**Best bang for buck:** ResNet50 + batch_size=256 + label_smoothing

```python
# Model
MODEL_CONFIG = {
    'architecture': 'resnet50',  # ← Better architecture
    'num_classes': 27,
    'stream1_channels': 3,
    'stream2_channels': 1,
    'device': 'cuda',
    'use_amp': True
}

# Dataset
DATASET_CONFIG = {
    'dataset_path': '/content/nyu_depth_v2_labeled.mat',
    'batch_size': 256,  # ← Larger batch
    'num_workers': 2,
    'target_size': (224, 224),  # Keep same initially
    'num_classes': 27
}

# Training
TRAINING_CONFIG = {
    'optimizer': 'sgd',
    'learning_rate': 0.2,  # ← Scaled with batch size
    'weight_decay': 1e-4,
    'momentum': 0.9,
    'loss': 'cross_entropy',
    'scheduler': 'cosine'
}

TRAIN_CONFIG = {
    'epochs': 120,  # ← Train longer
    'grad_clip_norm': 5.0,
    'early_stopping': True,
    'patience': 20,  # ← More patience
    'min_delta': 0.001,
    'monitor': 'val_accuracy',
    'restore_best_weights': True,
    'save_path': f"{checkpoint_dir}/best_model.pt"
}
```

**Expected Resources:**
- GPU RAM: ~15-20 GB (still only 25% of A100!)
- Training time: ~50-60 secs/epoch
- Total time: ~2 hours

**Expected Accuracy Gain:** +5-10% over current ResNet18 baseline

---

## 🔬 Aggressive Configuration (Maximum Accuracy)

If you want to push harder:

```python
MODEL_CONFIG = {
    'architecture': 'resnet50',
    'num_classes': 27,
    'stream1_channels': 3,
    'stream2_channels': 1,
    'device': 'cuda',
    'use_amp': True
}

DATASET_CONFIG = {
    'dataset_path': '/content/nyu_depth_v2_labeled.mat',
    'batch_size': 128,  # ← Reduce for larger images
    'num_workers': 2,
    'target_size': (384, 384),  # ← Higher resolution
    'num_classes': 27
}

TRAINING_CONFIG = {
    'optimizer': 'sgd',
    'learning_rate': 0.1,
    'weight_decay': 1e-4,
    'momentum': 0.9,
    'loss': 'cross_entropy',
    'scheduler': 'cosine'
}

TRAIN_CONFIG = {
    'epochs': 150,
    'grad_clip_norm': 5.0,
    'early_stopping': True,
    'patience': 25,
    'min_delta': 0.0005,
    'monitor': 'val_accuracy',
    'restore_best_weights': True,
    'save_path': f"{checkpoint_dir}/best_model.pt"
}
```

**Expected Resources:**
- GPU RAM: ~30-40 GB (still < 50% of A100!)
- Training time: ~2-3 mins/epoch
- Total time: ~5-7 hours

**Expected Accuracy Gain:** +10-15% over current baseline

---

## 📊 Summary Table

| Change | Accuracy Gain | GPU RAM | Training Time | Difficulty |
|--------|---------------|---------|---------------|------------|
| ResNet50 | +3-7% | +6 GB | +2.5x | Easy |
| Batch 256 | +1-3% | +2 GB | -20% | Easy |
| Batch 512 | +1-3% | +4 GB | -40% | Easy |
| Image 384×384 | +2-5% | +10 GB | +2x | Easy |
| Label Smoothing | +0.5-2% | 0 | 0 | Medium |
| Data Aug | +1-3% | 0 | +10% | Medium |
| Train 150 epochs | +1-2% | 0 | +1.7x | Easy |

---

## 🎯 Implementation Steps

1. **Quick Win (5 min):**
   - Change `architecture` to `resnet50`
   - Change `batch_size` to `256`
   - Adjust `learning_rate` to `0.2`
   - Run training

2. **If you want more (15 min):**
   - Add label smoothing to loss function
   - Increase epochs to 120-150
   - Increase patience to 20-25

3. **Experimental (30 min):**
   - Increase image resolution to 384×384
   - Add stronger data augmentation
   - Reduce batch size to 128 (for larger images)

---

## ⚠️ Things to Monitor

1. **Overfitting:** If train accuracy >> val accuracy, you need more regularization
2. **Underfitting:** If both train and val accuracy are low, you need more capacity/longer training
3. **Memory:** Watch `nvidia-smi` to ensure you're not OOM
4. **Learning Rate:** If loss diverges early, reduce LR

---

## 🔍 Debugging Tips

**If accuracy doesn't improve:**
- Check that labels are correct (use visualization script)
- Verify data augmentation isn't too aggressive
- Try reducing learning rate by 2x
- Check for class imbalance (some scenes might have very few samples)

**If training is unstable:**
- Reduce learning rate
- Increase gradient clipping (try `grad_clip_norm=1.0`)
- Reduce batch size
- Add more warmup epochs

---

## 📈 Expected Baseline Performance

**Current (ResNet18, batch=128, 224×224):**
- Expected: 60-70% validation accuracy

**Optimized (ResNet50, batch=256, 224×224, label smoothing):**
- Expected: 70-80% validation accuracy

**Aggressive (ResNet50, batch=128, 384×384, strong aug):**
- Expected: 75-85% validation accuracy
