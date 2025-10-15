# Recommended Training Configuration

**Date**: 2025-10-14
**Goal**: Reduce overfitting (28% gap → 10-12%) while improving val accuracy (67% → 70-73%)

## 🎯 Complete Recommended Config

```python
DATASET_CONFIG = {
    'data_root': LOCAL_DATASET_PATH,
    'batch_size': 96,
    'num_workers': 4,
    'target_size': (416, 544),
    'num_classes': 15
}

MODEL_CONFIG = {
    'architecture': 'resnet18',
    'num_classes': 15,
    'stream1_channels': 3,  # RGB
    'stream2_channels': 1,  # Depth
    'dropout_p': 0.6,  # ↓ from 0.7 (have augmentation now)
    'device': 'cuda',
    'use_amp': True
}

STREAM_SPECIFIC_CONFIG = {
    'optimizer': 'adamw',
    'learning_rate': 7e-5,           # Base LR (unchanged)
    'weight_decay': 5e-4,            # ↓ from 2e-2 (CRITICAL FIX!)

    # Stream-specific settings (rebalanced)
    'stream1_lr': 7e-5,              # ↑ from 5e-5 (boost RGB learning)
    'stream1_weight_decay': 3e-4,    # ↓ from 1e-2 (let RGB learn!)

    'stream2_lr': 2e-4,              # ↓ from 4e-4 (reduce Depth LR)
    'stream2_weight_decay': 2e-4,    # ↓ from 4e-3 (let Depth learn!)

    'loss': 'cross_entropy',
    'scheduler': 'quadratic_inout',  # Keep (works well with your use case)
    'label_smoothing': 0.15          # Keep
}

TRAIN_CONFIG = {
    'epochs': 70,
    'grad_clip_norm': 5.0,           # ↑ from 3.0 (allow larger updates)
    'early_stopping': True,
    'patience': 12,
    'min_delta': 0.001,
    'monitor': 'val_loss',           # or 'val_accuracy' (your choice)
    'restore_best_weights': True,
    'save_path': f"{checkpoint_dir}/best_model.pt",
    'stream_monitoring': True,

    # Scheduler settings
    't_max': 60,                     # ↑ from 50 (YOUR IDEA - softer decline)
    'eta_min': 1e-7,                 # ↓ from 1e-6 (lower floor)

    # Stream Early Stopping
    'stream_early_stopping': True,
    'stream1_patience': 15,          # ↑ from 10 (more time for RGB)
    'stream2_patience': 12,          # ↑ from 10 (more time for Depth)
    'stream_min_delta': 0.001,       # ↓ from 0.0015 (more sensitive)
}
```

## 📊 What Changed and Why

### 1. **Weight Decay Reduction** (MOST IMPORTANT!)

| Parameter | Before | After | Change | Reason |
|-----------|--------|-------|--------|--------|
| `weight_decay` | 2e-2 | 5e-4 | **÷40** | Was killing classifier learning |
| `stream1_weight_decay` | 1e-2 | 3e-4 | **÷33** | Stream1 at 49.5% - too weak! |
| `stream2_weight_decay` | 4e-3 | 2e-4 | **÷20** | Stream2 at 49.9% - too weak! |

**Why this fixes overfitting:**
- Current: Streams over-regularized (49% acc) → Main classifier compensates by memorizing
- Fixed: Streams learn better features (55-58% acc) → Main classifier doesn't need to memorize

### 2. **Stream Learning Rate Rebalance**

| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| `stream1_lr` | 5e-5 | 7e-5 | RGB needs faster learning |
| `stream2_lr` | 4e-4 | 2e-4 | Depth was learning too fast, overfitting |

**Ratio before**: Stream2 was 8x faster than Stream1
**Ratio after**: Stream2 is 2.9x faster than Stream1 (more balanced)

### 3. **Dropout Reduction**

| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| `dropout_p` | 0.7 | 0.6 | Have strong augmentation now, don't need both |

Combined with augmentation, this should be plenty of regularization.

### 4. **Scheduler (Your Idea)**

| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| `t_max` | 50 | 60 | Softer LR decline (YOUR IDEA - good!) |
| `eta_min` | 1e-6 | 1e-7 | Allow lower floor |

**At epoch 26** (when Stream1 froze):
- Before: LR = 2.4e-5
- After: LR = 3.2e-5 (+33% higher!)

### 5. **Stream Early Stopping**

| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| `stream1_patience` | 10 | 15 | RGB froze too early (epoch 26) |
| `stream2_patience` | 10 | 12 | Depth can be slightly less patient |
| `stream_min_delta` | 0.0015 | 0.001 | More sensitive to improvements |

### 6. **Gradient Clipping**

| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| `grad_clip_norm` | 3.0 | 5.0 | With high WD, was limiting learning |

## 📈 Expected Results

### Training Curves

| Metric | Before | Expected After | Improvement |
|--------|--------|----------------|-------------|
| **Train Accuracy** | 95.2% | 78-82% | ↓ 13-17% (less memorization) |
| **Val Accuracy** | 67.0% | 70-73% | ↑ 3-6% (better generalization) |
| **Train/Val Gap** | 28.2% | 8-12% | ↓ 16-20% (healthy overfitting) |
| | | | |
| **Stream1 Val Acc** | 49.5% | 55-58% | ↑ 5.5-8.5% (stronger RGB) |
| **Stream2 Val Acc** | 49.9% | 52-55% | ↑ 2-5% (stronger Depth) |
| **Stream1 Freeze** | Epoch 26 | Epoch 35-45 | Later (more learning time) |

### Visual Changes

**Epoch 10:**
```
Before:
  train_loss=1.71, train_acc=0.59, val_loss=1.76, val_acc=0.58
  Stream1: V_acc:0.46, Stream2: V_acc:0.32

After (Expected):
  train_loss=1.85, train_acc=0.52, val_loss=1.72, val_acc=0.60
  Stream1: V_acc:0.52, Stream2: V_acc:0.48
  (Lower train acc is GOOD - not memorizing!)
```

**Epoch 30:**
```
Before:
  train_loss=1.30, train_acc=0.80, val_loss=1.60, val_acc=0.65
  Stream1: V_acc:0.54, Stream2: V_acc:0.43

After (Expected):
  train_loss=1.52, train_acc=0.72, val_loss=1.55, val_acc=0.70
  Stream1: V_acc:0.56, Stream2: V_acc:0.52
  (Train/val gap closing!)
```

**Epoch 60:**
```
Before:
  train_loss=1.04, train_acc=0.95, val_loss=1.53, val_acc=0.67
  Stream1: V_acc:0.55, Stream2: V_acc:0.50

After (Expected):
  train_loss=1.38, train_acc=0.80, val_loss=1.48, val_acc=0.72
  Stream1: V_acc:0.57, Stream2: V_acc:0.54
  (Much healthier!)
```

## 🔬 Monitoring During Training

### Red Flags 🚩

1. **Train acc < 65%** at epoch 20
   - Augmentation too aggressive
   - Action: Reduce crop scale to (0.9, 1.0)

2. **Stream acc < 45%** at epoch 20
   - Weight decay still too high
   - Action: Further reduce to `stream1_wd=2e-4, stream2_wd=1e-4`

3. **Train/val gap > 20%** at epoch 40
   - Need more regularization
   - Action: Increase dropout to 0.7 or label_smoothing to 0.2

### Green Flags ✅

1. **Train acc = 75-82%** at epoch 40
   - Perfect balance!

2. **Stream1 > 52%, Stream2 > 48%** at epoch 20
   - Streams learning well!

3. **Train/val gap < 15%** at epoch 40
   - Healthy generalization!

## 🎛️ Fine-Tuning Guide

### If Still Overfitting (gap > 15%):

**Option 1**: Increase augmentation strength
```python
# In sunrgbd_dataset.py, line 152-154
scale=(0.7, 1.0),  # More aggressive crop
```

**Option 2**: Increase regularization
```python
'dropout_p': 0.7,           # ↑ from 0.6
'label_smoothing': 0.2,     # ↑ from 0.15
```

### If Underfitting (train acc < 70%):

**Option 1**: Reduce augmentation
```python
# In sunrgbd_dataset.py, line 152-154
scale=(0.9, 1.0),  # Less aggressive crop
```

**Option 2**: Reduce regularization
```python
'dropout_p': 0.5,           # ↓ from 0.6
'weight_decay': 3e-4,       # ↓ from 5e-4
```

### If Stream Imbalance (one stream much weaker):

**Stream1 (RGB) weak:**
```python
'stream1_lr': 1e-4,         # ↑ from 7e-5
'stream1_weight_decay': 2e-4,  # ↓ from 3e-4
```

**Stream2 (Depth) weak:**
```python
'stream2_lr': 3e-4,         # ↑ from 2e-4
'stream2_weight_decay': 1e-4,  # ↓ from 2e-4
```

## 🚀 Implementation Steps

1. **Update augmentation** (✅ Already done in `sunrgbd_dataset.py`)

2. **Update training config** in your notebook:
   ```python
   # Copy the STREAM_SPECIFIC_CONFIG and TRAIN_CONFIG above
   # Make sure to change weight_decay values!
   ```

3. **Run 20 epochs** as a test:
   ```python
   TRAIN_CONFIG['epochs'] = 20  # Quick test
   ```

4. **Check metrics at epoch 10 and 20**:
   - Train acc should be 65-75% (not 90%+)
   - Val acc should be improving
   - Stream acc should be > 50%

5. **If test successful**, run full 70 epochs

6. **Monitor and adjust** based on flags above

## 📝 Notes

- **Weight decay change is the most critical fix** - even without augmentation changes, this alone should help significantly
- **Your t_max=60 idea is good** - keep it with quadratic_inout scheduler
- **Don't change too many things at once** - if you want to test incrementally:
  1. First: Just weight decay changes
  2. Second: Add augmentation
  3. Third: Fine-tune LR ratios

Good luck with training! 🎉
