# 🚨 URGENT FIXES FOR COLAB TRAINING PLATEAU

## Problem Summary
Your training is plateauing at 37% validation accuracy with exploding loss because of configuration issues.

---

## Fix 1: Update Diagnostic Check (CRITICAL)

**Location:** Notebook cell "Pre-Training Diagnostics"

**FIND THIS (line 496):**
```python
# CRITICAL CHECK: Labels must be in [0, num_classes-1]
if labels.min() < 0 or labels.max() >= 13:
    raise ValueError(f"Labels out of range! Expected [0, 12], got [{labels.min()}, {labels.max()}]")
print(f"   ✅ Labels are in valid range [0, 12]")
```

**REPLACE WITH:**
```python
# CRITICAL CHECK: Labels must be in [0, num_classes-1]
num_classes = DATASET_CONFIG['num_classes']  # Use the actual num_classes (27)
if labels.min() < 0 or labels.max() >= num_classes:
    raise ValueError(f"Labels out of range! Expected [0, {num_classes-1}], got [{labels.min()}, {labels.max()}]")
print(f"   ✅ Labels are in valid range [0, {num_classes-1}]")
```

**Why:** The hardcoded check for 13 classes doesn't catch labels 13-26, which will cause CUDA index errors!

---

## Fix 2: Lower Initial Learning Rate

Your learning rate (0.1) might be too high for this small dataset (only 1159 training samples).

**Location:** Notebook cell "Compile Model"

**CHANGE:**
```python
TRAINING_CONFIG = {
    'optimizer': 'sgd',
    'learning_rate': 0.01,  # ← REDUCE from 0.1 to 0.01
    'weight_decay': 1e-4,
    'momentum': 0.9,
    'loss': 'cross_entropy',
    'scheduler': 'cosine'
}
```

**Why:** With only 10 batches per epoch, LR=0.1 causes huge gradient steps that destabilize training.

---

## Fix 3: Add Learning Rate Warmup

The model needs time to stabilize before using full learning rate.

**Location:** After model.compile(), before model.fit()

**ADD THIS CELL:**
```python
# Warmup: Train for 5 epochs with low LR to stabilize
print("=" * 60)
print("WARMUP TRAINING (5 epochs)")
print("=" * 60)

warmup_history = model.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=5,
    verbose=True
)

print(f"\n✅ Warmup complete!")
print(f"   Final warmup val_acc: {warmup_history['val_accuracy'][-1]*100:.2f}%")

# Now increase LR and train with scheduler
print("\nIncreasing learning rate for main training...")
for param_group in model.optimizer.param_groups:
    param_group['lr'] = 0.05  # Increase to 0.05 (still lower than 0.1)

print("=" * 60)
print("MAIN TRAINING")
print("=" * 60)
```

**Then modify the main training epochs:**
```python
TRAIN_CONFIG = {
    'epochs': 85,  # ← Reduce by 5 (since we did 5 warmup epochs)
    'grad_clip_norm': 5.0,
    'early_stopping': True,
    'patience': 15,
    'min_delta': 0.001,
    'monitor': 'val_accuracy',
    'restore_best_weights': True,
    'save_path': f"{checkpoint_dir}/best_model.pt"
}
```

---

## Fix 4: Reduce Gradient Clipping

Your gradient clipping at 5.0 might be too aggressive.

**Location:** TRAIN_CONFIG

**CHANGE:**
```python
TRAIN_CONFIG = {
    'epochs': 90,
    'grad_clip_norm': 1.0,  # ← REDUCE from 5.0 to 1.0
    'early_stopping': True,
    'patience': 20,  # ← INCREASE patience
    'min_delta': 0.0005,  # ← Make more sensitive
    'monitor': 'val_accuracy',
    'restore_best_weights': True,
    'save_path': f"{checkpoint_dir}/best_model.pt"
}
```

---

## Fix 5: Check for Class Imbalance

You might have severe class imbalance causing the model to ignore rare classes.

**ADD THIS DIAGNOSTIC CELL** (after loading dataset):

```python
import torch
import numpy as np

print("=" * 60)
print("CLASS DISTRIBUTION ANALYSIS")
print("=" * 60)

# Count labels in training set
all_labels = []
for _, _, labels in train_loader:
    all_labels.extend(labels.numpy())

all_labels = np.array(all_labels)
unique, counts = np.unique(all_labels, return_counts=True)

print(f"\nClass distribution in training set:")
print(f"{'Class':<20} {'Count':>8} {'Percent':>8}")
print("-" * 40)

scene_names = train_loader.dataset.scene_names
for label_idx, count in zip(unique, counts):
    percent = count / len(all_labels) * 100
    scene_name = scene_names[label_idx] if label_idx < len(scene_names) else f"Unknown_{label_idx}"
    print(f"{scene_name:<20} {count:>8} {percent:>7.2f}%")

print("-" * 40)
print(f"Total samples: {len(all_labels)}")
print(f"Unique classes: {len(unique)}")
print(f"Min samples per class: {counts.min()}")
print(f"Max samples per class: {counts.max()}")
print(f"Imbalance ratio: {counts.max() / counts.min():.2f}x")

# Check if we're missing any classes
if len(unique) < 27:
    missing = set(range(27)) - set(unique)
    print(f"\n⚠️  WARNING: Missing classes in training set: {missing}")

print("\n" + "=" * 60)
```

**If imbalanced:** You might need class weights in the loss function.

---

## Fix 6: Verify Model Output Shape

Make sure the model is outputting 27 classes, not 13.

**ADD TO DIAGNOSTICS:**
```python
# Check model output
print("\nModel output check:")
model.eval()
with torch.no_grad():
    rgb, depth, _ = next(iter(train_loader))
    outputs = model(rgb.cuda(), depth.cuda())
    print(f"  Model output shape: {outputs.shape}")
    print(f"  Expected: torch.Size([{DATASET_CONFIG['batch_size']}, 27])")

    if outputs.shape[1] != 27:
        raise ValueError(f"Model outputs {outputs.shape[1]} classes, expected 27!")
```

---

## Fix 7: Try Adam Optimizer Instead of SGD

SGD can be finicky with small datasets. Adam is more forgiving.

**CHANGE:**
```python
TRAINING_CONFIG = {
    'optimizer': 'adam',  # ← CHANGE from 'sgd'
    'learning_rate': 0.001,  # ← Adam uses much lower LR
    'weight_decay': 1e-4,
    # Remove momentum (not used in Adam)
    'loss': 'cross_entropy',
    'scheduler': 'cosine'
}
```

---

## 🎯 RECOMMENDED: Apply These in Order

**Minimal Fix (5 minutes):**
1. Fix diagnostic check (Fix 1) - CRITICAL
2. Lower learning rate to 0.01 (Fix 2)
3. Reduce grad clip to 1.0 (Fix 4)
4. Rerun training

**Better Fix (10 minutes):**
1. All minimal fixes above
2. Add class distribution analysis (Fix 5)
3. Verify model output shape (Fix 6)
4. If still bad, switch to Adam (Fix 7)

**Best Fix (15 minutes):**
1. All fixes above
2. Add warmup training (Fix 3)
3. Increase patience to 20 epochs

---

## 🔍 What to Look For After Fixes

**Good signs:**
- Validation loss should be < 10 by epoch 10
- Validation accuracy should be > 50% by epoch 20
- Training accuracy should be > 80% by epoch 50
- Loss should decrease smoothly (not jump around)

**Bad signs:**
- Val loss > 100 (model is broken)
- Val accuracy stuck at same value for 10+ epochs
- Training accuracy < 50% (underfitting)

---

## 📊 Expected Performance After Fixes

**With ResNet18 + batch_size=128:**
- Baseline (broken): 37% val accuracy (what you have now)
- After fixes: 55-65% val accuracy
- With ResNet50: 65-75% val accuracy

**Validation loss should stabilize around 1.5-2.5** (not 100+!)

---

## ⚠️ If Still Not Working

1. **Restart Colab runtime completely**
2. **Reclone the repository** (make sure you have latest code)
3. **Check that you're using num_classes=27** in both dataset AND model
4. **Run the class distribution analysis** to check for severe imbalance
5. **Try a smaller learning rate** (0.001 or 0.0001)
6. **Switch to Adam optimizer**

---

## 🚀 Quick Copy-Paste Block

Here's the minimal fix you can copy-paste:

```python
# FIX 1: Correct diagnostic check
num_classes = DATASET_CONFIG['num_classes']
if labels.min() < 0 or labels.max() >= num_classes:
    raise ValueError(f"Labels out of range! Expected [0, {num_classes-1}], got [{labels.min()}, {labels.max()}]")

# FIX 2 & 7: Use Adam with lower LR
TRAINING_CONFIG = {
    'optimizer': 'adam',
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'loss': 'cross_entropy',
    'scheduler': 'cosine'
}

# FIX 4: Better training config
TRAIN_CONFIG = {
    'epochs': 90,
    'grad_clip_norm': 1.0,  # Reduced
    'early_stopping': True,
    'patience': 20,  # Increased
    'min_delta': 0.0005,
    'monitor': 'val_accuracy',
    'restore_best_weights': True,
    'save_path': f"{checkpoint_dir}/best_model.pt"
}
```

Then recompile and retrain!
