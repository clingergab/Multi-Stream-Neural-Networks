# Optimizer Choice for MCResNet on NYU Depth V2

## Current Configuration
- **Optimizer:** SGD with momentum
- **Learning Rate:** 0.1
- **Momentum:** 0.9
- **Weight Decay:** 1e-4
- **Scheduler:** Cosine annealing

## SGD vs AdamW Analysis

### SGD with Momentum
**Advantages:**
- ✅ Better generalization (proven on ImageNet ResNets)
- ✅ Simpler, fewer hyperparameters
- ✅ Works well with cosine annealing
- ✅ Standard for ResNet architectures
- ✅ Momentum helps navigate loss landscape
- ✅ Lower memory usage (no adaptive moments)

**Disadvantages:**
- ⚠️ Requires careful LR tuning
- ⚠️ Slower initial convergence
- ⚠️ Sensitive to learning rate schedule

**Best for:**
- Long training runs (90+ epochs)
- When you have good LR schedule
- ImageNet-style tasks
- When generalization > fast convergence

### AdamW
**Advantages:**
- ✅ Faster initial convergence
- ✅ Adaptive learning rates per parameter
- ✅ Less sensitive to LR choice
- ✅ Works well with smaller datasets
- ✅ Better for fine-tuning

**Disadvantages:**
- ⚠️ Can overfit easier (especially on small datasets)
- ⚠️ Higher memory usage (2x optimizer state)
- ⚠️ May not generalize as well as SGD
- ⚠️ Requires different LR (typically 1e-3 to 1e-4)

**Best for:**
- Quick experiments
- Fine-tuning pre-trained models
- Smaller datasets
- When fast convergence matters

## NYU Depth V2 Context

**Dataset characteristics:**
- **Size:** 1,449 samples (1,159 train, 290 val)
- **Task:** Scene classification (13 classes)
- **Architecture:** MCResNet (dual-stream ResNet)
- **Training:** 90 epochs

**Key considerations:**
1. **Small dataset** → Risk of overfitting with AdamW
2. **ResNet architecture** → SGD is standard and proven
3. **From scratch training** → Need good generalization
4. **Dual-stream** → More parameters to optimize

## Recommendation

### 🏆 **Use SGD with Momentum (Current Choice is CORRECT)**

**Reasoning:**
1. **Small dataset (1,449 samples)** - AdamW would likely overfit
2. **ResNet baseline** - SGD has proven track record
3. **90 epoch training** - Enough time for SGD to converge
4. **Generalization critical** - Validation set is small (290 samples)

### Alternative Configuration (if SGD doesn't converge well)

**Option 1: AdamW for faster convergence**
```python
model.compile(
    optimizer='adamw',
    learning_rate=3e-4,  # Much lower than SGD
    weight_decay=0.01,   # Higher weight decay
    betas=(0.9, 0.999),
    scheduler='cosine'
)
```

**Option 2: SGD with warmup**
```python
model.compile(
    optimizer='sgd',
    learning_rate=0.1,
    momentum=0.9,
    weight_decay=1e-4,
    scheduler='onecycle',  # Includes warmup
    max_lr=0.1,
    pct_start=0.3  # 30% warmup
)
```

**Option 3: Hybrid approach - AdamW then SGD**
```python
# Phase 1: Quick convergence with AdamW (30 epochs)
model.compile(optimizer='adamw', learning_rate=1e-3, ...)
model.fit(epochs=30)

# Phase 2: Fine-tune with SGD (60 epochs)
model.compile(optimizer='sgd', learning_rate=0.01, ...)
model.fit(epochs=60)
```

## Benchmarks (Expected Performance)

### SGD (Current)
- **Convergence:** Slower (20-30 epochs to plateau)
- **Final accuracy:** 70-80% (better generalization)
- **Training time:** Standard
- **Validation gap:** Lower (less overfitting)

### AdamW (Alternative)
- **Convergence:** Faster (10-15 epochs to plateau)
- **Final accuracy:** 65-75% (may overfit)
- **Training time:** Similar
- **Validation gap:** Higher (more overfitting risk)

## Decision Matrix

| Criterion | SGD | AdamW | Winner |
|-----------|-----|-------|--------|
| Generalization | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | SGD |
| Convergence Speed | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | AdamW |
| Small Dataset | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | SGD |
| ResNet Architecture | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | SGD |
| Memory Usage | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | SGD |
| LR Sensitivity | ⭐⭐⭐ | ⭐⭐⭐⭐ | AdamW |
| **Overall** | **⭐⭐⭐⭐** | **⭐⭐⭐** | **SGD** |

## Final Recommendation

### ✅ **Keep SGD with current configuration**

```python
model.compile(
    optimizer='sgd',
    learning_rate=0.1,
    momentum=0.9,
    weight_decay=1e-4,
    scheduler='cosine'
)
```

**Why:**
1. Small dataset favors SGD (less overfitting)
2. ResNet standard is SGD
3. Better final generalization
4. Proven configuration for ImageNet-style tasks

### 🔄 **Try AdamW ONLY IF:**
- SGD doesn't converge after 30 epochs
- Training loss plateaus early
- Need faster prototyping

### 📊 **Monitor during training:**
- Train/val accuracy gap (> 15% = overfitting)
- Convergence speed (should improve by epoch 20)
- Final validation accuracy (target: 70-80%)

## Current Configuration Assessment

✅ **OPTIMAL** - Your current SGD configuration is the right choice for:
- Small dataset (1,449 samples)
- ResNet architecture
- From-scratch training
- Dual-stream network

**No change recommended.** Proceed with SGD + cosine annealing.
