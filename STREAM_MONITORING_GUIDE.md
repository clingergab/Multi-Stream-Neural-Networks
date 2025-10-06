

# Stream Monitoring Guide

## Overview

The `StreamMonitor` provides real-time insights into individual stream behavior during training, allowing you to:

1. **Detect stream-specific overfitting** - Identify which stream is memorizing training data
2. **Monitor gradient flow** - Track if gradients are balanced between streams
3. **Analyze weight evolution** - See if one stream dominates learning
4. **Get actionable recommendations** - Automatic suggestions for hyperparameter adjustments

---

## Quick Start

```python
from src.models.multi_channel import mc_resnet18
from src.models.utils import StreamMonitor

# Create and compile model
model = mc_resnet18(num_classes=27, fusion_type='weighted')
model.compile(
    optimizer='adamw',
    stream1_lr=5e-4,
    stream2_lr=1e-4,
    stream1_weight_decay=1e-3,
    stream2_weight_decay=3e-2
)

# Create monitor
monitor = StreamMonitor(model)

# During training loop
for epoch in range(epochs):
    # ... training code ...

    # Monitor gradients (call BEFORE optimizer.step())
    grad_stats = monitor.compute_stream_gradients()

    # Monitor overfitting
    overfit_stats = monitor.compute_stream_overfitting_indicators(
        train_loss, val_loss, train_acc, val_acc,
        train_loader, val_loader
    )

    # Log metrics
    monitor.log_metrics(epoch, {**grad_stats, **overfit_stats})

    # Get recommendations
    recommendations = monitor.get_recommendations()
    for rec in recommendations:
        print(rec)
```

---

## Key Metrics

### 1. Gradient Statistics

**What it measures:** Gradient flow through each stream

```python
grad_stats = monitor.compute_stream_gradients()
```

**Returns:**
```python
{
    'stream1_grad_norm': 14.51,      # L2 norm of all stream1 gradients
    'stream2_grad_norm': 14.58,      # L2 norm of all stream2 gradients
    'stream1_to_stream2_ratio': 1.00, # Ratio of gradient norms
    'stream1_grad_max': 0.23,        # Max gradient in stream1
    'stream2_grad_max': 0.19,        # Max gradient in stream2
}
```

**What to look for:**
- **Ratio close to 1.0** → Balanced gradient flow ✓
- **Ratio > 3.0** → Stream1 gradients dominating ⚠️
- **Ratio < 0.33** → Stream2 gradients dominating ⚠️
- **Very small norms** → Vanishing gradients ⚠️
- **Very large norms** → Exploding gradients ⚠️

**Actions:**
- If stream1 gradients too large: **reduce `stream1_lr`**
- If stream2 gradients too small: **increase `stream2_lr`**
- If stream1 gradients vanishing: **increase `stream1_lr` or reduce `stream1_weight_decay`**

### 2. Weight Statistics

**What it measures:** Weight magnitudes per stream

```python
weight_stats = monitor.compute_stream_weights()
```

**Returns:**
```python
{
    'stream1_weight_norm': 151.44,   # L2 norm of all stream1 weights
    'stream2_weight_norm': 114.16,   # L2 norm of all stream2 weights
    'weight_norm_ratio': 1.33,       # Ratio of weight norms
    'stream1_weight_mean': 0.02,     # Mean weight value
    'stream2_weight_mean': -0.01,    # Mean weight value
}
```

**What to look for:**
- **Ratio 1.0-1.5** → Balanced weight evolution ✓
- **Ratio > 2.0** → Stream1 weights growing too fast ⚠️
- **Ratio < 0.5** → Stream2 weights growing too fast ⚠️
- **Diverging over time** → Unstable training ⚠️

**Actions:**
- If stream1 weights too large: **increase `stream1_weight_decay`**
- If stream2 weights stagnant: **increase `stream2_lr`**

### 3. Overfitting Indicators

**What it measures:** Stream-specific overfitting

```python
overfit_stats = monitor.compute_stream_overfitting_indicators(
    train_loss, val_loss, train_acc, val_acc,
    train_loader, val_loader
)
```

**Returns:**
```python
{
    # Stream1 metrics
    'stream1_train_acc': 0.69,
    'stream1_val_acc': 0.09,
    'stream1_acc_gap': 0.60,         # train - val
    'stream1_loss_gap': 1.17,        # val - train
    'stream1_overfitting_score': 1.77, # Combined score

    # Stream2 metrics
    'stream2_train_acc': 1.00,
    'stream2_val_acc': 0.06,
    'stream2_acc_gap': 0.94,
    'stream2_loss_gap': 3.21,
    'stream2_overfitting_score': 4.15, # Combined score

    # Comparison
    'stream1_vs_stream2_overfit_ratio': 0.43,
}
```

**What to look for:**
- **Overfitting score < 0.5** → Healthy generalization ✓
- **Score 0.5-1.0** → Mild overfitting ⚠️
- **Score > 1.0** → Significant overfitting ⚠️
- **Stream2 score > Stream1 score × 2** → Stream2 overfitting more ⚠️

**Actions:**
- If stream1 overfitting: **increase `stream1_weight_decay`** or **reduce `stream1_lr`**
- If stream2 overfitting: **increase `stream2_weight_decay`** or **reduce `stream2_lr`**
- If both overfitting: **increase global `dropout_p`** or **add data augmentation**

---

## Usage Patterns

### Pattern 1: Monitor Every Epoch

```python
monitor = StreamMonitor(model)

for epoch in range(epochs):
    # Training
    train_loss, train_acc = train_epoch(model, train_loader)

    # Validation
    val_loss, val_acc = validate(model, val_loader)

    # Monitor streams
    grad_stats = monitor.compute_stream_gradients()
    weight_stats = monitor.compute_stream_weights()
    overfit_stats = monitor.compute_stream_overfitting_indicators(
        train_loss, val_loss, train_acc, val_acc,
        train_loader, val_loader
    )

    # Log all metrics
    all_metrics = {**grad_stats, **weight_stats, **overfit_stats}
    monitor.log_metrics(epoch, all_metrics)

    # Print summary
    if epoch % 10 == 0:
        print(monitor.get_summary())
        recommendations = monitor.get_recommendations()
        for rec in recommendations:
            print(rec)
```

### Pattern 2: Monitor Specific Batches

```python
monitor = StreamMonitor(model)

for epoch in range(epochs):
    for batch_idx, (s1, s2, targets) in enumerate(train_loader):
        # Forward
        outputs = model(s1, s2)
        loss = criterion(outputs, targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Monitor gradients BEFORE optimizer step
        if batch_idx % 100 == 0:  # Every 100 batches
            grad_stats = monitor.compute_stream_gradients()
            print(f"Batch {batch_idx}: S1 grad={grad_stats['stream1_grad_norm']:.2f}, "
                  f"S2 grad={grad_stats['stream2_grad_norm']:.2f}")

        optimizer.step()
```

### Pattern 3: Adaptive Learning Rate Adjustment

```python
monitor = StreamMonitor(model)

for epoch in range(epochs):
    # ... training ...

    # Check gradients
    grad_stats = monitor.compute_stream_gradients()
    ratio = grad_stats['stream1_to_stream2_ratio']

    # Adaptive adjustment
    if ratio > 3.0:
        # Stream1 gradients too large
        print("⚠️  Reducing stream1_lr")
        for pg in optimizer.param_groups:
            if 'stream1' in str(pg):  # Identify stream1 param group
                pg['lr'] *= 0.5

    elif ratio < 0.33:
        # Stream2 gradients too large
        print("⚠️  Reducing stream2_lr")
        for pg in optimizer.param_groups:
            if 'stream2' in str(pg):  # Identify stream2 param group
                pg['lr'] *= 0.5
```

---

## Interpretation Guide

### Scenario 1: Stream1 (RGB) Not Learning

**Symptoms:**
```
Stream1 grad norm: 0.15
Stream2 grad norm: 3.79
Ratio (S1/S2): 0.04  ← Very low!
Stream1 train acc: 0.10
Stream2 train acc: 0.85
```

**Diagnosis:** Stream1 gradients vanishing, not contributing to learning

**Solutions:**
1. **Increase `stream1_lr`** (e.g., from 1e-4 to 5e-4)
2. **Reduce `stream1_weight_decay`** (e.g., from 2e-2 to 1e-3)
3. **Check fusion weights** (if using WeightedFusion, stream1_weight might be near 0)
4. **Try different fusion strategy** (e.g., switch to GatedFusion)

### Scenario 2: Stream2 (Depth) Overfitting

**Symptoms:**
```
Stream2 train acc: 0.95
Stream2 val acc: 0.12
Stream2 overfitting score: 4.15  ← Very high!
Stream1 overfitting score: 0.85  ← Normal
```

**Diagnosis:** Stream2 memorizing training data

**Solutions:**
1. **Increase `stream2_weight_decay`** (e.g., from 2e-2 to 5e-2)
2. **Reduce `stream2_lr`** (e.g., from 1e-4 to 5e-5)
3. **Add stream2-specific dropout** (future feature)
4. **Augment depth channel more aggressively**

### Scenario 3: Balanced but Not Learning

**Symptoms:**
```
Stream1 grad norm: 2.15
Stream2 grad norm: 2.08
Ratio (S1/S2): 1.03  ← Balanced ✓
Both train acc: 0.11  ← Not learning!
```

**Diagnosis:** Gradients balanced but too small globally

**Solutions:**
1. **Increase both learning rates** (increase `learning_rate` base)
2. **Check loss function** (maybe using wrong loss)
3. **Check data preprocessing** (maybe inputs not normalized)
4. **Reduce global regularization** (reduce `weight_decay`)

### Scenario 4: One Stream Dominating

**Symptoms:**
```
Stream1 weight norm: 245.3
Stream2 weight norm: 87.1
Weight ratio: 2.82  ← Very high!
Stream1 contribution: 95%
Stream2 contribution: 15%
```

**Diagnosis:** Stream1 weights growing much faster, dominating predictions

**Solutions:**
1. **Increase `stream1_weight_decay`** to slow down stream1
2. **Increase `stream2_lr`** to boost stream2 learning
3. **Use WeightedFusion** to explicitly learn stream importance
4. **Check if fusion.stream2_weight is very small**

---

## Automatic Recommendations

The monitor provides automatic recommendations based on metrics:

```python
recommendations = monitor.get_recommendations()
```

**Example output:**
```
⚠️  Stream1 gradients 4.2x larger than Stream2 - consider reducing stream1_lr
⚠️  Stream2 overfitting (score: 3.54) - increase stream2_weight_decay or add dropout
⚠️  Stream2 overfitting significantly more - boost stream2 regularization
⚠️  Stream1 learning stalled - consider increasing stream1_lr
✓ Gradient ratio balanced (1.15)
✓ All metrics look healthy
```

---

## Integration with Training

### Full Training Loop with Monitoring

```python
from src.models.multi_channel import mc_resnet18
from src.models.utils import StreamMonitor

# Setup
model = mc_resnet18(num_classes=27, fusion_type='weighted')
model.compile(
    optimizer='adamw',
    learning_rate=1e-4,
    stream1_lr=5e-4,
    stream2_lr=1e-4,
    stream1_weight_decay=1e-3,
    stream2_weight_decay=3e-2
)

monitor = StreamMonitor(model)

# Training
for epoch in range(100):
    # Train
    model.train()
    for batch_idx, (s1, s2, targets) in enumerate(train_loader):
        outputs = model(s1, s2)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()

        # Monitor first batch gradients
        if batch_idx == 0:
            grad_stats = monitor.compute_stream_gradients()
            print(f"Epoch {epoch}: S1={grad_stats['stream1_grad_norm']:.2f}, "
                  f"S2={grad_stats['stream2_grad_norm']:.2f}")

        optimizer.step()

    # Validate
    model.eval()
    # ... validation code ...

    # Stream analysis (every 10 epochs)
    if epoch % 10 == 0:
        overfit_stats = monitor.compute_stream_overfitting_indicators(
            train_loss, val_loss, train_acc, val_acc,
            train_loader, val_loader
        )

        print(f"\nEpoch {epoch} Stream Analysis:")
        print(f"Stream1 overfit: {overfit_stats['stream1_overfitting_score']:.2f}")
        print(f"Stream2 overfit: {overfit_stats['stream2_overfitting_score']:.2f}")

        # Get recommendations
        recommendations = monitor.get_recommendations()
        if any('⚠️' in rec for rec in recommendations):
            print("\n⚠️  Recommendations:")
            for rec in recommendations:
                print(f"  {rec}")

    # Full pathway analysis (every 25 epochs)
    if epoch % 25 == 0:
        analysis = model.analyze_pathways(val_loader)
        print(f"\nPathway Contributions:")
        print(f"  Stream1: {analysis['accuracy']['color_contribution']:.1%}")
        print(f"  Stream2: {analysis['accuracy']['brightness_contribution']:.1%}")
```

---

## Visualization (Future Enhancement)

The monitoring data can be visualized:

```python
import matplotlib.pyplot as plt

# Plot gradient evolution
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(monitor.history['stream1_grad_norm'], label='Stream1')
plt.plot(monitor.history['stream2_grad_norm'], label='Stream2')
plt.xlabel('Epoch')
plt.ylabel('Gradient Norm')
plt.legend()
plt.title('Gradient Flow')

plt.subplot(1, 3, 2)
plt.plot(monitor.history['stream1_overfitting_score'], label='Stream1')
plt.plot(monitor.history['stream2_overfitting_score'], label='Stream2')
plt.xlabel('Epoch')
plt.ylabel('Overfitting Score')
plt.legend()
plt.title('Overfitting Indicators')

plt.subplot(1, 3, 3)
plt.plot(monitor.history['stream1_weight_norm'], label='Stream1')
plt.plot(monitor.history['stream2_weight_norm'], label='Stream2')
plt.xlabel('Epoch')
plt.ylabel('Weight Norm')
plt.legend()
plt.title('Weight Evolution')

plt.tight_layout()
plt.savefig('stream_monitoring.png')
```

---

## Best Practices

### 1. When to Monitor

- **Every epoch:** Overfitting indicators, weight statistics
- **Every N batches:** Gradient statistics (expensive to compute)
- **After major changes:** When adjusting hyperparameters

### 2. What to Monitor

**Early training (epochs 1-10):**
- Focus on gradient flow
- Ensure both streams learning

**Mid training (epochs 10-50):**
- Monitor overfitting scores
- Track weight evolution

**Late training (epochs 50+):**
- Check for divergence
- Validate pathway balance

### 3. Decision Making

**If metrics are stable:**
- Continue training
- Monitor less frequently

**If metrics show issues:**
- Adjust hyperparameters
- Monitor more frequently
- Consider early stopping

---

## Summary

The `StreamMonitor` provides comprehensive insights into multi-stream learning:

✅ **Gradient monitoring** - Detect vanishing/exploding gradients per stream
✅ **Overfitting detection** - Identify which stream memorizes training data
✅ **Weight analysis** - Track if one stream dominates
✅ **Automatic recommendations** - Get actionable suggestions
✅ **Historical tracking** - Monitor trends over time

**Use it to:**
- Make informed decisions about learning rates
- Adjust regularization per stream
- Detect and fix training issues early
- Achieve balanced multi-stream learning

Run `python3 example_stream_monitoring.py` to see it in action!
