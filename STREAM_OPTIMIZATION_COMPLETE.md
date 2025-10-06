# Stream-Specific Optimization Complete ✅

## Summary

Successfully implemented stream-specific optimization allowing different learning rates and weight decay for each stream (RGB and depth pathways). This addresses the depth pathway dominance issue by enabling targeted optimization strategies per stream.

## What Was Done

### 1. Extended `compile()` Method

Modified `src/models/abstracts/abstract_model.py`:

**Added 4 new parameters:**
- `stream1_lr`: Learning rate for stream1 pathway
- `stream2_lr`: Learning rate for stream2 pathway
- `stream1_weight_decay`: Weight decay for stream1 pathway
- `stream2_weight_decay`: Weight decay for stream2 pathway

### 2. Implemented Parameter Group Separation

**Logic:**
1. Detect if any stream-specific parameters are provided
2. If yes, separate all model parameters into 3 groups:
   - **Stream1 params**: Any parameter with 'stream1' in its name
   - **Stream2 params**: Any parameter with 'stream2' in its name
   - **Shared params**: All other parameters (fusion, FC layer, etc.)
3. Create optimizer with multiple parameter groups, each with its own LR and weight decay

**Parameter Distribution:**
- Stream1: ~11.2M parameters (conv1.stream1_weight, bn1.stream1_weight, etc.)
- Stream2: ~11.2M parameters (conv1.stream2_weight, bn1.stream2_weight, etc.)
- Shared: ~28K parameters (fc.weight, fc.bias, fusion params)

### 3. Backward Compatibility

If no stream-specific parameters are provided, the optimizer works exactly as before with a single parameter group.

### 4. Testing

Created `test_stream_optimization.py`:
- ✅ Parameter separation verified (61 stream1, 61 stream2, 2 shared tensors)
- ✅ Standard optimization works (1 param group)
- ✅ Stream-specific LR works (3 param groups with correct LRs)
- ✅ Stream-specific weight decay works (correct WD per group)
- ✅ Combined optimization works (both LR and WD per stream)
- ✅ Training step works (weights update correctly)
- ✅ Optimizer state persists correctly

## Usage Examples

### Standard Optimization (Same LR/WD for All)
```python
model.compile(
    optimizer='adamw',
    learning_rate=1e-3,
    weight_decay=2e-2
)
# Result: 1 parameter group, all params use same LR/WD
```

### Stream-Specific Learning Rates
```python
model.compile(
    optimizer='adamw',
    learning_rate=1e-3,      # Base LR (for shared params)
    weight_decay=2e-2,       # Base WD (for shared params)
    stream1_lr=2e-4,         # Lower LR for stream1 (RGB)
    stream2_lr=5e-4          # Higher LR for stream2 (depth)
)
# Result: 3 parameter groups with different LRs
```

### Stream-Specific Weight Decay
```python
model.compile(
    optimizer='adamw',
    learning_rate=1e-3,
    weight_decay=2e-2,
    stream1_weight_decay=1e-3,  # Lighter regularization for stream1
    stream2_weight_decay=4e-2    # Stronger regularization for stream2
)
# Result: 3 parameter groups with different weight decays
```

### Combined Stream-Specific Optimization (Recommended)
```python
model.compile(
    optimizer='adamw',
    learning_rate=1e-3,
    weight_decay=2e-2,
    stream1_lr=3e-4,            # Boost RGB learning
    stream2_lr=1e-4,            # Slow down depth learning
    stream1_weight_decay=1e-3,  # Light regularization for RGB
    stream2_weight_decay=5e-2   # Heavy regularization for depth
)
# Result: Balanced learning between streams
```

## Strategy for NYU Depth V2

Based on pathway analysis showing depth dominance (95% vs 61%):

### Problem
- Depth pathway learns too quickly and dominates predictions
- RGB pathway contributes weakly (only 61% relative contribution)
- Model accuracy stuck at ~22%

### Solution Strategy
```python
model.compile(
    optimizer='adamw',
    learning_rate=1e-4,          # Base LR
    weight_decay=2e-2,           # Base WD

    # Boost RGB learning
    stream1_lr=5e-4,             # 5x higher LR for RGB
    stream1_weight_decay=1e-3,   # Lighter regularization

    # Slow down depth
    stream2_lr=5e-5,             # 2x lower LR for depth
    stream2_weight_decay=5e-2    # Heavier regularization
)
```

**Rationale:**
1. **Higher RGB LR**: Forces RGB pathway to learn faster and contribute more
2. **Lower RGB WD**: Allows RGB to explore more diverse features
3. **Lower depth LR**: Prevents depth from dominating early in training
4. **Higher depth WD**: Regularizes depth pathway to prevent overfitting

## How It Works

### Parameter Group Structure
```python
optimizer.param_groups = [
    {
        'params': [stream1_params],      # 11.2M params
        'lr': stream1_lr,
        'weight_decay': stream1_weight_decay
    },
    {
        'params': [stream2_params],      # 11.2M params
        'lr': stream2_lr,
        'weight_decay': stream2_weight_decay
    },
    {
        'params': [shared_params],       # 28K params
        'lr': learning_rate,
        'weight_decay': weight_decay
    }
]
```

### What Gets Optimized Separately

**Stream1 Parameters (RGB pathway):**
- `conv1.stream1_weight`, `conv1.stream1_bias`
- `bn1.stream1_weight`, `bn1.stream1_bias`
- All layer1-4 stream1 conv/bn parameters
- `fusion.stream1_weight` (if WeightedFusion)

**Stream2 Parameters (Depth pathway):**
- `conv1.stream2_weight`, `conv1.stream2_bias`
- `bn2.stream2_weight`, `bn2.stream2_bias`
- All layer1-4 stream2 conv/bn parameters
- `fusion.stream2_weight` (if WeightedFusion)

**Shared Parameters:**
- `fc.weight`, `fc.bias` (classifier)
- `fusion.gate_network.*` (if GatedFusion)
- Dropout (if used)

## Comparison with Alternatives

### vs. Separate Optimizers
❌ **Don't need:** Separate optimizers for each stream
✅ **We have:** Single optimizer with multiple parameter groups
- Simpler to manage
- Works with all PyTorch schedulers
- Single `.step()` call

### vs. Manual Parameter Groups
❌ **Don't need:** Manual parameter filtering
✅ **We have:** Automatic separation by naming convention
- Just add `stream1_lr` parameter
- Automatic detection and grouping

### vs. Gradient Manipulation
❌ **Don't need:** Manual gradient scaling/clipping per stream
✅ **We have:** Native PyTorch optimization
- Cleaner implementation
- Better numerical stability

## Test Results

```bash
$ python3 test_stream_optimization.py

Testing Stream-Specific Optimization
======================================================================

1. Parameter Separation Test
----------------------------------------------------------------------
Stream1 parameters: 61 tensors, 11,176,513 values
Stream2 parameters: 61 tensors, 11,170,241 values
Shared parameters: 2 tensors, 27,675 values

5. Combined Stream-Specific Optimization
----------------------------------------------------------------------
✓ Parameter groups: 3
  Group 0: lr=1.0e-04, wd=1.0e-03, params=11,176,513
  Group 1: lr=5.0e-04, wd=4.0e-02, params=11,170,241
  Group 2: lr=1.0e-03, wd=2.0e-02, params=27,675

✅ All stream-specific optimization tests passed!
```

## Integration with Fusion Strategies

Stream-specific optimization works with all fusion types:

```python
# Concat fusion + stream-specific optimization
model = mc_resnet18(
    num_classes=27,
    fusion_type='concat'
)
model.compile(
    optimizer='adamw',
    stream1_lr=5e-4,
    stream2_lr=1e-4
)

# Weighted fusion + stream-specific optimization
model = mc_resnet18(
    num_classes=27,
    fusion_type='weighted'
)
model.compile(
    optimizer='adamw',
    stream1_lr=5e-4,
    stream2_lr=1e-4
)

# Gated fusion + stream-specific optimization
model = mc_resnet18(
    num_classes=27,
    fusion_type='gated'
)
model.compile(
    optimizer='adamw',
    stream1_lr=5e-4,
    stream2_lr=1e-4
)
```

## Files Modified

- ✅ `src/models/abstracts/abstract_model.py` - Added stream-specific params to `compile()`
- ✅ `test_stream_optimization.py` - Comprehensive test suite (created)
- ✅ `STREAM_OPTIMIZATION_COMPLETE.md` - Documentation (this file)

## Next Steps

### 1. Experiment on NYU Depth V2

Try different configurations:

**Configuration A: Boost RGB, Regularize Depth**
```python
stream1_lr=5e-4, stream2_lr=5e-5
stream1_weight_decay=1e-3, stream2_weight_decay=5e-2
```

**Configuration B: Moderate Boost**
```python
stream1_lr=3e-4, stream2_lr=1e-4
stream1_weight_decay=1e-2, stream2_weight_decay=3e-2
```

**Configuration C: Extreme Boost**
```python
stream1_lr=1e-3, stream2_lr=1e-5
stream1_weight_decay=1e-4, stream2_weight_decay=8e-2
```

### 2. Combine with Fusion Strategies

Test combinations:
- WeightedFusion + Stream-specific optimization
- GatedFusion + Stream-specific optimization
- Compare pathway balance improvements

### 3. Monitor Metrics

Track during training:
- Validation accuracy
- Pathway contribution (via `analyze_pathways()`)
- Stream1 vs Stream2 gradient norms
- Learning rate values per group

## Design Principles Followed

1. ✅ **Module-Agnostic**: Uses `stream1`/`stream2` naming (not RGB/depth)
2. ✅ **Backward Compatible**: Works without stream-specific params
3. ✅ **Clean Interface**: Simple parameters to `compile()`
4. ✅ **Automatic Detection**: No manual parameter filtering needed
5. ✅ **Works with All Optimizers**: AdamW, Adam, SGD, RMSprop
6. ✅ **Scheduler Compatible**: Works with all PyTorch schedulers

## Complete Example

```python
from src.models.multi_channel import mc_resnet18
from src.data_utils.nyu_depth_dataset import create_nyu_dataloaders

# Create model with gated fusion
model = mc_resnet18(
    num_classes=27,
    fusion_type='gated',
    dropout_p=0.3
)

# Compile with stream-specific optimization
model.compile(
    optimizer='adamw',
    learning_rate=1e-4,
    weight_decay=2e-2,
    # Boost RGB, regularize depth
    stream1_lr=5e-4,              # 5x boost for RGB
    stream2_lr=5e-5,              # 2x slower for depth
    stream1_weight_decay=1e-3,    # Light regularization
    stream2_weight_decay=5e-2,    # Heavy regularization
    scheduler='cosine',
    label_smoothing=0.1
)

# Load data
train_loader, val_loader = create_nyu_dataloaders(
    nyu_h5_path='nyu_depth_v2_labeled.mat',
    batch_size=64,
    num_workers=4
)

# Train
history = model.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100,
    early_stopping=True,
    patience=15
)

# Analyze pathway balance
analysis = model.analyze_pathways(val_loader)
print(f"RGB contribution: {analysis['accuracy']['color_contribution']:.2%}")
print(f"Depth contribution: {analysis['accuracy']['brightness_contribution']:.2%}")
```
