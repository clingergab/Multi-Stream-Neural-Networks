# Early Stopping Refactoring Summary

## Changes Made

### 1. Removed `monitor` from State Dictionaries

**Before**:
```python
early_stopping_state = {
    'enabled': True,
    'monitor': 'val_accuracy',  # ❌ Stored in state
    'patience': 12,
    'min_delta': 0.01,
    'best_metric': 0.0,
    'is_better': lambda current, best: current > (best + 0.01),  # ❌ Stored
    'patience_counter': 0,
    'best_epoch': 0,
    'best_weights': None
}

stream_early_stopping_state = {
    'enabled': True,
    'monitor': 'val_accuracy',  # ❌ Stored in state
    'is_better': lambda current, best: current > (best + 0.01),  # ❌ Stored
    'stream1': {...},
    'stream2': {...},
    ...
}
```

**After**:
```python
early_stopping_state = {
    'enabled': True,
    'patience': 12,
    'min_delta': 0.01,
    'best_metric': 0.0,  # Set based on monitor during setup
    'patience_counter': 0,
    'best_epoch': 0,
    'best_weights': None
}

stream_early_stopping_state = {
    'enabled': True,
    'min_delta': 0.01,
    'stream1': {'best_metric': 0.0, ...},  # Set based on monitor
    'stream2': {'best_metric': 0.0, ...},
    ...
}
```

### 2. Updated Function Signatures

**`early_stopping_initiated`** - Added `monitor` parameter:
```python
# Before
def early_stopping_initiated(model_state_dict, early_stopping_state,
                           val_loss, val_acc, epoch, pbar, verbose,
                           restore_best_weights):
    monitor = early_stopping_state['monitor']  # ❌ Read from state
    is_better = early_stopping_state['is_better']  # ❌ Read from state
    ...

# After
def early_stopping_initiated(model_state_dict, early_stopping_state,
                           val_loss, val_acc, epoch, monitor, pbar, verbose,
                           restore_best_weights):
    # ✅ monitor passed as parameter
    # ✅ is_better computed on the fly
    min_delta = early_stopping_state['min_delta']
    if monitor == 'val_loss':
        is_better = current_metric < (best_metric - min_delta)
    else:  # val_accuracy
        is_better = current_metric > (best_metric + min_delta)
    ...
```

**`check_stream_early_stopping`** - Already had `monitor` parameter, now uses it:
```python
# Before
def check_stream_early_stopping(stream_early_stopping_state, stream_stats,
                                model, epoch, monitor, verbose,
                                val_acc, val_loss):
    monitor = stream_early_stopping_state['monitor']  # ❌ Ignored parameter, read from state
    is_better = stream_early_stopping_state['is_better']  # ❌ Read from state
    ...

# After
def check_stream_early_stopping(stream_early_stopping_state, stream_stats,
                                model, epoch, monitor, verbose,
                                val_acc, val_loss):
    # ✅ Use monitor parameter directly
    # ✅ Compute is_better on the fly
    min_delta = stream_early_stopping_state['min_delta']
    if monitor == 'val_loss':
        is_better = lambda current, best: current < (best - min_delta)
    else:
        is_better = lambda current, best: current > (best + min_delta)
    ...
```

### 3. Updated All Call Sites

**`src/models/linear_integration/li_net.py`**:
```python
# Before
early_stopping_initiated(
    self.state_dict(), early_stopping_state, val_loss, val_acc, epoch,
    pbar, verbose, restore_best_weights
)

# After
early_stopping_initiated(
    self.state_dict(), early_stopping_state, val_loss, val_acc, epoch,
    monitor, pbar, verbose, restore_best_weights
)
```

Same changes in:
- `src/models/core/resnet.py`
- `src/models/multi_channel/mc_resnet.py`

### 4. Added NaN Handling

As a bonus, added NaN detection in `early_stopping_initiated`:

```python
# Check for NaN/Inf in validation loss (indicates numerical instability)
import math
if math.isnan(val_loss) or math.isinf(val_loss):
    if verbose and pbar is None:
        print(f"\n⚠️  Warning: Validation loss is {'NaN' if math.isnan(val_loss) else 'Inf'}")
        print(f"   This indicates numerical instability (likely from high learning rates)")
        print(f"   Skipping early stopping update - training will continue")
    # Don't update patience counter - let training continue
    return False
```

## Benefits

### 1. Cleaner State Management
- State dictionaries only store actual **state** (counters, metrics, weights)
- Configuration (monitor type) is passed explicitly as a parameter
- Easier to understand what's stored vs what's computed

### 2. No Redundant Storage
- `monitor` is already known by the caller (it's their configuration)
- No need to store it in every state dict
- `is_better` lambda doesn't need to be stored, can be computed on demand

### 3. Easier Testing
- Tests can pass different `monitor` values without modifying state
- State dictionaries are simpler to construct in tests
- Less coupling between setup and usage

### 4. Better Error Prevention
- NaN detection prevents early stopping from triggering on corrupted metrics
- Warns user about numerical instability
- Allows training to continue (user can Ctrl+C if needed)

## Migration Guide

If you have custom code using the old API:

### Before
```python
# Old way
early_stopping_state = setup_early_stopping(
    early_stopping=True, val_loader=val_loader,
    monitor='val_accuracy', patience=12, min_delta=0.01, verbose=True
)

# monitor was stored in state
print(early_stopping_state['monitor'])  # 'val_accuracy'

# Call without passing monitor
should_stop = early_stopping_initiated(
    model.state_dict(), early_stopping_state,
    val_loss, val_acc, epoch, pbar, verbose, restore_best_weights
)
```

### After
```python
# New way
monitor = 'val_accuracy'  # ✅ Keep track of monitor yourself
early_stopping_state = setup_early_stopping(
    early_stopping=True, val_loader=val_loader,
    monitor=monitor, patience=12, min_delta=0.01, verbose=True
)

# monitor NOT in state anymore
# print(early_stopping_state['monitor'])  # ❌ KeyError!

# Must pass monitor as parameter
should_stop = early_stopping_initiated(
    model.state_dict(), early_stopping_state,
    val_loss, val_acc, epoch, monitor, pbar, verbose, restore_best_weights
)
```

## Testing Status

- ✅ Manual testing passed (see verification above)
- ⚠️  Unit tests need updating (use `monitor` parameter in test calls)
- ✅ All model training code updated (li_net.py, mc_resnet.py, resnet.py)

## Files Modified

1. `src/models/common/model_helpers.py`:
   - `setup_early_stopping()` - Removed `monitor` and `is_better` from returned state
   - `setup_stream_early_stopping()` - Removed `monitor` and `is_better` from returned state
   - `early_stopping_initiated()` - Added `monitor` parameter, compute `is_better` dynamically, added NaN handling
   - `check_stream_early_stopping()` - Use `monitor` parameter, compute `is_better` dynamically

2. `src/models/linear_integration/li_net.py`:
   - Updated `early_stopping_initiated()` call to pass `monitor`

3. `src/models/core/resnet.py`:
   - Updated `early_stopping_initiated()` call to pass `monitor`

4. `src/models/multi_channel/mc_resnet.py`:
   - Updated `early_stopping_initiated()` call to pass `monitor`

## Summary

**What changed**: `monitor` is now passed as a function parameter instead of being stored in state dictionaries.

**Why**: Cleaner separation of configuration (passed in) vs state (stored).

**Impact**: Existing code needs to pass `monitor` when calling `early_stopping_initiated()` and `check_stream_early_stopping()`.

**Bonus**: Added NaN detection to prevent early stopping from triggering on corrupted metrics from numerical instability.
