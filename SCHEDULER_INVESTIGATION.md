# Scheduler Investigation: Stream-Specific Learning Rates

## Question
Does the PyTorch scheduler adjust stream-specific learning rates during training?

## Answer: ✅ YES!

PyTorch schedulers **automatically update ALL parameter groups**, including stream-specific learning rates.

---

## Test Results

### Test 1: PyTorch Behavior
```python
# Optimizer with 2 parameter groups (different initial LRs)
param_groups = [
    {'params': [param1], 'lr': 1e-3},   # Stream1
    {'params': [param2], 'lr': 5e-4},   # Stream2
]
optimizer = torch.optim.AdamW(param_groups)
scheduler = CosineAnnealingLR(optimizer, T_max=100)
```

**Result after 10 scheduler steps:**
```
Group 0: 0.001000 → 0.000976 (-2.45%)
Group 1: 0.000500 → 0.000488 (-2.45%)
LR Ratio: 2.0000 → 2.0000 (preserved!)
```

✅ **Both groups updated**
✅ **Relative ratio preserved**

---

## How It Works

### Scheduler Creation (in fit() method):
```python
# From mc_resnet.py:365
self.scheduler = setup_scheduler(
    self.optimizer,            # Optimizer with multiple param groups
    self.scheduler_type,        # 'cosine', 'step', etc.
    epochs,                     # Number of epochs
    len(train_loader),          # Batches per epoch
    **scheduler_kwargs
)
```

### Scheduler Stepping (in _train_epoch() method):
```python
# From mc_resnet.py:656
if self.scheduler is not None:
    if isinstance(self.scheduler, OneCycleLR):
        self.scheduler.step()  # ← Updates ALL param groups!
    elif not isinstance(self.scheduler, ReduceLROnPlateau):
        self.scheduler.step()  # ← Updates ALL param groups!
```

### CosineAnnealingLR Formula:
```
For each parameter group:
    new_lr = η_min + (η_max - η_min) * (1 + cos(π * T_cur / T_max)) / 2

Where:
    η_max = group['lr']  # Initial LR (different for each group!)
    η_min = 0            # Minimum LR
    T_cur = current step
    T_max = total steps
```

**Key Point:** Each group has its own η_max (initial LR), so they decay proportionally!

---

## Example: Stream-Specific Training

### Setup:
```python
model.compile(
    optimizer='adamw',
    learning_rate=1e-4,       # Base LR (fusion + classifier)
    stream1_lr=2e-4,          # RGB: 2x base
    stream2_lr=5e-5,          # Depth: 0.5x base
    scheduler='cosine'
)
```

### Parameter Groups Created:
```
Group 0 (RGB stream):    LR = 2e-4, WD = 1e-2
Group 1 (Depth stream):  LR = 5e-5, WD = 2e-2
Group 2 (Fusion + fc):   LR = 1e-4, WD = 1e-3
```

### During Training (cosine schedule):
```
Epoch 0:
  RGB:    2e-4
  Depth:  5e-5
  Shared: 1e-4
  Ratio (RGB/Depth): 4.0

Epoch 50 (midpoint):
  RGB:    1e-4  (50% of initial)
  Depth:  2.5e-5 (50% of initial)
  Shared: 5e-5  (50% of initial)
  Ratio (RGB/Depth): 4.0  ← Still 4x!

Epoch 100 (end):
  RGB:    ~0    (near minimum)
  Depth:  ~0    (near minimum)
  Shared: ~0    (near minimum)
  Ratio (RGB/Depth): 4.0  ← Still 4x!
```

✅ **All groups decay together**
✅ **Relative ratios preserved throughout training**

---

## Verification in Code

### Where Scheduler is Created:
**File:** `src/models/multi_channel/mc_resnet.py:365`
```python
self.scheduler = setup_scheduler(self.optimizer, self.scheduler_type, epochs, len(train_loader), **scheduler_kwargs)
```

### Where Scheduler Steps:
**File:** `src/models/multi_channel/mc_resnet.py:656` (inside _train_epoch)
```python
if self.scheduler is not None:
    if isinstance(self.scheduler, OneCycleLR):
        self.scheduler.step()
    elif not isinstance(self.scheduler, ReduceLROnPlateau):
        self.scheduler.step()
```

### Scheduler Setup Function:
**File:** `src/models/common/model_helpers.py:23-44`
```python
def setup_scheduler(optimizer, scheduler_type: str, epochs: int, train_loader_len: int, **scheduler_kwargs):
    if scheduler_type == 'cosine':
        t_max = scheduler_kwargs.get('t_max', epochs * train_loader_len)
        return CosineAnnealingLR(optimizer, T_max=t_max)  # ← Takes full optimizer (all groups!)
```

---

## Conclusion

### ✅ Your Code is Correct!

1. **Scheduler DOES adjust stream-specific LRs** - PyTorch handles this automatically
2. **Relative ratios are preserved** - If RGB starts at 2x Depth, it stays 2x throughout
3. **All parameter groups decay together** - According to the same schedule (cosine, step, etc.)

### 📊 What This Means:

If you set:
- RGB LR = 2e-4 (high, for slow-learning RGB)
- Depth LR = 5e-5 (low, for fast-learning Depth)

The scheduler will:
- Decay both LRs following the same curve (e.g., cosine)
- Maintain the 4x ratio throughout training
- **Example at epoch 50:** RGB = 1e-4, Depth = 2.5e-5 (still 4x)

### 🎯 No Action Needed!

The current implementation is working as intended. Stream-specific learning rates are being properly scheduled.

---

## Additional Notes

### Why Scheduler is Created in fit(), Not compile():
From `src/models/abstracts/abstract_model.py:378`:
```python
# Scheduler will be configured in fit() with actual training parameters
self.scheduler = None
```

**Reason:** Scheduler needs `T_max = epochs * batches_per_epoch`, which is only known when fit() is called with actual data.

### Monitoring LR During Training:
The fit() method tracks LRs in history:
```python
# From mc_resnet.py:413
current_lr = self.optimizer.param_groups[0]['lr']

# From mc_resnet.py:653
history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
```

**Note:** Currently only tracks Group 0 LR. To track all groups, modify to:
```python
history['learning_rates'].append([group['lr'] for group in self.optimizer.param_groups])
```
