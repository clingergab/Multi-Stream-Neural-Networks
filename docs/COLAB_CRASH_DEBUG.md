# Colab Training Crash - Debug Steps

## Symptoms
- Kernel crashes immediately when `model.fit()` is called
- Stack trace shows Python interpreter crash (not OOM or CUDA error)
- RAM usage is LOW (2.8 / 167 GB)
- Crash happens in _asyncio, suggesting event loop issue

## Likely Causes

### 1. tqdm Progress Bar Widget Issue (MOST LIKELY)
The smart tqdm detection may be crashing when trying to use notebook widgets:
```python
# In mc_resnet.py lines 32-46
try:
    from IPython import get_ipython
    if get_ipython() is not None:
        from tqdm.notebook import tqdm  # ← May crash on Colab
```

### 2. CUDA/Device Issue
First batch might trigger CUDA initialization crash

### 3. Data Loading in Training Loop
Workers might crash when first batch is requested during training

## Debug Steps

### Step 1: Add before training cell

```python
# Force disable tqdm notebook widgets
import os
os.environ['TQDM_DISABLE'] = '1'

# Test if model forward pass works
print("Testing model...")
model.eval()
with torch.no_grad():
    test_rgb, test_depth, test_labels = next(iter(train_loader))
    test_out = model(test_rgb.to('cuda'), test_depth.to('cuda'))
    print(f"✅ Model forward pass works: {test_out.shape}")

# Test optimizer step
model.train()
test_rgb, test_depth, test_labels = next(iter(train_loader))
test_rgb, test_depth, test_labels = test_rgb.to('cuda'), test_depth.to('cuda'), test_labels.to('cuda')
outputs = model(test_rgb, test_depth)
loss = model.criterion(outputs, test_labels)
loss.backward()
model.optimizer.step()
model.optimizer.zero_grad()
print(f"✅ Optimizer step works")
```

### Step 2: Try training with verbose=False

```python
# Disable progress bars
history = model.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=1,  # Just 1 epoch to test
    verbose=False,  # ← Disable tqdm
    ...
)
```

### Step 3: Check CUDA initialization

```python
# Test CUDA before training
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Test tensor on CUDA
test_tensor = torch.randn(10, 10).cuda()
print(f"✅ CUDA initialization works: {test_tensor.device}")
```

## Recommended Fix

### Option 1: Disable tqdm widgets (QUICKEST)

Add this BEFORE creating the model:

```python
import os
# Force tqdm to use text mode, not notebook widgets
os.environ['TQDM_DISABLE'] = '1'
```

### Option 2: Use verbose=False

```python
history = model.fit(
    ...,
    verbose=False  # No progress bars
)
```

### Option 3: Fix tqdm import in mc_resnet.py

Change the smart tqdm import to always use basic tqdm on Colab:

```python
# Force basic tqdm (not notebook) on Colab
import sys
if 'google.colab' in sys.modules:
    from tqdm import tqdm
else:
    # Smart detection for other environments
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm
    except:
        from tqdm import tqdm
```

## Testing

Add this cell BEFORE training to test each component:

```python
print("="*60)
print("PRE-TRAINING DIAGNOSTICS")
print("="*60)

# 1. Test CUDA
print("\n1. Testing CUDA...")
test_cuda = torch.randn(10, 10).cuda()
print(f"   ✅ CUDA works: {test_cuda.device}")

# 2. Test model forward
print("\n2. Testing model forward pass...")
model.eval()
with torch.no_grad():
    rgb, depth, labels = next(iter(train_loader))
    out = model(rgb.cuda(), depth.cuda())
    print(f"   ✅ Forward pass works: {out.shape}")

# 3. Test backward pass
print("\n3. Testing backward pass...")
model.train()
rgb, depth, labels = next(iter(train_loader))
outputs = model(rgb.cuda(), depth.cuda())
loss = model.criterion(outputs, labels.cuda())
loss.backward()
print(f"   ✅ Backward pass works: loss={loss.item():.4f}")

# 4. Test optimizer
print("\n4. Testing optimizer step...")
model.optimizer.step()
model.optimizer.zero_grad()
print(f"   ✅ Optimizer works")

# 5. Test tqdm
print("\n5. Testing tqdm...")
from tqdm import tqdm
for i in tqdm(range(10), desc="Test"):
    pass
print(f"   ✅ tqdm works")

print("\n" + "="*60)
print("ALL DIAGNOSTICS PASSED - Ready to train")
print("="*60)
```

If diagnostics pass but training still crashes, the issue is in the training loop logic itself.
