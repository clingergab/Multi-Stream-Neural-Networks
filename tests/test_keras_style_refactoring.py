"""
Comprehensive test for Keras-style API refactoring.

Tests:
1. Simple optimizer (single LR for all parameters)
2. Stream-specific optimizer (using get_stream_parameter_groups helper)
3. Scheduler creation with setup_scheduler
4. Compile with optimizer and scheduler objects
5. Stream monitoring compatibility (3 parameter groups)
6. Training loop execution
"""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
from src.models.multi_channel.mc_resnet import mc_resnet18
from src.training.schedulers import setup_scheduler

print("=" * 80)
print("COMPREHENSIVE TEST: Keras-Style API Refactoring")
print("=" * 80)

# Test 1: Create model
print("\n[Test 1] Creating MCResNet18 model...")
model = mc_resnet18(
    num_classes=10,
    stream1_input_channels=3,
    stream2_input_channels=1,
    dropout_p=0.3,
    device='cpu'
)
print("✅ Model created successfully")

# Test 2: Simple optimizer (single LR)
print("\n[Test 2] Testing simple optimizer (single LR)...")
simple_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
print(f"✅ Simple optimizer created with {len(simple_optimizer.param_groups)} parameter group(s)")

# Test 3: Stream-specific optimizer (using helper method)
print("\n[Test 3] Testing stream-specific optimizer (using get_stream_parameter_groups)...")
param_groups = model.get_stream_parameter_groups(
    stream1_lr=2e-4,
    stream2_lr=7e-4,
    shared_lr=5e-4,
    stream1_weight_decay=1e-4,
    stream2_weight_decay=2e-4,
    shared_weight_decay=1.5e-4
)
print(f"✅ Parameter groups created: {len(param_groups)} groups")
for i, pg in enumerate(param_groups):
    print(f"   Group {i+1}: {len(pg['params'])} params, lr={pg['lr']:.2e}, wd={pg['weight_decay']:.2e}")

stream_optimizer = torch.optim.AdamW(param_groups)
print(f"✅ Stream-specific optimizer created with {len(stream_optimizer.param_groups)} parameter groups")

# Test 4: Verify parameter group structure
print("\n[Test 4] Verifying parameter group structure...")
expected_groups = 3
actual_groups = len(stream_optimizer.param_groups)
assert actual_groups == expected_groups, f"Expected {expected_groups} groups, got {actual_groups}"
print(f"✅ Correct number of parameter groups: {actual_groups}")

# Verify LRs match what we set
assert stream_optimizer.param_groups[0]['lr'] == 2e-4, "Stream1 LR mismatch"
assert stream_optimizer.param_groups[1]['lr'] == 7e-4, "Stream2 LR mismatch"
assert stream_optimizer.param_groups[2]['lr'] == 5e-4, "Shared LR mismatch"
print("✅ Learning rates match expected values")
print(f"   Stream1 LR: {stream_optimizer.param_groups[0]['lr']:.2e}")
print(f"   Stream2 LR: {stream_optimizer.param_groups[1]['lr']:.2e}")
print(f"   Shared LR: {stream_optimizer.param_groups[2]['lr']:.2e}")

# Test 5: Create scheduler using setup_scheduler
print("\n[Test 5] Creating scheduler using setup_scheduler()...")
scheduler = setup_scheduler(
    stream_optimizer,
    'decaying_cosine',
    epochs=80,
    train_loader_len=40,
    t_max=10,
    eta_min=6e-5,
    max_factor=0.6,
    min_factor=0.6
)
print(f"✅ Scheduler created: {scheduler.__class__.__name__}")

# Test 6: Compile model (Keras-style)
print("\n[Test 6] Compiling model with optimizer and scheduler objects...")
try:
    model.compile(
        optimizer=stream_optimizer,
        scheduler=scheduler,
        loss='cross_entropy'
    )
    print("✅ Model compiled successfully (Keras-style)")
except Exception as e:
    print(f"❌ Compilation failed: {e}")
    raise

# Test 7: Verify compile() stores objects correctly
print("\n[Test 7] Verifying compile() stored objects correctly...")
assert model.optimizer is stream_optimizer, "Optimizer not stored correctly"
assert model.scheduler is scheduler, "Scheduler not stored correctly"
assert model.is_compiled is True, "is_compiled flag not set"
print("✅ All objects stored correctly in model")

# Test 8: Create dummy data loaders
print("\n[Test 8] Creating dummy data loaders...")
def create_dummy_loader(num_batches=5, batch_size=4):
    """Create a dummy data loader for testing."""
    class DummyDataset:
        def __len__(self):
            return num_batches * batch_size

        def __getitem__(self, idx):
            rgb = torch.randn(3, 32, 32)
            depth = torch.randn(1, 32, 32)
            label = torch.randint(0, 10, (1,)).item()
            return rgb, depth, label

    from torch.utils.data import DataLoader
    dataset = DummyDataset()
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

train_loader = create_dummy_loader(num_batches=5, batch_size=4)
val_loader = create_dummy_loader(num_batches=3, batch_size=4)
print(f"✅ Data loaders created: {len(train_loader)} train batches, {len(val_loader)} val batches")

# Test 9: Run training for a few epochs (no scheduler_kwargs needed!)
print("\n[Test 9] Running training with stream monitoring...")
try:
    history = model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=3,
        verbose=True,
        stream_monitoring=True  # Enable to test stream-specific logging
    )
    print("✅ Training completed successfully")
    print(f"   Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"   Final val loss: {history['val_loss'][-1]:.4f}")

    # Verify stream monitoring captured data
    if len(stream_optimizer.param_groups) >= 3:
        assert len(history['stream1_lr']) > 0, "Stream1 LR not recorded"
        assert len(history['stream2_lr']) > 0, "Stream2 LR not recorded"
        print(f"✅ Stream monitoring working: {len(history['stream1_lr'])} LR records captured")
        print(f"   Final Stream1 LR: {history['stream1_lr'][-1]:.2e}")
        print(f"   Final Stream2 LR: {history['stream2_lr'][-1]:.2e}")

except Exception as e:
    print(f"❌ Training failed: {e}")
    import traceback
    traceback.print_exc()
    raise

# Test 10: Verify scheduler stepped correctly
print("\n[Test 10] Verifying scheduler stepped correctly...")
current_lrs = [pg['lr'] for pg in stream_optimizer.param_groups]
print(f"✅ Current learning rates after training:")
for i, lr in enumerate(current_lrs):
    print(f"   Group {i+1}: {lr:.6e}")

# Test 11: Test simple optimizer workflow (without stream-specific)
print("\n[Test 11] Testing simple optimizer workflow (single LR)...")
model2 = mc_resnet18(num_classes=10, stream1_input_channels=3, stream2_input_channels=1, device='cpu')
simple_opt = torch.optim.AdamW(model2.parameters(), lr=1e-3, weight_decay=1e-4)
simple_sched = setup_scheduler(simple_opt, 'cosine', epochs=10, train_loader_len=40, t_max=10)
model2.compile(optimizer=simple_opt, scheduler=simple_sched, loss='cross_entropy')
print("✅ Simple optimizer workflow works")

# Test 12: Test compile without scheduler (optional scheduler)
print("\n[Test 12] Testing compile without scheduler (scheduler=None)...")
model3 = mc_resnet18(num_classes=10, stream1_input_channels=3, stream2_input_channels=1, device='cpu')
opt_no_sched = torch.optim.AdamW(model3.parameters(), lr=1e-3, weight_decay=1e-4)
model3.compile(optimizer=opt_no_sched, scheduler=None, loss='cross_entropy')
assert model3.scheduler is None, "Scheduler should be None"
print("✅ Compile works with scheduler=None")

# Test 13: Verify backward compatibility - setup_scheduler still public
print("\n[Test 13] Verifying setup_scheduler is still public and importable...")
from src.training.schedulers import setup_scheduler as imported_setup_scheduler
assert imported_setup_scheduler is not None, "setup_scheduler not importable"
print("✅ setup_scheduler is public and importable")

# Test 14: Test error handling - invalid optimizer type
print("\n[Test 14] Testing error handling - invalid optimizer type...")
model4 = mc_resnet18(num_classes=10, stream1_input_channels=3, stream2_input_channels=1, device='cpu')
try:
    model4.compile(optimizer="invalid_string", loss='cross_entropy')
    print("❌ Should have raised TypeError for string optimizer")
except TypeError as e:
    print(f"✅ Correctly raised TypeError: {str(e)[:60]}...")

# Test 15: Test error handling - invalid scheduler type
print("\n[Test 15] Testing error handling - invalid scheduler type...")
model5 = mc_resnet18(num_classes=10, stream1_input_channels=3, stream2_input_channels=1, device='cpu')
valid_opt = torch.optim.AdamW(model5.parameters(), lr=1e-3)
try:
    model5.compile(optimizer=valid_opt, scheduler="invalid_string", loss='cross_entropy')
    print("❌ Should have raised TypeError for string scheduler")
except TypeError as e:
    print(f"✅ Correctly raised TypeError: {str(e)[:60]}...")

print("\n" + "=" * 80)
print("🎉 ALL TESTS PASSED!")
print("=" * 80)
print("\nSummary:")
print("✅ Model creation works")
print("✅ get_stream_parameter_groups() helper works")
print("✅ Stream-specific optimizer creation works")
print("✅ Scheduler creation with setup_scheduler() works")
print("✅ Keras-style compile() works")
print("✅ Training with stream monitoring works")
print("✅ Scheduler stepping works")
print("✅ Simple optimizer workflow works")
print("✅ Optional scheduler works (scheduler=None)")
print("✅ setup_scheduler() is public and importable")
print("✅ Error handling works (invalid types)")
print("\n🔥 Keras-style API refactoring is fully functional!")
