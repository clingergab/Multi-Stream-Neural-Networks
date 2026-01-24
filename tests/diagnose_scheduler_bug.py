#!/usr/bin/env python3
"""
Comprehensive diagnostic to identify the LR scheduler bug.

Run this script to verify your scheduler setup works correctly in isolation.
If this works but your training doesn't, the bug is in your specific training environment.

Usage:
    python tests/diagnose_scheduler_bug.py
"""

import sys
sys.path.insert(0, '.')

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
from src.training.schedulers import setup_scheduler, PerGroupSchedulerWrapper


def test_basic_sequential():
    """Test basic SequentialLR with LinearLR warmup."""
    print("=" * 70)
    print("TEST 1: Basic SequentialLR with LinearLR warmup")
    print("=" * 70)

    model = torch.nn.Linear(10, 10)
    optimizer = AdamW(model.parameters(), lr=8.0e-6)

    warmup_epochs = 5
    start_factor = 0.2

    warmup = LinearLR(optimizer, start_factor=start_factor, end_factor=1.0, total_iters=warmup_epochs)
    main = CosineAnnealingLR(optimizer, T_max=120, eta_min=1e-7)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, main], milestones=[warmup_epochs])

    print(f"Config: warmup_epochs={warmup_epochs}, start_factor={start_factor}, base_lr=8.0e-6")
    print()

    history = []
    for epoch in range(10):
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        history.append(lr)

    # Check results
    expected_epoch5 = 8.0e-6  # Should reach full LR
    actual_epoch5 = history[4]  # 0-indexed

    print("LR history (first 7 epochs):")
    for i, lr in enumerate(history[:7]):
        if i < warmup_epochs:
            expected = 8.0e-6 * (start_factor + (1 - start_factor) * (i + 1) / warmup_epochs)
        else:
            expected = 8.0e-6
        match = "✓" if abs(lr - expected) / expected < 0.01 else "✗"
        print(f"  Epoch {i+1}: {lr:.6e} (expected {expected:.6e}) {match}")

    if abs(actual_epoch5 - expected_epoch5) / expected_epoch5 < 0.01:
        print("\n✅ TEST 1 PASSED: Warmup completes correctly")
        return True
    else:
        print(f"\n❌ TEST 1 FAILED: Epoch 5 should be {expected_epoch5:.2e}, got {actual_epoch5:.2e}")
        return False


def test_setup_scheduler_basic():
    """Test setup_scheduler without per-group eta_min."""
    print("\n" + "=" * 70)
    print("TEST 2: setup_scheduler() without per-group eta_min")
    print("=" * 70)

    model = torch.nn.Linear(10, 10)
    stream_lrs = [4.0e-5, 2.0e-4]
    shared_lr = 8.0e-6

    optimizer = AdamW([
        {'params': [torch.nn.Parameter(torch.zeros(10))], 'lr': stream_lrs[0]},
        {'params': [torch.nn.Parameter(torch.zeros(10))], 'lr': stream_lrs[1]},
        {'params': model.parameters(), 'lr': shared_lr},
    ])

    scheduler = setup_scheduler(
        optimizer=optimizer,
        scheduler_type='cosine',
        epochs=120,
        train_loader_len=100,
        warmup_epochs=5,
        warmup_start_factor=0.2,
        t_max=120,
        eta_min=1e-7  # Scalar eta_min
    )

    print(f"Scheduler type: {type(scheduler).__name__}")
    if hasattr(scheduler, '_schedulers'):
        print(f"Inner schedulers: {[type(s).__name__ for s in scheduler._schedulers]}")

    history = []
    for epoch in range(10):
        scheduler.step()
        lrs = [pg['lr'] for pg in optimizer.param_groups]
        history.append(lrs)

    # Check epoch 5 (index 4)
    epoch5_lrs = history[4]
    expected = [stream_lrs[0], stream_lrs[1], shared_lr]

    print("\nEpoch 5 LRs (end of warmup):")
    all_match = True
    for i, (actual, exp) in enumerate(zip(epoch5_lrs, expected)):
        match = abs(actual - exp) / exp < 0.01
        all_match = all_match and match
        print(f"  Group {i}: {actual:.6e} (expected {exp:.6e}) {'✓' if match else '✗'}")

    if all_match:
        print("\n✅ TEST 2 PASSED: All groups reach target LR")
        return True
    else:
        print("\n❌ TEST 2 FAILED: Some groups didn't reach target LR")
        return False


def test_setup_scheduler_per_group():
    """Test setup_scheduler WITH per-group eta_min (triggers PerGroupSchedulerWrapper)."""
    print("\n" + "=" * 70)
    print("TEST 3: setup_scheduler() WITH per-group eta_min")
    print("=" * 70)

    model = torch.nn.Linear(10, 10)
    stream_lrs = [4.0e-5, 2.0e-4]
    shared_lr = 8.0e-6

    optimizer = AdamW([
        {'params': [torch.nn.Parameter(torch.zeros(10))], 'lr': stream_lrs[0]},
        {'params': [torch.nn.Parameter(torch.zeros(10))], 'lr': stream_lrs[1]},
        {'params': model.parameters(), 'lr': shared_lr},
    ])

    scheduler = setup_scheduler(
        optimizer=optimizer,
        scheduler_type='cosine',
        epochs=120,
        train_loader_len=100,
        warmup_epochs=5,
        warmup_start_factor=0.2,
        t_max=120,
        eta_min=[1e-7, 5e-7, 1e-8]  # Per-group eta_min - triggers PerGroupSchedulerWrapper
    )

    print(f"Scheduler type: {type(scheduler).__name__}")
    if hasattr(scheduler, '_schedulers'):
        print(f"Inner schedulers: {[type(s).__name__ for s in scheduler._schedulers]}")
        for i, s in enumerate(scheduler._schedulers):
            if isinstance(s, PerGroupSchedulerWrapper):
                print(f"  Scheduler {i} is PerGroupSchedulerWrapper with {len(s.schedulers)} sub-schedulers")

    history = []
    for epoch in range(10):
        scheduler.step()
        lrs = [pg['lr'] for pg in optimizer.param_groups]
        history.append(lrs)

    # Check epoch 5 (index 4)
    epoch5_lrs = history[4]
    expected = [stream_lrs[0], stream_lrs[1], shared_lr]

    print("\nEpoch 5 LRs (end of warmup):")
    all_match = True
    for i, (actual, exp) in enumerate(zip(epoch5_lrs, expected)):
        match = abs(actual - exp) / exp < 0.01
        all_match = all_match and match
        print(f"  Group {i}: {actual:.6e} (expected {exp:.6e}) {'✓' if match else '✗'}")

    if all_match:
        print("\n✅ TEST 3 PASSED: PerGroupSchedulerWrapper works with warmup")
        return True
    else:
        print("\n❌ TEST 3 FAILED: PerGroupSchedulerWrapper has bug with warmup")
        return False


def test_your_exact_config():
    """Test with your exact config values from the question."""
    print("\n" + "=" * 70)
    print("TEST 4: Your exact config from the question")
    print("=" * 70)

    # Your EXACT config
    STREAM_SPECIFIC_CONFIG = {
        'stream_lrs': [4.0e-5, 2.0e-4],
        'stream_weight_decays': [7.0e-4, 9.0e-5],
        'shared_lr': 8.0e-6,
        'shared_weight_decay': 9e-4,
    }

    warmup_epochs = 5
    start_factor = 0.2

    model = torch.nn.Linear(10, 10)
    stream_lrs = STREAM_SPECIFIC_CONFIG['stream_lrs']
    shared_lr = STREAM_SPECIFIC_CONFIG['shared_lr']

    optimizer = AdamW([
        {'params': [torch.nn.Parameter(torch.zeros(10))], 'lr': stream_lrs[0],
         'weight_decay': STREAM_SPECIFIC_CONFIG['stream_weight_decays'][0]},
        {'params': [torch.nn.Parameter(torch.zeros(10))], 'lr': stream_lrs[1],
         'weight_decay': STREAM_SPECIFIC_CONFIG['stream_weight_decays'][1]},
        {'params': model.parameters(), 'lr': shared_lr,
         'weight_decay': STREAM_SPECIFIC_CONFIG['shared_weight_decay']},
    ])

    # Create scheduler like your notebook does
    scheduler = setup_scheduler(
        optimizer=optimizer,
        scheduler_type='cosine',
        epochs=120,
        train_loader_len=106,  # Your train_loader length
        warmup_epochs=warmup_epochs,
        warmup_start_factor=start_factor,
        t_max=120,
        eta_min=1e-7
    )

    print(f"stream_lrs = {stream_lrs}")
    print(f"shared_lr = {shared_lr}")
    print(f"warmup_epochs = {warmup_epochs}, start_factor = {start_factor}")
    print(f"Scheduler type: {type(scheduler).__name__}")

    # Simulate your training loop exactly
    history_learning_rates = []

    for epoch in range(10):  # 0-indexed like your code
        # Step scheduler (like line 990 in li_net.py)
        scheduler.step()

        # Get current_lr (like line 991)
        current_lr = optimizer.param_groups[-1]['lr']  # Shared group

        # Append to history (like line 623 in model_helpers.py)
        history_learning_rates.append(current_lr)

    print("\nSimulated history['learning_rates']:")
    your_data = [2.88e-06, 4.16e-06, 5.44e-06, 6.72e-06, 6.72e-06, 6.72e-06, 6.71e-06]

    all_match = True
    for i in range(7):
        sim = history_learning_rates[i]
        yours = your_data[i]

        if i < warmup_epochs:
            expected = shared_lr * (start_factor + (1 - start_factor) * (i + 1) / warmup_epochs)
        else:
            expected = shared_lr

        sim_match = abs(sim - expected) / expected < 0.02
        yours_match = abs(yours - expected) / expected < 0.02

        if not yours_match:
            all_match = False

        print(f"  Index {i} (Epoch {i+1}): simulated={sim:.6e}, yours={yours:.6e}, expected={expected:.6e}")
        print(f"                         simulated {'✓' if sim_match else '✗'}, yours {'✓' if yours_match else '✗'}")

    if all_match:
        print("\n✅ TEST 4 PASSED: Your data matches expected")
        return True
    else:
        print("\n❌ TEST 4 FAILED: Your data has the bug, but simulation doesn't!")
        print("\n   This means the bug is in your specific training environment, NOT in the scheduler code.")
        print("   Possible causes:")
        print("   1. You're running an older version of the scheduler code")
        print("   2. Something is stepping the scheduler twice per epoch")
        print("   3. You loaded from a checkpoint with corrupted scheduler state")
        print("   4. Something else is modifying optimizer LRs directly")
        return False


def main():
    print("=" * 70)
    print("SCHEDULER DIAGNOSTIC TESTS")
    print("=" * 70)
    print()

    results = []
    results.append(("Basic SequentialLR", test_basic_sequential()))
    results.append(("setup_scheduler (scalar eta_min)", test_setup_scheduler_basic()))
    results.append(("setup_scheduler (per-group eta_min)", test_setup_scheduler_per_group()))
    results.append(("Your exact config", test_your_exact_config()))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_passed = True
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {name}: {status}")
        all_passed = all_passed and passed

    if all_passed:
        print("\n✅ All tests passed! The scheduler code is working correctly.")
        print("   If your training still shows the bug, check:")
        print("   1. Are you running the latest version of schedulers.py?")
        print("   2. Are you loading from a checkpoint?")
        print("   3. Is something else modifying optimizer LRs?")
    else:
        print("\n❌ Some tests failed. Check the output above for details.")

    return all_passed


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore', message='.*lr_scheduler.step.*')
    success = main()
    sys.exit(0 if success else 1)
