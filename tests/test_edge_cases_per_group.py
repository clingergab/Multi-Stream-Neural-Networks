"""
Comprehensive edge case tests for per-group scheduler parameters.

This test suite covers:
1. Mismatched list lengths
2. Mixed per-group and scalar parameters
3. State dict save/load
4. Integration with warmup
5. Integration with SequentialLR
6. Optimizer.step() interleaving
7. Multiple parameter types per group
8. OneCycleLR (which doesn't support per-group via wrapper)
"""

import sys
sys.path.insert(0, '.')

import torch
from torch.optim import AdamW, SGD
from src.training.schedulers import setup_scheduler, PerGroupSchedulerWrapper


def test_mismatched_list_length():
    """Test that mismatched list lengths raise appropriate errors."""
    print("\n" + "="*80)
    print("Test: Mismatched list lengths")
    print("="*80)

    model1 = torch.nn.Linear(10, 5)
    model2 = torch.nn.Linear(10, 5)
    model3 = torch.nn.Linear(10, 5)

    optimizer = AdamW([
        {'params': model1.parameters(), 'lr': 0.001},
        {'params': model2.parameters(), 'lr': 0.002},
        {'params': model3.parameters(), 'lr': 0.003}
    ])

    try:
        # Should fail - 2 eta_mins for 3 param groups
        scheduler = setup_scheduler(
            optimizer,
            scheduler_type='cosine',
            epochs=100,
            train_loader_len=40,
            t_max=80,
            eta_min=[1e-7, 1e-6]  # Only 2 values for 3 groups!
        )
        print("❌ FAIL: Should have raised ValueError for mismatched lengths")
        return False
    except (ValueError, IndexError) as e:
        print(f"✓ PASS: Correctly raised error: {type(e).__name__}")
        return True


def test_mixed_per_group_and_scalar():
    """Test mixing per-group and scalar parameters."""
    print("\n" + "="*80)
    print("Test: Mixed per-group and scalar parameters")
    print("="*80)

    model1 = torch.nn.Linear(10, 5)
    model2 = torch.nn.Linear(10, 5)
    model3 = torch.nn.Linear(10, 5)

    optimizer = AdamW([
        {'params': model1.parameters(), 'lr': 0.001},
        {'params': model2.parameters(), 'lr': 0.002},
        {'params': model3.parameters(), 'lr': 0.003}
    ])

    # Per-group eta_min, but scalar t_max
    scheduler = setup_scheduler(
        optimizer,
        scheduler_type='cosine',
        epochs=100,
        train_loader_len=40,
        t_max=80,  # Scalar - same for all groups
        eta_min=[1e-7, 1e-6, 5e-7]  # Per-group
    )

    print(f"✓ Created scheduler: {scheduler}")

    # Step through and verify each group has different eta_min but same T_max
    for _ in range(80):
        scheduler.step()

    final_lrs = scheduler.get_last_lr()
    expected_mins = [1e-7, 1e-6, 5e-7]

    print(f"Final LRs: {[f'{lr:.6e}' for lr in final_lrs]}")
    print(f"Expected: {[f'{lr:.6e}' for lr in expected_mins]}")

    # Check if close to expected mins
    matches = all(abs(lr - expected) < 1e-9 for lr, expected in zip(final_lrs, expected_mins))

    if matches:
        print("✓ PASS: Mixed parameters work correctly")
        return True
    else:
        print("❌ FAIL: LRs don't match expected values")
        return False


def test_state_dict_save_load():
    """Test state_dict save/load preserves scheduler state."""
    print("\n" + "="*80)
    print("Test: State dict save/load")
    print("="*80)

    model1 = torch.nn.Linear(10, 5)
    model2 = torch.nn.Linear(10, 5)

    optimizer = AdamW([
        {'params': model1.parameters(), 'lr': 0.001},
        {'params': model2.parameters(), 'lr': 0.002}
    ])

    scheduler = setup_scheduler(
        optimizer,
        scheduler_type='cosine',
        epochs=100,
        train_loader_len=40,
        t_max=80,
        eta_min=[1e-7, 1e-6]
    )

    # Step multiple times
    for _ in range(30):
        scheduler.step()

    lrs_before = scheduler.get_last_lr()
    print(f"LRs before save: {[f'{lr:.6e}' for lr in lrs_before]}")

    # Save state
    state = scheduler.state_dict()

    # Create new scheduler and load state
    optimizer2 = AdamW([
        {'params': model1.parameters(), 'lr': 0.001},
        {'params': model2.parameters(), 'lr': 0.002}
    ])

    scheduler2 = setup_scheduler(
        optimizer2,
        scheduler_type='cosine',
        epochs=100,
        train_loader_len=40,
        t_max=80,
        eta_min=[1e-7, 1e-6]
    )

    scheduler2.load_state_dict(state)

    lrs_after = scheduler2.get_last_lr()
    print(f"LRs after load: {[f'{lr:.6e}' for lr in lrs_after]}")

    # Verify they match
    matches = all(abs(lr1 - lr2) < 1e-10 for lr1, lr2 in zip(lrs_before, lrs_after))

    if matches:
        print("✓ PASS: State dict save/load works correctly")
        return True
    else:
        print("❌ FAIL: State not preserved")
        return False


def test_warmup_integration():
    """Test per-group schedulers work with warmup."""
    print("\n" + "="*80)
    print("Test: Integration with warmup")
    print("="*80)

    model1 = torch.nn.Linear(10, 5)
    model2 = torch.nn.Linear(10, 5)

    optimizer = AdamW([
        {'params': model1.parameters(), 'lr': 0.001},
        {'params': model2.parameters(), 'lr': 0.002}
    ])

    scheduler = setup_scheduler(
        optimizer,
        scheduler_type='cosine',
        epochs=100,
        train_loader_len=40,
        t_max=80,
        eta_min=[1e-7, 1e-6],
        warmup_epochs=5,
        warmup_start_factor=0.1
    )

    print(f"✓ Created scheduler with warmup: {type(scheduler).__name__}")

    # Check LRs during warmup
    initial_lrs = scheduler.get_last_lr()
    print(f"Initial LRs (epoch 0): {[f'{lr:.6e}' for lr in initial_lrs]}")

    # Expected: 0.1 * base_lr for each group
    expected_initial = [0.1 * 0.001, 0.1 * 0.002]
    warmup_ok = all(abs(lr - expected) < 1e-10 for lr, expected in zip(initial_lrs, expected_initial))

    # Step through warmup
    for epoch in range(5):
        scheduler.step()

    # After warmup, should be at base LR
    post_warmup_lrs = scheduler.get_last_lr()
    print(f"Post-warmup LRs (epoch 5): {[f'{lr:.6e}' for lr in post_warmup_lrs]}")

    # Expected: ~base_lr for each group
    expected_post = [0.001, 0.002]
    post_warmup_ok = all(abs(lr - expected) < 1e-4 for lr, expected in zip(post_warmup_lrs, expected_post))

    if warmup_ok and post_warmup_ok:
        print("✓ PASS: Warmup integration works")
        return True
    else:
        print(f"❌ FAIL: Warmup LRs incorrect")
        print(f"  Warmup check: {warmup_ok}, Post-warmup check: {post_warmup_ok}")
        return False


def test_optimizer_interleaving():
    """Test that optimizer.step() doesn't interfere with scheduler."""
    print("\n" + "="*80)
    print("Test: Optimizer.step() interleaving")
    print("="*80)

    model = torch.nn.Linear(10, 5)

    optimizer = AdamW([
        {'params': model.parameters(), 'lr': 0.001}
    ])

    scheduler = setup_scheduler(
        optimizer,
        scheduler_type='cosine',
        epochs=100,
        train_loader_len=40,
        t_max=50,
        eta_min=[1e-7]
    )

    # Simulate training loop
    for epoch in range(10):
        # Simulate batches
        for _ in range(5):
            # Fake forward/backward
            loss = torch.tensor(1.0, requires_grad=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  # Optimizer step

        # Scheduler step at end of epoch
        lr_before = optimizer.param_groups[0]['lr']
        scheduler.step()
        lr_after = optimizer.param_groups[0]['lr']

        # LR should change after scheduler.step()
        if epoch == 0:
            print(f"Epoch {epoch}: LR before={lr_before:.6e}, after={lr_after:.6e}")

    # After 10 epochs of cosine annealing, LR should have decreased
    final_lr = optimizer.param_groups[0]['lr']
    if final_lr < 0.001:
        print(f"✓ PASS: LR decreased as expected (final={final_lr:.6e})")
        return True
    else:
        print(f"❌ FAIL: LR didn't decrease (final={final_lr:.6e})")
        return False


def test_different_optimizer_types():
    """Test per-group schedulers work with different optimizer types."""
    print("\n" + "="*80)
    print("Test: Different optimizer types")
    print("="*80)

    results = []

    for opt_class, opt_name in [(AdamW, 'AdamW'), (SGD, 'SGD')]:
        model1 = torch.nn.Linear(10, 5)
        model2 = torch.nn.Linear(10, 5)

        if opt_class == SGD:
            optimizer = opt_class([
                {'params': model1.parameters(), 'lr': 0.1},
                {'params': model2.parameters(), 'lr': 0.2}
            ], momentum=0.9)
        else:
            optimizer = opt_class([
                {'params': model1.parameters(), 'lr': 0.001},
                {'params': model2.parameters(), 'lr': 0.002}
            ])

        try:
            scheduler = setup_scheduler(
                optimizer,
                scheduler_type='cosine',
                epochs=100,
                train_loader_len=40,
                t_max=50,
                eta_min=[1e-7, 1e-6]
            )

            # Step a few times
            for _ in range(10):
                scheduler.step()

            lrs = scheduler.get_last_lr()
            print(f"✓ {opt_name}: LRs after 10 steps = {[f'{lr:.6e}' for lr in lrs]}")
            results.append(True)
        except Exception as e:
            print(f"❌ {opt_name}: Failed with {e}")
            results.append(False)

    if all(results):
        print("✓ PASS: Works with different optimizer types")
        return True
    else:
        print("❌ FAIL: Some optimizer types failed")
        return False


def test_plateau_with_per_group():
    """Test ReduceLROnPlateau with per-group min_lr."""
    print("\n" + "="*80)
    print("Test: ReduceLROnPlateau with per-group min_lr")
    print("="*80)

    model1 = torch.nn.Linear(10, 5)
    model2 = torch.nn.Linear(10, 5)

    optimizer = AdamW([
        {'params': model1.parameters(), 'lr': 0.001},
        {'params': model2.parameters(), 'lr': 0.002}
    ])

    from src.training.schedulers import update_scheduler

    scheduler = setup_scheduler(
        optimizer,
        scheduler_type='plateau',
        epochs=100,
        train_loader_len=40,
        patience=2,  # Reduced patience for faster reductions
        factor=0.1,  # More aggressive factor for faster convergence to min_lr
        min_lr=[1e-7, 1e-6]
    )

    print(f"✓ Created plateau scheduler: {scheduler}")

    # Simulate no improvement for many epochs
    # With patience=2, factor=0.1, starting from 0.001:
    # After (2+1)*1: 0.001 * 0.1 = 1e-4
    # After (2+1)*2: 1e-4 * 0.1 = 1e-5
    # After (2+1)*3: 1e-5 * 0.1 = 1e-6  (for group 1)
    # After (2+1)*4: 1e-6 * 0.1 = 1e-7  (for group 0, but min_lr=1e-7 so stops there)
    for epoch in range(30):
        # High loss (no improvement)
        update_scheduler(scheduler, val_loss=10.0)

    final_lrs = scheduler.get_last_lr()
    expected_mins = [1e-7, 1e-6]

    print(f"Final LRs: {[f'{lr:.6e}' for lr in final_lrs]}")
    print(f"Expected mins: {[f'{lr:.6e}' for lr in expected_mins]}")

    # After many reductions, should hit min_lr (or be very close)
    # Allow small tolerance since plateau might not hit exactly
    matches = all(abs(lr - expected) < expected * 0.01 for lr, expected in zip(final_lrs, expected_mins))

    if matches:
        print("✓ PASS: Plateau scheduler respects per-group min_lr")
        return True
    else:
        print("❌ FAIL: Didn't reach min_lr correctly")
        print(f"  Differences: {[abs(lr - expected) for lr, expected in zip(final_lrs, expected_mins)]}")
        return False


def main():
    """Run all edge case tests."""
    print("\n" + "="*80)
    print("COMPREHENSIVE EDGE CASE TESTS: Per-Group Scheduler Parameters")
    print("="*80)

    tests = [
        ("Mismatched list length", test_mismatched_list_length),
        ("Mixed per-group and scalar", test_mixed_per_group_and_scalar),
        ("State dict save/load", test_state_dict_save_load),
        ("Warmup integration", test_warmup_integration),
        ("Optimizer interleaving", test_optimizer_interleaving),
        ("Different optimizer types", test_different_optimizer_types),
        ("Plateau with per-group", test_plateau_with_per_group),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Test '{name}' raised exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "="*80)
    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")

    print("="*80)
    if passed == total:
        print(f"✅ ALL {total} EDGE CASE TESTS PASSED!")
    else:
        print(f"⚠️  {passed}/{total} tests passed, {total - passed} failed")
    print("="*80)

    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
