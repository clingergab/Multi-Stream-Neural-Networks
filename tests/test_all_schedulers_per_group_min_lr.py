"""
Comprehensive test for per-group min_lr/eta_min support across ALL scheduler types.
"""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
from src.training.schedulers import setup_scheduler


def create_dummy_optimizer():
    """Create optimizer with 3 parameter groups (stream1, stream2, shared)."""
    # Dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.stream1 = nn.Linear(10, 5)
            self.stream2 = nn.Linear(10, 5)
            self.shared = nn.Linear(10, 2)

    model = DummyModel()

    param_groups = [
        {'params': model.stream1.parameters(), 'lr': 3e-4},
        {'params': model.stream2.parameters(), 'lr': 1.2e-3},
        {'params': model.shared.parameters(), 'lr': 1.5e-4},
    ]
    return torch.optim.AdamW(param_groups)


def test_scheduler_with_per_group_min_lr(scheduler_type, scheduler_kwargs, eta_min_list):
    """Test a scheduler type with per-group minimum LRs."""

    print(f"\n{'='*80}")
    print(f"Testing: {scheduler_type}")
    print(f"{'='*80}")

    optimizer = create_dummy_optimizer()

    # Setup scheduler with per-group min LRs
    scheduler = setup_scheduler(
        optimizer,
        scheduler_type=scheduler_type,
        epochs=100,
        train_loader_len=40,
        **scheduler_kwargs
    )

    print(f"✓ Created scheduler: {scheduler}")
    print(f"  eta_min/min_lr: {eta_min_list}")

    # Step through epochs
    print(f"\nLearning rates over epochs:")
    print(f"{'Epoch':<8} {'Stream1 LR':<15} {'Stream2 LR':<15} {'Shared LR':<15}")
    print("-" * 60)

    test_epochs = [0, 20, 40, 60, 80]
    for epoch in test_epochs:
        while scheduler.last_epoch < epoch:
            # ReduceLROnPlateau requires metrics
            if scheduler_type == 'plateau':
                scheduler.step(metrics=2.0)  # Dummy loss value
            else:
                scheduler.step()

        lrs = scheduler.get_last_lr()
        print(f"{epoch:<8} {lrs[0]:<15.2e} {lrs[1]:<15.2e} {lrs[2]:<15.2e}")

    # Verify minimum LRs are enforced
    final_lrs = scheduler.get_last_lr()
    for i, (lr, min_lr) in enumerate(zip(final_lrs, eta_min_list)):
        assert lr >= min_lr * 0.99, \
            f"Group {i} LR ({lr:.2e}) fell below min_lr ({min_lr:.2e})"

    print(f"✓ All groups respect their minimum LRs")
    print(f"  Final LRs: [{final_lrs[0]:.2e}, {final_lrs[1]:.2e}, {final_lrs[2]:.2e}]")
    print(f"  Min LRs:   [{eta_min_list[0]:.2e}, {eta_min_list[1]:.2e}, {eta_min_list[2]:.2e}]")

    return True


def main():
    """Test all scheduler types with per-group min_lr/eta_min."""

    print("=" * 80)
    print("COMPREHENSIVE TEST: Per-Group min_lr/eta_min for ALL Schedulers")
    print("=" * 80)

    eta_min_list = [1e-7, 1e-6, 5e-7]  # [stream1, stream2, shared]

    tests = []

    # Test 1: cosine
    tests.append(('cosine', {
        't_max': 80,
        'eta_min': eta_min_list,
    }, eta_min_list))

    # Test 2: cosine with warmup
    tests.append(('cosine + warmup', {
        't_max': 80,
        'eta_min': eta_min_list,
        'warmup_epochs': 3,
        'warmup_start_factor': 0.2,
    }, eta_min_list))

    # Test 3: decaying_cosine
    tests.append(('decaying_cosine', {
        't_max': 40,
        'eta_min': eta_min_list,
        'max_factor': 0.85,
        'min_factor': 0.85,
    }, eta_min_list))

    # Test 4: cosine_restarts
    tests.append(('cosine_restarts', {
        't_0': 20,
        't_mult': 1,
        'eta_min': eta_min_list,
    }, eta_min_list))

    # Test 5: decaying_cosine_restarts
    tests.append(('decaying_cosine_restarts', {
        't_0': 20,
        't_mult': 1,
        'eta_min': eta_min_list,
        'restart_decay': 0.8,
    }, eta_min_list))

    # Test 6: quadratic_inout
    tests.append(('quadratic_inout', {
        't_max': 80,
        'eta_min': eta_min_list,
    }, eta_min_list))

    # Test 7: cubic_inout
    tests.append(('cubic_inout', {
        't_max': 80,
        'eta_min': eta_min_list,
    }, eta_min_list))

    # Test 8: plateau (uses min_lr instead of eta_min)
    tests.append(('plateau', {
        'patience': 5,
        'factor': 0.5,
        'min_lr': eta_min_list,  # Note: min_lr, not eta_min
    }, eta_min_list))

    # Run all tests
    passed = 0
    failed = 0

    for test_name, kwargs, expected_min_lrs in tests:
        try:
            # Extract scheduler_type (remove any suffixes like "+ warmup")
            scheduler_type = test_name.split(' ')[0]

            test_scheduler_with_per_group_min_lr(scheduler_type, kwargs, expected_min_lrs)
            passed += 1
        except Exception as e:
            print(f"\n❌ Test FAILED: {test_name}")
            print(f"   Error: {e}")
            failed += 1
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 80)
    if failed == 0:
        print(f"✅ ALL {passed} TESTS PASSED!")
    else:
        print(f"⚠️  {passed} tests passed, {failed} tests failed")
    print("=" * 80)

    return failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
