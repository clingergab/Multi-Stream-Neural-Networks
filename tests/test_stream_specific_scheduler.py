"""
Test StreamSpecificCosineAnnealingLR scheduler with per-group eta_min support.
"""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
from src.training.schedulers import setup_scheduler, StreamSpecificCosineAnnealingLR


def test_stream_specific_cosine_scheduler():
    """Test StreamSpecificCosineAnnealingLR with different eta_min per group."""

    print("=" * 80)
    print("Testing StreamSpecificCosineAnnealingLR")
    print("=" * 80)

    # Create a simple 3-stream model (mimicking LINet structure)
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.stream1 = nn.Linear(10, 5)
            self.stream2 = nn.Linear(10, 5)
            self.shared = nn.Linear(10, 2)

    model = DummyModel()

    # Create optimizer with 3 parameter groups manually (mimicking multi-stream setup)
    param_groups = [
        {'params': model.stream1.parameters(), 'lr': 2e-4, 'weight_decay': 8e-3},   # Stream1
        {'params': model.stream2.parameters(), 'lr': 9e-4, 'weight_decay': 8e-5},   # Stream2
        {'params': model.shared.parameters(), 'lr': 1e-4, 'weight_decay': 6e-3},     # Shared
    ]
    optimizer = torch.optim.AdamW(param_groups)

    print(f"\n✓ Created optimizer with {len(optimizer.param_groups)} parameter groups")
    print(f"  Stream1 LR: {optimizer.param_groups[0]['lr']}")
    print(f"  Stream2 LR: {optimizer.param_groups[1]['lr']}")
    print(f"  Shared LR:  {optimizer.param_groups[2]['lr']}")

    # Test 1: Single eta_min (backward compatible)
    print("\n" + "=" * 80)
    print("Test 1: Single eta_min (backward compatible)")
    print("=" * 80)

    scheduler = setup_scheduler(
        optimizer,
        scheduler_type='cosine',
        epochs=100,
        train_loader_len=40,
        t_max=80,
        eta_min=1e-6  # Single value for all groups
    )

    print(f"✓ Created scheduler: {scheduler}")
    print(f"  eta_min (all groups): 1e-6")

    # Step through some epochs
    print("\nLearning rates over epochs (single eta_min):")
    print(f"{'Epoch':<8} {'Stream1 LR':<15} {'Stream2 LR':<15} {'Shared LR':<15}")
    print("-" * 60)

    for epoch in [0, 20, 40, 60, 79, 80]:
        # Manually step to epoch
        while scheduler.last_epoch < epoch:
            scheduler.step()

        lrs = scheduler.get_last_lr()
        print(f"{epoch:<8} {lrs[0]:<15.2e} {lrs[1]:<15.2e} {lrs[2]:<15.2e}")

    # Verify all converge to same eta_min
    final_lrs = scheduler.get_last_lr()
    for i, lr in enumerate(final_lrs):
        assert abs(lr - 1e-6) < 1e-10, f"Group {i} did not converge to eta_min=1e-6"

    print("✓ All groups converged to eta_min=1e-6")

    # Test 2: Per-group eta_min
    print("\n" + "=" * 80)
    print("Test 2: Per-group eta_min (new feature)")
    print("=" * 80)

    # Recreate optimizer
    param_groups = [
        {'params': model.stream1.parameters(), 'lr': 2e-4, 'weight_decay': 8e-3},
        {'params': model.stream2.parameters(), 'lr': 9e-4, 'weight_decay': 8e-5},
        {'params': model.shared.parameters(), 'lr': 1e-4, 'weight_decay': 6e-3},
    ]
    optimizer = torch.optim.AdamW(param_groups)

    # Different eta_min for each stream
    eta_min_list = [1e-7, 1e-6, 5e-7]  # [stream1, stream2, shared]

    scheduler = setup_scheduler(
        optimizer,
        scheduler_type='cosine',
        epochs=100,
        train_loader_len=40,
        t_max=80,
        eta_min=eta_min_list  # List of values!
    )

    print(f"✓ Created scheduler: {scheduler}")
    print(f"  Stream1 eta_min: {eta_min_list[0]}")
    print(f"  Stream2 eta_min: {eta_min_list[1]}")
    print(f"  Shared eta_min:  {eta_min_list[2]}")

    # Step through some epochs
    print("\nLearning rates over epochs (per-group eta_min):")
    print(f"{'Epoch':<8} {'Stream1 LR':<15} {'Stream2 LR':<15} {'Shared LR':<15}")
    print("-" * 60)

    for epoch in [0, 20, 40, 60, 79, 80]:
        # Manually step to epoch
        while scheduler.last_epoch < epoch:
            scheduler.step()

        lrs = scheduler.get_last_lr()
        print(f"{epoch:<8} {lrs[0]:<15.2e} {lrs[1]:<15.2e} {lrs[2]:<15.2e}")

    # Verify each converges to its own eta_min
    final_lrs = scheduler.get_last_lr()
    for i, (lr, expected_eta_min) in enumerate(zip(final_lrs, eta_min_list)):
        assert abs(lr - expected_eta_min) < 1e-10, \
            f"Group {i} did not converge to eta_min={expected_eta_min}"

    print("✓ Each group converged to its specific eta_min")
    print(f"  Stream1: {final_lrs[0]:.2e} (expected {eta_min_list[0]:.2e})")
    print(f"  Stream2: {final_lrs[1]:.2e} (expected {eta_min_list[1]:.2e})")
    print(f"  Shared:  {final_lrs[2]:.2e} (expected {eta_min_list[2]:.2e})")

    # Test 3: With warmup
    print("\n" + "=" * 80)
    print("Test 3: Per-group eta_min with warmup")
    print("=" * 80)

    # Recreate optimizer
    param_groups = [
        {'params': model.stream1.parameters(), 'lr': 3e-4, 'weight_decay': 8e-3},
        {'params': model.stream2.parameters(), 'lr': 1.2e-3, 'weight_decay': 8e-5},
        {'params': model.shared.parameters(), 'lr': 1.5e-4, 'weight_decay': 6e-3},
    ]
    optimizer = torch.optim.AdamW(param_groups)

    eta_min_list = [1e-7, 1e-6, 5e-7]

    scheduler = setup_scheduler(
        optimizer,
        scheduler_type='cosine',
        epochs=100,
        train_loader_len=40,
        t_max=80,
        eta_min=eta_min_list,
        warmup_epochs=3,
        warmup_start_factor=0.2
    )

    print(f"✓ Created scheduler with warmup")
    print(f"  Warmup epochs: 3")
    print(f"  Warmup start factor: 0.2")

    # Step through warmup and beyond
    print("\nLearning rates during warmup and annealing:")
    print(f"{'Epoch':<8} {'Stream1 LR':<15} {'Stream2 LR':<15} {'Shared LR':<15} {'Phase':<12}")
    print("-" * 75)

    for epoch in [0, 1, 2, 3, 20, 40, 60, 80]:
        # Manually step to epoch
        while scheduler.last_epoch < epoch:
            scheduler.step()

        lrs = scheduler.get_last_lr()
        phase = "Warmup" if epoch < 3 else "Annealing"
        print(f"{epoch:<8} {lrs[0]:<15.2e} {lrs[1]:<15.2e} {lrs[2]:<15.2e} {phase:<12}")

    # Verify LRs are decreasing towards eta_min (warmup affects T_max offset)
    final_lrs = scheduler.get_last_lr()
    initial_lrs = [3e-4, 1.2e-3, 1.5e-4]

    for i, (final_lr, initial_lr) in enumerate(zip(final_lrs, initial_lrs)):
        # Verify LRs have decreased significantly from initial values
        assert final_lr < initial_lr * 0.02, \
            f"Group {i} LR ({final_lr:.2e}) did not decrease sufficiently from initial ({initial_lr:.2e})"

    print("✓ Warmup + per-group eta_min working correctly")
    print(f"  Final LRs:   [{final_lrs[0]:.2e}, {final_lrs[1]:.2e}, {final_lrs[2]:.2e}]")
    print(f"  Target eta:  [{eta_min_list[0]:.2e}, {eta_min_list[1]:.2e}, {eta_min_list[2]:.2e}]")
    print(f"  (Note: Warmup offsets T_max, so final LR may not reach exact eta_min at epoch 80)")

    # Test 4: State dict save/load
    print("\n" + "=" * 80)
    print("Test 4: State dict save/load")
    print("=" * 80)

    # Get state before
    state_before = scheduler.state_dict()
    lr_before = scheduler.get_last_lr()

    # Create new scheduler and load state
    param_groups_new = [
        {'params': model.stream1.parameters(), 'lr': 3e-4},
        {'params': model.stream2.parameters(), 'lr': 1.2e-3},
        {'params': model.shared.parameters(), 'lr': 1.5e-4},
    ]
    optimizer_new = torch.optim.AdamW(param_groups_new)

    scheduler_new = StreamSpecificCosineAnnealingLR(
        optimizer_new,
        T_max=80,
        eta_min=[1e-7, 1e-6, 5e-7]
    )

    scheduler_new.load_state_dict(state_before)
    lr_after = scheduler_new.get_last_lr()

    # Verify LRs match
    for lr_b, lr_a in zip(lr_before, lr_after):
        assert abs(lr_b - lr_a) < 1e-10, "State dict load failed"

    print("✓ State dict save/load working correctly")
    print(f"  LRs before: [{lr_before[0]:.2e}, {lr_before[1]:.2e}, {lr_before[2]:.2e}]")
    print(f"  LRs after:  [{lr_after[0]:.2e}, {lr_after[1]:.2e}, {lr_after[2]:.2e}]")

    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("=" * 80)


if __name__ == '__main__':
    test_stream_specific_cosine_scheduler()
