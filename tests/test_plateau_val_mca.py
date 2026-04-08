"""
Test: Plateau scheduler with val_mca monitoring.

Verifies that when monitor='val_mca' and scheduler mode='max',
the ReduceOnPlateau scheduler receives val_mca as its metric
and reduces LR when val_mca stops improving.
"""

import sys
sys.path.insert(0, '.')

import torch
from torch.utils.data import Dataset, DataLoader
from unittest.mock import patch, MagicMock

from src.models.linear_integration.li_net3.li_net import LINet
from src.models.linear_integration.li_net3 import li_resnet18
from src.training.schedulers import setup_scheduler

DEVICE = 'cpu'
NUM_CLASSES = 10
IMAGE_SIZE = 32
BATCH_SIZE = 4


class NStreamDataset(Dataset):
    def __init__(self, stream_data_list, labels):
        self.stream_data = stream_data_list
        self.labels = labels

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        # Model expects tuple: (stream1, stream2, ..., labels)
        return (*[s[idx] for s in self.stream_data], self.labels[idx])


def make_loaders(n_train=20, n_val=12):
    def make(n):
        rgb = torch.randn(n, 3, IMAGE_SIZE, IMAGE_SIZE)
        depth = torch.randn(n, 1, IMAGE_SIZE, IMAGE_SIZE)
        labels = torch.randint(0, NUM_CLASSES, (n,))
        return DataLoader(NStreamDataset([rgb, depth], labels),
                          batch_size=BATCH_SIZE, shuffle=False)
    return make(n_train), make(n_val)


def test_plateau_receives_val_mca():
    """Plateau scheduler.step() is called with val_mca when monitor='val_mca'."""
    print("\n" + "=" * 70)
    print("TEST: Plateau scheduler receives val_mca metric")
    print("=" * 70)

    model = li_resnet18(
        num_classes=NUM_CLASSES,
        stream_input_channels=[3, 1],
        width_multiplier=0.25,
        device=DEVICE,
        use_amp=False,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    scheduler = setup_scheduler(
        optimizer,
        scheduler_type='plateau',
        mode='max',
        scheduler_patience=2,
        factor=0.5,
        min_lr=1e-7,
        warmup_epochs=0,
    )

    model.compile(optimizer=optimizer, scheduler=scheduler, loss='cross_entropy')

    train_loader, val_loader = make_loaders()

    # Patch the scheduler's step to capture what metric it receives
    step_calls = []
    original_step = scheduler.step

    def tracking_step(metric=None):
        step_calls.append(metric)
        return original_step(metric)

    model.scheduler.step = tracking_step

    model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=3,
        verbose=False,
        monitor='val_mca',
    )

    assert len(step_calls) == 3, f"Expected 3 step calls, got {len(step_calls)}"
    for i, metric in enumerate(step_calls):
        assert metric is not None, f"Epoch {i}: scheduler.step() called without metric"
        assert isinstance(metric, float), f"Epoch {i}: metric should be float, got {type(metric)}"
        # val_mca is between 0 and 1
        assert 0.0 <= metric <= 1.0, f"Epoch {i}: metric {metric} not in [0,1] range for MCA"

    print(f"  Metrics passed to scheduler: {[f'{m:.4f}' for m in step_calls]}")
    print("  PASS: Plateau scheduler received val_mca for all epochs")


def test_plateau_reduces_lr_on_mca_plateau():
    """LR actually decreases when val_mca stops improving."""
    print("\n" + "=" * 70)
    print("TEST: LR reduces when val_mca plateaus")
    print("=" * 70)

    model = li_resnet18(
        num_classes=NUM_CLASSES,
        stream_input_channels=[3, 1],
        width_multiplier=0.25,
        device=DEVICE,
        use_amp=False,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # patience=1 so LR drops quickly after 1 epoch of no improvement
    scheduler = setup_scheduler(
        optimizer,
        scheduler_type='plateau',
        mode='max',
        scheduler_patience=1,
        factor=0.5,
        min_lr=1e-7,
        warmup_epochs=0,
    )

    model.compile(optimizer=optimizer, scheduler=scheduler, loss='cross_entropy')

    train_loader, val_loader = make_loaders()

    # Train enough epochs that the random model's val_mca will plateau
    history = model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=10,
        verbose=False,
        monitor='val_mca',
    )

    lrs = history['learning_rates']
    initial_lr = lrs[0]
    final_lr = lrs[-1]

    print(f"  Initial LR: {initial_lr:.2e}")
    print(f"  Final LR:   {final_lr:.2e}")
    print(f"  LR history: {[f'{lr:.2e}' for lr in lrs]}")

    # With patience=1, random val_mca should plateau and trigger LR reduction
    assert final_lr < initial_lr, (
        f"LR did not decrease: initial={initial_lr:.2e}, final={final_lr:.2e}. "
        "Plateau scheduler may not be receiving val_mca."
    )

    # Count reductions
    reductions = sum(1 for i in range(1, len(lrs)) if lrs[i] < lrs[i-1] * 0.9)
    print(f"  LR reductions: {reductions}")
    print("  PASS: LR decreased when val_mca plateaued")


def test_plateau_mode_min_still_uses_loss():
    """When mode='min', scheduler still uses val_loss (not val_mca), even if monitor='val_mca'."""
    print("\n" + "=" * 70)
    print("TEST: mode='min' still uses val_loss (backward compat)")
    print("=" * 70)

    model = li_resnet18(
        num_classes=NUM_CLASSES,
        stream_input_channels=[3, 1],
        width_multiplier=0.25,
        device=DEVICE,
        use_amp=False,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    scheduler = setup_scheduler(
        optimizer,
        scheduler_type='plateau',
        mode='min',
        scheduler_patience=2,
        factor=0.5,
        min_lr=1e-7,
        warmup_epochs=0,
    )

    model.compile(optimizer=optimizer, scheduler=scheduler, loss='cross_entropy')

    train_loader, val_loader = make_loaders()

    step_calls = []
    original_step = scheduler.step

    def tracking_step(metric=None):
        step_calls.append(metric)
        return original_step(metric)

    model.scheduler.step = tracking_step

    model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=3,
        verbose=False,
        monitor='val_mca',  # monitor is val_mca but scheduler mode is min
    )

    assert len(step_calls) == 3
    for i, metric in enumerate(step_calls):
        assert metric is not None
        # val_loss is typically > 1.0 for cross-entropy with 10 classes
        # val_mca is 0-1. This distinguishes them.
        assert metric > 0.5, (
            f"Epoch {i}: metric {metric:.4f} looks like MCA, not loss. "
            "mode='min' should use val_loss."
        )

    print(f"  Metrics passed to scheduler: {[f'{m:.4f}' for m in step_calls]}")
    print("  PASS: mode='min' correctly used val_loss, not val_mca")


def test_plateau_with_warmup_receives_val_mca():
    """WarmupReduceLROnPlateau also passes val_mca correctly after warmup."""
    print("\n" + "=" * 70)
    print("TEST: WarmupReduceLROnPlateau receives val_mca after warmup")
    print("=" * 70)

    model = li_resnet18(
        num_classes=NUM_CLASSES,
        stream_input_channels=[3, 1],
        width_multiplier=0.25,
        device=DEVICE,
        use_amp=False,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    scheduler = setup_scheduler(
        optimizer,
        scheduler_type='plateau',
        mode='max',
        scheduler_patience=1,
        factor=0.5,
        min_lr=1e-7,
        warmup_epochs=2,
        warmup_start_factor=0.1,
    )

    model.compile(optimizer=optimizer, scheduler=scheduler, loss='cross_entropy')

    train_loader, val_loader = make_loaders()

    # Track what the inner plateau scheduler receives
    from src.training.schedulers import WarmupReduceLROnPlateau
    assert isinstance(scheduler, WarmupReduceLROnPlateau), \
        f"Expected WarmupReduceLROnPlateau, got {type(scheduler)}"

    plateau_step_calls = []
    original_plateau_step = scheduler.plateau_scheduler.step

    def tracking_plateau_step(metrics=None):
        plateau_step_calls.append(metrics)
        return original_plateau_step(metrics)

    scheduler.plateau_scheduler.step = tracking_plateau_step

    history = model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=6,
        verbose=False,
        monitor='val_mca',
    )

    # First 2 epochs use warmup, epochs 3-6 use plateau
    print(f"  Warmup epochs: 2, Total epochs: 6")
    print(f"  Plateau step calls: {len(plateau_step_calls)}")
    print(f"  Plateau metrics: {[f'{m:.4f}' if m is not None else 'None' for m in plateau_step_calls]}")

    assert len(plateau_step_calls) == 4, \
        f"Expected 4 plateau step calls (epochs 3-6), got {len(plateau_step_calls)}"

    for i, metric in enumerate(plateau_step_calls):
        assert metric is not None, f"Plateau step {i}: called without metric"
        assert 0.0 <= metric <= 1.0, f"Plateau step {i}: metric {metric} not in MCA range"

    print("  PASS: WarmupReduceLROnPlateau correctly forwards val_mca to plateau scheduler")


if __name__ == '__main__':
    test_plateau_receives_val_mca()
    test_plateau_reduces_lr_on_mca_plateau()
    test_plateau_mode_min_still_uses_loss()
    test_plateau_with_warmup_receives_val_mca()

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
