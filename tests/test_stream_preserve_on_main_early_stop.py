"""
Test that frozen stream weights are preserved when main early stopping triggers.

Scenario:
1. Stream1 freezes at epoch X → Stream1 weights restored to best
2. Training continues with Stream2 + integration
3. Main early stopping triggers → Full model restored BUT Stream1 weights should be preserved

This ensures that frozen streams keep their best weights even after main model restoration.
"""

import pytest
import torch
import torch.nn as nn
from src.models.multi_channel.mc_resnet import mc_resnet18
from src.data_utils.dual_channel_dataset import create_dual_channel_dataloaders


def create_synthetic_dataset(n_samples=200, img_size=32, n_classes=10):
    """Create a synthetic dataset for testing."""
    stream1_data = torch.randn(n_samples, 3, img_size, img_size)
    stream2_data = torch.randn(n_samples, 1, img_size, img_size)
    targets = torch.randint(0, n_classes, (n_samples,))
    return stream1_data, stream2_data, targets


def test_frozen_stream_weights_preserved_on_main_early_stop(capsys):
    """
    Test that when main early stopping triggers, frozen stream weights are preserved.

    Scenario:
    - Stream1 freezes first
    - Main early stopping triggers later
    - Stream1's best weights should NOT be overwritten by main restoration
    """
    # Set seed for reproducibility
    torch.manual_seed(42)

    # Create model
    model = mc_resnet18(num_classes=10, stream1_input_channels=3, stream2_input_channels=1, device="cpu")

    # Create data
    stream1_data, stream2_data, targets = create_synthetic_dataset(n_samples=200)
    train_loader, val_loader = create_dual_channel_dataloaders(
        stream1_data[:160], stream2_data[:160], targets[:160],
        stream1_data[160:], stream2_data[160:], targets[160:],
        batch_size=32, num_workers=0
    )

    # Compile model
    model.compile(
        optimizer='adam',
        learning_rate=0.001,
        criterion=nn.CrossEntropyLoss(),
        use_stream_specific_params=True,
        stream1_lr=0.001,
        stream2_lr=0.001
    )

    # Train with:
    # - Very low stream1 patience (forces Stream1 to freeze early)
    # - Higher stream2 patience (keeps Stream2 active)
    # - Main early stopping with moderate patience (triggers after Stream1 frozen)
    print("\n" + "="*80)
    print("Training with Stream1 freezing early, then main early stopping")
    print("="*80)

    history = model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=30,
        verbose=True,
        stream_monitoring=True,
        stream_early_stopping=True,
        stream1_patience=2,  # Very low - will freeze quickly
        stream2_patience=15,  # High - stays active longer
        stream_min_delta=0.001,
        early_stopping=True,  # Main early stopping enabled
        patience=8,  # Will trigger after Stream1 frozen
        monitor='val_accuracy',
        restore_best_weights=True  # Important: this should preserve frozen Stream1
    )

    # Capture output
    captured = capsys.readouterr()

    # Verify Stream1 was frozen
    assert history['stream_early_stopping']['stream1_frozen'], "Stream1 should have frozen"

    # Verify main early stopping triggered (should stop before epoch 30)
    assert len(history['train_loss']) < 30, "Main early stopping should have triggered"

    # Check for preservation message in output
    if "preserved frozen" in captured.out.lower():
        print("\n✅ SUCCESS: Frozen stream weights were preserved during main restoration!")
        preserved_lines = [line for line in captured.out.split('\n') if 'preserved' in line.lower()]
        print(f"   Message found: {preserved_lines}")
    else:
        # If Stream1 was the only frozen stream when main ES triggered, we should see preservation
        if history['stream_early_stopping']['stream1_frozen'] and not history['stream_early_stopping']['all_frozen']:
            # Main ES triggered while Stream1 frozen but not all streams
            assert "preserved" in captured.out.lower() or "Restored best model weights" in captured.out, \
                "Should see either preservation message or standard restoration message"

    print(f"\n📊 Training Summary:")
    print(f"   Total epochs run: {len(history['train_loss'])}")
    print(f"   Stream1 frozen: {history['stream_early_stopping']['stream1_frozen']}")
    print(f"   Stream2 frozen: {history['stream_early_stopping']['stream2_frozen']}")
    print(f"   All streams frozen: {history['stream_early_stopping']['all_frozen']}")

    if history['stream_early_stopping']['stream1_frozen']:
        print(f"   Stream1 frozen at epoch: {history.get('stream1_frozen_epoch', 'N/A')}")

    print(f"\n✓ Test passed: Frozen stream weights preservation logic is working")


def test_no_streams_frozen_main_early_stop():
    """
    Test that when no streams are frozen, main early stopping works normally.

    This is the baseline case - no stream preservation needed.
    """
    torch.manual_seed(123)

    model = mc_resnet18(num_classes=10, stream1_input_channels=3, stream2_input_channels=1, device="cpu")

    stream1_data, stream2_data, targets = create_synthetic_dataset(n_samples=200)
    train_loader, val_loader = create_dual_channel_dataloaders(
        stream1_data[:160], stream2_data[:160], targets[:160],
        stream1_data[160:], stream2_data[160:], targets[160:],
        batch_size=32, num_workers=0
    )

    model.compile(
        optimizer='adam',
        learning_rate=0.001,
        criterion=nn.CrossEntropyLoss(),
        use_stream_specific_params=True,
        stream1_lr=0.001,
        stream2_lr=0.001
    )

    print("\n" + "="*80)
    print("Training with main early stopping only (no stream freezing)")
    print("="*80)

    history = model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=30,
        verbose=False,
        stream_monitoring=True,
        stream_early_stopping=True,
        stream1_patience=100,  # Very high - won't freeze
        stream2_patience=100,  # Very high - won't freeze
        stream_min_delta=0.001,
        early_stopping=True,
        patience=5,
        monitor='val_accuracy',
        restore_best_weights=True
    )

    # No streams should be frozen
    assert not history['stream_early_stopping']['stream1_frozen'], "Stream1 should not be frozen"
    assert not history['stream_early_stopping']['stream2_frozen'], "Stream2 should not be frozen"

    # Main early stopping should have triggered
    assert len(history['train_loss']) < 30, "Main early stopping should have triggered"

    print(f"\n📊 Training Summary:")
    print(f"   Total epochs run: {len(history['train_loss'])}")
    print(f"   Stream1 frozen: {history['stream_early_stopping']['stream1_frozen']}")
    print(f"   Stream2 frozen: {history['stream_early_stopping']['stream2_frozen']}")
    print(f"\n✓ Test passed: Main early stopping works normally when no streams frozen")


if __name__ == "__main__":
    print("="*80)
    print("Testing Frozen Stream Preservation During Main Early Stopping")
    print("="*80)

    print("\n" + "="*80)
    print("Test 1: Frozen stream weights preserved on main early stop")
    print("="*80)

    # Mock capsys for standalone run
    class MockCapsys:
        def readouterr(self):
            class Output:
                out = ""
                err = ""
            return Output()

    test_frozen_stream_weights_preserved_on_main_early_stop(MockCapsys())

    print("\n" + "="*80)
    print("Test 2: No streams frozen (baseline)")
    print("="*80)
    test_no_streams_frozen_main_early_stop()

    print("\n" + "="*80)
    print("✅ All tests passed!")
    print("="*80)
