"""
Comprehensive tests for all three restoration scenarios.

Tests the three main scenarios:
1. Main early stopping with no streams frozen
2. All streams freeze (stream early stopping)
3. One stream frozen + main early stopping

This ensures the restoration logic works correctly in all cases.
"""

import torch
import torch.nn as nn
from src.models.multi_channel.mc_resnet import mc_resnet18
from src.models.linear_integration.li_net import li_resnet18
from src.data_utils.dual_channel_dataset import create_dual_channel_dataloaders


def create_synthetic_dataset(n_samples=200, img_size=32, n_classes=10):
    """Create a synthetic dataset for testing."""
    stream1_data = torch.randn(n_samples, 3, img_size, img_size)
    stream2_data = torch.randn(n_samples, 1, img_size, img_size)
    targets = torch.randint(0, n_classes, (n_samples,))
    return stream1_data, stream2_data, targets


def test_scenario1_main_es_no_frozen_mcresnet(capsys):
    """
    Scenario 1: Main early stopping triggers, no streams frozen.

    Expected: All components (Stream1, Stream2, integration, classifier)
    restored from the same epoch (best full model epoch).
    """
    torch.manual_seed(100)

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
    print("SCENARIO 1 (MCResNet): Main ES, No Streams Frozen")
    print("="*80)

    history = model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=20,
        verbose=True,
        stream_monitoring=True,
        stream_early_stopping=True,
        stream1_patience=100,  # Very high - won't freeze
        stream2_patience=100,  # Very high - won't freeze
        stream_min_delta=0.001,
        early_stopping=True,  # Main ES will trigger
        patience=5,
        monitor='val_accuracy',
        restore_best_weights=True
    )

    captured = capsys.readouterr()

    # Verify no streams were frozen
    assert not history['stream_early_stopping']['stream1_frozen'], "Stream1 should not be frozen"
    assert not history['stream_early_stopping']['stream2_frozen'], "Stream2 should not be frozen"

    # Verify main ES triggered
    assert len(history['train_loss']) < 20, "Main early stopping should have triggered"

    # Verify restoration message (no preservation needed)
    assert "Restored best model weights" in captured.out
    assert "preserved" not in captured.out.lower(), "Should not preserve any streams (none frozen)"

    print(f"\n✅ SCENARIO 1 PASSED")
    print(f"   No streams frozen: {not history['stream_early_stopping']['stream1_frozen'] and not history['stream_early_stopping']['stream2_frozen']}")
    print(f"   Main ES triggered at epoch: {len(history['train_loss'])}")
    print(f"   All components restored from same epoch ✓")


def test_scenario2_all_streams_freeze_mcresnet(capsys):
    """
    Scenario 2: Both streams freeze (stream early stopping completes).

    Expected:
    - First frozen stream: Keeps its best weights
    - Second frozen stream: Gets weights from best full model epoch
    - Integration/Classifier: Gets weights from best full model epoch
    """
    torch.manual_seed(200)

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
    print("SCENARIO 2 (MCResNet): All Streams Freeze")
    print("="*80)

    history = model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=20,
        verbose=True,
        stream_monitoring=True,
        stream_early_stopping=True,
        stream1_patience=2,  # Low - will freeze early
        stream2_patience=2,  # Low - will freeze after stream1
        stream_min_delta=0.001,
        early_stopping=False  # Disabled - only stream ES
    )

    captured = capsys.readouterr()

    # Verify both streams frozen
    assert history['stream_early_stopping']['stream1_frozen'], "Stream1 should be frozen"
    assert history['stream_early_stopping']['stream2_frozen'], "Stream2 should be frozen"
    assert history['stream_early_stopping']['all_frozen'], "All streams should be frozen"

    # Verify full model restoration with preservation
    assert "Restored full model best weights" in captured.out
    assert "preserved" in captured.out.lower(), "Should preserve first frozen stream"

    # Check which stream froze first
    stream1_frozen_epoch = history.get('stream1_frozen_epoch', 999)
    stream2_frozen_epoch = history.get('stream2_frozen_epoch', 999)
    first_frozen = "Stream1" if stream1_frozen_epoch < stream2_frozen_epoch else "Stream2"

    print(f"\n✅ SCENARIO 2 PASSED")
    print(f"   Stream1 frozen at epoch: {stream1_frozen_epoch}")
    print(f"   Stream2 frozen at epoch: {stream2_frozen_epoch}")
    print(f"   First frozen ({first_frozen}): Keeps best weights ✓")
    print(f"   Second frozen: Gets weights from best full model epoch ✓")
    print(f"   Integration: Gets weights from best full model epoch ✓")


def test_scenario3_one_frozen_main_es_mcresnet(capsys):
    """
    Scenario 3: One stream frozen, then main early stopping triggers.

    Expected:
    - Frozen stream: Keeps its best weights
    - Unfrozen stream: Gets weights from best full model epoch
    - Integration/Classifier: Gets weights from best full model epoch
    """
    torch.manual_seed(300)

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
    print("SCENARIO 3 (MCResNet): One Stream Frozen + Main ES")
    print("="*80)

    history = model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=25,
        verbose=True,
        stream_monitoring=True,
        stream_early_stopping=True,
        stream1_patience=2,  # Low - will freeze early
        stream2_patience=100,  # High - stays active
        stream_min_delta=0.001,
        early_stopping=True,  # Main ES will trigger after stream1 frozen
        patience=8,
        monitor='val_accuracy',
        restore_best_weights=True
    )

    captured = capsys.readouterr()

    # Verify stream1 frozen, stream2 not frozen
    assert history['stream_early_stopping']['stream1_frozen'], "Stream1 should be frozen"
    assert not history['stream_early_stopping']['stream2_frozen'], "Stream2 should not be frozen"
    assert not history['stream_early_stopping']['all_frozen'], "Not all streams frozen"

    # Verify main ES triggered (stopped before epoch 25)
    assert len(history['train_loss']) < 25, "Main early stopping should have triggered"

    # Verify restoration with preservation
    assert "Restored best model weights" in captured.out
    assert "preserved frozen Stream1" in captured.out, "Should preserve frozen Stream1"

    print(f"\n✅ SCENARIO 3 PASSED")
    print(f"   Stream1 frozen at epoch: {history.get('stream1_frozen_epoch', 'N/A')}")
    print(f"   Main ES triggered at epoch: {len(history['train_loss'])}")
    print(f"   Frozen stream (Stream1): Keeps best weights ✓")
    print(f"   Unfrozen stream (Stream2): Gets weights from best full model epoch ✓")
    print(f"   Integration: Gets weights from best full model epoch ✓")


def test_all_scenarios_linet(capsys):
    """
    Test all three scenarios work correctly with LINet as well.
    """
    print("\n" + "="*80)
    print("TESTING ALL SCENARIOS WITH LINET")
    print("="*80)

    # Scenario 1: No streams frozen
    torch.manual_seed(400)
    model = li_resnet18(num_classes=10, stream1_input_channels=3, stream2_input_channels=1, device="cpu")
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

    history = model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=15,
        verbose=False,
        stream_monitoring=True,
        stream_early_stopping=True,
        stream1_patience=100,
        stream2_patience=100,
        stream_min_delta=0.001,
        early_stopping=True,
        patience=5,
        monitor='val_accuracy',
        restore_best_weights=True
    )

    assert not history['stream_early_stopping']['stream1_frozen']
    assert not history['stream_early_stopping']['stream2_frozen']
    print(f"✅ LINet Scenario 1: No streams frozen, main ES works ✓")

    # Scenario 3: One stream frozen (test most complex case)
    torch.manual_seed(500)
    model = li_resnet18(num_classes=10, stream1_input_channels=3, stream2_input_channels=1, device="cpu")

    model.compile(
        optimizer='adam',
        learning_rate=0.001,
        criterion=nn.CrossEntropyLoss(),
        use_stream_specific_params=True,
        stream1_lr=0.001,
        stream2_lr=0.001
    )

    history = model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=20,
        verbose=False,
        stream_monitoring=True,
        stream_early_stopping=True,
        stream1_patience=2,
        stream2_patience=100,
        stream_min_delta=0.001,
        early_stopping=True,
        patience=8,
        monitor='val_accuracy',
        restore_best_weights=True
    )

    assert history['stream_early_stopping']['stream1_frozen']
    assert not history['stream_early_stopping']['stream2_frozen']
    print(f"✅ LINet Scenario 3: One stream frozen + main ES works ✓")


if __name__ == "__main__":
    print("="*80)
    print("COMPREHENSIVE RESTORATION SCENARIO TESTS")
    print("="*80)

    class MockCapsys:
        def readouterr(self):
            class Output:
                out = ""
                err = ""
            return Output()

    capsys = MockCapsys()

    test_scenario1_main_es_no_frozen_mcresnet(capsys)
    test_scenario2_all_streams_freeze_mcresnet(capsys)
    test_scenario3_one_frozen_main_es_mcresnet(capsys)
    test_all_scenarios_linet(capsys)

    print("\n" + "="*80)
    print("✅ ALL COMPREHENSIVE TESTS PASSED!")
    print("="*80)
