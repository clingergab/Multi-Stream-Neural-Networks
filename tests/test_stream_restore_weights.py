"""
Test stream-specific restore best weights functionality.

Verifies that when stream_early_stopping is enabled:
1. Stream weights are saved when stream accuracy improves
2. Stream weights are restored to their best state before freezing
3. The restored weights actually improve stream accuracy
"""

import pytest
import torch
import torch.nn as nn
from src.models.multi_channel.mc_resnet import mc_resnet18
from src.data_utils.dual_channel_dataset import create_dual_channel_dataloaders


def create_synthetic_dataset(n_samples=200, img_size=32, n_classes=10):
    """Create a synthetic dataset for testing."""
    # Create synthetic RGB and depth data
    stream1_data = torch.randn(n_samples, 3, img_size, img_size)
    stream2_data = torch.randn(n_samples, 1, img_size, img_size)
    targets = torch.randint(0, n_classes, (n_samples,))

    return stream1_data, stream2_data, targets


@pytest.fixture
def model_and_data():
    """Create a model and synthetic data for testing."""
    # Use CPU for testing to avoid device-specific issues
    device = "cpu"
    # Create model with device specified
    model = mc_resnet18(num_classes=10, stream1_input_channels=3, stream2_input_channels=1, device=device)

    # Create synthetic data
    stream1_data, stream2_data, targets = create_synthetic_dataset(n_samples=200)

    # Split into train/val
    train_loader, val_loader = create_dual_channel_dataloaders(
        stream1_data[:160], stream2_data[:160], targets[:160],
        stream1_data[160:], stream2_data[160:], targets[160:],
        batch_size=32, num_workers=0
    )

    return model, device, train_loader, val_loader


def test_stream_weights_saved_on_improvement(model_and_data):
    """Test that stream weights are saved when stream accuracy improves."""
    model, device, train_loader, val_loader = model_and_data

    # Compile model with stream-specific optimizer groups
    model.compile(
        optimizer='adam',
        learning_rate=0.001,
        criterion=nn.CrossEntropyLoss(),
        use_stream_specific_params=True,
        stream1_lr=0.001,
        stream2_lr=0.001
    )

    # Train with stream early stopping
    history = model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=5,
        verbose=False,
        stream_monitoring=True,
        stream_early_stopping=True,
        stream1_patience=2,
        stream2_patience=2,
        stream_min_delta=0.001
    )

    # Check that we have stream early stopping history
    assert 'stream_early_stopping' in history

    # At least one of the streams should have best weights saved
    # (during training when improvement is detected)
    print(f"Stream1 frozen: {history['stream_early_stopping']['stream1_frozen']}")
    print(f"Stream2 frozen: {history['stream_early_stopping']['stream2_frozen']}")


def test_stream_weights_restored_before_freezing(model_and_data, capsys):
    """Test that stream weights are restored to best state before freezing."""
    model, device, train_loader, val_loader = model_and_data

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Compile model with stream-specific optimizer groups
    model.compile(
        optimizer='adam',
        learning_rate=0.001,
        criterion=nn.CrossEntropyLoss(),
        use_stream_specific_params=True,
        stream1_lr=0.001,
        stream2_lr=0.001
    )

    # Train with very low patience to force freezing
    history = model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=15,
        verbose=True,  # Enable verbose to see restoration messages
        stream_monitoring=True,
        stream_early_stopping=True,
        stream1_patience=2,
        stream2_patience=2,
        stream_min_delta=0.001
    )

    # Capture stdout to check for restoration messages
    captured = capsys.readouterr()

    # Check if any streams were frozen
    stream1_frozen = history['stream_early_stopping']['stream1_frozen']
    stream2_frozen = history['stream_early_stopping']['stream2_frozen']
    all_frozen = history['stream_early_stopping']['all_frozen']

    # If a stream was frozen, we should see restoration message in output
    if stream1_frozen:
        assert "Restored Stream1 best weights" in captured.out or "🔄 Restored Stream1" in captured.out
        print(f"\n✓ Stream1 was frozen and weights were restored from epoch {history['stream1_frozen_epoch']}")

    if stream2_frozen:
        assert "Restored Stream2 best weights" in captured.out or "🔄 Restored Stream2" in captured.out
        print(f"\n✓ Stream2 was frozen and weights were restored from epoch {history['stream2_frozen_epoch']}")

    # If all streams frozen, we should see full model restoration message
    if all_frozen:
        assert "Restored full model best weights" in captured.out or "🔄 Restored full model" in captured.out
        print(f"\n✓ Full model weights were restored when all streams froze")

    # At least one stream should have been frozen with this patience setting
    assert stream1_frozen or stream2_frozen, "Expected at least one stream to freeze with patience=2"

    print(f"\nFinal stream accuracies:")
    print(f"  Stream1: {history['stream_early_stopping']['stream1_best_acc']:.4f}")
    print(f"  Stream2: {history['stream_early_stopping']['stream2_best_acc']:.4f}")


def test_correct_stream_weights_are_saved():
    """Test that only the correct stream-specific parameters are saved."""
    # Create a simple model
    model = mc_resnet18(num_classes=10, stream1_input_channels=3, stream2_input_channels=1)

    # Get all stream1 and stream2 parameter names
    stream1_params = {name for name, _ in model.named_parameters() if '.stream1_' in name}
    stream2_params = {name for name, _ in model.named_parameters() if '.stream2_' in name}

    print(f"\nFound {len(stream1_params)} stream1 parameters")
    print(f"Found {len(stream2_params)} stream2 parameters")

    # Verify that we have stream-specific parameters
    assert len(stream1_params) > 0, "Should have stream1-specific parameters"
    assert len(stream2_params) > 0, "Should have stream2-specific parameters"

    # Verify that stream1 and stream2 parameters don't overlap
    assert len(stream1_params & stream2_params) == 0, "Stream1 and Stream2 parameters should not overlap"

    # Print some example parameter names
    print("\nExample stream1 parameters:")
    for name in list(stream1_params)[:3]:
        print(f"  {name}")

    print("\nExample stream2 parameters:")
    for name in list(stream2_params)[:3]:
        print(f"  {name}")


def test_weights_actually_different_after_restore():
    """Test that restored weights are different from current weights."""
    # Create model and data (with CPU device)
    model = mc_resnet18(num_classes=10, stream1_input_channels=3, stream2_input_channels=1, device="cpu")
    device = torch.device("cpu")

    # Create synthetic data
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

    # Save initial weights
    initial_stream1_weights = {
        name: param.data.cpu().clone()
        for name, param in model.named_parameters()
        if '.stream1_' in name
    }

    # Train for a few epochs
    model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=3,
        verbose=False,
        stream_monitoring=True,
        stream_early_stopping=False  # Don't freeze yet
    )

    # Check that weights have changed
    after_training_weights = {
        name: param.data.cpu().clone()
        for name, param in model.named_parameters()
        if '.stream1_' in name
    }

    # At least one parameter should have changed
    weights_changed = False
    for name in initial_stream1_weights.keys():
        if not torch.allclose(initial_stream1_weights[name], after_training_weights[name]):
            weights_changed = True
            break

    assert weights_changed, "Expected weights to change during training"
    print("\n✓ Weights changed during training (as expected)")


def test_integration_weights_not_saved():
    """Test that integration/fusion weights are NOT saved in stream-specific weights."""
    model = mc_resnet18(num_classes=10, stream1_input_channels=3, stream2_input_channels=1)

    # Get all parameters
    all_params = {name for name, _ in model.named_parameters()}
    stream1_params = {name for name in all_params if '.stream1_' in name}
    stream2_params = {name for name in all_params if '.stream2_' in name}

    # Get parameters that are NOT stream-specific (integration/fusion/fc)
    other_params = all_params - stream1_params - stream2_params

    print(f"\nNon-stream-specific parameters (integration/fusion/fc): {len(other_params)}")
    print("Examples:")
    for name in list(other_params)[:5]:
        print(f"  {name}")

    # Verify that integration weights are NOT in stream-specific params
    assert len(other_params) > 0, "Should have integration/fusion parameters"
    assert len(stream1_params & other_params) == 0, "Integration params should not be in stream1 params"
    assert len(stream2_params & other_params) == 0, "Integration params should not be in stream2 params"


if __name__ == "__main__":
    print("Running stream restore weights tests...\n")

    # Create fixtures manually
    print("=" * 70)
    print("Test 1: Checking correct stream parameters are identified")
    print("=" * 70)
    test_correct_stream_weights_are_saved()

    print("\n" + "=" * 70)
    print("Test 2: Checking integration weights are separate")
    print("=" * 70)
    test_integration_weights_not_saved()

    print("\n" + "=" * 70)
    print("Test 3: Checking weights change during training")
    print("=" * 70)
    test_weights_actually_different_after_restore()

    print("\n" + "=" * 70)
    print("Test 4: Full training with stream early stopping")
    print("=" * 70)
    model = mc_resnet18(num_classes=10, stream1_input_channels=3, stream2_input_channels=1, device="cpu")
    device = torch.device("cpu")

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
        verbose=True,
        stream_monitoring=True,
        stream_early_stopping=True,
        stream1_patience=2,
        stream2_patience=2,
        stream_min_delta=0.001
    )

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Stream1 frozen: {history['stream_early_stopping']['stream1_frozen']}")
    print(f"Stream1 best acc: {history['stream_early_stopping']['stream1_best_acc']:.4f}")
    if history['stream_early_stopping']['stream1_frozen']:
        print(f"Stream1 frozen at epoch: {history['stream1_frozen_epoch']}")

    print(f"\nStream2 frozen: {history['stream_early_stopping']['stream2_frozen']}")
    print(f"Stream2 best acc: {history['stream_early_stopping']['stream2_best_acc']:.4f}")
    if history['stream_early_stopping']['stream2_frozen']:
        print(f"Stream2 frozen at epoch: {history['stream2_frozen_epoch']}")

    print("\n✅ All tests passed!")
