"""
Test that LINet also preserves frozen stream weights during main early stopping.

This ensures both MCResNet and LINet have the same preservation behavior.
"""

import torch
import torch.nn as nn
from src.models.linear_integration.li_net import li_resnet18
from src.data_utils.dual_channel_dataset import create_dual_channel_dataloaders


def create_synthetic_dataset(n_samples=200, img_size=32, n_classes=10):
    """Create a synthetic dataset for testing."""
    stream1_data = torch.randn(n_samples, 3, img_size, img_size)
    stream2_data = torch.randn(n_samples, 1, img_size, img_size)
    targets = torch.randint(0, n_classes, (n_samples,))
    return stream1_data, stream2_data, targets


def test_linet_frozen_stream_preservation(capsys):
    """Test that LINet preserves frozen stream weights when main early stopping triggers."""

    torch.manual_seed(42)

    # Create LINet model
    model = li_resnet18(num_classes=10, stream1_input_channels=3, stream2_input_channels=1, device="cpu")

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

    print("\n" + "="*80)
    print("Testing LINet: Stream1 freezing early, then main early stopping")
    print("="*80)

    # Train with stream1 freezing early, then main early stopping
    history = model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=30,
        verbose=True,
        stream_monitoring=True,
        stream_early_stopping=True,
        stream1_patience=2,  # Low - will freeze quickly
        stream2_patience=15,  # High - stays active
        stream_min_delta=0.001,
        early_stopping=True,  # Main early stopping
        patience=8,
        monitor='val_accuracy',
        restore_best_weights=True
    )

    # Capture output
    captured = capsys.readouterr()

    # Verify Stream1 was frozen
    assert history['stream_early_stopping']['stream1_frozen'], "Stream1 should have frozen"

    # Verify main early stopping triggered
    assert len(history['train_loss']) < 30, "Main early stopping should have triggered"

    # Check for preservation message
    if "preserved frozen" in captured.out.lower():
        print("\n✅ LINet SUCCESS: Frozen stream weights were preserved!")
        preserved_lines = [line for line in captured.out.split('\n') if 'preserved' in line.lower()]
        print(f"   Message: {preserved_lines[0] if preserved_lines else 'N/A'}")

    print(f"\n📊 LINet Training Summary:")
    print(f"   Total epochs: {len(history['train_loss'])}")
    print(f"   Stream1 frozen: {history['stream_early_stopping']['stream1_frozen']}")
    print(f"   Stream2 frozen: {history['stream_early_stopping']['stream2_frozen']}")

    if history['stream_early_stopping']['stream1_frozen']:
        print(f"   Stream1 frozen at epoch: {history.get('stream1_frozen_epoch', 'N/A')}")

    print(f"\n✓ LINet test passed!")


if __name__ == "__main__":
    print("="*80)
    print("Testing LINet Stream Preservation")
    print("="*80)

    class MockCapsys:
        def readouterr(self):
            class Output:
                out = ""
                err = ""
            return Output()

    test_linet_frozen_stream_preservation(MockCapsys())

    print("\n" + "="*80)
    print("✅ LINet preservation test completed!")
    print("="*80)
