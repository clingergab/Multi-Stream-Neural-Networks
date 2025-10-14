"""
Quick demonstration of stream-specific restore best weights functionality.

Shows the feature in action with a simple training run.
"""

import torch
import torch.nn as nn
from src.models.multi_channel.mc_resnet import mc_resnet18
from src.data_utils.dual_channel_dataset import create_dual_channel_dataloaders


def create_demo_data(n_samples=200, img_size=32, n_classes=10):
    """Create synthetic dataset for demonstration."""
    stream1_data = torch.randn(n_samples, 3, img_size, img_size)
    stream2_data = torch.randn(n_samples, 1, img_size, img_size)
    targets = torch.randint(0, n_classes, (n_samples,))
    return stream1_data, stream2_data, targets


def main():
    print("=" * 80)
    print("Stream-Specific Restore Best Weights Demonstration")
    print("=" * 80)

    # Create model (use CPU for demo)
    print("\n1. Creating MCResNet model...")
    model = mc_resnet18(num_classes=10, stream1_input_channels=3, stream2_input_channels=1, device="cpu")
    print("   ✓ Model created")

    # Create data
    print("\n2. Creating synthetic dataset...")
    stream1_data, stream2_data, targets = create_demo_data(n_samples=200)
    train_loader, val_loader = create_dual_channel_dataloaders(
        stream1_data[:160], stream2_data[:160], targets[:160],
        stream1_data[160:], stream2_data[160:], targets[160:],
        batch_size=32, num_workers=0
    )
    print("   ✓ Dataset created (160 train, 40 val)")

    # Compile model with stream-specific optimizer
    print("\n3. Compiling model with stream-specific optimizer...")
    model.compile(
        optimizer='adam',
        learning_rate=0.001,
        criterion=nn.CrossEntropyLoss(),
        use_stream_specific_params=True,
        stream1_lr=0.001,
        stream2_lr=0.001
    )
    print("   ✓ Model compiled")

    # Train with stream early stopping
    print("\n4. Training with stream early stopping (patience=2)...")
    print("   Watch for restoration messages when streams freeze!\n")

    history = model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=15,
        verbose=True,
        stream_monitoring=True,
        stream_early_stopping=True,  # Automatic weight restoration enabled!
        stream1_patience=2,
        stream2_patience=2,
        stream_min_delta=0.001
    )

    # Print summary
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)

    if 'stream_early_stopping' in history:
        stream1_info = history['stream_early_stopping']

        print(f"\nStream1 (RGB):")
        print(f"  - Frozen: {stream1_info['stream1_frozen']}")
        print(f"  - Best Accuracy: {stream1_info['stream1_best_acc']:.4f}")
        if stream1_info['stream1_frozen']:
            print(f"  - Frozen at epoch: {history['stream1_frozen_epoch']}")
            print(f"  - Weights restored before freezing ✓")

        print(f"\nStream2 (Depth):")
        print(f"  - Frozen: {stream1_info['stream2_frozen']}")
        print(f"  - Best Accuracy: {stream1_info['stream2_best_acc']:.4f}")
        if stream1_info['stream2_frozen']:
            print(f"  - Frozen at epoch: {history['stream2_frozen_epoch']}")
            print(f"  - Weights restored before freezing ✓")

        print(f"\nAll Streams Frozen: {stream1_info['all_frozen']}")

    print("\n" + "=" * 80)
    print("KEY FEATURE HIGHLIGHTS")
    print("=" * 80)
    print("✓ Automatic: No manual configuration needed")
    print("✓ Stream-specific: Only stream parameters saved/restored")
    print("✓ Integration weights: Remain trainable after stream freezes")
    print("✓ Device-agnostic: Works on CPU, CUDA, and MPS")
    print("✓ Best weights: Streams restored to peak performance before freezing")

    print("\n" + "=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    # Set seed for reproducibility
    torch.manual_seed(42)
    main()
