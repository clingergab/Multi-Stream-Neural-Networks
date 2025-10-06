"""
End-to-end training test with synthetic data.

Tests full training pipeline with fusion + stream optimization.
"""

import torch
from torch.utils.data import TensorDataset, DataLoader
from src.models.multi_channel import mc_resnet18

def create_synthetic_dataset(n_samples=100, num_classes=10, img_size=64):
    """Create synthetic dual-channel dataset."""
    stream1_data = torch.randn(n_samples, 3, img_size, img_size)
    stream2_data = torch.randn(n_samples, 1, img_size, img_size)
    targets = torch.randint(0, num_classes, (n_samples,))
    return stream1_data, stream2_data, targets

def test_end_to_end_training():
    """Test complete training pipeline."""

    print("=" * 70)
    print("End-to-End Training Test")
    print("=" * 70)

    # Configuration
    num_classes = 10
    img_size = 64
    batch_size = 16
    epochs = 3

    # Create synthetic dataset
    print("\n1. Creating Synthetic Dataset")
    print("-" * 70)
    train_stream1, train_stream2, train_targets = create_synthetic_dataset(200, num_classes, img_size)
    val_stream1, val_stream2, val_targets = create_synthetic_dataset(50, num_classes, img_size)

    print(f"✓ Train samples: {len(train_targets)}")
    print(f"✓ Val samples: {len(val_targets)}")
    print(f"✓ Stream1 shape: {train_stream1.shape}")
    print(f"✓ Stream2 shape: {train_stream2.shape}")

    # Create DataLoaders
    from src.data_utils.dual_channel_dataset import create_dual_channel_dataloaders

    train_loader, val_loader = create_dual_channel_dataloaders(
        train_stream1, train_stream2, train_targets,
        val_stream1, val_stream2, val_targets,
        batch_size=batch_size,
        num_workers=0
    )

    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")

    # Test each fusion type
    fusion_types = ['concat', 'weighted', 'gated']

    for fusion_type in fusion_types:
        print(f"\n{'=' * 70}")
        print(f"Testing: {fusion_type.upper()} Fusion Training")
        print(f"{'=' * 70}")

        # Create model
        model = mc_resnet18(
            num_classes=num_classes,
            stream1_input_channels=3,
            stream2_input_channels=1,
            fusion_type=fusion_type,
            dropout_p=0.2,
            device='cpu'
        )

        # Compile with stream-specific optimization
        model.compile(
            optimizer='adamw',
            learning_rate=1e-3,
            weight_decay=1e-2,
            stream1_lr=5e-3,           # Higher LR for stream1
            stream2_lr=1e-3,           # Lower LR for stream2
            stream1_weight_decay=1e-3,
            stream2_weight_decay=2e-2,
            scheduler='cosine',
            loss='cross_entropy'
        )

        print(f"\n2. Model Configuration")
        print("-" * 70)
        print(f"Fusion: {model.fusion_strategy}")
        print(f"Optimizer groups: {len(model.optimizer.param_groups)}")
        for i, pg in enumerate(model.optimizer.param_groups):
            print(f"  Group {i}: lr={pg['lr']:.0e}, wd={pg['weight_decay']:.0e}")

        # Train
        print(f"\n3. Training for {epochs} epochs")
        print("-" * 70)

        history = model.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            verbose=True
        )

        # Verify training metrics
        print(f"\n4. Training Results")
        print("-" * 70)
        print(f"✓ Final train loss: {history['train_loss'][-1]:.4f}")
        print(f"✓ Final train acc: {history['train_accuracy'][-1]:.4f}")
        print(f"✓ Final val loss: {history['val_loss'][-1]:.4f}")
        print(f"✓ Final val acc: {history['val_accuracy'][-1]:.4f}")

        # Check for improvement
        initial_train_acc = history['train_accuracy'][0]
        final_train_acc = history['train_accuracy'][-1]
        assert final_train_acc > initial_train_acc, "Training accuracy should improve"
        print(f"✓ Training improved: {initial_train_acc:.4f} → {final_train_acc:.4f}")

        # Test evaluation
        print(f"\n5. Evaluation")
        print("-" * 70)
        eval_results = model.evaluate(val_loader)
        print(f"✓ Eval loss: {eval_results['loss']:.4f}")
        print(f"✓ Eval accuracy: {eval_results['accuracy']:.4f}")

        # Test prediction
        print(f"\n6. Prediction")
        print("-" * 70)
        predictions = model.predict(val_loader)
        print(f"✓ Predictions shape: {predictions.shape}")
        print(f"✓ Sample predictions: {predictions[:5].tolist()}")

        # Test pathway analysis
        print(f"\n7. Pathway Analysis")
        print("-" * 70)
        analysis = model.analyze_pathways(val_loader, num_samples=50)
        print(f"✓ Full model acc: {analysis['accuracy']['full_model']:.4f}")
        print(f"✓ Stream1 only acc: {analysis['accuracy']['color_only']:.4f}")
        print(f"✓ Stream2 only acc: {analysis['accuracy']['brightness_only']:.4f}")
        print(f"✓ Stream1 contrib: {analysis['accuracy']['color_contribution']:.2%}")
        print(f"✓ Stream2 contrib: {analysis['accuracy']['brightness_contribution']:.2%}")

        print(f"\n✅ {fusion_type.upper()} Training: PASSED")

    print(f"\n{'=' * 70}")
    print("✅ ALL END-TO-END TRAINING TESTS PASSED!")
    print(f"{'=' * 70}")

    print("\nVerified:")
    print("  ✓ Dataset creation and loading")
    print("  ✓ Model compilation with stream-specific optimization")
    print("  ✓ Training loop execution")
    print("  ✓ Training metrics tracking")
    print("  ✓ Validation during training")
    print("  ✓ Evaluation on test set")
    print("  ✓ Prediction generation")
    print("  ✓ Pathway analysis")
    print("  ✓ All fusion types work correctly")
    print("\n🚀 Ready for real dataset training!")

if __name__ == "__main__":
    test_end_to_end_training()
