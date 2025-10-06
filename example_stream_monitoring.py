"""
Example: Stream Monitoring During Training

Demonstrates how to monitor individual stream behavior during training
to detect overfitting, imbalanced learning, and make informed decisions.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
from src.models.multi_channel import mc_resnet18
from src.models.utils import StreamMonitor

def create_synthetic_data(n_samples=200, num_classes=10):
    """Create synthetic dual-channel dataset."""
    stream1 = torch.randn(n_samples, 3, 64, 64)
    stream2 = torch.randn(n_samples, 1, 64, 64)
    targets = torch.randint(0, num_classes, (n_samples,))
    return stream1, stream2, targets

def main():
    print("=" * 80)
    print("Stream Monitoring Example")
    print("=" * 80)

    # Create model
    model = mc_resnet18(
        num_classes=10,
        fusion_type='weighted',
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
        loss='cross_entropy'
    )

    # Create datasets
    train_s1, train_s2, train_t = create_synthetic_data(200, 10)
    val_s1, val_s2, val_t = create_synthetic_data(50, 10)

    from src.data_utils.dual_channel_dataset import create_dual_channel_dataloaders
    train_loader, val_loader = create_dual_channel_dataloaders(
        train_s1, train_s2, train_t,
        val_s1, val_s2, val_t,
        batch_size=16,
        num_workers=0
    )

    # Create stream monitor
    monitor = StreamMonitor(model)

    print("\n" + "=" * 80)
    print("Training with Stream Monitoring")
    print("=" * 80)

    epochs = 5

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 80)

        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (s1, s2, targets) in enumerate(train_loader):
            s1, s2, targets = s1.to(model.device), s2.to(model.device), targets.to(model.device)

            # Forward
            outputs = model(s1, s2)
            loss = model.criterion(outputs, targets)

            # Backward
            model.optimizer.zero_grad()
            loss.backward()

            # Monitor gradients BEFORE optimizer step
            if batch_idx == 0:  # Monitor first batch of each epoch
                grad_stats = monitor.compute_stream_gradients()
                print(f"\n📊 Gradient Statistics (Batch 0):")
                print(f"   Stream1 grad norm: {grad_stats['stream1_grad_norm']:.4f}")
                print(f"   Stream2 grad norm: {grad_stats['stream2_grad_norm']:.4f}")
                print(f"   Ratio (S1/S2): {grad_stats['stream1_to_stream2_ratio']:.2f}")

            model.optimizer.step()

            # Track metrics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for s1, s2, targets in val_loader:
                s1, s2, targets = s1.to(model.device), s2.to(model.device), targets.to(model.device)
                outputs = model(s1, s2)
                loss = model.criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total

        print(f"\n📈 Epoch Results:")
        print(f"   Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"   Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Compute stream-specific metrics
        print(f"\n🔍 Stream Analysis:")

        # Weight statistics
        weight_stats = monitor.compute_stream_weights()
        print(f"   Stream1 weight norm: {weight_stats['stream1_weight_norm']:.4f}")
        print(f"   Stream2 weight norm: {weight_stats['stream2_weight_norm']:.4f}")
        print(f"   Weight ratio (S1/S2): {weight_stats['weight_norm_ratio']:.2f}")

        # Overfitting indicators
        overfit_stats = monitor.compute_stream_overfitting_indicators(
            avg_train_loss, avg_val_loss, train_acc, val_acc,
            train_loader, val_loader
        )

        print(f"\n⚠️  Overfitting Indicators:")
        print(f"   Stream1 - Train: {overfit_stats['stream1_train_acc']:.3f}, Val: {overfit_stats['stream1_val_acc']:.3f}, Gap: {overfit_stats['stream1_acc_gap']:.3f}")
        print(f"   Stream2 - Train: {overfit_stats['stream2_train_acc']:.3f}, Val: {overfit_stats['stream2_val_acc']:.3f}, Gap: {overfit_stats['stream2_acc_gap']:.3f}")
        print(f"   Stream1 overfit score: {overfit_stats['stream1_overfitting_score']:.3f}")
        print(f"   Stream2 overfit score: {overfit_stats['stream2_overfitting_score']:.3f}")

        # Determine which stream is overfitting more
        if overfit_stats['stream1_overfitting_score'] > overfit_stats['stream2_overfitting_score'] * 1.5:
            print(f"   ⚠️  Stream1 is overfitting MORE than Stream2")
            print(f"   💡 Recommendation: Increase stream1_weight_decay or reduce stream1_lr")
        elif overfit_stats['stream2_overfitting_score'] > overfit_stats['stream1_overfitting_score'] * 1.5:
            print(f"   ⚠️  Stream2 is overfitting MORE than Stream1")
            print(f"   💡 Recommendation: Increase stream2_weight_decay or reduce stream2_lr")
        else:
            print(f"   ✓ Both streams have similar overfitting levels")

        # Log metrics
        all_metrics = {**grad_stats, **weight_stats, **overfit_stats}
        monitor.log_metrics(epoch, all_metrics)

        # Get recommendations
        if epoch > 1:  # After a few epochs
            print(f"\n💡 Recommendations:")
            recommendations = monitor.get_recommendations()
            for rec in recommendations:
                print(f"   {rec}")

    # Final summary
    print("\n" + "=" * 80)
    print("Training Complete - Final Analysis")
    print("=" * 80)

    print(monitor.get_summary())

    # Demonstrate pathway analysis
    print("\n" + "=" * 80)
    print("Detailed Pathway Analysis")
    print("=" * 80)

    analysis = model.analyze_pathways(val_loader, num_samples=50)
    print(f"\nAccuracy:")
    print(f"  Full model: {analysis['accuracy']['full_model']:.1%}")
    print(f"  Stream1 only: {analysis['accuracy']['color_only']:.1%}")
    print(f"  Stream2 only: {analysis['accuracy']['brightness_only']:.1%}")
    print(f"  Stream1 contribution: {analysis['accuracy']['color_contribution']:.1%}")
    print(f"  Stream2 contribution: {analysis['accuracy']['brightness_contribution']:.1%}")

    print(f"\nFeature Norms:")
    print(f"  Stream1 mean: {analysis['feature_norms']['color_mean']:.4f}")
    print(f"  Stream2 mean: {analysis['feature_norms']['brightness_mean']:.4f}")
    print(f"  Ratio (S1/S2): {analysis['feature_norms']['color_to_brightness_ratio']:.2f}")

    print("\n" + "=" * 80)
    print("Key Insights")
    print("=" * 80)

    print("\n1. Gradient Monitoring:")
    print("   - Track gradient norms per stream each batch")
    print("   - Detect if one stream has vanishing/exploding gradients")
    print("   - Adjust learning rates accordingly")

    print("\n2. Overfitting Detection:")
    print("   - Monitor train/val gap for each stream separately")
    print("   - Identify which stream needs more regularization")
    print("   - Adjust stream-specific weight decay")

    print("\n3. Weight Analysis:")
    print("   - Track weight norms to see if one stream dominates")
    print("   - Detect if weights are diverging or converging")

    print("\n4. Actionable Decisions:")
    print("   - If stream1 overfits more: ↑ stream1_weight_decay or ↓ stream1_lr")
    print("   - If stream2 gradients vanish: ↑ stream2_lr")
    print("   - If stream balance is off: adjust fusion strategy")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
