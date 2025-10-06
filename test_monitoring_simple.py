"""
Simple test with synthetic data to verify monitoring logic.
"""

import torch
import torch.nn as nn
from src.models.multi_channel.mc_resnet import mc_resnet18
from src.models.utils.stream_monitor import StreamMonitor
from torch.utils.data import TensorDataset, DataLoader

print("=" * 80)
print("TESTING STREAM MONITORING WITH SYNTHETIC DATA")
print("=" * 80)

# Create synthetic data
print("\n1. Creating synthetic data...")
n_samples = 64
rgb_data = torch.randn(n_samples, 3, 224, 224)
depth_data = torch.randn(n_samples, 1, 224, 224)
labels = torch.randint(0, 27, (n_samples,))

dataset = TensorDataset(rgb_data, depth_data, labels)
train_loader = DataLoader(dataset, batch_size=16, shuffle=False)
val_loader = DataLoader(dataset, batch_size=16, shuffle=False)

print(f"   Created {n_samples} samples")
print(f"   Train/Val loaders: {len(train_loader)} batches each")

# Create model
print("\n2. Creating model...")
model = mc_resnet18(
    num_classes=27,
    stream1_input_channels=3,
    stream2_input_channels=1,
    fusion_type='weighted',
    dropout_p=0.5,
    device='cpu'
)
model.compile(optimizer='adamw', learning_rate=1e-4, weight_decay=2e-2)

print(f"   Fusion type: {model.fusion.__class__.__name__}")
if hasattr(model.fusion, 'stream1_weight'):
    print(f"   Initial stream1_weight: {torch.sigmoid(model.fusion.stream1_weight).item():.4f}")
    print(f"   Initial stream2_weight: {torch.sigmoid(model.fusion.stream2_weight).item():.4f}")

# Create monitor
print("\n3. Creating monitor...")
monitor = StreamMonitor(model)

# Test with one batch manually
print("\n4. Manual isolation test on first batch...")
rgb_batch, depth_batch, labels_batch = next(iter(train_loader))

model.eval()
with torch.no_grad():
    # Full model
    full_outputs = model(rgb_batch, depth_batch)
    full_acc = (full_outputs.argmax(1) == labels_batch).float().mean().item()
    print(f"   Full model acc: {full_acc*100:.2f}%")

    # Stream1 (RGB) only
    stream1_features = model._forward_stream1_pathway(rgb_batch)
    stream2_dummy = torch.zeros_like(stream1_features)

    print(f"\n   Stream1 features:")
    print(f"     Shape: {stream1_features.shape}")
    print(f"     Mean: {stream1_features.mean():.4f}, Std: {stream1_features.std():.4f}")

    # Apply fusion
    fused_s1 = model.fusion(stream1_features, stream2_dummy)
    print(f"   After fusion (RGB + zeros):")
    print(f"     Shape: {fused_s1.shape}")
    print(f"     Mean: {fused_s1.mean():.4f}, Std: {fused_s1.std():.4f}")

    # Check what weighted fusion does
    if hasattr(model.fusion, 'stream1_weight'):
        w1 = torch.sigmoid(model.fusion.stream1_weight).item()
        w2 = torch.sigmoid(model.fusion.stream2_weight).item()
        print(f"   Fusion applies: w1={w1:.4f} to RGB, w2={w2:.4f} to zeros")
        print(f"   Result: cat([RGB*{w1:.4f}, zeros*{w2:.4f}])")

        # Verify manually
        expected_first_half = stream1_features * w1
        expected_second_half = stream2_dummy * w2
        manual_fused = torch.cat([expected_first_half, expected_second_half], dim=1)
        matches = torch.allclose(fused_s1, manual_fused, atol=1e-5)
        print(f"   Manual fusion matches: {matches}")

    # Through dropout and FC
    fused_s1_dropout = model.dropout(fused_s1)
    outputs_s1 = model.fc(fused_s1_dropout)
    stream1_acc = (outputs_s1.argmax(1) == labels_batch).float().mean().item()
    print(f"   Stream1 isolated acc: {stream1_acc*100:.2f}%")

    # Stream2 (Depth) only
    stream2_features = model._forward_stream2_pathway(depth_batch)
    stream1_dummy = torch.zeros_like(stream2_features)

    print(f"\n   Stream2 features:")
    print(f"     Shape: {stream2_features.shape}")
    print(f"     Mean: {stream2_features.mean():.4f}, Std: {stream2_features.std():.4f}")

    fused_s2 = model.fusion(stream1_dummy, stream2_features)
    print(f"   After fusion (zeros + Depth):")
    print(f"     Shape: {fused_s2.shape}")
    print(f"     Mean: {fused_s2.mean():.4f}, Std: {fused_s2.std():.4f}")

    fused_s2_dropout = model.dropout(fused_s2)
    outputs_s2 = model.fc(fused_s2_dropout)
    stream2_acc = (outputs_s2.argmax(1) == labels_batch).float().mean().item()
    print(f"   Stream2 isolated acc: {stream2_acc*100:.2f}%")

# Now test monitoring function
print("\n5. Testing monitoring function...")
overfitting_stats = monitor.compute_stream_overfitting_indicators(
    train_loss=1.0,
    val_loss=1.0,
    train_acc=0.25,
    val_acc=0.25,
    train_loader=train_loader,
    val_loader=val_loader,
    max_batches=2
)

print(f"\n   Monitoring results (2 batches):")
print(f"   Stream1 train acc: {overfitting_stats['stream1_train_acc']*100:.2f}%")
print(f"   Stream1 val acc: {overfitting_stats['stream1_val_acc']*100:.2f}%")
print(f"   Stream2 train acc: {overfitting_stats['stream2_train_acc']*100:.2f}%")
print(f"   Stream2 val acc: {overfitting_stats['stream2_val_acc']*100:.2f}%")
print(f"   Stream1 overfitting score: {overfitting_stats['stream1_overfitting_score']:.4f}")
print(f"   Stream2 overfitting score: {overfitting_stats['stream2_overfitting_score']:.4f}")

print("\n" + "=" * 80)
print("ANALYSIS:")
print("=" * 80)
print("For an untrained model with random weights:")
print("  - All accuracies should be ~3.7% (1/27 classes)")
print("  - Isolated stream accuracies might be slightly different but close")
print("  - The monitoring function should produce similar results to manual test")
print("=" * 80)
