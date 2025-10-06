"""
Test that the iterator fix prevents caching issues between monitoring calls.
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import sys
sys.path.insert(0, 'src')

from models.multi_channel.mc_resnet import mc_resnet18
from models.utils.stream_monitor import StreamMonitor

# Create dummy dataset with unique samples
num_samples = 500
torch.manual_seed(42)
rgb_data = torch.randn(num_samples, 3, 224, 224)
depth_data = torch.randn(num_samples, 1, 224, 224)
labels = torch.randint(0, 27, (num_samples,))

dataset = TensorDataset(rgb_data, depth_data, labels)
loader = DataLoader(dataset, batch_size=64, shuffle=False)

# Create model
model = mc_resnet18(num_classes=27, in_channels_stream1=3, in_channels_stream2=1)
model.criterion = nn.CrossEntropyLoss()
model.device = 'cpu'
model.eval()

# Create monitor
monitor = StreamMonitor(model)

print("Testing iterator fix...")
print("=" * 70)

# Simulate multiple epochs of monitoring (like training does)
# Each call should get the SAME first 5 batches, not cached/stale ones
results = []

for epoch in range(3):
    print(f"\nEpoch {epoch + 1}:")

    # Manually compute what monitoring should see
    with torch.no_grad():
        val_iter = iter(loader)
        batch_labels = []
        for i in range(5):  # max_batches=5
            try:
                _, _, batch_targets = next(val_iter)
                batch_labels.append(batch_targets)
            except StopIteration:
                break

        all_labels = torch.cat(batch_labels)
        print(f"  First label in first batch: {all_labels[0].item()}")
        print(f"  Total samples across 5 batches: {len(all_labels)}")
        print(f"  Mean label value: {all_labels.float().mean().item():.2f}")

        results.append({
            'first_label': all_labels[0].item(),
            'total_samples': len(all_labels),
            'mean_label': all_labels.float().mean().item()
        })

print("\n" + "=" * 70)
print("Consistency Check:")
print("=" * 70)

# All epochs should see the SAME data (first 5 batches)
first_labels = [r['first_label'] for r in results]
total_samples = [r['total_samples'] for r in results]
mean_labels = [r['mean_label'] for r in results]

print(f"First labels across epochs: {first_labels}")
print(f"Total samples across epochs: {total_samples}")
print(f"Mean labels across epochs: {[f'{m:.2f}' for m in mean_labels]}")

if len(set(first_labels)) == 1 and len(set(total_samples)) == 1:
    print("\n✓ PASS: All epochs see the same data (iterator resets correctly)")
else:
    print("\n✗ FAIL: Different data across epochs (iterator not resetting)")

print("\nNow testing actual monitoring function...")
print("=" * 70)

# Test actual monitoring function
monitoring_results = []
for epoch in range(3):
    print(f"\nEpoch {epoch + 1} - Calling monitor.compute_stream_overfitting_indicators():")

    stats = monitor.compute_stream_overfitting_indicators(
        train_loss=1.0,
        val_loss=1.5,
        train_acc=0.5,
        val_acc=0.3,
        train_loader=loader,
        val_loader=loader,
        max_batches=5
    )

    print(f"  Stream1 val acc: {stats['stream1_val_acc']:.4f}")
    print(f"  Stream2 val acc: {stats['stream2_val_acc']:.4f}")

    monitoring_results.append({
        's1_acc': stats['stream1_val_acc'],
        's2_acc': stats['stream2_val_acc']
    })

print("\n" + "=" * 70)
print("Monitoring Function Consistency Check:")
print("=" * 70)

s1_accs = [r['s1_acc'] for r in monitoring_results]
s2_accs = [r['s2_acc'] for r in monitoring_results]

print(f"Stream1 val acc across epochs: {[f'{a:.4f}' for a in s1_accs]}")
print(f"Stream2 val acc across epochs: {[f'{a:.4f}' for a in s2_accs]}")

# Check if values are identical (they should be, since we're using same data)
s1_identical = len(set([round(a, 6) for a in s1_accs])) == 1
s2_identical = len(set([round(a, 6) for a in s2_accs])) == 1

if s1_identical and s2_identical:
    print("\n✓ PASS: Monitoring returns consistent values across epochs")
else:
    print("\n✗ FAIL: Monitoring returns different values (BUG STILL EXISTS)")
    print("  This means the iterator is not being properly reset!")
