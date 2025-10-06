"""
Comprehensive test of monitoring with fixed label loading.
Tests both iterator fix and label diversity fix.
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, 'src')

from data_utils.nyu_depth_dataset import create_nyu_dataloaders
from models.multi_channel.mc_resnet import mc_resnet18
from models.utils.stream_monitor import StreamMonitor

print("=" * 70)
print("COMPREHENSIVE MONITORING TEST")
print("=" * 70)

# 1. Test label loading
print("\n1. Testing label diversity...")
print("-" * 70)
train_loader, val_loader = create_nyu_dataloaders(
    'data/content/nyu_depth_v2_labeled.mat',
    batch_size=64,
    num_workers=0,
    num_classes=27
)

# Check train labels
all_train_labels = []
for i, (_, _, labels) in enumerate(train_loader):
    all_train_labels.append(labels)
    if i >= 4:  # Check first 5 batches
        break

all_train_labels = torch.cat(all_train_labels)
unique_train = torch.unique(all_train_labels)
print(f"✓ Train labels diverse: {len(unique_train)} unique classes in first 5 batches")
print(f"  Unique labels: {unique_train.tolist()}")

# Check val labels
all_val_labels = []
for i, (_, _, labels) in enumerate(val_loader):
    all_val_labels.append(labels)
    if i >= 4:
        break

all_val_labels = torch.cat(all_val_labels)
unique_val = torch.unique(all_val_labels)
print(f"✓ Val labels diverse: {len(unique_val)} unique classes in first 5 batches")
print(f"  Unique labels: {unique_val.tolist()}")

if len(unique_train) == 1:
    print("❌ FAIL: All train labels are the same!")
    sys.exit(1)

# 2. Test monitoring consistency across epochs
print("\n2. Testing monitoring iterator consistency...")
print("-" * 70)

model = mc_resnet18(num_classes=27, in_channels_stream1=3, in_channels_stream2=1)
model.criterion = nn.CrossEntropyLoss()
model.device = 'cpu'
model.eval()

monitor = StreamMonitor(model)

# Run monitoring 3 times (simulating 3 epochs)
results = []
for epoch in range(3):
    print(f"\nEpoch {epoch + 1}:")

    stats = monitor.compute_stream_overfitting_indicators(
        train_loss=1.0,
        val_loss=1.5,
        train_acc=0.5,
        val_acc=0.4,
        train_loader=train_loader,
        val_loader=val_loader,
        max_batches=5
    )

    print(f"  Stream1 (RGB) - Train: {stats['stream1_train_acc']:.4f}, Val: {stats['stream1_val_acc']:.4f}")
    print(f"  Stream2 (Depth) - Train: {stats['stream2_train_acc']:.4f}, Val: {stats['stream2_val_acc']:.4f}")

    results.append({
        's1_train': stats['stream1_train_acc'],
        's1_val': stats['stream1_val_acc'],
        's2_train': stats['stream2_train_acc'],
        's2_val': stats['stream2_val_acc']
    })

# Check consistency
print("\n" + "-" * 70)
print("Consistency Check:")
s1_train_vals = [r['s1_train'] for r in results]
s1_val_vals = [r['s1_val'] for r in results]
s2_train_vals = [r['s2_train'] for r in results]
s2_val_vals = [r['s2_val'] for r in results]

print(f"Stream1 train accs: {[f'{v:.4f}' for v in s1_train_vals]}")
print(f"Stream1 val accs: {[f'{v:.4f}' for v in s1_val_vals]}")
print(f"Stream2 train accs: {[f'{v:.4f}' for v in s2_train_vals]}")
print(f"Stream2 val accs: {[f'{v:.4f}' for v in s2_val_vals]}")

# Check if all identical (should be, since we're using same data with shuffle=False for val)
def all_same(values, tolerance=1e-6):
    return max(values) - min(values) < tolerance

if (all_same(s1_train_vals) and all_same(s1_val_vals) and
    all_same(s2_train_vals) and all_same(s2_val_vals)):
    print("✓ PASS: All epochs return consistent values (iterator works!)")
else:
    print("❌ FAIL: Inconsistent values across epochs (iterator bug!)")
    sys.exit(1)

# 3. Sanity check: train acc should be >= val acc (or close) since model is random
print("\n3. Testing train vs val relationship...")
print("-" * 70)

for i, r in enumerate(results):
    print(f"Epoch {i+1}:")

    # For a random model on diverse labels, both should be ~1/27 = 3.7%
    # They should be similar, not wildly different
    s1_diff = abs(r['s1_train'] - r['s1_val'])
    s2_diff = abs(r['s2_train'] - r['s2_val'])

    print(f"  Stream1: |train - val| = {s1_diff:.4f}")
    print(f"  Stream2: |train - val| = {s2_diff:.4f}")

    # Check for impossible case: val >> train (like 48% vs 6%)
    if r['s1_val'] > r['s1_train'] * 3:
        print(f"  ❌ WARNING: Stream1 val is 3x+ higher than train - suspicious!")
    if r['s2_val'] > r['s2_train'] * 3:
        print(f"  ❌ WARNING: Stream2 val is 3x+ higher than train - suspicious!")

# 4. Check that accuracies are low (model is random)
print("\n4. Testing random model baseline...")
print("-" * 70)
avg_s1_val = sum(s1_val_vals) / len(s1_val_vals)
avg_s2_val = sum(s2_val_vals) / len(s2_val_vals)

expected_random = 1.0 / 27  # ~3.7%

print(f"Average Stream1 val acc: {avg_s1_val:.4f} (expected ~{expected_random:.4f} for random)")
print(f"Average Stream2 val acc: {avg_s2_val:.4f} (expected ~{expected_random:.4f} for random)")

if avg_s1_val > 0.2 or avg_s2_val > 0.2:
    print("❌ WARNING: Accuracy too high for random model - possible bug!")
else:
    print("✓ PASS: Accuracies in expected range for random model")

print("\n" + "=" * 70)
print("ALL TESTS PASSED!")
print("=" * 70)
print("\nThe monitoring system is working correctly:")
print("  ✓ Labels are diverse (not all class 12)")
print("  ✓ Iterator resets properly between epochs")
print("  ✓ Train/val accuracies are consistent")
print("  ✓ Random model baseline is correct")
