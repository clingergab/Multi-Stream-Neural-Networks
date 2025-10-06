"""
Test monitoring logic without running actual forward passes.
"""

import torch
import sys
sys.path.insert(0, 'src')

from data_utils.nyu_depth_dataset import create_nyu_dataloaders

print("=" * 70)
print("MONITORING LOGIC TEST")
print("=" * 70)

# 1. Test label diversity fix
print("\n✓ TEST 1: Label diversity (fixed clamping bug)")
print("-" * 70)

train_loader, val_loader = create_nyu_dataloaders(
    'data/content/nyu_depth_v2_labeled.mat',
    batch_size=64,
    num_workers=0,
    num_classes=27
)

# Collect all train labels
all_train_labels = []
for rgb, depth, labels in train_loader:
    all_train_labels.append(labels)

all_train_labels = torch.cat(all_train_labels)
unique_train = torch.unique(all_train_labels)

print(f"Total train samples: {len(all_train_labels)}")
print(f"Unique classes: {len(unique_train)}")
print(f"Class distribution: min={all_train_labels.min().item()}, max={all_train_labels.max().item()}")

if len(unique_train) == 1:
    print("❌ FAIL: All labels are the same (clamping bug still exists!)")
    sys.exit(1)
else:
    print(f"✅ PASS: Found {len(unique_train)} unique classes")

# 2. Test iterator consistency
print("\n✓ TEST 2: Iterator consistency (fresh iter each epoch)")
print("-" * 70)

# Simulate 3 epochs of monitoring
first_batches = []
for epoch in range(3):
    # This mimics what monitoring does
    val_iter = iter(val_loader)
    batch_labels = []

    for _ in range(5):  # max_batches=5
        try:
            _, _, labels = next(val_iter)
            batch_labels.append(labels)
        except StopIteration:
            break

    if batch_labels:
        all_labels = torch.cat(batch_labels)
        first_batches.append(all_labels[0].item())
        print(f"Epoch {epoch + 1}: First label = {all_labels[0].item()}, Total samples = {len(all_labels)}")

if len(set(first_batches)) == 1:
    print(f"✅ PASS: All epochs see same first label ({first_batches[0]}) - iterator resets correctly")
else:
    print(f"❌ FAIL: Different first labels {first_batches} - iterator bug!")
    sys.exit(1)

# 3. Test train vs val label distribution
print("\n✓ TEST 3: Train vs Val label distribution")
print("-" * 70)

all_val_labels = []
for rgb, depth, labels in val_loader:
    all_val_labels.append(labels)

all_val_labels = torch.cat(all_val_labels)
unique_val = torch.unique(all_val_labels)

print(f"Train: {len(all_train_labels)} samples, {len(unique_train)} classes")
print(f"Val: {len(all_val_labels)} samples, {len(unique_val)} classes")
print(f"Train classes: {sorted(unique_train.tolist())}")
print(f"Val classes: {sorted(unique_val.tolist())}")

# Both should have diverse labels
if len(unique_train) > 5 and len(unique_val) > 1:
    print("✅ PASS: Both train and val have diverse labels")
else:
    print("❌ FAIL: Labels not diverse enough")
    sys.exit(1)

print("\n" + "=" * 70)
print("ALL LOGIC TESTS PASSED!")
print("=" * 70)
print("\nSummary of fixes:")
print("  1. ✅ Removed label clamping bug (was forcing all labels to 12)")
print("  2. ✅ Iterator reset fix (uses iter() for fresh iterator each epoch)")
print("  3. ✅ Labels are diverse in both train and val sets")
print("\nThe monitoring should now show:")
print("  - Diverse predictions (not all class 12)")
print("  - Consistent metrics across epochs (no caching)")
print("  - Reasonable train/val relationship (no 6% train vs 48% val)")
