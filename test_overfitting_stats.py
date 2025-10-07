"""
Verify that overfitting_stats now contains proper full_loss_gap and full_acc_gap.
"""

import torch
from src.models.multi_channel.mc_resnet import mc_resnet18
from src.data_utils.sunrgbd_dataset import SUNRGBDDataset
from torch.utils.data import DataLoader
from src.models.utils.stream_monitor import StreamMonitor

print("=" * 80)
print("VERIFYING OVERFITTING STATS WITH REAL METRIC VALUES")
print("=" * 80)

# Create model
model = mc_resnet18(num_classes=15, stream1_input_channels=3, stream2_input_channels=1,
                    fusion_type='concat', device='cpu', use_amp=False)

model.compile(optimizer='adamw', learning_rate=1e-4, stream1_lr=2e-4, stream1_weight_decay=1e-2,
              stream2_lr=5e-5, stream2_weight_decay=2e-2, weight_decay=1e-3, loss='cross_entropy')

# Create small loaders
full_dataset = SUNRGBDDataset(data_root='data/sunrgbd_15', train=True)
small_dataset = torch.utils.data.Subset(full_dataset, indices=list(range(16)))
train_loader = DataLoader(small_dataset, batch_size=16, shuffle=False, num_workers=0)

full_val = SUNRGBDDataset(data_root='data/sunrgbd_15', train=False)
small_val = torch.utils.data.Subset(full_val, indices=list(range(16)))
val_loader = DataLoader(small_val, batch_size=16, shuffle=False, num_workers=0)

# Create monitor
monitor = StreamMonitor(model)

# Compute overfitting stats with REAL metric values
print("\nComputing overfitting_stats with real metric values:")
print("  train_loss=2.5, val_loss=2.8")
print("  train_acc=0.35, val_acc=0.30")

overfitting_stats = monitor.compute_stream_overfitting_indicators(
    train_loss=2.5,
    val_loss=2.8,
    train_acc=0.35,
    val_acc=0.30,
    train_loader=train_loader,
    val_loader=val_loader,
    max_batches=1
)

print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

# Check full_model_loss_gap and full_model_acc_gap (correct key names)
print(f"\n✅ full_model_loss_gap: {overfitting_stats.get('full_model_loss_gap', 'MISSING')}")
print(f"   Expected: 2.8 - 2.5 = 0.3")
print(f"   Actual: {overfitting_stats.get('full_model_loss_gap', 'MISSING')}")

print(f"\n✅ full_model_acc_gap: {overfitting_stats.get('full_model_acc_gap', 'MISSING')}")
print(f"   Expected: 0.35 - 0.30 = 0.05")
print(f"   Actual: {overfitting_stats.get('full_model_acc_gap', 'MISSING')}")

print(f"\n✅ stream1_train_acc: {overfitting_stats['stream1_train_acc']:.4f}")
print(f"✅ stream1_val_acc: {overfitting_stats['stream1_val_acc']:.4f}")
print(f"✅ stream2_train_acc: {overfitting_stats['stream2_train_acc']:.4f}")
print(f"✅ stream2_val_acc: {overfitting_stats['stream2_val_acc']:.4f}")

# Verify the values are correct
if 'full_model_loss_gap' in overfitting_stats:
    expected_loss_gap = 2.8 - 2.5
    actual_loss_gap = overfitting_stats['full_model_loss_gap']
    if abs(actual_loss_gap - expected_loss_gap) < 0.01:
        print(f"\n✅ SUCCESS: full_model_loss_gap is correct!")
    else:
        print(f"\n❌ ERROR: full_model_loss_gap mismatch!")
else:
    print(f"\n❌ ERROR: full_model_loss_gap not in overfitting_stats!")

if 'full_model_acc_gap' in overfitting_stats:
    expected_acc_gap = 0.35 - 0.30
    actual_acc_gap = overfitting_stats['full_model_acc_gap']
    if abs(actual_acc_gap - expected_acc_gap) < 0.01:
        print(f"✅ SUCCESS: full_model_acc_gap is correct!")
    else:
        print(f"❌ ERROR: full_model_acc_gap mismatch!")
else:
    print(f"❌ ERROR: full_model_acc_gap not in overfitting_stats!")

print("\n" + "=" * 80)
