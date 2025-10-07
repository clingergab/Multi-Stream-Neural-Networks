"""
Analyze how many batches are needed for accurate stream monitoring.

Compare stream accuracies computed with different max_batches values.
"""

import torch
from src.models.multi_channel.mc_resnet import mc_resnet18
from src.data_utils.sunrgbd_dataset import SUNRGBDDataset
from torch.utils.data import DataLoader
from src.models.utils.stream_monitor import StreamMonitor

print("=" * 80)
print("ANALYZING MAX_BATCHES FOR STREAM MONITORING")
print("=" * 80)

# Create model
model = mc_resnet18(num_classes=15, stream1_input_channels=3, stream2_input_channels=1,
                    fusion_type='concat', device='cpu', use_amp=False)

model.compile(optimizer='adamw', learning_rate=1e-4, stream1_lr=2e-4, stream1_weight_decay=1e-2,
              stream2_lr=5e-5, stream2_weight_decay=2e-2, weight_decay=1e-3, loss='cross_entropy')

# Load full dataset
print("\nLoading full dataset...")
train_dataset = SUNRGBDDataset(data_root='data/sunrgbd_15', train=True)
val_dataset = SUNRGBDDataset(data_root='data/sunrgbd_15', train=False)

# Different batch sizes to test
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

print(f"\nDataset info:")
print(f"  Train samples: {len(train_dataset)}")
print(f"  Val samples: {len(val_dataset)}")
print(f"  Batch size: {batch_size}")
print(f"  Train batches: {len(train_loader)}")
print(f"  Val batches: {len(val_loader)}")

# Create monitor
monitor = StreamMonitor(model)

# Test different max_batches values
max_batches_to_test = [1, 2, 5, 10, 20, 50, None]  # None = all batches

print("\n" + "=" * 80)
print("TESTING DIFFERENT MAX_BATCHES VALUES")
print("=" * 80)

results = []

for max_batches in max_batches_to_test:
    if max_batches is None:
        max_batches_actual = len(train_loader)
        label = "ALL"
    else:
        max_batches_actual = max_batches
        label = str(max_batches)

    print(f"\nTesting max_batches={label}...")
    print(f"  Evaluating {max_batches_actual} batches = {max_batches_actual * batch_size} samples")

    stats = monitor.compute_stream_overfitting_indicators(
        train_loss=2.5,
        val_loss=2.8,
        train_acc=0.35,
        val_acc=0.30,
        train_loader=train_loader,
        val_loader=val_loader,
        max_batches=max_batches_actual
    )

    results.append({
        'max_batches': label,
        'samples': max_batches_actual * batch_size,
        'stream1_train_acc': stats['stream1_train_acc'],
        'stream1_val_acc': stats['stream1_val_acc'],
        'stream2_train_acc': stats['stream2_train_acc'],
        'stream2_val_acc': stats['stream2_val_acc']
    })

    print(f"  Stream1 - Train: {stats['stream1_train_acc']*100:.2f}%, Val: {stats['stream1_val_acc']*100:.2f}%")
    print(f"  Stream2 - Train: {stats['stream2_train_acc']*100:.2f}%, Val: {stats['stream2_val_acc']*100:.2f}%")

# Analysis
print("\n" + "=" * 80)
print("COMPARISON TABLE")
print("=" * 80)

print(f"\n{'Batches':<10} {'Samples':<10} {'S1_Train':<10} {'S1_Val':<10} {'S2_Train':<10} {'S2_Val':<10}")
print("-" * 80)

for r in results:
    print(f"{r['max_batches']:<10} {r['samples']:<10} "
          f"{r['stream1_train_acc']*100:>9.2f}% {r['stream1_val_acc']*100:>9.2f}% "
          f"{r['stream2_train_acc']*100:>9.2f}% {r['stream2_val_acc']*100:>9.2f}%")

# Calculate variance
print("\n" + "=" * 80)
print("STABILITY ANALYSIS")
print("=" * 80)

# Use ALL batches as ground truth
ground_truth = results[-1]

print(f"\nGround truth (ALL batches = {ground_truth['samples']} samples):")
print(f"  Stream1 - Train: {ground_truth['stream1_train_acc']*100:.2f}%, Val: {ground_truth['stream1_val_acc']*100:.2f}%")
print(f"  Stream2 - Train: {ground_truth['stream2_train_acc']*100:.2f}%, Val: {ground_truth['stream2_val_acc']*100:.2f}%")

print(f"\nAccuracy deviation from ground truth:")
print(f"{'Batches':<10} {'Samples':<10} {'S1_Train Δ':<12} {'S1_Val Δ':<12} {'S2_Train Δ':<12} {'S2_Val Δ':<12} {'Avg Δ':<12}")
print("-" * 95)

for r in results[:-1]:  # Exclude ground truth
    s1_train_delta = abs(r['stream1_train_acc'] - ground_truth['stream1_train_acc']) * 100
    s1_val_delta = abs(r['stream1_val_acc'] - ground_truth['stream1_val_acc']) * 100
    s2_train_delta = abs(r['stream2_train_acc'] - ground_truth['stream2_train_acc']) * 100
    s2_val_delta = abs(r['stream2_val_acc'] - ground_truth['stream2_val_acc']) * 100
    avg_delta = (s1_train_delta + s1_val_delta + s2_train_delta + s2_val_delta) / 4

    print(f"{r['max_batches']:<10} {r['samples']:<10} "
          f"{s1_train_delta:>10.2f}% {s1_val_delta:>10.2f}% "
          f"{s2_train_delta:>10.2f}% {s2_val_delta:>10.2f}% "
          f"{avg_delta:>10.2f}%")

print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)

print(f"\nCurrent setting: max_batches=5 ({5 * batch_size} samples)")
print(f"\nIs 5 batches enough?")
print(f"  - Depends on acceptable error tolerance")
print(f"  - If avg deviation < 2%: Good enough")
print(f"  - If avg deviation > 5%: Too noisy, increase max_batches")

print("\n" + "=" * 80)
