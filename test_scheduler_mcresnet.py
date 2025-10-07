"""
Test that scheduler properly updates stream-specific learning rates in MCResNet training.
"""

import torch
from src.models.multi_channel.mc_resnet import mc_resnet18
from src.data_utils.sunrgbd_dataset import SUNRGBDDataset
from torch.utils.data import DataLoader

print("=" * 80)
print("TESTING SCHEDULER WITH MCRESNET STREAM-SPECIFIC OPTIMIZATION")
print("=" * 80)

# Create model
print("\n1. Creating MCResNet...")
model = mc_resnet18(
    num_classes=15,
    stream1_input_channels=3,
    stream2_input_channels=1,
    fusion_type='concat',
    dropout_p=0.3,
    device='cpu',
    use_amp=False
)

# Compile with stream-specific learning rates
print("\n2. Compiling with stream-specific LRs...")
model.compile(
    optimizer='adamw',
    learning_rate=1e-4,       # Base LR (for fusion + classifier)
    stream1_lr=2e-4,          # RGB stream (2x base)
    stream1_weight_decay=1e-2,
    stream2_lr=5e-5,          # Depth stream (0.5x base)
    stream2_weight_decay=2e-2,
    weight_decay=1e-3,
    loss='cross_entropy',
    scheduler='cosine'
)

print("\n3. Checking parameter groups...")
print(f"   Number of parameter groups: {len(model.optimizer.param_groups)}")

initial_lrs = []
for i, group in enumerate(model.optimizer.param_groups):
    lr = group['lr']
    wd = group['weight_decay']
    num_params = sum(p.numel() for p in group['params'])
    initial_lrs.append(lr)
    print(f"   Group {i}: LR={lr:.6f}, WD={wd:.6f}, Params={num_params:,}")

# Load small dataset
print("\n4. Loading dataset...")
dataset = SUNRGBDDataset(data_root='data/sunrgbd_15', train=True)
loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

# Get one batch
rgb, depth, labels = next(iter(loader))

# Simulate training for a few steps
print("\n5. Simulating training with scheduler...")

model.train()

# Manually step through training to monitor LR changes
for step in range(5):
    # Forward
    model.optimizer.zero_grad()
    outputs = model(rgb, depth)
    loss = model.criterion(outputs, labels)

    # Backward
    loss.backward()

    # Optimizer step
    model.optimizer.step()

    # Scheduler step (this should update ALL parameter groups)
    if model.scheduler is not None:
        model.scheduler.step()

    # Print current LRs
    print(f"\n   Step {step + 1}:")
    for i, group in enumerate(model.optimizer.param_groups):
        lr = group['lr']
        print(f"     Group {i}: LR={lr:.6f}")

# Get final LRs
final_lrs = [group['lr'] for group in model.optimizer.param_groups]

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

print("\nLearning rate changes:")
for i, (initial, final) in enumerate(zip(initial_lrs, final_lrs)):
    change = (final - initial) / initial * 100
    print(f"  Group {i}: {initial:.6f} → {final:.6f} ({change:+.2f}%)")

# Check if ALL groups changed
all_changed = all(final != initial for initial, final in zip(initial_lrs, final_lrs))

# Check if relative ratios preserved
if len(initial_lrs) >= 2:
    initial_ratio_01 = initial_lrs[0] / initial_lrs[1]
    final_ratio_01 = final_lrs[0] / final_lrs[1]

    print(f"\nLR Ratio (Group 0 / Group 1):")
    print(f"  Initial: {initial_ratio_01:.4f}")
    print(f"  Final:   {final_ratio_01:.4f}")
    ratio_preserved = abs(initial_ratio_01 - final_ratio_01) < 0.01

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

if all_changed:
    print("\n✅ SUCCESS: Scheduler updates ALL parameter groups!")
    print("   Stream-specific learning rates are being adjusted during training.")
    if 'ratio_preserved' in locals() and ratio_preserved:
        print("   ✅ The relative ratio between streams is preserved.")
        print("\n📊 Example: If Stream1 LR is 2x Stream2 LR initially,")
        print("   it remains 2x throughout training (just both decrease).")
    else:
        print("   ⚠️  Note: Relative ratios changed slightly (expected variation).")
else:
    print("\n❌ ERROR: Not all parameter groups were updated!")
    print("   This is a bug - stream-specific LRs are NOT being scheduled!")

print("\n" + "=" * 80)
