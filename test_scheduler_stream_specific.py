"""
Test if PyTorch schedulers properly update stream-specific learning rates.
"""

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

print("=" * 80)
print("TESTING SCHEDULER WITH STREAM-SPECIFIC LEARNING RATES")
print("=" * 80)

# Create dummy model
model = nn.Linear(10, 5)

# Create optimizer with MULTIPLE parameter groups (like stream-specific optimization)
param_groups = [
    {'params': [list(model.parameters())[0]], 'lr': 1e-3, 'weight_decay': 1e-2},  # Stream1
    {'params': [list(model.parameters())[1]], 'lr': 5e-4, 'weight_decay': 2e-2},  # Stream2
]

optimizer = torch.optim.AdamW(param_groups)

print("\nInitial learning rates:")
for i, group in enumerate(optimizer.param_groups):
    print(f"  Group {i}: LR = {group['lr']:.6f}")

# Create cosine scheduler
scheduler = CosineAnnealingLR(optimizer, T_max=100)

print(f"\n{'='*80}")
print("TESTING SCHEDULER UPDATES")
print(f"{'='*80}")

# Test scheduler for 10 steps
print("\nScheduler steps:")
for step in range(10):
    scheduler.step()

    print(f"\nStep {step + 1}:")
    for i, group in enumerate(optimizer.param_groups):
        print(f"  Group {i}: LR = {group['lr']:.6f}")

# Check if BOTH groups changed
initial_lr_0 = 1e-3
initial_lr_1 = 5e-4
final_lr_0 = optimizer.param_groups[0]['lr']
final_lr_1 = optimizer.param_groups[1]['lr']

print(f"\n{'='*80}")
print("ANALYSIS")
print(f"{'='*80}")

print(f"\nGroup 0 (Stream1):")
print(f"  Initial LR: {initial_lr_0:.6f}")
print(f"  Final LR:   {final_lr_0:.6f}")
print(f"  Change:     {(final_lr_0 - initial_lr_0) / initial_lr_0 * 100:.2f}%")

print(f"\nGroup 1 (Stream2):")
print(f"  Initial LR: {initial_lr_1:.6f}")
print(f"  Final LR:   {final_lr_1:.6f}")
print(f"  Change:     {(final_lr_1 - initial_lr_1) / initial_lr_1 * 100:.2f}%")

# Check if ratio is preserved
initial_ratio = initial_lr_0 / initial_lr_1
final_ratio = final_lr_0 / final_lr_1

print(f"\nLR Ratio (Group0 / Group1):")
print(f"  Initial: {initial_ratio:.4f}")
print(f"  Final:   {final_ratio:.4f}")
print(f"  Preserved: {'✅ YES' if abs(initial_ratio - final_ratio) < 0.01 else '❌ NO'}")

print(f"\n{'='*80}")
print("CONCLUSION")
print(f"{'='*80}")

if final_lr_0 < initial_lr_0 and final_lr_1 < initial_lr_1:
    print("\n✅ PyTorch scheduler DOES update all parameter groups!")
    print("   Both stream-specific LRs are being adjusted by the scheduler.")
    if abs(initial_ratio - final_ratio) < 0.01:
        print("   ✅ The relative ratio between streams is preserved.")
    else:
        print("   ⚠️  The relative ratio between streams changed slightly.")
else:
    print("\n❌ ERROR: Scheduler did not update all parameter groups!")

print(f"\n{'='*80}")
