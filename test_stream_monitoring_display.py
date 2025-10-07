"""
Test that stream_monitoring parameter displays stream-specific info during training.
"""

import torch
from src.models.multi_channel.mc_resnet import mc_resnet18
from src.data_utils.sunrgbd_dataset import SUNRGBDDataset
from torch.utils.data import DataLoader

print("=" * 80)
print("TESTING STREAM MONITORING DISPLAY")
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

# Compile with stream-specific LRs
print("\n2. Compiling with stream-specific LRs...")
model.compile(
    optimizer='adamw',
    learning_rate=1e-4,       # Base LR
    stream1_lr=2e-4,          # RGB: 2x base
    stream1_weight_decay=1e-2,
    stream2_lr=5e-5,          # Depth: 0.5x base
    stream2_weight_decay=2e-2,
    weight_decay=1e-3,
    loss='cross_entropy',
    scheduler='cosine'
)

print(f"\n3. Parameter groups: {len(model.optimizer.param_groups)}")
for i, group in enumerate(model.optimizer.param_groups):
    print(f"   Group {i}: LR={group['lr']:.6f}, WD={group['weight_decay']:.6f}")

# Load small dataset
print("\n4. Loading small dataset...")
dataset = SUNRGBDDataset(data_root='data/sunrgbd_15', train=True)
train_loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

val_dataset = SUNRGBDDataset(data_root='data/sunrgbd_15', train=False)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

# Train for 2 epochs with stream monitoring ENABLED
print("\n5. Training with stream_monitoring=True...")
print("=" * 80)

history = model.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=2,
    verbose=True,
    stream_monitoring=True,  # ← Enable stream monitoring!
    grad_clip_norm=5.0
)

print("\n" + "=" * 80)
print("EXPECTED OUTPUT:")
print("=" * 80)
print("After each epoch progress bar, you should see:")
print("  Stream Monitoring:")
print("    RGB   - LR: 0.000200, WD: 0.010000")
print("    Depth - LR: 0.000050, WD: 0.020000")
print("    Fuse  - LR: 0.000100, WD: 0.001000")
print("\n✅ Test complete!")
print("=" * 80)
