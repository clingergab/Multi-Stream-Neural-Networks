"""
Test full stream monitoring with per-stream accuracies.
"""

import torch
from src.models.multi_channel.mc_resnet import mc_resnet18
from src.data_utils.sunrgbd_dataset import SUNRGBDDataset
from torch.utils.data import DataLoader

print("=" * 80)
print("TESTING STREAM MONITORING WITH PER-STREAM ACCURACIES")
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
    learning_rate=1e-4,
    stream1_lr=2e-4,
    stream1_weight_decay=1e-2,
    stream2_lr=5e-5,
    stream2_weight_decay=2e-2,
    weight_decay=1e-3,
    loss='cross_entropy',
    scheduler='cosine'
)

# Load small dataset (just a few samples for quick test)
print("\n3. Loading small subset...")
full_dataset = SUNRGBDDataset(data_root='data/sunrgbd_15', train=True)
small_dataset = torch.utils.data.Subset(full_dataset, indices=list(range(32)))  # Just 32 samples
train_loader = DataLoader(small_dataset, batch_size=16, shuffle=False, num_workers=0)

full_val_dataset = SUNRGBDDataset(data_root='data/sunrgbd_15', train=False)
small_val_dataset = torch.utils.data.Subset(full_val_dataset, indices=list(range(32)))
val_loader = DataLoader(small_val_dataset, batch_size=16, shuffle=False, num_workers=0)

print(f"   Train samples: {len(small_dataset)}")
print(f"   Val samples: {len(small_val_dataset)}")

# Train for 1 epoch with stream monitoring
print("\n4. Training 1 epoch with stream_monitoring=True...")
print("=" * 80)

history = model.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=1,
    verbose=True,
    stream_monitoring=True,  # ← Enable stream monitoring
    grad_clip_norm=5.0
)

print("\n" * 2)
print("=" * 80)
print("✅ EXPECTED OUTPUT:")
print("=" * 80)
print("After the progress bar, you should see:")
print("  Stream Monitoring:")
print("    RGB   - LR: 0.000200, WD: 0.01000, Train: XX.XX%, Val: XX.XX%")
print("    Depth - LR: 0.000050, WD: 0.02000, Train: XX.XX%, Val: XX.XX%")
print("    Fuse  - LR: 0.000100, WD: 0.00100")
print("\n✅ Test complete!")
print("=" * 80)
