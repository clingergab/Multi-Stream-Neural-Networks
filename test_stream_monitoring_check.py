#!/usr/bin/env python3
"""Test if stream monitoring is being called during fit()"""

import torch
from torch.utils.data import DataLoader, TensorDataset
from src.models.multi_channel.mc_resnet import mc_resnet18

print("="*60)
print("STREAM MONITORING CHECK")
print("="*60)

# Create tiny fake dataset
torch.manual_seed(42)
n_samples = 64
rgb = torch.randn(n_samples, 3, 224, 224)
depth = torch.randn(n_samples, 1, 224, 224)
labels = torch.randint(0, 10, (n_samples,))

dataset = TensorDataset(rgb, depth, labels)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset, batch_size=32, shuffle=False)

print(f"\n✅ Created fake dataset: {n_samples} samples, 10 classes")

# Create model
model = mc_resnet18(
    num_classes=10,
    stream1_input_channels=3,
    stream2_input_channels=1,
    device='cpu'
)

print(f"✅ Model created")

# Compile with stream-specific settings
model.compile(
    optimizer='adamw',
    learning_rate=1e-3,
    weight_decay=1e-2,
    stream1_lr=5e-4,
    stream1_weight_decay=2e-2,
    stream2_lr=2e-3,
    stream2_weight_decay=5e-3,
    loss='cross_entropy'
)

print(f"✅ Model compiled with stream-specific settings")
print(f"   Param groups: {len(model.optimizer.param_groups)}")
for i, group in enumerate(model.optimizer.param_groups):
    num_params = sum(p.numel() for p in group['params'])
    print(f"   Group {i}: LR={group['lr']:.2e}, WD={group['weight_decay']:.2e}, Params={num_params:,}")

print(f"\n{'='*60}")
print("STARTING TRAINING WITH STREAM_MONITORING=TRUE")
print(f"{'='*60}\n")

# Train for 2 epochs with stream monitoring
history = model.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=2,
    verbose=True,
    stream_monitoring=True  # Should print stream monitoring!
)

print(f"\n{'='*60}")
print("✅ Training complete!")
print(f"{'='*60}")
