#!/usr/bin/env python3
"""Test with user's actual config"""

import torch
from torch.utils.data import DataLoader, TensorDataset
from src.models.multi_channel.mc_resnet import mc_resnet18

print("="*60)
print("TEST WITH USER'S ACTUAL CONFIG")
print("="*60)

# Create tiny fake dataset
torch.manual_seed(42)
n_samples = 64
rgb = torch.randn(n_samples, 3, 224, 224)
depth = torch.randn(n_samples, 1, 224, 224)
labels = torch.randint(0, 15, (n_samples,))

dataset = TensorDataset(rgb, depth, labels)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset, batch_size=32, shuffle=False)

# Create model
model = mc_resnet18(
    num_classes=15,
    stream1_input_channels=3,
    stream2_input_channels=1,
    fusion_type='concat',
    dropout_p=0.5,
    device='cpu'
)

# User's actual config from notebook
STREAM_SPECIFIC_CONFIG = {
    'optimizer': 'adamw',
    'learning_rate': 7e-5,           # Base LR for shared params (fusion, classifier)
    'weight_decay': 2e-4,             # Base weight decay

    # Stream-specific settings (adjusted based on research):
    'stream1_lr': 3e-5,               # RGB stream: lower LR (more regularization)
    'stream1_weight_decay': 5e-4,     # RGB stream: higher WD (prevent overfitting)
    'stream2_lr': 1e-4,               # Depth stream: higher LR (needs more learning)
    'stream2_weight_decay': 1e-4,     # Depth stream: lighter WD (less regularization)

    'loss': 'cross_entropy',
    'scheduler': 'cosine'
}

model.compile(**STREAM_SPECIFIC_CONFIG)

print(f"\n✅ Model compiled with user's config")
print(f"   Param groups: {len(model.optimizer.param_groups)}")
for i, group in enumerate(model.optimizer.param_groups):
    num_params = sum(p.numel() for p in group['params'])
    print(f"   Group {i}: LR={group['lr']:.2e}, WD={group['weight_decay']:.2e}, Params={num_params:,}")

print(f"\n{'='*60}")
print("EXPECTED: Progress bar should show lr=0.000070 (7e-5)")
print(f"{'='*60}\n")

# Train for 2 epochs
history = model.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=2,
    verbose=True,
    stream_monitoring=True
)

print(f"\n{'='*60}")
print("✅ Check the progress bar 'lr=' value - should be ~7e-5")
print("✅ Check stream monitoring appears after each epoch")
print(f"{'='*60}")
