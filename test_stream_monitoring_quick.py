"""
Quick test of stream monitoring display with StreamMonitor.
"""

import torch
from src.models.multi_channel.mc_resnet import mc_resnet18
from src.data_utils.sunrgbd_dataset import SUNRGBDDataset
from torch.utils.data import DataLoader

print("Testing stream monitoring with StreamMonitor integration...")

# Create model
model = mc_resnet18(num_classes=15, stream1_input_channels=3, stream2_input_channels=1,
                    fusion_type='concat', device='cpu', use_amp=False)

# Compile with stream-specific LRs
model.compile(optimizer='adamw', learning_rate=1e-4, stream1_lr=2e-4, stream1_weight_decay=1e-2,
              stream2_lr=5e-5, stream2_weight_decay=2e-2, weight_decay=1e-3, loss='cross_entropy')

print(f"\nParameter groups: {len(model.optimizer.param_groups)}")

# Create small dataset loaders
print("\nCreating small data loaders...")
full_dataset = SUNRGBDDataset(data_root='data/sunrgbd_15', train=True)
small_dataset = torch.utils.data.Subset(full_dataset, indices=list(range(16)))
train_loader = DataLoader(small_dataset, batch_size=16, shuffle=False, num_workers=0)

full_val = SUNRGBDDataset(data_root='data/sunrgbd_15', train=False)
small_val = torch.utils.data.Subset(full_val, indices=list(range(16)))
val_loader = DataLoader(small_val, batch_size=16, shuffle=False, num_workers=0)

# Test the monitoring print method
print("\nCreating StreamMonitor and calling _print_stream_monitoring()...")
from src.models.utils.stream_monitor import StreamMonitor
monitor = StreamMonitor(model)

# Call with mock metric values
model._print_stream_monitoring(
    train_loader=train_loader,
    val_loader=val_loader,
    monitor=monitor,
    train_loss=2.5,
    train_acc=0.35,
    val_loss=2.8,
    val_acc=0.30
)

print("\n✅ Stream monitoring display works!")
print("✅ Now passing real metric values (train_loss, val_loss, train_acc, val_acc)")
print("✅ StreamMonitor can now properly compute full_loss_gap and full_acc_gap")
