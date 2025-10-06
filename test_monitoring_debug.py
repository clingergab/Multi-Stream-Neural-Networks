"""
Debug version - add print statements to monitoring function.
"""

import torch
import torch.nn as nn
from src.models.multi_channel.mc_resnet import mc_resnet18
from torch.utils.data import TensorDataset, DataLoader

print("=" * 80)
print("DEBUGGING MONITORING FUNCTION")
print("=" * 80)

# Create synthetic data
n_samples = 64
rgb_data = torch.randn(n_samples, 3, 224, 224)
depth_data = torch.randn(n_samples, 1, 224, 224)
labels = torch.randint(0, 27, (n_samples,))

dataset = TensorDataset(rgb_data, depth_data, labels)
train_loader = DataLoader(dataset, batch_size=16, shuffle=False)
val_loader = DataLoader(dataset, batch_size=16, shuffle=False)

# Create model
model = mc_resnet18(
    num_classes=27,
    stream1_input_channels=3,
    stream2_input_channels=1,
    fusion_type='weighted',
    dropout_p=0.5,
    device='cpu'
)
model.compile(optimizer='adamw', learning_rate=1e-4, weight_decay=2e-2)

# Manual inline monitoring (copy of the function with debug prints)
print("\n1. Running monitoring manually with debug output...")

model.eval()
fusion_type = model.fusion.__class__.__name__
print(f"   Fusion type: {fusion_type}")

# Accumulators
stream1_train_correct = 0
stream1_train_total = 0
stream2_train_correct = 0
stream2_train_total = 0

max_batches = 2

with torch.no_grad():
    train_batches_processed = 0
    for batch_idx, (stream1_train, stream2_train, targets_train) in enumerate(train_loader):
        if batch_idx >= max_batches:
            break

        print(f"\n   Processing train batch {batch_idx}...")
        print(f"     Batch size: {targets_train.size(0)}")

        stream1_train = stream1_train.to(model.device)
        stream2_train = stream2_train.to(model.device)
        targets_train = targets_train.to(model.device)

        # Stream1 only performance
        stream1_train_features = model._forward_stream1_pathway(stream1_train)
        stream2_dummy_train = torch.zeros_like(stream1_train_features)
        stream1_train_fused = model.fusion(stream1_train_features, stream2_dummy_train)
        stream1_train_fused = model.dropout(stream1_train_fused)
        stream1_train_out = model.fc(stream1_train_fused)

        stream1_batch_correct = (stream1_train_out.argmax(1) == targets_train).sum().item()
        stream1_train_correct += stream1_batch_correct
        stream1_train_total += targets_train.size(0)

        print(f"     Stream1: {stream1_batch_correct}/{targets_train.size(0)} correct")
        print(f"     Stream1 running total: {stream1_train_correct}/{stream1_train_total}")

        # Stream2 only performance
        stream2_train_features = model._forward_stream2_pathway(stream2_train)
        stream1_dummy_train = torch.zeros_like(stream2_train_features)
        stream2_train_fused = model.fusion(stream1_dummy_train, stream2_train_features)
        stream2_train_fused = model.dropout(stream2_train_fused)
        stream2_train_out = model.fc(stream2_train_fused)

        stream2_batch_correct = (stream2_train_out.argmax(1) == targets_train).sum().item()
        stream2_train_correct += stream2_batch_correct
        stream2_train_total += targets_train.size(0)

        print(f"     Stream2: {stream2_batch_correct}/{targets_train.size(0)} correct")
        print(f"     Stream2 running total: {stream2_train_correct}/{stream2_train_total}")

        train_batches_processed += 1

    print(f"\n   Final train results:")
    stream1_train_acc = stream1_train_correct / max(stream1_train_total, 1)
    stream2_train_acc = stream2_train_correct / max(stream2_train_total, 1)
    print(f"     Stream1: {stream1_train_correct}/{stream1_train_total} = {stream1_train_acc*100:.2f}%")
    print(f"     Stream2: {stream2_train_correct}/{stream2_train_total} = {stream2_train_acc*100:.2f}%")

print("\n" + "=" * 80)
print("If Stream1 and Stream2 show THE SAME accuracy, there's a bug!")
print("They should be different for random predictions")
print("=" * 80)
