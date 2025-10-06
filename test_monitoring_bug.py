"""
Test the monitoring function to find the bug causing weird results.

The issue: Depth stream showing 48% accuracy while RGB shows 6%,
but full model also shows 48%. This suggests the monitoring is broken.
"""

import torch
import torch.nn as nn
from src.models.multi_channel.mc_resnet import mc_resnet18
from src.models.utils.stream_monitor import StreamMonitor
from src.data_utils.nyu_depth_dataset import create_nyu_dataloaders

print("=" * 80)
print("TESTING STREAM MONITORING FOR BUGS")
print("=" * 80)

# Create a small model
print("\n1. Creating model...")
model = mc_resnet18(
    num_classes=27,
    stream1_input_channels=3,
    stream2_input_channels=1,
    fusion_type='weighted',
    dropout_p=0.5,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)
model.compile(
    optimizer='adamw',
    learning_rate=1e-4,
    weight_decay=2e-2
)

print(f"   Model device: {model.device}")
print(f"   Fusion type: {model.fusion.__class__.__name__}")

# Create small dataloader
print("\n2. Creating small test dataset...")
train_loader, val_loader = create_nyu_dataloaders(
    h5_file_path='/Users/gclinger/Documents/datasets/nyu_depth_v2_labeled.mat',
    batch_size=32,
    num_workers=0,
    target_size=(224, 224),
    num_classes=27
)

print(f"   Train batches: {len(train_loader)}")
print(f"   Val batches: {len(val_loader)}")

# Create monitor
print("\n3. Creating monitor...")
monitor = StreamMonitor(model)

# Get one batch for testing
print("\n4. Getting test batch...")
rgb_batch, depth_batch, labels_batch = next(iter(train_loader))
print(f"   RGB shape: {rgb_batch.shape}")
print(f"   Depth shape: {depth_batch.shape}")
print(f"   Labels shape: {labels_batch.shape}")

# Test forward pass
print("\n5. Testing full model forward pass...")
model.eval()
with torch.no_grad():
    rgb_cuda = rgb_batch.to(model.device)
    depth_cuda = depth_batch.to(model.device)
    labels_cuda = labels_batch.to(model.device)

    outputs = model(rgb_cuda, depth_cuda)
    full_acc = (outputs.argmax(1) == labels_cuda).float().mean().item()
    print(f"   Full model accuracy on batch: {full_acc*100:.2f}%")

# Test Stream1 isolation
print("\n6. Testing Stream1 (RGB) isolation...")
with torch.no_grad():
    # Get RGB features
    stream1_features = model._forward_stream1_pathway(rgb_cuda)
    print(f"   Stream1 features shape: {stream1_features.shape}")
    print(f"   Stream1 features mean: {stream1_features.mean():.4f}")
    print(f"   Stream1 features std: {stream1_features.std():.4f}")

    # Create dummy Stream2 (zeros)
    stream2_dummy = torch.zeros_like(stream1_features)
    print(f"   Stream2 dummy shape: {stream2_dummy.shape}")
    print(f"   Stream2 dummy all zeros: {(stream2_dummy == 0).all()}")

    # Pass through fusion
    fused = model.fusion(stream1_features, stream2_dummy)
    print(f"   Fused shape: {fused.shape}")
    print(f"   Fused mean: {fused.mean():.4f}")
    print(f"   Fused std: {fused.std():.4f}")

    # Check fusion weights
    if hasattr(model.fusion, 'stream1_weight'):
        w1 = torch.sigmoid(model.fusion.stream1_weight).item()
        w2 = torch.sigmoid(model.fusion.stream2_weight).item()
        print(f"   Fusion weight1 (RGB): {w1:.6f}")
        print(f"   Fusion weight2 (Depth): {w2:.6f}")

    # Pass through dropout and FC
    fused_dropout = model.dropout(fused)
    outputs_stream1 = model.fc(fused_dropout)
    stream1_acc = (outputs_stream1.argmax(1) == labels_cuda).float().mean().item()
    print(f"   Stream1 isolated accuracy: {stream1_acc*100:.2f}%")

# Test Stream2 isolation
print("\n7. Testing Stream2 (Depth) isolation...")
with torch.no_grad():
    # Get Depth features
    stream2_features = model._forward_stream2_pathway(depth_cuda)
    print(f"   Stream2 features shape: {stream2_features.shape}")
    print(f"   Stream2 features mean: {stream2_features.mean():.4f}")
    print(f"   Stream2 features std: {stream2_features.std():.4f}")

    # Create dummy Stream1 (zeros)
    stream1_dummy = torch.zeros_like(stream2_features)
    print(f"   Stream1 dummy shape: {stream1_dummy.shape}")
    print(f"   Stream1 dummy all zeros: {(stream1_dummy == 0).all()}")

    # Pass through fusion
    fused = model.fusion(stream1_dummy, stream2_features)
    print(f"   Fused shape: {fused.shape}")
    print(f"   Fused mean: {fused.mean():.4f}")
    print(f"   Fused std: {fused.std():.4f}")

    # Pass through dropout and FC
    fused_dropout = model.dropout(fused)
    outputs_stream2 = model.fc(fused_dropout)
    stream2_acc = (outputs_stream2.argmax(1) == labels_cuda).float().mean().item()
    print(f"   Stream2 isolated accuracy: {stream2_acc*100:.2f}%")

# Now test the actual monitoring function
print("\n8. Testing monitoring function...")
print("   (This should give similar results to manual test above)")

# Run one training step first to get realistic state
model.train()
rgb_cuda = rgb_batch.to(model.device)
depth_cuda = depth_batch.to(model.device)
labels_cuda = labels_batch.to(model.device)

model.optimizer.zero_grad()
outputs = model(rgb_cuda, depth_cuda)
loss = model.criterion(outputs, labels_cuda)
loss.backward()
model.optimizer.step()

train_acc = (outputs.argmax(1) == labels_cuda).float().mean().item()
train_loss = loss.item()

# Now test monitoring
overfitting_stats = monitor.compute_stream_overfitting_indicators(
    train_loss=train_loss,
    val_loss=train_loss,  # Use same for simplicity
    train_acc=train_acc,
    val_acc=train_acc,  # Use same for simplicity
    train_loader=train_loader,
    val_loader=val_loader,
    max_batches=1  # Just one batch for testing
)

print(f"\n   Monitoring results:")
print(f"   Stream1 train acc: {overfitting_stats['stream1_train_acc']*100:.2f}%")
print(f"   Stream1 val acc: {overfitting_stats['stream1_val_acc']*100:.2f}%")
print(f"   Stream2 train acc: {overfitting_stats['stream2_train_acc']*100:.2f}%")
print(f"   Stream2 val acc: {overfitting_stats['stream2_val_acc']*100:.2f}%")

print("\n" + "=" * 80)
print("TEST COMPLETE - Check if results make sense!")
print("=" * 80)
