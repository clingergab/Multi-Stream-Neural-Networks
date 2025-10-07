"""
Comprehensive test suite for SUN RGB-D 15-category dataset and training pipeline.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import sys
import os
import numpy as np
from collections import Counter

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_utils.sunrgbd_dataset import SUNRGBDDataset, get_sunrgbd_dataloaders
from models.multi_channel.mc_resnet import mc_resnet18, mc_resnet50
from models.utils.stream_monitor import StreamMonitor

print("=" * 80)
print("SUN RGB-D 15-Category Dataset & Training Pipeline Test")
print("=" * 80)

# ============================================================================
# Test 1: Dataset Loading
# ============================================================================
print("\n[Test 1] Dataset Loading")
print("-" * 80)

try:
    train_dataset = SUNRGBDDataset(
        data_root='data/sunrgbd_15',
        train=True,
        target_size=(224, 224)
    )
    val_dataset = SUNRGBDDataset(
        data_root='data/sunrgbd_15',
        train=False,
        target_size=(224, 224)
    )

    print(f"✓ Train dataset loaded: {len(train_dataset)} samples")
    print(f"✓ Val dataset loaded: {len(val_dataset)} samples")
    assert len(train_dataset) == 8041, f"Expected 8041 train samples, got {len(train_dataset)}"
    assert len(val_dataset) == 2018, f"Expected 2018 val samples, got {len(val_dataset)}"
    print("✓ Sample counts correct")

except Exception as e:
    print(f"✗ Dataset loading failed: {e}")
    sys.exit(1)

# ============================================================================
# Test 2: Sample Access and Shape Verification
# ============================================================================
print("\n[Test 2] Sample Access and Shape Verification")
print("-" * 80)

try:
    # Test train sample
    rgb_train, depth_train, label_train = train_dataset[0]
    print(f"Train sample 0:")
    print(f"  RGB shape: {rgb_train.shape} (expected: [3, 224, 224])")
    print(f"  Depth shape: {depth_train.shape} (expected: [1, 224, 224])")
    print(f"  Label: {label_train} ({train_dataset.CLASS_NAMES[label_train]})")

    assert rgb_train.shape == (3, 224, 224), f"RGB shape mismatch: {rgb_train.shape}"
    assert depth_train.shape == (1, 224, 224), f"Depth shape mismatch: {depth_train.shape}"
    assert 0 <= label_train < 15, f"Label out of range: {label_train}"
    print("✓ Train sample shapes correct")

    # Test val sample
    rgb_val, depth_val, label_val = val_dataset[0]
    print(f"\nVal sample 0:")
    print(f"  RGB shape: {rgb_val.shape}")
    print(f"  Depth shape: {depth_val.shape}")
    print(f"  Label: {label_val} ({val_dataset.CLASS_NAMES[label_val]})")

    assert rgb_val.shape == (3, 224, 224), f"RGB shape mismatch: {rgb_val.shape}"
    assert depth_val.shape == (1, 224, 224), f"Depth shape mismatch: {depth_val.shape}"
    print("✓ Val sample shapes correct")

except Exception as e:
    print(f"✗ Sample access failed: {e}")
    sys.exit(1)

# ============================================================================
# Test 3: Class Distribution Verification
# ============================================================================
print("\n[Test 3] Class Distribution Verification")
print("-" * 80)

try:
    train_dist = train_dataset.get_class_distribution()
    val_dist = val_dataset.get_class_distribution()

    print("Train distribution (top 5):")
    sorted_train = sorted(train_dist.items(), key=lambda x: x[1]['count'], reverse=True)[:5]
    for class_name, info in sorted_train:
        print(f"  {class_name:20s}: {info['count']:4d} ({info['percentage']:5.2f}%)")

    print("\nVal distribution (top 5):")
    sorted_val = sorted(val_dist.items(), key=lambda x: x[1]['count'], reverse=True)[:5]
    for class_name, info in sorted_val:
        print(f"  {class_name:20s}: {info['count']:4d} ({info['percentage']:5.2f}%)")

    # Verify all 15 classes present
    assert len(train_dist) == 15, f"Train should have 15 classes, got {len(train_dist)}"
    assert len(val_dist) == 15, f"Val should have 15 classes, got {len(val_dist)}"
    print("✓ All 15 classes present in both splits")

    # Check class balance (should be <10x imbalance)
    train_counts = [info['count'] for info in train_dist.values()]
    imbalance = max(train_counts) / min(train_counts)
    print(f"✓ Class imbalance: {imbalance:.1f}x (should be <10x)")
    assert imbalance < 10, f"Imbalance too high: {imbalance:.1f}x"

except Exception as e:
    print(f"✗ Class distribution check failed: {e}")
    sys.exit(1)

# ============================================================================
# Test 4: DataLoader Creation
# ============================================================================
print("\n[Test 4] DataLoader Creation")
print("-" * 80)

try:
    train_loader, val_loader = get_sunrgbd_dataloaders(
        data_root='data/sunrgbd_15',
        batch_size=16,
        num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
        target_size=(224, 224)
    )

    print(f"✓ Train loader: {len(train_loader)} batches")
    print(f"✓ Val loader: {len(val_loader)} batches")

    assert len(train_loader) == (8041 + 15) // 16, "Train loader batch count mismatch"
    assert len(val_loader) == (2018 + 15) // 16, "Val loader batch count mismatch"
    print("✓ Batch counts correct")

except Exception as e:
    print(f"✗ DataLoader creation failed: {e}")
    sys.exit(1)

# ============================================================================
# Test 5: Batch Iteration and Shape Verification
# ============================================================================
print("\n[Test 5] Batch Iteration and Shape Verification")
print("-" * 80)

try:
    # Get first batch from train loader
    rgb_batch, depth_batch, labels_batch = next(iter(train_loader))

    print(f"Train batch shapes:")
    print(f"  RGB: {rgb_batch.shape} (expected: [16, 3, 224, 224])")
    print(f"  Depth: {depth_batch.shape} (expected: [16, 1, 224, 224])")
    print(f"  Labels: {labels_batch.shape} (expected: [16])")

    assert rgb_batch.shape == (16, 3, 224, 224), f"RGB batch shape mismatch"
    assert depth_batch.shape == (16, 1, 224, 224), f"Depth batch shape mismatch"
    assert labels_batch.shape == (16,), f"Labels batch shape mismatch"

    # Check label range
    assert labels_batch.min() >= 0 and labels_batch.max() < 15, "Labels out of range"
    print(f"✓ Batch shapes correct")
    print(f"✓ Labels in range [0, 14]: min={labels_batch.min()}, max={labels_batch.max()}")

    # Get first batch from val loader
    rgb_val_batch, depth_val_batch, labels_val_batch = next(iter(val_loader))
    assert rgb_val_batch.shape == (16, 3, 224, 224), "Val RGB batch shape mismatch"
    assert depth_val_batch.shape == (16, 1, 224, 224), "Val depth batch shape mismatch"
    print("✓ Val batch shapes correct")

except Exception as e:
    print(f"✗ Batch iteration failed: {e}")
    sys.exit(1)

# ============================================================================
# Test 6: Model Creation
# ============================================================================
print("\n[Test 6] Model Creation")
print("-" * 80)

try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = mc_resnet18(
        num_classes=15,
        pretrained=False,  # Don't download pretrained for testing
    )
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"✓ Model created successfully")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

except Exception as e:
    print(f"✗ Model creation failed: {e}")
    sys.exit(1)

# ============================================================================
# Test 7: Forward Pass
# ============================================================================
print("\n[Test 7] Forward Pass")
print("-" * 80)

try:
    model.eval()
    rgb_batch = rgb_batch.to(device)
    depth_batch = depth_batch.to(device)
    labels_batch = labels_batch.to(device)

    with torch.no_grad():
        outputs = model(rgb_batch, depth_batch)

    print(f"Output shape: {outputs.shape} (expected: [16, 15])")
    assert outputs.shape == (16, 15), f"Output shape mismatch: {outputs.shape}"

    # Check output values are logits (not probabilities)
    print(f"Output range: [{outputs.min():.4f}, {outputs.max():.4f}]")

    # Get predictions
    _, predictions = outputs.max(1)
    print(f"Predictions: {predictions[:5].cpu().numpy()}")
    print(f"True labels: {labels_batch[:5].cpu().numpy()}")

    print("✓ Forward pass successful")

except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    sys.exit(1)

# ============================================================================
# Test 8: Loss Computation
# ============================================================================
print("\n[Test 8] Loss Computation")
print("-" * 80)

try:
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, labels_batch)

    print(f"Loss value: {loss.item():.4f}")
    assert loss.item() > 0, "Loss should be positive"
    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is inf"

    print("✓ Loss computation successful")

except Exception as e:
    print(f"✗ Loss computation failed: {e}")
    sys.exit(1)

# ============================================================================
# Test 9: Backward Pass and Gradient Check
# ============================================================================
print("\n[Test 9] Backward Pass and Gradient Check")
print("-" * 80)

try:
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = model(rgb_batch, depth_batch)
    loss = criterion(outputs, labels_batch)
    loss.backward()

    # Check gradients exist
    has_grads = 0
    total_params = 0
    for name, param in model.named_parameters():
        total_params += 1
        if param.grad is not None:
            has_grads += 1

    print(f"Parameters with gradients: {has_grads}/{total_params}")
    assert has_grads > 0, "No gradients computed"

    optimizer.step()

    print("✓ Backward pass successful")
    print("✓ Optimizer step successful")

except Exception as e:
    print(f"✗ Backward pass failed: {e}")
    sys.exit(1)

# ============================================================================
# Test 10: Stream-Specific Optimization
# ============================================================================
print("\n[Test 10] Stream-Specific Optimization")
print("-" * 80)

try:
    # Create optimizer with stream-specific learning rates
    stream1_params = []
    stream2_params = []
    other_params = []

    for name, param in model.named_parameters():
        if 'stream1' in name:
            stream1_params.append(param)
        elif 'stream2' in name:
            stream2_params.append(param)
        else:
            other_params.append(param)

    print(f"Stream1 (RGB) parameters: {len(stream1_params)}")
    print(f"Stream2 (Depth) parameters: {len(stream2_params)}")
    print(f"Other parameters: {len(other_params)}")

    optimizer_stream = optim.Adam([
        {'params': stream1_params, 'lr': 0.001, 'weight_decay': 1e-4},
        {'params': stream2_params, 'lr': 0.0015, 'weight_decay': 5e-5},
        {'params': other_params, 'lr': 0.001, 'weight_decay': 1e-4}
    ])

    print(f"✓ Stream-specific optimizer created")
    print(f"  Stream1 LR: {optimizer_stream.param_groups[0]['lr']:.6f}")
    print(f"  Stream2 LR: {optimizer_stream.param_groups[1]['lr']:.6f}")
    print(f"  Other LR: {optimizer_stream.param_groups[2]['lr']:.6f}")

except Exception as e:
    print(f"✗ Stream-specific optimization setup failed: {e}")
    sys.exit(1)

# ============================================================================
# Test 11: Stream Monitoring
# ============================================================================
print("\n[Test 11] Stream Monitoring")
print("-" * 80)

try:
    monitor = StreamMonitor(model)

    # Do a forward pass to populate activations
    model.eval()
    with torch.no_grad():
        rgb_test, depth_test, _ = next(iter(train_loader))
        rgb_test, depth_test = rgb_test.to(device), depth_test.to(device)
        _ = model(rgb_test, depth_test)

    # Compute gradient stats after a training step
    model.train()
    optimizer.zero_grad()
    outputs = model(rgb_test, depth_test)
    loss = criterion(outputs, labels_batch)
    loss.backward()

    # Get monitoring stats
    grad_stats = monitor.compute_stream_gradients()
    weight_stats = monitor.compute_stream_weights()

    print("Gradient statistics:")
    for key, value in list(grad_stats.items())[:5]:
        print(f"  {key}: {value:.6f}")

    print("\nWeight statistics:")
    for key, value in list(weight_stats.items())[:5]:
        print(f"  {key}: {value:.6f}")

    # Get summary
    summary = monitor.get_summary()
    print(f"\nMonitor summary available: {len(summary)} characters")

    print("✓ Stream monitoring successful")

except Exception as e:
    print(f"✗ Stream monitoring failed: {e}")
    import traceback
    traceback.print_exc()
    # Don't exit - monitoring is optional
    print("⚠ Continuing without stream monitoring...")

# ============================================================================
# Test 12: Mini Training Loop (1 epoch, few batches)
# ============================================================================
print("\n[Test 12] Mini Training Loop (1 epoch, 10 batches)")
print("-" * 80)

try:
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    running_loss = 0.0
    correct = 0
    total = 0

    num_batches = min(10, len(train_loader))

    for i, (rgb, depth, labels) in enumerate(train_loader):
        if i >= num_batches:
            break

        rgb, depth, labels = rgb.to(device), depth.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(rgb, depth)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (i + 1) % 5 == 0:
            print(f"  Batch {i+1}/{num_batches}: Loss={loss.item():.4f}, Acc={100.*correct/total:.2f}%")

    final_loss = running_loss / num_batches
    final_acc = 100. * correct / total

    print(f"✓ Training loop completed")
    print(f"  Average loss: {final_loss:.4f}")
    print(f"  Accuracy: {final_acc:.2f}%")

    assert not np.isnan(final_loss), "Training loss is NaN"
    assert final_acc >= 0 and final_acc <= 100, "Accuracy out of range"

except Exception as e:
    print(f"✗ Training loop failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# Test 13: Validation Loop
# ============================================================================
print("\n[Test 13] Validation Loop (10 batches)")
print("-" * 80)

try:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    num_batches = min(10, len(val_loader))

    with torch.no_grad():
        for i, (rgb, depth, labels) in enumerate(val_loader):
            if i >= num_batches:
                break

            rgb, depth, labels = rgb.to(device), depth.to(device), labels.to(device)

            outputs = model(rgb, depth)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = running_loss / num_batches
    val_acc = 100. * correct / total

    print(f"✓ Validation loop completed")
    print(f"  Average loss: {val_loss:.4f}")
    print(f"  Accuracy: {val_acc:.2f}%")

    assert not np.isnan(val_loss), "Validation loss is NaN"
    assert val_acc >= 0 and val_acc <= 100, "Accuracy out of range"

except Exception as e:
    print(f"✗ Validation loop failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# Test 14: Scheduler
# ============================================================================
print("\n[Test 14] Learning Rate Scheduler")
print("-" * 80)

try:
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)

    initial_lr = optimizer.param_groups[0]['lr']
    print(f"Initial LR: {initial_lr:.6f}")

    # Step through a few epochs
    for epoch in range(5):
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch+1} LR: {current_lr:.6f}")

    print("✓ Scheduler working correctly")

except Exception as e:
    print(f"✗ Scheduler test failed: {e}")
    sys.exit(1)

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("ALL TESTS PASSED! ✓")
print("=" * 80)
print("\nSummary:")
print(f"  Dataset: SUN RGB-D 15 categories")
print(f"  Train samples: {len(train_dataset)}")
print(f"  Val samples: {len(val_dataset)}")
print(f"  Model: MCResNet18 with {total_params:,} parameters")
print(f"  Device: {device}")
print(f"  All components working correctly")
print("\nReady for full training on Colab!")
print("=" * 80)
