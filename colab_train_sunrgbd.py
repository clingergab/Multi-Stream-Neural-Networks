"""
Colab training script for SUN RGB-D 15-category scene classification.

This replaces the NYU Depth V2 training with the larger, more balanced SUN RGB-D dataset.

SETUP INSTRUCTIONS:
1. Upload data/sunrgbd_15/ folder to Google Drive → My Drive → datasets/
2. Mount Drive and run the dataset setup cell (see colab_sunrgbd_setup.py)
3. Run this training script
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
import os
from pathlib import Path

# Import custom modules
import sys
sys.path.append('/content/Multi-Stream-Neural-Networks/src')

from data_utils.sunrgbd_dataset import get_sunrgbd_dataloaders
from models.multi_channel.mc_resnet import mc_resnet18, mc_resnet50
from models.utils.stream_monitor import StreamMonitor

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Verify dataset exists
DATASET_PATH = '/content/data/sunrgbd_15'
if not Path(DATASET_PATH).exists():
    print(f"\n⚠ ERROR: Dataset not found at {DATASET_PATH}")
    print("Please run the dataset setup cell first (see colab_sunrgbd_setup.py)")
    raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    # Dataset
    'data_root': '/content/data/sunrgbd_15',
    'num_classes': 15,
    'batch_size': 64,
    'num_workers': 2,
    'image_size': (224, 224),

    # Model
    'model_type': 'resnet18',  # 'resnet18' or 'resnet50'
    'pretrained': True,

    # Training
    'num_epochs': 30,
    'base_lr': 0.001,
    'weight_decay': 1e-4,
    'scheduler': 'cosine',  # 'cosine' or 'step'

    # Stream-specific optimization (set to None to disable)
    'stream_specific': {
        'stream1_lr_mult': 1.0,    # RGB stream LR multiplier
        'stream2_lr_mult': 1.5,    # Depth stream LR multiplier (boost depth)
        'stream1_wd_mult': 1.0,    # RGB weight decay multiplier
        'stream2_wd_mult': 0.5,    # Depth weight decay multiplier (less regularization)
    },

    # Monitoring
    'monitor_streams': True,
    'monitor_interval': 1,  # Monitor every N epochs

    # Checkpointing
    'checkpoint_dir': '/content/drive/MyDrive/msnn_checkpoints',
    'save_best': True,
}

# ============================================================================
# Setup
# ============================================================================

# Create checkpoint directory
os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)

# Load dataloaders
print("\nLoading SUN RGB-D dataset...")
train_loader, val_loader = get_sunrgbd_dataloaders(
    data_root=CONFIG['data_root'],
    batch_size=CONFIG['batch_size'],
    num_workers=CONFIG['num_workers'],
    target_size=CONFIG['image_size'],
)

# Create model
print(f"\nCreating {CONFIG['model_type']} model...")
if CONFIG['model_type'] == 'resnet18':
    model = mc_resnet18(
        num_classes=CONFIG['num_classes'],
        pretrained=CONFIG['pretrained'],
    )
elif CONFIG['model_type'] == 'resnet50':
    model = mc_resnet50(
        num_classes=CONFIG['num_classes'],
        pretrained=CONFIG['pretrained'],
    )
else:
    raise ValueError(f"Unknown model type: {CONFIG['model_type']}")

model = model.to(device)

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer with stream-specific learning rates
if CONFIG['stream_specific'] is not None:
    print("\nUsing stream-specific optimization:")
    print(f"  Stream1 (RGB) LR: {CONFIG['base_lr'] * CONFIG['stream_specific']['stream1_lr_mult']:.6f}")
    print(f"  Stream2 (Depth) LR: {CONFIG['base_lr'] * CONFIG['stream_specific']['stream2_lr_mult']:.6f}")

    # Separate parameters for each stream
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

    optimizer = optim.Adam([
        {
            'params': stream1_params,
            'lr': CONFIG['base_lr'] * CONFIG['stream_specific']['stream1_lr_mult'],
            'weight_decay': CONFIG['weight_decay'] * CONFIG['stream_specific']['stream1_wd_mult']
        },
        {
            'params': stream2_params,
            'lr': CONFIG['base_lr'] * CONFIG['stream_specific']['stream2_lr_mult'],
            'weight_decay': CONFIG['weight_decay'] * CONFIG['stream_specific']['stream2_wd_mult']
        },
        {
            'params': other_params,
            'lr': CONFIG['base_lr'],
            'weight_decay': CONFIG['weight_decay']
        }
    ])
else:
    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG['base_lr'],
        weight_decay=CONFIG['weight_decay']
    )

# Scheduler
if CONFIG['scheduler'] == 'cosine':
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG['num_epochs'])
else:
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Stream monitor
if CONFIG['monitor_streams']:
    monitor = StreamMonitor(model)

# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc='Training')
    for rgb, depth, labels in pbar:
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

        pbar.set_postfix({
            'loss': f'{running_loss / (pbar.n + 1):.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })

    return running_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc='Validation')
        for rgb, depth, labels in pbar:
            rgb, depth, labels = rgb.to(device), depth.to(device), labels.to(device)

            outputs = model(rgb, depth)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f'{running_loss / (pbar.n + 1):.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

    return running_loss / len(loader), 100. * correct / total

# ============================================================================
# Training Loop
# ============================================================================

best_val_acc = 0.0
history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': [],
}

print("\n" + "="*80)
print("Starting training...")
print("="*80)

for epoch in range(1, CONFIG['num_epochs'] + 1):
    print(f"\nEpoch {epoch}/{CONFIG['num_epochs']}")
    print("-" * 80)

    # Train
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

    # Validate
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    # Update scheduler
    scheduler.step()

    # Record history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    # Print epoch summary
    current_lr = optimizer.param_groups[0]['lr']
    print(f"\nEpoch {epoch} Summary:")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
    print(f"  LR: {current_lr:.6f}")

    # Monitor streams
    if CONFIG['monitor_streams'] and epoch % CONFIG['monitor_interval'] == 0:
        print("\n" + "="*80)
        print(f"Stream Monitoring (Epoch {epoch})")
        print("="*80)
        monitor_stats = monitor.analyze_streams(
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            max_batches=20  # Sample 20 batches for speed
        )
        print(monitor_stats)

    # Save best model
    if CONFIG['save_best'] and val_acc > best_val_acc:
        best_val_acc = val_acc
        checkpoint_path = os.path.join(CONFIG['checkpoint_dir'], 'best_model.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
        }, checkpoint_path)
        print(f"\n✓ Saved best model (Val Acc: {val_acc:.2f}%)")

    print("-" * 80)

print("\n" + "="*80)
print("Training completed!")
print(f"Best validation accuracy: {best_val_acc:.2f}%")
print("="*80)

# Save final model
final_checkpoint_path = os.path.join(CONFIG['checkpoint_dir'], 'final_model.pth')
torch.save({
    'epoch': CONFIG['num_epochs'],
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'history': history,
}, final_checkpoint_path)
print(f"\nFinal model saved to: {final_checkpoint_path}")
