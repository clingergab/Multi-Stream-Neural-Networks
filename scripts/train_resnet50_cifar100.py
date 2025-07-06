#!/usr/bin/env python
"""
Training script for ResNet50 on CIFAR-100 dataset.
This script demonstrates how to use our modified ResNet implementation.
"""

import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models2.core.resnet import resnet50
from src.utils.cifar100_loader import load_cifar100_numpy

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

def get_args():
    parser = argparse.ArgumentParser(description='Train ResNet50 on CIFAR-100')
    parser.add_argument('--data-dir', type=str, default='./data/cifar-100', help='data directory containing CIFAR-100 files')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='checkpoint directory')
    parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
    return parser.parse_args()

# Custom dataset for transformations
class TransformDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y
            
    def __len__(self):
        return len(self.data)


def get_data_loaders(data_dir, batch_size):
    # CIFAR-100 normalization values
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]
    
    # Data transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean, std),
    ])
    
    transform_test = transforms.Compose([
        transforms.Normalize(mean, std),
    ])
    
    # Create datasets using custom loader
    try:
        print(f"Loading CIFAR-100 dataset from {data_dir}...")
        train_data, train_labels, test_data, test_labels, label_names = load_cifar100_numpy(data_dir=data_dir)
        
        # Convert numpy arrays to tensors
        train_data = torch.from_numpy(train_data)  # Already normalized in load_cifar100_numpy
        test_data = torch.from_numpy(test_data)    # Already normalized in load_cifar100_numpy
        train_labels = torch.from_numpy(train_labels).long()
        test_labels = torch.from_numpy(test_labels).long()
        
        # Create datasets
        train_dataset = TransformDataset(train_data, train_labels, transform_train)
        test_dataset = TransformDataset(test_data, test_labels, transform_test)
        
        print(f"Successfully loaded CIFAR-100 dataset:")
        print(f"  Train: {len(train_dataset)} samples")
        print(f"  Test: {len(test_dataset)} samples")
        print(f"  Classes: {len(label_names)}")
        
    except Exception as e:
        print(f"Error loading CIFAR-100 dataset: {e}")
        raise RuntimeError(f"Failed to load CIFAR-100 dataset from {data_dir}: {e}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    
    return train_loader, test_loader

def train(model, train_loader, criterion, optimizer, epoch, device):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 50 == 0:
            print(f'Epoch: {epoch} | Batch: {batch_idx}/{len(train_loader)} | Loss: {train_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.3f}%')
    
    return train_loss / len(train_loader), 100. * correct / total

def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    print(f'Test Loss: {test_loss/len(test_loader):.3f} | Test Acc: {100.*correct/total:.3f}%')
    
    return test_loss / len(test_loader), 100. * correct / total

def main():
    args = get_args()
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Get data loaders
    train_loader, test_loader = get_data_loaders(args.data_dir, args.batch_size)
    
    # Create model
    print("Creating ResNet50 model...")
    # CIFAR-100 has 100 classes
    model = resnet50(num_classes=100)
    
    # For CIFAR, we need to modify the first conv layer since images are 32x32
    # Replace the first 7x7 conv with a 3x3 conv with stride 1
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # Remove the max pooling layer after the first conv
    model.maxpool = nn.Identity()
    
    model = model.to(device)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), 
        lr=args.lr, 
        momentum=args.momentum, 
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_acc = 0
    
    if args.resume:
        checkpoint_path = os.path.join(args.checkpoint_dir, 'resnet50_cifar100.pth')
        if os.path.exists(checkpoint_path):
            print(f"Resuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint['best_acc']
            print(f"Loaded checkpoint from epoch {checkpoint['epoch']} with accuracy {best_acc:.2f}%")
        else:
            print(f"No checkpoint found at {checkpoint_path}")
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch: {epoch+1}/{args.epochs}")
        
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, epoch, device)
        test_loss, test_acc = test(model, test_loader, criterion, device)
        
        scheduler.step()
        
        # Save checkpoint if better than best accuracy
        if test_acc > best_acc:
            print(f"Saving checkpoint... (accuracy improved from {best_acc:.2f}% to {test_acc:.2f}%)")
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'best_acc': test_acc
            }
            torch.save(state, os.path.join(args.checkpoint_dir, 'resnet50_cifar100.pth'))
            best_acc = test_acc
    
    print(f"Best accuracy: {best_acc:.2f}%")

if __name__ == '__main__':
    main()
