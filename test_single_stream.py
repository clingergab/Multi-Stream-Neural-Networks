#!/usr/bin/env python3
"""
Quick test to verify if our multi-stream model can work with single RGB stream.
This will test if the issue is the multi-stream architecture or the data preprocessing.
"""

import torch
import torch.nn as nn
from src.models.basic_multi_channel.multi_channel_resnet_network import MultiChannelResNetNetwork
from src.utils.cifar100_loader import get_cifar100_datasets
from torch.utils.data import DataLoader
import time

def test_single_stream_training():
    """Test if feeding RGB to both streams allows proper training."""
    
    print("🔬 Testing Single Stream vs Multi-Stream Training")
    print("=" * 60)
    
    # Get CIFAR-100 data
    print("📊 Loading CIFAR-100 data...")
    train_dataset, test_dataset, class_names = get_cifar100_datasets(data_dir="data/cifar-100")
    
    # Create a small subset for quick testing
    subset_size = 1000  # Small subset for quick testing
    train_subset = torch.utils.data.Subset(train_dataset, range(subset_size))
    
    # Create DataLoader
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    
    # Create model for CIFAR-100 with reduced architecture
    print("🏗️  Creating CIFAR-100 optimized model...")
    model = MultiChannelResNetNetwork.for_cifar100(dropout=0.3)
    model.compile()
    
    print(f"✅ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"   Architecture: {'Reduced' if model.reduce_architecture else 'Full'}")
    
    # Test 1: Multi-stream with RGB + Brightness (current failing approach)
    print("\n📊 Test 1: Multi-stream RGB + Brightness (current approach)")
    test_multi_stream_training(model, train_loader, "RGB+Brightness")
    
    # Test 2: Single-stream with RGB to both pathways
    print("\n📊 Test 2: Single-stream RGB to both pathways")
    test_single_stream_rgb(model, train_loader, "RGB+RGB")
    
    # Test 3: Multi-stream with RGB + Noise (to test if independence matters)
    print("\n📊 Test 3: Multi-stream RGB + Random Noise")
    test_multi_stream_with_noise(model, train_loader, "RGB+Noise")

def test_multi_stream_training(model, train_loader, test_name):
    """Test standard multi-stream training (RGB + Brightness)."""
    model.train()
    losses = []
    accuracies = []
    
    from src.transforms.rgb_to_rgbl import RGBtoRGBL
    rgb_to_rgbl = RGBtoRGBL()
    
    start_time = time.time()
    for i, (images, labels) in enumerate(train_loader):
        if i >= 10:  # Only test 10 batches for speed
            break
            
        # Convert to RGB + Brightness streams
        rgb_stream, brightness_stream = rgb_to_rgbl(images)
        
        # Move to device
        rgb_stream = rgb_stream.to(model.device)
        brightness_stream = brightness_stream.to(model.device)
        labels = labels.to(model.device)
        
        # Forward pass
        model.optimizer.zero_grad()
        outputs = model(rgb_stream, brightness_stream)
        loss = model.criterion(outputs, labels)
        loss.backward()
        model.optimizer.step()
        
        # Track metrics
        losses.append(loss.item())
        with torch.no_grad():
            _, predicted = outputs.max(1)
            accuracy = predicted.eq(labels).float().mean().item()
            accuracies.append(accuracy)
    
    elapsed = time.time() - start_time
    avg_loss = sum(losses) / len(losses)
    avg_acc = sum(accuracies) / len(accuracies)
    
    print(f"   {test_name}: Loss={avg_loss:.4f}, Acc={avg_acc:.4f}, Time={elapsed:.2f}s")
    return avg_loss, avg_acc

def test_single_stream_rgb(model, train_loader, test_name):
    """Test feeding RGB to both color and brightness streams."""
    model.train()
    losses = []
    accuracies = []
    
    start_time = time.time()
    for i, (images, labels) in enumerate(train_loader):
        if i >= 10:  # Only test 10 batches for speed
            break
            
        # Use RGB for both streams (convert RGB to grayscale for brightness)
        rgb_stream = images
        brightness_stream = images.mean(dim=1, keepdim=True)  # Simple grayscale conversion
        
        # Move to device
        rgb_stream = rgb_stream.to(model.device)
        brightness_stream = brightness_stream.to(model.device)
        labels = labels.to(model.device)
        
        # Forward pass
        model.optimizer.zero_grad()
        outputs = model(rgb_stream, brightness_stream)
        loss = model.criterion(outputs, labels)
        loss.backward()
        model.optimizer.step()
        
        # Track metrics
        losses.append(loss.item())
        with torch.no_grad():
            _, predicted = outputs.max(1)
            accuracy = predicted.eq(labels).float().mean().item()
            accuracies.append(accuracy)
    
    elapsed = time.time() - start_time
    avg_loss = sum(losses) / len(losses)
    avg_acc = sum(accuracies) / len(accuracies)
    
    print(f"   {test_name}: Loss={avg_loss:.4f}, Acc={avg_acc:.4f}, Time={elapsed:.2f}s")
    return avg_loss, avg_acc

def test_multi_stream_with_noise(model, train_loader, test_name):
    """Test RGB + random noise to see if independence helps."""
    model.train()
    losses = []
    accuracies = []
    
    start_time = time.time()
    for i, (images, labels) in enumerate(train_loader):
        if i >= 10:  # Only test 10 batches for speed
            break
            
        # Use RGB + random noise
        rgb_stream = images
        noise_stream = torch.randn_like(images[:, :1, :, :])  # Random noise, same spatial size
        
        # Move to device
        rgb_stream = rgb_stream.to(model.device)
        noise_stream = noise_stream.to(model.device)
        labels = labels.to(model.device)
        
        # Forward pass
        model.optimizer.zero_grad()
        outputs = model(rgb_stream, noise_stream)
        loss = model.criterion(outputs, labels)
        loss.backward()
        model.optimizer.step()
        
        # Track metrics
        losses.append(loss.item())
        with torch.no_grad():
            _, predicted = outputs.max(1)
            accuracy = predicted.eq(labels).float().mean().item()
            accuracies.append(accuracy)
    
    elapsed = time.time() - start_time
    avg_loss = sum(losses) / len(losses)
    avg_acc = sum(accuracies) / len(accuracies)
    
    print(f"   {test_name}: Loss={avg_loss:.4f}, Acc={avg_acc:.4f}, Time={elapsed:.2f}s")
    return avg_loss, avg_acc

if __name__ == "__main__":
    test_single_stream_training()
