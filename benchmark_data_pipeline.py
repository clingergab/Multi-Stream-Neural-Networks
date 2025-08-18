#!/usr/bin/env python3
"""
Data Pipeline Bottleneck Analysis Script

This script identifies exactly what's causing the 2+ hour per epoch slowdown 
in ImageNet training by testing different data loading configurations.
"""

import torch
import time
import os
from pathlib import Path
import psutil
from torch.utils.data import DataLoader
from src.data_utils.streaming_dual_channel_dataset import (
    create_imagenet_train_val_dataloaders,
    create_default_imagenet_transforms
)
from src.models2.multi_channel.mc_resnet import mc_resnet50

def benchmark_data_loading_config():
    """Test different data loading configurations to find the bottleneck."""
    
    print("🔍 DATA PIPELINE BOTTLENECK ANALYSIS")
    print("=" * 60)
    
    # Check system resources
    cpu_count = os.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    print(f"System: {cpu_count} CPU cores, {memory_gb:.1f}GB RAM")
    
    # Configuration to test
    test_configs = [
        {"name": "Current Default", "num_workers": max(1, cpu_count // 2), "batch_size": 64, "prefetch_factor": 2},
        {"name": "Single Threaded", "num_workers": 0, "batch_size": 64, "prefetch_factor": None},
        {"name": "More Workers", "num_workers": min(8, cpu_count), "batch_size": 64, "prefetch_factor": 4},
        {"name": "Larger Batch", "num_workers": max(1, cpu_count // 2), "batch_size": 128, "prefetch_factor": 2},
        {"name": "High Throughput", "num_workers": min(8, cpu_count), "batch_size": 128, "prefetch_factor": 4},
    ]
    
    # Test paths (update these to your actual ImageNet data)
    TRAIN_FOLDERS = ["data/ImageNet-1K/train_images_0"]
    VAL_FOLDER = "data/ImageNet-1K/val_images"  
    TRUTH_FILE = "data/ImageNet-1K/ILSVRC2013_devkit/data/ILSVRC2013_clsloc_validation_ground_truth.txt"
    
    # Check if paths exist
    train_path = Path(TRAIN_FOLDERS[0])
    if not train_path.exists():
        print(f"❌ Training data not found at: {train_path}")
        print("Please update TRAIN_FOLDERS, VAL_FOLDER, and TRUTH_FILE paths")
        return
    
    print(f"✅ Using ImageNet data at: {train_path.parent}")
    
    # Create transforms
    train_transform, val_transform = create_default_imagenet_transforms(
        image_size=(224, 224)
    )
    
    results = []
    
    for config in test_configs:
        print(f"\n🧪 Testing: {config['name']}")
        print(f"   num_workers: {config['num_workers']}, batch_size: {config['batch_size']}")
        
        try:
            # Create DataLoader with specific config
            train_loader, _ = create_imagenet_train_val_dataloaders(
                train_folders=TRAIN_FOLDERS,
                val_folder=VAL_FOLDER,
                truth_file=TRUTH_FILE,
                train_transform=train_transform,
                val_transform=val_transform,
                batch_size=config['batch_size'],
                num_workers=config['num_workers'],
                pin_memory=True,
                persistent_workers=config['num_workers'] > 0,
                prefetch_factor=config['prefetch_factor']
            )
            
            # Benchmark data loading speed
            print(f"   📊 Benchmarking {len(train_loader)} total batches...")
            
            # Test first 50 batches for speed estimation
            test_batches = min(50, len(train_loader))
            start_time = time.time()
            
            for i, (rgb, labels) in enumerate(train_loader):
                if i >= test_batches:
                    break
                    
                # Simulate minimal processing (move to device)
                rgb = rgb.cuda() if torch.cuda.is_available() else rgb
                labels = labels.cuda() if torch.cuda.is_available() else labels
                
                if i % 10 == 0:
                    elapsed = time.time() - start_time
                    batches_per_sec = (i + 1) / elapsed if elapsed > 0 else 0
                    samples_per_sec = batches_per_sec * config['batch_size']
                    print(f"     Batch {i+1}/{test_batches}: {batches_per_sec:.2f} batches/sec, {samples_per_sec:.1f} samples/sec")
            
            total_time = time.time() - start_time
            batches_per_sec = test_batches / total_time
            samples_per_sec = batches_per_sec * config['batch_size']
            
            # Estimate full epoch time
            estimated_epoch_time = len(train_loader) / batches_per_sec / 60  # minutes
            
            result = {
                'config': config['name'],
                'batches_per_sec': batches_per_sec,
                'samples_per_sec': samples_per_sec,
                'estimated_epoch_time_min': estimated_epoch_time
            }
            results.append(result)
            
            print(f"   ✅ Result: {batches_per_sec:.2f} batches/sec, {samples_per_sec:.1f} samples/sec")
            print(f"   ⏱️  Estimated full epoch time: {estimated_epoch_time:.1f} minutes")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            continue
    
    # Print summary
    print(f"\n📋 BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"{'Configuration':<20} {'Batches/sec':<12} {'Samples/sec':<12} {'Epoch(min)':<12}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['config']:<20} {result['batches_per_sec']:<12.2f} {result['samples_per_sec']:<12.1f} {result['estimated_epoch_time_min']:<12.1f}")
    
    # Find best configuration
    if results:
        best = max(results, key=lambda x: x['samples_per_sec'])
        print(f"\n🏆 BEST CONFIGURATION: {best['config']}")
        print(f"   Expected epoch time: {best['estimated_epoch_time_min']:.1f} minutes")
        
        # Compare to your current slowdown
        if best['estimated_epoch_time_min'] > 60:
            print(f"\n⚠️  STILL TOO SLOW! Root causes to investigate:")
            print(f"   1. Disk I/O speed - check if using SSD vs HDD")
            print(f"   2. Image file format - JPEG decode overhead")
            print(f"   3. Network storage - local vs networked storage")
            print(f"   4. PIL resize overhead - consider pre-resized images")
            print(f"   5. Transform overhead - check augmentation pipeline")
        else:
            print(f"\n✅ This should give reasonable training speed!")


def benchmark_model_vs_data_pipeline():
    """Test whether bottleneck is in model or data pipeline."""
    
    print(f"\n🎯 MODEL vs DATA PIPELINE BOTTLENECK TEST")
    print("=" * 60)
    
    # Create simple synthetic data first
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 64
    
    print(f"Testing on device: {device}")
    
    # Test 1: Pure model performance with synthetic data
    print(f"\n1️⃣ Testing pure model performance (synthetic data)...")
    
    model = mc_resnet50(num_classes=1000, device=str(device))
    model.compile(
        optimizer='adamw',
        loss='cross_entropy',
        learning_rate=0.01
    )
    
    # Synthetic data
    synthetic_rgb = torch.randn(batch_size, 3, 224, 224, device=device)
    synthetic_brightness = torch.randn(batch_size, 1, 224, 224, device=device)
    synthetic_labels = torch.randint(0, 1000, (batch_size,), device=device)
    
    # Warm up
    for _ in range(5):
        outputs = model(synthetic_rgb, synthetic_brightness)
        loss = torch.nn.functional.cross_entropy(outputs, synthetic_labels)
        loss.backward()
        model.optimizer.step()
        model.optimizer.zero_grad()
    
    # Benchmark model speed
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    for i in range(50):
        outputs = model(synthetic_rgb, synthetic_brightness)
        loss = torch.nn.functional.cross_entropy(outputs, synthetic_labels)
        loss.backward()
        model.optimizer.step()
        model.optimizer.zero_grad()
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    model_time = time.time() - start_time
    
    model_batches_per_sec = 50 / model_time
    model_samples_per_sec = model_batches_per_sec * batch_size
    
    print(f"   Pure model speed: {model_batches_per_sec:.2f} batches/sec, {model_samples_per_sec:.1f} samples/sec")
    
    # Estimate how many samples we could process in 45 minutes with pure model
    pure_model_samples_45min = model_samples_per_sec * 45 * 60
    imagenet_samples = 1281167  # ImageNet training samples
    pure_model_epochs_45min = pure_model_samples_45min / imagenet_samples
    
    print(f"   Pure model could process {pure_model_epochs_45min:.2f} epochs in 45 minutes")
    
    if pure_model_epochs_45min < 1:
        print(f"   ❌ Model itself is too slow! Expected ~1 epoch in 45 min for A100")
    else:
        print(f"   ✅ Model speed is reasonable - bottleneck is in data pipeline!")


if __name__ == "__main__":
    # First test model performance alone
    benchmark_model_vs_data_pipeline()
    
    # Then test data loading configurations
    benchmark_data_loading_config()
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"1. Run this script to identify the exact bottleneck")
    print(f"2. If data loading is slow, optimize num_workers/batch_size")  
    print(f"3. If model is slow, we need to investigate MC-ResNet architecture")
    print(f"4. Consider pre-processing images to reduce I/O overhead")
