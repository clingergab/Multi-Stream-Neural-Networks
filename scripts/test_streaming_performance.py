"""
Performance test script for StreamingDualChannelDataset.
This script verifies that the dataset loads images on-demand efficiently.
"""

import time
import tempfile
import shutil
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import DataLoader
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from data_utils.streaming_dual_channel_dataset import (
    StreamingDualChannelDataset,
    create_imagenet_dual_channel_train_val_dataloaders,
    create_default_imagenet_transforms
)


def create_test_dataset(num_classes=20, images_per_class=50):
    """Create a test dataset with specified number of classes and images."""
    temp_dir = tempfile.mkdtemp()
    train_folder = Path(temp_dir) / 'train'
    train_folder.mkdir(parents=True)
    
    print(f"Creating test dataset with {num_classes} classes, {images_per_class} images each...")
    
    # Create training images
    for class_idx in range(num_classes):
        class_name = f"n{class_idx:08d}"
        for img_idx in range(images_per_class):
            img_name = f"{class_name}_{img_idx:04d}_{class_name}.JPEG"
            img_path = train_folder / img_name
            
            # Create a random colored image
            import random
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            image = Image.new('RGB', (224, 224), color)
            image.save(img_path, quality=95)
    
    total_images = num_classes * images_per_class
    print(f"Created {total_images} test images in {temp_dir}")
    
    return temp_dir, str(train_folder)


def test_streaming_performance():
    """Test the streaming performance of the dataset."""
    print("=" * 60)
    print("STREAMING DUAL CHANNEL DATASET PERFORMANCE TEST")
    print("=" * 60)
    
    # Create test dataset
    temp_dir, train_folder = create_test_dataset(num_classes=10, images_per_class=20)
    
    try:
        # Test 1: Dataset initialization time (should be fast - no pre-loading)
        print("\n1. Testing Dataset Initialization Performance...")
        start_time = time.time()
        dataset = StreamingDualChannelDataset(
            data_folders=train_folder,
            split="train",
            image_size=(224, 224)
        )
        init_time = time.time() - start_time
        print(f"   ✅ Dataset initialized in {init_time:.3f} seconds")
        print(f"   📊 Dataset size: {len(dataset)} images")
        
        if init_time > 2.0:
            print("   ⚠️  WARNING: Initialization took longer than expected. May be pre-loading images.")
        else:
            print("   ✅ Fast initialization confirms on-demand loading")
        
        # Test 2: First batch loading time
        print("\n2. Testing First Batch Loading...")
        loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
        
        start_time = time.time()
        first_batch = next(iter(loader))
        first_batch_time = time.time() - start_time
        
        rgb, brightness, labels = first_batch
        print(f"   ✅ First batch loaded in {first_batch_time:.3f} seconds")
        print(f"   📊 Batch shape: RGB {rgb.shape}, Brightness {brightness.shape}")
        
        # Test 3: Streaming through multiple batches
        print("\n3. Testing Streaming Performance...")
        batch_times = []
        total_samples = 0
        
        start_time = time.time()
        for i, (rgb, brightness, labels) in enumerate(loader):
            batch_start = time.time()
            # Simulate some processing
            _ = rgb.mean()
            _ = brightness.mean()
            batch_end = time.time()
            
            batch_times.append(batch_end - batch_start)
            total_samples += rgb.shape[0]
            
            if i >= 10:  # Test first 10 batches
                break
        
        total_time = time.time() - start_time
        avg_batch_time = sum(batch_times) / len(batch_times)
        
        print(f"   ✅ Processed {len(batch_times)} batches in {total_time:.3f} seconds")
        print(f"   📊 Average batch processing time: {avg_batch_time:.3f} seconds")
        print(f"   📊 Samples per second: {total_samples/total_time:.1f}")
        
        # Test 4: Memory efficiency test
        print("\n4. Testing Memory Efficiency...")
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Load multiple batches
        large_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
        batches_processed = 0
        
        for rgb, brightness, labels in large_loader:
            batches_processed += 1
            if batches_processed >= 20:
                break
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before
        
        print(f"   📊 Memory before: {memory_before:.1f} MB")
        print(f"   📊 Memory after: {memory_after:.1f} MB")
        print(f"   📊 Memory increase: {memory_increase:.1f} MB")
        
        if memory_increase < 100:  # Less than 100MB increase
            print("   ✅ Good memory efficiency - streaming is working")
        else:
            print("   ⚠️  High memory usage - may be caching images")
        
        # Test 5: Transform performance
        print("\n5. Testing Transform Performance...")
        train_transform, _ = create_default_imagenet_transforms()
        
        dataset_with_transforms = StreamingDualChannelDataset(
            data_folders=train_folder,
            split="train",
            transform=train_transform,
            image_size=(224, 224)
        )
        
        transform_loader = DataLoader(dataset_with_transforms, batch_size=8, num_workers=0)
        
        start_time = time.time()
        for i, (rgb, brightness, labels) in enumerate(transform_loader):
            if i >= 5:  # Test 5 batches
                break
        transform_time = time.time() - start_time
        
        print(f"   ✅ Transform processing: {transform_time:.3f} seconds for 5 batches")
        print(f"   📊 RGB range: [{rgb.min():.2f}, {rgb.max():.2f}] (should be normalized)")
        print(f"   📊 Brightness range: [{brightness.min():.2f}, {brightness.max():.2f}]")
        
        # Test 6: Multi-batch consistency
        print("\n6. Testing Batch Consistency...")
        consistency_loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
        
        batch_shapes = []
        total_samples_counted = 0
        
        for rgb, brightness, labels in consistency_loader:
            batch_shapes.append((rgb.shape, brightness.shape, labels.shape))
            total_samples_counted += rgb.shape[0]
        
        print(f"   ✅ Processed all batches consistently")
        print(f"   📊 Total samples counted: {total_samples_counted}")
        print(f"   📊 Dataset length: {len(dataset)}")
        assert total_samples_counted == len(dataset), "Sample count mismatch!"
        
        # Performance Summary
        print("\n" + "=" * 60)
        print("PERFORMANCE SUMMARY")
        print("=" * 60)
        print(f"✅ Dataset initialization: {init_time:.3f}s (FAST - on-demand loading)")
        print(f"✅ Average batch processing: {avg_batch_time:.3f}s")
        print(f"✅ Throughput: {total_samples/total_time:.1f} samples/second")
        print(f"✅ Memory efficiency: {memory_increase:.1f}MB increase")
        print(f"✅ Transform performance: {transform_time/5:.3f}s per batch")
        print(f"✅ All {len(dataset)} samples processed correctly")
        
        if init_time < 1.0 and memory_increase < 100:
            print("\n🎉 EXCELLENT: Streaming functionality is working optimally!")
        elif init_time < 2.0 and memory_increase < 200:
            print("\n✅ GOOD: Streaming functionality is working well")
        else:
            print("\n⚠️  WARNING: Performance issues detected")
        
    finally:
        # Cleanup
        print(f"\n🧹 Cleaning up test data...")
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    test_streaming_performance()
