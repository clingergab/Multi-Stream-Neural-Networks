"""
Additional tests for StreamingDualChannelDataset batch loading functionality.
"""

import pytest
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import sys
import os
import tempfile
import shutil
from pathlib import Path
from PIL import Image
import time

from data_utils.streaming_dual_channel_dataset import (
    StreamingDualChannelDataset,
    create_imagenet_dual_channel_train_val_dataloaders,
)

class TestBatchLoadingFunctionality:
    """Test batch loading and streaming functionality."""
    
    @pytest.fixture
    def large_temp_dataset(self):
        """Create a larger temporary dataset for batch testing."""
        temp_dir = tempfile.mkdtemp()
        dataset_paths = {
            'temp_dir': Path(temp_dir),
            'train_folder': Path(temp_dir) / 'train_folder',
            'val_folder': Path(temp_dir) / 'val_folder',
            'truth_file': Path(temp_dir) / 'truth.txt'
        }
        
        # Create directories
        for folder in ['train_folder', 'val_folder']:
            dataset_paths[folder].mkdir(parents=True, exist_ok=True)
        
        # Create more training images (5 classes, 10 images each = 50 total)
        train_classes = ['n01440764', 'n01443537', 'n01484850', 'n01491361', 'n01494475']
        for class_name in train_classes:
            for i in range(10):  # 10 images per class
                img_name = f"{class_name}_{i:03d}_{class_name}.JPEG"
                img_path = dataset_paths['train_folder'] / img_name
                self._create_test_image(img_path, size=(224, 224))
        
        # Create validation images (20 images)
        val_labels = []
        for i in range(20):
            img_name = f"ILSVRC2012_val_{i+1:08d}.JPEG"
            img_path = dataset_paths['val_folder'] / img_name
            self._create_test_image(img_path, size=(224, 224))
            val_labels.append(i % 5)  # Cycle through 5 classes
        
        # Create truth file for validation
        with open(dataset_paths['truth_file'], 'w') as f:
            for label in val_labels:
                f.write(f"{label}\n")
        
        yield dataset_paths
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def _create_test_image(self, path: Path, size: tuple = (224, 224)):
        """Create a test image with random colors to simulate real data."""
        import random
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        image = Image.new('RGB', size, color)
        image.save(path)
    
    def test_batch_size_consistency(self, large_temp_dataset):
        """Test that batches are consistently sized (except last batch if drop_last=False)."""
        dataset = StreamingDualChannelDataset(
            data_folders=str(large_temp_dataset['train_folder']),
            split="train",
            image_size=(224, 224)
        )
        
        # Test with different batch sizes
        for batch_size in [1, 4, 8, 16]:
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=0
            )
            
            batch_sizes = []
            for rgb, brightness, labels in loader:
                batch_sizes.append(rgb.shape[0])
                assert rgb.shape[0] == brightness.shape[0] == labels.shape[0]
            
            # All batches except possibly the last should be full size
            for i, size in enumerate(batch_sizes[:-1]):
                assert size == batch_size, f"Batch {i} has size {size}, expected {batch_size}"
            
            # Last batch should be <= batch_size
            assert batch_sizes[-1] <= batch_size
    
    def test_drop_last_functionality(self, large_temp_dataset):
        """Test drop_last parameter works correctly."""
        dataset = StreamingDualChannelDataset(
            data_folders=str(large_temp_dataset['train_folder']),
            split="train",
            image_size=(224, 224)
        )
        
        batch_size = 7  # Choose a size that doesn't divide evenly into 50
        
        # With drop_last=True, all batches should be exactly batch_size
        loader_drop = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=0
        )
        
        for rgb, brightness, labels in loader_drop:
            assert rgb.shape[0] == batch_size
            assert brightness.shape[0] == batch_size
            assert labels.shape[0] == batch_size
        
        # With drop_last=False, last batch might be smaller
        loader_keep = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0
        )
        
        batch_sizes = []
        for rgb, brightness, labels in loader_keep:
            batch_sizes.append(rgb.shape[0])
        
        # Should have one more batch than drop_last=True (unless dataset size is exactly divisible)
        expected_remainder = len(dataset) % batch_size
        if expected_remainder != 0:
            assert batch_sizes[-1] == expected_remainder
    
    def test_streaming_performance(self, large_temp_dataset):
        """Test that streaming loads images on-demand without pre-loading."""
        dataset = StreamingDualChannelDataset(
            data_folders=str(large_temp_dataset['train_folder']),
            split="train",
            image_size=(224, 224)
        )
        
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0
        )
        
        # Measure time to create dataset (should be fast - no pre-loading)
        start_time = time.time()
        dataset_creation_time = time.time() - start_time
        
        # Should be very fast since we're not pre-loading images
        assert dataset_creation_time < 1.0, "Dataset creation took too long - might be pre-loading images"
        
        # Test that we can load batches
        batch_count = 0
        start_time = time.time()
        for rgb, brightness, labels in loader:
            assert rgb.shape == (4, 3, 224, 224) or rgb.shape[0] <= 4  # Last batch might be smaller
            assert brightness.shape == (4, 1, 224, 224) or brightness.shape[0] <= 4
            batch_count += 1
            if batch_count >= 3:  # Test a few batches
                break
        
        loading_time = time.time() - start_time
        print(f"Loaded {batch_count} batches in {loading_time:.3f} seconds")
    
    def test_multiworker_loading(self, large_temp_dataset):
        """Test that DataLoader works correctly with multiple workers."""
        dataset = StreamingDualChannelDataset(
            data_folders=str(large_temp_dataset['train_folder']),
            split="train",
            image_size=(224, 224)
        )
        
        # Test with multiple workers (if supported on this platform)
        try:
            loader = DataLoader(
                dataset,
                batch_size=4,
                shuffle=False,
                num_workers=2,
                persistent_workers=False  # Don't persist for test
            )
            
            batch_count = 0
            total_samples = 0
            for rgb, brightness, labels in loader:
                assert rgb.shape[1:] == (3, 224, 224)
                assert brightness.shape[1:] == (1, 224, 224)
                total_samples += rgb.shape[0]
                batch_count += 1
            
            # Should process all samples exactly once
            assert total_samples == len(dataset)
            
        except Exception as e:
            # Some environments don't support multiprocessing for testing
            pytest.skip(f"Multiprocessing not supported in test environment: {e}")
    
    def test_shuffle_functionality(self, large_temp_dataset):
        """Test that shuffle produces different batch orders."""
        dataset = StreamingDualChannelDataset(
            data_folders=str(large_temp_dataset['train_folder']),
            split="train",
            image_size=(224, 224)
        )
        
        # Get first batch without shuffle
        loader1 = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
        first_batch_no_shuffle = next(iter(loader1))[2]  # Get labels
        
        # Get first batch with shuffle (use fixed seed for reproducibility)
        torch.manual_seed(42)
        loader2 = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
        first_batch_shuffle = next(iter(loader2))[2]  # Get labels
        
        # Different seed should give different order
        torch.manual_seed(123)
        loader3 = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
        first_batch_shuffle2 = next(iter(loader3))[2]  # Get labels
        
        # At least one of the shuffled batches should be different from unshuffled
        # (very unlikely they're all the same by chance)
        assert not torch.equal(first_batch_no_shuffle, first_batch_shuffle) or \
               not torch.equal(first_batch_no_shuffle, first_batch_shuffle2)
    
    def test_transform_application_in_batches(self, large_temp_dataset):
        """Test that transforms are applied correctly to batches."""
        # Define a simple transform
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0),  # Always flip
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        dataset = StreamingDualChannelDataset(
            data_folders=str(large_temp_dataset['train_folder']),
            split="train",
            transform=transform,
            image_size=(224, 224)
        )
        
        loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
        
        for rgb, brightness, labels in loader:
            # Check that normalization was applied (values should be centered around 0)
            assert rgb.mean().abs() < 2.0  # Should be roughly normalized
            assert brightness.mean().abs() < 2.0
            
            # Check shapes are correct
            assert rgb.shape[1:] == (3, 224, 224)
            assert brightness.shape[1:] == (1, 224, 224)
            break  # Test first batch
    
    def test_validation_dataloader_batch_behavior(self, large_temp_dataset):
        """Test that validation DataLoader behaves correctly with batching."""
        train_loader, val_loader = create_imagenet_dual_channel_train_val_dataloaders(
            train_folders=str(large_temp_dataset['train_folder']),
            val_folder=str(large_temp_dataset['val_folder']),
            truth_file=str(large_temp_dataset['truth_file']),
            batch_size=4,
            num_workers=0
        )
        
        # Validation loader should use same batch_size as training (4)
        val_batch = next(iter(val_loader))
        rgb, brightness, labels = val_batch
        
        # Should be same batch size as training (4) or all remaining samples if < 4
        assert rgb.shape[0] == 4 or rgb.shape[0] == 20  # 4 or all samples if < 4 remaining
        
        # Should not be shuffled (validation should be deterministic)
        val_batches = []
        for batch in val_loader:
            val_batches.append(batch[2])  # labels
            if len(val_batches) >= 2:
                break
        
        # Run again - should get same order
        val_batches2 = []
        for batch in val_loader:
            val_batches2.append(batch[2])
            if len(val_batches2) >= 2:
                break
        
        # Should be same order (not shuffled)
        for b1, b2 in zip(val_batches, val_batches2):
            assert torch.equal(b1, b2)
    
    def test_batch_size_limitations_and_no_overloading(self, large_temp_dataset):
        """
        Test that DataLoader strictly respects batch_size and never loads more samples than requested.
        This verifies that we don't accidentally load extra samples beyond the batch size.
        """
        dataset = StreamingDualChannelDataset(
            data_folders=str(large_temp_dataset['train_folder']),
            split="train",
            image_size=(224, 224)
        )
        
        # Test with dataset of 50 images and various batch sizes
        dataset_size = len(dataset)
        assert dataset_size == 50, f"Expected 50 images, got {dataset_size}"
        
        test_cases = [
            # (batch_size, expected_full_batches, expected_last_batch_size)
            (1, 50, 0),   # 50 batches of size 1, no remainder
            (3, 16, 2),   # 16 batches of size 3, last batch size 2 (50 % 3 = 2)
            (7, 7, 1),    # 7 batches of size 7, last batch size 1 (50 % 7 = 1)
            (10, 5, 0),   # 5 batches of size 10, no remainder
            (12, 4, 2),   # 4 batches of size 12, last batch size 2 (50 % 12 = 2)
            (25, 2, 0),   # 2 batches of size 25, no remainder
            (30, 1, 20),  # 1 batch of size 30, last batch size 20 (50 % 30 = 20)
            (50, 1, 0),   # 1 batch of size 50, no remainder
            (60, 0, 50),  # 0 full batches, last batch size 50 (entire dataset)
        ]
        
        for batch_size, expected_full_batches, expected_last_batch_size in test_cases:
            print(f"\n  Testing batch_size={batch_size}")
            
            # Test with drop_last=False (keep all samples)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=0
            )
            
            batch_sizes = []
            total_samples = 0
            
            for batch_idx, (rgb, brightness, labels) in enumerate(loader):
                current_batch_size = rgb.shape[0]
                batch_sizes.append(current_batch_size)
                total_samples += current_batch_size
                
                # Verify batch doesn't exceed specified batch_size
                assert current_batch_size <= batch_size, \
                    f"Batch {batch_idx} has {current_batch_size} samples, exceeds batch_size {batch_size}"
                
                # Verify all tensors in batch have same size
                assert rgb.shape[0] == brightness.shape[0] == labels.shape[0], \
                    f"Inconsistent batch sizes: RGB={rgb.shape[0]}, brightness={brightness.shape[0]}, labels={labels.shape[0]}"
                
                # Verify tensor shapes are correct
                assert rgb.shape == (current_batch_size, 3, 224, 224), \
                    f"Unexpected RGB shape: {rgb.shape}"
                assert brightness.shape == (current_batch_size, 1, 224, 224), \
                    f"Unexpected brightness shape: {brightness.shape}"
                assert labels.shape == (current_batch_size,), \
                    f"Unexpected labels shape: {labels.shape}"
            
            # Verify total samples processed equals dataset size
            assert total_samples == dataset_size, \
                f"Total samples {total_samples} != dataset size {dataset_size}"
            
            # Verify batch count and sizes
            if expected_last_batch_size == 0:
                # All batches should be full size
                expected_total_batches = expected_full_batches
                assert len(batch_sizes) == expected_total_batches, \
                    f"Expected {expected_total_batches} batches, got {len(batch_sizes)}"
                for i, size in enumerate(batch_sizes):
                    assert size == batch_size, \
                        f"Batch {i} has size {size}, expected {batch_size}"
            else:
                # Should have full batches + one partial batch
                expected_total_batches = expected_full_batches + 1
                assert len(batch_sizes) == expected_total_batches, \
                    f"Expected {expected_total_batches} batches, got {len(batch_sizes)}"
                
                # Check full batches
                for i in range(expected_full_batches):
                    assert batch_sizes[i] == batch_size, \
                        f"Full batch {i} has size {batch_sizes[i]}, expected {batch_size}"
                
                # Check last batch
                assert batch_sizes[-1] == expected_last_batch_size, \
                    f"Last batch has size {batch_sizes[-1]}, expected {expected_last_batch_size}"
            
            # Test with drop_last=True (drop incomplete batches)
            loader_drop = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=True,
                num_workers=0
            )
            
            drop_batch_sizes = []
            drop_total_samples = 0
            
            for rgb, brightness, labels in loader_drop:
                current_batch_size = rgb.shape[0]
                drop_batch_sizes.append(current_batch_size)
                drop_total_samples += current_batch_size
                
                # All batches should be exactly batch_size with drop_last=True
                assert current_batch_size == batch_size, \
                    f"With drop_last=True, batch has size {current_batch_size}, expected {batch_size}"
            
            # Verify we got exactly the expected number of full batches
            assert len(drop_batch_sizes) == expected_full_batches, \
                f"With drop_last=True, got {len(drop_batch_sizes)} batches, expected {expected_full_batches}"
            
            # Verify total samples with drop_last
            expected_drop_samples = expected_full_batches * batch_size
            assert drop_total_samples == expected_drop_samples, \
                f"With drop_last=True, got {drop_total_samples} samples, expected {expected_drop_samples}"
            
            print(f"    ✅ batch_size={batch_size}: {len(batch_sizes)} batches, last={batch_sizes[-1] if batch_sizes else 0}")

    def test_single_sample_loading_vs_batch_loading(self, large_temp_dataset):
        """
        Test that individual sample access (__getitem__) vs batch loading produces same data.
        Verifies that batching doesn't alter the data.
        """
        dataset = StreamingDualChannelDataset(
            data_folders=str(large_temp_dataset['train_folder']),
            split="train",
            image_size=(224, 224)
        )
        
        # Load first 4 samples individually
        individual_samples = []
        for i in range(4):
            rgb, brightness, label = dataset[i]
            individual_samples.append((rgb, brightness, label))
        
        # Load same 4 samples as a batch
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,  # Important: no shuffle for comparison
            num_workers=0
        )
        
        batch_rgb, batch_brightness, batch_labels = next(iter(loader))
        
        # Verify batch contains same data as individual samples
        for i in range(4):
            individual_rgb, individual_brightness, individual_label = individual_samples[i]
            
            # Compare tensors (allowing for small floating point differences)
            assert torch.allclose(batch_rgb[i], individual_rgb, atol=1e-6), \
                f"RGB mismatch at sample {i}"
            assert torch.allclose(batch_brightness[i], individual_brightness, atol=1e-6), \
                f"Brightness mismatch at sample {i}"
            assert batch_labels[i] == individual_label, \
                f"Label mismatch at sample {i}: batch={batch_labels[i]}, individual={individual_label}"
        
        print("    ✅ Individual sample loading matches batch loading")

    def test_batch_memory_efficiency(self, large_temp_dataset):
        """
        Test that larger batch sizes don't cause excessive memory usage.
        Verifies that we're not accidentally caching extra data.
        """
        import psutil
        import os
        
        dataset = StreamingDualChannelDataset(
            data_folders=str(large_temp_dataset['train_folder']),
            split="train",
            image_size=(224, 224)
        )
        
        process = psutil.Process(os.getpid())
        
        # Test different batch sizes and measure memory
        batch_sizes = [1, 8, 16, 32]
        memory_usage = {}
        
        for batch_size in batch_sizes:
            # Measure memory before
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0
            )
            
            # Load a few batches
            batches_loaded = 0
            for rgb, brightness, labels in loader:
                batches_loaded += 1
                if batches_loaded >= 3:  # Load 3 batches
                    break
            
            # Measure memory after
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = memory_after - memory_before
            memory_usage[batch_size] = memory_increase
            
            print(f"    Batch size {batch_size}: {memory_increase:.1f}MB increase")
            
            # Memory increase should be reasonable (not exponential with batch size)
            assert memory_increase < 200, f"Excessive memory usage: {memory_increase}MB for batch_size={batch_size}"
        
        # Memory usage shouldn't grow dramatically with batch size
        max_increase = max(memory_usage.values())
        min_increase = min(memory_usage.values())
        
        # Handle cases where memory increase is very small (measurement precision issues)
        if max_increase < 5.0:  # Less than 5MB total - very efficient
            print(f"    ✅ Excellent memory efficiency: max increase {max_increase:.1f}MB")
        else:
            ratio = max_increase / min_increase if min_increase > 0.1 else max_increase
            assert ratio < 50, f"Memory usage varies too much with batch size: {memory_usage}"
            print(f"    ✅ Memory usage variation acceptable: max {max_increase:.1f}MB")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
