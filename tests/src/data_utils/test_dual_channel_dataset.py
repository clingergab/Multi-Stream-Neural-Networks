"""
Tests for the DualChannelDataset implementation.
"""

import pytest
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import sys
import os
import numpy as np

from data_utils.dual_channel_dataset import (
    DualChannelDataset, 
    create_dual_channel_dataloaders,
    create_dual_channel_dataloader
)


class TestDualChannelDataset:
    """Test cases for DualChannelDataset."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample RGB data and labels for testing."""
        batch_size = 10
        rgb_data = torch.randn(batch_size, 3, 32, 32)
        brightness_data = torch.randn(batch_size, 1, 32, 32)
        labels = torch.randint(0, 10, (batch_size,))
        return rgb_data, brightness_data, labels
    
    @pytest.fixture
    def structured_test_data(self):
        """Create test data with distinctive patterns for verifying transforms."""
        batch_size = 4
        height, width = 32, 32
        
        # Create test images with distinct patterns
        rgb_data = torch.zeros(batch_size, 3, height, width)
        labels = torch.arange(batch_size)
        
        # Create distinctive patterns for each image
        for i in range(batch_size):
            # Create vertical stripes in red channel
            rgb_data[i, 0, :, ::4] = 1.0
            # Create horizontal stripes in green channel  
            rgb_data[i, 1, ::4, :] = 1.0
            # Create diagonal pattern in blue channel
            for j in range(min(height, width)):
                if j < height and j < width:
                    rgb_data[i, 2, j, j] = 1.0
        
        return rgb_data, labels

    def test_init_with_brightness_data(self, sample_data):
        """Test initialization with provided brightness data."""
        rgb_data, brightness_data, labels = sample_data
        
        dataset = DualChannelDataset(
            rgb_data=rgb_data,
            labels=labels,
            brightness_data=brightness_data
        )
        
        assert len(dataset) == len(rgb_data)
        assert dataset.rgb_data.shape == rgb_data.shape
        assert dataset.brightness_data.shape == brightness_data.shape
        assert dataset.labels.shape == labels.shape
    
    def test_init_auto_brightness(self, sample_data):
        """Test initialization with auto-computed brightness."""
        rgb_data, _, labels = sample_data
        
        dataset = DualChannelDataset(
            rgb_data=rgb_data,
            labels=labels
            # brightness_data=None by default, auto-computed
        )
        
        assert len(dataset) == len(rgb_data)
        assert dataset.brightness_data is None  # Computed on-the-fly
    
    def test_init_validation_errors(self, sample_data):
        """Test validation errors during initialization."""
        rgb_data, brightness_data, labels = sample_data
        
        # Mismatched RGB and label lengths
        with pytest.raises(ValueError, match="RGB data length.*labels length"):
            DualChannelDataset(
                rgb_data=rgb_data,
                labels=labels[:5],  # Shorter labels
                brightness_data=brightness_data
            )
        
        # Mismatched brightness and label lengths
        with pytest.raises(ValueError, match="Brightness data length.*labels length"):
            DualChannelDataset(
                rgb_data=rgb_data,
                labels=labels,
                brightness_data=brightness_data[:5]  # Shorter brightness
            )
    
    def test_getitem_with_brightness_data(self, sample_data):
        """Test __getitem__ with provided brightness data."""
        rgb_data, brightness_data, labels = sample_data
        
        dataset = DualChannelDataset(
            rgb_data=rgb_data,
            labels=labels,
            brightness_data=brightness_data
        )
        
        rgb, brightness, label = dataset[0]
        
        assert rgb.shape == rgb_data[0].shape
        assert brightness.shape == brightness_data[0].shape
        assert label == labels[0]
        assert torch.allclose(rgb, rgb_data[0])
        assert torch.allclose(brightness, brightness_data[0])
    
    def test_getitem_auto_brightness(self, sample_data):
        """Test __getitem__ with auto-computed brightness."""
        rgb_data, _, labels = sample_data
        
        dataset = DualChannelDataset(
            rgb_data=rgb_data,
            labels=labels
            # brightness_data=None by default, auto-computed
        )
        
        rgb, brightness, label = dataset[0]
        
        assert rgb.shape == rgb_data[0].shape
        assert brightness.shape == (1, 32, 32)  # Single channel
        assert label == labels[0]
        assert torch.allclose(rgb, rgb_data[0])
    
    def test_transforms_applied(self, sample_data):
        """Test that transforms are properly applied."""
        rgb_data, brightness_data, labels = sample_data
        
        # Create simple transform that doubles values
        transform = transforms.Lambda(lambda x: x * 2)
        
        dataset = DualChannelDataset(
            rgb_data=rgb_data,
            labels=labels,
            brightness_data=brightness_data,
            transform=transform
        )
        
        rgb, brightness, label = dataset[0]
        
        # Check transform was applied to both
        assert torch.allclose(rgb, rgb_data[0] * 2)
        assert torch.allclose(brightness, brightness_data[0] * 2)
        assert label == labels[0]
    
    def test_shared_transform_consistency(self, sample_data):
        """Test that transforms are applied consistently to both channels."""
        rgb_data, brightness_data, labels = sample_data
        
        # Use horizontal flip as transform
        transform = transforms.RandomHorizontalFlip(p=1.0)  # Always flip
        
        dataset = DualChannelDataset(
            rgb_data=rgb_data,
            labels=labels,
            brightness_data=brightness_data,
            transform=transform
        )
        
        # Get the same item multiple times to check consistency
        # Note: Due to random seeding, each call should be consistent within itself
        rgb1, brightness1, _ = dataset[0]
        rgb2, brightness2, _ = dataset[0]
        
        # Both should be flipped (different from original)
        assert not torch.allclose(rgb1, rgb_data[0])
        assert not torch.allclose(brightness1, brightness_data[0])
        # Each call generates its own seed, so they might differ between calls
        # but the transform is applied consistently within each call
    
    def test_synchronized_geometric_transforms(self, structured_test_data):
        """Test that geometric transforms are properly synchronized between RGB and brightness."""
        rgb_data, labels = structured_test_data
        
        # Define transform with guaranteed effects
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0),  # Always flip
            transforms.RandomRotation(degrees=90),   # Large rotation
        ])
        
        dataset = DualChannelDataset(
            rgb_data=rgb_data,
            labels=labels,
            transform=transform
        )
        
        # Test multiple samples
        for idx in range(min(3, len(dataset))):
            rgb, brightness, label = dataset[idx]
            
            # Verify shapes
            assert rgb.shape == (3, 32, 32)
            assert brightness.shape == (1, 32, 32)
            
            # Transform should change the data
            original_rgb = rgb_data[idx]
            rgb_changed = not torch.allclose(rgb, original_rgb, atol=1e-6)
            assert rgb_changed, "RGB should be changed by transforms"
            
            # Brightness should be consistent with transformed RGB
            expected_brightness = dataset._rgb_converter.get_brightness(rgb)
            brightness_consistent = torch.allclose(brightness, expected_brightness, atol=1e-4)
            assert brightness_consistent, "Brightness should match transformed RGB"
    
    def test_random_transform_variability(self, structured_test_data):
        """Test that random transforms produce different results on multiple calls."""
        rgb_data, labels = structured_test_data
        
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
        ])
        
        dataset = DualChannelDataset(
            rgb_data=rgb_data,
            labels=labels,
            transform=transform
        )
        
        # Get the same sample multiple times
        results = []
        for _ in range(5):
            rgb, brightness, label = dataset[0]
            results.append((rgb.clone(), brightness.clone()))
        
        # Check that at least some results are different (random transforms working)
        all_same = True
        for i in range(1, len(results)):
            if not torch.allclose(results[0][0], results[i][0], atol=1e-6):
                all_same = False
                break
        
        # With random transforms, we should get some variation
        # Note: There's a small chance all could be the same, but very unlikely
        assert not all_same or transform is None, "Random transforms should produce some variation"
    
    def test_no_transform_consistency(self, sample_data):
        """Test dataset behavior without transforms."""
        rgb_data, _, labels = sample_data
        
        dataset = DualChannelDataset(
            rgb_data=rgb_data,
            labels=labels,
            transform=None
        )
        
        # Test that data is unchanged without transforms
        for idx in range(min(3, len(dataset))):
            rgb, brightness, label = dataset[idx]
            
            # RGB should be identical to original
            assert torch.allclose(rgb, rgb_data[idx]), "RGB should be unchanged without transforms"
            
            # Brightness should match computed brightness from original RGB
            expected_brightness = dataset._rgb_converter.get_brightness(rgb_data[idx])
            assert torch.allclose(brightness, expected_brightness), "Brightness should match computed value"
    
    def test_brightness_channel_consistency(self, structured_test_data):
        """Test that brightness channel always has correct properties."""
        rgb_data, labels = structured_test_data
        
        # Test with various transforms
        transforms_to_test = [
            None,
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.RandomRotation(degrees=45),
            transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
            ])
        ]
        
        for transform in transforms_to_test:
            dataset = DualChannelDataset(
                rgb_data=rgb_data,
                labels=labels,
                transform=transform
            )
            
            for idx in range(len(dataset)):
                rgb, brightness, label = dataset[idx]
                
                # Brightness should always be single channel
                assert brightness.shape[0] == 1, f"Brightness should have 1 channel, got {brightness.shape[0]}"
                
                # Brightness values should be in reasonable range (0-1 typically for normalized images)
                # Note: This depends on the input data range, but check it's not completely invalid
                assert torch.isfinite(brightness).all(), "Brightness should not contain NaN or inf values"
                
                # Brightness should be non-negative if RGB is non-negative
                if (rgb >= 0).all():
                    assert (brightness >= 0).all(), "Brightness should be non-negative for non-negative RGB"

    def test_dataloader_integration(self, sample_data):
        """Test integration with PyTorch DataLoader."""
        rgb_data, brightness_data, labels = sample_data
        
        dataset = DualChannelDataset(
            rgb_data=rgb_data,
            labels=labels,
            brightness_data=brightness_data
        )
        
        dataloader = DataLoader(dataset, batch_size=3, shuffle=False)
        
        # Test first batch
        batch_rgb, batch_brightness, batch_labels = next(iter(dataloader))
        
        assert batch_rgb.shape == (3, 3, 32, 32)
        assert batch_brightness.shape == (3, 1, 32, 32)
        assert batch_labels.shape == (3,)
        assert torch.allclose(batch_rgb, rgb_data[:3])
        assert torch.allclose(batch_brightness, brightness_data[:3])
        assert torch.allclose(batch_labels, labels[:3])
    
    def test_dataloader_with_transforms(self, sample_data):
        """Test DataLoader integration with transforms applied."""
        rgb_data, _, labels = sample_data
        
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
        ])
        
        dataset = DualChannelDataset(
            rgb_data=rgb_data,
            labels=labels,
            transform=transform
        )
        
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        # Test multiple batches
        batch_count = 0
        for rgb_batch, brightness_batch, label_batch in dataloader:
            # Verify shapes
            assert rgb_batch.shape[1:] == (3, 32, 32), f"RGB shape should be (*, 3, 32, 32), got {rgb_batch.shape}"
            assert brightness_batch.shape[1:] == (1, 32, 32), f"Brightness shape should be (*, 1, 32, 32), got {brightness_batch.shape}"
            
            # Verify batch dimensions match
            batch_size = rgb_batch.shape[0]
            assert brightness_batch.shape[0] == batch_size, "RGB and brightness batch sizes should match"
            assert label_batch.shape[0] == batch_size, "Labels batch size should match"
            
            # Verify brightness is single channel
            assert brightness_batch.shape[1] == 1, "Brightness should have exactly 1 channel"
            
            batch_count += 1
            if batch_count >= 2:  # Test a few batches
                break


class TestCreateDualChannelDataloaders:
    """Test cases for create_dual_channel_dataloaders convenience function."""
    
    @pytest.fixture
    def sample_train_val_data(self):
        """Create sample train/val data."""
        train_rgb = torch.randn(20, 3, 32, 32)
        train_labels = torch.randint(0, 10, (20,))
        val_rgb = torch.randn(10, 3, 32, 32)
        val_labels = torch.randint(0, 10, (10,))
        
        return train_rgb, train_labels, val_rgb, val_labels
    
    def test_create_dataloaders_auto_brightness(self, sample_train_val_data):
        """Test creating dataloaders with auto-computed brightness."""
        train_rgb, train_labels, val_rgb, val_labels = sample_train_val_data
        
        train_loader, val_loader = create_dual_channel_dataloaders(
            train_rgb=train_rgb,
            train_brightness=None,  # Auto-compute
            train_labels=train_labels,
            val_rgb=val_rgb,
            val_brightness=None,  # Auto-compute
            val_labels=val_labels,
            batch_size=4
        )
        
        # Test train loader
        train_batch = next(iter(train_loader))
        assert len(train_batch) == 3  # RGB, brightness, labels
        assert train_batch[0].shape[0] == 4  # Batch size
        assert train_batch[1].shape[1] == 1  # Brightness channel
        
        # Test val loader
        val_batch = next(iter(val_loader))
        assert len(val_batch) == 3
        assert val_batch[0].shape[0] <= 8  # Validation uses batch_size*2 (might be smaller for last batch)
        assert val_batch[1].shape[1] == 1  # Brightness channel
    
    def test_create_dataloaders_provided_brightness(self, sample_train_val_data):
        """Test creating dataloaders with provided brightness data."""
        train_rgb, train_labels, val_rgb, val_labels = sample_train_val_data
        
        train_brightness = torch.randn(20, 1, 32, 32)
        val_brightness = torch.randn(10, 1, 32, 32)
        
        train_loader, val_loader = create_dual_channel_dataloaders(
            train_rgb=train_rgb,
            train_labels=train_labels,
            val_rgb=val_rgb,
            val_labels=val_labels,
            train_brightness=train_brightness,
            val_brightness=val_brightness,
            batch_size=4
        )
        
        # Test that data flows through correctly
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        
        assert len(train_batch) == 3
        assert len(val_batch) == 3
        assert train_batch[0].shape[0] == 4
        assert val_batch[0].shape[0] <= 8  # Validation uses batch_size*2
    
    def test_create_dataloaders_no_augmentation(self, sample_train_val_data):
        """Test creating dataloaders without augmentation."""
        train_rgb, train_labels, val_rgb, val_labels = sample_train_val_data
        
        train_loader, val_loader = create_dual_channel_dataloaders(
            train_rgb=train_rgb,
            train_brightness=None,  # Auto-compute
            train_labels=train_labels,
            val_rgb=val_rgb,
            val_brightness=None,  # Auto-compute
            val_labels=val_labels,
            batch_size=4
            # No augmentation - train_transform and val_transform are None by default
        )
        
        # Should still work
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        
        assert len(train_batch) == 3
        assert len(val_batch) == 3
    
    def test_create_dataloaders_with_transforms(self, sample_train_val_data):
        """Test creating dataloaders with transforms."""
        train_rgb, train_labels, val_rgb, val_labels = sample_train_val_data
        
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
        ])
        
        val_transform = transforms.Compose([
            transforms.CenterCrop(28),  # Different from train transform
        ])
        
        train_loader, val_loader = create_dual_channel_dataloaders(
            train_rgb=train_rgb,
            train_brightness=None,  # Auto-compute
            train_labels=train_labels,
            val_rgb=val_rgb,
            val_brightness=None,  # Auto-compute
            val_labels=val_labels,
            batch_size=4,
            train_transform=train_transform,
            val_transform=val_transform
        )
        
        # Test train loader
        train_batch = next(iter(train_loader))
        assert len(train_batch) == 3
        assert train_batch[0].shape[0] == 4
        assert train_batch[1].shape[1] == 1
        
        # Test val loader (should have different size due to CenterCrop)
        val_batch = next(iter(val_loader))
        assert len(val_batch) == 3
        assert val_batch[0].shape[0] <= 8  # Validation uses batch_size*2
        assert val_batch[1].shape[1] == 1
        # Validate crop was applied
        assert val_batch[0].shape[2] == 28, "CenterCrop should reduce height to 28"
        assert val_batch[0].shape[3] == 28, "CenterCrop should reduce width to 28"
    
    def test_create_dataloaders_gpu_optimization_params(self, sample_train_val_data):
        """Test creating dataloaders with GPU optimization parameters."""
        train_rgb, train_labels, val_rgb, val_labels = sample_train_val_data
        
        train_loader, val_loader = create_dual_channel_dataloaders(
            train_rgb=train_rgb,
            train_brightness=None,  # Auto-compute
            train_labels=train_labels,
            val_rgb=val_rgb,
            val_brightness=None,  # Auto-compute
            val_labels=val_labels,
            batch_size=4,
            num_workers=6,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=3
        )
        
        # Test that GPU optimization parameters are applied
        assert train_loader.num_workers == 6
        assert train_loader.pin_memory == True
        assert train_loader.persistent_workers == True
        
        assert val_loader.num_workers == 6
        assert val_loader.pin_memory == True
        assert val_loader.persistent_workers == True
        
        # Test data still flows correctly
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        
        assert len(train_batch) == 3
        assert len(val_batch) == 3

    def test_create_dataloaders_no_workers_disables_persistent(self, sample_train_val_data):
        """Test that num_workers=0 properly disables persistent_workers."""
        train_rgb, train_labels, val_rgb, val_labels = sample_train_val_data
        
        train_loader, val_loader = create_dual_channel_dataloaders(
            train_rgb=train_rgb,
            train_brightness=None,  # Auto-compute
            train_labels=train_labels,
            val_rgb=val_rgb,
            val_brightness=None,  # Auto-compute
            val_labels=val_labels,
            batch_size=4,
            num_workers=0,
            persistent_workers=True  # Should be overridden to False
        )
        
        # persistent_workers should be disabled when num_workers=0
        assert train_loader.num_workers == 0
        assert train_loader.persistent_workers == False
        
        assert val_loader.num_workers == 0
        assert val_loader.persistent_workers == False

    def test_create_dataloaders_custom_dataloader_params(self, sample_train_val_data):
        """Test creating dataloaders with custom DataLoader parameters."""
        train_rgb, train_labels, val_rgb, val_labels = sample_train_val_data
        
        train_loader, val_loader = create_dual_channel_dataloaders(
            train_rgb=train_rgb,
            train_brightness=None,  # Auto-compute
            train_labels=train_labels,
            val_rgb=val_rgb,
            val_brightness=None,  # Auto-compute
            val_labels=val_labels,
            batch_size=2,
            num_workers=2,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=1
        )
        
        # Check all custom parameters
        assert train_loader.batch_size == 2
        assert train_loader.num_workers == 2
        assert train_loader.pin_memory == False
        assert train_loader.persistent_workers == False
        
        assert val_loader.batch_size == 4  # Validation uses batch_size*2
        assert val_loader.num_workers == 2
        assert val_loader.pin_memory == False
        assert val_loader.persistent_workers == False

    def test_create_dataloaders_different_batch_sizes(self, sample_train_val_data):
        """
        Test that training and validation dataloaders use different batch sizes.
        
        This is a best practice: validation can use larger batch sizes than training
        because we don't compute gradients during validation, allowing more samples
        to fit in memory and faster validation processing.
        
        Expected behavior:
        - Training dataloader uses the specified batch_size
        - Validation dataloader uses batch_size * 2
        """
        train_rgb, train_labels, val_rgb, val_labels = sample_train_val_data
        
        batch_size = 8  # Use a specific batch size for clear testing
        train_loader, val_loader = create_dual_channel_dataloaders(
            train_rgb=train_rgb,
            train_brightness=None,  # Auto-compute
            train_labels=train_labels,
            val_rgb=val_rgb,
            val_brightness=None,  # Auto-compute
            val_labels=val_labels,
            batch_size=batch_size
        )
        
        # Verify that training uses the specified batch size
        assert train_loader.batch_size == batch_size, f"Training batch size should be {batch_size}, got {train_loader.batch_size}"
        
        # Verify that validation uses 2x the batch size
        expected_val_batch_size = batch_size * 2
        assert val_loader.batch_size == expected_val_batch_size, f"Validation batch size should be {expected_val_batch_size}, got {val_loader.batch_size}"
        
        # Verify actual batch loading behavior
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        
        # Training batch should match the specified batch size
        assert train_batch[0].shape[0] == batch_size, f"Actual training batch size is {train_batch[0].shape[0]}, expected {batch_size}"
        
        # Validation batch should be larger (up to 2x, depending on dataset size)
        assert val_batch[0].shape[0] <= expected_val_batch_size, f"Validation batch size {val_batch[0].shape[0]} exceeds expected max {expected_val_batch_size}"
        assert val_batch[0].shape[0] > batch_size, f"Validation batch size {val_batch[0].shape[0]} should be larger than training batch size {batch_size}"


class TestCreateDualChannelDataloader:
    """Test cases for create_dual_channel_dataloader convenience function."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for single dataloader."""
        rgb_data = torch.randn(15, 3, 32, 32)
        brightness_data = torch.randn(15, 1, 32, 32)
        labels = torch.randint(0, 10, (15,))
        return rgb_data, brightness_data, labels
    
    def test_create_single_dataloader_basic(self, sample_data):
        """Test creating a single dataloader with basic parameters."""
        rgb_data, brightness_data, labels = sample_data
        
        dataloader = create_dual_channel_dataloader(
            rgb_data=rgb_data,
            brightness_data=brightness_data,
            labels=labels,
            batch_size=5
        )
        
        assert isinstance(dataloader, DataLoader)
        assert dataloader.batch_size == 5
        
        # Test data flow
        batch = next(iter(dataloader))
        assert len(batch) == 3  # RGB, brightness, labels
        assert batch[0].shape[0] == 5  # Batch size
        assert batch[0].shape[1] == 3  # RGB channels
        assert batch[1].shape[1] == 1  # Brightness channel
        assert batch[2].shape[0] == 5  # Labels

    def test_create_single_dataloader_no_shuffle(self, sample_data):
        """Test creating a single dataloader with shuffle=False."""
        rgb_data, brightness_data, labels = sample_data
        
        dataloader = create_dual_channel_dataloader(
            rgb_data=rgb_data,
            brightness_data=brightness_data,
            labels=labels,
            batch_size=5,
            shuffle=False
        )
        
        # Should still work correctly
        batch = next(iter(dataloader))
        assert len(batch) == 3
        assert batch[0].shape[0] == 5

    def test_create_single_dataloader_gpu_params(self, sample_data):
        """Test creating a single dataloader with GPU optimization parameters."""
        rgb_data, brightness_data, labels = sample_data
        
        dataloader = create_dual_channel_dataloader(
            rgb_data=rgb_data,
            brightness_data=brightness_data,
            labels=labels,
            batch_size=3,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
        
        # Check GPU optimization parameters
        assert dataloader.batch_size == 3
        assert dataloader.num_workers == 4
        assert dataloader.pin_memory == True
        assert dataloader.persistent_workers == True
        
        # Test data flow
        batch = next(iter(dataloader))
        assert len(batch) == 3
        assert batch[0].shape[0] == 3

    def test_create_single_dataloader_with_transform(self, sample_data):
        """Test creating a single dataloader with transforms."""
        rgb_data, brightness_data, labels = sample_data
        
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
        ])
        
        dataloader = create_dual_channel_dataloader(
            rgb_data=rgb_data,
            brightness_data=brightness_data,
            labels=labels,
            batch_size=5,
            transform=transform
        )
        
        # Should work with transforms
        batch = next(iter(dataloader))
        assert len(batch) == 3
        assert batch[0].shape[0] == 5

    def test_create_single_dataloader_zero_workers(self, sample_data):
        """Test that num_workers=0 properly disables persistent_workers in single dataloader."""
        rgb_data, brightness_data, labels = sample_data
        
        dataloader = create_dual_channel_dataloader(
            rgb_data=rgb_data,
            brightness_data=brightness_data,
            labels=labels,
            batch_size=5,
            num_workers=0,
            persistent_workers=True  # Should be overridden to False
        )
        
        # persistent_workers should be disabled when num_workers=0
        assert dataloader.num_workers == 0
        assert dataloader.persistent_workers == False
        
        # Data should still flow correctly
        batch = next(iter(dataloader))
        assert len(batch) == 3

    def test_create_single_dataloader_auto_brightness(self):
        """Test creating a single dataloader that auto-computes brightness."""
        rgb_data = torch.randn(10, 3, 32, 32)
        labels = torch.randint(0, 10, (10,))
        
        # Pass None for brightness_data to test auto-computation
        dataloader = create_dual_channel_dataloader(
            rgb_data=rgb_data,
            brightness_data=None,
            labels=labels,
            batch_size=5
        )
        
        # Should work and auto-compute brightness
        batch = next(iter(dataloader))
        assert len(batch) == 3
        assert batch[0].shape[1] == 3  # RGB channels
        assert batch[1].shape[1] == 1  # Auto-computed brightness channel


if __name__ == "__main__":
    pytest.main([__file__])
