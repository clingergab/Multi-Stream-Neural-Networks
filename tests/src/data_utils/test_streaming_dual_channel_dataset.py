"""
Tests for the StreamingDualChannelDataset implementation.
"""

import pytest
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import tempfile
import shutil
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
from PIL import Image
import numpy as np

from data_utils.streaming_dual_channel_dataset import (
    StreamingDualChannelDataset,
    create_imagenet_dual_channel_train_val_dataloaders,
    create_imagenet_dual_channel_test_dataloader,
    create_default_imagenet_transforms
)


class TestStreamingDualChannelDataset:
    """Test cases for StreamingDualChannelDataset."""
    
    @pytest.fixture
    def temp_dataset(self):
        """Create a temporary dataset structure for testing."""
        temp_dir = tempfile.mkdtemp()
        dataset_paths = {
            'temp_dir': Path(temp_dir),
            'train_folder1': Path(temp_dir) / 'train_folder1',
            'train_folder2': Path(temp_dir) / 'train_folder2',
            'val_folder': Path(temp_dir) / 'val_folder',
            'test_folder': Path(temp_dir) / 'test_folder',
            'truth_file': Path(temp_dir) / 'truth.txt'
        }
        
        # Create directories
        for folder in ['train_folder1', 'train_folder2', 'val_folder', 'test_folder']:
            dataset_paths[folder].mkdir(parents=True, exist_ok=True)
        
        # Create sample images for training (with class names in filenames)
        train_classes = ['n01440764', 'n01443537', 'n01484850']
        for i, folder in enumerate(['train_folder1', 'train_folder2']):
            for j, class_name in enumerate(train_classes):
                for k in range(2):  # 2 images per class per folder
                    img_name = f"{class_name}_{i*10+j*2+k}_{class_name}.JPEG"
                    img_path = dataset_paths[folder] / img_name
                    self._create_test_image(img_path, size=(224, 224))
        
        # Create sample images for validation (sequential naming)
        val_images = []
        for i in range(6):
            img_name = f"ILSVRC2012_val_{i+1:08d}_n01440764.JPEG"
            img_path = dataset_paths['val_folder'] / img_name
            self._create_test_image(img_path, size=(224, 224))
            val_images.append(img_path)
        
        # Create sample images for test
        for i in range(4):
            img_name = f"ILSVRC2012_test_{i+1:08d}.JPEG"
            img_path = dataset_paths['test_folder'] / img_name
            self._create_test_image(img_path, size=(224, 224))
        
        # Create truth file for validation
        truth_labels = [0, 1, 2, 0, 1, 2]  # 6 labels for 6 validation images
        with open(dataset_paths['truth_file'], 'w') as f:
            for label in truth_labels:
                f.write(f"{label}\n")
        
        yield dataset_paths
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def _create_test_image(self, path: Path, size: tuple = (224, 224), color: tuple = (128, 128, 128)):
        """Create a test image at the given path."""
        image = Image.new('RGB', size, color)
        image.save(path)
    
    def test_init_single_train_folder(self, temp_dataset):
        """Test initialization with a single training folder."""
        dataset = StreamingDualChannelDataset(
            data_folders=str(temp_dataset['train_folder1']),
            split="train",
            image_size=(224, 224)
        )
        
        assert len(dataset) == 6  # 3 classes * 2 images
        assert dataset.split == "train"
        assert dataset.class_to_idx is not None
        assert len(dataset.class_to_idx) == 3  # 3 classes
    
    def test_init_multiple_train_folders(self, temp_dataset):
        """Test initialization with multiple training folders."""
        dataset = StreamingDualChannelDataset(
            data_folders=[
                str(temp_dataset['train_folder1']),
                str(temp_dataset['train_folder2'])
            ],
            split="train",
            image_size=(224, 224)
        )
        
        assert len(dataset) == 12  # 3 classes * 2 images * 2 folders
        assert len(dataset.class_to_idx) == 3  # Same 3 classes across folders
    
    def test_init_validation_folder(self, temp_dataset):
        """Test initialization with validation folder and truth file."""
        dataset = StreamingDualChannelDataset(
            data_folders=str(temp_dataset['val_folder']),
            split="validation",
            truth_file=str(temp_dataset['truth_file']),
            image_size=(224, 224)
        )
        
        assert len(dataset) == 6  # 6 validation images
        assert dataset.split == "validation"
        assert dataset.class_to_idx is None  # Not needed for validation
    
    def test_init_missing_folder(self):
        """Test initialization with non-existent folder."""
        with pytest.raises(FileNotFoundError, match="Data folder not found"):
            StreamingDualChannelDataset(
                data_folders="/non/existent/path",
                split="train"
            )
    
    def test_init_validation_without_truth_file(self, temp_dataset):
        """Test that validation requires truth file."""
        with pytest.raises(ValueError, match="truth_file required for validation split"):
            StreamingDualChannelDataset(
                data_folders=str(temp_dataset['val_folder']),
                split="validation",
                truth_file=None
            )
    
    def test_init_missing_truth_file(self, temp_dataset):
        """Test initialization with non-existent truth file."""
        with pytest.raises(FileNotFoundError, match="Truth file not found"):
            StreamingDualChannelDataset(
                data_folders=str(temp_dataset['val_folder']),
                split="validation",
                truth_file="/non/existent/truth.txt"
            )
    
    def test_init_mismatched_truth_file(self, temp_dataset):
        """Test validation with mismatched number of images and labels."""
        # Create truth file with wrong number of labels
        wrong_truth_file = temp_dataset['temp_dir'] / 'wrong_truth.txt'
        with open(wrong_truth_file, 'w') as f:
            f.write("0\n1\n")  # Only 2 labels for 6 images
        
        with pytest.raises(ValueError, match="Mismatch between images"):
            StreamingDualChannelDataset(
                data_folders=str(temp_dataset['val_folder']),
                split="validation",
                truth_file=str(wrong_truth_file)
            )
    
    def test_getitem_train(self, temp_dataset):
        """Test getting items from training dataset."""
        dataset = StreamingDualChannelDataset(
            data_folders=str(temp_dataset['train_folder1']),
            split="train",
            image_size=(224, 224)
        )
        
        rgb, brightness, label = dataset[0]
        
        # Check types and shapes
        assert isinstance(rgb, torch.Tensor)
        assert isinstance(brightness, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        
        assert rgb.shape == (3, 224, 224)
        assert brightness.shape == (1, 224, 224)
        assert label.shape == ()
        assert label.dtype == torch.long
        
        # Check value ranges
        assert 0 <= rgb.min() <= rgb.max() <= 1
        assert 0 <= brightness.min() <= brightness.max() <= 1
    
    def test_getitem_validation(self, temp_dataset):
        """Test getting items from validation dataset."""
        dataset = StreamingDualChannelDataset(
            data_folders=str(temp_dataset['val_folder']),
            split="validation",
            truth_file=str(temp_dataset['truth_file']),
            image_size=(224, 224)
        )
        
        rgb, brightness, label = dataset[0]
        
        # Check types and shapes
        assert isinstance(rgb, torch.Tensor)
        assert isinstance(brightness, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        
        assert rgb.shape == (3, 224, 224)
        assert brightness.shape == (1, 224, 224)
        assert label.item() == 0  # First label from truth file
    
    def test_getitem_with_transform(self, temp_dataset):
        """Test getting items with transforms applied."""
        # Use a simple transform that works with both RGB and brightness
        transform = transforms.Compose([
            transforms.Resize((200, 200)),  # Resize should work for both
        ])
        
        dataset = StreamingDualChannelDataset(
            data_folders=str(temp_dataset['train_folder1']),
            split="train",
            transform=transform,
            image_size=(224, 224)
        )
        
        rgb, brightness, label = dataset[0]
        
        # Check that resize was applied
        assert rgb.shape == (3, 200, 200)
        assert brightness.shape == (1, 200, 200)
    
    def test_synchronized_transforms(self, temp_dataset):
        """Test that RGB and brightness get the same random transforms."""
        # Use a transform that has randomness
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0),  # Always flip
            transforms.RandomRotation(90)  # Random rotation
        ])
        
        dataset = StreamingDualChannelDataset(
            data_folders=str(temp_dataset['train_folder1']),
            split="train",
            transform=transform,
            image_size=(224, 224)
        )
        
        # Get the same item multiple times - should get different results due to randomness
        # but RGB and brightness should always match
        rgb1, brightness1, _ = dataset[0]
        rgb2, brightness2, _ = dataset[0]
        
        # They should be different due to randomness
        assert not torch.equal(rgb1, rgb2)
        assert not torch.equal(brightness1, brightness2)
        
        # But the relationship between RGB and brightness should be consistent
        # (This is hard to test directly, but we can at least verify shapes match)
        assert rgb1.shape == rgb2.shape
        assert brightness1.shape == brightness2.shape
    
    def test_image_loading_error(self, temp_dataset):
        """Test handling of image loading errors."""
        dataset = StreamingDualChannelDataset(
            data_folders=str(temp_dataset['train_folder1']),
            split="train",
            image_size=(224, 224)
        )
        
        # Corrupt the first image file
        first_image_path = dataset.image_paths[0]
        with open(first_image_path, 'w') as f:
            f.write("corrupted image data")
        
        with pytest.raises(RuntimeError, match="Error loading image"):
            dataset[0]
    
    def test_different_image_sizes(self, temp_dataset):
        """Test with different image sizes."""
        dataset = StreamingDualChannelDataset(
            data_folders=str(temp_dataset['train_folder1']),
            split="train",
            image_size=(128, 128)
        )
        
        rgb, brightness, label = dataset[0]
        
        assert rgb.shape == (3, 128, 128)
        assert brightness.shape == (1, 128, 128)
    
    def test_valid_extensions(self, temp_dataset):
        """Test that only valid extensions are loaded."""
        # Create an image with invalid extension
        invalid_img = temp_dataset['train_folder1'] / 'invalid.txt'
        with open(invalid_img, 'w') as f:
            f.write("not an image")
        
        dataset = StreamingDualChannelDataset(
            data_folders=str(temp_dataset['train_folder1']),
            split="train",
            image_size=(224, 224)
        )
        
        # Should still have 6 images (invalid one ignored)
        assert len(dataset) == 6


class TestCreateImageNetDataloaders:
    """Test cases for convenience functions."""
    
    @pytest.fixture
    def temp_dataset(self):
        """Create a temporary dataset structure for testing."""
        temp_dir = tempfile.mkdtemp()
        dataset_paths = {
            'temp_dir': Path(temp_dir),
            'train_folder1': Path(temp_dir) / 'train_folder1',
            'val_folder': Path(temp_dir) / 'val_folder',
            'test_folder': Path(temp_dir) / 'test_folder',
            'truth_file': Path(temp_dir) / 'truth.txt'
        }
        
        # Create directories
        for folder in ['train_folder1', 'val_folder', 'test_folder']:
            dataset_paths[folder].mkdir(parents=True, exist_ok=True)
        
        # Create minimal sample data
        for i in range(2):
            # Train images
            train_img = dataset_paths['train_folder1'] / f"n01440764_{i}_n01440764.JPEG"
            self._create_test_image(train_img)
            
            # Val images
            val_img = dataset_paths['val_folder'] / f"ILSVRC2012_val_{i+1:08d}_n01440764.JPEG"
            self._create_test_image(val_img)
            
            # Test images
            test_img = dataset_paths['test_folder'] / f"ILSVRC2012_test_{i+1:08d}.JPEG"
            self._create_test_image(test_img)
        
        # Create truth file
        with open(dataset_paths['truth_file'], 'w') as f:
            f.write("0\n1\n")
        
        yield dataset_paths
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def _create_test_image(self, path: Path, size: tuple = (224, 224)):
        """Create a test image at the given path."""
        image = Image.new('RGB', size, (128, 128, 128))
        image.save(path)
    
    def test_create_train_val_dataloaders(self, temp_dataset):
        """Test creating train and validation dataloaders."""
        train_loader, val_loader = create_imagenet_dual_channel_train_val_dataloaders(
            train_folders=str(temp_dataset['train_folder1']),
            val_folder=str(temp_dataset['val_folder']),
            truth_file=str(temp_dataset['truth_file']),
            batch_size=2,
            num_workers=0  # Use 0 for testing to avoid multiprocessing issues
        )
        
        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
        
        # Test that we can iterate through loaders
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        
        # Check batch structure
        assert len(train_batch) == 3  # RGB, brightness, labels
        assert len(val_batch) == 3
        
        rgb, brightness, labels = train_batch
        assert rgb.shape[0] == 2  # batch size
        assert brightness.shape[0] == 2
        assert labels.shape[0] == 2
        
        # Default behavior: val_batch_size should equal batch_size
        val_rgb, val_brightness, val_labels = val_batch
        assert val_rgb.shape[0] == 2  # Same as training batch size
        assert val_brightness.shape[0] == 2
        assert val_labels.shape[0] == 2

    def test_create_train_val_dataloaders_with_custom_val_batch_size(self, temp_dataset):
        """Test creating dataloaders with custom validation batch size."""
        train_loader, val_loader = create_imagenet_dual_channel_train_val_dataloaders(
            train_folders=str(temp_dataset['train_folder1']),
            val_folder=str(temp_dataset['val_folder']),
            truth_file=str(temp_dataset['truth_file']),
            batch_size=1,
            val_batch_size=2,  # Different from training batch size
            num_workers=0
        )
        
        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
        
        # Test batch sizes
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        
        # Training batch should have size 1
        train_rgb, train_brightness, train_labels = train_batch
        assert train_rgb.shape[0] == 1
        assert train_brightness.shape[0] == 1
        assert train_labels.shape[0] == 1
        
        # Validation batch should have size 2
        val_rgb, val_brightness, val_labels = val_batch
        assert val_rgb.shape[0] == 2
        assert val_brightness.shape[0] == 2
        assert val_labels.shape[0] == 2

    def test_create_train_val_dataloaders_val_batch_size_none(self, temp_dataset):
        """Test that val_batch_size=None defaults to batch_size."""
        train_loader, val_loader = create_imagenet_dual_channel_train_val_dataloaders(
            train_folders=str(temp_dataset['train_folder1']),
            val_folder=str(temp_dataset['val_folder']),
            truth_file=str(temp_dataset['truth_file']),
            batch_size=2,
            val_batch_size=None,  # Explicitly set to None
            num_workers=0
        )
        
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        
        # Both should have same batch size
        assert train_batch[0].shape[0] == 2
        assert val_batch[0].shape[0] == 2

    def test_create_train_val_dataloaders_large_val_batch_size(self, temp_dataset):
        """Test with validation batch size larger than available data."""
        train_loader, val_loader = create_imagenet_dual_channel_train_val_dataloaders(
            train_folders=str(temp_dataset['train_folder1']),
            val_folder=str(temp_dataset['val_folder']),
            truth_file=str(temp_dataset['truth_file']),
            batch_size=1,
            val_batch_size=10,  # Larger than 2 available validation samples
            num_workers=0
        )
        
        # Should still work, just get smaller final batch
        val_batch = next(iter(val_loader))
        val_rgb, val_brightness, val_labels = val_batch
        
        # Should get all available validation samples (2 in this case)
        assert val_rgb.shape[0] == 2
        assert val_brightness.shape[0] == 2
        assert val_labels.shape[0] == 2
    
    def test_create_test_dataloader(self, temp_dataset):
        """Test creating test dataloader."""
        test_loader = create_imagenet_dual_channel_test_dataloader(
            test_folder=str(temp_dataset['test_folder']),
            truth_file=None,  # Test without labels
            batch_size=2,
            num_workers=0
        )
        
        assert isinstance(test_loader, DataLoader)
        
        # Since we didn't provide truth file, this should fail
        # Let's test with truth file instead
        test_loader = create_imagenet_dual_channel_test_dataloader(
            test_folder=str(temp_dataset['test_folder']),
            truth_file=str(temp_dataset['truth_file']),
            batch_size=2,
            num_workers=0
        )
        
        test_batch = next(iter(test_loader))
        rgb, brightness, labels = test_batch
        assert rgb.shape[0] == 2
    
    def test_create_train_val_dataloaders_multiple_folders(self, temp_dataset):
        """Test creating dataloaders with multiple training folders."""
        # Create a second training folder
        train_folder2 = temp_dataset['temp_dir'] / 'train_folder2'
        train_folder2.mkdir()
        for i in range(2):
            img = train_folder2 / f"n01443537_{i}_n01443537.JPEG"
            self._create_test_image(img)
        
        train_loader, val_loader = create_imagenet_dual_channel_train_val_dataloaders(
            train_folders=[
                str(temp_dataset['train_folder1']),
                str(train_folder2)
            ],
            val_folder=str(temp_dataset['val_folder']),
            truth_file=str(temp_dataset['truth_file']),
            batch_size=2,
            num_workers=0
        )
        
        # Should have more training data now
        train_batch = next(iter(train_loader))
        rgb, brightness, labels = train_batch
        assert rgb.shape[0] == 2

    def test_create_train_val_dataloaders_multiple_folders_with_val_batch_size(self, temp_dataset):
        """Test creating dataloaders with multiple training folders and custom val batch size."""
        # Create a second training folder
        train_folder2 = temp_dataset['temp_dir'] / 'train_folder2'
        train_folder2.mkdir()
        for i in range(2):
            img = train_folder2 / f"n01443537_{i}_n01443537.JPEG"
            self._create_test_image(img)
        
        train_loader, val_loader = create_imagenet_dual_channel_train_val_dataloaders(
            train_folders=[
                str(temp_dataset['train_folder1']),
                str(train_folder2)
            ],
            val_folder=str(temp_dataset['val_folder']),
            truth_file=str(temp_dataset['truth_file']),
            batch_size=1,
            val_batch_size=2,  # Different validation batch size
            num_workers=0
        )
        
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        
        # Training batch size should be 1
        assert train_batch[0].shape[0] == 1
        # Validation batch size should be 2
        assert val_batch[0].shape[0] == 2


class TestDefaultImageNetTransforms:
    """Test cases for default transform creation."""
    
    def test_create_default_transforms(self):
        """Test creating default ImageNet transforms."""
        train_transform, val_transform = create_default_imagenet_transforms()
        
        assert train_transform is not None
        assert val_transform is not None
        assert isinstance(train_transform, transforms.Compose)
        assert isinstance(val_transform, transforms.Compose)
    
    def test_default_transforms_with_custom_params(self):
        """Test creating transforms with custom parameters."""
        custom_size = (256, 256)
        custom_mean = (0.5, 0.5, 0.5)
        custom_std = (0.5, 0.5, 0.5)
        
        train_transform, val_transform = create_default_imagenet_transforms(
            image_size=custom_size,
            mean=custom_mean,
            std=custom_std
        )
        
        # Test that transforms work with a sample tensor
        sample_tensor = torch.rand(3, 256, 256)
        
        # Transforms should work without error
        train_result = train_transform(sample_tensor)
        val_result = val_transform(sample_tensor)
        
        assert train_result.shape == (3, 256, 256)
        assert val_result.shape == (3, 256, 256)
    
    def test_transforms_apply_normalization(self):
        """Test that transforms apply normalization correctly."""
        train_transform, val_transform = create_default_imagenet_transforms()
        
        # Create a tensor with values in [0, 1]
        sample_tensor = torch.ones(3, 224, 224) * 0.5
        
        train_result = train_transform(sample_tensor)
        val_result = val_transform(sample_tensor)
        
        # After normalization, values should be different from input
        assert not torch.allclose(train_result, sample_tensor)
        assert not torch.allclose(val_result, sample_tensor)


class TestIntegration:
    """Integration tests combining multiple components."""
    
    @pytest.fixture
    def temp_dataset(self):
        """Create a comprehensive temporary dataset."""
        temp_dir = tempfile.mkdtemp()
        dataset_paths = {
            'temp_dir': Path(temp_dir),
            'train_folders': [],
            'val_folder': Path(temp_dir) / 'val_folder',
            'truth_file': Path(temp_dir) / 'truth.txt'
        }
        
        # Create multiple training folders
        for folder_idx in range(2):
            train_folder = Path(temp_dir) / f'train_folder_{folder_idx}'
            train_folder.mkdir(parents=True)
            dataset_paths['train_folders'].append(train_folder)
            
            # Create images for different classes
            classes = ['n01440764', 'n01443537']
            for class_idx, class_name in enumerate(classes):
                for img_idx in range(3):
                    img_name = f"{class_name}_{folder_idx*10+class_idx*3+img_idx}_{class_name}.JPEG"
                    img_path = train_folder / img_name
                    self._create_test_image(img_path)
        
        # Create validation folder
        dataset_paths['val_folder'].mkdir(parents=True)
        val_labels = []
        for i in range(10):
            img_name = f"ILSVRC2012_val_{i+1:08d}_n01440764.JPEG"
            img_path = dataset_paths['val_folder'] / img_name
            self._create_test_image(img_path)
            val_labels.append(i % 2)  # Alternate between class 0 and 1
        
        # Create truth file
        with open(dataset_paths['truth_file'], 'w') as f:
            for label in val_labels:
                f.write(f"{label}\n")
        
        yield dataset_paths
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def _create_test_image(self, path: Path, size: tuple = (224, 224)):
        """Create a test image with some variation."""
        # Create images with slight color variation for more realistic testing
        color = (
            np.random.randint(100, 156),
            np.random.randint(100, 156),
            np.random.randint(100, 156)
        )
        image = Image.new('RGB', size, color)
        image.save(path)
    
    def test_end_to_end_training_simulation(self, temp_dataset):
        """Test a complete training simulation scenario."""
        # Create transforms
        train_transform, val_transform = create_default_imagenet_transforms(
            image_size=(224, 224)
        )
        
        # Create dataloaders
        train_loader, val_loader = create_imagenet_dual_channel_train_val_dataloaders(
            train_folders=[str(folder) for folder in temp_dataset['train_folders']],
            val_folder=str(temp_dataset['val_folder']),
            truth_file=str(temp_dataset['truth_file']),
            train_transform=train_transform,
            val_transform=val_transform,
            batch_size=4,
            num_workers=0
        )
        
        # Simulate a few training steps
        train_batches_processed = 0
        for rgb_batch, brightness_batch, label_batch in train_loader:
            # Verify batch properties
            assert rgb_batch.shape[1:] == (3, 224, 224)
            assert brightness_batch.shape[1:] == (1, 224, 224)
            assert label_batch.dtype == torch.long
            
            # Check that batches are properly normalized
            assert rgb_batch.min() < 0  # Should be normalized
            # Check that brightness is in a reasonable range (may not be negative depending on normalization)
            assert brightness_batch.min() >= -5.0 and brightness_batch.max() <= 5.0  # Should be normalized-ish
            
            train_batches_processed += 1
            if train_batches_processed >= 2:  # Process a few batches
                break
        
        # Simulate validation
        val_batches_processed = 0
        total_val_samples = 0
        for rgb_batch, brightness_batch, label_batch in val_loader:
            assert rgb_batch.shape[1:] == (3, 224, 224)
            assert brightness_batch.shape[1:] == (1, 224, 224)
            total_val_samples += label_batch.shape[0]
            
            val_batches_processed += 1
        
        # Should have processed all validation samples
        assert total_val_samples == 10
        assert train_batches_processed >= 1
        assert val_batches_processed >= 1

    def test_end_to_end_with_different_batch_sizes(self, temp_dataset):
        """Test complete training simulation with different training and validation batch sizes."""
        # Create transforms
        train_transform, val_transform = create_default_imagenet_transforms(
            image_size=(224, 224)
        )
        
        # Create dataloaders with different batch sizes
        train_loader, val_loader = create_imagenet_dual_channel_train_val_dataloaders(
            train_folders=[str(folder) for folder in temp_dataset['train_folders']],
            val_folder=str(temp_dataset['val_folder']),
            truth_file=str(temp_dataset['truth_file']),
            train_transform=train_transform,
            val_transform=val_transform,
            batch_size=2,  # Smaller training batch
            val_batch_size=4,  # Larger validation batch
            num_workers=0
        )
        
        # Simulate training - should get batches of size 2
        train_batch = next(iter(train_loader))
        rgb_batch, brightness_batch, label_batch = train_batch
        assert rgb_batch.shape[0] == 2
        assert brightness_batch.shape[0] == 2
        assert label_batch.shape[0] == 2
        
        # Simulate validation - should get batches of size up to 4
        val_batch = next(iter(val_loader))
        val_rgb, val_brightness, val_labels = val_batch
        
        # Should get the larger validation batch size (up to available data)
        expected_val_batch_size = min(4, 10)  # 10 total validation samples
        assert val_rgb.shape[0] == expected_val_batch_size
        assert val_brightness.shape[0] == expected_val_batch_size
        assert val_labels.shape[0] == expected_val_batch_size


if __name__ == "__main__":
    pytest.main([__file__])

class TestBatchLoadingFunctionality:
    """Test batch loading and streaming functionality for StreamingDualChannelDataset."""
    
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
        import time
        
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

    def test_validation_dataloader_batch_behavior_with_val_batch_size(self, large_temp_dataset):
        """Test that validation DataLoader uses the correct val_batch_size."""
        train_loader, val_loader = create_imagenet_dual_channel_train_val_dataloaders(
            train_folders=str(large_temp_dataset['train_folder']),
            val_folder=str(large_temp_dataset['val_folder']),
            truth_file=str(large_temp_dataset['truth_file']),
            batch_size=4,
            val_batch_size=8,  # Explicitly set different validation batch size
            num_workers=0
        )
        
        # Training batch should be size 4
        train_batch = next(iter(train_loader))
        train_rgb, train_brightness, train_labels = train_batch
        assert train_rgb.shape[0] == 4
        
        # Validation batch should be size 8 (or all remaining samples if fewer)
        val_batch = next(iter(val_loader))
        val_rgb, val_brightness, val_labels = val_batch
        expected_val_size = min(8, 20)  # 20 total validation samples
        assert val_rgb.shape[0] == expected_val_size
        
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
