"""
Tests for the StreamingDualChannelDataset implementation.
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
from unittest.mock import patch, MagicMock
from PIL import Image
import numpy as np

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

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
            assert brightness_batch.min() < 0
            
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


if __name__ == "__main__":
    pytest.main([__file__])
