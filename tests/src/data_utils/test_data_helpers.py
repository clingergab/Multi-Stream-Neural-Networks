"""Unit tests for src.data_utils.data_helpers module."""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from torch.utils.data import DataLoader, TensorDataset

from src.data_utils.data_helpers import (
    calculate_dataset_stats,
    get_class_weights
)


class TestCalculateDatasetStats:
    """Test cases for calculate_dataset_stats function."""
    
    @pytest.fixture
    def mock_dataloader(self):
        """Create a mock dataloader with known data."""
        # Create mock data
        color_data = torch.ones(2, 3, 4, 4) * 0.5  # Fixed values for predictable stats
        brightness_data = torch.ones(2, 1, 4, 4) * 0.3
        
        # Create mock batches
        batch1 = {
            'color': color_data[:1],  # Shape: [1, 3, 4, 4]
            'brightness': brightness_data[:1]  # Shape: [1, 1, 4, 4]
        }
        batch2 = {
            'color': color_data[1:],  # Shape: [1, 3, 4, 4]  
            'brightness': brightness_data[1:]  # Shape: [1, 1, 4, 4]
        }
        
        # Mock dataloader
        mock_loader = MagicMock()
        mock_loader.__iter__.return_value = iter([batch1, batch2])
        
        # Mock dataset for accessing individual items
        mock_dataset = MagicMock()
        mock_dataset.__getitem__.return_value = {
            'color': torch.ones(3, 4, 4),
            'brightness': torch.ones(1, 4, 4)
        }
        mock_loader.dataset = mock_dataset
        
        return mock_loader
    
    def test_calculate_stats_structure(self, mock_dataloader):
        """Test that the function returns the correct structure."""
        stats = calculate_dataset_stats(mock_dataloader)
        
        assert isinstance(stats, dict)
        assert 'color_mean' in stats
        assert 'color_std' in stats
        assert 'brightness_mean' in stats
        assert 'brightness_std' in stats
        assert 'num_samples' in stats
    
    def test_calculate_stats_types(self, mock_dataloader):
        """Test that the function returns correct data types."""
        stats = calculate_dataset_stats(mock_dataloader)
        
        assert isinstance(stats['color_mean'], list)
        assert isinstance(stats['color_std'], list)
        assert isinstance(stats['brightness_mean'], list)
        assert isinstance(stats['brightness_std'], list)
        assert isinstance(stats['num_samples'], int)
        
        # Check list lengths
        assert len(stats['color_mean']) == 3  # RGB channels
        assert len(stats['color_std']) == 3   # RGB channels
        assert len(stats['brightness_mean']) == 1  # Single brightness channel
        assert len(stats['brightness_std']) == 1   # Single brightness channel
    
    def test_calculate_stats_values(self, mock_dataloader):
        """Test that calculated statistics are reasonable."""
        stats = calculate_dataset_stats(mock_dataloader)
        
        # With constant values (0.5 for color, 0.3 for brightness), std should be 0
        assert all(std == 0.0 for std in stats['color_std'])
        assert stats['brightness_std'][0] == 0.0
        
        # Means should match the constant values
        assert all(abs(mean - 0.5) < 1e-6 for mean in stats['color_mean'])
        assert abs(stats['brightness_mean'][0] - 0.3) < 1e-6
        
        # Should have processed 2 samples
        assert stats['num_samples'] == 2
    
    def test_calculate_stats_real_dataloader(self):
        """Test with a real dataloader using realistic data."""
        # Create realistic data
        color_data = torch.rand(10, 3, 8, 8)
        brightness_data = torch.rand(10, 1, 8, 8)
        labels = torch.randint(0, 5, (10,))
        
        # Create a simple dataset that returns dict format
        class MultiStreamDataset:
            def __init__(self, color, brightness, labels):
                self.color = color
                self.brightness = brightness
                self.labels = labels
            
            def __len__(self):
                return len(self.labels)
            
            def __getitem__(self, idx):
                return {
                    'color': self.color[idx],
                    'brightness': self.brightness[idx],
                    'target': self.labels[idx]
                }
        
        dataset = MultiStreamDataset(color_data, brightness_data, labels)
        
        def collate_fn(batch):
            color_batch = torch.stack([item['color'] for item in batch])
            brightness_batch = torch.stack([item['brightness'] for item in batch])
            return {
                'color': color_batch,
                'brightness': brightness_batch
            }
        
        dataloader = DataLoader(dataset, batch_size=3, collate_fn=collate_fn)
        
        stats = calculate_dataset_stats(dataloader)
        
        # Check that stats are within reasonable ranges for random data
        assert all(0.0 <= mean <= 1.0 for mean in stats['color_mean'])
        assert all(0.0 <= std <= 1.0 for std in stats['color_std'])
        assert 0.0 <= stats['brightness_mean'][0] <= 1.0
        assert 0.0 <= stats['brightness_std'][0] <= 1.0
        assert stats['num_samples'] == 10


class TestGetClassWeights:
    """Test cases for get_class_weights function."""
    
    @pytest.fixture
    def balanced_dataset(self):
        """Create a balanced dataset for testing."""
        class MockDataset:
            def __init__(self):
                # 10 samples per class for 3 classes
                self.data = [
                    {'target': 0}, {'target': 0}, {'target': 0}, {'target': 0}, {'target': 0},
                    {'target': 0}, {'target': 0}, {'target': 0}, {'target': 0}, {'target': 0},
                    {'target': 1}, {'target': 1}, {'target': 1}, {'target': 1}, {'target': 1},
                    {'target': 1}, {'target': 1}, {'target': 1}, {'target': 1}, {'target': 1},
                    {'target': 2}, {'target': 2}, {'target': 2}, {'target': 2}, {'target': 2},
                    {'target': 2}, {'target': 2}, {'target': 2}, {'target': 2}, {'target': 2}
                ]
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        return MockDataset()
    
    @pytest.fixture
    def imbalanced_dataset(self):
        """Create an imbalanced dataset for testing."""
        class MockDataset:
            def __init__(self):
                # Class 0: 20 samples, Class 1: 5 samples, Class 2: 5 samples
                self.data = (
                    [{'target': 0}] * 20 +
                    [{'target': 1}] * 5 +
                    [{'target': 2}] * 5
                )
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        return MockDataset()
    
    @pytest.fixture
    def tensor_target_dataset(self):
        """Create a dataset with tensor targets."""
        class MockDataset:
            def __init__(self):
                self.data = [
                    {'target': torch.tensor(0)}, {'target': torch.tensor(0)},
                    {'target': torch.tensor(1)}, {'target': torch.tensor(1)},
                    {'target': torch.tensor(2)}, {'target': torch.tensor(2)}
                ]
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        return MockDataset()
    
    def test_balanced_dataset_weights(self, balanced_dataset):
        """Test class weights for balanced dataset."""
        weights = get_class_weights(balanced_dataset)
        
        assert isinstance(weights, torch.Tensor)
        assert weights.shape == (3,)  # 3 classes
        
        # For balanced dataset, all weights should be 1.0
        expected_weight = 1.0  # total_samples / (num_classes * count) = 30 / (3 * 10) = 1.0
        torch.testing.assert_close(weights, torch.tensor([1.0, 1.0, 1.0]))
    
    def test_imbalanced_dataset_weights(self, imbalanced_dataset):
        """Test class weights for imbalanced dataset."""
        weights = get_class_weights(imbalanced_dataset)
        
        assert isinstance(weights, torch.Tensor)
        assert weights.shape == (3,)  # 3 classes
        
        # Calculate expected weights
        # Class 0: 30 / (3 * 20) = 0.5
        # Class 1: 30 / (3 * 5) = 2.0
        # Class 2: 30 / (3 * 5) = 2.0
        expected_weights = torch.tensor([0.5, 2.0, 2.0])
        torch.testing.assert_close(weights, expected_weights)
        
        # Minority classes should have higher weights
        assert weights[1] > weights[0]
        assert weights[2] > weights[0]
        assert weights[1] == weights[2]  # Equal minority classes
    
    def test_tensor_targets(self, tensor_target_dataset):
        """Test class weights with tensor targets."""
        weights = get_class_weights(tensor_target_dataset)
        
        assert isinstance(weights, torch.Tensor)
        assert weights.shape == (3,)  # 3 classes
        
        # Each class has 2 samples out of 6 total
        # Weight = 6 / (3 * 2) = 1.0
        expected_weights = torch.tensor([1.0, 1.0, 1.0])
        torch.testing.assert_close(weights, expected_weights)
    
    def test_single_class_dataset(self):
        """Test class weights with single class."""
        class SingleClassDataset:
            def __init__(self):
                self.data = [{'target': 0}] * 5
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        dataset = SingleClassDataset()
        weights = get_class_weights(dataset)
        
        assert isinstance(weights, torch.Tensor)
        assert weights.shape == (1,)  # 1 class
        
        # Weight = 5 / (1 * 5) = 1.0
        expected_weights = torch.tensor([1.0])
        torch.testing.assert_close(weights, expected_weights)
    
    def test_empty_dataset(self):
        """Test class weights with empty dataset."""
        class EmptyDataset:
            def __len__(self):
                return 0
            
            def __getitem__(self, idx):
                raise IndexError("Empty dataset")
        
        dataset = EmptyDataset()
        weights = get_class_weights(dataset)
        
        assert isinstance(weights, torch.Tensor)
        assert weights.shape == (0,)  # No classes
    
    def test_non_sequential_classes(self):
        """Test class weights with non-sequential class indices."""
        class NonSequentialDataset:
            def __init__(self):
                # Classes 0, 2, 5 (skipping 1, 3, 4)
                self.data = [
                    {'target': 0}, {'target': 0},
                    {'target': 2}, {'target': 2},
                    {'target': 5}, {'target': 5}
                ]
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        dataset = NonSequentialDataset()
        weights = get_class_weights(dataset)
        
        assert isinstance(weights, torch.Tensor)
        assert weights.shape == (6,)  # Max class index + 1
        
        # Classes 0, 2, 5 should have weight 1.0
        # Classes 1, 3, 4 should have weight 0.0 (no samples)
        expected_weights = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 1.0])
        torch.testing.assert_close(weights, expected_weights)


if __name__ == "__main__":
    pytest.main([__file__])
