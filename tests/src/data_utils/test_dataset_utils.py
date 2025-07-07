"""Unit tests for src.data_utils.dataset_utils module."""

import pytest
import pickle
import torch
import numpy as np
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock
import tempfile
import os

from src.data_utils.dataset_utils import (
    _load_cifar100_pickle,
    load_cifar100_data,
    CIFAR100_FINE_LABELS
)


class TestLoadCIFAR100Pickle:
    """Test cases for _load_cifar100_pickle function."""
    
    def test_load_existing_file(self):
        """Test loading an existing pickle file."""
        # Create test data
        test_data = {'data': np.random.randn(10, 3072), 'labels': [1, 2, 3]}
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            pickle.dump(test_data, tmp_file)
            tmp_path = Path(tmp_file.name)
        
        try:
            # Test the function
            result = _load_cifar100_pickle(tmp_path)
            
            assert isinstance(result, dict)
            assert 'data' in result
            assert 'labels' in result
            np.testing.assert_array_equal(result['data'], test_data['data'])
            assert result['labels'] == test_data['labels']
        finally:
            # Clean up
            os.unlink(tmp_path)
    
    def test_load_nonexistent_file(self):
        """Test loading a non-existent file."""
        non_existent_path = Path('/path/that/does/not/exist.pkl')
        
        with pytest.raises(FileNotFoundError) as exc_info:
            _load_cifar100_pickle(non_existent_path)
        
        assert "CIFAR-100 file not found" in str(exc_info.value)
        assert str(non_existent_path) in str(exc_info.value)
    
    @patch('builtins.open', side_effect=IOError("Permission denied"))
    def test_load_file_io_error(self, mock_open):
        """Test handling of IO errors."""
        test_path = Path('test_file.pkl')
        
        # Mock Path.exists to return True
        with patch.object(Path, 'exists', return_value=True):
            with pytest.raises(IOError) as exc_info:
                _load_cifar100_pickle(test_path)
        
        assert "Failed to load CIFAR-100 pickle file" in str(exc_info.value)
    
    @patch('builtins.open', new_callable=mock_open, read_data=b'invalid_pickle_data')
    def test_load_corrupted_file(self, mock_file):
        """Test loading a corrupted pickle file."""
        test_path = Path('corrupted_file.pkl')
        
        with patch.object(Path, 'exists', return_value=True):
            with pytest.raises(IOError) as exc_info:
                _load_cifar100_pickle(test_path)
        
        assert "Failed to load CIFAR-100 pickle file" in str(exc_info.value)


class TestLoadCIFAR100Data:
    """Test cases for load_cifar100_data function."""
    
    @pytest.fixture
    def mock_cifar100_data(self):
        """Create mock CIFAR-100 data for testing."""
        # Create realistic CIFAR-100 data structure
        train_data = {
            b'data': np.random.randint(0, 256, (100, 3072), dtype=np.uint8),
            b'fine_labels': list(range(100))  # 100 samples with labels 0-99
        }
        
        test_data = {
            b'data': np.random.randint(0, 256, (50, 3072), dtype=np.uint8),
            b'fine_labels': list(range(50))   # 50 samples with labels 0-49
        }
        
        return train_data, test_data
    
    @patch('src.data_utils.dataset_utils._load_cifar100_pickle')
    @patch('pathlib.Path.exists')
    def test_load_torch_format(self, mock_exists, mock_load_pickle, mock_cifar100_data):
        """Test loading data in torch format."""
        mock_exists.return_value = True
        train_data, test_data = mock_cifar100_data
        mock_load_pickle.side_effect = [train_data, test_data]
        
        result = load_cifar100_data(
            data_dir="./test/path",
            return_type='torch',
            normalize=True,
            verbose=False
        )
        
        train_X, train_y, test_X, test_y = result
        
        # Check types
        assert isinstance(train_X, torch.Tensor)
        assert isinstance(train_y, torch.Tensor)
        assert isinstance(test_X, torch.Tensor)
        assert isinstance(test_y, torch.Tensor)
        
        # Check shapes
        assert train_X.shape == (100, 3, 32, 32)
        assert train_y.shape == (100,)
        assert test_X.shape == (50, 3, 32, 32)
        assert test_y.shape == (50,)
        
        # Check data types
        assert train_X.dtype == torch.float32
        assert train_y.dtype == torch.long
        assert test_X.dtype == torch.float32
        assert test_y.dtype == torch.long
        
        # Check normalization (values should be in [0, 1])
        assert torch.all(train_X >= 0.0)
        assert torch.all(train_X <= 1.0)
        assert torch.all(test_X >= 0.0)
        assert torch.all(test_X <= 1.0)
    
    @patch('src.data_utils.dataset_utils._load_cifar100_pickle')
    @patch('pathlib.Path.exists')
    def test_load_numpy_format(self, mock_exists, mock_load_pickle, mock_cifar100_data):
        """Test loading data in numpy format."""
        mock_exists.return_value = True
        train_data, test_data = mock_cifar100_data
        mock_load_pickle.side_effect = [train_data, test_data]
        
        result = load_cifar100_data(
            data_dir="./test/path",
            return_type='numpy',
            normalize=True,
            verbose=False
        )
        
        train_X, train_y, test_X, test_y = result
        
        # Check types
        assert isinstance(train_X, np.ndarray)
        assert isinstance(train_y, np.ndarray)
        assert isinstance(test_X, np.ndarray)
        assert isinstance(test_y, np.ndarray)
        
        # Check shapes
        assert train_X.shape == (100, 3, 32, 32)
        assert train_y.shape == (100,)
        assert test_X.shape == (50, 3, 32, 32)
        assert test_y.shape == (50,)
        
        # Check data types
        assert train_X.dtype == np.float32
        assert train_y.dtype == np.int64
        assert test_X.dtype == np.float32
        assert test_y.dtype == np.int64
    
    @patch('src.data_utils.dataset_utils._load_cifar100_pickle')
    @patch('pathlib.Path.exists')
    def test_load_without_normalization(self, mock_exists, mock_load_pickle, mock_cifar100_data):
        """Test loading data without normalization."""
        mock_exists.return_value = True
        train_data, test_data = mock_cifar100_data
        mock_load_pickle.side_effect = [train_data, test_data]
        
        result = load_cifar100_data(
            data_dir="./test/path",
            return_type='torch',
            normalize=False,
            verbose=False
        )
        
        train_X, train_y, test_X, test_y = result
        
        # Values should be in [0, 255] range when not normalized
        assert torch.all(train_X >= 0.0)
        assert torch.all(train_X <= 255.0)
        assert torch.all(test_X >= 0.0)
        assert torch.all(test_X <= 255.0)
    
    def test_invalid_return_type(self):
        """Test with invalid return_type parameter."""
        with pytest.raises(ValueError) as exc_info:
            load_cifar100_data(return_type='invalid', verbose=False)
        
        assert "return_type must be 'torch' or 'numpy'" in str(exc_info.value)
    
    def test_nonexistent_data_directory(self):
        """Test with non-existent data directory."""
        with patch('pathlib.Path.exists', return_value=False):
            with pytest.raises(FileNotFoundError) as exc_info:
                load_cifar100_data(data_dir="/nonexistent/path", verbose=False)
        
        assert "CIFAR-100 data directory not found" in str(exc_info.value)
    
    @patch('src.data_utils.dataset_utils._load_cifar100_pickle')
    @patch('pathlib.Path.exists')
    def test_load_with_string_path(self, mock_exists, mock_load_pickle, mock_cifar100_data):
        """Test loading with string path instead of Path object."""
        mock_exists.return_value = True
        train_data, test_data = mock_cifar100_data
        mock_load_pickle.side_effect = [train_data, test_data]
        
        result = load_cifar100_data(
            data_dir="./test/string/path",  # String instead of Path
            return_type='torch',
            verbose=False
        )
        
        train_X, train_y, test_X, test_y = result
        
        # Should work the same way
        assert isinstance(train_X, torch.Tensor)
        assert train_X.shape == (100, 3, 32, 32)
    
    @patch('src.data_utils.dataset_utils._load_cifar100_pickle')
    @patch('pathlib.Path.exists')
    def test_verbose_output(self, mock_exists, mock_load_pickle, mock_cifar100_data, capsys):
        """Test verbose output."""
        mock_exists.return_value = True
        train_data, test_data = mock_cifar100_data
        mock_load_pickle.side_effect = [train_data, test_data]
        
        load_cifar100_data(
            data_dir="./test/path",
            return_type='torch',
            verbose=True
        )
        
        captured = capsys.readouterr()
        assert "Loading CIFAR-100 from:" in captured.out
        assert "Loaded CIFAR-100 (torch format):" in captured.out
        assert "Training:" in captured.out
        assert "Test:" in captured.out

    @patch('src.data_utils.dataset_utils._load_cifar100_pickle')
    @patch('pathlib.Path.exists')
    def test_data_shapes_and_values(self, mock_exists, mock_load_pickle):
        """Test that data is properly reshaped and processed."""
        mock_exists.return_value = True
        
        # Create test data that mimics CIFAR-100 format
        # CIFAR-100 stores RGB data in channel-first format for each sample
        sample1_data = np.zeros(3072, dtype=np.uint8)
        sample1_data[0] = 100    # R channel, pixel (0,0)
        sample1_data[1024] = 150 # G channel, pixel (0,0) 
        sample1_data[2048] = 200 # B channel, pixel (0,0)
        
        sample2_data = np.zeros(3072, dtype=np.uint8)
        sample2_data[0] = 50     # R channel, pixel (0,0)
        sample2_data[1024] = 75  # G channel, pixel (0,0)
        sample2_data[2048] = 25  # B channel, pixel (0,0)
        
        train_data = {
            b'data': np.vstack([sample1_data, sample2_data]),
            b'fine_labels': [5, 10]
        }

        test_data = {
            b'data': sample1_data.reshape(1, -1),
            b'fine_labels': [15]
        }

        mock_load_pickle.side_effect = [train_data, test_data]

        result = load_cifar100_data(
            data_dir="./test/path",
            return_type='torch',
            normalize=False,
            verbose=False
        )

        train_X, train_y, test_X, test_y = result

        # Check that data was properly reshaped from (N, 3072) to (N, 3, 32, 32)
        assert train_X.shape == (2, 3, 32, 32)
        assert test_X.shape == (1, 3, 32, 32)

        # Check that labels are correct
        torch.testing.assert_close(train_y, torch.tensor([5, 10], dtype=torch.long))
        torch.testing.assert_close(test_y, torch.tensor([15], dtype=torch.long))

        # Verify that reshaping preserved data order
        # CIFAR-100 stores data in channel-first format: R[0:1024], G[1024:2048], B[2048:3072]
        # For the first sample, first pixel should have RGB values [100, 150, 200]
        expected_first_pixel = torch.tensor([100.0, 150.0, 200.0], dtype=torch.float32)
        actual_first_pixel = train_X[0, :, 0, 0]
        torch.testing.assert_close(actual_first_pixel, expected_first_pixel)
        
        # Verify second sample as well
        expected_second_pixel = torch.tensor([50.0, 75.0, 25.0], dtype=torch.float32)
        actual_second_pixel = train_X[1, :, 0, 0]
        torch.testing.assert_close(actual_second_pixel, expected_second_pixel)


class TestCIFAR100FineLabels:
    """Test cases for CIFAR100_FINE_LABELS constant."""
    
    def test_labels_count(self):
        """Test that there are exactly 100 labels."""
        assert len(CIFAR100_FINE_LABELS) == 100
    
    def test_labels_are_strings(self):
        """Test that all labels are strings."""
        assert all(isinstance(label, str) for label in CIFAR100_FINE_LABELS)
    
    def test_labels_not_empty(self):
        """Test that no labels are empty strings."""
        assert all(len(label) > 0 for label in CIFAR100_FINE_LABELS)
    
    def test_labels_unique(self):
        """Test that all labels are unique."""
        assert len(set(CIFAR100_FINE_LABELS)) == len(CIFAR100_FINE_LABELS)
    
    def test_specific_labels(self):
        """Test that specific known labels are present."""
        # Test a few known CIFAR-100 labels
        assert 'apple' in CIFAR100_FINE_LABELS
        assert 'bear' in CIFAR100_FINE_LABELS
        assert 'bicycle' in CIFAR100_FINE_LABELS
        assert 'tiger' in CIFAR100_FINE_LABELS
        assert 'woman' in CIFAR100_FINE_LABELS
    
    def test_labels_order(self):
        """Test that labels are in expected order (first few)."""
        # First few labels should be in this order based on CIFAR-100 specification
        expected_first_labels = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver']
        assert CIFAR100_FINE_LABELS[:5] == expected_first_labels
    
    def test_labels_format(self):
        """Test that labels follow expected format."""
        # Labels should be lowercase and use underscores for spaces
        for label in CIFAR100_FINE_LABELS:
            assert label.islower() or '_' in label
            assert ' ' not in label  # No spaces, should use underscores


class TestIntegration:
    """Integration tests for the entire module."""
    
    @patch('src.data_utils.dataset_utils._load_cifar100_pickle')
    @patch('pathlib.Path.exists')
    def test_full_pipeline_torch(self, mock_exists, mock_load_pickle):
        """Test the full data loading pipeline with torch format."""
        mock_exists.return_value = True
        
        # Create realistic data
        train_data = {
            b'data': np.random.randint(0, 256, (1000, 3072), dtype=np.uint8),
            b'fine_labels': np.random.randint(0, 100, 1000).tolist()
        }
        
        test_data = {
            b'data': np.random.randint(0, 256, (200, 3072), dtype=np.uint8),
            b'fine_labels': np.random.randint(0, 100, 200).tolist()
        }
        
        mock_load_pickle.side_effect = [train_data, test_data]
        
        # Load data
        train_X, train_y, test_X, test_y = load_cifar100_data(
            data_dir="./data/cifar-100",
            return_type='torch',
            normalize=True,
            verbose=False
        )
        
        # Verify complete pipeline
        assert isinstance(train_X, torch.Tensor)
        assert isinstance(train_y, torch.Tensor)
        assert isinstance(test_X, torch.Tensor)
        assert isinstance(test_y, torch.Tensor)
        
        assert train_X.shape == (1000, 3, 32, 32)
        assert train_y.shape == (1000,)
        assert test_X.shape == (200, 3, 32, 32)
        assert test_y.shape == (200,)
        
        # Check normalization
        assert torch.all(train_X >= 0.0) and torch.all(train_X <= 1.0)
        assert torch.all(test_X >= 0.0) and torch.all(test_X <= 1.0)
        
        # Check label ranges
        assert torch.all(train_y >= 0) and torch.all(train_y < 100)
        assert torch.all(test_y >= 0) and torch.all(test_y < 100)
    
    @patch('src.data_utils.dataset_utils._load_cifar100_pickle')
    @patch('pathlib.Path.exists')
    def test_full_pipeline_numpy(self, mock_exists, mock_load_pickle):
        """Test the full data loading pipeline with numpy format."""
        mock_exists.return_value = True
        
        # Create realistic data
        train_data = {
            b'data': np.random.randint(0, 256, (500, 3072), dtype=np.uint8),
            b'fine_labels': np.random.randint(0, 100, 500).tolist()
        }
        
        test_data = {
            b'data': np.random.randint(0, 256, (100, 3072), dtype=np.uint8),
            b'fine_labels': np.random.randint(0, 100, 100).tolist()
        }
        
        mock_load_pickle.side_effect = [train_data, test_data]
        
        # Load data
        train_X, train_y, test_X, test_y = load_cifar100_data(
            data_dir="./data/cifar-100",
            return_type='numpy',
            normalize=False,
            verbose=False
        )
        
        # Verify complete pipeline
        assert isinstance(train_X, np.ndarray)
        assert isinstance(train_y, np.ndarray)
        assert isinstance(test_X, np.ndarray)
        assert isinstance(test_y, np.ndarray)
        
        assert train_X.shape == (500, 3, 32, 32)
        assert train_y.shape == (500,)
        assert test_X.shape == (100, 3, 32, 32)
        assert test_y.shape == (100,)
        
        # Check no normalization (values in [0, 255])
        assert np.all(train_X >= 0.0) and np.all(train_X <= 255.0)
        assert np.all(test_X >= 0.0) and np.all(test_X <= 255.0)


if __name__ == "__main__":
    pytest.main([__file__])
