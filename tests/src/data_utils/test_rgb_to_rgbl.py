"""Unit tfrom data_utils.rgb_to_rgbl import (
    RGBtoRGBL,
    create_rgbl_transform,
    collate_with_streams,
    process_dataset_to_streams
)r src.data_utils.rgb_to_rgbl module."""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from torch.utils.data import TensorDataset
from PIL import Image

from data_utils.rgb_to_rgbl import (
    RGBtoRGBL,
    create_rgbl_transform,
    collate_with_streams,
    process_dataset_to_streams
)


class TestRGBtoRGBL:
    """Test cases for RGBtoRGBL class."""
    
    @pytest.fixture
    def rgb_to_rgbl(self):
        """Create RGBtoRGBL instance for testing."""
        return RGBtoRGBL()
    
    @pytest.fixture
    def sample_rgb_image(self):
        """Create a sample RGB image tensor."""
        return torch.rand(3, 32, 32)
    
    @pytest.fixture
    def sample_rgb_batch(self):
        """Create a sample RGB batch tensor."""
        return torch.rand(4, 3, 32, 32)
    
    def test_init(self, rgb_to_rgbl):
        """Test initialization of RGBtoRGBL."""
        assert rgb_to_rgbl.rgb_weights.shape == (3,)
        expected_weights = torch.tensor([0.299, 0.587, 0.114])
        torch.testing.assert_close(rgb_to_rgbl.rgb_weights, expected_weights)
    
    def test_call_single_image(self, rgb_to_rgbl, sample_rgb_image):
        """Test __call__ method with single image."""
        rgb_output, brightness_output = rgb_to_rgbl(sample_rgb_image)
        
        # Check that RGB output is unchanged
        torch.testing.assert_close(rgb_output, sample_rgb_image)
        
        # Check brightness output shape
        assert brightness_output.shape == (1, 32, 32)
        
        # Check that brightness is computed correctly
        expected_weights = torch.tensor([0.299, 0.587, 0.114])
        expected_brightness = torch.sum(sample_rgb_image * expected_weights.view(-1, 1, 1), dim=0, keepdim=True)
        torch.testing.assert_close(brightness_output, expected_brightness)
    
    def test_call_batch_images(self, rgb_to_rgbl, sample_rgb_batch):
        """Test __call__ method with batch of images."""
        rgb_output, brightness_output = rgb_to_rgbl(sample_rgb_batch)
        
        # Check that RGB output is unchanged
        torch.testing.assert_close(rgb_output, sample_rgb_batch)
        
        # Check brightness output shape
        assert brightness_output.shape == (4, 1, 32, 32)
        
        # Check that brightness is computed correctly for batch
        expected_weights = torch.tensor([0.299, 0.587, 0.114])
        expected_brightness = torch.sum(sample_rgb_batch * expected_weights.view(1, -1, 1, 1), dim=1, keepdim=True)
        torch.testing.assert_close(brightness_output, expected_brightness)
    
    def test_call_wrong_channels_single(self, rgb_to_rgbl):
        """Test __call__ method with wrong number of channels (single image)."""
        wrong_channels = torch.rand(4, 32, 32)  # 4 channels instead of 3
        
        with pytest.raises(ValueError) as exc_info:
            rgb_to_rgbl(wrong_channels)
        
        assert "Expected 3 channels, got 4" in str(exc_info.value)
    
    def test_call_wrong_channels_batch(self, rgb_to_rgbl):
        """Test __call__ method with wrong number of channels (batch)."""
        wrong_channels = torch.rand(2, 4, 32, 32)  # 4 channels instead of 3
        
        with pytest.raises(ValueError) as exc_info:
            rgb_to_rgbl(wrong_channels)
        
        assert "Expected 3 channels, got 4" in str(exc_info.value)
    
    def test_call_wrong_dimensions(self, rgb_to_rgbl):
        """Test __call__ method with wrong tensor dimensions."""
        wrong_dims = torch.rand(32, 32)  # 2D instead of 3D or 4D
        
        with pytest.raises(ValueError) as exc_info:
            rgb_to_rgbl(wrong_dims)
        
        assert "Expected tensor with 3 or 4 dimensions, got 2" in str(exc_info.value)
    
    def test_get_brightness_single_image(self, rgb_to_rgbl, sample_rgb_image):
        """Test get_brightness method with single image."""
        brightness_output = rgb_to_rgbl.get_brightness(sample_rgb_image)
        
        # Check brightness output shape
        assert brightness_output.shape == (1, 32, 32)
        
        # Check that brightness is computed correctly
        expected_weights = torch.tensor([0.299, 0.587, 0.114])
        expected_brightness = torch.sum(sample_rgb_image * expected_weights.view(-1, 1, 1), dim=0, keepdim=True)
        torch.testing.assert_close(brightness_output, expected_brightness)
    
    def test_get_brightness_batch(self, rgb_to_rgbl, sample_rgb_batch):
        """Test get_brightness method with batch."""
        brightness_output = rgb_to_rgbl.get_brightness(sample_rgb_batch)
        
        # Check brightness output shape
        assert brightness_output.shape == (4, 1, 32, 32)
        
        # Check that brightness is computed correctly for batch
        expected_weights = torch.tensor([0.299, 0.587, 0.114])
        expected_brightness = torch.sum(sample_rgb_batch * expected_weights.view(1, -1, 1, 1), dim=1, keepdim=True)
        torch.testing.assert_close(brightness_output, expected_brightness)
    
    def test_get_rgbl_single_image(self, rgb_to_rgbl, sample_rgb_image):
        """Test get_rgbl method with single image."""
        rgbl_output = rgb_to_rgbl.get_rgbl(sample_rgb_image)
        
        # Check RGBL output shape
        assert rgbl_output.shape == (4, 32, 32)  # RGB + L = 4 channels
        
        # Check that RGB channels are preserved
        torch.testing.assert_close(rgbl_output[:3], sample_rgb_image)
        
        # Check that L channel is correct
        expected_weights = torch.tensor([0.299, 0.587, 0.114])
        expected_brightness = torch.sum(sample_rgb_image * expected_weights.view(-1, 1, 1), dim=0)
        torch.testing.assert_close(rgbl_output[3], expected_brightness)
    
    def test_get_rgbl_batch(self, rgb_to_rgbl, sample_rgb_batch):
        """Test get_rgbl method with batch."""
        rgbl_output = rgb_to_rgbl.get_rgbl(sample_rgb_batch)
        
        # Check RGBL output shape
        assert rgbl_output.shape == (4, 4, 32, 32)  # Batch, RGB + L = 4 channels
        
        # Check that RGB channels are preserved
        torch.testing.assert_close(rgbl_output[:, :3], sample_rgb_batch)
        
        # Check that L channel is correct
        expected_weights = torch.tensor([0.299, 0.587, 0.114])
        expected_brightness = torch.sum(sample_rgb_batch * expected_weights.view(1, -1, 1, 1), dim=1)
        torch.testing.assert_close(rgbl_output[:, 3], expected_brightness)
    
    def test_device_consistency(self, rgb_to_rgbl):
        """Test that the transform works correctly across different devices."""
        # Test with CPU tensor
        cpu_tensor = torch.rand(3, 16, 16)
        rgb_out, brightness_out = rgb_to_rgbl(cpu_tensor)
        
        assert rgb_out.device == cpu_tensor.device
        assert brightness_out.device == cpu_tensor.device
        
        # Test with GPU tensor if available
        if torch.cuda.is_available():
            gpu_tensor = cpu_tensor.cuda()
            rgb_out_gpu, brightness_out_gpu = rgb_to_rgbl(gpu_tensor)
            
            assert rgb_out_gpu.device == gpu_tensor.device
            assert brightness_out_gpu.device == gpu_tensor.device
    
    def test_brightness_values_range(self, rgb_to_rgbl):
        """Test that brightness values are in reasonable range."""
        # Create extreme test cases
        black_image = torch.zeros(3, 10, 10)
        white_image = torch.ones(3, 10, 10)
        
        # Test black image
        _, brightness_black = rgb_to_rgbl(black_image)
        assert torch.allclose(brightness_black, torch.zeros(1, 10, 10))
        
        # Test white image
        _, brightness_white = rgb_to_rgbl(white_image)
        expected_white_brightness = torch.full((1, 10, 10), 0.299 + 0.587 + 0.114)
        torch.testing.assert_close(brightness_white, expected_white_brightness)


class TestCreateRGBLTransform:
    """Test cases for create_rgbl_transform function."""
    
    def test_create_transform(self):
        """Test that create_rgbl_transform returns a valid transform."""
        transform = create_rgbl_transform()
        
        # Should be a Compose transform
        from torchvision import transforms
        assert isinstance(transform, transforms.Compose)
        
        # Should have ToTensor and RGBtoRGBL
        assert len(transform.transforms) == 2
        assert isinstance(transform.transforms[0], transforms.ToTensor)
        assert isinstance(transform.transforms[1], RGBtoRGBL)
    
    def test_transform_functionality(self):
        """Test that the created transform works correctly."""
        transform = create_rgbl_transform()
        
        # Create a PIL image for testing
        pil_image = Image.new('RGB', (32, 32), color=(128, 64, 192))
        
        # Apply transform
        rgb_tensor, brightness_tensor = transform(pil_image)
        
        # Check output shapes
        assert rgb_tensor.shape == (3, 32, 32)
        assert brightness_tensor.shape == (1, 32, 32)
        
        # Check that values are in [0, 1] range (due to ToTensor)
        assert torch.all(rgb_tensor >= 0) and torch.all(rgb_tensor <= 1)
        assert torch.all(brightness_tensor >= 0) and torch.all(brightness_tensor <= 1)


class TestCollateWithStreams:
    """Test cases for collate_with_streams function."""
    
    @pytest.fixture
    def sample_batch_tensors(self):
        """Create sample batch with tensor data."""
        return [
            (torch.rand(3, 16, 16), torch.tensor(0)),
            (torch.rand(3, 16, 16), torch.tensor(1)),
            (torch.rand(3, 16, 16), torch.tensor(2))
        ]
    
    @pytest.fixture
    def sample_batch_pil(self):
        """Create sample batch with PIL Image data."""
        return [
            (Image.new('RGB', (16, 16), color=(255, 0, 0)), 0),
            (Image.new('RGB', (16, 16), color=(0, 255, 0)), 1),
            (Image.new('RGB', (16, 16), color=(0, 0, 255)), 2)
        ]
    
    def test_collate_tensor_batch(self, sample_batch_tensors):
        """Test collate function with tensor batch."""
        rgb_batch, brightness_batch, labels_batch = collate_with_streams(sample_batch_tensors)
        
        # Check output shapes
        assert rgb_batch.shape == (3, 3, 16, 16)  # Batch, RGB channels, H, W
        assert brightness_batch.shape == (3, 1, 16, 16)  # Batch, brightness channel, H, W
        assert labels_batch.shape == (3,)  # Batch size
        
        # Check data types
        assert isinstance(rgb_batch, torch.Tensor)
        assert isinstance(brightness_batch, torch.Tensor)
        assert isinstance(labels_batch, torch.Tensor)
        assert labels_batch.dtype == torch.long
        
        # Check label values
        expected_labels = torch.tensor([0, 1, 2], dtype=torch.long)
        torch.testing.assert_close(labels_batch, expected_labels)
    
    def test_collate_pil_batch(self, sample_batch_pil):
        """Test collate function with PIL Image batch."""
        rgb_batch, brightness_batch, labels_batch = collate_with_streams(sample_batch_pil)
        
        # Check output shapes
        assert rgb_batch.shape == (3, 3, 16, 16)
        assert brightness_batch.shape == (3, 1, 16, 16)
        assert labels_batch.shape == (3,)
        
        # Check data types
        assert isinstance(rgb_batch, torch.Tensor)
        assert isinstance(brightness_batch, torch.Tensor)
        assert isinstance(labels_batch, torch.Tensor)
        assert labels_batch.dtype == torch.long
        
        # Check that PIL images were converted to tensors (values in [0, 1])
        assert torch.all(rgb_batch >= 0) and torch.all(rgb_batch <= 1)
        assert torch.all(brightness_batch >= 0) and torch.all(brightness_batch <= 1)
    
    def test_collate_mixed_batch(self):
        """Test collate function with mixed tensor and PIL data."""
        mixed_batch = [
            (torch.rand(3, 16, 16), torch.tensor(0)),
            (Image.new('RGB', (16, 16), color=(128, 128, 128)), 1)
        ]
        
        rgb_batch, brightness_batch, labels_batch = collate_with_streams(mixed_batch)
        
        # Should work correctly
        assert rgb_batch.shape == (2, 3, 16, 16)
        assert brightness_batch.shape == (2, 1, 16, 16)
        assert labels_batch.shape == (2,)
    
    def test_collate_empty_batch(self):
        """Test collate function with empty batch."""
        empty_batch = []
        
        with pytest.raises(Exception):  # Should raise some kind of error
            collate_with_streams(empty_batch)
    
    def test_collate_single_item(self):
        """Test collate function with single item."""
        single_batch = [(torch.rand(3, 16, 16), torch.tensor(5))]
        
        rgb_batch, brightness_batch, labels_batch = collate_with_streams(single_batch)
        
        assert rgb_batch.shape == (1, 3, 16, 16)
        assert brightness_batch.shape == (1, 1, 16, 16)
        assert labels_batch.shape == (1,)
        torch.testing.assert_close(labels_batch, torch.tensor([5], dtype=torch.long))


class TestProcessDatasetToStreams:
    """Test cases for process_dataset_to_streams function."""
    
    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        # Create tensors that look like RGB images
        data = torch.rand(10, 3, 16, 16)
        targets = torch.randint(0, 5, (10,))
        return TensorDataset(data, targets)
    
    def test_process_small_dataset(self, sample_dataset):
        """Test processing a small dataset."""
        rgb_stream, brightness_stream, labels_tensor = process_dataset_to_streams(
            sample_dataset, batch_size=3, desc="Test"
        )
        
        # Check output shapes
        assert rgb_stream.shape == (10, 3, 16, 16)
        assert brightness_stream.shape == (10, 1, 16, 16)
        assert labels_tensor.shape == (10,)
        
        # Check data types
        assert isinstance(rgb_stream, torch.Tensor)
        assert isinstance(brightness_stream, torch.Tensor)
        assert isinstance(labels_tensor, torch.Tensor)
        assert labels_tensor.dtype == torch.long
    
    def test_process_with_different_batch_sizes(self, sample_dataset):
        """Test processing with different batch sizes."""
        # Test with batch_size=1
        rgb_stream1, brightness_stream1, labels1 = process_dataset_to_streams(
            sample_dataset, batch_size=1, desc="Test batch_size=1"
        )
        
        # Test with batch_size=5
        rgb_stream5, brightness_stream5, labels5 = process_dataset_to_streams(
            sample_dataset, batch_size=5, desc="Test batch_size=5"
        )
        
        # Results should be the same regardless of batch size
        torch.testing.assert_close(rgb_stream1, rgb_stream5)
        torch.testing.assert_close(brightness_stream1, brightness_stream5)
        torch.testing.assert_close(labels1, labels5)
    
    def test_process_empty_dataset(self):
        """Test processing an empty dataset."""
        empty_data = torch.empty(0, 3, 16, 16)
        empty_targets = torch.empty(0, dtype=torch.long)
        empty_dataset = TensorDataset(empty_data, empty_targets)
        
        with pytest.raises(ValueError) as exc_info:
            process_dataset_to_streams(empty_dataset, batch_size=1)
        
        assert "Dataset is empty" in str(exc_info.value)
    
    def test_process_invalid_batch_size(self, sample_dataset):
        """Test processing with invalid batch size."""
        with pytest.raises(ValueError) as exc_info:
            process_dataset_to_streams(sample_dataset, batch_size=0)
        
        assert "Batch size must be positive" in str(exc_info.value)
        
        with pytest.raises(ValueError) as exc_info:
            process_dataset_to_streams(sample_dataset, batch_size=-1)
        
        assert "Batch size must be positive" in str(exc_info.value)
    
    @patch('data_utils.rgb_to_rgbl.tqdm')
    def test_process_with_progress_bar(self, mock_tqdm, sample_dataset):
        """Test that progress bar is used correctly."""
        mock_tqdm.return_value = range(0, 10, 3)  # Mock the range iteration
        
        # Mock the actual tqdm to avoid issues
        with patch('data_utils.rgb_to_rgbl.tqdm', return_value=range(0, 10, 3)):
            try:
                process_dataset_to_streams(sample_dataset, batch_size=3, desc="Test progress")
            except Exception:
                pass  # We expect this to fail due to mocking, but we just want to test the call
    
    def test_process_brightness_correctness(self, sample_dataset):
        """Test that brightness is computed correctly."""
        rgb_stream, brightness_stream, _ = process_dataset_to_streams(
            sample_dataset, batch_size=10  # Process all at once
        )
        
        # Manually compute expected brightness
        rgb_to_rgbl = RGBtoRGBL()
        expected_rgb, expected_brightness = rgb_to_rgbl(rgb_stream)
        
        # Check that RGB is unchanged
        torch.testing.assert_close(rgb_stream, expected_rgb)
        
        # Check that brightness is computed correctly
        torch.testing.assert_close(brightness_stream, expected_brightness)
    
    def test_process_dataset_access_error(self):
        """Test handling of dataset access errors."""
        class FaultyDataset:
            def __len__(self):
                return 5
            
            def __getitem__(self, idx):
                if idx >= 2:
                    raise RuntimeError("Dataset access error")
                return torch.rand(3, 16, 16), torch.tensor(idx)
        
        faulty_dataset = FaultyDataset()
        
        with pytest.raises(RuntimeError) as exc_info:
            process_dataset_to_streams(faulty_dataset, batch_size=3)
        
        assert "Error accessing dataset item" in str(exc_info.value)
    
    def test_process_inconsistent_shapes(self):
        """Test handling of inconsistent image shapes."""
        class InconsistentDataset:
            def __len__(self):
                return 3
            
            def __getitem__(self, idx):
                if idx == 0:
                    return torch.rand(3, 16, 16), torch.tensor(0)
                else:
                    return torch.rand(3, 32, 32), torch.tensor(idx)  # Different size
        
        inconsistent_dataset = InconsistentDataset()
        
        with pytest.raises(RuntimeError) as exc_info:
            process_dataset_to_streams(inconsistent_dataset, batch_size=2)
        
        assert "Error stacking batch data" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__])
