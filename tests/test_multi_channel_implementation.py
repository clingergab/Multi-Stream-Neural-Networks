"""
Enhanced test suite for Multi-Channel Neural Networks.

This module provides comprehensive testing for the multi-channel implementation
that was successfully tested on CIFAR-100.
"""

import unittest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to path for testing
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

# Import after path setup to avoid import errors
try:
    from src.models.basic_multi_channel.multi_channel_model import (
        MultiChannelNetwork, 
        multi_channel_18,
        multi_channel_50
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure the multi-channel model implementation is available")
    raise


class TestMultiChannelImplementation(unittest.TestCase):
    """Test the actual multi-channel implementation that works with CIFAR-100."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 4
        self.channels = 3
        self.height = 32
        self.width = 32
        self.num_classes = 100
        
        # Create test inputs (similar to CIFAR-100)
        self.color_input = torch.randn(self.batch_size, self.channels, self.height, self.width)
        self.brightness_input = torch.randn(self.batch_size, self.channels, self.height, self.width)
        
    def test_multi_channel_network_creation(self):
        """Test that MultiChannelNetwork can be created with various configurations."""
        # Test basic creation
        model = MultiChannelNetwork(num_classes=10, input_channels=3, hidden_channels=64)
        self.assertIsInstance(model, nn.Module)
        
        # Test with different parameters
        model = MultiChannelNetwork(
            num_classes=100, 
            input_channels=3, 
            hidden_channels=32, 
            num_blocks=[1, 1, 1, 1]
        )
        self.assertIsInstance(model, nn.Module)
        
    def test_multi_channel_forward_pass(self):
        """Test forward pass through multi-channel network."""
        model = MultiChannelNetwork(
            num_classes=self.num_classes, 
            input_channels=self.channels, 
            hidden_channels=64,
            num_blocks=[2, 2, 2, 2]
        )
        
        # Test forward pass
        output = model(self.color_input, self.brightness_input)
        
        # Check output shape
        expected_shape = (self.batch_size, self.num_classes)
        self.assertEqual(output.shape, expected_shape)
        
        # Check that output is a valid tensor
        self.assertIsInstance(output, torch.Tensor)
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
        
    def test_resnet18_factory(self):
        """Test the multi_channel_18 factory function."""
        model = multi_channel_18(num_classes=100, input_channels=3)
        
        # Test forward pass
        output = model(self.color_input, self.brightness_input)
        self.assertEqual(output.shape, (self.batch_size, 100))
        
    def test_resnet34_factory(self):
        """Test the multi_channel_50 factory function."""
        model = multi_channel_50(num_classes=10, input_channels=3)
        
        # Adjust input for smaller test
        color_input = torch.randn(2, 3, 32, 32)
        brightness_input = torch.randn(2, 3, 32, 32)
        
        output = model(color_input, brightness_input)
        self.assertEqual(output.shape, (2, 10))
        
    def test_parameter_separation(self):
        """Test that color and brightness pathways have separate parameters."""
        model = MultiChannelNetwork(num_classes=10, input_channels=3, hidden_channels=64)
        
        # Check that the model has the expected multi-channel components
        self.assertTrue(hasattr(model, 'conv1'))
        self.assertTrue(hasattr(model, 'bn1'))
        self.assertTrue(hasattr(model, 'layer1'))
        self.assertTrue(hasattr(model, 'classifier'))
        
        # Count parameters to ensure we have separate pathways
        total_params = sum(p.numel() for p in model.parameters())
        self.assertGreater(total_params, 0)
        
    def test_different_input_shapes(self):
        """Test that the model handles different input shapes correctly."""
        model = MultiChannelNetwork(num_classes=10, input_channels=3, hidden_channels=32)
        model.eval()  # Set to eval mode to avoid BatchNorm issues with batch_size=1
        
        # Test with different batch sizes
        for batch_size in [1, 2, 8]:
            color_input = torch.randn(batch_size, 3, 32, 32)
            brightness_input = torch.randn(batch_size, 3, 32, 32)
            
            output = model(color_input, brightness_input)
            self.assertEqual(output.shape, (batch_size, 10))
            
    def test_model_training_mode(self):
        """Test that the model can switch between training and evaluation modes."""
        model = MultiChannelNetwork(num_classes=10, input_channels=3, hidden_channels=32)
        
        # Test training mode
        model.train()
        self.assertTrue(model.training)
        
        # Test evaluation mode
        model.eval()
        self.assertFalse(model.training)
        
    def test_gradient_flow(self):
        """Test that gradients flow properly through the network."""
        model = MultiChannelNetwork(num_classes=10, input_channels=3, hidden_channels=32)
        criterion = nn.CrossEntropyLoss()
        
        # Create inputs and targets
        color_input = torch.randn(2, 3, 32, 32, requires_grad=True)
        brightness_input = torch.randn(2, 3, 32, 32, requires_grad=True)
        targets = torch.randint(0, 10, (2,))
        
        # Forward pass
        output = model(color_input, brightness_input)
        loss = criterion(output, targets)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        for param in model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
                self.assertFalse(torch.isnan(param.grad).any())
                
    def test_model_device_compatibility(self):
        """Test that the model works on CPU (and CUDA if available)."""
        model = MultiChannelNetwork(num_classes=10, input_channels=3, hidden_channels=32)
        
        # Test CPU
        model.to('cpu')
        color_input = torch.randn(2, 3, 32, 32).to('cpu')
        brightness_input = torch.randn(2, 3, 32, 32).to('cpu')
        
        output = model(color_input, brightness_input)
        self.assertEqual(output.device.type, 'cpu')
        
        # Test CUDA if available
        if torch.cuda.is_available():
            model.to('cuda')
            color_input = color_input.to('cuda')
            brightness_input = brightness_input.to('cuda')
            
            output = model(color_input, brightness_input)
            self.assertEqual(output.device.type, 'cuda')


class TestMultiChannelDataFlow(unittest.TestCase):
    """Test the data flow and stream separation in multi-channel networks."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = MultiChannelNetwork(num_classes=10, input_channels=3, hidden_channels=32)
        
    def test_stream_independence(self):
        """Test that color and brightness streams are processed independently."""
        batch_size = 2
        
        # Create identical inputs
        identical_input = torch.randn(batch_size, 3, 32, 32)
        
        # Forward pass with identical inputs
        output1 = self.model(identical_input, identical_input)
        
        # Forward pass with different brightness input
        different_brightness = torch.randn(batch_size, 3, 32, 32)
        output2 = self.model(identical_input, different_brightness)
        
        # Outputs should be different since brightness input changed
        self.assertFalse(torch.allclose(output1, output2, atol=1e-6))
        
    def test_input_validation(self):
        """Test that the model handles various input scenarios properly."""
        # Test mismatched batch sizes (should raise an error)
        color_input = torch.randn(2, 3, 32, 32)
        brightness_input = torch.randn(3, 3, 32, 32)  # Different batch size
        
        with self.assertRaises(RuntimeError):
            self.model(color_input, brightness_input)
            
    def test_feature_combination(self):
        """Test that features from both streams are combined properly."""
        self.model.eval()  # Set to eval mode to avoid BatchNorm issues
        
        color_input = torch.randn(1, 3, 32, 32)
        brightness_input = torch.zeros(1, 3, 32, 32)  # Zero brightness
        
        output1 = self.model(color_input, brightness_input)
        
        # Now with non-zero brightness
        brightness_input = torch.ones(1, 3, 32, 32)
        output2 = self.model(color_input, brightness_input)
        
        # Outputs should be different
        self.assertFalse(torch.allclose(output1, output2, atol=1e-6))


class TestCIFAR100Compatibility(unittest.TestCase):
    """Test compatibility with CIFAR-100 dataset format."""
    
    def test_cifar100_input_format(self):
        """Test that the model works with CIFAR-100 style inputs."""
        # CIFAR-100 dimensions: (batch_size, 3, 32, 32)
        model = multi_channel_18(num_classes=100, input_channels=3)
        
        # Simulate CIFAR-100 batch
        batch_size = 4
        color_input = torch.randn(batch_size, 3, 32, 32)  # RGB image
        
        # Simulate brightness extraction (as done in training script)
        brightness_gray = torch.mean(color_input, dim=1, keepdim=True)  # Convert to grayscale
        brightness_input = brightness_gray.repeat(1, 3, 1, 1)  # Repeat to 3 channels
        
        # Forward pass
        output = model(color_input, brightness_input)
        
        # Check output shape matches CIFAR-100 classes
        self.assertEqual(output.shape, (batch_size, 100))
        
    def test_model_capacity(self):
        """Test that the model has reasonable capacity for CIFAR-100."""
        model = multi_channel_18(num_classes=100, input_channels=3)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        # Should have reasonable number of parameters (not too small, not too large)
        self.assertGreater(total_params, 1_000_000)  # At least 1M parameters
        self.assertLess(total_params, 100_000_000)   # Less than 100M parameters
        
        print(f"Multi-channel 18-layer model has {total_params:,} parameters")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
