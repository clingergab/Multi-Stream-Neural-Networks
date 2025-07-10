#!/usr/bin/env python3
"""
Comprehensive unit tests for multi-channel conv.py module.
Tests MCConv2d and MCBatchNorm2d classes.
"""

import unittest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent / "src"))

from src.models2.multi_channel.conv import MCConv2d, MCBatchNorm2d


class TestMCConv2d(unittest.TestCase):
    """Test cases for MCConv2d layer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.color_channels = 3
        self.brightness_channels = 1
        self.out_channels_color = 64
        self.out_channels_brightness = 32
        
        self.conv = MCConv2d(
            color_in_channels=self.color_channels,
            brightness_in_channels=self.brightness_channels,
            color_out_channels=self.out_channels_color,
            brightness_out_channels=self.out_channels_brightness,
            kernel_size=3,
            padding=1,
            bias=False  # Set bias=False for this test
        )
        
        # Test inputs
        self.batch_size = 4
        self.input_size = 32
        self.color_input = torch.randn(self.batch_size, self.color_channels, self.input_size, self.input_size)
        self.brightness_input = torch.randn(self.batch_size, self.brightness_channels, self.input_size, self.input_size)
    
    def test_initialization(self):
        """Test proper initialization of MCConv2d."""
        # Check attributes
        self.assertEqual(self.conv.color_in_channels, self.color_channels)
        self.assertEqual(self.conv.brightness_in_channels, self.brightness_channels)
        self.assertEqual(self.conv.color_out_channels, self.out_channels_color)
        self.assertEqual(self.conv.brightness_out_channels, self.out_channels_brightness)
        
        # Check weight shapes
        expected_color_weight_shape = (self.out_channels_color, self.color_channels, 3, 3)
        expected_brightness_weight_shape = (self.out_channels_brightness, self.brightness_channels, 3, 3)
        
        self.assertEqual(self.conv.color_weight.shape, expected_color_weight_shape)
        self.assertEqual(self.conv.brightness_weight.shape, expected_brightness_weight_shape)
        
        # Check bias is None by default
        self.assertIsNone(self.conv.bias)
    
    def test_forward_dual_channel(self):
        """Test forward pass with both color and brightness inputs."""
        color_out, brightness_out = self.conv(self.color_input, self.brightness_input)
        
        # Check output shapes
        expected_color_shape = (self.batch_size, self.out_channels_color, self.input_size, self.input_size)
        expected_brightness_shape = (self.batch_size, self.out_channels_brightness, self.input_size, self.input_size)
        
        self.assertEqual(color_out.shape, expected_color_shape)
        self.assertEqual(brightness_out.shape, expected_brightness_shape)
        
        # Check outputs are tensors
        self.assertIsInstance(color_out, torch.Tensor)
        self.assertIsInstance(brightness_out, torch.Tensor)
    
    def test_forward_color_only(self):
        """Test forward pass through color pathway only."""
        color_out = self.conv.forward_color(self.color_input)
        
        expected_shape = (self.batch_size, self.out_channels_color, self.input_size, self.input_size)
        self.assertEqual(color_out.shape, expected_shape)
        self.assertIsInstance(color_out, torch.Tensor)
    
    def test_forward_brightness_only(self):
        """Test forward pass through brightness pathway only."""
        brightness_out = self.conv.forward_brightness(self.brightness_input)
        
        expected_shape = (self.batch_size, self.out_channels_brightness, self.input_size, self.input_size)
        self.assertEqual(brightness_out.shape, expected_shape)
        self.assertIsInstance(brightness_out, torch.Tensor)
    
    def test_stride_and_padding(self):
        """Test convolution with different stride and padding."""
        conv_stride = MCConv2d(
            color_in_channels=3,
            brightness_in_channels=1,
            color_out_channels=64,
            brightness_out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1
        )
        
        color_out, brightness_out = conv_stride(self.color_input, self.brightness_input)
        
        # With stride=2, output should be half the input size
        expected_size = self.input_size // 2
        expected_color_shape = (self.batch_size, 64, expected_size, expected_size)
        expected_brightness_shape = (self.batch_size, 32, expected_size, expected_size)
        
        self.assertEqual(color_out.shape, expected_color_shape)
        self.assertEqual(brightness_out.shape, expected_brightness_shape)
    
    def test_with_bias(self):
        """Test convolution with bias enabled."""
        conv_bias = MCConv2d(
            color_in_channels=3,
            brightness_in_channels=1,
            color_out_channels=64,
            brightness_out_channels=32,
            kernel_size=1,
            bias=True
        )
        
        # Check bias parameters exist
        self.assertIsNotNone(conv_bias.color_bias)
        self.assertIsNotNone(conv_bias.brightness_bias)
        
        # Check bias shapes
        self.assertEqual(conv_bias.color_bias.shape, (64,))
        self.assertEqual(conv_bias.brightness_bias.shape, (32,))
        
        # Test forward pass
        color_out, brightness_out = conv_bias(self.color_input, self.brightness_input)
        self.assertIsInstance(color_out, torch.Tensor)
        self.assertIsInstance(brightness_out, torch.Tensor)
    
    def test_gradient_flow(self):
        """Test that gradients flow correctly through both pathways."""
        # Set requires_grad for inputs
        color_input = self.color_input.clone().requires_grad_(True)
        brightness_input = self.brightness_input.clone().requires_grad_(True)
        
        # Forward pass
        color_out, brightness_out = self.conv(color_input, brightness_input)
        
        # Backward pass
        loss = color_out.sum() + brightness_out.sum()
        loss.backward()
        
        # Check gradients exist
        self.assertIsNotNone(color_input.grad)
        self.assertIsNotNone(brightness_input.grad)
        self.assertIsNotNone(self.conv.color_weight.grad)
        self.assertIsNotNone(self.conv.brightness_weight.grad)


class TestMCBatchNorm2d(unittest.TestCase):
    """Test cases for MCBatchNorm2d layer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.color_channels = 64
        self.brightness_channels = 32
        
        self.bn = MCBatchNorm2d(
            color_num_features=self.color_channels,
            brightness_num_features=self.brightness_channels
        )
        
        # Test inputs
        self.batch_size = 4
        self.input_size = 32
        self.color_input = torch.randn(self.batch_size, self.color_channels, self.input_size, self.input_size)
        self.brightness_input = torch.randn(self.batch_size, self.brightness_channels, self.input_size, self.input_size)
    
    def test_initialization(self):
        """Test proper initialization of MCBatchNorm2d."""
        # Check attributes
        self.assertEqual(self.bn.color_num_features, self.color_channels)
        self.assertEqual(self.bn.brightness_num_features, self.brightness_channels)
        
        # Check parameter shapes
        self.assertEqual(self.bn.color_weight.shape, (self.color_channels,))
        self.assertEqual(self.bn.brightness_weight.shape, (self.brightness_channels,))
        self.assertEqual(self.bn.color_bias.shape, (self.color_channels,))
        self.assertEqual(self.bn.brightness_bias.shape, (self.brightness_channels,))
        
        # Check running statistics shapes
        self.assertEqual(self.bn.color_running_mean.shape, (self.color_channels,))
        self.assertEqual(self.bn.brightness_running_mean.shape, (self.brightness_channels,))
        self.assertEqual(self.bn.color_running_var.shape, (self.color_channels,))
        self.assertEqual(self.bn.brightness_running_var.shape, (self.brightness_channels,))
    
    def test_forward_dual_channel(self):
        """Test forward pass with both color and brightness inputs."""
        color_out, brightness_out = self.bn(self.color_input, self.brightness_input)
        
        # Check output shapes match input shapes
        self.assertEqual(color_out.shape, self.color_input.shape)
        self.assertEqual(brightness_out.shape, self.brightness_input.shape)
        
        # Check outputs are tensors
        self.assertIsInstance(color_out, torch.Tensor)
        self.assertIsInstance(brightness_out, torch.Tensor)
    
    def test_forward_color_only(self):
        """Test forward pass through color pathway only."""
        color_out = self.bn.forward_color(self.color_input)
        
        self.assertEqual(color_out.shape, self.color_input.shape)
        self.assertIsInstance(color_out, torch.Tensor)
    
    def test_forward_brightness_only(self):
        """Test forward pass through brightness pathway only."""
        brightness_out = self.bn.forward_brightness(self.brightness_input)
        
        self.assertEqual(brightness_out.shape, self.brightness_input.shape)
        self.assertIsInstance(brightness_out, torch.Tensor)
    
    def test_training_mode(self):
        """Test batch normalization in training mode."""
        self.bn.train()
        
        # Store initial running statistics
        initial_color_mean = self.bn.color_running_mean.clone()
        initial_brightness_mean = self.bn.brightness_running_mean.clone()
        
        # Forward pass
        color_out, brightness_out = self.bn(self.color_input, self.brightness_input)
        
        # Running statistics should be updated
        self.assertFalse(torch.equal(initial_color_mean, self.bn.color_running_mean))
        self.assertFalse(torch.equal(initial_brightness_mean, self.bn.brightness_running_mean))
    
    def test_eval_mode(self):
        """Test batch normalization in evaluation mode."""
        self.bn.eval()
        
        # Store running statistics
        initial_color_mean = self.bn.color_running_mean.clone()
        initial_brightness_mean = self.bn.brightness_running_mean.clone()
        
        # Forward pass
        color_out, brightness_out = self.bn(self.color_input, self.brightness_input)
        
        # Running statistics should not change in eval mode
        self.assertTrue(torch.equal(initial_color_mean, self.bn.color_running_mean))
        self.assertTrue(torch.equal(initial_brightness_mean, self.bn.brightness_running_mean))
    
    def test_gradient_flow(self):
        """Test that gradients flow correctly through both pathways."""
        # Set requires_grad for inputs
        color_input = self.color_input.clone().requires_grad_(True)
        brightness_input = self.brightness_input.clone().requires_grad_(True)
        
        # Forward pass
        color_out, brightness_out = self.bn(color_input, brightness_input)
        
        # Backward pass
        loss = color_out.sum() + brightness_out.sum()
        loss.backward()
        
        # Check gradients exist
        self.assertIsNotNone(color_input.grad)
        self.assertIsNotNone(brightness_input.grad)
        self.assertIsNotNone(self.bn.color_weight.grad)
        self.assertIsNotNone(self.bn.brightness_weight.grad)


class TestMCConv2dErrorConditions(unittest.TestCase):
    """Test error conditions and edge cases for MCConv2d."""
    
    def test_invalid_groups(self):
        """Test invalid groups parameter."""
        with self.assertRaises(ValueError) as cm:
            MCConv2d(3, 1, 64, 32, kernel_size=3, groups=0)
        self.assertIn("groups must be a positive integer", str(cm.exception))
        
        with self.assertRaises(ValueError) as cm:
            MCConv2d(3, 1, 64, 32, kernel_size=3, groups=-1)
        self.assertIn("groups must be a positive integer", str(cm.exception))
    
    def test_channels_not_divisible_by_groups(self):
        """Test channels not divisible by groups."""
        # Color in channels not divisible by groups
        with self.assertRaises(ValueError) as cm:
            MCConv2d(5, 2, 8, 4, kernel_size=3, groups=2)
        self.assertIn("color_in_channels must be divisible by groups", str(cm.exception))
        
        # Brightness in channels not divisible by groups
        with self.assertRaises(ValueError) as cm:
            MCConv2d(4, 3, 8, 4, kernel_size=3, groups=2)
        self.assertIn("brightness_in_channels must be divisible by groups", str(cm.exception))
        
        # Color out channels not divisible by groups
        with self.assertRaises(ValueError) as cm:
            MCConv2d(4, 2, 7, 4, kernel_size=3, groups=2)
        self.assertIn("color_out_channels must be divisible by groups", str(cm.exception))
        
        # Brightness out channels not divisible by groups
        with self.assertRaises(ValueError) as cm:
            MCConv2d(4, 2, 8, 3, kernel_size=3, groups=2)
        self.assertIn("brightness_out_channels must be divisible by groups", str(cm.exception))
    
    def test_invalid_padding_string(self):
        """Test invalid padding string."""
        with self.assertRaises(ValueError) as cm:
            MCConv2d(3, 1, 64, 32, kernel_size=3, padding="invalid")
        self.assertIn("Invalid padding string", str(cm.exception))
    
    def test_same_padding_with_stride(self):
        """Test 'same' padding with stride > 1."""
        with self.assertRaises(ValueError) as cm:
            MCConv2d(3, 1, 64, 32, kernel_size=3, padding="same", stride=2)
        self.assertIn("padding='same' is not supported for strided convolutions", str(cm.exception))
    
    def test_valid_padding_strings(self):
        """Test valid padding strings."""
        # Test 'same' padding with stride=1
        conv_same = MCConv2d(3, 1, 64, 32, kernel_size=3, padding="same", stride=1)
        self.assertIsInstance(conv_same, MCConv2d)
        
        # Test 'valid' padding
        conv_valid = MCConv2d(3, 1, 64, 32, kernel_size=3, padding="valid")
        self.assertIsInstance(conv_valid, MCConv2d)
    
    def test_device_and_dtype_kwargs(self):
        """Test device and dtype factory kwargs."""
        if torch.cuda.is_available():
            conv_cuda = MCConv2d(3, 1, 64, 32, kernel_size=3, device='cuda')
            self.assertTrue(conv_cuda.color_weight.is_cuda)
            self.assertTrue(conv_cuda.brightness_weight.is_cuda)
        
        # Test different dtypes
        conv_float64 = MCConv2d(3, 1, 64, 32, kernel_size=3, dtype=torch.float64)
        self.assertEqual(conv_float64.color_weight.dtype, torch.float64)
        self.assertEqual(conv_float64.brightness_weight.dtype, torch.float64)


class TestMCBatchNorm2dErrorConditions(unittest.TestCase):
    """Test error conditions and edge cases for MCBatchNorm2d."""
    
    def test_device_and_dtype_kwargs(self):
        """Test device and dtype factory kwargs."""
        if torch.cuda.is_available():
            bn_cuda = MCBatchNorm2d(64, 32, device='cuda')
            self.assertTrue(bn_cuda.color_weight.is_cuda)
            self.assertTrue(bn_cuda.brightness_weight.is_cuda)
        
        # Test different dtypes
        bn_float64 = MCBatchNorm2d(64, 32, dtype=torch.float64)
        self.assertEqual(bn_float64.color_weight.dtype, torch.float64)
        self.assertEqual(bn_float64.brightness_weight.dtype, torch.float64)
    
    def test_affine_false(self):
        """Test batch norm with affine=False."""
        bn_no_affine = MCBatchNorm2d(64, 32, affine=False)
        
        # Should not have weight and bias parameters when affine=False
        self.assertIsNone(bn_no_affine.color_weight)
        self.assertIsNone(bn_no_affine.brightness_weight)
        self.assertIsNone(bn_no_affine.color_bias)
        self.assertIsNone(bn_no_affine.brightness_bias)
    
    def test_track_running_stats_false(self):
        """Test batch norm with track_running_stats=False."""
        bn_no_stats = MCBatchNorm2d(64, 32, track_running_stats=False)
        
        # Should not have running mean and var when track_running_stats=False
        self.assertIsNone(bn_no_stats.color_running_mean)
        self.assertIsNone(bn_no_stats.brightness_running_mean)
        self.assertIsNone(bn_no_stats.color_running_var)
        self.assertIsNone(bn_no_stats.brightness_running_var)
    
    def test_both_false(self):
        """Test batch norm with both affine=False and track_running_stats=False."""
        bn_minimal = MCBatchNorm2d(64, 32, affine=False, track_running_stats=False)
        
        # Should not have any parameters or buffers
        self.assertIsNone(bn_minimal.color_weight)
        self.assertIsNone(bn_minimal.brightness_weight)
        self.assertIsNone(bn_minimal.color_bias)
        self.assertIsNone(bn_minimal.brightness_bias)
        self.assertIsNone(bn_minimal.color_running_mean)
        self.assertIsNone(bn_minimal.brightness_running_mean)
        self.assertIsNone(bn_minimal.color_running_var)
        self.assertIsNone(bn_minimal.brightness_running_var)


if __name__ == "__main__":
    unittest.main()
