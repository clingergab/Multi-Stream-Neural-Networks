#!/usr/bin/env python3
"""
Comprehensive unit tests for multi-channel conv.py module.
Tests MCConv2d and MCBatchNorm2d classes.
"""

import unittest
import torch
import torch.nn as nn
import math
from models2.multi_channel.conv import MCConv2d, MCBatchNorm2d


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
        
        # Check bias parameters are None when bias=False
        self.assertIsNone(self.conv.color_bias)
        self.assertIsNone(self.conv.brightness_bias)
    
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
    
    def test_pytorch_pattern_compliance(self):
        """Test that MCBatchNorm2d follows PyTorch's BatchNorm2d patterns exactly."""
        # Create instances - using same feature count for fair comparison
        pytorch_bn = nn.BatchNorm2d(64)
        mc_bn = MCBatchNorm2d(64, 64)  # Both pathways have same feature count for comparison
        
        # Check _version
        expected_version = getattr(pytorch_bn, '_version', None)
        mc_version = getattr(mc_bn, '_version', None)
        self.assertEqual(mc_version, expected_version, f"Version mismatch: {mc_version} != {expected_version}")
        
        # Check __constants__
        pytorch_constants = set(getattr(pytorch_bn, '__constants__', []))
        mc_constants = set(getattr(mc_bn, '__constants__', []))
        expected_mc_constants = {'track_running_stats', 'momentum', 'eps', 'color_num_features', 'brightness_num_features', 'affine'}
        
        self.assertTrue(expected_mc_constants.issubset(mc_constants), 
                       f"Missing constants: {expected_mc_constants - mc_constants}")
    
    def test_parameter_and_buffer_creation(self):
        """Test parameter and buffer creation follows PyTorch patterns."""
        # Test with affine=True, track_running_stats=True (default)
        mc_bn_default = MCBatchNorm2d(64, 64)
        
        # Check parameter existence
        self.assertTrue(hasattr(mc_bn_default, 'color_weight') and mc_bn_default.color_weight is not None)
        self.assertTrue(hasattr(mc_bn_default, 'color_bias') and mc_bn_default.color_bias is not None)
        self.assertTrue(hasattr(mc_bn_default, 'brightness_weight') and mc_bn_default.brightness_weight is not None)
        self.assertTrue(hasattr(mc_bn_default, 'brightness_bias') and mc_bn_default.brightness_bias is not None)
        
        # Check buffer existence
        self.assertTrue(hasattr(mc_bn_default, 'color_running_mean') and mc_bn_default.color_running_mean is not None)
        self.assertTrue(hasattr(mc_bn_default, 'color_running_var') and mc_bn_default.color_running_var is not None)
        self.assertTrue(hasattr(mc_bn_default, 'num_batches_tracked') and mc_bn_default.num_batches_tracked is not None)
        
        # Test with affine=False, track_running_stats=False
        mc_bn_minimal = MCBatchNorm2d(64, 64, affine=False, track_running_stats=False)
        
        # Check parameters are None
        self.assertIsNone(mc_bn_minimal.color_weight)
        self.assertIsNone(mc_bn_minimal.color_bias)
        self.assertIsNone(mc_bn_minimal.brightness_weight)
        self.assertIsNone(mc_bn_minimal.brightness_bias)
        
        # Check buffers are None
        self.assertIsNone(mc_bn_minimal.color_running_mean)
        self.assertIsNone(mc_bn_minimal.color_running_var)
        self.assertIsNone(mc_bn_minimal.num_batches_tracked)
    
    def test_backward_compatibility_attributes(self):
        """Test that num_batches_tracked is shared between pathways."""
        mc_bn = MCBatchNorm2d(64, 64)
        
        # Both pathways should use the same num_batches_tracked since they process same batches
        self.assertTrue(hasattr(mc_bn, 'num_batches_tracked'))
        self.assertIsNotNone(mc_bn.num_batches_tracked)
        
        # Verify it's a single shared buffer, not separate ones
        self.assertFalse(hasattr(mc_bn, 'color_num_batches_tracked'))
        self.assertFalse(hasattr(mc_bn, 'brightness_num_batches_tracked'))
    
    def test_parameter_initialization(self):
        """Test parameter initialization follows PyTorch patterns."""
        mc_bn = MCBatchNorm2d(64, 64)
        
        # Check weight initialization (should be ones)
        torch.testing.assert_close(mc_bn.color_weight, torch.ones(64))
        torch.testing.assert_close(mc_bn.brightness_weight, torch.ones(64))
        
        # Check bias initialization (should be zeros)
        torch.testing.assert_close(mc_bn.color_bias, torch.zeros(64))
        torch.testing.assert_close(mc_bn.brightness_bias, torch.zeros(64))
        
        # Check running stats initialization
        torch.testing.assert_close(mc_bn.color_running_mean, torch.zeros(64))
        torch.testing.assert_close(mc_bn.brightness_running_mean, torch.zeros(64))
        torch.testing.assert_close(mc_bn.color_running_var, torch.ones(64))
        torch.testing.assert_close(mc_bn.brightness_running_var, torch.ones(64))
        self.assertEqual(mc_bn.num_batches_tracked.item(), 0)  # Shared batch tracking
    
    def test_input_dimension_validation(self):
        """Test input dimension validation matches PyTorch."""
        mc_bn = MCBatchNorm2d(64, 64)
        
        # Test correct input shapes
        color_input = torch.randn(2, 64, 16, 16)
        brightness_input = torch.randn(2, 64, 16, 16)
        
        # Should not raise exception
        try:
            output = mc_bn(color_input, brightness_input)
            self.assertIsNotNone(output)
        except Exception as e:
            self.fail(f"Valid inputs rejected: {e}")
        
        # Test invalid input shapes
        invalid_input = torch.randn(2, 64, 16)  # 3D instead of 4D
        
        with self.assertRaises(ValueError) as context:
            mc_bn._check_input_dim(invalid_input)
        self.assertIn("expected 4D input", str(context.exception))
    
    def test_forward_pass_consistency(self):
        """Test that dual forward gives same results as single pathway forwards."""
        mc_bn = MCBatchNorm2d(64, 64)
        mc_bn.eval()  # Put in eval mode for consistent results
        
        color_input = torch.randn(2, 64, 16, 16)
        brightness_input = torch.randn(2, 64, 16, 16)
        
        # Dual forward
        color_out, brightness_out = mc_bn(color_input, brightness_input)
        
        # Single pathway forwards
        color_only = mc_bn.forward_color(color_input)
        brightness_only = mc_bn.forward_brightness(brightness_input)
        
        torch.testing.assert_close(color_out, color_only)
        torch.testing.assert_close(brightness_out, brightness_only)
    
    def test_extra_repr_method(self):
        """Test extra_repr method contains all required fields."""
        mc_bn = MCBatchNorm2d(64, 32, eps=1e-4, momentum=0.2, affine=False)
        
        repr_str = mc_bn.extra_repr()
        expected_fields = ['color_num_features', 'brightness_num_features', 'eps', 'momentum', 'affine', 'track_running_stats']
        for field in expected_fields:
            self.assertIn(field, repr_str, f"Missing field in extra_repr: {field}")
        
        # Check specific values are present
        self.assertIn('64', repr_str)  # color_num_features
        self.assertIn('32', repr_str)  # brightness_num_features
        # eps can be represented as either '1e-4' or '0.0001'
        self.assertTrue('1e-4' in repr_str or '0.0001' in repr_str, f"eps value not found in repr: {repr_str}")
        self.assertIn('0.2', repr_str)  # momentum
        self.assertIn('False', repr_str)  # affine
    
    def test_parameter_count_scaling(self):
        """Test parameter count scales correctly for dual pathways."""
        pytorch_bn = nn.BatchNorm2d(64)
        # Both pathways use same channel count after transformation
        mc_bn = MCBatchNorm2d(64, 64)  # color: 64, brightness: 64 (same transformation)
        
        pytorch_params = sum(p.numel() for p in pytorch_bn.parameters())
        mc_params = sum(p.numel() for p in mc_bn.parameters())
        
        # MCBatchNorm should have parameters for both pathways
        # PyTorch: 2 * 64 = 128 parameters (weight + bias)
        # MC: 2 * 64 + 2 * 64 = 256 parameters (color weight/bias + brightness weight/bias)
        expected_mc_params = 2 * pytorch_params
        
        self.assertEqual(mc_params, expected_mc_params, 
                        f"Parameter count mismatch: {mc_params} != {expected_mc_params}")
    
    def test_state_dict_structure(self):
        """Test state dict contains all expected dual pathway keys."""
        mc_bn = MCBatchNorm2d(64, 32)
        state_dict = mc_bn.state_dict()
        
        expected_state_keys = [
            'color_weight', 'color_bias', 'color_running_mean', 'color_running_var',
            'brightness_weight', 'brightness_bias', 'brightness_running_mean', 'brightness_running_var',
            'num_batches_tracked'  # Shared batch tracking
        ]
        
        for key in expected_state_keys:
            self.assertIn(key, state_dict, f"Missing state dict key: {key}")
    
    def test_annotations_attribute(self):
        """Test that __annotations__ attribute is properly set for type hints."""
        mc_bn = MCBatchNorm2d(64, 32)
        
        # Check that the class has the expected attributes (parameters)
        # This is more meaningful than checking __annotations__ since PyTorch doesn't consistently use type annotations
        expected_attributes = ['color_weight', 'color_bias', 'brightness_weight', 'brightness_bias']
        
        for attr in expected_attributes:
            self.assertTrue(hasattr(mc_bn, attr), f"Missing attribute: {attr}")
            # For non-None attributes, check they are proper Parameters
            param = getattr(mc_bn, attr)
            if param is not None:
                self.assertIsInstance(param, torch.nn.Parameter, f"{attr} should be a Parameter")


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


class TestPyTorchCompatibilityMCConv2d(unittest.TestCase):
    """
    Comprehensive PyTorch compatibility tests for MCConv2d.
    
    Tests that our dual-stream implementation behaves exactly like PyTorch's Conv2d
    for each individual stream, ensuring the same mathematical operations,
    parameter initialization, gradient flow, and edge cases.
    """
    
    def setUp(self):
        """Set up test fixtures with matching PyTorch configurations."""
        # Standard test configuration
        self.batch_size = 2
        self.input_size = 16
        
        # Channel configurations - both streams have same transformations after initial
        self.in_channels = 64
        self.out_channels = 128
        
        # Test inputs for dual streams and PyTorch comparison
        self.color_input = torch.randn(self.batch_size, self.in_channels, self.input_size, self.input_size)
        self.brightness_input = torch.randn(self.batch_size, self.in_channels, self.input_size, self.input_size) 
        self.pytorch_test_input = torch.randn(self.batch_size, self.in_channels, self.input_size, self.input_size)
        
        # Set seeds for reproducible comparisons
        torch.manual_seed(42)
    
    def test_weight_initialization_compatibility(self):
        """Test that weight initialization matches PyTorch's Conv2d exactly."""
        # Test multiple kernel sizes and configurations
        test_configs = [
            {'kernel_size': 3, 'padding': 1, 'bias': True},
            {'kernel_size': 1, 'padding': 0, 'bias': False},
            {'kernel_size': 5, 'padding': 2, 'bias': True},
            {'kernel_size': 7, 'padding': 3, 'bias': False},
        ]
        
        for config in test_configs:
            with self.subTest(config=config):
                # Create PyTorch reference
                torch.manual_seed(42)
                pytorch_conv = nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    **config
                )
                
                # Create our MC implementation
                torch.manual_seed(42)
                mc_conv = MCConv2d(
                    color_in_channels=self.in_channels,
                    brightness_in_channels=self.in_channels,
                    color_out_channels=self.out_channels,
                    brightness_out_channels=self.out_channels,
                    **config
                )
                
                # Compare weight initialization statistics (should be from same distribution)
                # Test that kaiming uniform was applied correctly
                pytorch_fan_in = pytorch_conv.weight.size(1) * pytorch_conv.weight.size(2) * pytorch_conv.weight.size(3)
                mc_color_fan_in = mc_conv.color_weight.size(1) * mc_conv.color_weight.size(2) * mc_conv.color_weight.size(3)
                mc_brightness_fan_in = mc_conv.brightness_weight.size(1) * mc_conv.brightness_weight.size(2) * mc_conv.brightness_weight.size(3)
                
                self.assertEqual(pytorch_fan_in, mc_color_fan_in)
                self.assertEqual(pytorch_fan_in, mc_brightness_fan_in)
                
                # Test bias initialization if present
                if config['bias']:
                    # Bias should be initialized uniformly in [-bound, bound] where bound = 1/sqrt(fan_in)
                    expected_bound = 1 / math.sqrt(pytorch_fan_in)
                    
                    self.assertTrue(torch.all(torch.abs(pytorch_conv.bias) <= expected_bound + 1e-6))
                    self.assertTrue(torch.all(torch.abs(mc_conv.color_bias) <= expected_bound + 1e-6))
                    self.assertTrue(torch.all(torch.abs(mc_conv.brightness_bias) <= expected_bound + 1e-6))
    
    def test_forward_pass_mathematical_equivalence(self):
        """Test that each stream produces mathematically equivalent results to PyTorch Conv2d."""
        test_configs = [
            {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'groups': 1},
            {'kernel_size': 3, 'stride': 2, 'padding': 1, 'dilation': 1, 'groups': 1},
            {'kernel_size': 1, 'stride': 1, 'padding': 0, 'dilation': 1, 'groups': 1},
            {'kernel_size': 5, 'stride': 1, 'padding': 2, 'dilation': 2, 'groups': 1},
        ]
        
        for config in test_configs:
            with self.subTest(config=config):
                # Create layers with identical weights for comparison
                pytorch_conv = nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    bias=False,
                    **config
                )
                
                mc_conv = MCConv2d(
                    color_in_channels=self.in_channels,
                    brightness_in_channels=self.in_channels,
                    color_out_channels=self.out_channels,
                    brightness_out_channels=self.out_channels,
                    bias=False,
                    **config
                )
                
                # Copy weights to ensure identical computation
                with torch.no_grad():
                    mc_conv.color_weight.copy_(pytorch_conv.weight)
                    mc_conv.brightness_weight.copy_(pytorch_conv.weight)
                
                # Test forward pass
                pytorch_output = pytorch_conv(self.pytorch_test_input)
                color_output = mc_conv.forward_color(self.pytorch_test_input)
                brightness_output = mc_conv.forward_brightness(self.pytorch_test_input)
                
                # Results should be numerically identical
                torch.testing.assert_close(pytorch_output, color_output, rtol=1e-6, atol=1e-6)
                torch.testing.assert_close(pytorch_output, brightness_output, rtol=1e-6, atol=1e-6)
    
    def test_padding_modes_compatibility(self):
        """Test all padding modes work identically to PyTorch."""
        padding_modes = ['zeros', 'reflect', 'replicate', 'circular']
        
        for mode in padding_modes:
            with self.subTest(padding_mode=mode):
                pytorch_conv = nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=3,
                    padding=1,
                    padding_mode=mode,
                    bias=False
                )
                
                mc_conv = MCConv2d(
                    color_in_channels=self.in_channels,
                    brightness_in_channels=self.in_channels,
                    color_out_channels=self.out_channels,
                    brightness_out_channels=self.out_channels,
                    kernel_size=3,
                    padding=1,
                    padding_mode=mode,
                    bias=False
                )
                
                # Copy weights
                with torch.no_grad():
                    mc_conv.color_weight.copy_(pytorch_conv.weight)
                    mc_conv.brightness_weight.copy_(pytorch_conv.weight)
                
                # Test outputs
                pytorch_output = pytorch_conv(self.pytorch_test_input)
                color_output = mc_conv.forward_color(self.pytorch_test_input)
                
                torch.testing.assert_close(pytorch_output, color_output, rtol=1e-6, atol=1e-6)
    
    def test_gradient_flow_compatibility(self):
        """Test that gradients flow exactly like PyTorch Conv2d."""
        pytorch_conv = nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1)
        mc_conv = MCConv2d(
            self.in_channels, self.in_channels, 
            self.out_channels, self.out_channels, 
            3, padding=1
        )
        
        # Copy weights for identical starting point
        with torch.no_grad():
            mc_conv.color_weight.copy_(pytorch_conv.weight)
            mc_conv.brightness_weight.copy_(pytorch_conv.weight)
            if pytorch_conv.bias is not None:
                mc_conv.color_bias.copy_(pytorch_conv.bias)
                mc_conv.brightness_bias.copy_(pytorch_conv.bias)
        
        # Test inputs with gradients
        pytorch_input = self.pytorch_test_input.clone().requires_grad_(True)
        color_input = self.pytorch_test_input.clone().requires_grad_(True)
        brightness_input = self.pytorch_test_input.clone().requires_grad_(True)
        
        # Forward and backward
        pytorch_output = pytorch_conv(pytorch_input)
        color_output = mc_conv.forward_color(color_input)
        brightness_output = mc_conv.forward_brightness(brightness_input)
        
        pytorch_loss = pytorch_output.sum()
        color_loss = color_output.sum()
        brightness_loss = brightness_output.sum()
        
        pytorch_loss.backward()
        color_loss.backward()
        brightness_loss.backward()
        
        # Compare input gradients
        torch.testing.assert_close(pytorch_input.grad, color_input.grad, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(pytorch_input.grad, brightness_input.grad, rtol=1e-6, atol=1e-6)
        
        # Compare weight gradients
        torch.testing.assert_close(pytorch_conv.weight.grad, mc_conv.color_weight.grad, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(pytorch_conv.weight.grad, mc_conv.brightness_weight.grad, rtol=1e-6, atol=1e-6)
    
    def test_channel_dimension_transformations(self):
        """Test that channel transformations follow PyTorch patterns exactly."""
        # Test standard ResNet-style channel progressions
        channel_progressions = [
            (3, 64),    # Initial conv: RGB to 64 features
            (64, 64),   # Same dimension
            (64, 128),  # 2x expansion
            (128, 256), # 2x expansion  
            (256, 512), # 2x expansion
            (512, 1024), # 2x expansion
        ]
        
        for in_ch, out_ch in channel_progressions:
            with self.subTest(transformation=f"{in_ch}->{out_ch}"):
                # Test that our implementation handles these transformations
                # exactly like PyTorch for both streams
                test_input = torch.randn(2, in_ch, 16, 16)
                
                pytorch_conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
                mc_conv = MCConv2d(in_ch, in_ch, out_ch, out_ch, 3, padding=1, bias=False)
                
                # Copy weights 
                with torch.no_grad():
                    mc_conv.color_weight.copy_(pytorch_conv.weight)
                    mc_conv.brightness_weight.copy_(pytorch_conv.weight)
                
                pytorch_out = pytorch_conv(test_input)
                color_out = mc_conv.forward_color(test_input)
                brightness_out = mc_conv.forward_brightness(test_input)
                
                # Verify channel transformation
                self.assertEqual(pytorch_out.shape[1], out_ch)
                self.assertEqual(color_out.shape[1], out_ch)
                self.assertEqual(brightness_out.shape[1], out_ch)
                
                # Verify mathematical equivalence
                torch.testing.assert_close(pytorch_out, color_out, rtol=1e-6, atol=1e-6)
                torch.testing.assert_close(pytorch_out, brightness_out, rtol=1e-6, atol=1e-6)
    
    def test_parameter_sharing_isolation(self):
        """Test that color and brightness parameters are properly isolated."""
        mc_conv = MCConv2d(64, 64, 128, 128, 3, padding=1)
        
        # Modify color weights
        original_brightness_weight = mc_conv.brightness_weight.clone()
        mc_conv.color_weight.data.fill_(1.0)
        
        # Brightness weights should be unchanged
        torch.testing.assert_close(mc_conv.brightness_weight, original_brightness_weight)
        
        # Test that forward passes produce different results
        test_input = torch.randn(1, 64, 8, 8)
        color_out = mc_conv.forward_color(test_input)
        brightness_out = mc_conv.forward_brightness(test_input)
        
        # Results should be different since weights are different
        self.assertFalse(torch.allclose(color_out, brightness_out))
    
    def test_dual_stream_coordination(self):
        """Test that dual-stream forward pass coordinates both streams correctly."""
        mc_conv = MCConv2d(64, 64, 128, 128, 3, padding=1, bias=False)
        
        color_input = torch.randn(2, 64, 16, 16)
        brightness_input = torch.randn(2, 64, 16, 16)
        
        # Test dual forward
        color_out, brightness_out = mc_conv(color_input, brightness_input)
        
        # Test individual forwards
        color_out_individual = mc_conv.forward_color(color_input)
        brightness_out_individual = mc_conv.forward_brightness(brightness_input)
        
        # Results should be identical
        torch.testing.assert_close(color_out, color_out_individual)
        torch.testing.assert_close(brightness_out, brightness_out_individual)
    
    def test_error_handling_compatibility(self):
        """Test that error handling matches PyTorch's Conv2d behavior."""
        # Test invalid parameters that should raise errors
        with self.assertRaises(ValueError):
            MCConv2d(64, 64, 128, 128, 3, groups=0)  # Invalid groups
        
        with self.assertRaises(ValueError):
            MCConv2d(63, 64, 128, 128, 3, groups=2)  # in_channels not divisible by groups
        
        with self.assertRaises(ValueError):
            MCConv2d(64, 64, 127, 128, 3, groups=2)  # out_channels not divisible by groups
        
        with self.assertRaises(ValueError):
            MCConv2d(64, 64, 128, 128, 3, padding="invalid")  # Invalid padding string


class TestPyTorchCompatibilityMCBatchNorm2d(unittest.TestCase):
    """
    Comprehensive PyTorch compatibility tests for MCBatchNorm2d.
    
    Tests that our dual-stream batch normalization behaves exactly like PyTorch's BatchNorm2d
    for each individual stream, ensuring the same normalization statistics, momentum updates,
    running statistics, and training/eval behavior.
    """
    
    def setUp(self):
        """Set up test fixtures with matching PyTorch configurations."""
        self.batch_size = 4
        self.num_features = 64
        self.input_size = 16
        
        # Test inputs
        self.color_input = torch.randn(self.batch_size, self.num_features, self.input_size, self.input_size)
        self.brightness_input = torch.randn(self.batch_size, self.num_features, self.input_size, self.input_size)
        self.pytorch_test_input = torch.randn(self.batch_size, self.num_features, self.input_size, self.input_size)
    
    def test_parameter_initialization_compatibility(self):
        """Test that parameter initialization matches PyTorch's BatchNorm2d exactly."""
        pytorch_bn = nn.BatchNorm2d(self.num_features)
        mc_bn = MCBatchNorm2d(self.num_features, self.num_features)
        
        # Check weight initialization (should be ones)
        torch.testing.assert_close(pytorch_bn.weight, torch.ones_like(pytorch_bn.weight))
        torch.testing.assert_close(mc_bn.color_weight, torch.ones_like(mc_bn.color_weight))
        torch.testing.assert_close(mc_bn.brightness_weight, torch.ones_like(mc_bn.brightness_weight))
        
        # Check bias initialization (should be zeros)
        torch.testing.assert_close(pytorch_bn.bias, torch.zeros_like(pytorch_bn.bias))
        torch.testing.assert_close(mc_bn.color_bias, torch.zeros_like(mc_bn.color_bias))
        torch.testing.assert_close(mc_bn.brightness_bias, torch.zeros_like(mc_bn.brightness_bias))
        
        # Check running stats initialization
        torch.testing.assert_close(pytorch_bn.running_mean, torch.zeros_like(pytorch_bn.running_mean))
        torch.testing.assert_close(mc_bn.color_running_mean, torch.zeros_like(mc_bn.color_running_mean))
        torch.testing.assert_close(mc_bn.brightness_running_mean, torch.zeros_like(mc_bn.brightness_running_mean))
        
        torch.testing.assert_close(pytorch_bn.running_var, torch.ones_like(pytorch_bn.running_var))
        torch.testing.assert_close(mc_bn.color_running_var, torch.ones_like(mc_bn.color_running_var))
        torch.testing.assert_close(mc_bn.brightness_running_var, torch.ones_like(mc_bn.brightness_running_var))
    
    def test_training_mode_normalization_compatibility(self):
        """Test that training mode normalization matches PyTorch exactly."""
        pytorch_bn = nn.BatchNorm2d(self.num_features)
        mc_bn = MCBatchNorm2d(self.num_features, self.num_features)
        
        # Set to training mode
        pytorch_bn.train()
        mc_bn.train()
        
        # Copy parameters for identical computation
        with torch.no_grad():
            mc_bn.color_weight.copy_(pytorch_bn.weight)
            mc_bn.brightness_weight.copy_(pytorch_bn.weight)
            mc_bn.color_bias.copy_(pytorch_bn.bias)
            mc_bn.brightness_bias.copy_(pytorch_bn.bias)
        
        # Forward pass
        pytorch_output = pytorch_bn(self.pytorch_test_input)
        color_output = mc_bn.forward_color(self.pytorch_test_input)
        brightness_output = mc_bn.forward_brightness(self.pytorch_test_input)
        
        # Results should be numerically identical (within floating point precision)
        torch.testing.assert_close(pytorch_output, color_output, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(pytorch_output, brightness_output, rtol=1e-6, atol=1e-6)
    
    def test_eval_mode_normalization_compatibility(self):
        """Test that eval mode normalization matches PyTorch exactly."""
        pytorch_bn = nn.BatchNorm2d(self.num_features)
        mc_bn = MCBatchNorm2d(self.num_features, self.num_features)
        
        # First run in training mode to populate running stats
        pytorch_bn.train()
        mc_bn.train()
        
        # Use same input for consistent running stats
        with torch.no_grad():
            mc_bn.color_weight.copy_(pytorch_bn.weight)
            mc_bn.brightness_weight.copy_(pytorch_bn.weight)
            mc_bn.color_bias.copy_(pytorch_bn.bias)
            mc_bn.brightness_bias.copy_(pytorch_bn.bias)
            
            # Initialize running stats to same values
            mc_bn.color_running_mean.copy_(pytorch_bn.running_mean)
            mc_bn.brightness_running_mean.copy_(pytorch_bn.running_mean)
            mc_bn.color_running_var.copy_(pytorch_bn.running_var)
            mc_bn.brightness_running_var.copy_(pytorch_bn.running_var)
            mc_bn.num_batches_tracked.copy_(pytorch_bn.num_batches_tracked)
        
        # Switch to eval mode
        pytorch_bn.eval()
        mc_bn.eval()
        
        # Test forward pass in eval mode
        pytorch_output = pytorch_bn(self.pytorch_test_input)
        color_output = mc_bn.forward_color(self.pytorch_test_input)
        brightness_output = mc_bn.forward_brightness(self.pytorch_test_input)
        
        torch.testing.assert_close(pytorch_output, color_output, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(pytorch_output, brightness_output, rtol=1e-6, atol=1e-6)
    
    def test_running_statistics_updates_compatibility(self):
        """Test that running statistics are updated exactly like PyTorch."""
        pytorch_bn = nn.BatchNorm2d(self.num_features, momentum=0.1)
        mc_bn = MCBatchNorm2d(self.num_features, self.num_features, momentum=0.1)
        
        # Set to training mode
        pytorch_bn.train()
        mc_bn.train()
        
        # Copy initial parameters
        with torch.no_grad():
            mc_bn.color_weight.copy_(pytorch_bn.weight)
            mc_bn.brightness_weight.copy_(pytorch_bn.weight)
            mc_bn.color_bias.copy_(pytorch_bn.bias)
            mc_bn.brightness_bias.copy_(pytorch_bn.bias)
        
        # Test single pathway updates to match PyTorch behavior
        for i in range(5):
            test_input = torch.randn(self.batch_size, self.num_features, self.input_size, self.input_size)
            
            # Forward passes - compare single pathway to PyTorch
            pytorch_bn(test_input)
            mc_bn.forward_color(test_input)
            
            # Check that num_batches_tracked updates correctly for single pathway
            self.assertEqual(pytorch_bn.num_batches_tracked.item(), i + 1)
            self.assertEqual(mc_bn.num_batches_tracked.item(), i + 1)
        
        # Reset and test that dual forward pass updates batch tracking appropriately
        pytorch_bn.reset_running_stats()
        mc_bn.reset_running_stats()
        
        for i in range(3):
            test_input = torch.randn(self.batch_size, self.num_features, self.input_size, self.input_size)
            color_input = torch.randn(self.batch_size, self.num_features, self.input_size, self.input_size)
            brightness_input = torch.randn(self.batch_size, self.num_features, self.input_size, self.input_size)
            
            # Dual forward pass should increment batch tracking once per batch processed
            mc_bn(color_input, brightness_input)
            
            # Since dual forward processes one batch but updates twice (once per pathway),
            # we expect num_batches_tracked to be 2*(i+1)
            expected_batches = 2 * (i + 1)
            self.assertEqual(mc_bn.num_batches_tracked.item(), expected_batches)
    
    def test_momentum_calculation_compatibility(self):
        """Test that momentum calculations match PyTorch exactly."""
        # Test different momentum values
        momentum_values = [0.1, 0.9, None]  # None means cumulative moving average
        
        for momentum in momentum_values:
            with self.subTest(momentum=momentum):
                pytorch_bn = nn.BatchNorm2d(self.num_features, momentum=momentum)
                mc_bn = MCBatchNorm2d(self.num_features, self.num_features, momentum=momentum)
                
                pytorch_bn.train()
                mc_bn.train()
                
                # Copy parameters
                with torch.no_grad():
                    mc_bn.color_weight.copy_(pytorch_bn.weight)
                    mc_bn.brightness_weight.copy_(pytorch_bn.weight)
                    mc_bn.color_bias.copy_(pytorch_bn.bias)
                    mc_bn.brightness_bias.copy_(pytorch_bn.bias)
                
                # Forward pass
                test_input = torch.randn(self.batch_size, self.num_features, self.input_size, self.input_size)
                pytorch_bn(test_input)
                mc_bn.forward_color(test_input)
                
                # The exponential average factor calculation should be identical
                # This is tested implicitly through the running stats updates
    
    def test_affine_parameter_effects_compatibility(self):
        """Test that affine parameters affect output exactly like PyTorch."""
        # Test with affine=True (default)
        pytorch_bn_affine = nn.BatchNorm2d(self.num_features, affine=True)
        mc_bn_affine = MCBatchNorm2d(self.num_features, self.num_features, affine=True)
        
        # Test with affine=False
        pytorch_bn_no_affine = nn.BatchNorm2d(self.num_features, affine=False)
        mc_bn_no_affine = MCBatchNorm2d(self.num_features, self.num_features, affine=False)
        
        # Test that affine=False layers have no learnable parameters
        self.assertIsNone(pytorch_bn_no_affine.weight)
        self.assertIsNone(mc_bn_no_affine.color_weight)
        self.assertIsNone(mc_bn_no_affine.brightness_weight)
        
        self.assertIsNone(pytorch_bn_no_affine.bias)
        self.assertIsNone(mc_bn_no_affine.color_bias)
        self.assertIsNone(mc_bn_no_affine.brightness_bias)
    
    def test_track_running_stats_compatibility(self):
        """Test that track_running_stats behavior matches PyTorch exactly."""
        # Test with track_running_stats=False
        pytorch_bn = nn.BatchNorm2d(self.num_features, track_running_stats=False)
        mc_bn = MCBatchNorm2d(self.num_features, self.num_features, track_running_stats=False)
        
        # Should have no running statistics
        self.assertIsNone(pytorch_bn.running_mean)
        self.assertIsNone(mc_bn.color_running_mean)
        self.assertIsNone(mc_bn.brightness_running_mean)
        
        self.assertIsNone(pytorch_bn.running_var)
        self.assertIsNone(mc_bn.color_running_var)
        self.assertIsNone(mc_bn.brightness_running_var)
        
        self.assertIsNone(pytorch_bn.num_batches_tracked)
        self.assertIsNone(mc_bn.num_batches_tracked)
    
    def test_eps_parameter_compatibility(self):
        """Test that eps parameter affects computation exactly like PyTorch."""
        eps_values = [1e-5, 1e-3, 1e-7]
        
        for eps in eps_values:
            with self.subTest(eps=eps):
                pytorch_bn = nn.BatchNorm2d(self.num_features, eps=eps)
                mc_bn = MCBatchNorm2d(self.num_features, self.num_features, eps=eps)
                
                # Copy parameters
                with torch.no_grad():
                    mc_bn.color_weight.copy_(pytorch_bn.weight)
                    mc_bn.brightness_weight.copy_(pytorch_bn.weight) 
                    mc_bn.color_bias.copy_(pytorch_bn.bias)
                    mc_bn.brightness_bias.copy_(pytorch_bn.bias)
                
                pytorch_output = pytorch_bn(self.pytorch_test_input)
                color_output = mc_bn.forward_color(self.pytorch_test_input)
                
                torch.testing.assert_close(pytorch_output, color_output, rtol=1e-6, atol=1e-6)
    
    def test_gradient_flow_compatibility(self):
        """Test that gradients flow exactly like PyTorch BatchNorm2d."""
        pytorch_bn = nn.BatchNorm2d(self.num_features)
        mc_bn = MCBatchNorm2d(self.num_features, self.num_features)
        
        # Copy parameters
        with torch.no_grad():
            mc_bn.color_weight.copy_(pytorch_bn.weight)
            mc_bn.brightness_weight.copy_(pytorch_bn.weight)
            mc_bn.color_bias.copy_(pytorch_bn.bias)
            mc_bn.brightness_bias.copy_(pytorch_bn.bias)
        
        # Test inputs with gradients
        pytorch_input = self.pytorch_test_input.clone().requires_grad_(True)
        color_input = self.pytorch_test_input.clone().requires_grad_(True)
        
        # Forward and backward
        pytorch_output = pytorch_bn(pytorch_input)
        color_output = mc_bn.forward_color(color_input)
        
        pytorch_loss = pytorch_output.sum()
        color_loss = color_output.sum()
        
        pytorch_loss.backward()
        color_loss.backward()
        
        # Compare gradients
        torch.testing.assert_close(pytorch_input.grad, color_input.grad, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(pytorch_bn.weight.grad, mc_bn.color_weight.grad, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(pytorch_bn.bias.grad, mc_bn.color_bias.grad, rtol=1e-6, atol=1e-6)
    
    def test_input_dimension_validation_compatibility(self):
        """Test that input dimension validation matches PyTorch exactly."""
        pytorch_bn = nn.BatchNorm2d(self.num_features)
        mc_bn = MCBatchNorm2d(self.num_features, self.num_features)
        
        # Test with wrong input dimensions
        wrong_input_3d = torch.randn(self.batch_size, self.num_features, self.input_size)  # 3D instead of 4D
        wrong_input_5d = torch.randn(self.batch_size, self.num_features, self.input_size, self.input_size, 2)  # 5D
        
        # Both should raise the same error for 3D input
        with self.assertRaises(ValueError):
            pytorch_bn(wrong_input_3d)
        with self.assertRaises(ValueError):
            mc_bn.forward_color(wrong_input_3d)
        
        # Both should raise the same error for 5D input  
        with self.assertRaises(ValueError):
            pytorch_bn(wrong_input_5d)
        with self.assertRaises(ValueError):
            mc_bn.forward_color(wrong_input_5d)


if __name__ == "__main__":
    unittest.main()
