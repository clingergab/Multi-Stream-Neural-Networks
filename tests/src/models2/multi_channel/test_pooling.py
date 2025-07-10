"""
Unit tests for Multi-Channel pooling modules.
"""

import unittest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent / "src"))

from models2.multi_channel.pooling import MCMaxPool2d, MCAdaptiveAvgPool2d


class TestMCMaxPool2d(unittest.TestCase):
    """Test cases for MCMaxPool2d."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample inputs
        self.color_input = torch.randn(2, 3, 8, 8)
        self.brightness_input = torch.randn(2, 1, 8, 8)
    
    def test_init_basic(self):
        """Test basic MCMaxPool2d initialization."""
        pool = MCMaxPool2d(kernel_size=2)
        
        self.assertEqual(pool.kernel_size, (2, 2))
        self.assertEqual(pool.stride, (2, 2))
        self.assertEqual(pool.padding, (0, 0))
        self.assertEqual(pool.dilation, (1, 1))
        self.assertFalse(pool.return_indices)
        self.assertFalse(pool.ceil_mode)
    
    def test_init_with_all_params(self):
        """Test MCMaxPool2d initialization with all parameters."""
        pool = MCMaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=2,
            return_indices=True,
            ceil_mode=True
        )
        
        self.assertEqual(pool.kernel_size, (3, 3))
        self.assertEqual(pool.stride, (2, 2))
        self.assertEqual(pool.padding, (1, 1))
        self.assertEqual(pool.dilation, (2, 2))
        self.assertTrue(pool.return_indices)
        self.assertTrue(pool.ceil_mode)
    
    def test_init_tuple_params(self):
        """Test MCMaxPool2d initialization with tuple parameters."""
        pool = MCMaxPool2d(
            kernel_size=(3, 2),
            stride=(2, 1),
            padding=(1, 0),
            dilation=(2, 1)
        )
        
        self.assertEqual(pool.kernel_size, (3, 2))
        self.assertEqual(pool.stride, (2, 1))
        self.assertEqual(pool.padding, (1, 0))
        self.assertEqual(pool.dilation, (2, 1))
    
    def test_forward_basic(self):
        """Test basic forward pass."""
        pool = MCMaxPool2d(kernel_size=2, stride=2)
        
        color_out, brightness_out = pool(self.color_input, self.brightness_input)
        
        # Check output shapes - should be halved due to stride=2
        expected_shape_color = (2, 3, 4, 4)
        expected_shape_brightness = (2, 1, 4, 4)
        
        self.assertEqual(color_out.shape, expected_shape_color)
        self.assertEqual(brightness_out.shape, expected_shape_brightness)
    
    def test_forward_with_padding(self):
        """Test forward pass with padding."""
        pool = MCMaxPool2d(kernel_size=3, stride=2, padding=1)
        
        color_out, brightness_out = pool(self.color_input, self.brightness_input)
        
        # With padding=1 and stride=2, output size should be ceil((8+2*1-3)/2 + 1) = 4
        expected_shape_color = (2, 3, 4, 4)
        expected_shape_brightness = (2, 1, 4, 4)
        
        self.assertEqual(color_out.shape, expected_shape_color)
        self.assertEqual(brightness_out.shape, expected_shape_brightness)
    
    def test_forward_color_pathway(self):
        """Test forward pass through color pathway only."""
        pool = MCMaxPool2d(kernel_size=2, stride=2)
        
        color_out = pool.forward_color(self.color_input)
        expected_shape = (2, 3, 4, 4)
        
        self.assertEqual(color_out.shape, expected_shape)
    
    def test_forward_brightness_pathway(self):
        """Test forward pass through brightness pathway only."""
        pool = MCMaxPool2d(kernel_size=2, stride=2)
        
        brightness_out = pool.forward_brightness(self.brightness_input)
        expected_shape = (2, 1, 4, 4)
        
        self.assertEqual(brightness_out.shape, expected_shape)
    
    def test_forward_return_indices(self):
        """Test forward pass with return_indices=True."""
        pool = MCMaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        # When return_indices=True, the functional returns (output, indices) tuple
        # Our implementation returns ((color_out, color_indices), (brightness_out, brightness_indices))
        color_result, brightness_result = pool(self.color_input, self.brightness_input)
        
        # Each result should be a tuple of (output, indices)
        self.assertIsInstance(color_result, tuple)
        self.assertIsInstance(brightness_result, tuple)
        self.assertEqual(len(color_result), 2)
        self.assertEqual(len(brightness_result), 2)
        
        # Extract outputs and indices
        color_out, color_indices = color_result
        brightness_out, brightness_indices = brightness_result
        
        # Check output shapes
        self.assertEqual(color_out.shape, (2, 3, 4, 4))
        self.assertEqual(brightness_out.shape, (2, 1, 4, 4))
        
        # Check indices shapes (should be same as outputs)
        self.assertEqual(color_indices.shape, (2, 3, 4, 4))
        self.assertEqual(brightness_indices.shape, (2, 1, 4, 4))
    
    def test_forward_ceil_mode(self):
        """Test forward pass with ceil_mode=True."""
        pool = MCMaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        
        color_out, brightness_out = pool(self.color_input, self.brightness_input)
        
        # With ceil_mode=True, output size should be ceil((8-3)/2 + 1) = 4
        expected_shape_color = (2, 3, 4, 4)
        expected_shape_brightness = (2, 1, 4, 4)
        
        self.assertEqual(color_out.shape, expected_shape_color)
        self.assertEqual(brightness_out.shape, expected_shape_brightness)
    
    def test_gradient_flow(self):
        """Test that gradients flow through MCMaxPool2d."""
        pool = MCMaxPool2d(kernel_size=2, stride=2)
        
        color_input = torch.randn(1, 3, 4, 4, requires_grad=True)
        brightness_input = torch.randn(1, 1, 4, 4, requires_grad=True)
        
        color_out, brightness_out = pool(color_input, brightness_input)
        loss = color_out.sum() + brightness_out.sum()
        loss.backward()
        
        # Check that gradients exist
        self.assertIsNotNone(color_input.grad)
        self.assertIsNotNone(brightness_input.grad)
    
    def test_extra_repr(self):
        """Test string representation."""
        pool = MCMaxPool2d(kernel_size=3, stride=2, padding=1, dilation=2, ceil_mode=True)
        
        repr_str = pool.extra_repr()
        self.assertIn("kernel_size=(3, 3)", repr_str)
        self.assertIn("stride=(2, 2)", repr_str)
        self.assertIn("padding=(1, 1)", repr_str)
        self.assertIn("dilation=(2, 2)", repr_str)
        self.assertIn("ceil_mode=True", repr_str)


class TestMCAdaptiveAvgPool2d(unittest.TestCase):
    """Test cases for MCAdaptiveAvgPool2d."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample inputs
        self.color_input = torch.randn(2, 3, 8, 8)
        self.brightness_input = torch.randn(2, 1, 8, 8)
    
    def test_init_single_size(self):
        """Test MCAdaptiveAvgPool2d initialization with single size."""
        pool = MCAdaptiveAvgPool2d(7)
        
        self.assertEqual(pool.output_size, 7)
    
    def test_init_tuple_size(self):
        """Test MCAdaptiveAvgPool2d initialization with tuple size."""
        pool = MCAdaptiveAvgPool2d((5, 7))
        
        self.assertEqual(pool.output_size, (5, 7))
    
    def test_init_none_size(self):
        """Test MCAdaptiveAvgPool2d initialization with None in tuple."""
        pool = MCAdaptiveAvgPool2d((None, 7))
        
        self.assertEqual(pool.output_size, (None, 7))
    
    def test_forward_single_size(self):
        """Test forward pass with single output size."""
        pool = MCAdaptiveAvgPool2d(4)
        
        color_out, brightness_out = pool(self.color_input, self.brightness_input)
        
        # Output should be 4x4 regardless of input size
        expected_shape_color = (2, 3, 4, 4)
        expected_shape_brightness = (2, 1, 4, 4)
        
        self.assertEqual(color_out.shape, expected_shape_color)
        self.assertEqual(brightness_out.shape, expected_shape_brightness)
    
    def test_forward_tuple_size(self):
        """Test forward pass with tuple output size."""
        pool = MCAdaptiveAvgPool2d((3, 5))
        
        color_out, brightness_out = pool(self.color_input, self.brightness_input)
        
        # Output should be 3x5
        expected_shape_color = (2, 3, 3, 5)
        expected_shape_brightness = (2, 1, 3, 5)
        
        self.assertEqual(color_out.shape, expected_shape_color)
        self.assertEqual(brightness_out.shape, expected_shape_brightness)
    
    def test_forward_none_dimension(self):
        """Test forward pass with None in one dimension."""
        pool = MCAdaptiveAvgPool2d((None, 4))
        
        color_out, brightness_out = pool(self.color_input, self.brightness_input)
        
        # Height should remain 8, width should be 4
        expected_shape_color = (2, 3, 8, 4)
        expected_shape_brightness = (2, 1, 8, 4)
        
        self.assertEqual(color_out.shape, expected_shape_color)
        self.assertEqual(brightness_out.shape, expected_shape_brightness)
    
    def test_forward_color_pathway(self):
        """Test forward pass through color pathway only."""
        pool = MCAdaptiveAvgPool2d(4)
        
        color_out = pool.forward_color(self.color_input)
        expected_shape = (2, 3, 4, 4)
        
        self.assertEqual(color_out.shape, expected_shape)
    
    def test_forward_brightness_pathway(self):
        """Test forward pass through brightness pathway only."""
        pool = MCAdaptiveAvgPool2d(4)
        
        brightness_out = pool.forward_brightness(self.brightness_input)
        expected_shape = (2, 1, 4, 4)
        
        self.assertEqual(brightness_out.shape, expected_shape)
    
    def test_forward_larger_output(self):
        """Test forward pass with larger output than input."""
        pool = MCAdaptiveAvgPool2d(12)  # Larger than input size of 8
        
        color_out, brightness_out = pool(self.color_input, self.brightness_input)
        
        # Should work and upsample to 12x12
        expected_shape_color = (2, 3, 12, 12)
        expected_shape_brightness = (2, 1, 12, 12)
        
        self.assertEqual(color_out.shape, expected_shape_color)
        self.assertEqual(brightness_out.shape, expected_shape_brightness)
    
    def test_forward_different_input_sizes(self):
        """Test forward pass with different input sizes."""
        pool = MCAdaptiveAvgPool2d(4)
        
        # Test with different input sizes
        color_input1 = torch.randn(1, 3, 16, 16)
        brightness_input1 = torch.randn(1, 1, 16, 16)
        
        color_input2 = torch.randn(1, 3, 6, 10)
        brightness_input2 = torch.randn(1, 1, 6, 10)
        
        color_out1, brightness_out1 = pool(color_input1, brightness_input1)
        color_out2, brightness_out2 = pool(color_input2, brightness_input2)
        
        # Both should output 4x4 regardless of input size
        self.assertEqual(color_out1.shape, (1, 3, 4, 4))
        self.assertEqual(brightness_out1.shape, (1, 1, 4, 4))
        self.assertEqual(color_out2.shape, (1, 3, 4, 4))
        self.assertEqual(brightness_out2.shape, (1, 1, 4, 4))
    
    def test_gradient_flow(self):
        """Test that gradients flow through MCAdaptiveAvgPool2d."""
        pool = MCAdaptiveAvgPool2d(4)
        
        color_input = torch.randn(1, 3, 8, 8, requires_grad=True)
        brightness_input = torch.randn(1, 1, 8, 8, requires_grad=True)
        
        color_out, brightness_out = pool(color_input, brightness_input)
        loss = color_out.sum() + brightness_out.sum()
        loss.backward()
        
        # Check that gradients exist
        self.assertIsNotNone(color_input.grad)
        self.assertIsNotNone(brightness_input.grad)
    
    def test_extra_repr(self):
        """Test string representation."""
        pool = MCAdaptiveAvgPool2d((5, 7))
        
        repr_str = pool.extra_repr()
        self.assertIn("output_size=(5, 7)", repr_str)
    
    def test_consistency_with_pytorch(self):
        """Test that our implementation gives same results as PyTorch's."""
        import torch.nn.functional as F
        
        pool = MCAdaptiveAvgPool2d(4)
        
        # Apply our multi-channel version
        color_out, brightness_out = pool(self.color_input, self.brightness_input)
        
        # Apply PyTorch's version separately
        expected_color = F.adaptive_avg_pool2d(self.color_input, 4)
        expected_brightness = F.adaptive_avg_pool2d(self.brightness_input, 4)
        
        # Should be identical
        torch.testing.assert_close(color_out, expected_color)
        torch.testing.assert_close(brightness_out, expected_brightness)


class TestPoolingLayersIntegration(unittest.TestCase):
    """Integration tests for pooling layers."""
    
    def test_maxpool_with_conv(self):
        """Test MCMaxPool2d integration with conv layers."""
        from models2.multi_channel.conv import MCConv2d
        
        conv = MCConv2d(3, 1, 16, 16, kernel_size=3, padding=1)
        pool = MCMaxPool2d(kernel_size=2, stride=2)
        
        color_input = torch.randn(1, 3, 8, 8)
        brightness_input = torch.randn(1, 1, 8, 8)
        
        # Apply conv then pool
        color_conv, brightness_conv = conv(color_input, brightness_input)
        color_out, brightness_out = pool(color_conv, brightness_conv)
        
        # Should work without errors
        self.assertEqual(color_out.shape, (1, 16, 4, 4))
        self.assertEqual(brightness_out.shape, (1, 16, 4, 4))
    
    def test_adaptive_pool_with_conv(self):
        """Test MCAdaptiveAvgPool2d integration with conv layers."""
        from models2.multi_channel.conv import MCConv2d
        
        conv = MCConv2d(3, 1, 16, 16, kernel_size=3, padding=1)
        pool = MCAdaptiveAvgPool2d(1)  # Global average pooling
        
        color_input = torch.randn(1, 3, 8, 8)
        brightness_input = torch.randn(1, 1, 8, 8)
        
        # Apply conv then pool
        color_conv, brightness_conv = conv(color_input, brightness_input)
        color_out, brightness_out = pool(color_conv, brightness_conv)
        
        # Should produce 1x1 outputs (global pooling)
        self.assertEqual(color_out.shape, (1, 16, 1, 1))
        self.assertEqual(brightness_out.shape, (1, 16, 1, 1))
    
    def test_sequential_pooling(self):
        """Test sequential application of pooling layers."""
        pool1 = MCMaxPool2d(kernel_size=2, stride=2)  # 8x8 -> 4x4
        pool2 = MCAdaptiveAvgPool2d(2)  # 4x4 -> 2x2
        
        color_input = torch.randn(1, 3, 8, 8)
        brightness_input = torch.randn(1, 1, 8, 8)
        
        # Apply pools sequentially
        color_out1, brightness_out1 = pool1(color_input, brightness_input)
        color_out2, brightness_out2 = pool2(color_out1, brightness_out1)
        
        # Final output should be 2x2
        self.assertEqual(color_out2.shape, (1, 3, 2, 2))
        self.assertEqual(brightness_out2.shape, (1, 1, 2, 2))


if __name__ == "__main__":
    unittest.main()
