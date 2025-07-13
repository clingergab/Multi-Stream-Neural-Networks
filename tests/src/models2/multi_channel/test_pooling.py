"""
Unit tests for Multi-Channel pooling modules.
"""

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import gc
import io
import pickle
from pathlib import Path

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
        
        # PyTorch stores raw values, not converted tuples
        self.assertEqual(pool.kernel_size, 2)
        self.assertEqual(pool.stride, 2)
        self.assertEqual(pool.padding, 0)
        self.assertEqual(pool.dilation, 1)
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
        
        # PyTorch stores raw values, not converted tuples
        self.assertEqual(pool.kernel_size, 3)
        self.assertEqual(pool.stride, 2)
        self.assertEqual(pool.padding, 1)
        self.assertEqual(pool.dilation, 2)
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
        self.assertIn("kernel_size=3", repr_str)
        self.assertIn("stride=2", repr_str)
        self.assertIn("padding=1", repr_str)
        self.assertIn("dilation=2", repr_str)
        self.assertIn("ceil_mode=True", repr_str)
    
    def test_consistency_with_pytorch(self):
        """Test that our implementation gives same results as PyTorch's."""
        pool = MCMaxPool2d(kernel_size=2, stride=2)
        pytorch_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Create one PyTorch version of what we're implementing
        # Use the same input for both MC pathways and PyTorch layer
        test_input = torch.randn(2, 3, 8, 8)
        
        # Apply PyTorch's version
        pytorch_out = pytorch_pool(test_input)
        
        # Apply our multi-channel version with same input for both pathways
        mc_color_out, mc_brightness_out = pool(test_input, test_input)
        
        # Compare each MC pathway output to the PyTorch output
        torch.testing.assert_close(mc_color_out, pytorch_out)
        torch.testing.assert_close(mc_brightness_out, pytorch_out)


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
        
        # Create one PyTorch version of what we're implementing
        # Use the same input for both MC pathways and PyTorch layer
        test_input = torch.randn(2, 3, 8, 8)
        
        # Apply PyTorch's version
        pytorch_out = F.adaptive_avg_pool2d(test_input, 4)
        
        # Apply our multi-channel version with same input for both pathways
        mc_color_out, mc_brightness_out = pool(test_input, test_input)
        
        # Compare each MC pathway output to the PyTorch output
        torch.testing.assert_close(mc_color_out, pytorch_out)
        torch.testing.assert_close(mc_brightness_out, pytorch_out)


class TestChannelTransformationPatterns(unittest.TestCase):
    """Test pooling layers handle channel transformation patterns correctly."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.input_size = 32
        
        # Test different stages of channel transformation
        # Stage 1: Different initial channels
        self.color_input_stage1 = torch.randn(self.batch_size, 3, self.input_size, self.input_size)
        self.brightness_input_stage1 = torch.randn(self.batch_size, 1, self.input_size, self.input_size)
        
        # Stage 2: After initial conv - both have 64 channels  
        self.color_input_stage2 = torch.randn(self.batch_size, 64, self.input_size, self.input_size)
        self.brightness_input_stage2 = torch.randn(self.batch_size, 64, self.input_size, self.input_size)
        
        # Stage 3: Mid-network - both have 256 channels
        self.color_input_stage3 = torch.randn(self.batch_size, 256, self.input_size, self.input_size)
        self.brightness_input_stage3 = torch.randn(self.batch_size, 256, self.input_size, self.input_size)
    
    def test_pooling_preserves_channel_counts(self):
        """Test that pooling preserves channel counts at each stage."""
        pool = MCMaxPool2d(kernel_size=2, stride=2)
        
        # Stage 1: Different channels should be preserved
        color_out1, brightness_out1 = pool(self.color_input_stage1, self.brightness_input_stage1)
        self.assertEqual(color_out1.shape[1], 3)  # Color channels preserved
        self.assertEqual(brightness_out1.shape[1], 1)  # Brightness channels preserved
        
        # Stage 2: Same channels should be preserved
        color_out2, brightness_out2 = pool(self.color_input_stage2, self.brightness_input_stage2)
        self.assertEqual(color_out2.shape[1], 64)  # Both preserved
        self.assertEqual(brightness_out2.shape[1], 64)
        
        # Stage 3: Larger channel counts should be preserved
        color_out3, brightness_out3 = pool(self.color_input_stage3, self.brightness_input_stage3)
        self.assertEqual(color_out3.shape[1], 256)
        self.assertEqual(brightness_out3.shape[1], 256)
    
    def test_adaptive_pooling_channel_transformations(self):
        """Test adaptive pooling with various channel transformation stages."""
        pool = MCAdaptiveAvgPool2d(1)  # Global average pooling
        
        # Test all stages produce correct output shapes
        color_out1, brightness_out1 = pool(self.color_input_stage1, self.brightness_input_stage1)
        self.assertEqual(color_out1.shape, (self.batch_size, 3, 1, 1))
        self.assertEqual(brightness_out1.shape, (self.batch_size, 1, 1, 1))
        
        color_out2, brightness_out2 = pool(self.color_input_stage2, self.brightness_input_stage2)
        self.assertEqual(color_out2.shape, (self.batch_size, 64, 1, 1))
        self.assertEqual(brightness_out2.shape, (self.batch_size, 64, 1, 1))
        
        color_out3, brightness_out3 = pool(self.color_input_stage3, self.brightness_input_stage3)
        self.assertEqual(color_out3.shape, (self.batch_size, 256, 1, 1))
        self.assertEqual(brightness_out3.shape, (self.batch_size, 256, 1, 1))
    
    def test_progressive_pooling_channel_transformation(self):
        """Test progressive pooling through typical ResNet-style channel progression."""
        # Simulate typical ResNet progression with pooling
        # 3,1 -> 64,64 -> pool -> 128,128 -> pool -> 256,256 -> global pool
        
        from models2.multi_channel.conv import MCConv2d, MCBatchNorm2d
        
        # Stage 1: Initial conv (3,1 -> 64,64)
        initial_conv = MCConv2d(3, 1, 64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        pool1 = MCMaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Stage 2: Mid conv (64,64 -> 128,128)  
        mid_conv = MCConv2d(64, 64, 128, 128, kernel_size=3, stride=2, padding=1, bias=False)
        pool2 = MCMaxPool2d(kernel_size=2, stride=2)
        
        # Stage 3: Final conv (128,128 -> 256,256)
        final_conv = MCConv2d(128, 128, 256, 256, kernel_size=3, stride=2, padding=1, bias=False)
        global_pool = MCAdaptiveAvgPool2d(1)
        
        # Start with different initial channels
        color_input = torch.randn(2, 3, 224, 224)
        brightness_input = torch.randn(2, 1, 224, 224)
        
        # Apply progressive transformation
        color_out, brightness_out = initial_conv(color_input, brightness_input)
        color_out, brightness_out = pool1(color_out, brightness_out)
        # Both should now have 64 channels, spatial size ~56x56
        
        color_out, brightness_out = mid_conv(color_out, brightness_out)
        color_out, brightness_out = pool2(color_out, brightness_out)
        # Both should now have 128 channels, smaller spatial size
        
        color_out, brightness_out = final_conv(color_out, brightness_out)
        color_out, brightness_out = global_pool(color_out, brightness_out)
        
        # Final output: both pathways should have same channels and 1x1 spatial
        expected_shape = (2, 256, 1, 1)
        self.assertEqual(color_out.shape, expected_shape)
        self.assertEqual(brightness_out.shape, expected_shape)


class TestAdvancedPyTorchCompatibility(unittest.TestCase):
    """Advanced PyTorch compatibility tests for edge cases and complex scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.color_input = torch.randn(2, 64, 16, 16)
        self.brightness_input = torch.randn(2, 64, 16, 16)
        self.pytorch_test_input = torch.randn(2, 64, 16, 16)
    
    def test_mcmaxpool2d_edge_cases_vs_pytorch(self):
        """Test MCMaxPool2d edge cases match PyTorch behavior."""
        edge_configs = [
            {"kernel_size": 1},  # Identity pooling
            {"kernel_size": 16, "stride": 1, "padding": 0},  # Kernel size = input size
            {"kernel_size": 2, "stride": 1, "padding": 1},  # Padding = kernel_size/2
            {"kernel_size": 3, "dilation": 2, "padding": 1},  # Large dilation with valid padding
            {"kernel_size": (1, 3), "stride": (1, 2)},  # Asymmetric kernel
        ]
        
        for config in edge_configs:
            with self.subTest(config=config):
                mc_pool = MCMaxPool2d(**config)
                pytorch_pool = nn.MaxPool2d(**config)
                
                # Use same input for both MC pathways and PyTorch
                test_input = self.pytorch_test_input
                
                # Test color pathway against PyTorch
                pytorch_out = pytorch_pool(test_input)
                mc_color_out = mc_pool.forward_color(test_input)
                
                torch.testing.assert_close(mc_color_out, pytorch_out,
                                         msg=f"Color pathway edge case config {config} failed")
                
                # Test brightness pathway against same PyTorch output  
                mc_brightness_out = mc_pool.forward_brightness(test_input)
                
                torch.testing.assert_close(mc_brightness_out, pytorch_out,
                                         msg=f"Brightness pathway edge case config {config} failed")
                
                # Test dual pathway - both outputs should match PyTorch
                mc_color_dual, mc_brightness_dual = mc_pool(test_input, test_input)
                
                torch.testing.assert_close(mc_color_dual, pytorch_out,
                                         msg=f"Dual-stream color output differs from PyTorch for config {config}")
                torch.testing.assert_close(mc_brightness_dual, pytorch_out,
                                         msg=f"Dual-stream brightness output differs from PyTorch for config {config}")
    
    def test_mcadaptive_avgpool2d_edge_cases_vs_pytorch(self):
        """Test MCAdaptiveAvgPool2d edge cases match PyTorch behavior."""
        edge_cases = [
            (1, 1),  # 1x1 output
            (16, 16),  # Same as input size
            (32, 32),  # Larger than input
            (None, 8),  # One dimension preserved
            (8, None),  # Other dimension preserved
        ]
        
        for output_size in edge_cases:
            with self.subTest(output_size=output_size):
                mc_pool = MCAdaptiveAvgPool2d(output_size)
                pytorch_pool = nn.AdaptiveAvgPool2d(output_size)
                
                # Use same input for both MC pathways and PyTorch
                test_input = self.pytorch_test_input
                
                # Test color pathway against PyTorch
                pytorch_out = pytorch_pool(test_input)
                mc_color_out = mc_pool.forward_color(test_input)
                
                torch.testing.assert_close(mc_color_out, pytorch_out,
                                         msg=f"Color pathway edge case output_size {output_size} failed")
                
                # Test brightness pathway against same PyTorch output
                mc_brightness_out = mc_pool.forward_brightness(test_input)
                
                torch.testing.assert_close(mc_brightness_out, pytorch_out,
                                         msg=f"Brightness pathway edge case output_size {output_size} failed")
                
                # Test dual pathway - both outputs should match PyTorch
                mc_color_dual, mc_brightness_dual = mc_pool(test_input, test_input)
                
                torch.testing.assert_close(mc_color_dual, pytorch_out,
                                         msg=f"Dual-stream color output differs from PyTorch for output_size {output_size}")
                torch.testing.assert_close(mc_brightness_dual, pytorch_out,
                                         msg=f"Dual-stream brightness output differs from PyTorch for output_size {output_size}")
    
    def test_mixed_precision_compatibility(self):
        """Test pooling layers work with mixed precision (half/float)."""
        mc_maxpool = MCMaxPool2d(kernel_size=2)
        mc_adaptive = MCAdaptiveAvgPool2d(4)
        
        # Test with half precision
        color_half = self.color_input.half()
        brightness_half = self.brightness_input.half()
        
        color_out_half, brightness_out_half = mc_maxpool(color_half, brightness_half)
        self.assertEqual(color_out_half.dtype, torch.float16)
        self.assertEqual(brightness_out_half.dtype, torch.float16)
        
        color_adaptive_half, brightness_adaptive_half = mc_adaptive(color_half, brightness_half)
        self.assertEqual(color_adaptive_half.dtype, torch.float16)
        self.assertEqual(brightness_adaptive_half.dtype, torch.float16)
    
    def test_device_compatibility(self):
        """Test pooling layers maintain device placement."""
        mc_maxpool = MCMaxPool2d(kernel_size=2)
        mc_adaptive = MCAdaptiveAvgPool2d(4)
        
        # Test CPU inputs stay on CPU
        color_out_cpu, brightness_out_cpu = mc_maxpool(self.color_input, self.brightness_input)
        self.assertEqual(color_out_cpu.device.type, 'cpu')
        self.assertEqual(brightness_out_cpu.device.type, 'cpu')
        
        # Test with CUDA if available
        if torch.cuda.is_available():
            color_cuda = self.color_input.cuda()
            brightness_cuda = self.brightness_input.cuda()
            
            color_out_cuda, brightness_out_cuda = mc_maxpool(color_cuda, brightness_cuda)
            self.assertEqual(color_out_cuda.device.type, 'cuda')
            self.assertEqual(brightness_out_cuda.device.type, 'cuda')
    
    def test_memory_efficiency_vs_pytorch(self):
        """Test memory usage is reasonable compared to PyTorch."""
        import gc
        
        # Large inputs to make memory differences visible
        large_input = torch.randn(4, 64, 128, 128)
        
        # Test PyTorch memory usage
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
        pytorch_pool = nn.MaxPool2d(kernel_size=2)
        pytorch_out = pytorch_pool(large_input)
        pytorch_memory = pytorch_out.numel() * pytorch_out.element_size()
        
        # Test MC memory usage
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
        mc_pool = MCMaxPool2d(kernel_size=2)
        mc_color_out, mc_brightness_out = mc_pool(large_input, large_input)
        mc_memory = (mc_color_out.numel() + mc_brightness_out.numel()) * mc_color_out.element_size()
        
        # MC should use roughly 2x memory (two outputs vs one)
        memory_ratio = mc_memory / pytorch_memory
        self.assertGreater(memory_ratio, 1.8)  # Close to 2x
        self.assertLess(memory_ratio, 2.2)  # But not much more
    
    def test_serialization_compatibility(self):
        """Test that pooling layers can be serialized/deserialized like PyTorch layers."""
        import pickle
        import io
        
        # Test MCMaxPool2d serialization
        mc_maxpool = MCMaxPool2d(kernel_size=3, stride=2, padding=1, dilation=2, ceil_mode=True)
        
        # Serialize
        buffer = io.BytesIO()
        pickle.dump(mc_maxpool, buffer)
        
        # Deserialize
        buffer.seek(0)
        mc_maxpool_loaded = pickle.load(buffer)
        
        # Test that deserialized version works identically
        color_out1, brightness_out1 = mc_maxpool(self.color_input, self.brightness_input)
        color_out2, brightness_out2 = mc_maxpool_loaded(self.color_input, self.brightness_input)
        
        torch.testing.assert_close(color_out1, color_out2)
        torch.testing.assert_close(brightness_out1, brightness_out2)
        
        # Test MCAdaptiveAvgPool2d serialization
        mc_adaptive = MCAdaptiveAvgPool2d((5, 7))
        
        buffer = io.BytesIO()
        pickle.dump(mc_adaptive, buffer)
        buffer.seek(0)
        mc_adaptive_loaded = pickle.load(buffer)
        
        color_out1, brightness_out1 = mc_adaptive(self.color_input, self.brightness_input)
        color_out2, brightness_out2 = mc_adaptive_loaded(self.color_input, self.brightness_input)
        
        torch.testing.assert_close(color_out1, color_out2)
        torch.testing.assert_close(brightness_out1, brightness_out2)


class TestPyTorchCompatibilityMCMaxPool2d(unittest.TestCase):
    """
    Comprehensive PyTorch compatibility tests for MCMaxPool2d.
    
    Tests that our dual-stream implementation behaves exactly like PyTorch's MaxPool2d
    for each individual stream, ensuring the same mathematical operations,
    parameter handling, output shapes, and edge cases.
    """
    
    def setUp(self):
        """Set up test fixtures with matching PyTorch configurations."""
        # Standard test configuration
        self.batch_size = 2
        self.input_size = 16
        
        # Channel configurations - both streams have same transformations after initial
        self.in_channels = 64
        
        # Test inputs for dual streams and PyTorch comparison
        self.color_input = torch.randn(self.batch_size, self.in_channels, self.input_size, self.input_size)
        self.brightness_input = torch.randn(self.batch_size, self.in_channels, self.input_size, self.input_size) 
        self.pytorch_test_input = torch.randn(self.batch_size, self.in_channels, self.input_size, self.input_size)
        
        # Set seeds for reproducible comparisons
        torch.manual_seed(42)
    
    def test_parameter_initialization_compatibility(self):
        """Test that parameter initialization matches PyTorch's MaxPool2d exactly."""
        # Test multiple configurations
        test_configs = [
            {'kernel_size': 2},
            {'kernel_size': 3, 'stride': 2, 'padding': 1},
            {'kernel_size': (3, 2), 'stride': (2, 1), 'padding': (1, 0)},
            {'kernel_size': 5, 'stride': 3, 'padding': 2, 'dilation': 2},
            {'kernel_size': 2, 'ceil_mode': True},
            {'kernel_size': 3, 'return_indices': True},
        ]
        
        for config in test_configs:
            with self.subTest(config=config):
                # Create PyTorch reference
                pytorch_pool = nn.MaxPool2d(**config)
                
                # Create our MC implementation
                mc_pool = MCMaxPool2d(**config)
                
                # Compare all attributes
                self.assertEqual(pytorch_pool.kernel_size, mc_pool.kernel_size)
                self.assertEqual(pytorch_pool.stride, mc_pool.stride) 
                self.assertEqual(pytorch_pool.padding, mc_pool.padding)
                self.assertEqual(pytorch_pool.dilation, mc_pool.dilation)
                self.assertEqual(pytorch_pool.return_indices, mc_pool.return_indices)
                self.assertEqual(pytorch_pool.ceil_mode, mc_pool.ceil_mode)
    
    def test_output_shape_compatibility(self):
        """Test that output shapes match PyTorch exactly."""
        test_configs = [
            {'kernel_size': 2, 'stride': 2},  # Standard downsampling
            {'kernel_size': 3, 'stride': 1, 'padding': 1},  # Same size output
            {'kernel_size': 3, 'stride': 2, 'padding': 1},  # Half size output
            {'kernel_size': 5, 'stride': 1, 'padding': 2},  # Same size with larger kernel
            {'kernel_size': 2, 'stride': 1},  # Overlapping windows
            {'kernel_size': 3, 'stride': 2, 'padding': 1, 'dilation': 2},  # With dilation
        ]
        
        for config in test_configs:
            with self.subTest(config=config):
                pytorch_pool = nn.MaxPool2d(**config)
                mc_pool = MCMaxPool2d(**config)
                
                # Test output shapes
                pytorch_output = pytorch_pool(self.pytorch_test_input)
                color_output = mc_pool.forward_color(self.pytorch_test_input)
                brightness_output = mc_pool.forward_brightness(self.pytorch_test_input)
                
                # Shapes should be identical
                self.assertEqual(pytorch_output.shape, color_output.shape)
                self.assertEqual(pytorch_output.shape, brightness_output.shape)
    
    def test_mathematical_equivalence(self):
        """Test that pooling operations are mathematically equivalent to PyTorch."""
        test_configs = [
            {'kernel_size': 2, 'stride': 2},
            {'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'kernel_size': 3, 'stride': 2, 'padding': 1, 'ceil_mode': False},
            {'kernel_size': 3, 'stride': 2, 'padding': 1, 'ceil_mode': True},
        ]
        
        for config in test_configs:
            with self.subTest(config=config):
                pytorch_pool = nn.MaxPool2d(**config)
                mc_pool = MCMaxPool2d(**config)
                
                # Use identical input
                test_input = torch.randn(2, 32, 8, 8)
                
                pytorch_output = pytorch_pool(test_input)
                color_output = mc_pool.forward_color(test_input)
                brightness_output = mc_pool.forward_brightness(test_input)
                
                # Results should be numerically identical
                torch.testing.assert_close(pytorch_output, color_output, rtol=1e-6, atol=1e-6)
                torch.testing.assert_close(pytorch_output, brightness_output, rtol=1e-6, atol=1e-6)
    
    def test_return_indices_compatibility(self):
        """Test that return_indices behavior matches PyTorch exactly."""
        # Test with return_indices=True
        pytorch_pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        mc_pool = MCMaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        test_input = torch.randn(1, 16, 8, 8)
        
        pytorch_output, pytorch_indices = pytorch_pool(test_input)
        color_output, color_indices = mc_pool.forward_color(test_input)
        brightness_output, brightness_indices = mc_pool.forward_brightness(test_input)
        
        # Results should be identical
        torch.testing.assert_close(pytorch_output, color_output)
        torch.testing.assert_close(pytorch_indices, color_indices)
        torch.testing.assert_close(pytorch_output, brightness_output)
        torch.testing.assert_close(pytorch_indices, brightness_indices)
    
    def test_ceil_mode_compatibility(self):
        """Test that ceil_mode behavior matches PyTorch exactly."""
        # Test with different input sizes to see ceil_mode effect
        test_inputs = [
            torch.randn(1, 8, 7, 7),   # Odd size
            torch.randn(1, 8, 9, 9),   # Another odd size
            torch.randn(1, 8, 6, 8),   # Different H and W
        ]
        
        for ceil_mode in [False, True]:
            with self.subTest(ceil_mode=ceil_mode):
                pytorch_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=ceil_mode)
                mc_pool = MCMaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=ceil_mode)
                
                for test_input in test_inputs:
                    pytorch_output = pytorch_pool(test_input)
                    color_output = mc_pool.forward_color(test_input)
                    
                    # Shape and values should match
                    self.assertEqual(pytorch_output.shape, color_output.shape)
                    torch.testing.assert_close(pytorch_output, color_output)
    
    def test_dilation_compatibility(self):
        """Test that dilation behavior matches PyTorch exactly."""
        test_configs = [
            {'kernel_size': 3, 'dilation': 1},
            {'kernel_size': 3, 'dilation': 2},
            {'kernel_size': 3, 'dilation': 3},
            {'kernel_size': (3, 2), 'dilation': (2, 1)},
        ]
        
        for config in test_configs:
            with self.subTest(config=config):
                pytorch_pool = nn.MaxPool2d(**config)
                mc_pool = MCMaxPool2d(**config)
                
                test_input = torch.randn(1, 16, 12, 12)
                
                pytorch_output = pytorch_pool(test_input)
                color_output = mc_pool.forward_color(test_input)
                
                # Results should be identical
                torch.testing.assert_close(pytorch_output, color_output)
    
    def test_gradient_flow_compatibility(self):
        """Test that gradients flow exactly like PyTorch MaxPool2d."""
        pytorch_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        mc_pool = MCMaxPool2d(kernel_size=2, stride=2)
        
        # Test inputs with gradients
        pytorch_input = self.pytorch_test_input.clone().requires_grad_(True)
        color_input = self.pytorch_test_input.clone().requires_grad_(True)
        brightness_input = self.pytorch_test_input.clone().requires_grad_(True)
        
        # Forward and backward
        pytorch_output = pytorch_pool(pytorch_input)
        color_output = mc_pool.forward_color(color_input)
        brightness_output = mc_pool.forward_brightness(brightness_input)
        
        pytorch_loss = pytorch_output.sum()
        color_loss = color_output.sum()
        brightness_loss = brightness_output.sum()
        
        pytorch_loss.backward()
        color_loss.backward()
        brightness_loss.backward()
        
        # Compare input gradients
        torch.testing.assert_close(pytorch_input.grad, color_input.grad, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(pytorch_input.grad, brightness_input.grad, rtol=1e-6, atol=1e-6)
    
    def test_dual_stream_coordination(self):
        """Test that dual-stream forward pass coordinates both streams correctly."""
        mc_pool = MCMaxPool2d(kernel_size=2, stride=2)
        
        color_input = torch.randn(2, 32, 8, 8)
        brightness_input = torch.randn(2, 32, 8, 8)
        
        # Test dual forward
        color_out, brightness_out = mc_pool(color_input, brightness_input)
        
        # Test individual forwards
        color_out_individual = mc_pool.forward_color(color_input)
        brightness_out_individual = mc_pool.forward_brightness(brightness_input)
        
        # Results should be identical
        torch.testing.assert_close(color_out, color_out_individual)
        torch.testing.assert_close(brightness_out, brightness_out_individual)
    
    def test_parameter_sharing_isolation(self):
        """Test that both streams use same pooling parameters but process independently."""
        mc_pool = MCMaxPool2d(kernel_size=2, stride=2)
        
        # Different inputs should produce different outputs
        color_input = torch.randn(1, 16, 8, 8)
        brightness_input = torch.randn(1, 16, 8, 8)
        
        color_out = mc_pool.forward_color(color_input)
        brightness_out = mc_pool.forward_brightness(brightness_input)
        
        # Since inputs are different, outputs should be different
        self.assertFalse(torch.allclose(color_out, brightness_out))
        
        # But same input should produce same output for both streams
        same_input = torch.randn(1, 16, 8, 8)
        color_out_same = mc_pool.forward_color(same_input)
        brightness_out_same = mc_pool.forward_brightness(same_input)
        
        torch.testing.assert_close(color_out_same, brightness_out_same)
    
    def test_constants_attribute_compatibility(self):
        """Test that __constants__ attribute matches PyTorch pattern."""
        pytorch_pool = nn.MaxPool2d(kernel_size=2)
        mc_pool = MCMaxPool2d(kernel_size=2)
        
        # Check that our constants include the same fields as PyTorch
        pytorch_constants = set(pytorch_pool.__constants__)
        mc_constants = set(mc_pool.__constants__)
        
        # Our implementation should have at least the same constants as PyTorch
        self.assertTrue(pytorch_constants.issubset(mc_constants), 
                       f"Missing constants: {pytorch_constants - mc_constants}")
    
    def test_extra_repr_compatibility(self):
        """Test that extra_repr method output is consistent with PyTorch pattern."""
        test_configs = [
            {'kernel_size': 2},
            {'kernel_size': 3, 'stride': 2, 'padding': 1},
            {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 2, 'ceil_mode': True},
        ]
        
        for config in test_configs:
            with self.subTest(config=config):
                pytorch_pool = nn.MaxPool2d(**config)
                mc_pool = MCMaxPool2d(**config)
                
                pytorch_repr = pytorch_pool.extra_repr()
                mc_repr = mc_pool.extra_repr()
                
                # Both should contain the same parameter information
                # Check that key parameters are present in both
                for param in ['kernel_size', 'stride', 'padding']:
                    self.assertIn(param, pytorch_repr)
                    self.assertIn(param, mc_repr)


class TestPyTorchCompatibilityMCAdaptiveAvgPool2d(unittest.TestCase):
    """
    Comprehensive PyTorch compatibility tests for MCAdaptiveAvgPool2d.
    
    Tests that our dual-stream implementation behaves exactly like PyTorch's AdaptiveAvgPool2d
    for each individual stream, ensuring the same mathematical operations,
    parameter handling, output shapes, and edge cases.
    """
    
    def setUp(self):
        """Set up test fixtures with matching PyTorch configurations."""
        # Standard test configuration
        self.batch_size = 2
        self.input_size = 16
        
        # Channel configurations - both streams have same transformations after initial
        self.in_channels = 64
        
        # Test inputs for dual streams and PyTorch comparison
        self.color_input = torch.randn(self.batch_size, self.in_channels, self.input_size, self.input_size)
        self.brightness_input = torch.randn(self.batch_size, self.in_channels, self.input_size, self.input_size) 
        self.pytorch_test_input = torch.randn(self.batch_size, self.in_channels, self.input_size, self.input_size)
        
        # Set seeds for reproducible comparisons
        torch.manual_seed(42)
    
    def test_parameter_initialization_compatibility(self):
        """Test that parameter initialization matches PyTorch's AdaptiveAvgPool2d exactly."""
        # Test multiple output size configurations
        test_configs = [
            {'output_size': 1},                    # Global average pooling
            {'output_size': 7},                    # Single size
            {'output_size': (5, 7)},              # Tuple size
            {'output_size': (None, 7)},           # None for one dimension
            {'output_size': (5, None)},           # None for other dimension
            {'output_size': (None, None)},        # Both None (identity)
        ]
        
        for config in test_configs:
            with self.subTest(config=config):
                # Create PyTorch reference
                pytorch_pool = nn.AdaptiveAvgPool2d(**config)
                
                # Create our MC implementation
                mc_pool = MCAdaptiveAvgPool2d(**config)
                
                # Compare output_size attribute
                self.assertEqual(pytorch_pool.output_size, mc_pool.output_size)
    
    def test_output_shape_compatibility(self):
        """Test that output shapes match PyTorch exactly."""
        test_configs = [
            {'output_size': 1, 'expected_shape': (2, 64, 1, 1)},
            {'output_size': 7, 'expected_shape': (2, 64, 7, 7)},
            {'output_size': (5, 7), 'expected_shape': (2, 64, 5, 7)},
            {'output_size': (None, 7), 'expected_shape': (2, 64, 16, 7)},  # Height preserved
            {'output_size': (5, None), 'expected_shape': (2, 64, 5, 16)},  # Width preserved
        ]
        
        for config in test_configs:
            with self.subTest(config=config):
                expected_shape = config.pop('expected_shape')
                
                pytorch_pool = nn.AdaptiveAvgPool2d(**config)
                mc_pool = MCAdaptiveAvgPool2d(**config)
                
                # Test output shapes
                pytorch_output = pytorch_pool(self.pytorch_test_input)
                color_output = mc_pool.forward_color(self.pytorch_test_input)
                brightness_output = mc_pool.forward_brightness(self.pytorch_test_input)
                
                # Shapes should match expected and be identical across implementations
                self.assertEqual(pytorch_output.shape, expected_shape)
                self.assertEqual(color_output.shape, expected_shape)
                self.assertEqual(brightness_output.shape, expected_shape)
    
    def test_mathematical_equivalence(self):
        """Test that pooling operations are mathematically equivalent to PyTorch."""
        test_configs = [
            {'output_size': 1},      # Global pooling - deterministic result
            {'output_size': 4},      # Downsampling
            {'output_size': 8},      # Half size
            {'output_size': (4, 8)}, # Asymmetric
        ]
        
        for config in test_configs:
            with self.subTest(config=config):
                pytorch_pool = nn.AdaptiveAvgPool2d(**config)
                mc_pool = MCAdaptiveAvgPool2d(**config)
                
                # Use identical input
                test_input = torch.randn(2, 32, 16, 16)
                
                pytorch_output = pytorch_pool(test_input)
                color_output = mc_pool.forward_color(test_input)
                brightness_output = mc_pool.forward_brightness(test_input)
                
                # Results should be numerically identical
                torch.testing.assert_close(pytorch_output, color_output, rtol=1e-6, atol=1e-6)
                torch.testing.assert_close(pytorch_output, brightness_output, rtol=1e-6, atol=1e-6)
    
    def test_global_average_pooling_compatibility(self):
        """Test that global average pooling (output_size=1) works exactly like PyTorch."""
        pytorch_pool = nn.AdaptiveAvgPool2d(1)
        mc_pool = MCAdaptiveAvgPool2d(1)
        
        # Test with different input sizes
        test_inputs = [
            torch.randn(1, 16, 7, 7),
            torch.randn(1, 32, 14, 14),
            torch.randn(2, 64, 28, 28),
            torch.randn(1, 8, 1, 1),  # Already 1x1
        ]
        
        for test_input in test_inputs:
            with self.subTest(input_shape=test_input.shape):
                pytorch_output = pytorch_pool(test_input)
                color_output = mc_pool.forward_color(test_input)
                
                # Should be global average - same values
                torch.testing.assert_close(pytorch_output, color_output, rtol=1e-6, atol=1e-6)
                
                # Output should be 1x1
                self.assertEqual(pytorch_output.shape[-2:], (1, 1))
                self.assertEqual(color_output.shape[-2:], (1, 1))
    
    def test_upsampling_compatibility(self):
        """Test that upsampling (output > input) works exactly like PyTorch."""
        # Test upsampling case
        pytorch_pool = nn.AdaptiveAvgPool2d(32)  # Larger than input (16)
        mc_pool = MCAdaptiveAvgPool2d(32)
        
        test_input = torch.randn(1, 16, 8, 8)
        
        pytorch_output = pytorch_pool(test_input)
        color_output = mc_pool.forward_color(test_input)
        
        # Results should be identical
        torch.testing.assert_close(pytorch_output, color_output, rtol=1e-6, atol=1e-6)
        
        # Output should be 32x32
        self.assertEqual(pytorch_output.shape[-2:], (32, 32))
        self.assertEqual(color_output.shape[-2:], (32, 32))
    
    def test_none_dimension_compatibility(self):
        """Test that None dimensions work exactly like PyTorch."""
        test_configs = [
            {'output_size': (None, 4), 'input_shape': (1, 8, 10, 12)},  # Preserve height
            {'output_size': (6, None), 'input_shape': (1, 8, 10, 12)},  # Preserve width
            {'output_size': (None, None), 'input_shape': (1, 8, 10, 12)},  # Identity
        ]
        
        for config in test_configs:
            with self.subTest(config=config):
                input_shape = config.pop('input_shape')
                test_input = torch.randn(*input_shape)
                
                pytorch_pool = nn.AdaptiveAvgPool2d(**config)
                mc_pool = MCAdaptiveAvgPool2d(**config)
                
                pytorch_output = pytorch_pool(test_input)
                color_output = mc_pool.forward_color(test_input)
                
                # Results should be identical
                torch.testing.assert_close(pytorch_output, color_output, rtol=1e-6, atol=1e-6)
                
                # Check expected shape based on None handling
                if config['output_size'] == (None, None):
                    # Should be identity
                    self.assertEqual(pytorch_output.shape, test_input.shape)
    
    def test_gradient_flow_compatibility(self):
        """Test that gradients flow exactly like PyTorch AdaptiveAvgPool2d."""
        pytorch_pool = nn.AdaptiveAvgPool2d(4)
        mc_pool = MCAdaptiveAvgPool2d(4)
        
        # Test inputs with gradients
        pytorch_input = self.pytorch_test_input.clone().requires_grad_(True)
        color_input = self.pytorch_test_input.clone().requires_grad_(True)
        brightness_input = self.pytorch_test_input.clone().requires_grad_(True)
        
        # Forward and backward
        pytorch_output = pytorch_pool(pytorch_input)
        color_output = mc_pool.forward_color(color_input)
        brightness_output = mc_pool.forward_brightness(brightness_input)
        
        pytorch_loss = pytorch_output.sum()
        color_loss = color_output.sum()
        brightness_loss = brightness_output.sum()
        
        pytorch_loss.backward()
        color_loss.backward()
        brightness_loss.backward()
        
        # Compare input gradients
        torch.testing.assert_close(pytorch_input.grad, color_input.grad, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(pytorch_input.grad, brightness_input.grad, rtol=1e-6, atol=1e-6)
    
    def test_dual_stream_coordination(self):
        """Test that dual-stream forward pass coordinates both streams correctly."""
        mc_pool = MCAdaptiveAvgPool2d(8)
        
        color_input = torch.randn(2, 32, 16, 16)
        brightness_input = torch.randn(2, 32, 16, 16)
        
        # Test dual forward
        color_out, brightness_out = mc_pool(color_input, brightness_input)
        
        # Test individual forwards
        color_out_individual = mc_pool.forward_color(color_input)
        brightness_out_individual = mc_pool.forward_brightness(brightness_input)
        
        # Results should be identical
        torch.testing.assert_close(color_out, color_out_individual)
        torch.testing.assert_close(brightness_out, brightness_out_individual)
    
    def test_parameter_sharing_isolation(self):
        """Test that both streams use same pooling parameters but process independently."""
        mc_pool = MCAdaptiveAvgPool2d(4)
        
        # Different inputs should produce different outputs
        color_input = torch.randn(1, 16, 8, 8)
        brightness_input = torch.randn(1, 16, 8, 8)
        
        color_out = mc_pool.forward_color(color_input)
        brightness_out = mc_pool.forward_brightness(brightness_input)
        
        # Since inputs are different, outputs should be different
        self.assertFalse(torch.allclose(color_out, brightness_out))
        
        # But same input should produce same output for both streams
        same_input = torch.randn(1, 16, 8, 8)
        color_out_same = mc_pool.forward_color(same_input)
        brightness_out_same = mc_pool.forward_brightness(same_input)
        
        torch.testing.assert_close(color_out_same, brightness_out_same)
    
    def test_constants_attribute_compatibility(self):
        """Test that __constants__ attribute matches PyTorch pattern."""
        pytorch_pool = nn.AdaptiveAvgPool2d(4)
        mc_pool = MCAdaptiveAvgPool2d(4)
        
        # Check that our constants include the same fields as PyTorch
        pytorch_constants = set(pytorch_pool.__constants__)
        mc_constants = set(mc_pool.__constants__)
        
        # Our implementation should have at least the same constants as PyTorch
        self.assertTrue(pytorch_constants.issubset(mc_constants), 
                       f"Missing constants: {pytorch_constants - mc_constants}")
    
    def test_extra_repr_compatibility(self):
        """Test that extra_repr method output is consistent with PyTorch pattern."""
        test_configs = [
            {'output_size': 1},
            {'output_size': 7},
            {'output_size': (5, 7)},
            {'output_size': (None, 7)},
        ]
        
        for config in test_configs:
            with self.subTest(config=config):
                pytorch_pool = nn.AdaptiveAvgPool2d(**config)
                mc_pool = MCAdaptiveAvgPool2d(**config)
                
                pytorch_repr = pytorch_pool.extra_repr()
                mc_repr = mc_pool.extra_repr()
                
                # Both should contain the output_size information
                self.assertIn('output_size', pytorch_repr)
                self.assertIn('output_size', mc_repr)
                
                # Should contain the same output_size value
                if config['output_size'] is not None:
                    self.assertIn(str(config['output_size']), mc_repr)


class TestPyTorchCompatibilityBaseClasses(unittest.TestCase):
    """
    Test PyTorch compatibility for base classes _MCMaxPoolNd and _MCAdaptiveAvgPoolNd.
    
    These tests ensure our base classes follow PyTorch's base class patterns exactly,
    including attribute handling, constants, and method signatures.
    """
    
    def test_mc_maxpool_base_class_compatibility(self):
        """Test _MCMaxPoolNd base class matches PyTorch's _MaxPoolNd pattern."""
        # Test that our base class has the same constants as PyTorch
        from torch.nn.modules.pooling import _MaxPoolNd
        
        mc_pool = MCMaxPool2d(kernel_size=2)
        pytorch_pool = nn.MaxPool2d(kernel_size=2)
        
        # Check __constants__ attribute
        pytorch_constants = set(pytorch_pool.__constants__)
        mc_constants = set(mc_pool.__constants__)
        
        # Our implementation should have at least the same constants as PyTorch
        self.assertTrue(pytorch_constants.issubset(mc_constants), 
                       f"Missing constants: {pytorch_constants - mc_constants}")
        
        # Test that all expected attributes are present
        for const in pytorch_constants:
            self.assertTrue(hasattr(mc_pool, const), f"Missing attribute: {const}")
            self.assertTrue(hasattr(pytorch_pool, const), f"PyTorch missing attribute: {const}")
    
    def test_mc_adaptive_avgpool_base_class_compatibility(self):
        """Test _MCAdaptiveAvgPoolNd base class matches PyTorch's _AdaptiveAvgPoolNd pattern."""
        from torch.nn.modules.pooling import _AdaptiveAvgPoolNd
        
        mc_pool = MCAdaptiveAvgPool2d(output_size=7)
        pytorch_pool = nn.AdaptiveAvgPool2d(output_size=7)
        
        # Check __constants__ attribute
        pytorch_constants = set(pytorch_pool.__constants__)
        mc_constants = set(mc_pool.__constants__)
        
        # Our implementation should have at least the same constants as PyTorch
        self.assertTrue(pytorch_constants.issubset(mc_constants), 
                       f"Missing constants: {pytorch_constants - mc_constants}")
        
        # Test that all expected attributes are present
        for const in pytorch_constants:
            self.assertTrue(hasattr(mc_pool, const), f"Missing attribute: {const}")
            self.assertTrue(hasattr(pytorch_pool, const), f"PyTorch missing attribute: {const}")


class TestResNetChannelTransformationCompatibility(unittest.TestCase):
    """
    Test PyTorch compatibility through ResNet-style channel transformations.
    
    This ensures that as channels transform through typical ResNet patterns
    (3→64→128→256→512), both our color and brightness streams behave 
    exactly like individual PyTorch layers processing the same transformations.
    """
    
    def setUp(self):
        """Set up test fixtures with ResNet-style channel progression."""
        self.batch_size = 2
        self.spatial_size = 32
        
        # ResNet-style channel progression: 3 → 64 → 128 → 256 → 512
        self.channel_stages = [
            {"in_channels": 3, "out_channels": 64, "stage": "initial"},
            {"in_channels": 64, "out_channels": 64, "stage": "block1"},  
            {"in_channels": 64, "out_channels": 128, "stage": "block2"},
            {"in_channels": 128, "out_channels": 256, "stage": "block3"},
            {"in_channels": 256, "out_channels": 512, "stage": "block4"},
        ]
        
        # Create test inputs for each stage
        self.test_inputs = {}
        for stage_info in self.channel_stages:
            stage = stage_info["stage"]
            channels = stage_info["in_channels"]
            
            # For initial stage, color has 3 channels, brightness has 1
            if stage == "initial":
                color_channels = 3
                brightness_channels = 1
            else:
                # After initial conv, both streams have same channel count
                color_channels = channels
                brightness_channels = channels
            
            self.test_inputs[stage] = {
                "color": torch.randn(self.batch_size, color_channels, self.spatial_size, self.spatial_size),
                "brightness": torch.randn(self.batch_size, brightness_channels, self.spatial_size, self.spatial_size),
                "pytorch_reference": torch.randn(self.batch_size, color_channels, self.spatial_size, self.spatial_size)
            }
        
        torch.manual_seed(42)
    
    def test_maxpool_channel_preservation_through_stages(self):
        """Test MCMaxPool2d preserves channels correctly through ResNet stages."""
        pool_configs = [
            {"kernel_size": 2, "stride": 2},                    # Standard downsampling
            {"kernel_size": 3, "stride": 2, "padding": 1},      # ResNet-style downsampling
            {"kernel_size": 3, "stride": 1, "padding": 1},      # Same-size pooling
        ]
        
        for pool_config in pool_configs:
            with self.subTest(pool_config=pool_config):
                mc_pool = MCMaxPool2d(**pool_config)
                pytorch_pool = nn.MaxPool2d(**pool_config)
                
                for stage_info in self.channel_stages:
                    stage = stage_info["stage"]
                    
                    with self.subTest(stage=stage):
                        inputs = self.test_inputs[stage]
                        
                        # Test that PyTorch layer matches our color stream
                        pytorch_out = pytorch_pool(inputs["pytorch_reference"])
                        color_out = mc_pool.forward_color(inputs["pytorch_reference"])
                        
                        # Exact mathematical equivalence
                        torch.testing.assert_close(pytorch_out, color_out, rtol=1e-6, atol=1e-6)
                        
                        # Test dual stream preserves channel counts
                        color_dual, brightness_dual = mc_pool(inputs["color"], inputs["brightness"])
                        
                        # Channel counts should be preserved
                        self.assertEqual(color_dual.shape[1], inputs["color"].shape[1])
                        self.assertEqual(brightness_dual.shape[1], inputs["brightness"].shape[1])
                        
                        # If both inputs have same channels, outputs should have same channels
                        if inputs["color"].shape[1] == inputs["brightness"].shape[1]:
                            self.assertEqual(color_dual.shape[1], brightness_dual.shape[1])
    
    def test_adaptive_avgpool_channel_preservation_through_stages(self):
        """Test MCAdaptiveAvgPool2d preserves channels correctly through ResNet stages."""
        pool_configs = [
            {"output_size": 1},        # Global average pooling
            {"output_size": 7},        # Fixed size output
            {"output_size": (4, 8)},   # Asymmetric output
        ]
        
        for pool_config in pool_configs:
            with self.subTest(pool_config=pool_config):
                mc_pool = MCAdaptiveAvgPool2d(**pool_config)
                pytorch_pool = nn.AdaptiveAvgPool2d(**pool_config)
                
                for stage_info in self.channel_stages:
                    stage = stage_info["stage"]
                    
                    with self.subTest(stage=stage):
                        inputs = self.test_inputs[stage]
                        
                        # Test that PyTorch layer matches our color stream
                        pytorch_out = pytorch_pool(inputs["pytorch_reference"])
                        color_out = mc_pool.forward_color(inputs["pytorch_reference"])
                        
                        # Exact mathematical equivalence
                        torch.testing.assert_close(pytorch_out, color_out, rtol=1e-6, atol=1e-6)
                        
                        # Test dual stream preserves channel counts
                        color_dual, brightness_dual = mc_pool(inputs["color"], inputs["brightness"])
                        
                        # Channel counts should be preserved
                        self.assertEqual(color_dual.shape[1], inputs["color"].shape[1])
                        self.assertEqual(brightness_dual.shape[1], inputs["brightness"].shape[1])
                        
                        # If both inputs have same channels, outputs should have same channels
                        if inputs["color"].shape[1] == inputs["brightness"].shape[1]:
                            self.assertEqual(color_dual.shape[1], brightness_dual.shape[1])
    
    def test_progressive_channel_transformation_consistency(self):
        """Test that channel transformations through pooling maintain PyTorch consistency."""
        # Simulate progressive ResNet-style transformation with pooling
        from models2.multi_channel.conv import MCConv2d
        
        # Stage 1: Initial 3,1 → 64,64 transformation
        initial_conv = MCConv2d(3, 1, 64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        pool1 = MCMaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Stage 2: 64,64 → 128,128 transformation  
        conv2 = MCConv2d(64, 64, 128, 128, kernel_size=3, stride=2, padding=1, bias=False)
        pool2 = MCMaxPool2d(kernel_size=2, stride=2)
        
        # Stage 3: Final pooling to 1x1
        global_pool = MCAdaptiveAvgPool2d(1)
        
        # Create PyTorch reference path
        pytorch_conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        pytorch_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        pytorch_conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        pytorch_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        pytorch_global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Sync weights for comparison
        with torch.no_grad():
            # Copy initial conv weights
            pytorch_conv1.weight.copy_(initial_conv.color_weight)
            pytorch_conv2.weight.copy_(conv2.color_weight)
        
        # Test inputs
        color_input = torch.randn(1, 3, 112, 112)
        brightness_input = torch.randn(1, 1, 112, 112) 
        pytorch_input = color_input.clone()
        
        # Apply transformations
        # MC path
        color_out, brightness_out = initial_conv(color_input, brightness_input)
        color_out, brightness_out = pool1(color_out, brightness_out)
        # Both should now have 64 channels
        self.assertEqual(color_out.shape[1], 64)
        self.assertEqual(brightness_out.shape[1], 64)
        
        color_out, brightness_out = conv2(color_out, brightness_out)
        color_out, brightness_out = pool2(color_out, brightness_out)
        # Both should now have 128 channels
        self.assertEqual(color_out.shape[1], 128)
        self.assertEqual(brightness_out.shape[1], 128)
        
        color_final, brightness_final = global_pool(color_out, brightness_out)
        
        # PyTorch reference path (for color stream)
        pytorch_out = pytorch_conv1(pytorch_input)
        pytorch_out = pytorch_pool1(pytorch_out)
        pytorch_out = pytorch_conv2(pytorch_out)
        pytorch_out = pytorch_pool2(pytorch_out)
        pytorch_final = pytorch_global_pool(pytorch_out)
        
        # Color stream should match PyTorch exactly
        torch.testing.assert_close(color_final, pytorch_final, rtol=1e-5, atol=1e-5)
        
        # Final outputs should be 1x1 with 128 channels
        expected_final_shape = (1, 128, 1, 1)
        self.assertEqual(color_final.shape, expected_final_shape)
        self.assertEqual(brightness_final.shape, expected_final_shape)
    
    def test_batch_dimension_consistency_through_stages(self):
        """Test that batch dimension handling matches PyTorch through channel transformations."""
        batch_sizes = [1, 4, 8, 16]
        
        for batch_size in batch_sizes:
            with self.subTest(batch_size=batch_size):
                mc_pool = MCMaxPool2d(kernel_size=2, stride=2)
                pytorch_pool = nn.MaxPool2d(kernel_size=2, stride=2)
                
                for stage_info in self.channel_stages:
                    stage = stage_info["stage"] 
                    
                    with self.subTest(stage=stage):
                        # Create batch-specific inputs
                        if stage == "initial":
                            color_channels, brightness_channels = 3, 1
                        else:
                            color_channels = brightness_channels = stage_info["in_channels"]
                        
                        color_input = torch.randn(batch_size, color_channels, 16, 16)
                        pytorch_input = torch.randn(batch_size, color_channels, 16, 16)
                        
                        # Test outputs
                        pytorch_out = pytorch_pool(pytorch_input)
                        color_out = mc_pool.forward_color(pytorch_input)
                        
                        # Batch dimension should be preserved and outputs should match
                        self.assertEqual(pytorch_out.shape[0], batch_size)
                        self.assertEqual(color_out.shape[0], batch_size)
                        torch.testing.assert_close(pytorch_out, color_out, rtol=1e-6, atol=1e-6)


class TestAdvancedPyTorchEdgeCases(unittest.TestCase):
    """
    Test advanced PyTorch compatibility edge cases and boundary conditions.
    
    These tests ensure our implementation handles all edge cases exactly like PyTorch,
    including unusual input sizes, extreme parameter values, and corner cases.
    """
    
    def setUp(self):
        """Set up edge case test fixtures."""
        torch.manual_seed(42)
    
    def test_minimal_input_sizes(self):
        """Test pooling with minimal input sizes (1x1, 2x2, etc.)."""
        minimal_sizes = [(1, 1), (2, 2), (3, 3)]
        
        for input_h, input_w in minimal_sizes:
            with self.subTest(input_size=(input_h, input_w)):
                # Test MCMaxPool2d
                mc_maxpool = MCMaxPool2d(kernel_size=1)  # 1x1 kernel to preserve size
                pytorch_maxpool = nn.MaxPool2d(kernel_size=1)
                
                test_input = torch.randn(1, 16, input_h, input_w)
                
                pytorch_out = pytorch_maxpool(test_input)
                mc_color_out = mc_maxpool.forward_color(test_input)
                
                torch.testing.assert_close(pytorch_out, mc_color_out)
                
                # Test MCAdaptiveAvgPool2d
                mc_adaptive = MCAdaptiveAvgPool2d((input_h, input_w))
                pytorch_adaptive = nn.AdaptiveAvgPool2d((input_h, input_w))
                
                pytorch_out = pytorch_adaptive(test_input)
                mc_color_out = mc_adaptive.forward_color(test_input)
                
                torch.testing.assert_close(pytorch_out, mc_color_out)
    
    def test_large_input_sizes(self):
        """Test pooling with large input sizes to ensure scalability."""
        large_sizes = [(224, 224), (512, 512)]
        
        for input_h, input_w in large_sizes:
            with self.subTest(input_size=(input_h, input_w)):
                # Use smaller batch and channel sizes for memory efficiency
                test_input = torch.randn(1, 8, input_h, input_w)
                
                # Test with downsampling pooling
                mc_maxpool = MCMaxPool2d(kernel_size=8, stride=8)
                pytorch_maxpool = nn.MaxPool2d(kernel_size=8, stride=8)
                
                pytorch_out = pytorch_maxpool(test_input)
                mc_color_out = mc_maxpool.forward_color(test_input)
                
                torch.testing.assert_close(pytorch_out, mc_color_out)
                
                # Test adaptive pooling to fixed size
                mc_adaptive = MCAdaptiveAvgPool2d(7)
                pytorch_adaptive = nn.AdaptiveAvgPool2d(7)
                
                pytorch_out = pytorch_adaptive(test_input)
                mc_color_out = mc_adaptive.forward_color(test_input)
                
                torch.testing.assert_close(pytorch_out, mc_color_out)
    
    def test_extreme_kernel_sizes(self):
        """Test pooling with extreme kernel sizes."""
        # Test very large kernels relative to input
        test_input = torch.randn(1, 8, 16, 16)
        
        # Kernel size equal to input size
        mc_maxpool = MCMaxPool2d(kernel_size=16)
        pytorch_maxpool = nn.MaxPool2d(kernel_size=16)
        
        pytorch_out = pytorch_maxpool(test_input)
        mc_color_out = mc_maxpool.forward_color(test_input)
        
        torch.testing.assert_close(pytorch_out, mc_color_out)
        
        # Output should be 1x1
        self.assertEqual(pytorch_out.shape[-2:], (1, 1))
        self.assertEqual(mc_color_out.shape[-2:], (1, 1))
    
    def test_asymmetric_kernels_and_strides(self):
        """Test pooling with highly asymmetric kernels and strides."""
        asymmetric_configs = [
            {"kernel_size": (1, 8), "stride": (1, 4)},    # Horizontal pooling
            {"kernel_size": (8, 1), "stride": (4, 1)},    # Vertical pooling  
            {"kernel_size": (7, 3), "stride": (3, 2)},    # Mixed asymmetric
        ]
        
        for config in asymmetric_configs:
            with self.subTest(config=config):
                test_input = torch.randn(1, 16, 32, 32)
                
                mc_maxpool = MCMaxPool2d(**config)
                pytorch_maxpool = nn.MaxPool2d(**config)
                
                pytorch_out = pytorch_maxpool(test_input)
                mc_color_out = mc_maxpool.forward_color(test_input)
                
                torch.testing.assert_close(pytorch_out, mc_color_out)
    
    def test_single_channel_compatibility(self):
        """Test that single-channel inputs work exactly like PyTorch."""
        # This is important for brightness stream which often starts with 1 channel
        single_channel_input = torch.randn(2, 1, 16, 16)
        
        # Test MCMaxPool2d
        mc_maxpool = MCMaxPool2d(kernel_size=2, stride=2)
        pytorch_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        pytorch_out = pytorch_maxpool(single_channel_input)
        mc_brightness_out = mc_maxpool.forward_brightness(single_channel_input)
        
        torch.testing.assert_close(pytorch_out, mc_brightness_out)
        
        # Test MCAdaptiveAvgPool2d
        mc_adaptive = MCAdaptiveAvgPool2d(4)
        pytorch_adaptive = nn.AdaptiveAvgPool2d(4)
        
        pytorch_out = pytorch_adaptive(single_channel_input)
        mc_brightness_out = mc_adaptive.forward_brightness(single_channel_input)
        
        torch.testing.assert_close(pytorch_out, mc_brightness_out)
    
    def test_very_large_channel_counts(self):
        """Test pooling with very large channel counts."""
        large_channel_counts = [512, 1024, 2048]
        
        for channels in large_channel_counts:
            with self.subTest(channels=channels):
                # Use smaller spatial size for memory efficiency
                test_input = torch.randn(1, channels, 8, 8)
                
                # Test MCMaxPool2d
                mc_maxpool = MCMaxPool2d(kernel_size=2, stride=2)
                pytorch_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
                
                pytorch_out = pytorch_maxpool(test_input)
                mc_color_out = mc_maxpool.forward_color(test_input)
                
                torch.testing.assert_close(pytorch_out, mc_color_out)
                
                # Channel count should be preserved
                self.assertEqual(pytorch_out.shape[1], channels)
                self.assertEqual(mc_color_out.shape[1], channels)


if __name__ == '__main__':
    unittest.main()
