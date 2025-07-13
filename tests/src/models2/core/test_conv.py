"""
Unit tests for core convolution functions.
"""

import unittest
import torch
import sys
from pathlib import Path

from models2.core.conv import conv1x1, conv3x3


class TestConvFunctions(unittest.TestCase):
    """Test cases for core convolution functions."""
    
    def test_conv3x3_basic(self):
        """Test basic functionality of conv3x3."""
        # Test with standard parameters
        in_channels = 64
        out_channels = 128
        conv = conv3x3(in_channels, out_channels)
        
        # Check type
        self.assertIsInstance(conv, torch.nn.Conv2d)
        
        # Check parameters
        self.assertEqual(conv.in_channels, in_channels)
        self.assertEqual(conv.out_channels, out_channels)
        self.assertEqual(conv.kernel_size, (3, 3))
        self.assertEqual(conv.stride, (1, 1))
        self.assertEqual(conv.padding, (1, 1))
        self.assertEqual(conv.dilation, (1, 1))
        self.assertEqual(conv.groups, 1)
        self.assertFalse(conv.bias is not None)
    
    def test_conv3x3_stride(self):
        """Test conv3x3 with custom stride."""
        in_channels = 64
        out_channels = 128
        stride = 2
        conv = conv3x3(in_channels, out_channels, stride=stride)
        
        self.assertEqual(conv.stride, (stride, stride))
    
    def test_conv3x3_groups(self):
        """Test conv3x3 with groups."""
        in_channels = 64
        out_channels = 128
        groups = 4
        conv = conv3x3(in_channels, out_channels, groups=groups)
        
        self.assertEqual(conv.groups, groups)
    
    def test_conv3x3_dilation(self):
        """Test conv3x3 with dilation."""
        in_channels = 64
        out_channels = 128
        dilation = 2
        conv = conv3x3(in_channels, out_channels, dilation=dilation)
        
        self.assertEqual(conv.dilation, (dilation, dilation))
        self.assertEqual(conv.padding, (dilation, dilation))
    
    def test_conv1x1_basic(self):
        """Test basic functionality of conv1x1."""
        # Test with standard parameters
        in_channels = 64
        out_channels = 128
        conv = conv1x1(in_channels, out_channels)
        
        # Check type
        self.assertIsInstance(conv, torch.nn.Conv2d)
        
        # Check parameters
        self.assertEqual(conv.in_channels, in_channels)
        self.assertEqual(conv.out_channels, out_channels)
        self.assertEqual(conv.kernel_size, (1, 1))
        self.assertEqual(conv.stride, (1, 1))
        self.assertEqual(conv.padding, (0, 0))
        self.assertEqual(conv.groups, 1)
        self.assertFalse(conv.bias is not None)
    
    def test_conv1x1_stride(self):
        """Test conv1x1 with custom stride."""
        in_channels = 64
        out_channels = 128
        stride = 2
        conv = conv1x1(in_channels, out_channels, stride=stride)
        
        self.assertEqual(conv.stride, (stride, stride))
    
    def test_forward_pass(self):
        """Test forward pass through convolution layers."""
        # Create sample input
        batch_size = 4
        in_channels = 64
        out_channels = 128
        height, width = 32, 32
        x = torch.randn(batch_size, in_channels, height, width)
        
        # Test conv3x3
        conv3 = conv3x3(in_channels, out_channels)
        output3 = conv3(x)
        self.assertEqual(output3.shape, (batch_size, out_channels, height, width))
        
        # Test conv1x1
        conv1 = conv1x1(in_channels, out_channels)
        output1 = conv1(x)
        self.assertEqual(output1.shape, (batch_size, out_channels, height, width))
        
        # Test with stride
        stride = 2
        conv3_stride = conv3x3(in_channels, out_channels, stride=stride)
        output3_stride = conv3_stride(x)
        self.assertEqual(output3_stride.shape, (batch_size, out_channels, height//stride, width//stride))
        
        conv1_stride = conv1x1(in_channels, out_channels, stride=stride)
        output1_stride = conv1_stride(x)
        self.assertEqual(output1_stride.shape, (batch_size, out_channels, height//stride, width//stride))


if __name__ == "__main__":
    unittest.main()
