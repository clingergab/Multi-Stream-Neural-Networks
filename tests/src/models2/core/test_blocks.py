"""
Unit tests for core blocks.
"""

import unittest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent / "src"))

from src.models2.core.blocks import BasicBlock, Bottleneck
from src.models2.core.conv import conv1x1, conv3x3


class TestBasicBlock(unittest.TestCase):
    """Test cases for the BasicBlock class."""
    
    def test_init(self):
        """Test initialization with different parameters."""
        # Standard initialization
        inplanes = 64
        planes = 128
        block = BasicBlock(inplanes, planes)
        
        # Check instance and expansion
        self.assertIsInstance(block, BasicBlock)
        self.assertEqual(block.expansion, 1)
        
        # Check that the convolutional layers have the correct parameters
        self.assertEqual(block.conv1.in_channels, inplanes)
        self.assertEqual(block.conv1.out_channels, planes)
        self.assertEqual(block.conv2.in_channels, planes)
        self.assertEqual(block.conv2.out_channels, planes)
    
    def test_forward(self):
        """Test forward pass with different inputs."""
        # Setup
        batch_size = 4
        channels = 64
        input_size = 32
        x = torch.randn(batch_size, channels, input_size, input_size)
        
        # Test with identity (no downsample)
        block = BasicBlock(channels, channels)
        output = block(x)
        
        # Check shape preserved
        self.assertEqual(output.shape, x.shape)
        
        # Test with downsample (stride=2)
        block_down = BasicBlock(
            channels, 
            channels * 2, 
            stride=2, 
            downsample=torch.nn.Sequential(
                conv1x1(channels, channels * 2, stride=2),
                torch.nn.BatchNorm2d(channels * 2),
            )
        )
        output_down = block_down(x)
        expected_shape = (batch_size, channels * 2, input_size // 2, input_size // 2)
        self.assertEqual(output_down.shape, expected_shape)
    
    def test_invalid_params(self):
        """Test that invalid parameters raise appropriate errors."""
        # Test with groups != 1
        with self.assertRaises(ValueError):
            BasicBlock(64, 64, groups=2)
        
        # Test with base_width != 64
        with self.assertRaises(ValueError):
            BasicBlock(64, 64, base_width=32)
        
        # Test with dilation > 1
        with self.assertRaises(NotImplementedError):
            BasicBlock(64, 64, dilation=2)


class TestBottleneck(unittest.TestCase):
    """Test cases for the Bottleneck class."""
    
    def test_init(self):
        """Test initialization with different parameters."""
        # Standard initialization
        inplanes = 64
        planes = 64  # Output will be planes * expansion = 64 * 4 = 256
        block = Bottleneck(inplanes, planes)
        
        # Check instance and expansion
        self.assertIsInstance(block, Bottleneck)
        self.assertEqual(block.expansion, 4)
        
        # Check that the convolutional layers have the correct parameters
        self.assertEqual(block.conv1.in_channels, inplanes)
        self.assertEqual(block.conv1.out_channels, planes)  # width = planes * (base_width/64) * groups
        self.assertEqual(block.conv2.in_channels, planes)
        self.assertEqual(block.conv2.out_channels, planes)
        self.assertEqual(block.conv3.in_channels, planes)
        self.assertEqual(block.conv3.out_channels, planes * block.expansion)
    
    def test_forward(self):
        """Test forward pass with different inputs."""
        # Setup
        batch_size = 4
        channels = 64
        input_size = 32
        x = torch.randn(batch_size, channels, input_size, input_size)
        
        # Test with identity (no downsample)
        block = Bottleneck(channels, channels // 4)  # Output will be (channels // 4) * 4 = channels
        output = block(x)
        
        # Check shape preserved
        self.assertEqual(output.shape, x.shape)
        
        # Test with downsample (stride=2)
        output_channels = (channels // 2) * 4  # (channels // 2) * expansion
        block_down = Bottleneck(
            channels, 
            channels // 2, 
            stride=2, 
            downsample=torch.nn.Sequential(
                conv1x1(channels, output_channels, stride=2),
                torch.nn.BatchNorm2d(output_channels),
            )
        )
        output_down = block_down(x)
        expected_shape = (batch_size, output_channels, input_size // 2, input_size // 2)
        self.assertEqual(output_down.shape, expected_shape)
    
    def test_groups_and_width(self):
        """Test Bottleneck with different groups and width_per_group."""
        # Setup
        batch_size = 4
        channels = 64
        input_size = 32
        x = torch.randn(batch_size, channels, input_size, input_size)
        
        # Test with groups=2, base_width=64
        block = Bottleneck(channels, channels // 4, groups=2)
        output = block(x)
        self.assertEqual(output.shape, x.shape)
        
        # Test with base_width=32
        block = Bottleneck(channels, channels // 4, base_width=32)
        output = block(x)
        self.assertEqual(output.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
