"""
Comprehensive unit tests for multi-channel blocks.py module.
Tests MCBasicBlock and MCBottleneck classes.
"""

import unittest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent / "src"))

from src.models2.multi_channel.blocks import MCBasicBlock, MCBottleneck, mc_conv3x3, mc_conv1x1
from src.models2.multi_channel.conv import MCConv2d, MCBatchNorm2d
from src.models2.multi_channel.container import MCSequential


class TestMCBasicBlock(unittest.TestCase):
    """Test cases for MCBasicBlock."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.color_inplanes = 64
        self.brightness_inplanes = 64
        self.color_planes = 64
        self.brightness_planes = 64
        
        self.block = MCBasicBlock(
            color_inplanes=self.color_inplanes,
            brightness_inplanes=self.brightness_inplanes,
            color_planes=self.color_planes,
            brightness_planes=self.brightness_planes
        )
        
        # Test inputs
        self.batch_size = 4
        self.input_size = 32
        self.color_input = torch.randn(self.batch_size, self.color_inplanes, self.input_size, self.input_size)
        self.brightness_input = torch.randn(self.batch_size, self.brightness_inplanes, self.input_size, self.input_size)
    
    def test_initialization(self):
        """Test proper initialization of MCBasicBlock."""
        # Check expansion factor
        self.assertEqual(MCBasicBlock.expansion, 1)
        
        # Check output planes calculation
        self.assertEqual(self.block.color_outplanes, self.color_planes * MCBasicBlock.expansion)
        self.assertEqual(self.block.brightness_outplanes, self.brightness_planes * MCBasicBlock.expansion)
        
        # Check layer components exist
        self.assertIsInstance(self.block.conv1, MCConv2d)
        self.assertIsInstance(self.block.conv2, MCConv2d)
        self.assertIsInstance(self.block.bn1, MCBatchNorm2d)
        self.assertIsInstance(self.block.bn2, MCBatchNorm2d)
        self.assertIsInstance(self.block.relu, nn.ReLU)
    
    def test_forward_dual_channel(self):
        """Test forward pass with both color and brightness inputs."""
        color_out, brightness_out = self.block(self.color_input, self.brightness_input)
        
        # Check output shapes
        expected_color_shape = (self.batch_size, self.color_planes, self.input_size, self.input_size)
        expected_brightness_shape = (self.batch_size, self.brightness_planes, self.input_size, self.input_size)
        
        self.assertEqual(color_out.shape, expected_color_shape)
        self.assertEqual(brightness_out.shape, expected_brightness_shape)
        
        # Check outputs are tensors
        self.assertIsInstance(color_out, torch.Tensor)
        self.assertIsInstance(brightness_out, torch.Tensor)
    
    def test_forward_color_only(self):
        """Test forward pass through color pathway only."""
        color_out = self.block.forward_color(self.color_input)
        
        expected_shape = (self.batch_size, self.color_planes, self.input_size, self.input_size)
        self.assertEqual(color_out.shape, expected_shape)
        self.assertIsInstance(color_out, torch.Tensor)
    
    def test_forward_brightness_only(self):
        """Test forward pass through brightness pathway only."""
        brightness_out = self.block.forward_brightness(self.brightness_input)
        
        expected_shape = (self.batch_size, self.brightness_planes, self.input_size, self.input_size)
        self.assertEqual(brightness_out.shape, expected_shape)
        self.assertIsInstance(brightness_out, torch.Tensor)
    
    def test_stride_downsampling(self):
        """Test block with stride for downsampling."""
        # Create a downsample module for the skip connection
        downsample = MCSequential(
            MCConv2d(64, 64, 128, 128, kernel_size=1, stride=2, bias=False),
            MCBatchNorm2d(128, 128)
        )
        
        block_stride = MCBasicBlock(
            color_inplanes=64,
            brightness_inplanes=64,
            color_planes=128,
            brightness_planes=128,
            stride=2,
            downsample=downsample
        )
        
        color_out, brightness_out = block_stride(self.color_input, self.brightness_input)
        
        # With stride=2, output should be half the input size
        expected_size = self.input_size // 2
        expected_color_shape = (self.batch_size, 128, expected_size, expected_size)
        expected_brightness_shape = (self.batch_size, 128, expected_size, expected_size)
        
        self.assertEqual(color_out.shape, expected_color_shape)
        self.assertEqual(brightness_out.shape, expected_brightness_shape)
    
    def test_with_downsample(self):
        """Test block with downsample layer."""
        downsample = MCSequential(
            MCConv2d(64, 64, 128, 128, kernel_size=1, stride=2),
            MCBatchNorm2d(128, 128)
        )
        
        block_downsample = MCBasicBlock(
            color_inplanes=64,
            brightness_inplanes=64,
            color_planes=128,
            brightness_planes=128,
            stride=2,
            downsample=downsample
        )
        
        color_out, brightness_out = block_downsample(self.color_input, self.brightness_input)
        
        expected_size = self.input_size // 2
        expected_color_shape = (self.batch_size, 128, expected_size, expected_size)
        expected_brightness_shape = (self.batch_size, 128, expected_size, expected_size)
        
        self.assertEqual(color_out.shape, expected_color_shape)
        self.assertEqual(brightness_out.shape, expected_brightness_shape)
    
    def test_gradient_flow(self):
        """Test that gradients flow correctly through both pathways."""
        color_input = self.color_input.clone().requires_grad_(True)
        brightness_input = self.brightness_input.clone().requires_grad_(True)
        
        color_out, brightness_out = self.block(color_input, brightness_input)
        loss = color_out.sum() + brightness_out.sum()
        loss.backward()
        
        # Check gradients exist
        self.assertIsNotNone(color_input.grad)
        self.assertIsNotNone(brightness_input.grad)


class TestMCBottleneck(unittest.TestCase):
    """Test cases for MCBottleneck."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.color_inplanes = 64
        self.brightness_inplanes = 64
        self.color_planes = 64
        self.brightness_planes = 64
        
        # Create downsample for proper dimension matching
        self.downsample = MCSequential(
            MCConv2d(64, 64, 256, 256, kernel_size=1),  # 64 * 4 = 256 (expansion)
            MCBatchNorm2d(256, 256)
        )
        
        self.block = MCBottleneck(
            color_inplanes=self.color_inplanes,
            brightness_inplanes=self.brightness_inplanes,
            color_planes=self.color_planes,
            brightness_planes=self.brightness_planes,
            downsample=self.downsample
        )
        
        # Test inputs
        self.batch_size = 4
        self.input_size = 32
        self.color_input = torch.randn(self.batch_size, self.color_inplanes, self.input_size, self.input_size)
        self.brightness_input = torch.randn(self.batch_size, self.brightness_inplanes, self.input_size, self.input_size)
    
    def test_initialization(self):
        """Test proper initialization of MCBottleneck."""
        # Check expansion factor
        self.assertEqual(MCBottleneck.expansion, 4)
        
        # Check output planes calculation
        expected_color_outplanes = self.color_planes * MCBottleneck.expansion
        expected_brightness_outplanes = self.brightness_planes * MCBottleneck.expansion
        self.assertEqual(self.block.color_outplanes, expected_color_outplanes)
        self.assertEqual(self.block.brightness_outplanes, expected_brightness_outplanes)
        
        # Check layer components exist
        self.assertIsInstance(self.block.conv1, MCConv2d)
        self.assertIsInstance(self.block.conv2, MCConv2d)
        self.assertIsInstance(self.block.conv3, MCConv2d)
        self.assertIsInstance(self.block.bn1, MCBatchNorm2d)
        self.assertIsInstance(self.block.bn2, MCBatchNorm2d)
        self.assertIsInstance(self.block.bn3, MCBatchNorm2d)
        self.assertIsInstance(self.block.relu, nn.ReLU)
    
    def test_forward_dual_channel(self):
        """Test forward pass with both color and brightness inputs."""
        color_out, brightness_out = self.block(self.color_input, self.brightness_input)
        
        # Check output shapes (expanded by factor of 4)
        expected_color_shape = (self.batch_size, self.color_planes * 4, self.input_size, self.input_size)
        expected_brightness_shape = (self.batch_size, self.brightness_planes * 4, self.input_size, self.input_size)
        
        self.assertEqual(color_out.shape, expected_color_shape)
        self.assertEqual(brightness_out.shape, expected_brightness_shape)
        
        # Check outputs are tensors
        self.assertIsInstance(color_out, torch.Tensor)
        self.assertIsInstance(brightness_out, torch.Tensor)
    
    def test_forward_color_only(self):
        """Test forward pass through color pathway only."""
        color_out = self.block.forward_color(self.color_input)
        
        expected_shape = (self.batch_size, self.color_planes * 4, self.input_size, self.input_size)
        self.assertEqual(color_out.shape, expected_shape)
        self.assertIsInstance(color_out, torch.Tensor)
    
    def test_forward_brightness_only(self):
        """Test forward pass through brightness pathway only."""
        brightness_out = self.block.forward_brightness(self.brightness_input)
        
        expected_shape = (self.batch_size, self.brightness_planes * 4, self.input_size, self.input_size)
        self.assertEqual(brightness_out.shape, expected_shape)
        self.assertIsInstance(brightness_out, torch.Tensor)
    
    def test_width_and_groups(self):
        """Test bottleneck with different width and groups."""
        block_width = MCBottleneck(
            color_inplanes=64,
            brightness_inplanes=64,
            color_planes=64,
            brightness_planes=64,
            base_width=128,  # Double width
            groups=1,
            downsample=self.downsample
        )
        
        color_out, brightness_out = block_width(self.color_input, self.brightness_input)
        
        # Should still produce same output shape
        expected_color_shape = (self.batch_size, 256, self.input_size, self.input_size)
        expected_brightness_shape = (self.batch_size, 256, self.input_size, self.input_size)
        
        self.assertEqual(color_out.shape, expected_color_shape)
        self.assertEqual(brightness_out.shape, expected_brightness_shape)


class TestConvHelpers(unittest.TestCase):
    """Test cases for convolution helper functions."""
    
    def test_mc_conv3x3(self):
        """Test mc_conv3x3 helper function."""
        conv = mc_conv3x3(3, 1, 64, 32)
        
        self.assertIsInstance(conv, MCConv2d)
        self.assertEqual(conv.kernel_size, (3, 3))
        self.assertEqual(conv.padding, (1, 1))
        self.assertFalse(conv.bias is not None)  # bias=False by default
    
    def test_mc_conv1x1(self):
        """Test mc_conv1x1 helper function."""
        conv = mc_conv1x1(64, 32, 128, 64)
        
        self.assertIsInstance(conv, MCConv2d)
        self.assertEqual(conv.kernel_size, (1, 1))
        self.assertEqual(conv.padding, (0, 0))
        self.assertFalse(conv.bias is not None)  # bias=False by default
    
    def test_mc_conv3x3_with_stride(self):
        """Test mc_conv3x3 with stride parameter."""
        conv = mc_conv3x3(3, 1, 64, 32, stride=2)
        
        self.assertEqual(conv.stride, (2, 2))
    
    def test_mc_conv1x1_with_stride(self):
        """Test mc_conv1x1 with stride parameter."""
        conv = mc_conv1x1(64, 32, 128, 64, stride=2)
        
        self.assertEqual(conv.stride, (2, 2))


if __name__ == "__main__":
    unittest.main()
