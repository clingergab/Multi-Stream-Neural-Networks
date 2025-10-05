"""
Comprehensive unit tests for multi-channel blocks.py module.
Tests MCBasicBlock and MCBottleneck classes.
"""

import unittest
import torch
import torch.nn as nn
import sys
from pathlib import Path

from models.multi_channel.blocks import MCBasicBlock, MCBottleneck, mc_conv3x3, mc_conv1x1
from models.multi_channel.conv import MCConv2d, MCBatchNorm2d
from models.multi_channel.container import MCSequential, MCReLU


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
        self.assertIsInstance(self.block.relu, MCReLU)
    
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
        self.assertIsInstance(self.block.relu, MCReLU)
    
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
        self.assertIsNone(conv.color_bias)  # bias=False by default
        self.assertIsNone(conv.brightness_bias)  # bias=False by default
    
    def test_mc_conv1x1(self):
        """Test mc_conv1x1 helper function."""
        conv = mc_conv1x1(64, 32, 128, 64)
        
        self.assertIsInstance(conv, MCConv2d)
        self.assertEqual(conv.kernel_size, (1, 1))
        self.assertEqual(conv.padding, (0, 0))
        self.assertIsNone(conv.color_bias)  # bias=False by default
        self.assertIsNone(conv.brightness_bias)  # bias=False by default
    
    def test_mc_conv3x3_with_stride(self):
        """Test mc_conv3x3 with stride parameter."""
        conv = mc_conv3x3(3, 1, 64, 32, stride=2)
        
        self.assertEqual(conv.stride, (2, 2))
    
    def test_mc_conv1x1_with_stride(self):
        """Test mc_conv1x1 with stride parameter."""
        conv = mc_conv1x1(64, 32, 128, 64, stride=2)
        
        self.assertEqual(conv.stride, (2, 2))


class TestPyTorchCompatibility(unittest.TestCase):
    """Test PyTorch compatibility for all multi-channel block layers."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Both pathways follow same channel transformations after initial conv
        # Example: color 3->64->64, brightness 1->64->64 (same after first transformation)
        self.input_size = 32
        self.batch_size = 2
        
        # Inputs after initial transformation - both pathways have same channels
        self.color_input_64 = torch.randn(self.batch_size, 64, self.input_size, self.input_size)
        self.brightness_input_64 = torch.randn(self.batch_size, 64, self.input_size, self.input_size)
        
        # For PyTorch comparison
        self.pytorch_test_input = torch.randn(self.batch_size, 64, self.input_size, self.input_size)
    
    def test_mcbasicblock_vs_pytorch_basicblock(self):
        """Test MCBasicBlock gives identical results to PyTorch BasicBlock from torchvision."""
        from torchvision.models.resnet import BasicBlock
        
        # Both pathways use same transformation: 64 -> 64
        mc_block = MCBasicBlock(
            color_inplanes=64,
            brightness_inplanes=64, 
            color_planes=64,
            brightness_planes=64
        )
        
        # PyTorch BasicBlock for comparison
        pytorch_block = BasicBlock(inplanes=64, planes=64)
        
        # Copy weights from MC block to PyTorch block for fair comparison
        # This ensures we're testing the algorithm, not random weight differences
        with torch.no_grad():
            # Copy conv1 weights
            pytorch_block.conv1.weight.copy_(mc_block.conv1.color_weight)
            pytorch_block.bn1.weight.copy_(mc_block.bn1.color_weight)
            pytorch_block.bn1.bias.copy_(mc_block.bn1.color_bias)
            pytorch_block.bn1.running_mean.copy_(mc_block.bn1.color_running_mean)
            pytorch_block.bn1.running_var.copy_(mc_block.bn1.color_running_var)
            
            # Copy conv2 weights
            pytorch_block.conv2.weight.copy_(mc_block.conv2.color_weight)
            pytorch_block.bn2.weight.copy_(mc_block.bn2.color_weight)
            pytorch_block.bn2.bias.copy_(mc_block.bn2.color_bias)
            pytorch_block.bn2.running_mean.copy_(mc_block.bn2.color_running_mean)
            pytorch_block.bn2.running_var.copy_(mc_block.bn2.color_running_var)
        
        # Set all to eval mode for consistent batch norm behavior
        mc_block.eval()
        pytorch_block.eval()
        
        # Use same input for both MC pathways and PyTorch layer
        test_input = self.pytorch_test_input
        
        # Test color pathway against PyTorch
        pytorch_color_out = pytorch_block(test_input)
        mc_color_out = mc_block.forward_color(test_input)
        
        torch.testing.assert_close(mc_color_out, pytorch_color_out, atol=1e-5, rtol=1e-5,
                                 msg="Color pathway output differs from PyTorch BasicBlock")
        
        # Test brightness pathway - copy brightness weights to same PyTorch block
        with torch.no_grad():
            # Copy brightness pathway weights
            pytorch_block.conv1.weight.copy_(mc_block.conv1.brightness_weight)
            pytorch_block.bn1.weight.copy_(mc_block.bn1.brightness_weight)
            pytorch_block.bn1.bias.copy_(mc_block.bn1.brightness_bias)
            pytorch_block.bn1.running_mean.copy_(mc_block.bn1.brightness_running_mean)
            pytorch_block.bn1.running_var.copy_(mc_block.bn1.brightness_running_var)
            pytorch_block.conv2.weight.copy_(mc_block.conv2.brightness_weight)
            pytorch_block.bn2.weight.copy_(mc_block.bn2.brightness_weight)
            pytorch_block.bn2.bias.copy_(mc_block.bn2.brightness_bias)
            pytorch_block.bn2.running_mean.copy_(mc_block.bn2.brightness_running_mean)
            pytorch_block.bn2.running_var.copy_(mc_block.bn2.brightness_running_var)
        
        pytorch_brightness_out = pytorch_block(test_input)
        mc_brightness_out = mc_block.forward_brightness(test_input)
        
        torch.testing.assert_close(mc_brightness_out, pytorch_brightness_out, atol=1e-5, rtol=1e-5,
                                 msg="Brightness pathway output differs from PyTorch BasicBlock")
        
        # Test dual pathway forward - restore color weights for final test
        with torch.no_grad():
            pytorch_block.conv1.weight.copy_(mc_block.conv1.color_weight)
            pytorch_block.bn1.weight.copy_(mc_block.bn1.color_weight)
            pytorch_block.bn1.bias.copy_(mc_block.bn1.color_bias)
            pytorch_block.bn1.running_mean.copy_(mc_block.bn1.color_running_mean)
            pytorch_block.bn1.running_var.copy_(mc_block.bn1.color_running_var)
            pytorch_block.conv2.weight.copy_(mc_block.conv2.color_weight)
            pytorch_block.bn2.weight.copy_(mc_block.bn2.color_weight)
            pytorch_block.bn2.bias.copy_(mc_block.bn2.color_bias)
            pytorch_block.bn2.running_mean.copy_(mc_block.bn2.color_running_mean)
            pytorch_block.bn2.running_var.copy_(mc_block.bn2.color_running_var)
        
        mc_color_dual, mc_brightness_dual = mc_block(test_input, test_input)
        
        torch.testing.assert_close(mc_color_dual, pytorch_color_out, atol=1e-5, rtol=1e-5,
                                 msg="Dual-stream color output differs from single pathway")
    
    def test_mcbottleneck_vs_pytorch_bottleneck(self):
        """Test MCBottleneck gives identical results to PyTorch Bottleneck from torchvision."""
        from torchvision.models.resnet import Bottleneck
        
        # Both pathways use same transformation: 64 -> 16 -> 64 (with expansion=4)
        mc_block = MCBottleneck(
            color_inplanes=64,
            brightness_inplanes=64,
            color_planes=16,  # Bottleneck reduces then expands
            brightness_planes=16
        )
        
        # PyTorch Bottleneck for comparison  
        pytorch_block = Bottleneck(inplanes=64, planes=16)
        
        # Copy weights for fair comparison
        with torch.no_grad():
            # Copy conv1 weights (1x1)
            pytorch_block.conv1.weight.copy_(mc_block.conv1.color_weight)
            pytorch_block.bn1.weight.copy_(mc_block.bn1.color_weight)
            pytorch_block.bn1.bias.copy_(mc_block.bn1.color_bias)
            pytorch_block.bn1.running_mean.copy_(mc_block.bn1.color_running_mean)
            pytorch_block.bn1.running_var.copy_(mc_block.bn1.color_running_var)
            
            # Copy conv2 weights (3x3)
            pytorch_block.conv2.weight.copy_(mc_block.conv2.color_weight)
            pytorch_block.bn2.weight.copy_(mc_block.bn2.color_weight)
            pytorch_block.bn2.bias.copy_(mc_block.bn2.color_bias)
            pytorch_block.bn2.running_mean.copy_(mc_block.bn2.color_running_mean)
            pytorch_block.bn2.running_var.copy_(mc_block.bn2.color_running_var)
            
            # Copy conv3 weights (1x1)
            pytorch_block.conv3.weight.copy_(mc_block.conv3.color_weight)
            pytorch_block.bn3.weight.copy_(mc_block.bn3.color_weight)
            pytorch_block.bn3.bias.copy_(mc_block.bn3.color_bias)
            pytorch_block.bn3.running_mean.copy_(mc_block.bn3.color_running_mean)
            pytorch_block.bn3.running_var.copy_(mc_block.bn3.color_running_var)
        
        # Set all to eval mode
        mc_block.eval()
        pytorch_block.eval()
        
        # Use same input for both MC pathways and PyTorch layer
        test_input = self.pytorch_test_input
        
        # Test color pathway against PyTorch
        pytorch_color_out = pytorch_block(test_input)
        mc_color_out = mc_block.forward_color(test_input)
        
        torch.testing.assert_close(mc_color_out, pytorch_color_out, atol=1e-5, rtol=1e-5,
                                 msg="Color pathway output differs from PyTorch Bottleneck")
        
        # Test brightness pathway - copy brightness weights and use same input
        with torch.no_grad():
            # Copy brightness weights to same PyTorch block
            pytorch_block.conv1.weight.copy_(mc_block.conv1.brightness_weight)
            pytorch_block.bn1.weight.copy_(mc_block.bn1.brightness_weight)
            pytorch_block.bn1.bias.copy_(mc_block.bn1.brightness_bias)
            pytorch_block.bn1.running_mean.copy_(mc_block.bn1.brightness_running_mean)
            pytorch_block.bn1.running_var.copy_(mc_block.bn1.brightness_running_var)
            pytorch_block.conv2.weight.copy_(mc_block.conv2.brightness_weight)
            pytorch_block.bn2.weight.copy_(mc_block.bn2.brightness_weight)
            pytorch_block.bn2.bias.copy_(mc_block.bn2.brightness_bias)
            pytorch_block.bn2.running_mean.copy_(mc_block.bn2.brightness_running_mean)
            pytorch_block.bn2.running_var.copy_(mc_block.bn2.brightness_running_var)
            pytorch_block.conv3.weight.copy_(mc_block.conv3.brightness_weight)
            pytorch_block.bn3.weight.copy_(mc_block.bn3.brightness_weight)
            pytorch_block.bn3.bias.copy_(mc_block.bn3.brightness_bias)
            pytorch_block.bn3.running_mean.copy_(mc_block.bn3.brightness_running_mean)
            pytorch_block.bn3.running_var.copy_(mc_block.bn3.brightness_running_var)
        
        pytorch_brightness_out = pytorch_block(test_input)
        mc_brightness_out = mc_block.forward_brightness(test_input)
        
        torch.testing.assert_close(mc_brightness_out, pytorch_brightness_out, atol=1e-5, rtol=1e-5,
                                 msg="Brightness pathway output differs from PyTorch Bottleneck")
        
        torch.testing.assert_close(mc_brightness_out, pytorch_brightness_out, atol=1e-5, rtol=1e-5,
                                 msg="Brightness pathway output differs from PyTorch Bottleneck")
        
        # Test dual pathway forward
        mc_color_dual, mc_brightness_dual = mc_block(test_input, test_input)
        
        torch.testing.assert_close(mc_color_dual, pytorch_color_out, atol=1e-5, rtol=1e-5,
                                 msg="Dual-stream color output differs from single pathway")
        torch.testing.assert_close(mc_brightness_dual, pytorch_brightness_out, atol=1e-5, rtol=1e-5,
                                 msg="Dual-stream brightness output differs from single pathway")
    
    def test_mcbasicblock_with_downsample_vs_pytorch(self):
        """Test MCBasicBlock with downsampling matches PyTorch behavior."""
        from torchvision.models.resnet import BasicBlock
        
        # Test with stride=2 (downsampling case)
        mc_block = MCBasicBlock(
            color_inplanes=64,
            brightness_inplanes=64,
            color_planes=128,  # Increase channels
            brightness_planes=128,
            stride=2
        )
        
        pytorch_block = BasicBlock(inplanes=64, planes=128, stride=2)
        
        # Neither should have downsample modules when not explicitly provided
        # (This matches PyTorch behavior - downsample must be created externally)
        self.assertIsNone(mc_block.downsample)
        self.assertIsNone(pytorch_block.downsample)
        
        # Set to eval mode
        mc_block.eval()
        pytorch_block.eval()
        
        # Test forward pass (don't copy weights since we're testing structure)
        # Use same input size - note that without downsample, this will fail due to dimension mismatch
        # But we can test the structure is correct by checking the error is the same
        with self.assertRaises(RuntimeError) as mc_cm:
            mc_color_out, mc_brightness_out = mc_block(self.color_input_64, self.brightness_input_64)
        
        with self.assertRaises(RuntimeError) as pytorch_cm:
            pytorch_out = pytorch_block(self.pytorch_test_input)
        
        # Both should fail with similar error messages about tensor size mismatch
        self.assertIn("must match the size", str(mc_cm.exception))
        self.assertIn("must match the size", str(pytorch_cm.exception))
    
    def test_dual_stream_consistency(self):
        """Test that dual-stream forward gives same results as individual pathways."""
        # Test MCBasicBlock
        mc_basic = MCBasicBlock(
            color_inplanes=64,
            brightness_inplanes=64,
            color_planes=64,
            brightness_planes=64
        )
        mc_basic.eval()
        
        # Dual stream forward
        color_dual, brightness_dual = mc_basic(self.color_input_64, self.brightness_input_64)
        
        # Individual pathway forwards
        color_single = mc_basic.forward_color(self.color_input_64)
        brightness_single = mc_basic.forward_brightness(self.brightness_input_64)
        
        torch.testing.assert_close(color_dual, color_single)
        torch.testing.assert_close(brightness_dual, brightness_single)
        
        # Test MCBottleneck
        mc_bottleneck = MCBottleneck(
            color_inplanes=64,
            brightness_inplanes=64,
            color_planes=16,
            brightness_planes=16
        )
        mc_bottleneck.eval()
        
        # Dual stream forward
        color_dual_bn, brightness_dual_bn = mc_bottleneck(self.color_input_64, self.brightness_input_64)
        
        # Individual pathway forwards
        color_single_bn = mc_bottleneck.forward_color(self.color_input_64)
        brightness_single_bn = mc_bottleneck.forward_brightness(self.brightness_input_64)
        
        torch.testing.assert_close(color_dual_bn, color_single_bn)
        torch.testing.assert_close(brightness_dual_bn, brightness_single_bn)
    
    def test_block_parameter_compatibility(self):
        """Test that block parameters follow PyTorch patterns."""
        from torchvision.models.resnet import BasicBlock, Bottleneck
        
        # Test BasicBlock parameters
        mc_basic = MCBasicBlock(64, 64, 64, 64)
        pytorch_basic = BasicBlock(64, 64)
        
        # Count parameters (MC should have ~2x due to dual pathways)
        mc_basic_params = sum(p.numel() for p in mc_basic.parameters())
        pytorch_basic_params = sum(p.numel() for p in pytorch_basic.parameters())
        
        # MC should have approximately 2x parameters (slight variations due to different channel counts possible)
        self.assertGreater(mc_basic_params, 1.5 * pytorch_basic_params)
        self.assertLess(mc_basic_params, 2.5 * pytorch_basic_params)
        
        # Test Bottleneck parameters
        mc_bottleneck = MCBottleneck(64, 64, 16, 16)
        pytorch_bottleneck = Bottleneck(64, 16)
        
        mc_bottleneck_params = sum(p.numel() for p in mc_bottleneck.parameters())
        pytorch_bottleneck_params = sum(p.numel() for p in pytorch_bottleneck.parameters())
        
        self.assertGreater(mc_bottleneck_params, 1.5 * pytorch_bottleneck_params)
        self.assertLess(mc_bottleneck_params, 2.5 * pytorch_bottleneck_params)
    
    def test_conv_helper_functions_vs_pytorch(self):
        """Test mc_conv3x3 and mc_conv1x1 helper functions match PyTorch patterns."""
        # Test mc_conv3x3
        mc_conv3x3_layer = mc_conv3x3(64, 64, 128, 128, stride=2)
        
        # Equivalent PyTorch conv
        pytorch_conv3x3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Test structure
        self.assertEqual(mc_conv3x3_layer.kernel_size, pytorch_conv3x3.kernel_size)
        self.assertEqual(mc_conv3x3_layer.stride, pytorch_conv3x3.stride)
        self.assertEqual(mc_conv3x3_layer.padding, pytorch_conv3x3.padding)
        self.assertEqual(mc_conv3x3_layer.color_bias, pytorch_conv3x3.bias)  # Both should be None
        
        # Test mc_conv1x1
        mc_conv1x1_layer = mc_conv1x1(64, 64, 256, 256, stride=1)
        
        # Equivalent PyTorch conv
        pytorch_conv1x1 = nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False)
        
        # Test structure
        self.assertEqual(mc_conv1x1_layer.kernel_size, pytorch_conv1x1.kernel_size)
        self.assertEqual(mc_conv1x1_layer.stride, pytorch_conv1x1.stride)
        self.assertEqual(mc_conv1x1_layer.padding, pytorch_conv1x1.padding)
        self.assertEqual(mc_conv1x1_layer.color_bias, pytorch_conv1x1.bias)  # Both should be None
    
    def test_activation_function_compatibility(self):
        """Test that activation functions behave identically."""
        # Both pathways should use same ReLU activation
        mc_block = MCBasicBlock(64, 64, 64, 64)
        
        # Test that ReLU is the expected type
        self.assertIsInstance(mc_block.relu, MCReLU)
        
        # Test ReLU behavior with dual inputs
        test_input = torch.randn(2, 64, 16, 16)
        test_negative = torch.randn(2, 64, 16, 16) - 1.0  # Ensure some negative values
        
        # Test dual pathway
        color_output, brightness_output = mc_block.relu(test_negative, test_negative)
        
        # All outputs should be non-negative
        self.assertTrue(torch.all(color_output >= 0))
        self.assertTrue(torch.all(brightness_output >= 0))
        
        # Should match F.relu behavior
        import torch.nn.functional as F
        expected_output = F.relu(test_negative)
        torch.testing.assert_close(color_output, expected_output)
        torch.testing.assert_close(brightness_output, expected_output)
        
        # Test single pathway methods
        color_only = mc_block.relu.forward_color(test_negative)
        brightness_only = mc_block.relu.forward_brightness(test_negative)
        
        self.assertTrue(torch.all(color_only >= 0))
        self.assertTrue(torch.all(brightness_only >= 0))
        torch.testing.assert_close(color_only, expected_output)
        torch.testing.assert_close(brightness_only, expected_output)


class TestBlocksIntegration(unittest.TestCase):
    """Integration tests for blocks with other multi-channel components."""
    
    def test_basicblock_with_sequential(self):
        """Test MCBasicBlock integration with MCSequential."""
        from models.multi_channel.container import MCSequential
        
        # Create a sequence of blocks (typical ResNet layer)
        layer = MCSequential(
            MCBasicBlock(64, 64, 64, 64),
            MCBasicBlock(64, 64, 64, 64),
        )
        
        color_input = torch.randn(2, 64, 16, 16)
        brightness_input = torch.randn(2, 64, 16, 16)
        
        color_out, brightness_out = layer(color_input, brightness_input)
        
        # Should maintain same shape through residual connections
        self.assertEqual(color_out.shape, color_input.shape)
        self.assertEqual(brightness_out.shape, brightness_input.shape)
    
    def test_bottleneck_with_sequential(self):
        """Test MCBottleneck integration with MCSequential.""" 
        from models.multi_channel.container import MCSequential
        
        # Create a sequence of bottleneck blocks
        layer = MCSequential(
            MCBottleneck(64, 64, 16, 16),  # 64 -> 64 (via 16 bottleneck)
            MCBottleneck(64, 64, 16, 16),  # 64 -> 64 (via 16 bottleneck)
        )
        
        color_input = torch.randn(2, 64, 16, 16)
        brightness_input = torch.randn(2, 64, 16, 16)
        
        color_out, brightness_out = layer(color_input, brightness_input)
        
        # Should maintain same shape through residual connections
        self.assertEqual(color_out.shape, color_input.shape)
        self.assertEqual(brightness_out.shape, brightness_input.shape)
    
    def test_mixed_block_types(self):
        """Test mixing different block types in sequential."""
        from models.multi_channel.container import MCSequential
        
        # Mix BasicBlock and Bottleneck (not typical but should work)
        layer = MCSequential(
            MCBasicBlock(64, 64, 64, 64),
            MCBottleneck(64, 64, 16, 16),  # 64 -> 64 via bottleneck
        )
        
        color_input = torch.randn(2, 64, 16, 16)
        brightness_input = torch.randn(2, 64, 16, 16)
        
        color_out, brightness_out = layer(color_input, brightness_input)
        
        self.assertEqual(color_out.shape, (2, 64, 16, 16))
        self.assertEqual(brightness_out.shape, (2, 64, 16, 16))
    
    def test_blocks_with_different_channel_progressions(self):
        """Test blocks handle the dual-stream channel transformation pattern."""
        # Simulate typical ResNet progression: 64 -> 128 with stride=2
        # Create proper downsample module for channel/spatial dimension changes
        downsample = MCSequential(
            MCConv2d(64, 64, 128, 128, kernel_size=1, stride=2, bias=False),
            MCBatchNorm2d(128, 128)
        )
        
        block_with_downsample = MCBasicBlock(
            color_inplanes=64,
            brightness_inplanes=64,
            color_planes=128,
            brightness_planes=128,
            stride=2,
            downsample=downsample
        )
        
        color_input = torch.randn(2, 64, 32, 32)
        brightness_input = torch.randn(2, 64, 32, 32)
        
        color_out, brightness_out = block_with_downsample(color_input, brightness_input)
        
        # Both pathways should follow same transformation: 64->128, spatial/2
        expected_shape = (2, 128, 16, 16)
        self.assertEqual(color_out.shape, expected_shape)
        self.assertEqual(brightness_out.shape, expected_shape)


class TestComprehensivePyTorchCompatibility(unittest.TestCase):
    """Comprehensive PyTorch compatibility tests ensuring our MC implementations match PyTorch exactly."""
    
    def setUp(self):
        """Set up test fixtures for comprehensive PyTorch compatibility testing."""
        self.batch_size = 2
        self.input_size = 32
        
        # Test different channel progressions like real ResNet
        # Stage 1: After initial conv - both pathways same channels
        self.color_input_64 = torch.randn(self.batch_size, 64, self.input_size, self.input_size)
        self.brightness_input_64 = torch.randn(self.batch_size, 64, self.input_size, self.input_size)
        
        # Stage 2: Mid-network - larger channels
        self.color_input_128 = torch.randn(self.batch_size, 128, self.input_size, self.input_size)
        self.brightness_input_128 = torch.randn(self.batch_size, 128, self.input_size, self.input_size)
        
        # Stage 3: Deeper network - even larger channels
        self.color_input_256 = torch.randn(self.batch_size, 256, self.input_size, self.input_size)
        self.brightness_input_256 = torch.randn(self.batch_size, 256, self.input_size, self.input_size)
        
        # PyTorch comparison input (same shape as our pathways)
        self.pytorch_input_64 = torch.randn(self.batch_size, 64, self.input_size, self.input_size)
        self.pytorch_input_128 = torch.randn(self.batch_size, 128, self.input_size, self.input_size)
        self.pytorch_input_256 = torch.randn(self.batch_size, 256, self.input_size, self.input_size)
    
    def test_mc_conv3x3_vs_pytorch_conv3x3(self):
        """Test mc_conv3x3 function matches PyTorch Conv2d exactly."""
        from models.multi_channel.blocks import mc_conv3x3
        
        # Test various channel transformations
        test_cases = [
            (64, 64, 64, 64),      # Same channels
            (64, 64, 128, 128),    # Channel expansion
            (128, 128, 64, 64),    # Channel reduction
        ]
        
        for color_in, brightness_in, color_out, brightness_out in test_cases:
            with self.subTest(channels=(color_in, brightness_in, color_out, brightness_out)):
                # Create MC conv3x3
                mc_conv = mc_conv3x3(color_in, brightness_in, color_out, brightness_out)
                
                # Create equivalent PyTorch conv
                pytorch_conv = nn.Conv2d(color_in, color_out, kernel_size=3, padding=1, bias=False)
                
                # Copy weights for fair comparison
                with torch.no_grad():
                    pytorch_conv.weight.copy_(mc_conv.color_weight)
                
                # Test color pathway
                test_input = torch.randn(2, color_in, 16, 16)
                mc_color_out = mc_conv.forward_color(test_input)
                pytorch_out = pytorch_conv(test_input)
                
                torch.testing.assert_close(mc_color_out, pytorch_out, atol=1e-6, rtol=1e-6,
                                         msg=f"mc_conv3x3 color pathway differs from PyTorch for {test_cases}")
                
                # Test brightness pathway with brightness weights
                with torch.no_grad():
                    pytorch_conv.weight.copy_(mc_conv.brightness_weight)
                
                test_input_brightness = torch.randn(2, brightness_in, 16, 16)
                mc_brightness_out = mc_conv.forward_brightness(test_input_brightness)
                pytorch_brightness_out = pytorch_conv(test_input_brightness)
                
                torch.testing.assert_close(mc_brightness_out, pytorch_brightness_out, atol=1e-6, rtol=1e-6,
                                         msg=f"mc_conv3x3 brightness pathway differs from PyTorch for {test_cases}")
    
    def test_mc_conv1x1_vs_pytorch_conv1x1(self):
        """Test mc_conv1x1 function matches PyTorch Conv2d exactly."""
        from models.multi_channel.blocks import mc_conv1x1
        
        # Test various channel transformations typical in ResNet
        test_cases = [
            (64, 64, 16, 16),      # Bottleneck reduction
            (16, 16, 64, 64),      # Bottleneck expansion  
            (128, 128, 256, 256),  # Large channel transformation
        ]
        
        for color_in, brightness_in, color_out, brightness_out in test_cases:
            with self.subTest(channels=(color_in, brightness_in, color_out, brightness_out)):
                # Create MC conv1x1
                mc_conv = mc_conv1x1(color_in, brightness_in, color_out, brightness_out)
                
                # Create equivalent PyTorch conv
                pytorch_conv = nn.Conv2d(color_in, color_out, kernel_size=1, bias=False)
                
                # Test color pathway
                with torch.no_grad():
                    pytorch_conv.weight.copy_(mc_conv.color_weight)
                
                test_input = torch.randn(2, color_in, 16, 16)
                mc_color_out = mc_conv.forward_color(test_input)
                pytorch_out = pytorch_conv(test_input)
                
                torch.testing.assert_close(mc_color_out, pytorch_out, atol=1e-6, rtol=1e-6,
                                         msg=f"mc_conv1x1 color pathway differs from PyTorch for {test_cases}")
                
                # Test brightness pathway
                with torch.no_grad():
                    pytorch_conv.weight.copy_(mc_conv.brightness_weight)
                
                test_input_brightness = torch.randn(2, brightness_in, 16, 16)
                mc_brightness_out = mc_conv.forward_brightness(test_input_brightness)
                pytorch_brightness_out = pytorch_conv(test_input_brightness)
                
                torch.testing.assert_close(mc_brightness_out, pytorch_brightness_out, atol=1e-6, rtol=1e-6,
                                         msg=f"mc_conv1x1 brightness pathway differs from PyTorch for {test_cases}")
    
    def test_mcbasicblock_channel_progression_vs_pytorch(self):
        """Test MCBasicBlock handles ResNet-style channel progressions like PyTorch."""
        from torchvision.models.resnet import BasicBlock
        from models.multi_channel.container import MCSequential
        
        # Test typical ResNet channel progressions
        progression_tests = [
            # (inplanes, planes, expected_outplanes, needs_downsample)
            (64, 64, 64, False),      # Layer 1: 64 -> 64 (no downsample needed)
            (64, 128, 128, True),     # Layer 2: 64 -> 128 (needs downsample)
            (128, 128, 128, False),   # Layer 2: 128 -> 128 (no downsample needed)
            (128, 256, 256, True),    # Layer 3: 128 -> 256 (needs downsample)
            (256, 256, 256, False),   # Layer 3: 256 -> 256 (no downsample needed)
            (256, 512, 512, True),    # Layer 4: 256 -> 512 (needs downsample)
        ]
        
        for inplanes, planes, expected_outplanes, needs_downsample in progression_tests:
            with self.subTest(progression=(inplanes, planes, expected_outplanes, needs_downsample)):
                # Create downsample if needed (like PyTorch does)
                downsample = None
                pytorch_downsample = None
                if needs_downsample:
                    downsample = MCSequential(
                        MCConv2d(inplanes, inplanes, expected_outplanes, expected_outplanes, 
                                kernel_size=1, stride=1, bias=False),
                        MCBatchNorm2d(expected_outplanes, expected_outplanes)
                    )
                    pytorch_downsample = nn.Sequential(
                        nn.Conv2d(inplanes, expected_outplanes, kernel_size=1, stride=1, bias=False),
                        nn.BatchNorm2d(expected_outplanes)
                    )
                
                # Create MC block
                mc_block = MCBasicBlock(
                    color_inplanes=inplanes,
                    brightness_inplanes=inplanes,
                    color_planes=planes,
                    brightness_planes=planes,
                    downsample=downsample
                )
                
                # Create PyTorch block
                pytorch_block = BasicBlock(inplanes=inplanes, planes=planes, downsample=pytorch_downsample)
                
                # Verify output channel calculation matches
                self.assertEqual(mc_block.color_outplanes, expected_outplanes)
                self.assertEqual(mc_block.brightness_outplanes, expected_outplanes)
                self.assertEqual(pytorch_block.expansion * planes, expected_outplanes)
                
                # Test forward pass shape consistency
                test_input = torch.randn(2, inplanes, 32, 32)
                
                mc_color_out = mc_block.forward_color(test_input)
                mc_brightness_out = mc_block.forward_brightness(test_input)
                pytorch_out = pytorch_block(test_input)
                
                # All outputs should have same shape
                self.assertEqual(mc_color_out.shape, pytorch_out.shape)
                self.assertEqual(mc_brightness_out.shape, pytorch_out.shape)
                self.assertEqual(mc_color_out.shape[1], expected_outplanes)
    
    def test_mcbottleneck_channel_progression_vs_pytorch(self):
        """Test MCBottleneck handles ResNet-style channel progressions like PyTorch."""
        from torchvision.models.resnet import Bottleneck
        from models.multi_channel.container import MCSequential
        
        # Test typical ResNet50+ channel progressions
        progression_tests = [
            # (inplanes, planes, expected_outplanes, needs_downsample) - expansion=4 for Bottleneck
            (64, 64, 256, True),       # Layer 1: 64 -> 64*4 = 256 (needs downsample)
            (256, 128, 512, True),     # Layer 2: 256 -> 128*4 = 512 (needs downsample)  
            (512, 128, 512, False),    # Layer 2: 512 -> 128*4 = 512 (no downsample needed)
            (512, 256, 1024, True),    # Layer 3: 512 -> 256*4 = 1024 (needs downsample)
            (1024, 256, 1024, False),  # Layer 3: 1024 -> 256*4 = 1024 (no downsample needed)
            (1024, 512, 2048, True),   # Layer 4: 1024 -> 512*4 = 2048 (needs downsample)
        ]
        
        for inplanes, planes, expected_outplanes, needs_downsample in progression_tests:
            with self.subTest(progression=(inplanes, planes, expected_outplanes, needs_downsample)):
                # Create downsample if needed (like PyTorch does)
                downsample = None
                pytorch_downsample = None
                if needs_downsample:
                    downsample = MCSequential(
                        MCConv2d(inplanes, inplanes, expected_outplanes, expected_outplanes, 
                                kernel_size=1, stride=1, bias=False),
                        MCBatchNorm2d(expected_outplanes, expected_outplanes)
                    )
                    pytorch_downsample = nn.Sequential(
                        nn.Conv2d(inplanes, expected_outplanes, kernel_size=1, stride=1, bias=False),
                        nn.BatchNorm2d(expected_outplanes)
                    )
                
                # Create MC block
                mc_block = MCBottleneck(
                    color_inplanes=inplanes,
                    brightness_inplanes=inplanes,
                    color_planes=planes,
                    brightness_planes=planes,
                    downsample=downsample
                )
                
                # Create PyTorch block
                pytorch_block = Bottleneck(inplanes=inplanes, planes=planes, downsample=pytorch_downsample)
                
                # Verify output channel calculation matches
                self.assertEqual(mc_block.color_outplanes, expected_outplanes)
                self.assertEqual(mc_block.brightness_outplanes, expected_outplanes)
                self.assertEqual(pytorch_block.expansion * planes, expected_outplanes)
                
                # Test forward pass shape consistency
                test_input = torch.randn(2, inplanes, 32, 32)
                
                mc_color_out = mc_block.forward_color(test_input)
                mc_brightness_out = mc_block.forward_brightness(test_input)
                pytorch_out = pytorch_block(test_input)
                
                # All outputs should have same shape
                self.assertEqual(mc_color_out.shape, pytorch_out.shape)
                self.assertEqual(mc_brightness_out.shape, pytorch_out.shape)
                self.assertEqual(mc_color_out.shape[1], expected_outplanes)
    
    def test_mcbasicblock_with_stride_vs_pytorch(self):
        """Test MCBasicBlock with stride matches PyTorch BasicBlock stride behavior."""
        from torchvision.models.resnet import BasicBlock
        from models.multi_channel.container import MCSequential
        
        # Test stride=2 for downsampling (typical in ResNet)
        stride_tests = [
            (1, 32, 32, False),  # stride=1: spatial size unchanged, no downsample needed for same channels
            (2, 16, 16, True),   # stride=2: spatial size halved, needs downsample for different channels
        ]
        
        for stride, expected_h, expected_w, needs_downsample in stride_tests:
            with self.subTest(stride=stride):
                # Create downsample if needed and channels change
                downsample = None
                pytorch_downsample = None
                inplanes, planes = 64, 128
                
                if needs_downsample or inplanes != planes:
                    downsample = MCSequential(
                        MCConv2d(inplanes, inplanes, planes, planes, 
                                kernel_size=1, stride=stride, bias=False),
                        MCBatchNorm2d(planes, planes)
                    )
                    pytorch_downsample = nn.Sequential(
                        nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm2d(planes)
                    )
                
                # Create MC block with stride
                mc_block = MCBasicBlock(
                    color_inplanes=inplanes,
                    brightness_inplanes=inplanes,
                    color_planes=planes,
                    brightness_planes=planes,
                    stride=stride,
                    downsample=downsample
                )
                
                # Create PyTorch block with same stride
                pytorch_block = BasicBlock(inplanes=inplanes, planes=planes, stride=stride, downsample=pytorch_downsample)
                
                # Test with 32x32 input
                test_input = torch.randn(2, inplanes, 32, 32)
                
                mc_color_out = mc_block.forward_color(test_input)
                mc_brightness_out = mc_block.forward_brightness(test_input)
                pytorch_out = pytorch_block(test_input)
                
                # Check spatial dimensions match PyTorch behavior
                expected_shape = (2, planes, expected_h, expected_w)
                self.assertEqual(mc_color_out.shape, expected_shape)
                self.assertEqual(mc_brightness_out.shape, expected_shape)
                self.assertEqual(pytorch_out.shape, expected_shape)
    
    def test_mcbottleneck_width_calculation_vs_pytorch(self):
        """Test MCBottleneck width calculation matches PyTorch exactly."""
        from torchvision.models.resnet import Bottleneck
        
        # Test different base_width and groups like PyTorch ResNeXt
        width_tests = [
            (64, 1, 64),    # Standard ResNet: base_width=64, groups=1
            (64, 1, 128),   # Standard ResNet with different planes
            (4, 32, 64),    # ResNeXt-style: base_width=4, groups=32
        ]
        
        for base_width, groups, planes in width_tests:
            with self.subTest(width_config=(base_width, groups, planes)):
                # Create MC block
                mc_block = MCBottleneck(
                    color_inplanes=256,
                    brightness_inplanes=256,
                    color_planes=planes,
                    brightness_planes=planes,
                    base_width=base_width,
                    groups=groups
                )
                
                # Create PyTorch block
                pytorch_block = Bottleneck(
                    inplanes=256,
                    planes=planes,
                    base_width=base_width,
                    groups=groups
                )
                
                # Calculate expected width (same as PyTorch formula)
                expected_width = int(planes * (base_width / 64.0)) * groups
                
                # Test that our width calculation matches PyTorch
                # Check conv2 (the 3x3 conv) has correct input/output channels
                self.assertEqual(mc_block.conv2.color_in_channels, expected_width)
                self.assertEqual(mc_block.conv2.color_out_channels, expected_width)
                self.assertEqual(mc_block.conv2.brightness_in_channels, expected_width)
                self.assertEqual(mc_block.conv2.brightness_out_channels, expected_width)
                
                # PyTorch conv2 should have same width
                self.assertEqual(pytorch_block.conv2.in_channels, expected_width)
                self.assertEqual(pytorch_block.conv2.out_channels, expected_width)
    
    def test_dual_stream_identical_behavior(self):
        """Test that both streams behave identically when given same input and weights."""
        # This ensures our dual-stream is truly "PyTorch behavior × 2"
        
        # Test BasicBlock
        mc_basic = MCBasicBlock(64, 64, 64, 64)
        
        # Make both streams have identical weights
        with torch.no_grad():
            mc_basic.conv1.brightness_weight.copy_(mc_basic.conv1.color_weight)
            mc_basic.conv2.brightness_weight.copy_(mc_basic.conv2.color_weight)
            mc_basic.bn1.brightness_weight.copy_(mc_basic.bn1.color_weight)
            mc_basic.bn1.brightness_bias.copy_(mc_basic.bn1.color_bias)
            mc_basic.bn2.brightness_weight.copy_(mc_basic.bn2.color_weight)
            mc_basic.bn2.brightness_bias.copy_(mc_basic.bn2.color_bias)
        
        # Use same input for both streams
        test_input = torch.randn(2, 64, 32, 32)
        
        # Both streams should produce identical output
        color_out, brightness_out = mc_basic(test_input, test_input)
        torch.testing.assert_close(color_out, brightness_out, atol=1e-6, rtol=1e-6,
                                 msg="Identical streams with identical weights should produce identical outputs")
        
        # Test Bottleneck
        mc_bottleneck = MCBottleneck(64, 64, 16, 16)
        
        # Make both streams have identical weights
        with torch.no_grad():
            mc_bottleneck.conv1.brightness_weight.copy_(mc_bottleneck.conv1.color_weight)
            mc_bottleneck.conv2.brightness_weight.copy_(mc_bottleneck.conv2.color_weight)
            mc_bottleneck.conv3.brightness_weight.copy_(mc_bottleneck.conv3.color_weight)
            mc_bottleneck.bn1.brightness_weight.copy_(mc_bottleneck.bn1.color_weight)
            mc_bottleneck.bn1.brightness_bias.copy_(mc_bottleneck.bn1.color_bias)
            mc_bottleneck.bn2.brightness_weight.copy_(mc_bottleneck.bn2.color_weight)
            mc_bottleneck.bn2.brightness_bias.copy_(mc_bottleneck.bn2.color_bias)
            mc_bottleneck.bn3.brightness_weight.copy_(mc_bottleneck.bn3.color_weight)
            mc_bottleneck.bn3.brightness_bias.copy_(mc_bottleneck.bn3.color_bias)
        
        color_out, brightness_out = mc_bottleneck(test_input, test_input)
        torch.testing.assert_close(color_out, brightness_out, atol=1e-6, rtol=1e-6,
                                 msg="Identical bottleneck streams should produce identical outputs")
    
    def test_gradient_flow_identical_to_pytorch(self):
        """Test that gradients flow identically to PyTorch blocks."""
        from torchvision.models.resnet import BasicBlock, Bottleneck
        
        # Test BasicBlock gradient flow
        mc_basic = MCBasicBlock(64, 64, 64, 64)
        pytorch_basic = BasicBlock(64, 64)
        
        # Copy weights for fair comparison
        with torch.no_grad():
            pytorch_basic.conv1.weight.copy_(mc_basic.conv1.color_weight)
            pytorch_basic.conv2.weight.copy_(mc_basic.conv2.color_weight)
            pytorch_basic.bn1.weight.copy_(mc_basic.bn1.color_weight)
            pytorch_basic.bn1.bias.copy_(mc_basic.bn1.color_bias)
            pytorch_basic.bn2.weight.copy_(mc_basic.bn2.color_weight)
            pytorch_basic.bn2.bias.copy_(mc_basic.bn2.color_bias)
        
        # Same input with gradients
        test_input = torch.randn(1, 64, 32, 32, requires_grad=True)
        pytorch_input = test_input.clone().detach().requires_grad_(True)
        
        # Forward pass
        mc_color_out = mc_basic.forward_color(test_input)
        pytorch_out = pytorch_basic(pytorch_input)
        
        # Backward pass with same loss
        mc_loss = mc_color_out.sum()
        pytorch_loss = pytorch_out.sum()
        
        mc_loss.backward()
        pytorch_loss.backward()
        
        # Check that input gradients are identical
        torch.testing.assert_close(test_input.grad, pytorch_input.grad, atol=1e-5, rtol=1e-5,
                                 msg="Input gradients should be identical to PyTorch")
        
        # Check that parameter gradients are identical
        torch.testing.assert_close(mc_basic.conv1.color_weight.grad, pytorch_basic.conv1.weight.grad, 
                                 atol=1e-5, rtol=1e-5, msg="Conv1 weight gradients should be identical")
        torch.testing.assert_close(mc_basic.conv2.color_weight.grad, pytorch_basic.conv2.weight.grad, 
                                 atol=1e-5, rtol=1e-5, msg="Conv2 weight gradients should be identical")
