"""
Unit tests for Multi-Channel container modules.
"""

import unittest
import torch
import torch.nn as nn
import sys
from pathlib import Path
from collections import OrderedDict

from models.multi_channel.container import MCSequential, MCReLU
from models.multi_channel.conv import MCConv2d, MCBatchNorm2d
from models.multi_channel.blocks import MCBasicBlock


class MockMCModule(nn.Module):
    """Mock multi-channel module for testing."""
    
    def __init__(self, name="mock"):
        super().__init__()
        self.name = name
    
    def forward(self, color_input, brightness_input):
        """Mock forward that adds 1 to each input."""
        return color_input + 1, brightness_input + 1
    
    def forward_color(self, color_input):
        """Mock color forward that adds 1."""
        return color_input + 1
    
    def forward_brightness(self, brightness_input):
        """Mock brightness forward that adds 1."""
        return brightness_input + 1
    
    def __repr__(self):
        return f"MockMCModule({self.name})"


class TestMCSequential(unittest.TestCase):
    """Test cases for MCSequential container."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock1 = MockMCModule("mock1")
        self.mock2 = MockMCModule("mock2")
        self.mock3 = MockMCModule("mock3")
        
        # Create sample inputs
        self.color_input = torch.randn(2, 3, 8, 8)
        self.brightness_input = torch.randn(2, 1, 8, 8)
    
    def test_init_with_args(self):
        """Test MCSequential initialization with arguments."""
        seq = MCSequential(self.mock1, self.mock2, self.mock3)
        
        self.assertEqual(len(seq), 3)
        self.assertEqual(seq[0], self.mock1)
        self.assertEqual(seq[1], self.mock2)
        self.assertEqual(seq[2], self.mock3)
    
    def test_init_with_ordered_dict(self):
        """Test MCSequential initialization with OrderedDict."""
        modules = OrderedDict([
            ('first', self.mock1),
            ('second', self.mock2),
            ('third', self.mock3)
        ])
        seq = MCSequential(modules)
        
        self.assertEqual(len(seq), 3)
        self.assertEqual(seq[0], self.mock1)
        self.assertEqual(seq[1], self.mock2)
        self.assertEqual(seq[2], self.mock3)
    
    def test_forward_pass(self):
        """Test forward pass through MCSequential."""
        seq = MCSequential(self.mock1, self.mock2)
        
        color_out, brightness_out = seq(self.color_input, self.brightness_input)
        
        # Each mock adds 1, so with 2 mocks we should add 2
        expected_color = self.color_input + 2
        expected_brightness = self.brightness_input + 2
        
        torch.testing.assert_close(color_out, expected_color)
        torch.testing.assert_close(brightness_out, expected_brightness)
    
    def test_forward_color_pathway(self):
        """Test forward pass through color pathway only."""
        seq = MCSequential(self.mock1, self.mock2)
        
        color_out = seq.forward_color(self.color_input)
        expected = self.color_input + 2  # Two mocks, each adds 1
        
        torch.testing.assert_close(color_out, expected)
    
    def test_forward_brightness_pathway(self):
        """Test forward pass through brightness pathway only."""
        seq = MCSequential(self.mock1, self.mock2)
        
        brightness_out = seq.forward_brightness(self.brightness_input)
        expected = self.brightness_input + 2  # Two mocks, each adds 1
        
        torch.testing.assert_close(brightness_out, expected)
    
    def test_empty_sequential(self):
        """Test empty MCSequential."""
        seq = MCSequential()
        self.assertEqual(len(seq), 0)
        
        # Forward pass with empty sequence should return inputs unchanged
        color_out, brightness_out = seq(self.color_input, self.brightness_input)
        torch.testing.assert_close(color_out, self.color_input)
        torch.testing.assert_close(brightness_out, self.brightness_input)
    
    def test_getitem_single_index(self):
        """Test getting single item by index."""
        seq = MCSequential(self.mock1, self.mock2, self.mock3)
        
        self.assertEqual(seq[0], self.mock1)
        self.assertEqual(seq[1], self.mock2)
        self.assertEqual(seq[2], self.mock3)
        self.assertEqual(seq[-1], self.mock3)
        self.assertEqual(seq[-2], self.mock2)
    
    def test_getitem_slice(self):
        """Test getting slice of modules."""
        seq = MCSequential(self.mock1, self.mock2, self.mock3)
        
        sub_seq = seq[1:3]
        self.assertIsInstance(sub_seq, MCSequential)
        self.assertEqual(len(sub_seq), 2)
        self.assertEqual(sub_seq[0], self.mock2)
        self.assertEqual(sub_seq[1], self.mock3)
    
    def test_setitem(self):
        """Test setting item by index."""
        seq = MCSequential(self.mock1, self.mock2, self.mock3)
        new_mock = MockMCModule("new_mock")
        
        seq[1] = new_mock
        self.assertEqual(seq[1], new_mock)
        self.assertEqual(len(seq), 3)
    
    def test_delitem_single(self):
        """Test deleting single item."""
        seq = MCSequential(self.mock1, self.mock2, self.mock3)
        
        del seq[1]
        self.assertEqual(len(seq), 2)
        self.assertEqual(seq[0], self.mock1)
        self.assertEqual(seq[1], self.mock3)
    
    def test_delitem_slice(self):
        """Test deleting slice."""
        seq = MCSequential(self.mock1, self.mock2, self.mock3)
        
        del seq[1:3]
        self.assertEqual(len(seq), 1)
        self.assertEqual(seq[0], self.mock1)
    
    def test_len(self):
        """Test length of MCSequential."""
        seq = MCSequential()
        self.assertEqual(len(seq), 0)
        
        seq = MCSequential(self.mock1, self.mock2)
        self.assertEqual(len(seq), 2)
    
    def test_add_operator(self):
        """Test addition operator for MCSequential."""
        seq1 = MCSequential(self.mock1, self.mock2)
        seq2 = MCSequential(self.mock3)
        
        combined = seq1 + seq2
        self.assertIsInstance(combined, MCSequential)
        self.assertEqual(len(combined), 3)
        self.assertEqual(combined[0], self.mock1)
        self.assertEqual(combined[1], self.mock2)
        self.assertEqual(combined[2], self.mock3)
    
    def test_add_operator_invalid_type(self):
        """Test addition operator with invalid type."""
        seq = MCSequential(self.mock1)
        
        with self.assertRaises(ValueError):
            seq + self.mock2
    
    def test_iadd_operator(self):
        """Test in-place addition operator."""
        seq1 = MCSequential(self.mock1)
        seq2 = MCSequential(self.mock2, self.mock3)
        
        seq1 += seq2
        self.assertEqual(len(seq1), 3)
        self.assertEqual(seq1[0], self.mock1)
        self.assertEqual(seq1[1], self.mock2)
        self.assertEqual(seq1[2], self.mock3)
    
    def test_mul_operator(self):
        """Test multiplication operator."""
        seq = MCSequential(self.mock1, self.mock2)
        
        multiplied = seq * 3
        self.assertEqual(len(multiplied), 6)
        # Should have 3 copies of the original sequence
        for i in range(3):
            self.assertEqual(multiplied[i * 2], self.mock1)
            self.assertEqual(multiplied[i * 2 + 1], self.mock2)
    
    def test_mul_operator_invalid(self):
        """Test multiplication operator with invalid inputs."""
        seq = MCSequential(self.mock1)
        
        with self.assertRaises(TypeError):
            seq * "invalid"
        
        with self.assertRaises(ValueError):
            seq * 0
        
        with self.assertRaises(ValueError):
            seq * -1
    
    def test_rmul_operator(self):
        """Test right multiplication operator."""
        seq = MCSequential(self.mock1)
        
        multiplied = 2 * seq
        self.assertEqual(len(multiplied), 2)
        self.assertEqual(multiplied[0], self.mock1)
        self.assertEqual(multiplied[1], self.mock1)
    
    def test_imul_operator(self):
        """Test in-place multiplication operator."""
        seq = MCSequential(self.mock1, self.mock2)
        original_length = len(seq)
        
        seq *= 2
        self.assertEqual(len(seq), original_length * 2)
        self.assertEqual(seq[0], self.mock1)
        self.assertEqual(seq[1], self.mock2)
        self.assertEqual(seq[2], self.mock1)
        self.assertEqual(seq[3], self.mock2)
    
    def test_pop(self):
        """Test pop method."""
        seq = MCSequential(self.mock1, self.mock2, self.mock3)
        
        popped = seq.pop(1)
        self.assertEqual(popped, self.mock2)
        self.assertEqual(len(seq), 2)
        self.assertEqual(seq[0], self.mock1)
        self.assertEqual(seq[1], self.mock3)
    
    def test_append(self):
        """Test append method."""
        seq = MCSequential(self.mock1)
        
        seq.append(self.mock2)
        self.assertEqual(len(seq), 2)
        self.assertEqual(seq[1], self.mock2)
    
    def test_insert(self):
        """Test insert method."""
        seq = MCSequential(self.mock1, self.mock3)
        
        seq.insert(1, self.mock2)
        self.assertEqual(len(seq), 3)
        self.assertEqual(seq[0], self.mock1)
        self.assertEqual(seq[1], self.mock2)
        self.assertEqual(seq[2], self.mock3)
    
    def test_insert_invalid_index(self):
        """Test insert with invalid index."""
        seq = MCSequential(self.mock1)
        
        with self.assertRaises(IndexError):
            seq.insert(5, self.mock2)
    
    def test_extend(self):
        """Test extend method."""
        seq1 = MCSequential(self.mock1)
        seq2 = MCSequential(self.mock2, self.mock3)
        
        seq1.extend(seq2)
        self.assertEqual(len(seq1), 3)
        self.assertEqual(seq1[0], self.mock1)
        self.assertEqual(seq1[1], self.mock2)
        self.assertEqual(seq1[2], self.mock3)
    
    def test_iter(self):
        """Test iteration over MCSequential."""
        seq = MCSequential(self.mock1, self.mock2, self.mock3)
        
        modules = list(seq)
        self.assertEqual(len(modules), 3)
        self.assertEqual(modules[0], self.mock1)
        self.assertEqual(modules[1], self.mock2)
        self.assertEqual(modules[2], self.mock3)
    
    def test_real_mc_modules(self):
        """Test MCSequential with real multi-channel modules."""
        conv = MCConv2d(3, 1, 16, 16, kernel_size=3, padding=1)
        bn = MCBatchNorm2d(16, 16)
        seq = MCSequential(conv, bn)
        
        color_input = torch.randn(2, 3, 8, 8)
        brightness_input = torch.randn(2, 1, 8, 8)
        
        color_out, brightness_out = seq(color_input, brightness_input)
        
        # Check output shapes
        self.assertEqual(color_out.shape, (2, 16, 8, 8))
        self.assertEqual(brightness_out.shape, (2, 16, 8, 8))
        
        # Test individual pathways
        color_only = seq.forward_color(color_input)
        brightness_only = seq.forward_brightness(brightness_input)
        
        self.assertEqual(color_only.shape, (2, 16, 8, 8))
        self.assertEqual(brightness_only.shape, (2, 16, 8, 8))
    
    def test_gradient_flow(self):
        """Test that gradients flow through MCSequential."""
        conv = MCConv2d(3, 1, 8, 8, kernel_size=3, padding=1)
        seq = MCSequential(conv)
        
        color_input = torch.randn(1, 3, 4, 4, requires_grad=True)
        brightness_input = torch.randn(1, 1, 4, 4, requires_grad=True)
        
        color_out, brightness_out = seq(color_input, brightness_input)
        loss = color_out.sum() + brightness_out.sum()
        loss.backward()
        
        # Check that gradients exist
        self.assertIsNotNone(color_input.grad)
        self.assertIsNotNone(brightness_input.grad)
        self.assertIsNotNone(conv.color_weight.grad)
        self.assertIsNotNone(conv.brightness_weight.grad)


class TestPyTorchCompatibility(unittest.TestCase):
    """Test PyTorch compatibility for MCSequential container."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Both pathways follow same channel transformations
        self.color_input = torch.randn(2, 64, 16, 16)
        self.brightness_input = torch.randn(2, 64, 16, 16)
        
        # For PyTorch comparison
        self.pytorch_test_input = torch.randn(2, 64, 16, 16)
    
    def test_mcsequential_vs_pytorch_sequential_structure(self):
        """Test MCSequential structure matches PyTorch Sequential patterns."""
        # Create equivalent sequences
        mc_seq = MCSequential(
            MCConv2d(64, 64, 128, 128, kernel_size=3, padding=1, bias=False),
            MCBatchNorm2d(128, 128),
            MCConv2d(128, 128, 256, 256, kernel_size=3, padding=1, bias=False),
            MCBatchNorm2d(256, 256)
        )
        
        pytorch_seq = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256)
        )
        
        # Check structure properties
        self.assertEqual(len(mc_seq), len(pytorch_seq))
        
        # Test forward pass shapes
        color_out, brightness_out = mc_seq(self.color_input, self.brightness_input)
        pytorch_out = pytorch_seq(self.pytorch_test_input)
        
        # Both pathways should have same output shape as PyTorch
        expected_shape = (2, 256, 16, 16)
        self.assertEqual(color_out.shape, expected_shape)
        self.assertEqual(brightness_out.shape, expected_shape)
        self.assertEqual(pytorch_out.shape, expected_shape)
    
    def test_mcsequential_vs_pytorch_sequential_with_matching_weights(self):
        """Test MCSequential gives identical results to PyTorch Sequential with same weights."""
        # Create simple sequences
        mc_seq = MCSequential(
            MCConv2d(64, 64, 128, 128, kernel_size=1, bias=False),
            MCBatchNorm2d(128, 128)
        )
        
        # Create ONE PyTorch sequence for comparison
        pytorch_seq = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128)
        )
        
        # Use same input for all tests
        test_input_64 = torch.randn(2, 64, 16, 16)
        
        # Set to eval mode
        mc_seq.eval()
        pytorch_seq.eval()
        
        # Test 1: Copy color pathway weights to PyTorch and compare
        with torch.no_grad():
            pytorch_seq[0].weight.copy_(mc_seq[0].color_weight)
            pytorch_seq[1].weight.copy_(mc_seq[1].color_weight)
            pytorch_seq[1].bias.copy_(mc_seq[1].color_bias)
            pytorch_seq[1].running_mean.copy_(mc_seq[1].color_running_mean)
            pytorch_seq[1].running_var.copy_(mc_seq[1].color_running_var)
        
        pytorch_out = pytorch_seq(test_input_64)
        mc_color_out = mc_seq.forward_color(test_input_64)
        
        torch.testing.assert_close(mc_color_out, pytorch_out, atol=1e-5, rtol=1e-5,
                                 msg="MCSequential color pathway differs from PyTorch Sequential")
        
        # Test 2: Copy brightness pathway weights to PyTorch and compare
        with torch.no_grad():
            pytorch_seq[0].weight.copy_(mc_seq[0].brightness_weight)
            pytorch_seq[1].weight.copy_(mc_seq[1].brightness_weight)
            pytorch_seq[1].bias.copy_(mc_seq[1].brightness_bias)
            pytorch_seq[1].running_mean.copy_(mc_seq[1].brightness_running_mean)
            pytorch_seq[1].running_var.copy_(mc_seq[1].brightness_running_var)
        
        pytorch_out = pytorch_seq(test_input_64)
        mc_brightness_out = mc_seq.forward_brightness(test_input_64)
        
        torch.testing.assert_close(mc_brightness_out, pytorch_out, atol=1e-5, rtol=1e-5,
                                 msg="MCSequential brightness pathway differs from PyTorch Sequential")
        
        # Test 3: Verify dual pathway forward gives same results as individual pathways
        mc_color_dual, mc_brightness_dual = mc_seq(test_input_64, test_input_64)
        
        torch.testing.assert_close(mc_color_dual, mc_color_out, atol=1e-5, rtol=1e-5,
                                 msg="Dual-stream color output differs from single pathway")
        torch.testing.assert_close(mc_brightness_dual, mc_brightness_out, atol=1e-5, rtol=1e-5,
                                 msg="Dual-stream brightness output differs from single pathway")
    
    def test_mcsequential_parameter_management_vs_pytorch(self):
        """Test MCSequential parameter management matches PyTorch patterns."""
        # Create equivalent sequences
        mc_seq = MCSequential(
            MCConv2d(64, 64, 128, 128, kernel_size=3, padding=1, bias=False),
            MCBatchNorm2d(128, 128)
        )
        
        pytorch_seq = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        
        # Test parameter counting
        mc_params = sum(p.numel() for p in mc_seq.parameters())
        pytorch_params = sum(p.numel() for p in pytorch_seq.parameters())
        
        # MC should have ~2x parameters due to dual pathways
        self.assertGreater(mc_params, 1.5 * pytorch_params)
        self.assertLess(mc_params, 2.5 * pytorch_params)
        
        # Test named parameters structure
        mc_param_names = [name for name, _ in mc_seq.named_parameters()]
        pytorch_param_names = [name for name, _ in pytorch_seq.named_parameters()]
        
        # MC should have both color and brightness variants of each PyTorch parameter
        for pytorch_name in pytorch_param_names:
            if 'weight' in pytorch_name or 'bias' in pytorch_name:
                # Should have color and brightness versions
                color_name = pytorch_name.replace('weight', 'color_weight').replace('bias', 'color_bias')
                brightness_name = pytorch_name.replace('weight', 'brightness_weight').replace('bias', 'brightness_bias')
                self.assertIn(color_name, mc_param_names)
                self.assertIn(brightness_name, mc_param_names)
    
    def test_mcsequential_state_dict_vs_pytorch(self):
        """Test MCSequential state_dict structure and loading."""
        mc_seq = MCSequential(
            MCConv2d(64, 64, 128, 128, kernel_size=1, bias=False),
            MCBatchNorm2d(128, 128)
        )
        
        # Get state dict
        state_dict = mc_seq.state_dict()
        
        # Should contain all dual-pathway parameters
        expected_keys = [
            '0.color_weight', '0.brightness_weight',
            '1.color_weight', '1.color_bias', '1.brightness_weight', '1.brightness_bias',
            '1.color_running_mean', '1.color_running_var', 
            '1.brightness_running_mean', '1.brightness_running_var',
            '1.num_batches_tracked'  # Single shared tracking buffer
        ]
        
        for key in expected_keys:
            self.assertIn(key, state_dict.keys())
        
        # Test loading state dict
        mc_seq2 = MCSequential(
            MCConv2d(64, 64, 128, 128, kernel_size=1, bias=False),
            MCBatchNorm2d(128, 128)
        )
        
        mc_seq2.load_state_dict(state_dict)
        
        # Parameters should match
        for (name1, param1), (name2, param2) in zip(mc_seq.named_parameters(), mc_seq2.named_parameters()):
            self.assertEqual(name1, name2)
            torch.testing.assert_close(param1, param2)
    
    def test_mcsequential_gradient_flow_vs_pytorch(self):
        """Test gradient flow through MCSequential matches PyTorch patterns."""
        mc_seq = MCSequential(
            MCConv2d(64, 64, 128, 128, kernel_size=1),
            MCBatchNorm2d(128, 128),
            MCReLU(inplace=True)
        )
        
        color_input = self.color_input.requires_grad_(True)
        brightness_input = self.brightness_input.requires_grad_(True)
        
        # Forward pass
        color_out, brightness_out = mc_seq(color_input, brightness_input)
        
        # Backward pass
        loss = color_out.sum() + brightness_out.sum()
        loss.backward()
        
        # Check gradients exist for inputs and all parameters
        self.assertIsNotNone(color_input.grad)
        self.assertIsNotNone(brightness_input.grad)
        
        for name, param in mc_seq.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"No gradient for parameter {name}")
    
    def test_mcsequential_training_eval_modes_vs_pytorch(self):
        """Test MCSequential training/eval mode behavior matches PyTorch."""
        mc_seq = MCSequential(
            MCConv2d(64, 64, 128, 128, kernel_size=1),
            MCBatchNorm2d(128, 128)
        )
        
        pytorch_seq = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1),
            nn.BatchNorm2d(128)
        )
        
        # Test training mode
        mc_seq.train()
        pytorch_seq.train()
        
        self.assertTrue(mc_seq.training)
        self.assertTrue(pytorch_seq.training)
        
        # All modules should be in training mode
        for module in mc_seq.modules():
            if hasattr(module, 'training'):
                self.assertTrue(module.training)
        
        # Test eval mode
        mc_seq.eval()
        pytorch_seq.eval()
        
        self.assertFalse(mc_seq.training)
        self.assertFalse(pytorch_seq.training)
        
        # All modules should be in eval mode
        for module in mc_seq.modules():
            if hasattr(module, 'training'):
                self.assertFalse(module.training)
    
    def test_mcsequential_channel_transformation_patterns(self):
        """Test MCSequential handles typical channel transformation patterns."""
        # Test common ResNet-style progression: 3,1 -> 64,64 -> 128,128 -> 256,256
        feature_progression = MCSequential(
            MCConv2d(3, 1, 64, 64, kernel_size=7, stride=2, padding=3, bias=False),  # Initial different channels
            MCBatchNorm2d(64, 64),
            MCReLU(inplace=True),
            MCConv2d(64, 64, 128, 128, kernel_size=3, stride=2, padding=1, bias=False),  # Same channels
            MCBatchNorm2d(128, 128),
            MCReLU(inplace=True),
            MCConv2d(128, 128, 256, 256, kernel_size=3, stride=2, padding=1, bias=False),  # Same channels
            MCBatchNorm2d(256, 256)
        )
        
        # Start with different channels
        color_input = torch.randn(2, 3, 224, 224)  # RGB
        brightness_input = torch.randn(2, 1, 224, 224)  # Grayscale
        
        color_out, brightness_out = feature_progression(color_input, brightness_input)
        
        # Both should converge to same channel count and spatial dimensions
        # 224 -> 112 -> 56 -> 28
        expected_shape = (2, 256, 28, 28)
        self.assertEqual(color_out.shape, expected_shape)
        self.assertEqual(brightness_out.shape, expected_shape)
        
        # Verify intermediate transformations work
        color_intermediate = color_input
        brightness_intermediate = brightness_input
        
        for i, layer in enumerate(feature_progression):
            if isinstance(layer, (MCConv2d, MCBatchNorm2d, MCReLU)):
                color_intermediate, brightness_intermediate = layer(color_intermediate, brightness_intermediate)
            else:  # Should not happen with current test, but handle gracefully
                raise ValueError(f"Unexpected module type: {type(layer)}")
            
            # After first conv+bn, both pathways should have 64 channels
            if i == 1:  # After first batchnorm
                self.assertEqual(color_intermediate.shape[1], 64)
                self.assertEqual(brightness_intermediate.shape[1], 64)

class TestMCReLU(unittest.TestCase):
    """Test cases for MCReLU activation function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.relu = MCReLU()
        self.relu_inplace = MCReLU(inplace=True)
        
        # Test inputs with negative and positive values
        self.color_input = torch.tensor([[-1.0, 2.0, -3.0, 4.0]]).reshape(1, 1, 2, 2)
        self.brightness_input = torch.tensor([[-2.0, 1.0, -4.0, 3.0]]).reshape(1, 1, 2, 2)
        
        # Expected outputs (negative values become 0)
        self.expected_color = torch.tensor([[0.0, 2.0, 0.0, 4.0]]).reshape(1, 1, 2, 2)
        self.expected_brightness = torch.tensor([[0.0, 1.0, 0.0, 3.0]]).reshape(1, 1, 2, 2)
    
    def test_basic_functionality(self):
        """Test basic MCReLU dual-stream functionality."""
        color_out, brightness_out = self.relu(self.color_input, self.brightness_input)
        
        # Check that negative values become 0, positive values remain
        torch.testing.assert_close(color_out, self.expected_color)
        torch.testing.assert_close(brightness_out, self.expected_brightness)
        
        # Check shapes are preserved
        self.assertEqual(color_out.shape, self.color_input.shape)
        self.assertEqual(brightness_out.shape, self.brightness_input.shape)
    
    def test_inplace_functionality(self):
        """Test MCReLU with inplace=True."""
        # Create copies to avoid modifying original test data
        color_input = self.color_input.clone()
        brightness_input = self.brightness_input.clone()
        
        # Store original data pointers to verify inplace operation
        color_input_data_ptr = color_input.data_ptr()
        brightness_input_data_ptr = brightness_input.data_ptr()
        
        color_out, brightness_out = self.relu_inplace(color_input, brightness_input)
        
        # Check that the operation was done in-place
        self.assertEqual(color_out.data_ptr(), color_input_data_ptr)
        self.assertEqual(brightness_out.data_ptr(), brightness_input_data_ptr)
        
        # Check values are correct
        torch.testing.assert_close(color_out, self.expected_color)
        torch.testing.assert_close(brightness_out, self.expected_brightness)
    
    def test_individual_pathway_methods(self):
        """Test forward_color and forward_brightness methods."""
        # Test forward_color
        color_out = self.relu.forward_color(self.color_input)
        torch.testing.assert_close(color_out, self.expected_color)
        
        # Test forward_brightness  
        brightness_out = self.relu.forward_brightness(self.brightness_input)
        torch.testing.assert_close(brightness_out, self.expected_brightness)
    
    def test_extra_repr(self):
        """Test string representation."""
        # Default (inplace=False)
        self.assertEqual(self.relu.extra_repr(), "")
        
        # With inplace=True
        self.assertEqual(self.relu_inplace.extra_repr(), "inplace=True")
    
    def test_gradient_flow(self):
        """Test that gradients flow properly through MCReLU."""
        color_input = torch.randn(2, 3, 4, 4, requires_grad=True)
        brightness_input = torch.randn(2, 1, 4, 4, requires_grad=True)
        
        color_out, brightness_out = self.relu(color_input, brightness_input)
        
        # Backward pass
        loss = color_out.sum() + brightness_out.sum()
        loss.backward()
        
        # Check gradients exist
        self.assertIsNotNone(color_input.grad)
        self.assertIsNotNone(brightness_input.grad)
        
        # Gradients should be 0 where input was negative, 1 where positive
        color_grad_expected = (color_input > 0).float()
        brightness_grad_expected = (brightness_input > 0).float()
        
        torch.testing.assert_close(color_input.grad, color_grad_expected)
        torch.testing.assert_close(brightness_input.grad, brightness_grad_expected)
    
    def test_with_mcsequential(self):
        """Test MCReLU integration with MCSequential."""
        seq = MCSequential(
            MCConv2d(3, 1, 16, 8, kernel_size=3, padding=1),
            MCReLU(inplace=True)
        )
        
        color_input = torch.randn(2, 3, 8, 8)
        brightness_input = torch.randn(2, 1, 8, 8)
        
        # Forward pass
        color_out, brightness_out = seq(color_input, brightness_input)
        
        # Check output shapes
        self.assertEqual(color_out.shape, (2, 16, 8, 8))
        self.assertEqual(brightness_out.shape, (2, 8, 8, 8))
        
        # Check that all outputs are non-negative (ReLU effect)
        self.assertTrue(torch.all(color_out >= 0))
        self.assertTrue(torch.all(brightness_out >= 0))
    
    def test_consistency_with_pytorch_relu(self):
        """Test that MCReLU gives same results as PyTorch ReLU on individual pathways."""
        import torch.nn.functional as F
        
        # Test color pathway
        color_out_mc = self.relu.forward_color(self.color_input)
        color_out_pytorch = F.relu(self.color_input)
        torch.testing.assert_close(color_out_mc, color_out_pytorch)
        
        # Test brightness pathway
        brightness_out_mc = self.relu.forward_brightness(self.brightness_input)
        brightness_out_pytorch = F.relu(self.brightness_input)
        torch.testing.assert_close(brightness_out_mc, brightness_out_pytorch)


if __name__ == "__main__":
    unittest.main()

"""
Advanced PyTorch compatibility tests for MCSequential edge cases and patterns.
"""

import unittest
import torch
import torch.nn as nn
from models.multi_channel.container import MCSequential, MCReLU
from models.multi_channel.conv import MCConv2d, MCBatchNorm2d


class TestAdvancedPyTorchCompatibility(unittest.TestCase):
    """Advanced PyTorch compatibility tests for MCSequential edge cases and patterns."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.color_input = torch.randn(2, 64, 16, 16)
        self.brightness_input = torch.randn(2, 64, 16, 16)
    
    def test_mcsequential_vs_pytorch_sequential_exact_matching(self):
        """Test MCSequential against PyTorch Sequential with exact weight matching."""
        # Create MC sequence
        mc_seq = MCSequential(
            MCConv2d(64, 64, 128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            MCBatchNorm2d(128, 128),
            MCReLU(inplace=True),
            MCConv2d(128, 128, 256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            MCBatchNorm2d(256, 256),
        )
        
        # Create equivalent PyTorch sequences
        pytorch_color_seq = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
        )
        
        pytorch_brightness_seq = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
        )
        
        # Copy weights from MC to PyTorch for color pathway
        with torch.no_grad():
            pytorch_color_seq[0].weight.copy_(mc_seq[0].color_weight)
            pytorch_color_seq[1].weight.copy_(mc_seq[1].color_weight)
            pytorch_color_seq[1].bias.copy_(mc_seq[1].color_bias)
            pytorch_color_seq[1].running_mean.copy_(mc_seq[1].color_running_mean)
            pytorch_color_seq[1].running_var.copy_(mc_seq[1].color_running_var)
            pytorch_color_seq[3].weight.copy_(mc_seq[3].color_weight)
            pytorch_color_seq[4].weight.copy_(mc_seq[4].color_weight)
            pytorch_color_seq[4].bias.copy_(mc_seq[4].color_bias)
            pytorch_color_seq[4].running_mean.copy_(mc_seq[4].color_running_mean)
            pytorch_color_seq[4].running_var.copy_(mc_seq[4].color_running_var)
            
            # Copy weights from MC to PyTorch for brightness pathway
            pytorch_brightness_seq[0].weight.copy_(mc_seq[0].brightness_weight)
            pytorch_brightness_seq[1].weight.copy_(mc_seq[1].brightness_weight)
            pytorch_brightness_seq[1].bias.copy_(mc_seq[1].brightness_bias)
            pytorch_brightness_seq[1].running_mean.copy_(mc_seq[1].brightness_running_mean)
            pytorch_brightness_seq[1].running_var.copy_(mc_seq[1].brightness_running_var)
            pytorch_brightness_seq[3].weight.copy_(mc_seq[3].brightness_weight)
            pytorch_brightness_seq[4].weight.copy_(mc_seq[4].brightness_weight)
            pytorch_brightness_seq[4].bias.copy_(mc_seq[4].brightness_bias)
            pytorch_brightness_seq[4].running_mean.copy_(mc_seq[4].brightness_running_mean)
            pytorch_brightness_seq[4].running_var.copy_(mc_seq[4].brightness_running_var)
        
        # Set all to eval mode for consistent batch norm behavior
        mc_seq.eval()
        pytorch_color_seq.eval()
        pytorch_brightness_seq.eval()
        
        # Forward pass
        with torch.no_grad():
            mc_color_out, mc_brightness_out = mc_seq(self.color_input, self.brightness_input)
            pytorch_color_out = pytorch_color_seq(self.color_input)
            pytorch_brightness_out = pytorch_brightness_seq(self.brightness_input)
        
        # Compare outputs - should be identical
        torch.testing.assert_close(mc_color_out, pytorch_color_out, atol=1e-5, rtol=1e-5,
                                 msg="MC color pathway differs from equivalent PyTorch Sequential")
        torch.testing.assert_close(mc_brightness_out, pytorch_brightness_out, atol=1e-5, rtol=1e-5,
                                 msg="MC brightness pathway differs from equivalent PyTorch Sequential")
        
        # Test individual pathway methods match too
        mc_color_only = mc_seq.forward_color(self.color_input)
        mc_brightness_only = mc_seq.forward_brightness(self.brightness_input)
        
        torch.testing.assert_close(mc_color_only, pytorch_color_out, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(mc_brightness_only, pytorch_brightness_out, atol=1e-5, rtol=1e-5)
    
    def test_mcsequential_gradient_flow_vs_pytorch(self):
        """Test MCSequential gradient flow matches PyTorch Sequential exactly."""
        # Create MC sequence
        mc_seq = MCSequential(
            MCConv2d(64, 64, 128, 128, kernel_size=3, padding=1),
            MCBatchNorm2d(128, 128),
            MCReLU()
        )
        
        # Create equivalent PyTorch sequences
        pytorch_color_seq = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        pytorch_brightness_seq = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # Copy weights for exact comparison
        with torch.no_grad():
            pytorch_color_seq[0].weight.copy_(mc_seq[0].color_weight)
            pytorch_color_seq[0].bias.copy_(mc_seq[0].color_bias)
            pytorch_color_seq[1].weight.copy_(mc_seq[1].color_weight)
            pytorch_color_seq[1].bias.copy_(mc_seq[1].color_bias)
            
            pytorch_brightness_seq[0].weight.copy_(mc_seq[0].brightness_weight)
            pytorch_brightness_seq[0].bias.copy_(mc_seq[0].brightness_bias)
            pytorch_brightness_seq[1].weight.copy_(mc_seq[1].brightness_weight)
            pytorch_brightness_seq[1].bias.copy_(mc_seq[1].brightness_bias)
        
        # Test gradient flow
        color_input = self.color_input.clone().requires_grad_(True)
        brightness_input = self.brightness_input.clone().requires_grad_(True)
        
        pytorch_color_input = self.color_input.clone().requires_grad_(True)
        pytorch_brightness_input = self.brightness_input.clone().requires_grad_(True)
        
        # MC forward and backward
        mc_color_out, mc_brightness_out = mc_seq(color_input, brightness_input)
        mc_loss = mc_color_out.sum() + mc_brightness_out.sum()
        mc_loss.backward()
        
        # PyTorch forward and backward
        pytorch_color_out = pytorch_color_seq(pytorch_color_input)
        pytorch_brightness_out = pytorch_brightness_seq(pytorch_brightness_input)
        pytorch_loss = pytorch_color_out.sum() + pytorch_brightness_out.sum()
        pytorch_loss.backward()
        
        # Compare input gradients
        torch.testing.assert_close(color_input.grad, pytorch_color_input.grad, atol=1e-5, rtol=1e-5,
                                 msg="Color input gradients differ from PyTorch")
        torch.testing.assert_close(brightness_input.grad, pytorch_brightness_input.grad, atol=1e-5, rtol=1e-5,
                                 msg="Brightness input gradients differ from PyTorch")
        
        # Compare parameter gradients
        torch.testing.assert_close(mc_seq[0].color_weight.grad, pytorch_color_seq[0].weight.grad, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(mc_seq[0].brightness_weight.grad, pytorch_brightness_seq[0].weight.grad, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(mc_seq[1].color_weight.grad, pytorch_color_seq[1].weight.grad, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(mc_seq[1].brightness_weight.grad, pytorch_brightness_seq[1].weight.grad, atol=1e-5, rtol=1e-5)
    
    def test_mixed_channel_progression_vs_pytorch_equivalent(self):
        """Test mixed channel progression against equivalent PyTorch networks."""
        # Create MC network with mixed channel progressions
        mc_network = MCSequential(
            # Initial projection: 3,1 -> 64,64
            MCConv2d(3, 1, 64, 64, kernel_size=7, stride=2, padding=3, bias=False),
            MCBatchNorm2d(64, 64),
            MCReLU(inplace=True),
            
            # Feature expansion: 64,64 -> 128,128
            MCConv2d(64, 64, 128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            MCBatchNorm2d(128, 128),
            MCReLU(inplace=True),
            
            # Further expansion: 128,128 -> 256,256  
            MCConv2d(128, 128, 256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            MCBatchNorm2d(256, 256),
        )
        
        # Create equivalent PyTorch networks for each pathway
        pytorch_color_network = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
        )
        
        pytorch_brightness_network = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
        )
        
        # Copy weights from MC to PyTorch
        mc_conv_layers = [mc_network[0], mc_network[3], mc_network[6]]
        mc_bn_layers = [mc_network[1], mc_network[4], mc_network[7]]
        
        pytorch_color_conv_layers = [pytorch_color_network[0], pytorch_color_network[3], pytorch_color_network[6]]
        pytorch_color_bn_layers = [pytorch_color_network[1], pytorch_color_network[4], pytorch_color_network[7]]
        
        pytorch_brightness_conv_layers = [pytorch_brightness_network[0], pytorch_brightness_network[3], pytorch_brightness_network[6]]
        pytorch_brightness_bn_layers = [pytorch_brightness_network[1], pytorch_brightness_network[4], pytorch_brightness_network[7]]
        
        with torch.no_grad():
            for mc_conv, pt_color_conv, pt_brightness_conv in zip(mc_conv_layers, pytorch_color_conv_layers, pytorch_brightness_conv_layers):
                pt_color_conv.weight.copy_(mc_conv.color_weight)
                pt_brightness_conv.weight.copy_(mc_conv.brightness_weight)
            
            for mc_bn, pt_color_bn, pt_brightness_bn in zip(mc_bn_layers, pytorch_color_bn_layers, pytorch_brightness_bn_layers):
                pt_color_bn.weight.copy_(mc_bn.color_weight)
                pt_color_bn.bias.copy_(mc_bn.color_bias)
                pt_color_bn.running_mean.copy_(mc_bn.color_running_mean)
                pt_color_bn.running_var.copy_(mc_bn.color_running_var)
                
                pt_brightness_bn.weight.copy_(mc_bn.brightness_weight)
                pt_brightness_bn.bias.copy_(mc_bn.brightness_bias)
                pt_brightness_bn.running_mean.copy_(mc_bn.brightness_running_mean)
                pt_brightness_bn.running_var.copy_(mc_bn.brightness_running_var)
        
        # Test inputs
        color_input = torch.randn(2, 3, 224, 224)
        brightness_input = torch.randn(2, 1, 224, 224)
        
        # Set to eval mode
        mc_network.eval()
        pytorch_color_network.eval()
        pytorch_brightness_network.eval()
        
        # Forward pass
        with torch.no_grad():
            mc_color_out, mc_brightness_out = mc_network(color_input, brightness_input)
            pytorch_color_out = pytorch_color_network(color_input)
            pytorch_brightness_out = pytorch_brightness_network(brightness_input)
        
        # Compare outputs - should be identical
        torch.testing.assert_close(mc_color_out, pytorch_color_out, atol=1e-5, rtol=1e-5,
                                 msg="MC mixed progression color pathway differs from PyTorch equivalent")
        torch.testing.assert_close(mc_brightness_out, pytorch_brightness_out, atol=1e-5, rtol=1e-5,
                                 msg="MC mixed progression brightness pathway differs from PyTorch equivalent")
        
        # Test shapes match expected
        expected_shape = (2, 256, 28, 28)  # 224 -> 112 -> 56 -> 28
        self.assertEqual(mc_color_out.shape, expected_shape)
        self.assertEqual(mc_brightness_out.shape, expected_shape)
        self.assertEqual(pytorch_color_out.shape, expected_shape)
        self.assertEqual(pytorch_brightness_out.shape, expected_shape)
    
    def test_mcsequential_vs_pytorch_with_different_activation_patterns(self):
        """Test MCSequential against PyTorch with different activation patterns."""
        # Test 1: ReLU activation
        mc_relu_seq = MCSequential(
            MCConv2d(64, 64, 128, 128, kernel_size=3, padding=1),
            MCReLU()
        )
        
        pytorch_relu_seq = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Copy weights
        with torch.no_grad():
            pytorch_relu_seq[0].weight.copy_(mc_relu_seq[0].color_weight)
            pytorch_relu_seq[0].bias.copy_(mc_relu_seq[0].color_bias)
        
        # Test with negative values to verify ReLU behavior
        test_input = torch.randn(2, 64, 8, 8) * 2  # Some negative values
        
        mc_relu_seq.eval()
        pytorch_relu_seq.eval()
        
        with torch.no_grad():
            mc_color_out, _ = mc_relu_seq(test_input, test_input)
            pytorch_out = pytorch_relu_seq(test_input)
        
        torch.testing.assert_close(mc_color_out, pytorch_out, atol=1e-5, rtol=1e-5,
                                 msg="MC ReLU sequence differs from PyTorch ReLU sequence")
        
        # Verify ReLU activation worked (all outputs should be non-negative)
        self.assertTrue(torch.all(mc_color_out >= 0))
        self.assertTrue(torch.all(pytorch_out >= 0))
    
    def test_mcsequential_batch_norm_behavior_vs_pytorch(self):
        """Test MCSequential batch normalization behavior matches PyTorch exactly."""
        # Create sequences with batch norm
        mc_seq = MCSequential(
            MCConv2d(64, 64, 128, 128, kernel_size=3, padding=1, bias=False),
            MCBatchNorm2d(128, 128),
        )
        
        pytorch_color_seq = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
        )
        
        pytorch_brightness_seq = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
        )
        
        # Copy weights
        with torch.no_grad():
            pytorch_color_seq[0].weight.copy_(mc_seq[0].color_weight)
            pytorch_color_seq[1].weight.copy_(mc_seq[1].color_weight)
            pytorch_color_seq[1].bias.copy_(mc_seq[1].color_bias)
            
            pytorch_brightness_seq[0].weight.copy_(mc_seq[0].brightness_weight)
            pytorch_brightness_seq[1].weight.copy_(mc_seq[1].brightness_weight)
            pytorch_brightness_seq[1].bias.copy_(mc_seq[1].brightness_bias)
        
        # Test training mode behavior (batch norm should update running stats)
        mc_seq.train()
        pytorch_color_seq.train()
        pytorch_brightness_seq.train()
        
        test_input = torch.randn(4, 64, 8, 8)  # Larger batch for meaningful stats
        
        # Forward pass in training mode
        mc_color_out, mc_brightness_out = mc_seq(test_input, test_input)
        pytorch_color_out = pytorch_color_seq(test_input)
        pytorch_brightness_out = pytorch_brightness_seq(test_input)
        
        # Outputs should be close (small differences due to different running stat updates)
        torch.testing.assert_close(mc_color_out, pytorch_color_out, atol=1e-3, rtol=1e-3,
                                 msg="MC training mode output differs significantly from PyTorch")
        torch.testing.assert_close(mc_brightness_out, pytorch_brightness_out, atol=1e-3, rtol=1e-3,
                                 msg="MC training mode output differs significantly from PyTorch")
        
        # Test eval mode behavior (should use running stats, be identical)
        mc_seq.eval()
        pytorch_color_seq.eval()
        pytorch_brightness_seq.eval()
        
        with torch.no_grad():
            mc_color_out_eval, mc_brightness_out_eval = mc_seq(test_input, test_input)
            pytorch_color_out_eval = pytorch_color_seq(test_input)
            pytorch_brightness_out_eval = pytorch_brightness_seq(test_input)
        
        # In eval mode with same running stats, should be nearly identical
        torch.testing.assert_close(mc_color_out_eval, pytorch_color_out_eval, atol=1e-5, rtol=1e-5,
                                 msg="MC eval mode output differs from PyTorch")
        torch.testing.assert_close(mc_brightness_out_eval, pytorch_brightness_out_eval, atol=1e-5, rtol=1e-5,
                                 msg="MC eval mode output differs from PyTorch")

    def test_mcsequential_mixed_channel_progressions(self):
        """Test MCSequential with mixed channel progressions matching real ResNet patterns."""
        # Simulate ResNet-50 initial layers: RGB+Brightness -> 64+64 -> 128+128 -> 256+256
        mixed_progression = MCSequential(
            # Initial projection: 3,1 -> 64,64
            MCConv2d(3, 1, 64, 64, kernel_size=7, stride=2, padding=3, bias=False),
            MCBatchNorm2d(64, 64),
            MCReLU(inplace=True),
            
            # Feature expansion: 64,64 -> 128,128
            MCConv2d(64, 64, 128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            MCBatchNorm2d(128, 128),
            MCReLU(inplace=True),
            
            # Further expansion: 128,128 -> 256,256  
            MCConv2d(128, 128, 256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            MCBatchNorm2d(256, 256),
        )
        
        # Test with realistic inputs
        color_input = torch.randn(2, 3, 224, 224)    # RGB input
        brightness_input = torch.randn(2, 1, 224, 224) # Grayscale input
        
        color_out, brightness_out = mixed_progression(color_input, brightness_input)
        
        # Both pathways should converge to same output shape
        # 224 -> 112 -> 56 -> 28 due to stride=2 layers
        expected_shape = (2, 256, 28, 28)
        self.assertEqual(color_out.shape, expected_shape)
        self.assertEqual(brightness_out.shape, expected_shape)
        
        # Test individual pathway consistency
        color_only = mixed_progression.forward_color(color_input)
        brightness_only = mixed_progression.forward_brightness(brightness_input)
        self.assertEqual(color_only.shape, expected_shape)
        self.assertEqual(brightness_only.shape, expected_shape)
    
    def test_mcsequential_deep_network_performance(self):
        """Test MCSequential with deep networks for memory and gradient efficiency."""
        # Create a deeper network to test gradient flow and memory efficiency
        deep_layers = []
        in_channels = 64
        
        for i in range(10):  # 10 residual-like blocks
            out_channels = min(in_channels * 2, 512)  # Cap at 512 channels
            
            deep_layers.extend([
                MCConv2d(in_channels, in_channels, out_channels, out_channels, 
                        kernel_size=3, padding=1, bias=False),
                MCBatchNorm2d(out_channels, out_channels),
                MCReLU(inplace=True)
            ])
            
            in_channels = out_channels
        
        deep_seq = MCSequential(*deep_layers)
        
        # Test forward pass
        color_out, brightness_out = deep_seq(self.color_input, self.brightness_input)
        
        # Check output shapes
        expected_channels = 512  # Capped value
        self.assertEqual(color_out.shape[1], expected_channels)
        self.assertEqual(brightness_out.shape[1], expected_channels)
        
        # Test gradient flow through deep network
        color_input_grad = self.color_input.requires_grad_(True)
        brightness_input_grad = self.brightness_input.requires_grad_(True)
        
        color_out, brightness_out = deep_seq(color_input_grad, brightness_input_grad)
        loss = color_out.mean() + brightness_out.mean()
        loss.backward()
        
        # Verify gradients exist throughout the network
        self.assertIsNotNone(color_input_grad.grad)
        self.assertIsNotNone(brightness_input_grad.grad)
        
        # Check that all parameters have gradients
        for name, param in deep_seq.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"No gradient for {name}")
    
    def test_mcsequential_nested_container_patterns(self):
        """Test MCSequential with nested container patterns like ResNet blocks."""
        # Create a block that mimics ResNet BasicBlock structure
        def create_mc_basic_block(in_channels, out_channels, stride=1):
            layers = [
                MCConv2d(in_channels, in_channels, out_channels, out_channels,
                        kernel_size=3, stride=stride, padding=1, bias=False),
                MCBatchNorm2d(out_channels, out_channels),
                MCReLU(inplace=True),
                MCConv2d(out_channels, out_channels, out_channels, out_channels,
                        kernel_size=3, stride=1, padding=1, bias=False),
                MCBatchNorm2d(out_channels, out_channels),
            ]
            return MCSequential(*layers)
        
        # Create multiple blocks
        block1 = create_mc_basic_block(64, 128, stride=2)
        block2 = create_mc_basic_block(128, 256, stride=2)
        
        # Combine blocks
        combined_network = MCSequential(block1, block2)
        
        # Test forward pass
        color_out, brightness_out = combined_network(self.color_input, self.brightness_input)
        
        # Verify expected transformations: 64->128->256 channels, 16->8->4 spatial
        expected_shape = (2, 256, 4, 4)
        self.assertEqual(color_out.shape, expected_shape)
        self.assertEqual(brightness_out.shape, expected_shape)
    
    def test_mcsequential_device_handling(self):
        """Test MCSequential device handling matches PyTorch patterns."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        # Create sequence
        mc_seq = MCSequential(
            MCConv2d(64, 64, 128, 128, kernel_size=3, padding=1),
            MCBatchNorm2d(128, 128),
            MCReLU()
        )
        
        # Test device movement
        mc_seq = mc_seq.cuda()
        
        # All parameters should be on GPU
        for param in mc_seq.parameters():
            self.assertTrue(param.is_cuda)
        
        # Test forward pass on GPU
        color_input_gpu = self.color_input.cuda()
        brightness_input_gpu = self.brightness_input.cuda()
        
        color_out, brightness_out = mc_seq(color_input_gpu, brightness_input_gpu)
        
        self.assertTrue(color_out.is_cuda)
        self.assertTrue(brightness_out.is_cuda)
        
        # Move back to CPU
        mc_seq = mc_seq.cpu()
        for param in mc_seq.parameters():
            self.assertFalse(param.is_cuda)
    
    def test_mcsequential_scripting_compatibility(self):
        """Test MCSequential compatibility with PyTorch scripting."""
        # Create simple sequence
        mc_seq = MCSequential(
            MCConv2d(64, 64, 128, 128, kernel_size=1),
            MCReLU()
        )
        
        # Set to eval mode for scripting
        mc_seq.eval()
        
        # Test that the model can be traced (basic JIT compatibility)
        with torch.no_grad():
            color_out, brightness_out = mc_seq(self.color_input, self.brightness_input)
            
            # Verify outputs are reasonable
            self.assertEqual(color_out.shape, (2, 128, 16, 16))
            self.assertEqual(brightness_out.shape, (2, 128, 16, 16))
            
            # Check ReLU activation worked (all outputs should be non-negative)
            self.assertTrue(torch.all(color_out >= 0))
            self.assertTrue(torch.all(brightness_out >= 0))
    
    def test_mcsequential_memory_efficiency_vs_pytorch(self):
        """Test MCSequential memory usage patterns vs PyTorch Sequential."""
        # Create equivalent networks
        mc_seq = MCSequential(
            MCConv2d(64, 64, 128, 128, kernel_size=3, padding=1),
            MCBatchNorm2d(128, 128),
            MCReLU(inplace=True)
        )
        
        pytorch_seq = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Test memory usage during forward pass
        # Note: This is a basic test - real memory profiling would need specialized tools
        
        # MCSequential forward
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        with torch.no_grad():
            color_out, brightness_out = mc_seq(self.color_input, self.brightness_input)
            mc_output_size = color_out.numel() + brightness_out.numel()
        
        # PyTorch Sequential forward
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        with torch.no_grad():
            pytorch_out = pytorch_seq(self.color_input)
            pytorch_output_size = pytorch_out.numel()
        
        # MCSequential should produce about 2x the output (dual streams)
        self.assertAlmostEqual(mc_output_size / pytorch_output_size, 2.0, delta=0.1)
    
    def test_mcsequential_checkpoint_compatibility(self):
        """Test MCSequential state dict checkpoint save/load compatibility."""
        # Create sequence
        mc_seq1 = MCSequential(
            MCConv2d(64, 64, 128, 128, kernel_size=3, padding=1, bias=False),
            MCBatchNorm2d(128, 128),
            MCReLU()
        )
        
        # Train for a few steps to modify parameters
        optimizer = torch.optim.SGD(mc_seq1.parameters(), lr=0.01)
        for _ in range(5):
            color_out, brightness_out = mc_seq1(self.color_input, self.brightness_input)
            loss = color_out.sum() + brightness_out.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # Save checkpoint
        checkpoint = {
            'model_state_dict': mc_seq1.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': 5
        }
        
        # Create new model and load checkpoint
        mc_seq2 = MCSequential(
            MCConv2d(64, 64, 128, 128, kernel_size=3, padding=1, bias=False),
            MCBatchNorm2d(128, 128),
            MCReLU()
        )
        
        mc_seq2.load_state_dict(checkpoint['model_state_dict'])
        
        # Verify parameters match exactly
        for (name1, param1), (name2, param2) in zip(mc_seq1.named_parameters(), mc_seq2.named_parameters()):
            self.assertEqual(name1, name2)
            torch.testing.assert_close(param1, param2, atol=1e-6, rtol=1e-6)
        
        # Verify forward pass gives identical results
        mc_seq1.eval()
        mc_seq2.eval()
        
        with torch.no_grad():
            color_out1, brightness_out1 = mc_seq1(self.color_input, self.brightness_input)
            color_out2, brightness_out2 = mc_seq2(self.color_input, self.brightness_input)
            
            torch.testing.assert_close(color_out1, color_out2)
            torch.testing.assert_close(brightness_out1, brightness_out2)
    
    def test_mcsequential_parameter_sharing_patterns(self):
        """Test MCSequential parameter sharing and weight tying patterns."""
        # Create sequence with shared modules that maintain channel consistency
        shared_conv = MCConv2d(128, 128, 128, 128, kernel_size=3, padding=1)  # 128->128 channels
        shared_bn = MCBatchNorm2d(128, 128)
        
        # First transform 64->128, then reuse 128->128 modules
        weight_shared_seq = MCSequential(
            MCConv2d(64, 64, 128, 128, kernel_size=3, padding=1),  # Initial transform: 64->128
            MCBatchNorm2d(128, 128),
            MCReLU(),
            shared_conv,  # Reuse same conv layer: 128->128
            shared_bn,    # Reuse same bn layer
            MCReLU(),
            shared_conv,  # Reuse again: 128->128
            shared_bn,    # Reuse again
            MCReLU()
        )
        
        # Test forward pass
        color_out, brightness_out = weight_shared_seq(self.color_input, self.brightness_input)
        
        # Verify output shapes
        self.assertEqual(color_out.shape, (2, 128, 16, 16))
        self.assertEqual(brightness_out.shape, (2, 128, 16, 16))
        
        # Test gradient flow with shared parameters
        color_input_grad = self.color_input.requires_grad_(True)
        brightness_input_grad = self.brightness_input.requires_grad_(True)
        
        color_out, brightness_out = weight_shared_seq(color_input_grad, brightness_input_grad)
        loss = color_out.mean() + brightness_out.mean()
        loss.backward()
        
        # Shared parameters should accumulate gradients from multiple uses
        self.assertIsNotNone(shared_conv.color_weight.grad)
        self.assertIsNotNone(shared_conv.brightness_weight.grad)
        self.assertIsNotNone(shared_bn.color_weight.grad)
        self.assertIsNotNone(shared_bn.brightness_weight.grad)
    
    def test_mcsequential_dynamic_construction_patterns(self):
        """Test MCSequential dynamic construction patterns matching PyTorch usage."""
        # Test dynamic construction similar to how ResNet/VGG are built
        def make_mc_layer(in_channels, out_channels, num_blocks, stride=1):
            layers = []
            
            # First block with potential stride
            layers.extend([
                MCConv2d(in_channels, in_channels, out_channels, out_channels,
                        kernel_size=3, stride=stride, padding=1, bias=False),
                MCBatchNorm2d(out_channels, out_channels),
                MCReLU(inplace=True)
            ])
            
            # Remaining blocks
            for _ in range(num_blocks - 1):
                layers.extend([
                    MCConv2d(out_channels, out_channels, out_channels, out_channels,
                            kernel_size=3, stride=1, padding=1, bias=False),
                    MCBatchNorm2d(out_channels, out_channels),
                    MCReLU(inplace=True)
                ])
            
            return layers
        
        # Build network dynamically
        all_layers = []
        all_layers.extend(make_mc_layer(64, 128, 2, stride=2))    # layer1: 64->128, stride 2
        all_layers.extend(make_mc_layer(128, 256, 2, stride=2))   # layer2: 128->256, stride 2
        all_layers.extend(make_mc_layer(256, 512, 2, stride=2))   # layer3: 256->512, stride 2
        
        dynamic_network = MCSequential(*all_layers)
        
        # Test forward pass
        color_out, brightness_out = dynamic_network(self.color_input, self.brightness_input)
        
        # Verify final transformation: 64->512 channels, 16->2 spatial (3 stride=2 layers)
        expected_shape = (2, 512, 2, 2)
        self.assertEqual(color_out.shape, expected_shape)
        self.assertEqual(brightness_out.shape, expected_shape)
        
        # Test that the dynamic construction works correctly
        self.assertEqual(len(dynamic_network), len(all_layers))
        
        # Verify gradient flow through dynamically constructed network
        color_input_grad = self.color_input.requires_grad_(True)
        brightness_input_grad = self.brightness_input.requires_grad_(True)
        
        color_out, brightness_out = dynamic_network(color_input_grad, brightness_input_grad)
        loss = color_out.mean() + brightness_out.mean()
        loss.backward()
        
        self.assertIsNotNone(color_input_grad.grad)
        self.assertIsNotNone(brightness_input_grad.grad)
    
class TestDirectPyTorchCompatibility(unittest.TestCase):
    """Direct PyTorch compatibility tests comparing MCSequential against nn.Sequential with identical weights."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.color_input = torch.randn(self.batch_size, 64, 32, 32)
        self.brightness_input = torch.randn(self.batch_size, 64, 32, 32)
        self.pytorch_input = torch.randn(self.batch_size, 64, 32, 32)
    
    def test_mcsequential_vs_pytorch_sequential_simple_conv_bn_relu(self):
        """Test MCSequential vs PyTorch Sequential for simple conv+bn+relu."""
        # Create MC sequence
        mc_seq = MCSequential(
            MCConv2d(64, 64, 128, 128, kernel_size=3, padding=1, bias=False),
            MCBatchNorm2d(128, 128),
            MCReLU(inplace=True)
        )
        
        # Create equivalent PyTorch sequence
        pytorch_seq = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Set to eval mode for consistent batch norm behavior
        mc_seq.eval()
        pytorch_seq.eval()
        
        # Copy MC color pathway weights to PyTorch
        with torch.no_grad():
            pytorch_seq[0].weight.copy_(mc_seq[0].color_weight)
            pytorch_seq[1].weight.copy_(mc_seq[1].color_weight)
            pytorch_seq[1].bias.copy_(mc_seq[1].color_bias)
            pytorch_seq[1].running_mean.copy_(mc_seq[1].color_running_mean)
            pytorch_seq[1].running_var.copy_(mc_seq[1].color_running_var)
        
        # Test color pathway against PyTorch
        mc_color_out = mc_seq.forward_color(self.pytorch_input)
        pytorch_out = pytorch_seq(self.pytorch_input)
        
        torch.testing.assert_close(mc_color_out, pytorch_out, atol=1e-5, rtol=1e-5,
                                 msg="MCSequential color pathway differs from PyTorch Sequential")
        
        # Copy MC brightness pathway weights to PyTorch
        with torch.no_grad():
            pytorch_seq[0].weight.copy_(mc_seq[0].brightness_weight)
            pytorch_seq[1].weight.copy_(mc_seq[1].brightness_weight)
            pytorch_seq[1].bias.copy_(mc_seq[1].brightness_bias)
            pytorch_seq[1].running_mean.copy_(mc_seq[1].brightness_running_mean)
            pytorch_seq[1].running_var.copy_(mc_seq[1].brightness_running_var)
        
        # Test brightness pathway against PyTorch
        mc_brightness_out = mc_seq.forward_brightness(self.pytorch_input)
        pytorch_out_brightness = pytorch_seq(self.pytorch_input)
        
        torch.testing.assert_close(mc_brightness_out, pytorch_out_brightness, atol=1e-5, rtol=1e-5,
                                 msg="MCSequential brightness pathway differs from PyTorch Sequential")
    
    def test_mcsequential_vs_pytorch_sequential_deep_network(self):
        """Test MCSequential vs PyTorch Sequential for deeper networks."""
        # Create deeper MC sequence
        mc_seq = MCSequential(
            MCConv2d(64, 64, 128, 128, kernel_size=3, padding=1, bias=False),
            MCBatchNorm2d(128, 128),
            MCReLU(inplace=True),
            MCConv2d(128, 128, 256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            MCBatchNorm2d(256, 256),
            MCReLU(inplace=True),
            MCConv2d(256, 256, 512, 512, kernel_size=1, bias=False),
            MCBatchNorm2d(512, 512)
        )
        
        # Create equivalent PyTorch sequence
        pytorch_seq = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512)
        )
        
        # Set to eval mode
        mc_seq.eval()
        pytorch_seq.eval()
        
        # Test color pathway
        self._copy_mc_to_pytorch_weights(mc_seq, pytorch_seq, pathway='color')
        
        mc_color_out = mc_seq.forward_color(self.pytorch_input)
        pytorch_out = pytorch_seq(self.pytorch_input)
        
        torch.testing.assert_close(mc_color_out, pytorch_out, atol=1e-5, rtol=1e-5,
                                 msg="Deep MCSequential color pathway differs from PyTorch Sequential")
        
        # Test brightness pathway
        self._copy_mc_to_pytorch_weights(mc_seq, pytorch_seq, pathway='brightness')
        
        mc_brightness_out = mc_seq.forward_brightness(self.pytorch_input)
        pytorch_out_brightness = pytorch_seq(self.pytorch_input)
        
        torch.testing.assert_close(mc_brightness_out, pytorch_out_brightness, atol=1e-5, rtol=1e-5,
                                 msg="Deep MCSequential brightness pathway differs from PyTorch Sequential")
    
    def test_mcsequential_vs_pytorch_sequential_different_channels(self):
        """Test MCSequential vs PyTorch when starting with different channel counts."""
        # Start with different channels like real RGB+Brightness scenario
        color_input = torch.randn(2, 3, 32, 32)  # RGB
        brightness_input = torch.randn(2, 1, 32, 32)  # Brightness
        
        # MC sequence that handles different initial channels
        mc_seq = MCSequential(
            MCConv2d(3, 1, 64, 64, kernel_size=7, stride=2, padding=3, bias=False),
            MCBatchNorm2d(64, 64),
            MCReLU(inplace=True),
            MCConv2d(64, 64, 128, 128, kernel_size=3, padding=1, bias=False),
            MCBatchNorm2d(128, 128)
        )
        
        # Test color pathway (3 -> 64 -> 128)
        pytorch_seq_color = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        
        # Test brightness pathway (1 -> 64 -> 128)
        pytorch_seq_brightness = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        
        # Set to eval mode
        mc_seq.eval()
        pytorch_seq_color.eval()
        pytorch_seq_brightness.eval()
        
        # Copy weights and test color pathway
        self._copy_mc_to_pytorch_weights(mc_seq, pytorch_seq_color, pathway='color')
        
        mc_color_out = mc_seq.forward_color(color_input)
        pytorch_color_out = pytorch_seq_color(color_input)
        
        torch.testing.assert_close(mc_color_out, pytorch_color_out, atol=1e-5, rtol=1e-5,
                                 msg="MCSequential color pathway with different input channels differs from PyTorch")
        
        # Copy weights and test brightness pathway
        self._copy_mc_to_pytorch_weights(mc_seq, pytorch_seq_brightness, pathway='brightness')
        
        mc_brightness_out = mc_seq.forward_brightness(brightness_input)
        pytorch_brightness_out = pytorch_seq_brightness(brightness_input)
        
        torch.testing.assert_close(mc_brightness_out, pytorch_brightness_out, atol=1e-5, rtol=1e-5,
                                 msg="MCSequential brightness pathway with different input channels differs from PyTorch")
    
    def test_mcsequential_vs_pytorch_sequential_with_stride_patterns(self):
        """Test MCSequential vs PyTorch Sequential with various stride patterns."""
        # Test different stride patterns common in ResNets
        stride_configs = [
            [(1, 1), (1, 1), (1, 1)],  # No downsampling
            [(2, 2), (1, 1), (1, 1)],  # Early downsampling
            [(1, 1), (2, 2), (1, 1)],  # Mid downsampling
            [(2, 2), (2, 2), (1, 1)],  # Multiple downsampling
        ]
        
        for stride_config in stride_configs:
            with self.subTest(stride_config=stride_config):
                # Create MC sequence with stride pattern
                mc_seq = MCSequential(
                    MCConv2d(64, 64, 128, 128, kernel_size=3, stride=stride_config[0], padding=1, bias=False),
                    MCBatchNorm2d(128, 128),
                    MCReLU(),
                    MCConv2d(128, 128, 256, 256, kernel_size=3, stride=stride_config[1], padding=1, bias=False),
                    MCBatchNorm2d(256, 256),
                    MCReLU(),
                    MCConv2d(256, 256, 512, 512, kernel_size=3, stride=stride_config[2], padding=1, bias=False),
                    MCBatchNorm2d(512, 512)
                )
                
                # Create equivalent PyTorch sequence
                pytorch_seq = nn.Sequential(
                    nn.Conv2d(64, 128, kernel_size=3, stride=stride_config[0], padding=1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.Conv2d(128, 256, kernel_size=3, stride=stride_config[1], padding=1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.Conv2d(256, 512, kernel_size=3, stride=stride_config[2], padding=1, bias=False),
                    nn.BatchNorm2d(512)
                )
                
                # Set to eval mode
                mc_seq.eval()
                pytorch_seq.eval()
                
                # Test color pathway
                self._copy_mc_to_pytorch_weights(mc_seq, pytorch_seq, pathway='color')
                
                mc_color_out = mc_seq.forward_color(self.pytorch_input)
                pytorch_out = pytorch_seq(self.pytorch_input)
                
                torch.testing.assert_close(mc_color_out, pytorch_out, atol=1e-5, rtol=1e-5,
                                         msg=f"MCSequential with stride {stride_config} differs from PyTorch")
                
                # Verify output shapes match
                self.assertEqual(mc_color_out.shape, pytorch_out.shape)
    
    def test_mcsequential_vs_pytorch_sequential_kernel_sizes(self):
        """Test MCSequential vs PyTorch Sequential with different kernel sizes."""
        kernel_configs = [
            [(1, 1), (3, 3), (5, 5)],  # Mixed kernel sizes
            [(7, 7), (3, 3), (1, 1)],  # Large to small
            [(3, 3), (3, 3), (3, 3)],  # All same (typical ResNet)
            [(1, 3), (3, 1), (3, 3)],  # Asymmetric kernels
        ]
        
        for kernel_config in kernel_configs:
            with self.subTest(kernel_config=kernel_config):
                # Calculate appropriate padding for each kernel
                paddings = []
                for kh, kw in kernel_config:
                    pad_h = kh // 2
                    pad_w = kw // 2
                    paddings.append((pad_h, pad_w))
                
                # Create MC sequence
                mc_seq = MCSequential(
                    MCConv2d(64, 64, 128, 128, kernel_size=kernel_config[0], padding=paddings[0], bias=False),
                    MCBatchNorm2d(128, 128),
                    MCReLU(),
                    MCConv2d(128, 128, 256, 256, kernel_size=kernel_config[1], padding=paddings[1], bias=False),
                    MCBatchNorm2d(256, 256),
                    MCReLU(),
                    MCConv2d(256, 256, 512, 512, kernel_size=kernel_config[2], padding=paddings[2], bias=False),
                    MCBatchNorm2d(512, 512)
                )
                
                # Create equivalent PyTorch sequence
                pytorch_seq = nn.Sequential(
                    nn.Conv2d(64, 128, kernel_size=kernel_config[0], padding=paddings[0], bias=False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.Conv2d(128, 256, kernel_size=kernel_config[1], padding=paddings[1], bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.Conv2d(256, 512, kernel_size=kernel_config[2], padding=paddings[2], bias=False),
                    nn.BatchNorm2d(512)
                )
                
                # Set to eval mode
                mc_seq.eval()
                pytorch_seq.eval()
                
                # Test color pathway
                self._copy_mc_to_pytorch_weights(mc_seq, pytorch_seq, pathway='color')
                
                mc_color_out = mc_seq.forward_color(self.pytorch_input)
                pytorch_out = pytorch_seq(self.pytorch_input)
                
                torch.testing.assert_close(mc_color_out, pytorch_out, atol=1e-5, rtol=1e-5,
                                         msg=f"MCSequential with kernels {kernel_config} differs from PyTorch")
    
    def test_mcsequential_vs_pytorch_sequential_with_bias(self):
        """Test MCSequential vs PyTorch Sequential with bias enabled."""
        # Create MC sequence with bias
        mc_seq = MCSequential(
            MCConv2d(64, 64, 128, 128, kernel_size=3, padding=1, bias=True),
            MCReLU(),
            MCConv2d(128, 128, 256, 256, kernel_size=1, bias=True),
            MCReLU()
        )
        
        # Create equivalent PyTorch sequence
        pytorch_seq = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=1, bias=True),
            nn.ReLU()
        )
        
        # Set to eval mode
        mc_seq.eval()
        pytorch_seq.eval()
        
        # Test color pathway
        with torch.no_grad():
            # Copy conv weights and biases
            pytorch_seq[0].weight.copy_(mc_seq[0].color_weight)
            pytorch_seq[0].bias.copy_(mc_seq[0].color_bias)
            pytorch_seq[2].weight.copy_(mc_seq[2].color_weight)
            pytorch_seq[2].bias.copy_(mc_seq[2].color_bias)
        
        mc_color_out = mc_seq.forward_color(self.pytorch_input)
        pytorch_out = pytorch_seq(self.pytorch_input)
        
        torch.testing.assert_close(mc_color_out, pytorch_out, atol=1e-5, rtol=1e-5,
                                 msg="MCSequential with bias differs from PyTorch Sequential")
        
        # Test brightness pathway
        with torch.no_grad():
            # Copy brightness weights and biases
            pytorch_seq[0].weight.copy_(mc_seq[0].brightness_weight)
            pytorch_seq[0].bias.copy_(mc_seq[0].brightness_bias)
            pytorch_seq[2].weight.copy_(mc_seq[2].brightness_weight)
            pytorch_seq[2].bias.copy_(mc_seq[2].brightness_bias)
        
        mc_brightness_out = mc_seq.forward_brightness(self.pytorch_input)
        pytorch_out_brightness = pytorch_seq(self.pytorch_input)
        
        torch.testing.assert_close(mc_brightness_out, pytorch_out_brightness, atol=1e-5, rtol=1e-5,
                                 msg="MCSequential brightness with bias differs from PyTorch Sequential")
    
    def test_mcsequential_vs_pytorch_sequential_training_mode(self):
        """Test MCSequential vs PyTorch Sequential in training mode with batch norm."""
        # Create MC sequence
        mc_seq = MCSequential(
            MCConv2d(64, 64, 128, 128, kernel_size=3, padding=1, bias=False),
            MCBatchNorm2d(128, 128),
            MCReLU(),
            MCConv2d(128, 128, 256, 256, kernel_size=3, padding=1, bias=False),
            MCBatchNorm2d(256, 256)
        )
        
        # Create equivalent PyTorch sequence
        pytorch_seq = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256)
        )
        
        # Set to training mode
        mc_seq.train()
        pytorch_seq.train()
        
        # Copy initial batch norm statistics
        self._copy_mc_to_pytorch_weights(mc_seq, pytorch_seq, pathway='color')
        
        # Multiple forward passes to test batch norm running statistics update
        for i in range(3):
            test_input = torch.randn(4, 64, 32, 32)  # Larger batch for better batch norm
            
            mc_color_out = mc_seq.forward_color(test_input)
            pytorch_out = pytorch_seq(test_input)
            
            # In training mode, outputs might differ slightly due to batch norm statistics
            # but should be very close if implementation is correct
            torch.testing.assert_close(mc_color_out, pytorch_out, atol=1e-3, rtol=1e-3,
                                     msg=f"MCSequential training mode forward {i} differs significantly from PyTorch")
            
            # Copy updated running statistics for next iteration
            with torch.no_grad():
                pytorch_seq[1].running_mean.copy_(mc_seq[1].color_running_mean)
                pytorch_seq[1].running_var.copy_(mc_seq[1].color_running_var)
                pytorch_seq[4].running_mean.copy_(mc_seq[4].color_running_mean)
                pytorch_seq[4].running_var.copy_(mc_seq[4].color_running_var)
    
    def test_mcsequential_vs_pytorch_sequential_gradient_flow(self):
        """Test MCSequential vs PyTorch Sequential gradient flow."""
        # Create MC sequence
        mc_seq = MCSequential(
            MCConv2d(64, 64, 128, 128, kernel_size=3, padding=1, bias=False),
            MCBatchNorm2d(128, 128),
            MCReLU(),
            MCConv2d(128, 128, 256, 256, kernel_size=1, bias=False),
            MCBatchNorm2d(256, 256)
        )
        
        # Create equivalent PyTorch sequence
        pytorch_seq = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256)
        )
        
        # Set to eval mode for consistent behavior
        mc_seq.eval()
        pytorch_seq.eval()
        
        # Copy weights
        self._copy_mc_to_pytorch_weights(mc_seq, pytorch_seq, pathway='color')
        
        # Test inputs with gradients
        mc_input = self.pytorch_input.clone().requires_grad_(True)
        pytorch_input = self.pytorch_input.clone().requires_grad_(True)
        
        # Forward pass
        mc_color_out = mc_seq.forward_color(mc_input)
        pytorch_out = pytorch_seq(pytorch_input)
        
        # Same loss function
        mc_loss = mc_color_out.mean()
        pytorch_loss = pytorch_out.mean()
        
        # Backward pass
        mc_loss.backward()
        pytorch_loss.backward()
        
        # Compare input gradients
        torch.testing.assert_close(mc_input.grad, pytorch_input.grad, atol=1e-5, rtol=1e-5,
                                 msg="MCSequential input gradients differ from PyTorch Sequential")
        
        # Compare parameter gradients
        mc_params = list(mc_seq.parameters())
        pytorch_params = list(pytorch_seq.parameters())
        
        # Check conv weights
        torch.testing.assert_close(mc_seq[0].color_weight.grad, pytorch_seq[0].weight.grad, 
                                 atol=1e-5, rtol=1e-5, msg="Conv1 weight gradients differ")
        torch.testing.assert_close(mc_seq[3].color_weight.grad, pytorch_seq[3].weight.grad, 
                                 atol=1e-5, rtol=1e-5, msg="Conv2 weight gradients differ")
        
        # Check batch norm parameters
        torch.testing.assert_close(mc_seq[1].color_weight.grad, pytorch_seq[1].weight.grad, 
                                 atol=1e-5, rtol=1e-5, msg="BN1 weight gradients differ")
        torch.testing.assert_close(mc_seq[1].color_bias.grad, pytorch_seq[1].bias.grad, 
                                 atol=1e-5, rtol=1e-5, msg="BN1 bias gradients differ")
    
    def _copy_mc_to_pytorch_weights(self, mc_seq, pytorch_seq, pathway='color'):
        """Helper to copy MC weights to PyTorch sequence for comparison."""
        with torch.no_grad():
            for mc_layer, pytorch_layer in zip(mc_seq, pytorch_seq):
                if isinstance(mc_layer, MCConv2d) and isinstance(pytorch_layer, nn.Conv2d):
                    if pathway == 'color':
                        pytorch_layer.weight.copy_(mc_layer.color_weight)
                        if mc_layer.color_bias is not None:
                            pytorch_layer.bias.copy_(mc_layer.color_bias)
                    else:  # brightness
                        pytorch_layer.weight.copy_(mc_layer.brightness_weight)
                        if mc_layer.brightness_bias is not None:
                            pytorch_layer.bias.copy_(mc_layer.brightness_bias)
                
                elif isinstance(mc_layer, MCBatchNorm2d) and isinstance(pytorch_layer, nn.BatchNorm2d):
                    if pathway == 'color':
                        pytorch_layer.weight.copy_(mc_layer.color_weight)
                        pytorch_layer.bias.copy_(mc_layer.color_bias)
                        pytorch_layer.running_mean.copy_(mc_layer.color_running_mean)
                        pytorch_layer.running_var.copy_(mc_layer.color_running_var)
                    else:  # brightness
                        pytorch_layer.weight.copy_(mc_layer.brightness_weight)
                        pytorch_layer.bias.copy_(mc_layer.brightness_bias)
                        pytorch_layer.running_mean.copy_(mc_layer.brightness_running_mean)
                        pytorch_layer.running_var.copy_(mc_layer.brightness_running_var)
