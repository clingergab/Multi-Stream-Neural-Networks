"""
Unit tests for Multi-Channel container modules.
"""

import unittest
import torch
import torch.nn as nn
import sys
from pathlib import Path
from collections import OrderedDict

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent / "src"))

from src.models2.multi_channel.container import MCSequential
from src.models2.multi_channel.conv import MCConv2d, MCBatchNorm2d
from src.models2.multi_channel.blocks import MCBasicBlock


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


if __name__ == "__main__":
    unittest.main()
