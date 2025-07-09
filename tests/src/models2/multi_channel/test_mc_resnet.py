"""
Unit tests for Multi-Channel ResNet models.
"""

import unittest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent / "src"))

from src.models2.multi_channel.mc_resnet import MCResNet
from src.models2.multi_channel.blocks import MCBasicBlock, MCBottleneck


class TestMCResNet(unittest.TestCase):
    """Test cases for Multi-Channel ResNet models."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a minimal MCResNet model for testing
        self.model = MCResNet(
            block=MCBasicBlock,
            layers=[2, 2, 2, 2],
            num_classes=10,
            device='cpu'  # Explicit device for consistent test behavior
        )
    
    def test_init(self):
        """Test model initialization."""
        # Check that model is an instance of MCResNet
        self.assertIsInstance(self.model, MCResNet)
        
        # Check model structure - only test runtime-relevant attributes
        self.assertEqual(self.model.num_classes, 10)  # num_classes is kept as instance variable
        
        # Check MCResNet-specific attributes
        self.assertEqual(self.model.color_input_channels, 3)
        self.assertEqual(self.model.brightness_input_channels, 1)
        
        # Verify that the network was built correctly (layers exist)
        self.assertTrue(hasattr(self.model, 'conv1'))
        self.assertTrue(hasattr(self.model, 'layer1'))
        self.assertTrue(hasattr(self.model, 'layer2'))
        self.assertTrue(hasattr(self.model, 'layer3'))
        self.assertTrue(hasattr(self.model, 'layer4'))
        self.assertTrue(hasattr(self.model, 'fc'))
    
    @unittest.skip("Implementation pending")
    def test_build_network(self):
        """Test network building functionality."""
        # This test should be implemented once the _build_network method is completed
        pass
    
    @unittest.skip("Implementation pending")
    def test_initialize_weights(self):
        """Test weight initialization."""
        # This test should be implemented once the _initialize_weights method is completed
        pass
    
    @unittest.skip("Implementation pending")
    def test_forward_pass(self):
        """Test forward pass with multi-channel input."""
        # Setup input data
        batch_size = 4
        channels = 3  # RGB
        input_size = 224
        x = torch.randn(batch_size, channels, input_size, input_size)
        
        # Test forward pass
        # output = self.model(x)
        
        # Check output shape
        # expected_shape = (batch_size, self.model.num_classes)
        # self.assertEqual(output.shape, expected_shape)
        pass
    
    @unittest.skip("Implementation pending")
    def test_different_fusion_types(self):
        """Test model with different fusion types."""
        # Note: MCResNet doesn't currently have configurable fusion types
        # This test is skipped until that feature is implemented
        pass
    
    @unittest.skip("Implementation pending")
    def test_bottleneck_block(self):
        """Test MCResNet with Bottleneck blocks."""
        model = MCResNet(
            block=MCBottleneck,
            layers=[3, 4, 6, 3],  # ResNet-50 configuration
            num_classes=10
        )
        
        # Check that the network was built correctly - test network structure instead of stored parameters
        self.assertTrue(hasattr(model, 'conv1'))
        self.assertTrue(hasattr(model, 'layer1'))
        
        # Test forward pass once implemented
        pass
    
    def test_compile_with_standard_losses_only(self):
        """Test that MCResNet supports standard loss functions but not multi_stream."""
        # Test cross_entropy
        try:
            self.model.compile(optimizer='adam', loss='cross_entropy', device='cpu')
            self.assertIsNotNone(self.model.criterion)
        except Exception as e:
            self.fail(f"MCResNet should support cross_entropy loss but got error: {e}")
        
        # Test focal loss  
        try:
            self.model.compile(optimizer='adam', loss='focal', device='cpu', alpha=1.0, gamma=2.0)
            from src.training.losses import FocalLoss
            self.assertIsInstance(self.model.criterion, FocalLoss)
        except Exception as e:
            self.fail(f"MCResNet should support focal loss but got error: {e}")
        
        # Test that multi_stream is now rejected (since we're keeping MCResNet similar to ResNet)
        with self.assertRaises(ValueError) as context:
            self.model.compile(optimizer='adam', loss='multi_stream', device='cpu')
        self.assertIn("Supported losses: 'cross_entropy', 'focal'", str(context.exception))


if __name__ == "__main__":
    unittest.main()
