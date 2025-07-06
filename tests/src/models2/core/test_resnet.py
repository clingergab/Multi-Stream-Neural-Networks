"""
Unit tests for ResNet models.
"""

import unittest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent / "src"))

from src.models2.core.resnet import (
    ResNet, resnet18, resnet34, resnet50, resnet101, resnet152
)
from src.models2.core.blocks import BasicBlock, Bottleneck


class TestResNetBlocks(unittest.TestCase):
    """Test cases for ResNet building blocks."""
    
    def test_basic_block(self):
        """Test the BasicBlock functionality."""
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
                torch.nn.Conv2d(channels, channels * 2, kernel_size=1, stride=2, bias=False),
                torch.nn.BatchNorm2d(channels * 2),
            )
        )
        output_down = block_down(x)
        expected_shape = (batch_size, channels * 2, input_size // 2, input_size // 2)
        self.assertEqual(output_down.shape, expected_shape)
    
    def test_bottleneck_block(self):
        """Test the Bottleneck functionality."""
        # Setup
        batch_size = 4
        channels = 64
        input_size = 32
        x = torch.randn(batch_size, channels, input_size, input_size)
        
        # Test with identity (no downsample)
        block = Bottleneck(channels, channels // 4)  # Expansion is 4
        output = block(x)
        
        # Check shape preserved
        self.assertEqual(output.shape, x.shape)
        
        # Test with downsample (stride=2)
        block_down = Bottleneck(
            channels, 
            channels // 2, 
            stride=2, 
            downsample=torch.nn.Sequential(
                torch.nn.Conv2d(channels, channels * 2, kernel_size=1, stride=2, bias=False),
                torch.nn.BatchNorm2d(channels * 2),
            )
        )
        output_down = block_down(x)
        expected_shape = (batch_size, channels * 2, input_size // 2, input_size // 2)
        self.assertEqual(output_down.shape, expected_shape)
    

class TestResNetModels(unittest.TestCase):
    """Test cases for ResNet models."""
    
    def test_resnet_building(self):
        """Test the ResNet model construction."""
        # Create a mini ResNet
        model = ResNet(
            block=BasicBlock,
            layers=[1, 1, 1, 1],
            num_classes=10
        )
        
        # Check model type
        self.assertIsInstance(model, ResNet)
        
        # Verify layers exist
        self.assertTrue(hasattr(model, 'layer1'))
        self.assertTrue(hasattr(model, 'layer2'))
        self.assertTrue(hasattr(model, 'layer3'))
        self.assertTrue(hasattr(model, 'layer4'))
        self.assertTrue(hasattr(model, 'fc'))
        
        # Check final classifier
        self.assertEqual(model.fc.out_features, 10)
    
    def test_resnet_forward(self):
        """Test the forward pass of ResNet models."""
        # Reduced-size test model
        model = ResNet(
            block=BasicBlock,
            layers=[1, 1, 1, 1],
            num_classes=10
        )
        
        # Test with RGB input
        batch_size = 4
        channels = 3
        input_size = 224
        x = torch.randn(batch_size, channels, input_size, input_size)
        
        output = model(x)
        expected_shape = (batch_size, 10)  # Output is class probabilities
        self.assertEqual(output.shape, expected_shape)
    
    def test_resnet_variants(self):
        """Test different ResNet variants."""
        # Create small test input
        batch_size = 1
        channels = 3
        input_size = 64  # Small size for testing
        x = torch.randn(batch_size, channels, input_size, input_size)
        
        # List of models to test with output size
        models = [
            (resnet18(num_classes=10), (batch_size, 10)),
            (resnet34(num_classes=10), (batch_size, 10)),
            (resnet50(num_classes=10), (batch_size, 10)),
        ]
        
        # Test each model
        for model, expected_shape in models:
            # Switch to eval mode for inference
            model.eval()
            with torch.no_grad():
                output = model(x)
                self.assertEqual(output.shape, expected_shape)
    
    def test_model_initialization(self):
        """Test model weight initialization."""
        model = resnet18(num_classes=10)
        
        # Check initialization of different layer types
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                # Conv layers should have been initialized with kaiming
                self.assertFalse(torch.allclose(m.weight, torch.zeros_like(m.weight)))
            elif isinstance(m, torch.nn.BatchNorm2d):
                # BatchNorm weight should be 1, bias should be 0
                self.assertTrue(torch.allclose(m.weight, torch.ones_like(m.weight)))
                self.assertTrue(torch.allclose(m.bias, torch.zeros_like(m.bias)))


if __name__ == "__main__":
    unittest.main()
