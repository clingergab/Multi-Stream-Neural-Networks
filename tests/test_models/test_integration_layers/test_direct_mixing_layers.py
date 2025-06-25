"""Tests for direct mixing layers."""

import unittest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "src"))

from models.layers.integration_layers.direct_mixing_layers import (
    ScalarMixingLayer, ChannelWiseMixingLayer, DynamicMixingLayer, SpatialMixingLayer
)


class TestDirectMixingLayers(unittest.TestCase):
    """Test cases for direct mixing layers."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 4
        self.channels = 64
        self.height = 32
        self.width = 32
        
        self.color_features = torch.randn(self.batch_size, self.channels, self.height, self.width)
        self.brightness_features = torch.randn(self.batch_size, self.channels, self.height, self.width)
    
    def test_scalar_mixing_layer(self):
        """Test scalar mixing layer."""
        layer = ScalarMixingLayer()
        
        output = layer(self.color_features, self.brightness_features)
        
        # Check output shape
        expected_shape = (self.batch_size, self.channels, self.height, self.width)
        self.assertEqual(output.shape, expected_shape)
        
        # Check parameters exist
        self.assertTrue(hasattr(layer, 'alpha'))
        self.assertTrue(hasattr(layer, 'beta'))
        self.assertTrue(hasattr(layer, 'gamma'))
    
    def test_channel_wise_mixing_layer(self):
        """Test channel-wise mixing layer."""
        layer = ChannelWiseMixingLayer(self.channels)
        
        output = layer(self.color_features, self.brightness_features)
        
        # Check output shape
        expected_shape = (self.batch_size, self.channels, self.height, self.width)
        self.assertEqual(output.shape, expected_shape)
        
        # Check parameter shapes
        self.assertEqual(layer.alpha.shape, (self.channels,))
        self.assertEqual(layer.beta.shape, (self.channels,))
        self.assertEqual(layer.gamma.shape, (self.channels,))
    
    def test_dynamic_mixing_layer(self):
        """Test dynamic mixing layer."""
        layer = DynamicMixingLayer(self.channels)
        
        output = layer(self.color_features, self.brightness_features)
        
        # Check output shape
        expected_shape = (self.batch_size, self.channels, self.height, self.width)
        self.assertEqual(output.shape, expected_shape)
        
        # Check weight network exists
        self.assertTrue(hasattr(layer, 'weight_network'))
    
    def test_spatial_mixing_layer(self):
        """Test spatial mixing layer."""
        layer = SpatialMixingLayer(self.channels)
        
        output = layer(self.color_features, self.brightness_features)
        
        # Check output shape
        expected_shape = (self.batch_size, self.channels, self.height, self.width)
        self.assertEqual(output.shape, expected_shape)
        
        # Check attention networks exist
        self.assertTrue(hasattr(layer, 'alpha_attention'))
        self.assertTrue(hasattr(layer, 'beta_attention'))
        self.assertTrue(hasattr(layer, 'gamma_attention'))


if __name__ == '__main__':
    unittest.main()