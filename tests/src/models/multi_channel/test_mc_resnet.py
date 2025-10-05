"""
Unit tests for Multi-Channel ResNet models.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import patch, MagicMock

from models.multi_channel.mc_resnet import MCResNet
from models.multi_channel.blocks import MCBasicBlock, MCBottleneck
from data_utils.dual_channel_dataset import create_dual_channel_dataloader


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
    
    def test_build_network(self):
        """Test network building functionality."""
        # Test that all required layers are built
        self.assertTrue(hasattr(self.model, 'conv1'))
        self.assertTrue(hasattr(self.model, 'bn1'))
        self.assertTrue(hasattr(self.model, 'relu'))
        self.assertTrue(hasattr(self.model, 'maxpool'))
        self.assertTrue(hasattr(self.model, 'layer1'))
        self.assertTrue(hasattr(self.model, 'layer2'))
        self.assertTrue(hasattr(self.model, 'layer3'))
        self.assertTrue(hasattr(self.model, 'layer4'))
        self.assertTrue(hasattr(self.model, 'avgpool'))
        self.assertTrue(hasattr(self.model, 'fc'))
        
        # Test layer types
        from models.multi_channel.conv import MCConv2d, MCBatchNorm2d
        from models.multi_channel.pooling import MCMaxPool2d, MCAdaptiveAvgPool2d
        from models.multi_channel.container import MCSequential
        
        self.assertIsInstance(self.model.conv1, MCConv2d)
        self.assertIsInstance(self.model.bn1, MCBatchNorm2d)
        self.assertIsInstance(self.model.maxpool, MCMaxPool2d)
        self.assertIsInstance(self.model.avgpool, MCAdaptiveAvgPool2d)
        
        # Test that layers are MCSequential containers
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            layer = getattr(self.model, layer_name)
            self.assertIsInstance(layer, MCSequential)
    
    def test_initialize_weights(self):
        """Test weight initialization."""
        # Create a new model to test weight initialization
        model = MCResNet(
            block=MCBasicBlock,
            layers=[1, 1, 1, 1],
            num_classes=10,
            device='cpu'
        )
        
        # Check that weights are initialized (not all zeros)
        conv_weights = []
        bn_weights = []
        
        for name, param in model.named_parameters():
            if 'conv' in name and 'weight' in name:
                conv_weights.append(param)
            elif 'bn' in name and 'weight' in name:
                bn_weights.append(param)
        
        # Check that we have conv and bn weights
        self.assertGreater(len(conv_weights), 0)
        self.assertGreater(len(bn_weights), 0)
        
        # Check that weights are not all zeros
        for weight in conv_weights:
            self.assertFalse(torch.allclose(weight, torch.zeros_like(weight)))
        
        # Check that batch norm weights are initialized to 1
        for weight in bn_weights:
            # Most BN weights should be close to 1 after initialization
            self.assertTrue(torch.allclose(weight, torch.ones_like(weight), atol=1e-3))
    
    def test_forward_pass(self):
        """Test forward pass with multi-channel input."""
        batch_size = 4
        color_input = torch.randn(batch_size, 3, 32, 32)
        brightness_input = torch.randn(batch_size, 1, 32, 32)
        
        # Test forward pass
        output = self.model(color_input, brightness_input)
        
        # Check output shape
        expected_shape = (batch_size, self.model.num_classes)
        self.assertEqual(output.shape, expected_shape)
        
        # Check that output is a tensor
        self.assertIsInstance(output, torch.Tensor)
        
        # Check that gradients can flow
        loss = output.sum()
        loss.backward()
        
        # Check that some parameters have gradients
        has_gradients = any(p.grad is not None for p in self.model.parameters())
        self.assertTrue(has_gradients)
    
    @unittest.skip("Implementation pending")
    def test_different_fusion_types(self):
        """Test model with different fusion types."""
        # Note: MCResNet doesn't currently have configurable fusion types
        # This test is skipped until that feature is implemented
        pass
    
    def test_bottleneck_block(self):
        """Test MCResNet with Bottleneck blocks."""
        model = MCResNet(
            block=MCBottleneck,
            layers=[3, 4, 6, 3],  # ResNet-50 configuration
            num_classes=10,
            device='cpu'
        )
        
        # Check that the network was built correctly
        self.assertTrue(hasattr(model, 'conv1'))
        self.assertTrue(hasattr(model, 'layer1'))
        
        # Test forward pass
        color_input = torch.randn(2, 3, 32, 32)
        brightness_input = torch.randn(2, 1, 32, 32)
        
        output = model(color_input, brightness_input)
        expected_shape = (2, 10)
        self.assertEqual(output.shape, expected_shape)
        
        # Check that the first layer contains MCBottleneck blocks
        first_block = model.layer1[0]
        self.assertIsInstance(first_block, MCBottleneck)
    
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
            from training.losses import FocalLoss
            self.assertIsInstance(self.model.criterion, FocalLoss)
        except Exception as e:
            self.fail(f"MCResNet should support focal loss but got error: {e}")
        
        # Test that multi_stream is now rejected (since we're keeping MCResNet similar to ResNet)
        with self.assertRaises(ValueError) as context:
            self.model.compile(optimizer='adam', loss='multi_stream', device='cpu')
        self.assertIn("Supported losses: 'cross_entropy', 'focal'", str(context.exception))

    def _create_sample_data(self, batch_size=8, input_size=32, num_classes=10):
        """Helper method to create sample data for testing analysis methods."""
        color_data = torch.randn(batch_size, 3, input_size, input_size)
        brightness_data = torch.randn(batch_size, 1, input_size, input_size)
        targets = torch.randint(0, num_classes, (batch_size,))
        
        return color_data, brightness_data, targets
    
    def _setup_compiled_model(self):
        """Helper method to setup a compiled model for testing."""
        model = MCResNet(
            block=MCBasicBlock,
            layers=[1, 1, 1, 1],  # Minimal layers for faster testing
            num_classes=10,
            device='cpu'
        )
        model.compile(optimizer='adam', loss='cross_entropy', device='cpu')
        return model


class TestMCResNetAnalysis(unittest.TestCase):
    """Test cases for MCResNet analysis methods."""
    
    def setUp(self):
        """Set up test fixtures for analysis methods."""
        self.model = MCResNet(
            block=MCBasicBlock,
            layers=[1, 1, 1, 1],  # Minimal layers for faster testing
            num_classes=10,
            device='cpu'
        )
        self.model.compile(optimizer='adam', loss='cross_entropy', device='cpu')
        
        # Create sample data
        self.batch_size = 8
        self.input_size = 32
        self.num_classes = 10
        self.color_data = torch.randn(self.batch_size, 3, self.input_size, self.input_size)
        self.brightness_data = torch.randn(self.batch_size, 1, self.input_size, self.input_size) 
        self.targets = torch.randint(0, self.num_classes, (self.batch_size,))
        
        # Create data loader
        self.data_loader = create_dual_channel_dataloader(
            self.color_data, self.brightness_data, self.targets,
            batch_size=4, num_workers=0
        )
    
    def test_analyze_pathways_with_dataloader(self):
        """Test analyze_pathways method with DataLoader input."""
        result = self.model.analyze_pathways(
            data_loader=self.data_loader,
            num_samples=self.batch_size
        )
        
        # Check structure of returned dictionary
        self.assertIn('accuracy', result)
        self.assertIn('loss', result)
        self.assertIn('feature_norms', result)
        self.assertIn('samples_analyzed', result)
        
        # Check accuracy metrics
        accuracy_dict = result['accuracy']
        self.assertIn('full_model', accuracy_dict)
        self.assertIn('color_only', accuracy_dict)
        self.assertIn('brightness_only', accuracy_dict)
        self.assertIn('color_contribution', accuracy_dict)
        self.assertIn('brightness_contribution', accuracy_dict)
        
        # Check that accuracies are in valid range [0, 1]
        for key in ['full_model', 'color_only', 'brightness_only']:
            self.assertGreaterEqual(accuracy_dict[key], 0.0)
            self.assertLessEqual(accuracy_dict[key], 1.0)
        
        # Check loss metrics
        loss_dict = result['loss']
        self.assertIn('full_model', loss_dict)
        self.assertIn('color_only', loss_dict)
        self.assertIn('brightness_only', loss_dict)
        
        # Check feature norms
        feature_norms = result['feature_norms']
        self.assertIn('color_mean', feature_norms)
        self.assertIn('color_std', feature_norms)
        self.assertIn('brightness_mean', feature_norms)
        self.assertIn('brightness_std', feature_norms)
        self.assertIn('color_to_brightness_ratio', feature_norms)
        
        # Check that feature norms are positive
        self.assertGreater(feature_norms['color_mean'], 0)
        self.assertGreater(feature_norms['brightness_mean'], 0)
        self.assertGreater(feature_norms['color_std'], 0)
        self.assertGreater(feature_norms['brightness_std'], 0)
        
        # Check samples analyzed
        self.assertEqual(result['samples_analyzed'], self.batch_size)
    
    def test_analyze_pathways_with_tensors(self):
        """Test analyze_pathways method with tensor inputs."""
        result = self.model.analyze_pathways(
            color_data=self.color_data,
            brightness_data=self.brightness_data,
            targets=self.targets,
            batch_size=4,
            num_samples=self.batch_size
        )
        
        # Basic structure checks
        self.assertIn('accuracy', result)
        self.assertIn('loss', result)
        self.assertIn('feature_norms', result)
        self.assertIn('samples_analyzed', result)
        
        # Check that we get valid results
        self.assertEqual(result['samples_analyzed'], self.batch_size)
        self.assertIsInstance(result['accuracy']['full_model'], float)
        self.assertIsInstance(result['loss']['full_model'], float)
    
    def test_analyze_pathways_invalid_input(self):
        """Test analyze_pathways with invalid input combinations."""
        # Test with no data provided
        with self.assertRaises(ValueError) as context:
            self.model.analyze_pathways()
        self.assertIn("Either provide data_loader or all of color_data, brightness_data, and targets", 
                     str(context.exception))
        
        # Test with incomplete tensor data
        with self.assertRaises(ValueError) as context:
            self.model.analyze_pathways(color_data=self.color_data)
        self.assertIn("Either provide data_loader or all of color_data, brightness_data, and targets", 
                     str(context.exception))
        
        # Test with uncompiled model
        uncompiled_model = MCResNet(
            block=MCBasicBlock,
            layers=[1, 1, 1, 1],
            num_classes=10,
            device='cpu'
        )
        with self.assertRaises(ValueError) as context:
            uncompiled_model.analyze_pathways(data_loader=self.data_loader)
        self.assertIn("Model not compiled. Call compile() before analyze_pathways()", 
                     str(context.exception))
    
    def test_analyze_pathways_sample_limiting(self):
        """Test that analyze_pathways properly limits samples for efficiency."""
        # Create larger dataset
        large_color_data = torch.randn(200, 3, self.input_size, self.input_size)
        large_brightness_data = torch.randn(200, 1, self.input_size, self.input_size)
        large_targets = torch.randint(0, self.num_classes, (200,))
        
        result = self.model.analyze_pathways(
            color_data=large_color_data,
            brightness_data=large_brightness_data,
            targets=large_targets,
            batch_size=4,
            num_samples=50  # Should limit to 50 samples
        )
        
        # Should analyze exactly 50 samples, not 200
        self.assertEqual(result['samples_analyzed'], 50)
    
    def test_analyze_pathway_weights(self):
        """Test analyze_pathway_weights method."""
        result = self.model.analyze_pathway_weights()
        
        # Check structure of returned dictionary
        self.assertIn('color_pathway', result)
        self.assertIn('brightness_pathway', result)
        self.assertIn('ratio_analysis', result)
        
        # Check color pathway analysis
        color_pathway = result['color_pathway']
        self.assertIn('layer_weights', color_pathway)
        self.assertIn('total_norm', color_pathway)
        self.assertIn('mean_norm', color_pathway)
        self.assertIn('num_layers', color_pathway)
        
        # Check brightness pathway analysis
        brightness_pathway = result['brightness_pathway']
        self.assertIn('layer_weights', brightness_pathway)
        self.assertIn('total_norm', brightness_pathway)
        self.assertIn('mean_norm', brightness_pathway)
        self.assertIn('num_layers', brightness_pathway)
        
        # Check ratio analysis
        ratio_analysis = result['ratio_analysis']
        self.assertIn('color_to_brightness_norm_ratio', ratio_analysis)
        self.assertIn('layer_ratios', ratio_analysis)
        
        # Check that norms are non-negative (may be 0 if no multi-channel layers found)
        self.assertGreaterEqual(color_pathway['total_norm'], 0)
        self.assertGreaterEqual(brightness_pathway['total_norm'], 0)
        self.assertGreaterEqual(color_pathway['mean_norm'], 0)
        self.assertGreaterEqual(brightness_pathway['mean_norm'], 0)
        
        # Check that we have equal number of layers for both pathways
        self.assertEqual(color_pathway['num_layers'], brightness_pathway['num_layers'])
        
        # Check layer weights structure if layers exist
        if color_pathway['layer_weights']:
            # Get first layer weight info
            first_layer = next(iter(color_pathway['layer_weights'].values()))
            self.assertIn('mean', first_layer)
            self.assertIn('std', first_layer)
            self.assertIn('norm', first_layer)
            self.assertIn('shape', first_layer)
            
            # Check that shape is a list
            self.assertIsInstance(first_layer['shape'], list)
            self.assertGreater(len(first_layer['shape']), 0)
    
    def test_get_pathway_importance_gradient_method(self):
        """Test get_pathway_importance with gradient method."""
        result = self.model.get_pathway_importance(
            data_loader=self.data_loader,
            method='gradient'
        )
        
        # Check structure
        self.assertIn('method', result)
        self.assertIn('color_importance', result)
        self.assertIn('brightness_importance', result)
        self.assertIn('raw_gradients', result)
        
        # Check method
        self.assertEqual(result['method'], 'gradient')
        
        # Check that importance values are in [0, 1] and sum to 1
        color_imp = result['color_importance']
        brightness_imp = result['brightness_importance']
        self.assertGreaterEqual(color_imp, 0.0)
        self.assertLessEqual(color_imp, 1.0)
        self.assertGreaterEqual(brightness_imp, 0.0)
        self.assertLessEqual(brightness_imp, 1.0)
        self.assertAlmostEqual(color_imp + brightness_imp, 1.0, places=5)
        
        # Check raw gradients
        raw_grads = result['raw_gradients']
        self.assertIn('color_avg', raw_grads)
        self.assertIn('brightness_avg', raw_grads)
        self.assertGreaterEqual(raw_grads['color_avg'], 0)
        self.assertGreaterEqual(raw_grads['brightness_avg'], 0)
    
    def test_get_pathway_importance_ablation_method(self):
        """Test get_pathway_importance with ablation method."""
        result = self.model.get_pathway_importance(
            color_data=self.color_data,
            brightness_data=self.brightness_data,
            targets=self.targets,
            method='ablation'
        )
        
        # Check structure
        self.assertIn('method', result)
        self.assertIn('color_importance', result)
        self.assertIn('brightness_importance', result)
        self.assertIn('performance_drops', result)
        self.assertIn('individual_accuracies', result)
        
        # Check method
        self.assertEqual(result['method'], 'ablation')
        
        # Check importance values
        color_imp = result['color_importance']
        brightness_imp = result['brightness_importance']
        self.assertGreaterEqual(color_imp, 0.0)
        self.assertLessEqual(color_imp, 1.0)
        self.assertGreaterEqual(brightness_imp, 0.0)
        self.assertLessEqual(brightness_imp, 1.0)
        
        # Check performance drops
        perf_drops = result['performance_drops']
        self.assertIn('without_color', perf_drops)
        self.assertIn('without_brightness', perf_drops)
        self.assertGreaterEqual(perf_drops['without_color'], 0)
        self.assertGreaterEqual(perf_drops['without_brightness'], 0)
        
        # Check individual accuracies
        accuracies = result['individual_accuracies']
        self.assertIn('full_model', accuracies)
        self.assertIn('color_only', accuracies)
        self.assertIn('brightness_only', accuracies)
        for acc in accuracies.values():
            self.assertGreaterEqual(acc, 0.0)
            self.assertLessEqual(acc, 1.0)
    
    def test_get_pathway_importance_feature_norm_method(self):
        """Test get_pathway_importance with feature_norm method."""
        result = self.model.get_pathway_importance(
            data_loader=self.data_loader,
            method='feature_norm'
        )
        
        # Check structure
        self.assertIn('method', result)
        self.assertIn('color_importance', result)
        self.assertIn('brightness_importance', result)
        self.assertIn('feature_norms', result)
        
        # Check method
        self.assertEqual(result['method'], 'feature_norm')
        
        # Check importance values
        color_imp = result['color_importance']
        brightness_imp = result['brightness_importance']
        self.assertGreaterEqual(color_imp, 0.0)
        self.assertLessEqual(color_imp, 1.0)
        self.assertGreaterEqual(brightness_imp, 0.0)
        self.assertLessEqual(brightness_imp, 1.0)
        self.assertAlmostEqual(color_imp + brightness_imp, 1.0, places=5)
        
        # Check feature norms
        feature_norms = result['feature_norms']
        self.assertIn('color_mean', feature_norms)
        self.assertIn('brightness_mean', feature_norms)
        self.assertIn('ratio', feature_norms)
        self.assertGreater(feature_norms['color_mean'], 0)
        self.assertGreater(feature_norms['brightness_mean'], 0)
    
    def test_get_pathway_importance_invalid_method(self):
        """Test get_pathway_importance with invalid method."""
        with self.assertRaises(ValueError) as context:
            self.model.get_pathway_importance(
                data_loader=self.data_loader,
                method='invalid_method'
            )
        self.assertIn("Unknown importance method: invalid_method", str(context.exception))
        self.assertIn("Choose from 'gradient', 'ablation', 'feature_norm'", str(context.exception))
    
    def test_get_pathway_importance_uncompiled_model(self):
        """Test get_pathway_importance with uncompiled model for gradient method."""
        uncompiled_model = MCResNet(
            block=MCBasicBlock,
            layers=[1, 1, 1, 1],
            num_classes=10,
            device='cpu'
        )
        
        # Gradient method should fail without compiled model
        with self.assertRaises(ValueError) as context:
            uncompiled_model.get_pathway_importance(
                data_loader=self.data_loader,
                method='gradient'
            )
        self.assertIn("Model not compiled. Call compile() before calculating importance", 
                     str(context.exception))
    
    def test_forward_color_pathway(self):
        """Test _forward_color_pathway method."""
        color_input = torch.randn(2, 3, 32, 32)
        
        # Test that method exists and returns tensor
        result = self.model._forward_color_pathway(color_input)
        
        # Check that result is a tensor
        self.assertIsInstance(result, torch.Tensor)
        
        # Check that result is flattened (2D: batch_size x features)
        self.assertEqual(len(result.shape), 2)
        self.assertEqual(result.shape[0], 2)  # batch size
        
        # Check that features dimension is reasonable
        self.assertGreater(result.shape[1], 0)
    
    def test_forward_brightness_pathway(self):
        """Test _forward_brightness_pathway method."""
        brightness_input = torch.randn(2, 1, 32, 32)
        
        # Test that method exists and returns tensor
        result = self.model._forward_brightness_pathway(brightness_input)
        
        # Check that result is a tensor
        self.assertIsInstance(result, torch.Tensor)
        
        # Check that result is flattened (2D: batch_size x features)
        self.assertEqual(len(result.shape), 2)
        self.assertEqual(result.shape[0], 2)  # batch size
        
        # Check that features dimension is reasonable
        self.assertGreater(result.shape[1], 0)
    
    def test_pathway_analysis_consistency(self):
        """Test that pathway analysis methods give consistent results."""
        # Run analyze_pathways
        pathway_result = self.model.analyze_pathways(
            data_loader=self.data_loader,
            num_samples=self.batch_size
        )
        
        # Run get_pathway_importance with feature_norm method
        importance_result = self.model.get_pathway_importance(
            data_loader=self.data_loader,
            method='feature_norm'
        )
        
        # The feature norms should be consistent between methods
        pathway_color_norm = pathway_result['feature_norms']['color_mean']
        pathway_brightness_norm = pathway_result['feature_norms']['brightness_mean']
        
        importance_color_norm = importance_result['feature_norms']['color_mean']
        importance_brightness_norm = importance_result['feature_norms']['brightness_mean']
        
        # Should be approximately equal (allowing for small floating point differences)
        self.assertAlmostEqual(pathway_color_norm, importance_color_norm, places=4)
        self.assertAlmostEqual(pathway_brightness_norm, importance_brightness_norm, places=4)
    
    def test_analysis_with_different_input_sizes(self):
        """Test analysis methods work with different input sizes."""
        # Test with smaller input
        small_color = torch.randn(4, 3, 16, 16)
        small_brightness = torch.randn(4, 1, 16, 16)
        small_targets = torch.randint(0, 10, (4,))
        
        # Should work without errors
        result = self.model.analyze_pathways(
            color_data=small_color,
            brightness_data=small_brightness,
            targets=small_targets
        )
        
        self.assertIn('accuracy', result)
        self.assertEqual(result['samples_analyzed'], 4)
    
    def test_analysis_with_single_batch(self):
        """Test analysis methods work with single batch."""
        single_color = torch.randn(1, 3, 32, 32)
        single_brightness = torch.randn(1, 1, 32, 32)
        single_targets = torch.randint(0, 10, (1,))
        
        # Should work without errors
        result = self.model.get_pathway_importance(
            color_data=single_color,
            brightness_data=single_brightness,
            targets=single_targets,
            method='ablation'
        )
        
        self.assertIn('method', result)
        self.assertEqual(result['method'], 'ablation')


class TestMCResNetTrainingAndInference(unittest.TestCase):
    """Test cases for MCResNet training and inference methods."""
    
    def setUp(self):
        """Set up test fixtures for training and inference methods."""
        self.model = MCResNet(
            block=MCBasicBlock,
            layers=[1, 1, 1, 1],  # Minimal layers for faster testing
            num_classes=10,
            device='cpu'
        )
        self.model.compile(optimizer='adam', loss='cross_entropy', device='cpu')
        
        # Create sample data
        self.batch_size = 8
        self.input_size = 32
        self.num_classes = 10
        self.color_data = torch.randn(self.batch_size, 3, self.input_size, self.input_size)
        self.brightness_data = torch.randn(self.batch_size, 1, self.input_size, self.input_size) 
        self.targets = torch.randint(0, self.num_classes, (self.batch_size,))
        
        # Create data loader
        self.data_loader = create_dual_channel_dataloader(
            self.color_data, self.brightness_data, self.targets,
            batch_size=4, num_workers=0
        )
        
        # Create validation data
        self.val_color_data = torch.randn(4, 3, self.input_size, self.input_size)
        self.val_brightness_data = torch.randn(4, 1, self.input_size, self.input_size)
        self.val_targets = torch.randint(0, self.num_classes, (4,))
        self.val_data_loader = create_dual_channel_dataloader(
            self.val_color_data, self.val_brightness_data, self.val_targets,
            batch_size=4, num_workers=0
        )
    
    def test_fit_with_dataloader(self):
        """Test fit method with DataLoader input."""
        history = self.model.fit(
            train_loader=self.data_loader,
            val_loader=self.val_data_loader,
            epochs=2,
            verbose=False
        )
        
        # Check history structure
        self.assertIn('train_loss', history)
        self.assertIn('val_loss', history)
        self.assertIn('train_accuracy', history)
        self.assertIn('val_accuracy', history)
        self.assertIn('learning_rates', history)
        
        # Check that we have the right number of epochs
        self.assertEqual(len(history['train_loss']), 2)
        self.assertEqual(len(history['val_loss']), 2)
        self.assertEqual(len(history['train_accuracy']), 2)
        self.assertEqual(len(history['val_accuracy']), 2)
        
        # Check that values are reasonable
        for loss in history['train_loss']:
            self.assertGreater(loss, 0)
        for acc in history['train_accuracy']:
            self.assertGreaterEqual(acc, 0.0)
            self.assertLessEqual(acc, 1.0)
    
    def test_fit_with_tensors(self):
        """Test fit method with tensor inputs."""
        history = self.model.fit(
            train_color_data=self.color_data,
            train_brightness_data=self.brightness_data,
            train_targets=self.targets,
            val_color_data=self.val_color_data,
            val_brightness_data=self.val_brightness_data,
            val_targets=self.val_targets,
            epochs=2,
            batch_size=4,
            verbose=False
        )
        
        # Check basic structure
        self.assertIn('train_loss', history)
        self.assertIn('val_loss', history)
        self.assertEqual(len(history['train_loss']), 2)
    
    def test_fit_without_validation(self):
        """Test fit method without validation data."""
        history = self.model.fit(
            train_loader=self.data_loader,
            epochs=2,
            verbose=False
        )
        
        # Should still have training metrics
        self.assertIn('train_loss', history)
        self.assertIn('train_accuracy', history)
        self.assertEqual(len(history['train_loss']), 2)
        
        # Validation metrics should be empty lists
        self.assertEqual(len(history['val_loss']), 0)
        self.assertEqual(len(history['val_accuracy']), 0)
    
    def test_fit_early_stopping(self):
        """Test fit method with early stopping."""
        history = self.model.fit(
            train_loader=self.data_loader,
            val_loader=self.val_data_loader,
            epochs=10,  # High number to potentially trigger early stopping
            verbose=False,
            early_stopping=True,
            patience=2,
            min_delta=0.001
        )
        
        # Check that early stopping info is in history
        self.assertIn('early_stopping', history)
        early_stopping_info = history['early_stopping']
        
        self.assertIn('stopped_early', early_stopping_info)
        self.assertIn('best_epoch', early_stopping_info)
        self.assertIn('best_metric', early_stopping_info)
        self.assertIn('monitor', early_stopping_info)
        self.assertIn('patience', early_stopping_info)
        self.assertIn('min_delta', early_stopping_info)
        
        # Check that we didn't train for all 10 epochs if early stopping triggered
        if early_stopping_info['stopped_early']:
            self.assertLess(len(history['train_loss']), 10)
    
    def test_fit_invalid_input(self):
        """Test fit method with invalid inputs."""
        # Test with uncompiled model
        uncompiled_model = MCResNet(
            block=MCBasicBlock,
            layers=[1, 1, 1, 1],
            num_classes=10,
            device='cpu'
        )
        
        with self.assertRaises(ValueError):
            uncompiled_model.fit(train_loader=self.data_loader, epochs=1)
    
    def test_evaluate_with_dataloader(self):
        """Test evaluate method with DataLoader input."""
        result = self.model.evaluate(data_loader=self.data_loader)
        
        # Check structure
        self.assertIn('loss', result)
        self.assertIn('accuracy', result)
        
        # Check that values are reasonable
        self.assertGreater(result['loss'], 0)
        self.assertGreaterEqual(result['accuracy'], 0.0)
        self.assertLessEqual(result['accuracy'], 1.0)
    
    def test_evaluate_with_tensors(self):
        """Test evaluate method with tensor inputs."""
        result = self.model.evaluate(
            color_data=self.color_data,
            brightness_data=self.brightness_data,
            targets=self.targets,
            batch_size=4
        )
        
        # Check structure
        self.assertIn('loss', result)
        self.assertIn('accuracy', result)
        
        # Check that values are reasonable
        self.assertGreater(result['loss'], 0)
        self.assertGreaterEqual(result['accuracy'], 0.0)
        self.assertLessEqual(result['accuracy'], 1.0)
    
    def test_evaluate_invalid_input(self):
        """Test evaluate method with invalid inputs."""
        # Test with uncompiled model
        uncompiled_model = MCResNet(
            block=MCBasicBlock,
            layers=[1, 1, 1, 1],
            num_classes=10,
            device='cpu'
        )
        
        with self.assertRaises(ValueError):
            uncompiled_model.evaluate(data_loader=self.data_loader)
        
        # Test with incomplete tensor data
        with self.assertRaises(ValueError):
            self.model.evaluate(color_data=self.color_data)
    
    def test_predict_with_dataloader(self):
        """Test predict method with DataLoader input."""
        predictions = self.model.predict(data_loader=self.data_loader)
        
        # Check shape
        expected_shape = (self.batch_size,)
        self.assertEqual(predictions.shape, expected_shape)
        
        # Check that predictions are class indices
        self.assertTrue(torch.all(predictions >= 0))
        self.assertTrue(torch.all(predictions < self.num_classes))
        
        # Check data type
        self.assertTrue(predictions.dtype in [torch.int64, torch.long])
    
    def test_predict_with_tensors(self):
        """Test predict method with tensor inputs."""
        predictions = self.model.predict(
            color_data=self.color_data,
            brightness_data=self.brightness_data,
            batch_size=4
        )
        
        # Check shape
        expected_shape = (self.batch_size,)
        self.assertEqual(predictions.shape, expected_shape)
        
        # Check that predictions are valid class indices
        self.assertTrue(torch.all(predictions >= 0))
        self.assertTrue(torch.all(predictions < self.num_classes))
    
    def test_predict_invalid_input(self):
        """Test predict method with invalid inputs."""
        # Test with incomplete tensor data
        with self.assertRaises(ValueError) as context:
            self.model.predict(color_data=self.color_data)
        self.assertIn("Either provide data_loader or both color_data and brightness_data", 
                     str(context.exception))
    
    def test_predict_proba_with_dataloader(self):
        """Test predict_proba method with DataLoader input."""
        probabilities = self.model.predict_proba(data_loader=self.data_loader)
        
        # Check shape
        expected_shape = (self.batch_size, self.num_classes)
        self.assertEqual(probabilities.shape, expected_shape)
        
        # Check that probabilities sum to 1
        prob_sums = probabilities.sum(axis=1)
        np.testing.assert_allclose(prob_sums, 1.0, rtol=1e-5)
        
        # Check that all probabilities are non-negative
        self.assertTrue(np.all(probabilities >= 0))
        
        # Check data type
        self.assertEqual(probabilities.dtype, np.float32)
    
    def test_predict_proba_with_tensors(self):
        """Test predict_proba method with tensor inputs."""
        probabilities = self.model.predict_proba(
            color_data=self.color_data,
            brightness_data=self.brightness_data,
            batch_size=4
        )
        
        # Check shape
        expected_shape = (self.batch_size, self.num_classes)
        self.assertEqual(probabilities.shape, expected_shape)
        
        # Check that probabilities sum to 1
        prob_sums = probabilities.sum(axis=1)
        np.testing.assert_allclose(prob_sums, 1.0, rtol=1e-5)
    
    def test_predict_proba_invalid_input(self):
        """Test predict_proba method with invalid inputs."""
        # Test with incomplete tensor data
        with self.assertRaises(ValueError) as context:
            self.model.predict_proba(color_data=self.color_data)
        self.assertIn("Either provide data_loader or both color_data and brightness_data", 
                     str(context.exception))
    
    def test_fusion_type_property(self):
        """Test fusion_type property."""
        fusion_type = self.model.fusion_type
        self.assertEqual(fusion_type, "concatenation")
        self.assertIsInstance(fusion_type, str)
    
    def test_training_mode_switching(self):
        """Test that training mode is properly switched during fit/evaluate."""
        # Start in eval mode
        self.model.eval()
        self.assertFalse(self.model.training)
        
        # After fit, should be in train mode
        self.model.fit(
            train_loader=self.data_loader,
            epochs=1,
            verbose=False
        )
        # fit() doesn't change mode after completion
        
        # After evaluate, should be in eval mode
        self.model.evaluate(data_loader=self.data_loader)
        self.assertFalse(self.model.training)
        
        # After predict, should be in eval mode
        self.model.predict(data_loader=self.data_loader)
        self.assertFalse(self.model.training)
    
    def test_device_handling(self):
        """Test that device is properly handled."""
        # Model should be on CPU
        self.assertEqual(str(self.model.device), 'cpu')
        
        # Test that model moves data to correct device
        predictions = self.model.predict(
            color_data=self.color_data,
            brightness_data=self.brightness_data
        )
        
        # Predictions should be on CPU
        self.assertEqual(predictions.device.type, 'cpu')
    
    def test_consistency_between_methods(self):
        """Test consistency between predict and predict_proba."""
        # Set model to eval mode for consistent results
        self.model.eval()
        
        # Set deterministic behavior
        torch.manual_seed(42)
        
        # Get predictions from both methods using the same data tensors to avoid dataloader iteration differences
        predictions = self.model.predict(
            color_data=self.color_data,
            brightness_data=self.brightness_data,
            batch_size=len(self.color_data)
        )
        
        # Reset seed for consistency
        torch.manual_seed(42)
        probabilities = self.model.predict_proba(
            color_data=self.color_data,
            brightness_data=self.brightness_data,
            batch_size=len(self.color_data)
        )
        
        # Get predicted classes from probabilities
        prob_predictions = torch.from_numpy(probabilities.argmax(axis=1))
        
        # Should be the same
        torch.testing.assert_close(predictions, prob_predictions)
    
    def test_different_batch_sizes(self):
        """Test methods work with different batch sizes."""
        # Create data with different batch size
        small_color = torch.randn(3, 3, self.input_size, self.input_size)
        small_brightness = torch.randn(3, 1, self.input_size, self.input_size)
        small_targets = torch.randint(0, self.num_classes, (3,))
        
        # Test predict
        predictions = self.model.predict(
            color_data=small_color,
            brightness_data=small_brightness,
            batch_size=2  # Different from data size
        )
        self.assertEqual(predictions.shape, (3,))
        
        # Test evaluate
        result = self.model.evaluate(
            color_data=small_color,
            brightness_data=small_brightness,
            targets=small_targets,
            batch_size=2
        )
        self.assertIn('accuracy', result)


class TestMCResNetHelperFunctions(unittest.TestCase):
    """Test cases for MCResNet helper functions."""
    
    def test_mc_resnet18(self):
        """Test mc_resnet18 helper function."""
        from models.multi_channel.mc_resnet import mc_resnet18
        
        # Test basic creation with explicit device
        model = mc_resnet18(num_classes=10, device='cpu')
        
        # Check that it's an MCResNet instance
        self.assertIsInstance(model, MCResNet)
        
        # Check that it has the right number of classes
        self.assertEqual(model.num_classes, 10)
        
        # Check that it uses BasicBlock (ResNet-18 configuration)
        first_block = model.layer1[0]
        self.assertIsInstance(first_block, MCBasicBlock)
        
        # Test forward pass with eval mode to avoid batch norm issues with batch_size=1
        model.eval()
        color_input = torch.randn(1, 3, 64, 64)  # Use larger input to avoid spatial size issues
        brightness_input = torch.randn(1, 1, 64, 64)
        output = model(color_input, brightness_input)
        
        self.assertEqual(output.shape, (1, 10))
    
    def test_mc_resnet18_with_kwargs(self):
        """Test mc_resnet18 with additional keyword arguments."""
        from models.multi_channel.mc_resnet import mc_resnet18
        
        # Test with device argument
        model = mc_resnet18(num_classes=5, device='cpu')
        
        # Check that device is set correctly
        self.assertEqual(str(model.device), 'cpu')
        self.assertEqual(model.num_classes, 5)
    
    def test_mc_resnet18_default_classes(self):
        """Test mc_resnet18 with default number of classes."""
        from models.multi_channel.mc_resnet import mc_resnet18
        
        model = mc_resnet18()
        
        # Should default to 1000 classes (ImageNet)
        self.assertEqual(model.num_classes, 1000)


# Also add a test for module imports to ensure everything is properly exported
class TestMCResNetImports(unittest.TestCase):
    """Test that all required classes and functions can be imported."""
    
    def test_import_mcresnet(self):
        """Test that MCResNet can be imported."""
        from models.multi_channel.mc_resnet import MCResNet
        self.assertTrue(callable(MCResNet))
    
    def test_import_helper_functions(self):
        """Test that helper functions can be imported."""
        from models.multi_channel.mc_resnet import mc_resnet18
        self.assertTrue(callable(mc_resnet18))
    
    def test_import_blocks(self):
        """Test that block classes can be imported."""
        from models.multi_channel.mc_resnet import MCBasicBlock, MCBottleneck
        self.assertTrue(callable(MCBasicBlock))
        self.assertTrue(callable(MCBottleneck))
    

class TestMCResNetSpecialCases(unittest.TestCase):
    """Test special cases and edge conditions for MCResNet."""
    
    def test_zero_init_residual_with_bottleneck(self):
        """Test zero initialization of residual connections with Bottleneck blocks."""
        model = MCResNet(
            block=MCBottleneck,
            layers=[3, 4, 6, 3],  # ResNet-50 configuration
            num_classes=10,
            zero_init_residual=True,  # Enable zero init
            device='cpu'
        )
        
        # Check that the last BN in Bottleneck blocks has zero weights
        found_bottleneck = False
        for m in model.modules():
            if isinstance(m, MCBottleneck):
                found_bottleneck = True
                if hasattr(m, 'bn3') and m.bn3.affine:
                    # Check if weights are initialized to zero
                    self.assertTrue(torch.allclose(m.bn3.color_weight, torch.zeros_like(m.bn3.color_weight)))
                    self.assertTrue(torch.allclose(m.bn3.brightness_weight, torch.zeros_like(m.bn3.brightness_weight)))
                break
        
        self.assertTrue(found_bottleneck, "Should have found at least one Bottleneck block")
    
    def test_zero_init_residual_with_basic_block(self):
        """Test zero initialization of residual connections with BasicBlock blocks."""
        model = MCResNet(
            block=MCBasicBlock,
            layers=[2, 2, 2, 2],
            num_classes=10,
            zero_init_residual=True,  # Enable zero init
            device='cpu'
        )
        
        # Check that the last BN in BasicBlock blocks has zero weights
        found_basic_block = False
        for m in model.modules():
            if isinstance(m, MCBasicBlock):
                found_basic_block = True
                if hasattr(m, 'bn2') and m.bn2.affine:
                    # Check if weights are initialized to zero
                    self.assertTrue(torch.allclose(m.bn2.color_weight, torch.zeros_like(m.bn2.color_weight)))
                    self.assertTrue(torch.allclose(m.bn2.brightness_weight, torch.zeros_like(m.bn2.brightness_weight)))
                break
        
        self.assertTrue(found_basic_block, "Should have found at least one BasicBlock")
    
    def test_custom_input_channels(self):
        """Test MCResNet with custom input channels."""
        model = MCResNet(
            block=MCBasicBlock,
            layers=[2, 2, 2, 2],
            num_classes=10,
            color_input_channels=4,  # Custom color channels (e.g., RGBA)
            brightness_input_channels=2,  # Custom brightness channels
            device='cpu'
        )
        
        self.assertEqual(model.color_input_channels, 4)
        self.assertEqual(model.brightness_input_channels, 2)
        
        # Test with appropriate input sizes
        color_input = torch.randn(2, 4, 32, 32)
        brightness_input = torch.randn(2, 2, 32, 32)
        
        output = model(color_input, brightness_input)
        self.assertEqual(output.shape, (2, 10))
    
    def test_groups_and_width_per_group(self):
        """Test MCResNet with custom groups and width_per_group parameters."""
        model = MCResNet(
            block=MCBottleneck,
            layers=[3, 4, 6, 3],
            num_classes=10,
            groups=2,
            width_per_group=32,
            device='cpu'
        )
        
        # Model should initialize successfully with custom group settings
        self.assertIsInstance(model, MCResNet)
        
        # Test forward pass
        color_input = torch.randn(2, 3, 32, 32)
        brightness_input = torch.randn(2, 1, 32, 32)
        output = model(color_input, brightness_input)
        self.assertEqual(output.shape, (2, 10))
    
    def test_dilation_settings(self):
        """Test MCResNet with dilation settings using Bottleneck blocks."""
        # MCBasicBlock doesn't support dilation > 1, so use MCBottleneck
        model = MCResNet(
            block=MCBottleneck,
            layers=[3, 4, 6, 3],
            num_classes=10,
            replace_stride_with_dilation=[False, True, True],
            device='cpu'
        )
        
        # Model should initialize successfully with dilation
        self.assertIsInstance(model, MCResNet)
        
        # Test forward pass
        color_input = torch.randn(2, 3, 32, 32)
        brightness_input = torch.randn(2, 1, 32, 32)
        output = model(color_input, brightness_input)
        self.assertEqual(output.shape, (2, 10))
    
    def test_custom_norm_layer(self):
        """Test MCResNet with custom normalization layer."""
        from models.multi_channel.conv import MCBatchNorm2d
        
        model = MCResNet(
            block=MCBasicBlock,
            layers=[2, 2, 2, 2],
            num_classes=10,
            norm_layer=MCBatchNorm2d,  # Explicitly set norm layer
            device='cpu'
        )
        
        # Model should initialize successfully
        self.assertIsInstance(model, MCResNet)
        
        # Test forward pass
        color_input = torch.randn(2, 3, 32, 32)
        brightness_input = torch.randn(2, 1, 32, 32)
        output = model(color_input, brightness_input)
        self.assertEqual(output.shape, (2, 10))
    
    @patch('models2.multi_channel.mc_resnet.get_ipython')
    def test_tqdm_import_jupyter_environment(self, mock_get_ipython):
        """Test tqdm import in Jupyter environment."""
        # Mock Jupyter environment
        mock_ipython = MagicMock()
        mock_ipython.__class__.__name__ = 'ZMQInteractiveShell'
        mock_get_ipython.return_value = mock_ipython
        
        # This test mainly ensures the import logic is covered
        # The actual tqdm import happens at module level, but we can test related functionality
        model = MCResNet(
            block=MCBasicBlock,
            layers=[2, 2, 2, 2],
            num_classes=10,
            device='cpu'
        )
        self.assertIsInstance(model, MCResNet)
    
    @patch('models2.multi_channel.mc_resnet.get_ipython')
    def test_tqdm_import_exception_handling(self, mock_get_ipython):
        """Test tqdm import exception handling."""
        # Mock an exception during IPython detection
        mock_get_ipython.side_effect = Exception("IPython not available")
        
        # This should fall back to regular tqdm import
        model = MCResNet(
            block=MCBasicBlock,
            layers=[2, 2, 2, 2],
            num_classes=10,
            device='cpu'
        )
        self.assertIsInstance(model, MCResNet)


class TestMCResNetGradientAccumulation(unittest.TestCase):
    """Test gradient accumulation functionality in MCResNet."""
    
    def setUp(self):
        """Set up test fixtures for gradient accumulation tests."""
        self.model = MCResNet(
            block=MCBasicBlock,
            layers=[2, 2, 2, 2],
            num_classes=10,
            device='cpu'
        )
        
        # Compile model with basic optimizer string instead of object
        self.model.compile(
            optimizer='sgd',
            learning_rate=0.01,
            loss='cross_entropy'
        )
        
        # Create test data
        self.color_data = torch.randn(16, 3, 32, 32)
        self.brightness_data = torch.randn(16, 1, 32, 32)
        self.targets = torch.randint(0, 10, (16,))
        
        self.train_loader = create_dual_channel_dataloader(
            self.color_data, self.brightness_data, self.targets,
            batch_size=4, num_workers=0
        )
    
    def test_gradient_accumulation_default(self):
        """Test that gradient_accumulation_steps=1 works as default."""
        history = self.model.fit(
            train_loader=self.train_loader,
            epochs=1,
            verbose=False,
            gradient_accumulation_steps=1
        )
        
        self.assertIn('train_loss', history)
        self.assertIn('train_accuracy', history)
        self.assertEqual(len(history['train_loss']), 1)
    
    def test_gradient_accumulation_steps_parameter(self):
        """Test that gradient_accumulation_steps parameter is properly handled."""
        # Test with accumulation steps = 2
        history = self.model.fit(
            train_loader=self.train_loader,
            epochs=1,
            verbose=False,
            gradient_accumulation_steps=2
        )
        
        self.assertIn('train_loss', history)
        self.assertIn('train_accuracy', history)
        self.assertEqual(len(history['train_loss']), 1)
    
    def test_gradient_accumulation_parameter_validation(self):
        """Test that invalid gradient_accumulation_steps values are handled."""
        # Test with gradient_accumulation_steps = 0 (should work but be effectively 1)
        history = self.model.fit(
            train_loader=self.train_loader,
            epochs=1,
            verbose=False,
            gradient_accumulation_steps=0
        )
        
        self.assertIn('train_loss', history)
        
        # Test with very large accumulation steps
        history = self.model.fit(
            train_loader=self.train_loader,
            epochs=1,
            verbose=False,
            gradient_accumulation_steps=100
        )
        
        self.assertIn('train_loss', history)
    
    def test_gradient_accumulation_with_amp(self):
        """Test gradient accumulation with automatic mixed precision."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available for AMP testing")
        
        model_amp = MCResNet(
            block=MCBasicBlock,
            layers=[2, 2, 2, 2],
            num_classes=10,
            device='cuda',
            use_amp=True
        )
        
        model_amp.compile(
            optimizer='sgd',
            learning_rate=0.01,
            loss='cross_entropy'
        )
        
        # Move test data to GPU
        color_data_gpu = self.color_data.cuda()
        brightness_data_gpu = self.brightness_data.cuda()
        targets_gpu = self.targets.cuda()
        
        train_loader_gpu = create_dual_channel_dataloader(
            color_data_gpu, brightness_data_gpu, targets_gpu,
            batch_size=4, num_workers=0
        )
        
        history = model_amp.fit(
            train_loader=train_loader_gpu,
            epochs=1,
            verbose=False,
            gradient_accumulation_steps=2
        )
        
        self.assertIn('train_loss', history)
        self.assertIn('train_accuracy', history)
    
    def test_gradient_accumulation_with_onecycle_scheduler(self):
        """Test gradient accumulation with OneCycleLR scheduler."""
        # Create model with OneCycleLR scheduler
        model = MCResNet(
            block=MCBasicBlock,
            layers=[2, 2, 2, 2],
            num_classes=10,
            device='cpu'
        )
        
        model.compile(
            optimizer='sgd',
            learning_rate=0.01,
            loss='cross_entropy',
            scheduler='onecycle'
        )
        
        history = model.fit(
            train_loader=self.train_loader,
            epochs=2,
            verbose=False,
            gradient_accumulation_steps=2,
            # OneCycleLR specific parameters
            steps_per_epoch=len(self.train_loader),
            max_lr=0.1
        )
        
        self.assertIn('train_loss', history)
        self.assertIn('learning_rates', history)
        # Should have learning rate updates when using OneCycleLR
        self.assertGreater(len(history['learning_rates']), 0)
    
    def test_gradient_accumulation_training_epoch_method(self):
        """Test the _train_epoch method directly with gradient accumulation."""
        self.model.train()
        
        # Initialize history
        history = {
            'train_loss': [], 
            'val_loss': [], 
            'train_accuracy': [],
            'val_accuracy': [],
            'learning_rates': []
        }
        
        # Test with different accumulation steps
        for accumulation_steps in [1, 2, 4]:
            avg_loss, accuracy = self.model._train_epoch(
                self.train_loader, 
                history, 
                pbar=None, 
                gradient_accumulation_steps=accumulation_steps
            )
            
            self.assertIsInstance(avg_loss, float)
            self.assertIsInstance(accuracy, float)
            self.assertGreaterEqual(accuracy, 0.0)
            self.assertLessEqual(accuracy, 1.0)
            
            self.assertIsInstance(avg_loss, float)
            self.assertIsInstance(accuracy, float)
            self.assertGreaterEqual(accuracy, 0.0)
            self.assertLessEqual(accuracy, 1.0)
    
    def test_gradient_accumulation_backward_compatibility(self):
        """Test that existing code works without gradient_accumulation_steps parameter."""
        # Test that fit() works without the gradient_accumulation_steps parameter
        history = self.model.fit(
            train_loader=self.train_loader,
            epochs=1,
            verbose=False
            # No gradient_accumulation_steps parameter - should default to 1
        )
        
        self.assertIn('train_loss', history)
        self.assertIn('train_accuracy', history)
    
    def test_gradient_accumulation_effective_batch_size(self):
        """Test that gradient accumulation simulates larger batch sizes correctly."""
        # Create smaller batches for testing accumulation
        small_batch_loader = create_dual_channel_dataloader(
            self.color_data, self.brightness_data, self.targets,
            batch_size=2, num_workers=0  # Smaller batch size
        )
        
        # Train with gradient accumulation (effective batch size = 2 * 2 = 4)
        history_accumulated = self.model.fit(
            train_loader=small_batch_loader,
            epochs=1,
            verbose=False,
            gradient_accumulation_steps=2
        )
        
        # Should complete training without errors
        self.assertIn('train_loss', history_accumulated)
        self.assertIn('train_accuracy', history_accumulated)
    
    def test_gradient_accumulation_edge_cases(self):
        """Test edge cases for gradient accumulation."""
        # Test with accumulation steps larger than number of batches
        very_large_steps = len(self.train_loader) + 10
        
        history = self.model.fit(
            train_loader=self.train_loader,
            epochs=1,
            verbose=False,
            gradient_accumulation_steps=very_large_steps
        )
        
        self.assertIn('train_loss', history)
        
        # Test with single batch loader
        single_batch_loader = create_dual_channel_dataloader(
            self.color_data[:4], self.brightness_data[:4], self.targets[:4],
            batch_size=4, num_workers=0
        )
        
        history = self.model.fit(
            train_loader=single_batch_loader,
            epochs=1,
            verbose=False,
            gradient_accumulation_steps=2
        )
        
        self.assertIn('train_loss', history)
        
        # Test with single batch loader
        single_batch_loader = create_dual_channel_dataloader(
            self.color_data[:4], self.brightness_data[:4], self.targets[:4],
            batch_size=4, num_workers=0
        )
        
        history = self.model.fit(
            train_loader=single_batch_loader,
            epochs=1,
            verbose=False,
            gradient_accumulation_steps=2
        )
        
        self.assertIn('train_loss', history)
    
    def test_gradient_accumulation_memory_efficiency(self):
        """Test that gradient accumulation doesn't cause memory issues."""
        # Create larger test data to test memory management
        large_color_data = torch.randn(32, 3, 64, 64)
        large_brightness_data = torch.randn(32, 1, 64, 64)
        large_targets = torch.randint(0, 10, (32,))
        
        large_loader = create_dual_channel_dataloader(
            large_color_data, large_brightness_data, large_targets,
            batch_size=8, num_workers=0
        )
        
        # Train with gradient accumulation
        history = self.model.fit(
            train_loader=large_loader,
            epochs=1,
            verbose=False,
            gradient_accumulation_steps=4
        )
        
        self.assertIn('train_loss', history)
        self.assertIn('train_accuracy', history)
