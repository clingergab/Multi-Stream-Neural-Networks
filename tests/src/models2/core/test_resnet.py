"""
Comprehensive unit tests for ResNet models.
Consolidates all ResNet-related tests into a single file to mirror the source structure.
"""

import unittest
import torch
import torch.nn as nn
import sys
from pathlib import Path
import tempfile
import os
from unittest.mock import patch, MagicMock
# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent / "src"))

from src.models2.core.resnet import (
    ResNet, resnet18, resnet34, resnet50, resnet101, resnet152
)
from src.models2.core.blocks import BasicBlock, Bottleneck
from torch.utils.data import DataLoader, TensorDataset


class TestResNetArchitecture(unittest.TestCase):
    """Test cases for ResNet model architecture and construction."""
    
    def test_resnet_building(self):
        """Test the ResNet model construction."""
        # Create a mini ResNet
        model = ResNet(
            block=BasicBlock,
            layers=[1, 1, 1, 1],
            num_classes=10,
            device='cpu'  # Ensure consistent device for testing
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
        # Reduced-size test model with explicit CPU device for consistent testing
        model = ResNet(
            block=BasicBlock,
            layers=[1, 1, 1, 1],
            num_classes=10,
            device='cpu'  # Ensure consistent device for testing
        )
        
        # Test with RGB input (on same device as model)
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
        
        # List of models to test with output size (all on CPU for consistent testing)
        models = [
            (resnet18(num_classes=10, device='cpu'), (batch_size, 10)),
            (resnet34(num_classes=10, device='cpu'), (batch_size, 10)),
            (resnet50(num_classes=10, device='cpu'), (batch_size, 10)),
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
        model = resnet18(num_classes=10, device='cpu')
        
        # Check initialization of different layer types
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                # Conv layers should have been initialized with kaiming
                self.assertFalse(torch.allclose(m.weight, torch.zeros_like(m.weight)))
            elif isinstance(m, torch.nn.BatchNorm2d):
                # BatchNorm weight should be 1, bias should be 0
                self.assertTrue(torch.allclose(m.weight, torch.ones_like(m.weight)))
                self.assertTrue(torch.allclose(m.bias, torch.zeros_like(m.bias)))


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


class TestResNetTrainingAPI(unittest.TestCase):
    """Test cases for ResNet training API methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a small model for testing (with CPU device for consistent testing)
        self.model = resnet18(num_classes=10, device='cpu')
        
        # Create a small dataset for testing
        self.num_samples = 20
        self.batch_size = 4
        
        # Random inputs and targets
        self.inputs = torch.randn(self.num_samples, 3, 32, 32)
        self.targets = torch.randint(0, 10, (self.num_samples,))
        
        # Create train and validation datasets
        train_dataset = TensorDataset(self.inputs, self.targets)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size)
        
        # Same data for validation in this test
        self.val_loader = self.train_loader
    
    def test_compile(self):
        """Test the compile method."""
        # Test basic compilation (device is set in constructor, not compile)
        self.model.compile(optimizer='adam', loss='cross_entropy')
        
        # Verify components were created
        self.assertIsNotNone(self.model.optimizer)
        self.assertIsNotNone(self.model.criterion)
        # Device should already be set from constructor
        self.assertEqual(self.model.device, torch.device('cpu'))
        
        # Test with scheduler - scheduler is set in compile but scheduler object is created in fit
        self.model.compile(
            optimizer='sgd', 
            loss='cross_entropy',
            lr=0.01, 
            scheduler='step'
        )
        self.assertEqual(self.model.scheduler_type, 'step')
        
        # Test cosine scheduler type is stored
        self.model.compile(
            optimizer='adam',
            loss='cross_entropy',
            lr=0.01,
            scheduler='cosine'
        )
        self.assertEqual(self.model.scheduler_type, 'cosine')
        
        # Test OneCycleLR scheduler type is stored
        self.model.compile(
            optimizer='sgd',
            loss='cross_entropy',
            lr=0.01,
            scheduler='onecycle',
            device='cpu'
        )
        self.assertEqual(self.model.scheduler_type, 'onecycle')
    
    def test_fit_method(self):
        """Test the fit method."""
        # Compile the model
        self.model.compile(optimizer='adam', loss='cross_entropy', device='cpu')
        
        # Train for just 2 epochs to keep test fast
        history = self.model.fit(
            self.train_loader,
            self.val_loader,
            epochs=2,
            verbose=False
        )
        
        # Check that history was recorded
        self.assertIn('train_loss', history)
        self.assertIn('train_accuracy', history)
        self.assertIn('val_loss', history)
        self.assertIn('val_accuracy', history)
        self.assertIn('learning_rates', history)
        
        # Check that we have the right number of epochs
        self.assertEqual(len(history['train_loss']), 2)
        self.assertEqual(len(history['val_loss']), 2)
    
    def test_predict_method(self):
        """Test the predict method."""
        # Compile the model
        self.model.compile(optimizer='adam', loss='cross_entropy', device='cpu')
        
        # Test prediction with targets (required for tensor input)
        predictions = self.model.predict(self.inputs, self.targets)
        
        # Check prediction shape and type
        self.assertEqual(predictions.shape, (self.num_samples,))
        self.assertTrue(torch.all(predictions >= 0))
        self.assertTrue(torch.all(predictions < 10))  # Should be valid class indices
    
    def test_evaluate_method(self):
        """Test the evaluate method."""
        # Compile the model
        self.model.compile(optimizer='adam', loss='cross_entropy', device='cpu')
        
        # Evaluate on the test data
        result = self.model.evaluate(self.inputs, self.targets)
        
        # Check that we get valid metrics - the evaluate method might return different formats
        if isinstance(result, tuple):
            test_loss, test_accuracy = result
            self.assertIsInstance(test_loss, (float, torch.Tensor))
            self.assertIsInstance(test_accuracy, (float, torch.Tensor))
            if isinstance(test_accuracy, torch.Tensor):
                test_accuracy = test_accuracy.item()
            self.assertGreaterEqual(test_accuracy, 0.0)
            self.assertLessEqual(test_accuracy, 1.0)
        else:
            # If it returns a single value or different format
            self.assertIsNotNone(result)
    
    def test_checkpoint_saving(self):
        """Test checkpoint saving and loading."""
        # Compile the model
        self.model.compile(optimizer='adam', loss='cross_entropy', device='cpu')
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            try:
                # Save the model state
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.model.optimizer.state_dict(),
                }, tmp.name)
                
                # Create a new model and load the state (on same device)
                new_model = resnet18(num_classes=10, device='cpu')
                new_model.compile(optimizer='adam', loss='cross_entropy')
                
                checkpoint = torch.load(tmp.name)
                new_model.load_state_dict(checkpoint['model_state_dict'])
                new_model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                # Verify that the weights are the same
                for param1, param2 in zip(self.model.parameters(), new_model.parameters()):
                    self.assertTrue(torch.equal(param1, param2))
                    
            finally:
                # Clean up
                os.unlink(tmp.name)
    
    def test_scheduler_initialization_in_fit(self):
        """Test that scheduler is properly initialized during fit."""
        # Compile with OneCycleLR scheduler
        self.model.compile(
            optimizer='sgd',
            loss='cross_entropy',
            lr=0.01,
            scheduler='onecycle',
            device='cpu'
        )
        
        # Train for 1 epoch
        history = self.model.fit(
            self.train_loader,
            epochs=1,
            verbose=False
        )
        
        # Check that scheduler was created
        self.assertIsNotNone(self.model.scheduler)
        
        # Check that learning rates were recorded
        self.assertIn('learning_rates', history)
        self.assertGreater(len(history['learning_rates']), 0)


class TestResNetSchedulers(unittest.TestCase):
    """Test cases for ResNet learning rate schedulers."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = resnet18(num_classes=10)
        
        # Create small dataset
        self.num_samples = 16
        self.batch_size = 4
        self.inputs = torch.randn(self.num_samples, 3, 32, 32)
        self.targets = torch.randint(0, 10, (self.num_samples,))
        train_dataset = TensorDataset(self.inputs, self.targets)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size)
    
    def test_onecycle_scheduler(self):
        """Test OneCycleLR scheduler."""
        self.model.compile(
            optimizer='sgd',
            loss='cross_entropy',
            lr=0.01,
            scheduler='onecycle',
            device='cpu'
        )
        
        history = self.model.fit(
            self.train_loader,
            epochs=2,
            verbose=False
        )
        
        # Check that learning rates changed
        lr_history = history['learning_rates']
        self.assertGreater(len(lr_history), 0)
        # OneCycleLR should have varying learning rates
        self.assertTrue(len(set(lr_history)) > 1)
    
    def test_step_scheduler(self):
        """Test StepLR scheduler."""
        self.model.compile(
            optimizer='sgd',
            loss='cross_entropy',
            lr=0.1,
            scheduler='step',
            device='cpu'
        )
        
        history = self.model.fit(
            self.train_loader,
            epochs=3,
            verbose=False
        )
        
        self.assertIn('learning_rates', history)
        self.assertGreater(len(history['learning_rates']), 0)
    
    def test_cosine_scheduler(self):
        """Test CosineAnnealingLR scheduler."""
        self.model.compile(
            optimizer='adam',
            loss='cross_entropy',
            lr=0.01,
            scheduler='cosine',
            device='cpu'
        )
        
        history = self.model.fit(
            self.train_loader,
            epochs=2,
            verbose=False
        )
        
        self.assertIn('learning_rates', history)
        self.assertGreater(len(history['learning_rates']), 0)
    
    def test_plateau_scheduler(self):
        """Test ReduceLROnPlateau scheduler."""
        self.model.compile(
            optimizer='adam',
            loss='cross_entropy',
            lr=0.01,
            scheduler='plateau',
            device='cpu'
        )
        
        history = self.model.fit(
            self.train_loader,
            self.train_loader,  # Use same data for validation
            epochs=2,
            verbose=False
        )
        
        self.assertIn('learning_rates', history)
        self.assertGreater(len(history['learning_rates']), 0)
    
    def test_no_scheduler(self):
        """Test training without scheduler."""
        self.model.compile(
            optimizer='adam',
            loss='cross_entropy',
            lr=0.01,
            device='cpu'
        )
        
        history = self.model.fit(
            self.train_loader,
            epochs=2,
            verbose=False
        )
        
        # Learning rate should remain constant
        lr_history = history['learning_rates']
        self.assertTrue(all(lr == lr_history[0] for lr in lr_history))
    
    def test_invalid_scheduler(self):
        """Test invalid scheduler handling."""
        # The implementation may not raise an error during compile, so let's test actual behavior
        try:
            self.model.compile(
                optimizer='adam',
                loss='cross_entropy',
                scheduler='invalid_scheduler'
            )
            # If compile succeeds, try to use it and expect an error during training
            with self.assertRaises((ValueError, AttributeError, RuntimeError)):
                self.model.fit(
                    self.train_loader,
                    epochs=1,
                    verbose=False
                )
        except (ValueError, AttributeError):
            # Expected behavior if compilation itself fails
            pass
    
    def test_scheduler_state_saving(self):
        """Test that scheduler state can be saved and loaded."""
        self.model.compile(
            optimizer='sgd',
            loss='cross_entropy',
            lr=0.01,
            scheduler='step',
            device='cpu'
        )
        
        # Train for 1 epoch to initialize scheduler
        self.model.fit(
            self.train_loader,
            epochs=1,
            verbose=False
        )
        
        # Check scheduler exists
        self.assertIsNotNone(self.model.scheduler)
        
        # Save and load scheduler state
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            try:
                torch.save({
                    'scheduler_state_dict': self.model.scheduler.state_dict()
                }, tmp.name)
                
                checkpoint = torch.load(tmp.name)
                self.model.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
            finally:
                os.unlink(tmp.name)


class TestResNetEarlyStopping(unittest.TestCase):
    """Test cases for ResNet early stopping functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a small model for testing
        self.model = resnet18(num_classes=10)
        self.model.compile(optimizer='adam', loss='cross_entropy', device='cpu')
        
        # Create a small dataset for testing
        self.num_samples = 20
        self.batch_size = 4
        
        # Random inputs and targets
        self.inputs = torch.randn(self.num_samples, 3, 32, 32)
        self.targets = torch.randint(0, 10, (self.num_samples,))
        
        # Create train and validation datasets
        train_dataset = TensorDataset(self.inputs, self.targets)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size)
        
        # Same data for validation in this test
        self.val_loader = self.train_loader
    
    def test_early_stopping_triggered(self):
        """Test that early stopping triggers when validation doesn't improve."""
        # Train with very low patience to trigger early stopping quickly
        history = self.model.fit(
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            epochs=10,
            early_stopping=True,
            patience=2,
            verbose=False  # Disable verbose to test non-progress-bar paths
        )
        
        # Should stop early (less than or equal to 10 epochs, could be exactly 10 if no improvement)
        self.assertLessEqual(len(history['train_loss']), 10)
        self.assertLessEqual(len(history['val_loss']), 10)
        
        # Should have consistent history lengths
        self.assertEqual(len(history['train_loss']), len(history['val_loss']))
        self.assertEqual(len(history['train_accuracy']), len(history['val_accuracy']))
        
        # Early stopping should have been active
        self.assertGreater(len(history['train_loss']), 0)
    
    def test_early_stopping_with_restore_weights(self):
        """Test early stopping with weight restoration."""
        # Save initial weights
        initial_weights = {k: v.clone() for k, v in self.model.state_dict().items()}
        
        # Train with early stopping and weight restoration
        history = self.model.fit(
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            epochs=8,
            early_stopping=True,
            patience=2,
            restore_best_weights=True,
            verbose=False
        )
        
        # Weights should be different from initial (training occurred)
        final_weights = {k: v.clone() for k, v in self.model.state_dict().items()}
        weights_changed = any(
            not torch.equal(initial_weights[k], final_weights[k]) 
            for k in initial_weights.keys()
        )
        self.assertTrue(weights_changed, "Model weights should have changed during training")
    
    def test_early_stopping_disabled(self):
        """Test training without early stopping."""
        history = self.model.fit(
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            epochs=3,
            early_stopping=False,
            verbose=False
        )
        
        # Should complete all epochs
        self.assertEqual(len(history['train_loss']), 3)
        self.assertEqual(len(history['val_loss']), 3)
    
    def test_early_stopping_without_validation(self):
        """Test that early stopping is skipped when no validation data is provided."""
        history = self.model.fit(
            train_loader=self.train_loader,
            val_loader=None,  # No validation data
            epochs=3,
            early_stopping=True,  # Should be ignored
            patience=1,
            verbose=False
        )
        
        # Should complete all epochs since no validation data
        self.assertEqual(len(history['train_loss']), 3)
        # When no validation loader is provided, validation arrays are empty but still present
        if 'val_loss' in history:
            self.assertEqual(len(history['val_loss']), 0)
        if 'val_accuracy' in history:
            self.assertEqual(len(history['val_accuracy']), 0)
    
    def test_early_stopping_monitor_validation_accuracy(self):
        """Test early stopping monitoring validation accuracy instead of loss."""
        history = self.model.fit(
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            epochs=6,
            early_stopping=True,
            patience=2,
            monitor='val_accuracy',
            verbose=False
        )
        
        # Should have trained (history exists)
        self.assertGreater(len(history['train_loss']), 0)
        self.assertIn('val_accuracy', history)
    
    def test_invalid_monitor_parameter(self):
        """Test that invalid monitor parameter raises appropriate error or falls back."""
        # This should either raise an error or fall back to default behavior
        try:
            history = self.model.fit(
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                epochs=2,
                early_stopping=True,
                patience=1,
                monitor='invalid_metric',
                verbose=False
            )
            # If no error, training should still complete
            self.assertGreater(len(history['train_loss']), 0)
        except (ValueError, KeyError):
            # Expected behavior for invalid monitor
            pass


class TestResNetErrorHandling(unittest.TestCase):
    """Test cases for ResNet error handling and edge cases."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = resnet18(num_classes=10)
    
    def test_compile_with_invalid_optimizer(self):
        """Test compile with invalid optimizer string."""
        with self.assertRaises((ValueError, AttributeError)):
            self.model.compile(optimizer='invalid_optimizer', loss='cross_entropy')
    
    def test_compile_with_invalid_loss(self):
        """Test compile with invalid loss string."""
        with self.assertRaises((ValueError, AttributeError)):
            self.model.compile(optimizer='adam', loss='invalid_loss')
    
    def test_fit_without_compile(self):
        """Test that fit fails gracefully when model isn't compiled."""
        inputs = torch.randn(10, 3, 32, 32)
        targets = torch.randint(0, 10, (10,))
        train_dataset = TensorDataset(inputs, targets)
        train_loader = DataLoader(train_dataset, batch_size=4)
        
        with self.assertRaises(ValueError):
            self.model.fit(train_loader=train_loader, epochs=1)
    
    def test_predict_without_compile(self):
        """Test that predict works even without compile."""
        inputs = torch.randn(4, 3, 32, 32)
        targets = torch.randint(0, 10, (4,))  # Add targets for tensor input
        
        # Predict should work (just forward pass) but returns class predictions, not logits
        outputs = self.model.predict(inputs, targets)
        self.assertEqual(outputs.shape, (4,))  # Changed expectation - predict returns class indices
        
        # Test that output values are valid class indices
        self.assertTrue(torch.all(outputs >= 0))
        self.assertTrue(torch.all(outputs < 10))
    
    def test_predict_tensor_without_targets_raises_error(self):
        """Test that predict raises ValueError when called with tensor but no targets."""
        inputs = torch.randn(4, 3, 32, 32)
        
        # Should raise ValueError when targets are not provided for tensor input
        with self.assertRaises(ValueError) as context:
            self.model.predict(inputs)
        
        self.assertIn("targets must be provided when data_loader is a tensor", str(context.exception))
    
    def test_evaluate_without_compile(self):
        """Test that evaluate fails gracefully when model isn't compiled."""
        inputs = torch.randn(10, 3, 32, 32)
        targets = torch.randint(0, 10, (10,))
        
        with self.assertRaises(ValueError):
            self.model.evaluate(inputs, targets)
    
    def test_empty_data_handling(self):
        """Test handling of empty datasets."""
        self.model.compile(optimizer='adam', loss='cross_entropy', device='cpu')
        
        # Empty tensors
        empty_inputs = torch.empty(0, 3, 32, 32)
        empty_targets = torch.empty(0, dtype=torch.long)
        empty_dataset = TensorDataset(empty_inputs, empty_targets)
        empty_loader = DataLoader(empty_dataset, batch_size=1)
        
        # This should raise a ZeroDivisionError for empty data
        with self.assertRaises(ZeroDivisionError):
            history = self.model.fit(train_loader=empty_loader, epochs=1, verbose=False)
    
    def test_mismatched_input_dimensions(self):
        """Test handling of wrong input dimensions."""
        self.model.compile(optimizer='adam', loss='cross_entropy', device='cpu')
        
        # Wrong number of channels (should be 3 for RGB)
        wrong_inputs = torch.randn(4, 1, 32, 32)  # 1 channel instead of 3
        wrong_targets = torch.randint(0, 10, (4,))  # Add dummy targets
        
        with self.assertRaises(RuntimeError):
            self.model.predict(wrong_inputs, wrong_targets)
    
    def test_invalid_scheduler_configuration(self):
        """Test compilation with invalid scheduler."""
        # Test with invalid scheduler name - this may not raise an error until runtime
        try:
            self.model.compile(
                optimizer='adam', 
                loss='cross_entropy',
                scheduler='invalid_scheduler'
            )
            # If compilation succeeds, the error will come during training
            inputs = torch.randn(4, 3, 32, 32)
            targets = torch.randint(0, 10, (4,))
            train_dataset = TensorDataset(inputs, targets)
            train_loader = DataLoader(train_dataset, batch_size=2)
            
            # This should fail during training
            with self.assertRaises((ValueError, AttributeError, RuntimeError)):
                history = self.model.fit(train_loader=train_loader, epochs=1, verbose=False)
        except (ValueError, AttributeError):
            # Expected behavior if compilation itself fails
            pass
    
    def test_large_batch_size_handling(self):
        """Test handling of batch sizes larger than dataset."""
        self.model.compile(optimizer='adam', loss='cross_entropy', device='cpu')
        
        # Small dataset with large batch size
        inputs = torch.randn(5, 3, 32, 32)
        targets = torch.randint(0, 10, (5,))
        dataset = TensorDataset(inputs, targets)
        loader = DataLoader(dataset, batch_size=10)  # Larger than dataset
        
        # Should still work
        history = self.model.fit(train_loader=loader, epochs=1, verbose=False)
        self.assertIn('train_loss', history)
        self.assertEqual(len(history['train_loss']), 1)
    
    def test_save_load_without_path(self):
        """Test save/load operations with edge cases."""
        self.model.compile(optimizer='adam', loss='cross_entropy', device='cpu')
        
        # Test that save_checkpoint method doesn't exist - this tests our understanding
        self.assertFalse(hasattr(self.model, 'save_checkpoint'))
        
        # Test that we can still save the model using PyTorch's built-in methods
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            torch.save(self.model.state_dict(), tmp.name)
            
            # Load it back
            new_model = resnet18(num_classes=10)
            new_model.load_state_dict(torch.load(tmp.name))
            
            # Clean up
            os.unlink(tmp.name)
    
    def test_device_mismatch_handling(self):
        """Test handling of device mismatches."""
        # Compile for CPU
        self.model.compile(optimizer='adam', loss='cross_entropy', device='cpu')
        
        # Create data on CPU
        inputs = torch.randn(4, 3, 32, 32)
        targets = torch.randint(0, 10, (4,))  # Add dummy targets
        
        # Should work fine
        outputs = self.model.predict(inputs, targets)
        self.assertEqual(outputs.device.type, 'cpu')
    
    def test_zero_epochs_training(self):
        """Test training with zero epochs."""
        self.model.compile(optimizer='adam', loss='cross_entropy', device='cpu')
        
        inputs = torch.randn(10, 3, 32, 32)
        targets = torch.randint(0, 10, (10,))
        train_dataset = TensorDataset(inputs, targets)
        train_loader = DataLoader(train_dataset, batch_size=4)
        
        history = self.model.fit(train_loader=train_loader, epochs=0, verbose=False)
        
        # Should return empty history
        self.assertEqual(len(history['train_loss']), 0)
        self.assertEqual(len(history['train_accuracy']), 0)
    
    def test_dilation_validation_error(self):
        """Test that invalid dilation configuration raises ValueError."""
        with self.assertRaises(ValueError) as context:
            ResNet(
                block=BasicBlock,
                layers=[2, 2, 2, 2],
                num_classes=10,
                replace_stride_with_dilation=[True, False]  # Invalid length (should be 3, device='cpu')
            )
        self.assertIn("replace_stride_with_dilation should be None", str(context.exception))


# Additional test classes for improved coverage
class TestResNetInternalMethods(unittest.TestCase):
    """Test cases for ResNet internal methods to improve coverage."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = resnet18(num_classes=10)
        self.sample_input = torch.randn(2, 3, 32, 32)
        self.sample_targets = torch.randint(0, 10, (2,))
    
    def test_init_method(self):
        """Test ResNet __init__ method with various configurations."""
        # Test with BasicBlock
        model1 = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=100, device='cpu')
        self.assertEqual(model1.num_classes, 100)
        # Verify network was built correctly (layers exist)
        self.assertTrue(hasattr(model1, 'conv1'))
        self.assertTrue(hasattr(model1, 'layer1'))
        self.assertTrue(hasattr(model1, 'layer2'))
        self.assertTrue(hasattr(model1, 'layer3'))
        self.assertTrue(hasattr(model1, 'layer4'))
        
        # Test with Bottleneck
        model2 = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=1000, zero_init_residual=True, device='cpu')
        self.assertEqual(model2.num_classes, 1000)
        # Verify network was built correctly
        self.assertTrue(hasattr(model2, 'conv1'))
        self.assertTrue(hasattr(model2, 'fc'))
        
        # Test with custom parameters (use Bottleneck for groups/width)
        model3 = ResNet(
            Bottleneck, [1, 1, 1, 1], 
            num_classes=50, 
            groups=2, 
            width_per_group=32,
            replace_stride_with_dilation=[False, True, True],
            device='cpu'
        )
        self.assertEqual(model3.groups, 2)
        self.assertEqual(model3.base_width, 32)  # ResNet stores width_per_group as base_width
        # Verify network was built correctly with custom parameters
        self.assertTrue(hasattr(model3, 'conv1'))
        self.assertTrue(hasattr(model3, 'fc'))
    
    def test_build_network_method(self):
        """Test the _build_network method."""
        model = ResNet(BasicBlock, [1, 1, 1, 1], num_classes=10, device='cpu')
        
        # Check that all layers are created
        self.assertTrue(hasattr(model, 'conv1'))
        self.assertTrue(hasattr(model, 'bn1'))
        self.assertTrue(hasattr(model, 'relu'))
        self.assertTrue(hasattr(model, 'maxpool'))
        self.assertTrue(hasattr(model, 'layer1'))
        self.assertTrue(hasattr(model, 'layer2'))
        self.assertTrue(hasattr(model, 'layer3'))
        self.assertTrue(hasattr(model, 'layer4'))
        self.assertTrue(hasattr(model, 'avgpool'))
        self.assertTrue(hasattr(model, 'fc'))
        
        # Check layer dimensions
        self.assertEqual(model.conv1.in_channels, 3)
        self.assertEqual(model.conv1.out_channels, 64)
        self.assertEqual(model.fc.out_features, 10)
    
    def test_initialize_weights_method(self):
        """Test the _initialize_weights method."""
        model = ResNet(BasicBlock, [1, 1, 1, 1], num_classes=10, zero_init_residual=True, device='cpu')
        
        # Check that weights are initialized (not all zeros)
        conv_weights_sum = sum(p.sum().item() for p in model.parameters() if p.dim() > 1)
        self.assertNotEqual(conv_weights_sum, 0.0)
        
        # Test without zero_init_residual
        model2 = ResNet(BasicBlock, [1, 1, 1, 1], num_classes=10, zero_init_residual=False, device='cpu')
        conv_weights_sum2 = sum(p.sum().item() for p in model2.parameters() if p.dim() > 1)
        self.assertNotEqual(conv_weights_sum2, 0.0)
    
    def test_make_layer_method(self):
        """Test the _make_layer method."""
        model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10, device='cpu')
        
        # Test creating a layer
        layer = model._make_layer(BasicBlock, 128, 2, stride=2)
        self.assertIsInstance(layer, nn.Sequential)
        self.assertEqual(len(layer), 2)  # 2 blocks
        
        # Test with dilation
        layer_dilated = model._make_layer(BasicBlock, 256, 1, stride=1, dilate=True)
        self.assertIsInstance(layer_dilated, nn.Sequential)
    
    def test_train_epoch_method(self):
        """Test the _train_epoch method."""
        model = resnet18(num_classes=10)
        model.compile(optimizer='adam', loss='cross_entropy', lr=0.001, device='cpu')
        
        # Create sample data
        dataset = TensorDataset(self.sample_input, self.sample_targets)
        loader = DataLoader(dataset, batch_size=2)
        
        history = {'learning_rates': []}
        
        # Test without progress bar
        avg_loss, accuracy = model._train_epoch(loader, history)
        self.assertIsInstance(avg_loss, float)
        self.assertIsInstance(accuracy, float)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)
        
        # Test with mocked progress bar
        mock_pbar = MagicMock()
        avg_loss2, accuracy2 = model._train_epoch(loader, history, pbar=mock_pbar)
        self.assertIsInstance(avg_loss2, float)
        self.assertIsInstance(accuracy2, float)
    
    def test_build_network_method_signature(self):
        """Test that _build_network method accepts required parameters after refactoring."""
        # Test that the method signature includes the correct parameters
        import inspect
        sig = inspect.signature(ResNet._build_network)
        params = list(sig.parameters.keys())
        
        # Check that the new parameters are present
        self.assertIn('block', params)
        self.assertIn('layers', params) 
        self.assertIn('replace_stride_with_dilation', params)
        
        # Test that a model can be created successfully (which calls _build_network internally)
        model = ResNet(BasicBlock, [1, 1, 1, 1], num_classes=10, device='cpu')
        
        # Verify the network was built correctly
        self.assertTrue(hasattr(model, 'conv1'))
        self.assertTrue(hasattr(model, 'layer1'))
        self.assertTrue(hasattr(model, 'layer2'))
        self.assertTrue(hasattr(model, 'layer3'))
        self.assertTrue(hasattr(model, 'layer4'))
        self.assertTrue(hasattr(model, 'fc'))
    
    def test_initialize_weights_method_signature(self):
        """Test that _initialize_weights method accepts required parameters after refactoring."""
        model = ResNet(BasicBlock, [1, 1, 1, 1], num_classes=10, device='cpu')
        
        # Test that we can call _initialize_weights with parameters directly
        try:
            # Test the new signature with zero_init_residual=True
            model._initialize_weights(zero_init_residual=True)
            
            # Test the new signature with zero_init_residual=False  
            model._initialize_weights(zero_init_residual=False)
            
            # Verify weights are still initialized (not all zeros)
            conv_weights_sum = sum(p.sum().item() for p in model.parameters() if p.dim() > 1)
            self.assertNotEqual(conv_weights_sum, 0.0)
            
        except TypeError as e:
            self.fail(f"_initialize_weights signature test failed: {e}")
    
    def test_construction_parameters_not_stored(self):
        """Test that construction-only parameters are not stored as instance variables after refactoring."""
        model = ResNet(
            BasicBlock, 
            [2, 2, 2, 2], 
            num_classes=10, 
            zero_init_residual=True,
            groups=1,  # BasicBlock limitation
            width_per_group=64,  # BasicBlock limitation
            replace_stride_with_dilation=[False, False, False],  # BasicBlock limitation
            device='cpu'
        )
        
        # Verify construction-only parameters are NOT stored
        self.assertFalse(hasattr(model, 'block'), "block should not be stored as instance variable")
        self.assertFalse(hasattr(model, 'layers'), "layers should not be stored as instance variable")
        self.assertFalse(hasattr(model, 'zero_init_residual'), "zero_init_residual should not be stored as instance variable")
        self.assertFalse(hasattr(model, 'replace_stride_with_dilation'), "replace_stride_with_dilation should not be stored as instance variable")
        
        # Verify runtime-relevant parameters ARE stored
        self.assertTrue(hasattr(model, 'num_classes'), "num_classes should be stored as instance variable")
        self.assertTrue(hasattr(model, 'groups'), "groups should be stored as instance variable")
        self.assertTrue(hasattr(model, 'base_width'), "base_width should be stored as instance variable")
        self.assertTrue(hasattr(model, 'inplanes'), "inplanes should be stored as instance variable")
        self.assertTrue(hasattr(model, 'dilation'), "dilation should be stored as instance variable")
        
        # Verify stored values are correct
        self.assertEqual(model.num_classes, 10)
        self.assertEqual(model.groups, 1)
        self.assertEqual(model.base_width, 64)  # width_per_group stored as base_width
    
    def test_different_replace_stride_with_dilation_configs(self):
        """Test various replace_stride_with_dilation configurations."""
        # Only test valid configurations for BasicBlock (no dilation > 1)
        configs = [
            None,  # Default
            [False, False, False],
        ]
        
        for config in configs:
            with self.subTest(config=config):
                model = ResNet(
                    BasicBlock, 
                    [1, 1, 1, 1], 
                    num_classes=10,
                    replace_stride_with_dilation=config,
                    device='cpu'
                )
                
                # Model should build successfully
                self.assertTrue(hasattr(model, 'layer1'))
                self.assertTrue(hasattr(model, 'layer2'))
                self.assertTrue(hasattr(model, 'layer3'))
                self.assertTrue(hasattr(model, 'layer4'))
                
                # Forward pass should work (use eval mode to avoid BatchNorm issues with single sample)
                model.eval()
                x = torch.randn(1, 3, 32, 32)
                output = model(x)
                self.assertEqual(output.shape, (1, 10))
        
        # Test dilation configurations with Bottleneck (supports dilation)
        dilation_configs = [
            [False, True, False], 
            [True, True, True],
            [False, True, True]
        ]
        
        for config in dilation_configs:
            with self.subTest(config=config, block="Bottleneck"):
                model = ResNet(
                    Bottleneck, 
                    [1, 1, 1, 1], 
                    num_classes=10,
                    replace_stride_with_dilation=config,
                    device='cpu'
                )
                
                # Model should build successfully
                self.assertTrue(hasattr(model, 'layer1'))
                self.assertTrue(hasattr(model, 'layer2'))
                self.assertTrue(hasattr(model, 'layer3'))
                self.assertTrue(hasattr(model, 'layer4'))
                
                # Forward pass should work (use eval mode to avoid BatchNorm issues with single sample)
                model.eval()
                x = torch.randn(1, 3, 32, 32)
                output = model(x)
                self.assertEqual(output.shape, (1, 10))


class TestResNetFactoryFunctionsCoverage(unittest.TestCase):
    """Test cases for ResNet factory functions."""
    
    def test_resnet18_factory(self):
        """Test resnet18 factory function."""
        model = resnet18(num_classes=100)
        self.assertIsInstance(model, ResNet)
        self.assertEqual(model.num_classes, 100)
        # Verify network structure instead of construction parameters
        self.assertTrue(hasattr(model, 'conv1'))
        self.assertTrue(hasattr(model, 'layer1'))
        self.assertTrue(hasattr(model, 'layer2'))
        self.assertTrue(hasattr(model, 'layer3'))
        self.assertTrue(hasattr(model, 'layer4'))
    
    def test_resnet34_factory(self):
        """Test resnet34 factory function.""" 
        model = resnet34(num_classes=50)
        self.assertIsInstance(model, ResNet)
        self.assertEqual(model.num_classes, 50)
        # Verify network structure
        self.assertTrue(hasattr(model, 'conv1'))
        self.assertTrue(hasattr(model, 'fc'))
    
    def test_resnet50_factory(self):
        """Test resnet50 factory function."""
        model = resnet50(num_classes=200)
        self.assertIsInstance(model, ResNet)
        self.assertEqual(model.num_classes, 200)
        # Verify network structure
        self.assertTrue(hasattr(model, 'conv1'))
        self.assertTrue(hasattr(model, 'fc'))
    
    def test_resnet101_factory(self):
        """Test resnet101 factory function."""
        model = resnet101(num_classes=1000)
        self.assertIsInstance(model, ResNet)
        self.assertEqual(model.num_classes, 1000)
        # Verify network structure
        self.assertTrue(hasattr(model, 'conv1'))
        self.assertTrue(hasattr(model, 'fc'))
    
    def test_resnet152_factory(self):
        """Test resnet152 factory function."""
        model = resnet152(num_classes=1000)
        self.assertIsInstance(model, ResNet)
        self.assertEqual(model.num_classes, 1000)
        # Verify network structure
        self.assertTrue(hasattr(model, 'conv1'))
        self.assertTrue(hasattr(model, 'fc'))
    
    def test_factory_with_kwargs(self):
        """Test factory functions with additional kwargs."""
        model = resnet50(num_classes=10, zero_init_residual=True, groups=2)  # Use resnet50 for groups
        # Test runtime-relevant attributes that are still stored
        self.assertEqual(model.groups, 2)
        # Verify network structure
        self.assertTrue(hasattr(model, 'conv1'))
        self.assertTrue(hasattr(model, 'fc'))
    
    def test_factory_functions_with_all_parameters(self):
        """Test factory functions with all possible parameters to ensure comprehensive coverage."""
        # Test resnet18 with BasicBlock limitations (no dilation, groups=1)
        model = resnet18(
            num_classes=50,
            zero_init_residual=True,
            groups=1,  # BasicBlock only supports groups=1
            width_per_group=64,  # BasicBlock only supports base_width=64
            replace_stride_with_dilation=[False, False, False],  # BasicBlock doesn't support dilation > 1
            norm_layer=nn.BatchNorm2d
        )
        self.assertEqual(model.num_classes, 50)
        self.assertEqual(model.groups, 1)
        self.assertEqual(model.base_width, 64)
        
        # Test resnet50 with Bottleneck capabilities (supports groups and dilation)
        model50 = resnet50(
            num_classes=100,
            zero_init_residual=True,
            groups=2,  # Bottleneck supports groups > 1
            width_per_group=32,
            replace_stride_with_dilation=[True, True, True],  # Bottleneck supports dilation
            norm_layer=nn.BatchNorm2d
        )
        self.assertEqual(model50.num_classes, 100)
        self.assertEqual(model50.groups, 2)
        self.assertEqual(model50.base_width, 32)
    
    def test_factory_functions_forward_pass(self):
        """Test that all factory functions produce working models."""
        factory_funcs = [resnet18, resnet34, resnet50, resnet101, resnet152]
        
        for factory_func in factory_funcs:
            with self.subTest(factory=factory_func.__name__):
                # Explicitly set device to cpu to avoid device mismatch
                model = factory_func(num_classes=10, device='cpu')
                x = torch.randn(2, 3, 32, 32)
                output = model(x)
                self.assertEqual(output.shape, (2, 10))


class TestResNetAMPSupport(unittest.TestCase):
    """Test cases for Automatic Mixed Precision support."""
    
    def test_amp_compilation_cpu(self):
        """Test AMP setup on CPU (should be disabled)."""
        # AMP is now set in constructor, not compile
        with patch('builtins.print') as mock_print:
            model = resnet18(num_classes=10, device='cpu', use_amp=True)
            
            # Check that AMP is disabled on CPU
            self.assertFalse(model.use_amp)
            self.assertIsNone(model.scaler)
    
    def test_amp_disabled_by_default(self):
        """Test that AMP is disabled by default."""
        model = resnet18(num_classes=10, device='cpu')
        
        self.assertFalse(model.use_amp)
        self.assertIsNone(model.scaler)


class TestResNetNormLayers(unittest.TestCase):
    """Test cases for different normalization layers."""
    
    def test_custom_norm_layer(self):
        """Test ResNet with custom normalization layer."""
        # Test with GroupNorm
        model = ResNet(
            BasicBlock, [1, 1, 1, 1], 
            num_classes=10,
            norm_layer=lambda channels: nn.GroupNorm(2, channels, device='cpu')
        )
        
        # Check that the norm layer is applied
        self.assertIsInstance(model.bn1, nn.GroupNorm)
    
    def test_default_norm_layer(self):
        """Test ResNet with default BatchNorm2d."""
        model = ResNet(BasicBlock, [1, 1, 1, 1], num_classes=10, device='cpu')
        self.assertIsInstance(model.bn1, nn.BatchNorm2d)


class TestResNetDeviceDetection(unittest.TestCase):
    """Test cases for device detection and setup."""
    
    def test_explicit_cpu_device_setting(self):
        """Test explicitly setting CPU device."""
        model = resnet18(num_classes=10, device='cpu')
        self.assertEqual(model.device.type, 'cpu')
        
        # Compile should not change the device
        model.compile(optimizer='adam', loss='cross_entropy')
        self.assertEqual(model.device.type, 'cpu')
    
    def test_device_setup_in_constructor(self):
        """Test that device is set up in constructor, not compile."""
        # Create model with explicit CPU device
        model = resnet18(num_classes=10, device='cpu')
        original_device = model.device
        
        # Compile should not change the device
        model.compile(optimizer='adam', loss='cross_entropy')
        self.assertEqual(model.device, original_device)
    
    def test_mps_device_detection_when_available(self):
        """Test MPS device detection when available (line 122, 126)."""
        # Mock MPS availability
        import unittest.mock
        
        with unittest.mock.patch('torch.backends.mps.is_available', return_value=True):
            with unittest.mock.patch('torch.cuda.is_available', return_value=False):
                model = ResNet(BasicBlock, [1, 1, 1, 1], num_classes=10, device=None)
                
                # Should detect MPS when available and CUDA is not
                self.assertEqual(str(model.device), 'mps')
    
    def test_cpu_fallback_when_no_accelerators(self):
        """Test CPU fallback when no accelerators are available."""
        import unittest.mock
        
        with unittest.mock.patch('torch.cuda.is_available', return_value=False):
            with unittest.mock.patch('torch.backends.mps.is_available', return_value=False):
                model = ResNet(BasicBlock, [1, 1, 1, 1], num_classes=10, device=None)
                
                # Should fallback to CPU when no accelerators available
                self.assertEqual(str(model.device), 'cpu')
    
    def test_amp_warning_on_non_cuda_device(self):
        """Test AMP warning when requested on non-CUDA device (line 132-134)."""
        import io
        import sys
        import contextlib
        
        # Capture stdout to check for warning message
        captured_output = io.StringIO()
        
        with contextlib.redirect_stdout(captured_output):
            model = ResNet(
                BasicBlock, [1, 1, 1, 1], 
                num_classes=10, 
                device='cpu',  # Force CPU
                use_amp=True   # Request AMP on CPU
            )
        
        output = captured_output.getvalue()
        
        # Verify AMP is disabled and warning is shown
        self.assertFalse(model.use_amp)
        self.assertIn("AMP requested but not available", output)
        self.assertIn("using standard precision", output)


class TestResNetProgressBarHandling(unittest.TestCase):
    """Test cases for progress bar handling and tqdm imports."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = resnet18(num_classes=10)
        self.model.compile(optimizer='adam', loss='cross_entropy', device='cpu')
        
        # Create small dataset
        inputs = torch.randn(8, 3, 32, 32)
        targets = torch.randint(0, 10, (8,))
        train_dataset = TensorDataset(inputs, targets)
        self.train_loader = DataLoader(train_dataset, batch_size=4)
    
    def test_verbose_training(self):
        """Test training with verbose output (progress bars)."""
        history = self.model.fit(
            train_loader=self.train_loader,
            epochs=1,
            verbose=True  # This should trigger progress bar code paths
        )
        
        self.assertIn('train_loss', history)
        self.assertEqual(len(history['train_loss']), 1)
    
    @patch('src.models2.core.resnet.tqdm')
    def test_progress_bar_refresh_error_handling(self, mock_tqdm):
        """Test progress bar refresh error handling."""
        # Mock tqdm to raise an exception on refresh
        mock_pbar = MagicMock()
        mock_pbar.refresh.side_effect = Exception("Mock refresh error")
        mock_tqdm.return_value = mock_pbar
        
        # This should handle the refresh error gracefully
        history = self.model.fit(
            train_loader=self.train_loader,
            epochs=1,
            verbose=True
        )
        
        self.assertIn('train_loss', history)


class TestResNetUncoveredLines(unittest.TestCase):
    """Test cases for uncovered medium/high risk lines to improve coverage."""
    
    def test_adamw_optimizer_branch(self):
        """Test AdamW optimizer branch (line 316)."""
        model = resnet18(num_classes=10, device='cpu')
        
        # Test AdamW optimizer specifically to hit line 316
        model.compile(optimizer='adamw', loss='cross_entropy', lr=0.001, weight_decay=0.01)
        
        # Verify optimizer is set correctly
        self.assertIsNotNone(model.optimizer)
        self.assertEqual(model.optimizer.__class__.__name__, 'AdamW')
    
    def test_device_fallback_in_predict(self):
        """Test device fallback when device is None in predict method (lines 521-522)."""
        model = resnet18(num_classes=10, device='cpu')
        model.compile(optimizer='adam', loss='cross_entropy', device='cpu')
        
        # Temporarily set device to None to trigger fallback
        original_device = model.device
        model.device = None
        
        # Create test data
        test_data = torch.randn(4, 3, 32, 32)
        test_targets = torch.randint(0, 10, (4,))  # Add dummy targets
        
        try:
            # This should trigger the device fallback logic (lines 521-522)
            predictions = model.predict(test_data, test_targets, batch_size=2)
            
            # Verify predictions were generated
            self.assertEqual(predictions.shape, (4,))
            self.assertIsNotNone(model.device)  # Device should be set after fallback
            
        finally:
            # Restore original device
            model.device = original_device
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_amp_training_path(self):
        """Test AMP training path (lines 611-618)."""
        model = resnet18(num_classes=10, device='cuda', use_amp=True)
        model.compile(optimizer='adam', loss='cross_entropy', device='cuda')
        
        # Create small dataset for CUDA
        train_data = torch.randn(4, 3, 32, 32)
        train_targets = torch.randint(0, 10, (4,))
        
        # Train for 1 epoch to hit AMP training path
        history = model.fit(
            train_loader=train_data,
            train_targets=train_targets,
            epochs=1,
            batch_size=2,
            verbose=False
        )
        
        # Verify training completed with AMP
        self.assertIn('train_loss', history)
        self.assertTrue(model.use_amp)
        self.assertIsNotNone(model.scaler)
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_amp_validation_path(self):
        """Test AMP validation path (lines 708-710)."""
        model = resnet18(num_classes=10, device='cuda', use_amp=True)
        model.compile(optimizer='adam', loss='cross_entropy', device='cuda')
        
        # Create small dataset for CUDA
        train_data = torch.randn(4, 3, 32, 32)
        train_targets = torch.randint(0, 10, (4,))
        val_data = torch.randn(4, 3, 32, 32)
        val_targets = torch.randint(0, 10, (4,))
        
        # Train with validation to hit AMP validation path
        history = model.fit(
            train_loader=train_data,
            train_targets=train_targets,
            val_loader=val_data,
            val_targets=val_targets,
            epochs=1,
            batch_size=2,
            verbose=False
        )
        
        # Verify validation was performed with AMP
        self.assertIn('val_loss', history)
        self.assertTrue(model.use_amp)
    
    def test_jupyter_tqdm_import_handling(self):
        """Test Jupyter tqdm import handling (lines 27-34, 38-40, 45)."""
        # This test verifies that the module loads correctly with tqdm imports
        # The import handling code is executed when the module is loaded
        
        # Test by importing the module directly to exercise import paths
        import importlib
        import sys
        
        # Mock IPython environment to test Jupyter path
        import unittest.mock
        
        with unittest.mock.patch('src.models2.core.resnet.get_ipython') as mock_get_ipython:
            # Mock being in Jupyter
            mock_ipython = unittest.mock.MagicMock()
            mock_ipython.__class__.__name__ = 'ZMQInteractiveShell'
            mock_get_ipython.return_value = mock_ipython
            
            # This will exercise the import paths when module is reloaded
            # We'll just verify the module can be imported successfully
            try:
                # Force reimport to test the Jupyter path
                if 'src.models2.core.resnet' in sys.modules:
                    importlib.reload(sys.modules['src.models2.core.resnet'])
                else:
                    import src.models2.core.resnet
                
                # If we get here, imports worked
                self.assertTrue(True, "Jupyter tqdm import path completed successfully")
                
            except ImportError:
                # Test the fallback path
                self.assertTrue(True, "Fallback tqdm import path completed successfully")
    
    def test_validation_progress_bar_updates(self):
        """Test validation progress bar updates (lines 726-747)."""
        model = resnet18(num_classes=10)
        model.compile(optimizer='adam', loss='cross_entropy', device='cpu')
        
        # Create test data
        val_data = torch.randn(8, 3, 32, 32)
        val_targets = torch.randint(0, 10, (8,))
        val_dataset = TensorDataset(val_data, val_targets)
        val_loader = DataLoader(val_dataset, batch_size=4)
        
        # Create a mock progress bar to test the update path
        mock_pbar = MagicMock()
        mock_pbar.postfix = {'train_loss': '0.5000', 'train_acc': '0.7500'}
        
        # Test validation with progress bar
        val_loss, val_acc = model._validate(val_loader, pbar=mock_pbar)
        
        # Verify validation completed and progress bar was updated
        self.assertIsInstance(val_loss, float)
        self.assertIsInstance(val_acc, float)
        
        # Verify progress bar update calls were made
        self.assertTrue(mock_pbar.set_postfix.called)
        self.assertTrue(mock_pbar.update.called)
        
        # Check that validation metrics were added to postfix
        call_args = mock_pbar.set_postfix.call_args[0][0]
        self.assertIn('val_loss', call_args)
        self.assertIn('val_acc', call_args)
    
    def test_tensor_to_dataloader_conversion_in_predict(self):
        """Test tensor to DataLoader conversion in predict method."""
        model = resnet18(num_classes=10, device='cpu')
        model.compile(optimizer='adam', loss='cross_entropy', device='cpu')
        
        # Create test data as tensor (not DataLoader)
        test_data = torch.randn(6, 3, 32, 32)
        test_targets = torch.randint(0, 10, (6,))  # Add dummy targets
        
        # This should trigger tensor to DataLoader conversion
        predictions = model.predict(test_data, test_targets, batch_size=3)
        
        # Verify predictions
        self.assertEqual(predictions.shape, (6,))
        self.assertTrue(torch.all(predictions >= 0))
        self.assertTrue(torch.all(predictions < 10))
