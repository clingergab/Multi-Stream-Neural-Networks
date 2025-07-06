"""
Tests for ResNet compile, fit, predict, and evaluate methods.
"""

import unittest
import torch
import sys
from pathlib import Path
import tempfile
import os

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent / "src"))

from src.models2.core.resnet import resnet18
from torch.utils.data import DataLoader, TensorDataset


class TestResNetTrainingAPI(unittest.TestCase):
    """Test cases for ResNet training API methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a small model for testing
        self.model = resnet18(num_classes=10)
        
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
        # Test basic compilation
        self.model.compile(optimizer='adam', loss='cross_entropy', device='cpu')
        
        # Verify components were created
        self.assertIsNotNone(self.model.optimizer)
        self.assertIsNotNone(self.model.criterion)
        self.assertEqual(self.model.device, torch.device('cpu'))
        
        # Test with scheduler - scheduler is set in compile but scheduler object is created in fit
        self.model.compile(
            optimizer='sgd', 
            loss='cross_entropy',
            lr=0.01, 
            scheduler='step',
            device='cpu'
        )
        self.assertEqual(self.model.scheduler_type, 'step')
        
        # Test cosine scheduler type is stored
        self.model.compile(
            optimizer='adam',
            loss='cross_entropy',
            lr=0.01,
            scheduler='cosine',
            device='cpu'
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
        
        # Check that we have the right number of entries
        self.assertEqual(len(history['train_loss']), 2)
        self.assertEqual(len(history['train_accuracy']), 2)
        self.assertEqual(len(history['val_loss']), 2)
        self.assertEqual(len(history['val_accuracy']), 2)
        self.assertEqual(len(history['learning_rates']), 2)
    
    def test_scheduler_initialization_in_fit(self):
        """Test that schedulers are properly initialized in fit method."""
        # Test with step scheduler
        self.model.compile(
            optimizer='adam', 
            loss='cross_entropy', 
            scheduler='step',
            device='cpu'
        )
        
        # Scheduler should be None before fit
        self.assertIsNone(self.model.scheduler)
        
        # After fit, scheduler should be initialized
        self.model.fit(
            self.train_loader,
            epochs=1,
            verbose=False,
            step_size=1,  # scheduler kwargs
            gamma=0.5
        )
        
        # Now scheduler should exist
        self.assertIsNotNone(self.model.scheduler)
        self.assertIsInstance(self.model.scheduler, torch.optim.lr_scheduler.StepLR)
        
        # Test with cosine scheduler
        self.model.compile(
            optimizer='adam', 
            loss='cross_entropy', 
            scheduler='cosine',
            device='cpu'
        )
        
        self.model.fit(
            self.train_loader,
            epochs=2,
            verbose=False,
            t_max=2  # scheduler kwargs
        )
        
        self.assertIsInstance(self.model.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
        self.assertEqual(self.model.scheduler.T_max, 2)
        
        # Test with no scheduler
        self.model.compile(
            optimizer='adam', 
            loss='cross_entropy',
            device='cpu'
        )
        
        self.model.fit(
            self.train_loader,
            epochs=1,
            verbose=False
        )
        
        # Scheduler should remain None
        self.assertIsNone(self.model.scheduler)
    
    def test_predict_method(self):
        """Test the predict method."""
        self.model.compile(optimizer='adam', loss='cross_entropy', device='cpu')
        
        # Get predictions
        predictions = self.model.predict(self.train_loader)
        
        # Check shape
        self.assertEqual(predictions.shape, (self.num_samples,))
        
        # Check data type
        self.assertTrue(predictions.dtype == torch.int64)
    
    def test_evaluate_method(self):
        """Test the evaluate method."""
        self.model.compile(optimizer='adam', loss='cross_entropy', device='cpu')
        
        # Evaluate model
        metrics = self.model.evaluate(self.train_loader)
        
        # Check metrics
        self.assertIn('loss', metrics)
        self.assertIn('accuracy', metrics)
        
        # Loss should be a float
        self.assertIsInstance(metrics['loss'], float)
        
        # Accuracy should be a float
        self.assertIsInstance(metrics['accuracy'], float)
    
    def test_checkpoint_saving(self):
        """Test saving checkpoints."""
        self.model.compile(optimizer='adam', loss='cross_entropy', device='cpu')
        
        # Create a temporary file
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = os.path.join(tmp_dir, "model/checkpoint.pt")
            
            # Train with checkpoint saving
            self.model.fit(
                self.train_loader,
                self.val_loader,
                epochs=1,
                verbose=False,
                save_path=checkpoint_path
            )
            
            # Check that the file exists
            self.assertTrue(os.path.exists(checkpoint_path))
            
            # Load the checkpoint and verify
            checkpoint = torch.load(checkpoint_path)
            self.assertIn('model_state_dict', checkpoint)
            self.assertIn('optimizer_state_dict', checkpoint)
            self.assertIn('history', checkpoint)


if __name__ == "__main__":
    unittest.main()
