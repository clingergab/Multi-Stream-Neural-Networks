"""
Comprehensive tests for all scheduler types in ResNet models.
"""

import unittest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent / "src"))

from src.models2.core.resnet import resnet18
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import (
    StepLR, 
    CosineAnnealingLR, 
    ReduceLROnPlateau, 
    OneCycleLR
)


class TestResNetSchedulers(unittest.TestCase):
    """Test cases for ResNet scheduler functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a small model for testing
        self.model = resnet18(num_classes=10)
        
        # Create a tiny dataset for testing
        self.num_samples = 16
        self.batch_size = 4
        
        # Random inputs and targets
        self.inputs = torch.randn(self.num_samples, 3, 32, 32)
        self.targets = torch.randint(0, 10, (self.num_samples,))
        
        # Create train dataset
        train_dataset = TensorDataset(self.inputs, self.targets)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size)
    
    def test_step_scheduler(self):
        """Test StepLR scheduler."""
        # Compile with step scheduler
        self.model.compile(
            optimizer='sgd',
            loss='cross_entropy',
            lr=0.1,
            scheduler='step',
            device='cpu'
        )
        
        # Train with specific step scheduler parameters
        history = self.model.fit(
            self.train_loader,
            epochs=3,
            verbose=False,
            step_size=2,  # Step every 2 epochs
            gamma=0.5     # Multiply by 0.5
        )
        
        # Verify scheduler was created
        self.assertIsInstance(self.model.scheduler, StepLR)
        self.assertEqual(self.model.scheduler.step_size, 2)
        self.assertEqual(self.model.scheduler.gamma, 0.5)
    
    def test_cosine_scheduler(self):
        """Test CosineAnnealingLR scheduler."""
        # Compile with cosine scheduler
        self.model.compile(
            optimizer='adam',
            loss='cross_entropy',
            lr=0.01,
            scheduler='cosine',
            device='cpu'
        )
        
        # Train with specific cosine scheduler parameters
        epochs = 5
        history = self.model.fit(
            self.train_loader,
            epochs=epochs,
            verbose=False,
            t_max=epochs  # Should cycle over the number of epochs
        )
        
        # Verify scheduler was created
        self.assertIsInstance(self.model.scheduler, CosineAnnealingLR)
        self.assertEqual(self.model.scheduler.T_max, epochs)
    
    def test_plateau_scheduler(self):
        """Test ReduceLROnPlateau scheduler."""
        # Compile with plateau scheduler
        self.model.compile(
            optimizer='adam',
            loss='cross_entropy',
            lr=0.01,
            scheduler='plateau',
            device='cpu'
        )
        
        # Train with specific plateau scheduler parameters
        history = self.model.fit(
            self.train_loader,
            self.train_loader,  # Use same as val_loader
            epochs=3,
            verbose=False,
            scheduler_patience=2,  # Scheduler patience (renamed to avoid conflict with early stopping)
            factor=0.8
        )
        
        # Verify scheduler was created
        self.assertIsInstance(self.model.scheduler, ReduceLROnPlateau)
        self.assertEqual(self.model.scheduler.patience, 2)
        self.assertEqual(self.model.scheduler.factor, 0.8)
    
    def test_onecycle_scheduler(self):
        """Test OneCycleLR scheduler."""
        # Compile with onecycle scheduler
        self.model.compile(
            optimizer='sgd',
            loss='cross_entropy',
            lr=0.01,
            scheduler='onecycle',
            device='cpu'
        )
        
        epochs = 3
        steps_per_epoch = len(self.train_loader)
        max_lr = 0.1
        
        # Train with specific onecycle scheduler parameters
        history = self.model.fit(
            self.train_loader,
            epochs=epochs,
            verbose=False,
            steps_per_epoch=steps_per_epoch,
            max_lr=max_lr,
            pct_start=0.3,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=1e4
        )
        
        # Verify scheduler was created
        self.assertIsInstance(self.model.scheduler, OneCycleLR)
        self.assertEqual(self.model.scheduler.total_steps, epochs * steps_per_epoch)
    
    def test_no_scheduler(self):
        """Test that no scheduler is created when not specified."""
        # Compile without scheduler
        self.model.compile(
            optimizer='adam',
            loss='cross_entropy',
            lr=0.01,
            device='cpu'
        )
        
        # Train
        history = self.model.fit(
            self.train_loader,
            epochs=2,
            verbose=False
        )
        
        # Verify no scheduler was created
        self.assertIsNone(self.model.scheduler)
        self.assertIsNone(self.model.scheduler_type)
    
    def test_invalid_scheduler(self):
        """Test that invalid scheduler raises error."""
        # Compile with invalid scheduler
        self.model.compile(
            optimizer='adam',
            loss='cross_entropy',
            lr=0.01,
            scheduler='invalid_scheduler',
            device='cpu'
        )
        
        # Training should raise ValueError
        with self.assertRaises(ValueError):
            self.model.fit(
                self.train_loader,
                epochs=1,
                verbose=False
            )
    
    def test_scheduler_state_saving(self):
        """Test that scheduler state is saved in checkpoints."""
        import tempfile
        import os
        
        # Compile with scheduler
        self.model.compile(
            optimizer='adam',
            loss='cross_entropy',
            lr=0.01,
            scheduler='step',
            device='cpu'
        )
        
        # Create a temporary file
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = os.path.join(tmp_dir, "model_checkpoint.pt")
            
            # Train with checkpoint saving
            self.model.fit(
                self.train_loader,
                self.train_loader,  # Use same as val_loader
                epochs=1,
                verbose=False,
                save_path=checkpoint_path,
                step_size=1,
                gamma=0.9
            )
            
            # Load the checkpoint and verify scheduler state is saved
            checkpoint = torch.load(checkpoint_path)
            self.assertIn('scheduler_state_dict', checkpoint)
            
            # If scheduler exists, state dict should not be None
            if self.model.scheduler is not None:
                self.assertIsNotNone(checkpoint['scheduler_state_dict'])


if __name__ == "__main__":
    unittest.main()
