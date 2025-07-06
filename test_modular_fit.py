#!/usr/bin/env python3
"""
Quick test to verify the modularized fit method works correctly.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models2.core.resnet import resnet18

def test_modular_fit():
    """Test that the modularized fit method works correctly."""
    print("Testing modularized fit method...")
    
    # Create a simple dataset
    X = torch.randn(100, 3, 32, 32)
    y = torch.randint(0, 10, (100,))
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    # Create model
    model = resnet18(num_classes=10)
    
    # Compile model
    model.compile(
        optimizer='adam',
        criterion='crossentropy',
        scheduler='step'
    )
    
    # Train for 2 epochs
    history = model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=2,
        verbose=True,
        step_size=1,
        gamma=0.5
    )
    
    # Verify history structure
    expected_keys = ['train_loss', 'val_loss', 'train_accuracy', 'val_accuracy', 'learning_rates']
    for key in expected_keys:
        assert key in history, f"Missing key: {key}"
        print(f"✓ {key}: {len(history[key])} entries")
    
    # Verify train_accuracy is included
    assert len(history['train_accuracy']) == 2, "Should have train_accuracy for 2 epochs"
    assert all(0 <= acc <= 1 for acc in history['train_accuracy']), "Train accuracy should be between 0 and 1"
    
    print("✓ All tests passed! Modularized fit method works correctly.")
    print(f"✓ Train accuracies: {[f'{acc:.3f}' for acc in history['train_accuracy']]}")
    return history

if __name__ == "__main__":
    history = test_modular_fit()
