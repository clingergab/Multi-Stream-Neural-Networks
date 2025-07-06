#!/usr/bin/env python3
"""
Test tensor input functionality for ResNet model.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models2.core.resnet import resnet18

def test_tensor_inputs():
    """Test that the ResNet model can handle tensor inputs as well as DataLoaders."""
    print("Testing tensor input functionality...")
    
    # Create test data
    X_train = torch.randn(100, 3, 32, 32)
    y_train = torch.randint(0, 10, (100,))
    X_val = torch.randn(50, 3, 32, 32)
    y_val = torch.randint(0, 10, (50,))
    X_test = torch.randn(20, 3, 32, 32)
    y_test = torch.randint(0, 10, (20,))
    
    # Create model
    model = resnet18(num_classes=10)
    model.compile(optimizer='adam', loss='cross_entropy', lr=0.001)
    
    print("\n1. Testing fit() with tensor inputs...")
    # Test fit with tensors
    history = model.fit(
        train_loader=X_train,
        train_targets=y_train,
        val_loader=X_val,
        val_targets=y_val,
        epochs=2,
        batch_size=16,
        verbose=True
    )
    
    # Verify history structure
    expected_keys = ['train_loss', 'val_loss', 'train_accuracy', 'val_accuracy', 'learning_rates']
    for key in expected_keys:
        assert key in history, f"Missing key: {key}"
        print(f"✓ {key}: {len(history[key])} entries")
    
    print("\n2. Testing predict() with tensor inputs...")
    # Test predict with tensor
    predictions = model.predict(X_test, batch_size=8)
    assert predictions.shape == (20,), f"Expected shape (20,), got {predictions.shape}"
    print(f"✓ Predictions shape: {predictions.shape}")
    print(f"✓ Sample predictions: {predictions[:5]}")
    
    print("\n3. Testing evaluate() with tensor inputs...")
    # Test evaluate with tensors
    eval_results = model.evaluate(X_test, y_test, batch_size=8)
    assert 'loss' in eval_results and 'accuracy' in eval_results
    print(f"✓ Evaluation results: Loss={eval_results['loss']:.4f}, Accuracy={eval_results['accuracy']:.2f}%")
    
    print("\n4. Testing mixed inputs (DataLoader + tensors)...")
    # Create DataLoader for training
    train_dataset = TensorDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    # Use DataLoader for training, tensor for validation
    history_mixed = model.fit(
        train_loader=train_dataloader,
        val_loader=X_val,
        val_targets=y_val,
        epochs=1,
        verbose=True
    )
    
    print(f"✓ Mixed input training completed")
    
    print("\n5. Testing error handling...")
    # Test error cases
    try:
        model.fit(X_train, epochs=1)  # Missing train_targets
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Correctly caught error: {e}")
    
    try:
        model.evaluate(X_test)  # Missing targets
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Correctly caught error: {e}")
    
    print("\n✅ All tensor input tests passed!")
    return history, predictions, eval_results

if __name__ == "__main__":
    history, predictions, eval_results = test_tensor_inputs()
