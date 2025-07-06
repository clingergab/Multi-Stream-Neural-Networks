"""
Test to verify that history is no longer an instance variable but returned from fit().
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent / "src"))

from src.models2.core.resnet import resnet18


def test_history_not_instance_variable():
    """Test that history is not stored as instance variable."""
    print("Testing that history is not an instance variable...")
    
    # Create model
    model = resnet18(num_classes=10)
    
    # Check that model doesn't have history attribute initially
    assert not hasattr(model, 'history'), "Model should not have history attribute"
    print("✓ Model does not have history attribute initially")
    
    # Compile model
    model.compile(optimizer='adam', loss='cross_entropy', device='cpu')
    
    # Check that model still doesn't have history attribute after compilation
    assert not hasattr(model, 'history'), "Model should not have history attribute after compile"
    print("✓ Model does not have history attribute after compile")
    
    # Create sample data
    inputs = torch.randn(16, 3, 32, 32)
    targets = torch.randint(0, 10, (16,))
    dataset = TensorDataset(inputs, targets)
    train_loader = DataLoader(dataset, batch_size=8)
    
    # Train the model and get history
    history = model.fit(
        train_loader,
        epochs=2,
        verbose=False
    )
    
    # Check that model still doesn't have history attribute after training
    assert not hasattr(model, 'history'), "Model should not have history attribute after fit"
    print("✓ Model does not have history attribute after fit")
    
    # Check that history was returned correctly
    assert isinstance(history, dict), "History should be a dictionary"
    assert 'train_loss' in history, "History should contain train_loss"
    assert 'val_loss' in history, "History should contain val_loss"
    assert 'val_accuracy' in history, "History should contain val_accuracy"
    assert 'learning_rates' in history, "History should contain learning_rates"
    print("✓ History returned from fit contains expected keys")
    
    # Check that history has correct number of entries
    assert len(history['train_loss']) == 2, "Should have 2 epochs of train loss"
    print("✓ History has correct number of entries")
    
    print("\n✅ All tests passed! History is properly managed as local variable.")
    
    return True


if __name__ == "__main__":
    test_history_not_instance_variable()
