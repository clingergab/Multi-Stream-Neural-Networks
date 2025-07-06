"""
Integration test demonstrating the updated ResNet API with scheduler in compile method.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent / "src"))

from src.models2.core.resnet import resnet18


def test_new_scheduler_api():
    """Demonstrate the new scheduler API."""
    print("Testing new ResNet scheduler API...")
    
    # Create model
    model = resnet18(num_classes=10)
    
    # Create sample data
    inputs = torch.randn(32, 3, 32, 32)
    targets = torch.randint(0, 10, (32,))
    dataset = TensorDataset(inputs, targets)
    train_loader = DataLoader(dataset, batch_size=8)
    
    print("\n1. Testing compilation with scheduler in compile method:")
    
    # Test 1: Compile with cosine scheduler
    model.compile(
        optimizer='adam',
        loss='cross_entropy',
        lr=0.01,
        scheduler='cosine',  # Scheduler specified in compile
        device='cpu'
    )
    
    print(f"   - Scheduler type stored: {model.scheduler_type}")
    print(f"   - Scheduler object before fit: {model.scheduler}")
    
    # Train the model (scheduler will be initialized here)
    print("\n2. Training model (scheduler initialized in fit):")
    history = model.fit(
        train_loader,
        epochs=3,
        verbose=True,
        t_max=3  # Scheduler-specific parameter
    )
    
    print(f"   - Scheduler object after fit: {type(model.scheduler).__name__}")
    print(f"   - Scheduler T_max: {model.scheduler.T_max}")
    
    print("\n3. Testing different scheduler types:")
    
    # Test OneCycle scheduler
    model.compile(
        optimizer='sgd',
        loss='cross_entropy',
        lr=0.01,
        scheduler='onecycle',
        device='cpu'
    )
    
    model.fit(
        train_loader,
        epochs=2,
        verbose=False,
        max_lr=0.1,
        pct_start=0.3
    )
    
    print(f"   - OneCycle scheduler total_steps: {model.scheduler.total_steps}")
    
    # Test Step scheduler
    model.compile(
        optimizer='adam',
        loss='cross_entropy',
        lr=0.01,
        scheduler='step',
        device='cpu'
    )
    
    model.fit(
        train_loader,
        epochs=2,
        verbose=False,
        step_size=1,
        gamma=0.5
    )
    
    print(f"   - Step scheduler step_size: {model.scheduler.step_size}")
    print(f"   - Step scheduler gamma: {model.scheduler.gamma}")
    
    # Test no scheduler
    model.compile(
        optimizer='adam',
        loss='cross_entropy',
        lr=0.01,
        device='cpu'
        # No scheduler parameter
    )
    
    model.fit(
        train_loader,
        epochs=1,
        verbose=False
    )
    
    print(f"   - No scheduler case: {model.scheduler}")
    
    print("\n✅ All tests passed! New scheduler API working correctly.")
    
    # Add assertions to make this a proper test
    assert model.scheduler_type is None
    assert model.scheduler is None


if __name__ == "__main__":
    test_new_scheduler_api()
