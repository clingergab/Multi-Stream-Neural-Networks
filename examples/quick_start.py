#!/usr/bin/env python3
"""Quick start example for basic multi-channel neural networks."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import torch
from src.models.basic_multi_channel.multi_channel_model import multi_channel_18
from src.transforms.rgb_to_rgbl import RGBtoRGBL
from src.data_utils.dataloaders import create_train_dataloader


def main():
    """Quick start example."""
    print("Multi-Stream Neural Networks - Quick Start Example")
    print("=" * 50)
    
    # 1. Create a model
    print("1. Creating model...")
    model = create_model(
        model_type='direct_mixing_scalar',
        num_classes=10,
        hidden_dim=64
    )
    print(f"   Created {model.__class__.__name__}")
    
    # 2. Create dummy dataset (replace with real data)
    print("2. Creating dataset...")
    # For demo purposes, we'll create a minimal dataset
    class DummyDataset:
        def __len__(self):
            return 100
        
        def __getitem__(self, idx):
            return {
                'color': torch.randn(3, 32, 32),
                'brightness': torch.randn(1, 32, 32),
                'target': torch.randint(0, 10, (1,)).item()
            }
    
    train_dataset = DummyDataset()
    val_dataset = DummyDataset()
    
    # 3. Create data loaders
    print("3. Creating data loaders...")
    train_loader = MultiStreamDataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = MultiStreamDataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # 4. Create trainer
    print("4. Setting up trainer...")
    trainer = MultiStreamTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer='adam',
        lr=0.001,
        device='cpu'  # Use CPU for demo
    )
    
    # 5. Train for a few epochs
    print("5. Training model...")
    results = trainer.train(num_epochs=3)
    
    print("\nTraining completed!")
    print(f"Final validation accuracy: {results['val_accuracies'][-1]:.2f}%")
    
    # 6. Analyze mixing weights
    print("\n6. Analyzing mixing weights...")
    for name, param in model.named_parameters():
        if name in ['alpha', 'beta', 'gamma']:
            print(f"   {name}: {param.data.item():.4f}")
    
    print("\nQuick start example completed successfully!")


if __name__ == '__main__':
    main()