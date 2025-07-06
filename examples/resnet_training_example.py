"""
Example script demonstrating how to use the ResNet training API.

This script shows how to:
1. Create a ResNet model
2. Compile it with an optimizer and loss function
3. Train it on CIFAR-10 data
4. Evaluate it on test data
5. Use it to make predictions
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models2.core.resnet import resnet18


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define transformations
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Load CIFAR-10 dataset
    print("Loading CIFAR-10 dataset...")
    
    # Training dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True,
        download=True, 
        transform=transform_train
    )
    
    # Split into training and validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    # Test dataset
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False,
        download=True, 
        transform=transform_test
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
    
    # Create model
    print("Creating ResNet-18 model...")
    model = resnet18(num_classes=10)
    
    # Compile model
    print("Compiling model...")
    model.compile(
        optimizer='adam',
        loss='cross_entropy',
        lr=0.001,
        weight_decay=1e-4,
        scheduler_type='cosine',
        # With lr=0.001, t_max would default to 10/0.001 = 10000
        # We'll override it for this example to make it shorter
        t_max=50
    )
    
    # Train model
    print("Training model...")
    history = model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=5,  # Just a few epochs for demonstration
        save_path='saved_models/resnet18_cifar10.pt'
    )
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    metrics = model.evaluate(test_loader)
    print(f"Test loss: {metrics['loss']:.4f}")
    print(f"Test accuracy: {metrics['accuracy']:.2f}%")
    
    # Make predictions
    print("\nMaking predictions on a batch of test data...")
    batch_iter = iter(test_loader)
    images, labels = next(batch_iter)
    
    # Get predictions
    predictions = model.predict(DataLoader(
        torch.utils.data.TensorDataset(images), 
        batch_size=128
    ))
    
    # Print first 10 predictions vs actual
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    print("\nSample predictions:")
    print("Predicted\tActual")
    print("-" * 25)
    for i in range(10):
        pred_class = classes[predictions[i].item()]
        true_class = classes[labels[i].item()]
        print(f"{pred_class}\t\t{true_class}")


if __name__ == "__main__":
    main()
