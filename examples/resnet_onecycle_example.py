"""
Example script demonstrating how to use ResNet with OneCycleLR scheduler.

This script shows how to:
1. Create a ResNet model
2. Compile it with the OneCycleLR learning rate scheduler
3. Train it on CIFAR-10 data
4. Plot the learning rate over time
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
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
    
    # Create data loaders
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Calculate steps_per_epoch
    steps_per_epoch = len(train_loader)
    print(f"Steps per epoch: {steps_per_epoch}")
    
    # Create model
    print("Creating ResNet-18 model...")
    model = resnet18(num_classes=10)
    
    # Learning rate parameters
    initial_lr = 0.001
    max_lr = 0.01  # 10x initial learning rate
    
    # Compile model with OneCycleLR scheduler
    print("Compiling model with OneCycleLR scheduler...")
    model.compile(
        optimizer='sgd',
        loss='cross_entropy',
        lr=initial_lr,
        weight_decay=1e-4,
        scheduler_type='onecycle',
        steps_per_epoch=steps_per_epoch,
        epochs=5,
        max_lr=max_lr,
        pct_start=0.3,  # First 30% of training increases learning rate
        div_factor=25.0,  # initial_lr = max_lr/25
        final_div_factor=1e4,  # final_lr = initial_lr/10000
        device=str(device)
    )
    
    # For tracking learning rates
    lrs = []
    
    # Create a hook to record learning rates
    def lr_recorder(engine):
        lrs.append(model.optimizer.param_groups[0]['lr'])
    
    # Create a simple callback
    class LRCallback:
        def on_epoch_end(self, epoch, logs):
            current_lr = model.optimizer.param_groups[0]['lr']
            print(f"Learning rate at epoch {epoch+1} end: {current_lr:.6f}")
    
    # Train model with learning rate tracking
    print("Training model...")
    
    # Train for just a few epochs to demonstrate the scheduler
    history = model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=5,
        callbacks=[LRCallback()],
        save_path='saved_models/resnet18_onecycle.pt'
    )
    
    # Get the tracked learning rates (always available in history)
    lrs = history['learning_rates']
    
    # Calculate steps for each epoch to plot learning rate
    steps = []
    for i in range(steps_per_epoch * 5):
        steps.append(i+1)
    
    # Plot the learning rate schedule
    plt.figure(figsize=(10, 5))
    plt.plot(steps[:len(lrs)], lrs)
    plt.title("OneCycleLR Learning Rate Schedule")
    plt.xlabel("Training Steps")
    plt.ylabel("Learning Rate")
    plt.grid()
    plt.savefig("onecycle_lr_schedule.png")
    plt.show()
    
    print("Training complete!")
    print(f"Results saved to saved_models/resnet18_onecycle.pt")
    print(f"Learning rate schedule plot saved to onecycle_lr_schedule.png")


if __name__ == "__main__":
    main()
