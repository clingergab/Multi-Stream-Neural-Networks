"""
Test script to compare reduced vs full MultiChannelResNetNetwork on CIFAR-100.

This script trains both the reduced and full-sized MultiChannelResNetNetwork models
on CIFAR-100 to evaluate the performance trade-offs and overfitting characteristics.
"""

import torch
import os
import sys
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

# Add project root to path to ensure imports work
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import local modules
from src.models.basic_multi_channel.multi_channel_resnet_network import MultiChannelResNetNetwork
from src.utils.cifar100_loader import get_cifar100_datasets
from src.transforms.rgb_to_rgbl import RGBtoRGBL


class CIFAR100WithRGBL(Dataset):
    """
    Dataset wrapper to transform CIFAR-100 to RGBL format for multi-channel networks.
    """
    def __init__(self, base_dataset, transform=None):
        self.base_dataset = base_dataset
        self.transform = transform or RGBtoRGBL()
        
    def __len__(self):
        return len(self.base_dataset)
        
    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        # Apply RGB to RGBL transformation
        color, brightness = self.transform(image)
        return color, brightness, label


def create_model(reduced=True, device=None):
    """
    Create either a reduced or full-sized MultiChannelResNetNetwork.
    
    Args:
        reduced: Whether to use the reduced architecture for CIFAR-100
        device: The device to put the model on
        
    Returns:
        The initialized model
    """
    # Create the base model
    model = MultiChannelResNetNetwork(
        num_classes=100,  # CIFAR-100 has 100 classes
        color_input_channels=3,  # RGB channels
        brightness_input_channels=1,  # L channel
        num_blocks=[2, 2, 2, 2],  # ResNet-18 style configuration
        block_type='basic',
        activation='relu',
        dropout=0.3,
        use_shared_classifier=True,
        reduce_architecture=reduced,  # Key parameter we're testing
        device='auto' if device is None else device
    )
    
    # Print model configuration summary
    params = sum(p.numel() for p in model.parameters())
    print(f"\n--- Created {'reduced' if reduced else 'full-sized'} MultiChannelResNetNetwork (reduce_architecture={reduced}) ---")
    print(f"   Parameters: {params:,}")
    
    return model


def main():
    # CIFAR-100 data directory
    data_dir = 'data/cifar-100'
    
    # Hyperparameters
    batch_size = 128
    epochs = 5  # Reduced for faster testing
    learning_rate = 0.001
    weight_decay = 0.0001
    gradient_clip = 1.0
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load CIFAR-100 datasets
    try:
        print(f"Loading CIFAR-100 from: {data_dir}")
        train_dataset, test_dataset, class_names = get_cifar100_datasets(data_dir=data_dir)
        
        # Create RGB to RGBL transform
        rgb_to_rgbl = RGBtoRGBL()
        
        # Wrap datasets with RGBL transform
        rgbl_train_dataset = CIFAR100WithRGBL(train_dataset, rgb_to_rgbl)
        rgbl_test_dataset = CIFAR100WithRGBL(test_dataset, rgb_to_rgbl)
        
        # Create data loaders
        train_loader = DataLoader(
            rgbl_train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4 if device.type == 'cuda' else 0,
            pin_memory=True if device.type == 'cuda' else False
        )
        
        test_loader = DataLoader(
            rgbl_test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4 if device.type == 'cuda' else 0,
            pin_memory=True if device.type == 'cuda' else False
        )
        
        print(f"CIFAR-100 dataloaders created with {len(rgbl_train_dataset)} training and {len(rgbl_test_dataset)} test samples.")
        
    except Exception as e:
        print(f"Error loading CIFAR-100: {e}")
        return
    
    # Create reduced and full MultiChannelResNetNetwork models
    reduced_model = create_model(reduced=True, device=device).to(device)
    full_model = create_model(reduced=False, device=device).to(device)
    
    # Compile models
    print("\n--- Compiling Reduced MultiChannelResNetNetwork ---")
    reduced_model.compile(
        optimizer='adamw',
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        loss='cross_entropy',
        gradient_clip=gradient_clip,
        scheduler='onecycle'
    )
    
    print("\n--- Compiling Full MultiChannelResNetNetwork ---")
    full_model.compile(
        optimizer='adamw',
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        loss='cross_entropy',
        gradient_clip=gradient_clip,
        scheduler='onecycle'
    )
    
    # Train Reduced model
    print("\n=== Training Reduced MultiChannelResNetNetwork ===")
    reduced_history = reduced_model.fit(
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=epochs,
        early_stopping_patience=epochs + 1,  # Disable early stopping for this test
        verbose=1
    )
    
    # Train Full model
    print("\n=== Training Full MultiChannelResNetNetwork ===")
    full_history = full_model.fit(
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=epochs,
        early_stopping_patience=epochs + 1,  # Disable early stopping for this test
        verbose=1
    )
    
    # Save models
    torch.save(reduced_model.state_dict(), 'reduced_multichannelresnetnetwork_model.pth')
    torch.save(full_model.state_dict(), 'full_multichannelresnetnetwork_model.pth')
    print("Models saved to reduced_multichannelresnetnetwork_model.pth and full_multichannelresnetnetwork_model.pth")
    
    # Ensure both history dictionaries have all required keys
    required_keys = ['train_loss', 'val_loss', 'train_accuracy', 'val_accuracy']
    for key in required_keys:
        if key not in reduced_history:
            print(f"Warning: '{key}' not found in reduced_history. Using empty list.")
            reduced_history[key] = []
        if key not in full_history:
            print(f"Warning: '{key}' not found in full_history. Using empty list.")
            full_history[key] = []

    # Plot comparison
    plt.figure(figsize=(15, 10))
    
    # Plot training loss
    plt.subplot(2, 2, 1)
    plt.plot(reduced_history['train_loss'], 'g-', label='Reduced')
    plt.plot(full_history['train_loss'], 'b-', label='Full')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot validation loss
    plt.subplot(2, 2, 2)
    plt.plot(reduced_history['val_loss'], 'g-', label='Reduced')
    plt.plot(full_history['val_loss'], 'b-', label='Full')
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot training accuracy
    plt.subplot(2, 2, 3)
    plt.plot(reduced_history['train_accuracy'], 'g-', label='Reduced')
    plt.plot(full_history['train_accuracy'], 'b-', label='Full')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot validation accuracy
    plt.subplot(2, 2, 4)
    plt.plot(reduced_history['val_accuracy'], 'g-', label='Reduced')
    plt.plot(full_history['val_accuracy'], 'b-', label='Full')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    
    # Ensure results directory exists before saving the plot
    if not os.path.exists('results'):
        os.makedirs('results')
        print("Created results directory")

    plt.savefig('results/reduced_vs_full_comparison.png')
    print("Comparison plot saved to results/reduced_vs_full_comparison.png")
    
    # Final evaluation
    reduced_model.eval()
    full_model.eval()
    
    reduced_correct = 0
    full_correct = 0
    total = 0
    
    with torch.no_grad():
        for color, brightness, labels in test_loader:
            # Prepare inputs
            color_input = color.to(device)
            brightness_input = brightness.to(device)
            labels = labels.to(device)
            
            # Forward pass for both models
            reduced_outputs = reduced_model(color_input, brightness_input)
            full_outputs = full_model(color_input, brightness_input)
            
            # Calculate accuracy
            _, reduced_predicted = torch.max(reduced_outputs.data, 1)
            _, full_predicted = torch.max(full_outputs.data, 1)
            
            total += labels.size(0)
            reduced_correct += (reduced_predicted == labels).sum().item()
            full_correct += (full_predicted == labels).sum().item()
    
    reduced_accuracy = 100 * reduced_correct / total
    full_accuracy = 100 * full_correct / total
    
    # Calculate final training accuracies
    reduced_final_train_acc = reduced_history['train_accuracy'][-1] * 100
    full_final_train_acc = full_history['train_accuracy'][-1] * 100
    
    # Calculate final validation accuracies
    reduced_final_val_acc = reduced_history['val_accuracy'][-1] * 100
    full_final_val_acc = full_history['val_accuracy'][-1] * 100
    
    # Calculate the gap between training and validation accuracy as a measure of overfitting
    reduced_gap = reduced_final_train_acc - reduced_final_val_acc
    full_gap = full_final_train_acc - full_final_val_acc
    
    print("\n=== Final Test Results ===")
    print(f"Reduced MultiChannelResNetNetwork accuracy: {reduced_accuracy:.2f}%")
    print(f"Full MultiChannelResNetNetwork accuracy: {full_accuracy:.2f}%")
    
    print("\n=== Overfitting Analysis ===")
    print(f"Reduced ResNet final training accuracy: {reduced_final_train_acc:.2f}%")
    print(f"Reduced ResNet final validation accuracy: {reduced_final_val_acc:.2f}%")
    print(f"Reduced accuracy gap (train-val): {reduced_gap:.2f}%")
    print()
    print(f"Full ResNet final training accuracy: {full_final_train_acc:.2f}%")
    print(f"Full ResNet final validation accuracy: {full_final_val_acc:.2f}%")
    print(f"Full accuracy gap (train-val): {full_gap:.2f}%")
    
    if reduced_gap < full_gap:
        gap_reduction = full_gap - reduced_gap
        print(f"\n✅ The reduced architecture shows {gap_reduction:.2f}% less overfitting than the full architecture")
    else:
        print("\n⚠️ The reduced architecture does not show less overfitting than the full architecture")
        
    if reduced_accuracy > full_accuracy:
        print(f"✅ The reduced architecture performs better on the test set by {reduced_accuracy - full_accuracy:.2f}%")
    elif abs(reduced_accuracy - full_accuracy) < 1.0:
        print("✅ The reduced and full architectures perform similarly on the test set (within 1%)")
    else:
        print(f"⚠️ The full architecture performs better on the test set by {full_accuracy - reduced_accuracy:.2f}%")


if __name__ == "__main__":
    main()
