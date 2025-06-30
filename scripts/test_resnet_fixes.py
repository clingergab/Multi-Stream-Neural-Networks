"""
Test script to evaluate MultiChannelResNetNetwork on CIFAR-100.

This script trains both MultiChannelResNetNetwork and BaseMultiChannelNetwork
using the updated architecture and training configuration to compare performance
and verify that the learning issues have been fixed.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path to ensure imports work
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import local modules
from src.models.basic_multi_channel.multi_channel_resnet_network import MultiChannelResNetNetwork
from src.models.basic_multi_channel.base_multi_channel_network import BaseMultiChannelNetwork
from src.utils.cifar100_loader import get_cifar100_datasets
from src.transforms.rgb_to_rgbl import RGBtoRGBL
from torch.utils.data import DataLoader, Dataset


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
    
    # Create MultiChannelResNetNetwork using factory method
    resnet_model = MultiChannelResNetNetwork.for_cifar100(
        use_shared_classifier=True,
        dropout=0.3,
        block_type='basic'
    ).to(device)
    
    # Create BaseMultiChannelNetwork for comparison
    base_model = BaseMultiChannelNetwork(
        color_input_size=3072,  # 32x32x3 flattened
        brightness_input_size=1024,  # 32x32x1 flattened
        hidden_sizes=[512, 256, 128],
        num_classes=100,
        dropout=0.3,
        use_shared_classifier=True,
        device=device
    ).to(device)
    
    # Compile models
    print("\n--- Compiling MultiChannelResNetNetwork ---")
    resnet_model.compile(
        optimizer='adamw',
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        loss='cross_entropy',
        gradient_clip=gradient_clip,
        scheduler='onecycle'
    )
    
    # Check if BaseMultiChannelNetwork has compile method
    if hasattr(base_model, 'compile'):
        print("\n--- Compiling BaseMultiChannelNetwork ---")
        base_model.compile(
            optimizer='adamw',
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            loss='cross_entropy'
        )
    else:
        print("\n--- Setting up optimizer for BaseMultiChannelNetwork ---")
        base_optimizer = optim.AdamW(base_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        base_criterion = nn.CrossEntropyLoss()
    
    # Train ResNet model
    print("\n=== Training MultiChannelResNetNetwork ===")
    resnet_history = resnet_model.fit(
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=epochs,
        early_stopping_patience=epochs + 1,  # Disable early stopping for this test
        verbose=1
    )
    
    # Ensure resnet_history has all required keys
    if not isinstance(resnet_history, dict):
        print("Warning: resnet_model.fit() didn't return a history dictionary. Creating one.")
        resnet_history = {
            'train_loss': [0.0] * epochs,
            'train_accuracy': [0.0] * epochs,
            'val_loss': [0.0] * epochs,
            'val_accuracy': [0.0] * epochs
        }
    else:
        # Make sure all required keys exist
        for key in ['train_loss', 'train_accuracy', 'val_loss', 'val_accuracy']:
            if key not in resnet_history:
                print(f"Warning: '{key}' not found in resnet_history. Adding empty list.")
                resnet_history[key] = [0.0] * epochs
    
    # Train Base model
    print("\n=== Training BaseMultiChannelNetwork ===")
    if hasattr(base_model, 'fit'):
        base_history = base_model.fit(
            train_loader=train_loader,
            val_loader=test_loader,
            epochs=epochs,
            early_stopping_patience=epochs + 1,  # Disable early stopping for this test
            verbose=1
        )
        
        # Ensure base_history has all required keys
        if not isinstance(base_history, dict):
            print("Warning: base_model.fit() didn't return a history dictionary. Creating one.")
            base_history = {
                'train_loss': [0.0] * epochs,
                'train_accuracy': [0.0] * epochs,
                'val_loss': [0.0] * epochs,
                'val_accuracy': [0.0] * epochs
            }
        else:
            # Make sure all required keys exist
            for key in ['train_loss', 'train_accuracy', 'val_loss', 'val_accuracy']:
                if key not in base_history:
                    print(f"Warning: '{key}' not found in base_history. Adding empty list.")
                    base_history[key] = [0.0] * epochs
    else:
        # Manual training loop for BaseMultiChannelNetwork
        base_history = {'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': []}
        
        for epoch in range(epochs):
            # Training phase
            base_model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for color, brightness, labels in train_loader:
                # Flatten inputs for BaseMultiChannelNetwork
                color_flat = color.view(color.size(0), -1).to(device)
                brightness_flat = brightness.view(brightness.size(0), -1).to(device)
                labels = labels.to(device)
                
                # Forward pass
                base_optimizer.zero_grad()
                outputs = base_model(color_flat, brightness_flat)
                loss = base_criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                base_optimizer.step()
                
                # Track metrics
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # Calculate training metrics
            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = train_correct / train_total
            
            # Validation phase
            base_model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for color, brightness, labels in test_loader:
                    # Flatten inputs
                    color_flat = color.view(color.size(0), -1).to(device)
                    brightness_flat = brightness.view(brightness.size(0), -1).to(device)
                    labels = labels.to(device)
                    
                    # Forward pass
                    outputs = base_model(color_flat, brightness_flat)
                    loss = base_criterion(outputs, labels)
                    
                    # Track metrics
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            # Calculate validation metrics
            avg_val_loss = val_loss / len(test_loader)
            val_accuracy = val_correct / val_total
            
            # Store history - ensure keys exist before appending
            if 'train_loss' not in base_history:
                print("Warning: 'train_loss' not found in base_history. Adding empty list.")
                base_history['train_loss'] = []
            if 'train_accuracy' not in base_history:
                print("Warning: 'train_accuracy' not found in base_history. Adding empty list.")
                base_history['train_accuracy'] = []
            if 'val_loss' not in base_history:
                print("Warning: 'val_loss' not found in base_history. Adding empty list.")
                base_history['val_loss'] = []
            if 'val_accuracy' not in base_history:
                print("Warning: 'val_accuracy' not found in base_history. Adding empty list.")
                base_history['val_accuracy'] = []
                
            base_history['train_loss'].append(avg_train_loss)
            base_history['train_accuracy'].append(train_accuracy)
            base_history['val_loss'].append(avg_val_loss)
            base_history['val_accuracy'].append(val_accuracy)
            
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"loss: {avg_train_loss:.4f}, acc: {train_accuracy:.4f}, "
                  f"val_loss: {avg_val_loss:.4f}, val_acc: {val_accuracy:.4f}")
    
    # Save models
    torch.save(resnet_model.state_dict(), 'fixed_multichannelresnetnetwork_model.pth')
    torch.save(base_model.state_dict(), 'base_multichannelnetwork_model.pth')
    print("Models saved to fixed_multichannelresnetnetwork_model.pth and base_multichannelnetwork_model.pth")
    
    # Ensure both history dictionaries have all required keys
    required_keys = ['train_loss', 'val_loss', 'train_accuracy', 'val_accuracy']
    for key in required_keys:
        if key not in resnet_history:
            print(f"Warning: '{key}' not found in resnet_history. Using empty list.")
            resnet_history[key] = []
        if key not in base_history:
            print(f"Warning: '{key}' not found in base_history. Using empty list.")
            base_history[key] = []

    # Plot comparison
    plt.figure(figsize=(15, 5))
    
    # Plot training loss
    plt.subplot(1, 3, 1)
    plt.plot(resnet_history['train_loss'], 'b-', label='ResNet')
    plt.plot(base_history['train_loss'], 'r-', label='Base')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot validation loss
    plt.subplot(1, 3, 2)
    plt.plot(resnet_history['val_loss'], 'b-', label='ResNet')
    plt.plot(base_history['val_loss'], 'r-', label='Base')
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot validation accuracy
    plt.subplot(1, 3, 3)
    plt.plot(resnet_history['val_accuracy'], 'b-', label='ResNet')
    plt.plot(base_history['val_accuracy'], 'r-', label='Base')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    
    # Ensure results directory exists before saving the plot
    if not os.path.exists('results'):
        os.makedirs('results')
        print("Created results directory")

    plt.savefig('results/model_comparison.png')
    print("Comparison plot saved to results/model_comparison.png")
    
    # Final evaluation
    resnet_model.eval()
    base_model.eval()
    
    resnet_correct = 0
    base_correct = 0
    total = 0
    
    with torch.no_grad():
        for color, brightness, labels in test_loader:
            # Prepare inputs
            color_cnn = color.to(device)
            brightness_cnn = brightness.to(device)
            color_flat = color.view(color.size(0), -1).to(device)
            brightness_flat = brightness.view(brightness.size(0), -1).to(device)
            labels = labels.to(device)
            
            # Forward pass for both models
            resnet_outputs = resnet_model(color_cnn, brightness_cnn)
            base_outputs = base_model(color_flat, brightness_flat)
            
            # Calculate accuracy
            _, resnet_predicted = torch.max(resnet_outputs.data, 1)
            _, base_predicted = torch.max(base_outputs.data, 1)
            
            total += labels.size(0)
            resnet_correct += (resnet_predicted == labels).sum().item()
            base_correct += (base_predicted == labels).sum().item()
    
    resnet_accuracy = 100 * resnet_correct / total
    base_accuracy = 100 * base_correct / total
    
    print("\n=== Final Test Results ===")
    print(f"MultiChannelResNetNetwork accuracy: {resnet_accuracy:.2f}%")
    print(f"BaseMultiChannelNetwork accuracy: {base_accuracy:.2f}%")
    
    if resnet_accuracy > base_accuracy:
        print("✅ Success! The MultiChannelResNetNetwork is now learning better than BaseMultiChannelNetwork.")
    elif abs(resnet_accuracy - base_accuracy) < 2.0:  # Within 2%
        print("✅ Success! The MultiChannelResNetNetwork is now learning on par with BaseMultiChannelNetwork.")
    else:
        print("❌ The fixes may not be fully effective yet. MultiChannelResNetNetwork is still learning worse than BaseMultiChannelNetwork.")


if __name__ == "__main__":
    main()
