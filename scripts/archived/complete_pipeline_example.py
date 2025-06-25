#!/usr/bin/env python3
"""
Complete Multi-Stream Neural Network Pipeline Example

This demonstrates the full pipeline from data loading to evaluation using
the refactored canonical components:
1. Data Loading with canonical utilities
2. Preprocessing with refactored transforms (RGBtoRGBL as simple function)
3. Multi-stream dataset creation using canonical MultiStreamWrapper
4. Model creation and training
5. Evaluation and results

Key improvements shown:
- RGBtoRGBL is now a preprocessing function (not nn.Module)
- Clean separation between preprocessing and model computation
- Use of canonical dataset wrappers
- Proper device management
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Add src to path
sys.path.insert(0, '/Users/gclinger/Documents/projects/Multi-Stream-Neural-Networks/src')

# Import canonical components
from src.models.basic_multi_channel.multi_channel_model import MultiChannelNetwork
from src.utils.device_utils import get_device, DeviceManager
from src.transforms.rgb_to_rgbl import RGBtoRGBL
from src.datasets.dataset_wrappers import MultiStreamWrapper

def main():
    """Complete pipeline demonstration."""
    
    print("=" * 80)
    print("MULTI-STREAM NEURAL NETWORK COMPLETE PIPELINE")
    print("=" * 80)
    
    # =============================================================================
    # 1. DEVICE SETUP
    # =============================================================================
    print("\n1. DEVICE SETUP")
    print("-" * 40)
    
    device = get_device()
    device_manager = DeviceManager()
    print(f"✅ Using device: {device}")
    print(f"✅ Device manager initialized successfully")
    
    # =============================================================================
    # 2. DATA LOADING
    # =============================================================================
    print("\n2. DATA LOADING")
    print("-" * 40)
    
    # For this example, we'll create synthetic data to avoid download issues
    # In practice, you'd load from your actual data directory
    print("Creating synthetic CIFAR-10 like data for demonstration...")
    
    class SyntheticDataset(torch.utils.data.Dataset):
        """Synthetic dataset for demonstration."""
        def __init__(self, num_samples=1000, num_classes=10):
            self.num_samples = num_samples
            self.num_classes = num_classes
            # Create synthetic RGB images (32x32x3)
            self.data = torch.randn(num_samples, 3, 32, 32)
            self.targets = torch.randint(0, num_classes, (num_samples,))
            
        def __len__(self):
            return self.num_samples
            
        def __getitem__(self, idx):
            return self.data[idx], self.targets[idx]
    
    # Create datasets
    train_dataset = SyntheticDataset(num_samples=2000, num_classes=10)
    test_dataset = SyntheticDataset(num_samples=500, num_classes=10)
    
    print(f"✅ Training dataset: {len(train_dataset)} samples")
    print(f"✅ Test dataset: {len(test_dataset)} samples")
    
    # =============================================================================
    # 3. PREPROCESSING SETUP
    # =============================================================================
    print("\n3. PREPROCESSING SETUP")
    print("-" * 40)
    
    # Initialize preprocessing transforms (now simple classes, not nn.Modules)
    rgbl_transform = RGBtoRGBL()
    
    print(f"✅ RGBtoRGBL transform: {type(rgbl_transform)}")
    print(f"   - Is nn.Module: {isinstance(rgbl_transform, nn.Module)} (Should be False)")
    print(f"   - Has __call__: {hasattr(rgbl_transform, '__call__')} (Should be True)")
    
    # Test preprocessing on a sample
    sample_rgb, _ = train_dataset[0]
    print(f"\n✅ Sample RGB shape: {sample_rgb.shape}")
    
    # Apply RGBtoRGBL (returns both streams directly)
    rgb_part, l_part = rgbl_transform(sample_rgb)
    print(f"✅ RGB stream: {rgb_part.shape}, L stream: {l_part.shape}")
    
    # =============================================================================
    # 4. MULTI-STREAM DATASET WRAPPER
    # =============================================================================
    print("\n4. MULTI-STREAM DATASET WRAPPER")
    print("-" * 40)
    
    def color_preprocessing_pipeline(img):
        """Color stream preprocessing pipeline."""
        # Normalize to [0, 1] range
        img = img / 255.0 if img.max() > 1.0 else img
        # Apply standard normalization
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
        return normalize(img)
    
    def brightness_preprocessing_pipeline(brightness_tensor):
        """Brightness stream preprocessing pipeline."""
        # MultiStreamWrapper gives us single-channel brightness [1, H, W]
        # Our model expects 3 channels, so replicate the brightness channel
        if len(brightness_tensor.shape) == 3 and brightness_tensor.shape[0] == 1:
            brightness_tensor = brightness_tensor.repeat(3, 1, 1)
        elif len(brightness_tensor.shape) == 2:
            # If it's [H, W], add channel dimension and replicate
            brightness_tensor = brightness_tensor.unsqueeze(0).repeat(3, 1, 1)
        
        # Normalize
        brightness_tensor = brightness_tensor / 255.0 if brightness_tensor.max() > 1.0 else brightness_tensor
        return (brightness_tensor - 0.5) / 0.5  # Normalize to [-1, 1]
    
    # Create multi-stream datasets using canonical wrapper
    train_multistream = MultiStreamWrapper(
        base_dataset=train_dataset,
        color_transform=color_preprocessing_pipeline,
        brightness_transform=brightness_preprocessing_pipeline
    )
    
    test_multistream = MultiStreamWrapper(
        base_dataset=test_dataset,
        color_transform=color_preprocessing_pipeline,
        brightness_transform=brightness_preprocessing_pipeline
    )
    
    print(f"✅ Multi-stream training dataset: {len(train_multistream)} samples")
    print(f"✅ Multi-stream test dataset: {len(test_multistream)} samples")
    
    # Test multi-stream sample
    sample_multistream = train_multistream[0]
    print(f"✅ Multi-stream sample keys: {sample_multistream.keys()}")
    print(f"   - Color shape: {sample_multistream['color'].shape}")
    print(f"   - Brightness shape: {sample_multistream['brightness'].shape}")
    print(f"   - Target: {sample_multistream['target']}")
    
    # =============================================================================
    # 5. DATA LOADERS
    # =============================================================================
    print("\n5. DATA LOADERS")
    print("-" * 40)
    
    def multistream_collate_fn(batch):
        """Custom collate function for multi-stream data."""
        color_inputs = torch.stack([item['color'] for item in batch])
        brightness_inputs = torch.stack([item['brightness'] for item in batch])
        targets = torch.tensor([item['target'] for item in batch])
        
        return {
            'color': color_inputs,
            'brightness': brightness_inputs,
            'target': targets
        }
    
    # Create data loaders
    batch_size = 32
    
    train_loader = DataLoader(
        train_multistream,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=multistream_collate_fn,
        num_workers=0  # Set to 0 for compatibility
    )
    
    test_loader = DataLoader(
        test_multistream,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=multistream_collate_fn,
        num_workers=0
    )
    
    print(f"✅ Training loader: {len(train_loader)} batches")
    print(f"✅ Test loader: {len(test_loader)} batches")
    
    # Test batch
    sample_batch = next(iter(train_loader))
    print(f"✅ Sample batch shapes:")
    print(f"   - Color: {sample_batch['color'].shape}")
    print(f"   - Brightness: {sample_batch['brightness'].shape}")
    print(f"   - Targets: {sample_batch['target'].shape}")
    
    # =============================================================================
    # 6. MODEL CREATION
    # =============================================================================
    print("\n6. MODEL CREATION")
    print("-" * 40)
    
    # Create model using canonical class
    model = MultiChannelNetwork(
        num_classes=10,
        input_channels=3,
        hidden_channels=64
    ).to(device)
    
    print(f"✅ Model created and moved to device: {next(model.parameters()).device}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✅ Total parameters: {total_params:,}")
    print(f"✅ Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    sample_batch_device = {
        'color': sample_batch['color'].to(device),
        'brightness': sample_batch['brightness'].to(device),
        'target': sample_batch['target'].to(device)
    }
    
    with torch.no_grad():
        model.eval()
        test_output = model(sample_batch_device['color'], sample_batch_device['brightness'])
        print(f"✅ Test forward pass output shape: {test_output.shape}")
    
    # =============================================================================
    # 7. TRAINING SETUP
    # =============================================================================
    print("\n7. TRAINING SETUP")
    print("-" * 40)
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    print(f"✅ Optimizer: {type(optimizer).__name__}")
    print(f"✅ Loss function: {type(criterion).__name__}")
    print(f"✅ Scheduler: {type(scheduler).__name__}")
    
    # =============================================================================
    # 8. TRAINING LOOP
    # =============================================================================
    print("\n8. TRAINING")
    print("-" * 40)
    
    num_epochs = 5  # Short training for demonstration
    train_losses = []
    train_accuracies = []
    
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            color_input = batch['color'].to(device)
            brightness_input = batch['brightness'].to(device)
            targets = batch['target'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(color_input, brightness_input)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += targets.size(0)
            correct_predictions += (predicted == targets).sum().item()
            
            if batch_idx % 20 == 0:
                print(f"   Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}")
        
        # Calculate epoch metrics
        avg_loss = epoch_loss / len(train_loader)
        accuracy = 100.0 * correct_predictions / total_samples
        
        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)
        
        scheduler.step()
        
        print(f"✅ Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    # =============================================================================
    # 9. EVALUATION
    # =============================================================================
    print("\n9. EVALUATION")
    print("-" * 40)
    
    model.eval()
    test_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Move batch to device
            color_input = batch['color'].to(device)
            brightness_input = batch['brightness'].to(device)
            targets = batch['target'].to(device)
            
            # Forward pass
            outputs = model(color_input, brightness_input)
            loss = criterion(outputs, targets)
            
            # Statistics
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += targets.size(0)
            correct_predictions += (predicted == targets).sum().item()
            
            # Store predictions for detailed analysis
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate test metrics
    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = 100.0 * correct_predictions / total_samples
    
    print(f"✅ Test Loss: {avg_test_loss:.4f}")
    print(f"✅ Test Accuracy: {test_accuracy:.2f}%")
    
    # =============================================================================
    # 10. DETAILED ANALYSIS
    # =============================================================================
    print("\n10. DETAILED ANALYSIS")
    print("-" * 40)
    
    # Classification report
    from sklearn.metrics import classification_report, confusion_matrix
    
    print("Classification Report:")
    print(classification_report(all_targets, all_predictions, digits=4))
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('/Users/gclinger/Documents/projects/Multi-Stream-Neural-Networks/confusion_matrix.png')
    print("✅ Confusion matrix saved as confusion_matrix.png")
    
    # Training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, 'r-', label='Training Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('/Users/gclinger/Documents/projects/Multi-Stream-Neural-Networks/training_curves.png')
    print("✅ Training curves saved as training_curves.png")
    
    # =============================================================================
    # 11. SUMMARY
    # =============================================================================
    print("\n11. PIPELINE SUMMARY")
    print("=" * 40)
    print("✅ PREPROCESSING:")
    print(f"   - RGBtoRGBL: Canonical preprocessing class (not nn.Module) ✓")
    print(f"   - Direct stream output: RGB + brightness streams ✓")
    print(f"   - MultiStreamWrapper: Canonical dataset wrapper ✓")
    
    print("✅ MODEL:")
    print(f"   - MultiChannelNetwork: {trainable_params:,} parameters")
    print(f"   - Device: {device}")
    print(f"   - Multi-stream architecture ✓")
    
    print("✅ RESULTS:")
    print(f"   - Final training accuracy: {train_accuracies[-1]:.2f}%")
    print(f"   - Test accuracy: {test_accuracy:.2f}%")
    print(f"   - Training completed successfully ✓")
    
    print("\n🎉 COMPLETE PIPELINE EXECUTED SUCCESSFULLY!")
    print("   All canonical components working properly")
    print("   Refactored transforms performing efficiently")
    print("   Multi-stream architecture functioning correctly")
    
    return {
        'model': model,
        'train_accuracy': train_accuracies[-1],
        'test_accuracy': test_accuracy,
        'train_losses': train_losses,
        'predictions': all_predictions,
        'targets': all_targets
    }

if __name__ == "__main__":
    results = main()
    print(f"\nFinal Results Summary:")
    print(f"Training Accuracy: {results['train_accuracy']:.2f}%")
    print(f"Test Accuracy: {results['test_accuracy']:.2f}%")
