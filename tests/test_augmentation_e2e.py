#!/usr/bin/env python3
"""
End-to-end test for augmentation pipeline with multi-stream models.

This test validates the complete pipeline using our consolidated augmentation module with
both BaseMultiChannelNetwork and MultiChannelResNetNetwork models.

Tests include:
- Data loading and preprocessing with RGBtoRGBL transform
- Augmentation with CIFAR100Augmentation and MixUp
- AugmentedMultiStreamDataset and DataLoader creation
- Model training with augmented data
- Comparison between augmented and non-augmented training
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Subset
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our utilities and models
from src.utils.cifar100_loader import get_cifar100_datasets, create_validation_split
from src.models.basic_multi_channel.base_multi_channel_network import BaseMultiChannelNetwork
from src.models.basic_multi_channel.multi_channel_resnet_network import MultiChannelResNetNetwork
from src.transforms.rgb_to_rgbl import RGBtoRGBL
from src.transforms.augmentation import (
    CIFAR100Augmentation,
    AugmentedMultiStreamDataset,
    MixUp,
    create_augmented_dataloaders,
    create_test_dataloader
)


class TestConfig:
    """Test configuration."""
    # Data settings
    batch_size = 32
    num_workers = 0  # Set to 0 to avoid multiprocessing issues
    data_root = './data'
    
    # Training settings
    num_epochs = 2  # Short for testing
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Sample sizes
    train_samples = 1000  # Use 1000 training samples for testing
    val_samples = 200    # Use 200 validation samples
    test_samples = 200   # Use 200 test samples
    
    # Augmentation settings
    augmentation_config = {
        'horizontal_flip_prob': 0.5,
        'rotation_degrees': 10.0,
        'translate_range': 0.1,
        'scale_range': (0.9, 1.1),
        'color_jitter_strength': 0.3,
        'gaussian_noise_std': 0.01,
        'cutout_prob': 0.3,
        'cutout_size': 8,
        'enabled': True
    }
    
    # MixUp settings
    mixup_alpha = 0.2


def process_dataset_to_streams(dataset, batch_size=1000, desc="Processing"):
    """
    Convert RGB dataset to RGB + Brightness streams efficiently.
    
    Args:
        dataset: Dataset with RGB images (PyTorch dataset format)
        batch_size: Size of batches for memory-efficient processing
        desc: Description for progress bar
        
    Returns:
        Tuple of (rgb_stream, brightness_stream, labels_tensor)
    """
    rgb_to_rgbl = RGBtoRGBL()
    rgb_tensors = []
    brightness_tensors = []
    labels = []
    
    # Process in batches to manage memory
    for i in tqdm(range(0, len(dataset), batch_size), desc=desc):
        batch_end = min(i + batch_size, len(dataset))
        batch_data = []
        batch_labels = []
        
        # Collect batch data
        for j in range(i, batch_end):
            data, label = dataset[j]
            batch_data.append(data)
            batch_labels.append(label)
        
        # Convert to tensor batch
        batch_tensor = torch.stack(batch_data)
        
        # Apply RGB to RGB+L transform
        rgb_batch, brightness_batch = rgb_to_rgbl(batch_tensor)
        
        rgb_tensors.append(rgb_batch)
        brightness_tensors.append(brightness_batch)
        labels.extend(batch_labels)
    
    # Concatenate all batches
    rgb_stream = torch.cat(rgb_tensors, dim=0)
    brightness_stream = torch.cat(brightness_tensors, dim=0)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    return rgb_stream, brightness_stream, labels_tensor


def prepare_data():
    """
    Prepare CIFAR-100 data with and without augmentation.
    
    Returns:
        Dictionary containing all data loaders and tensors
    """
    print("📊 Loading and processing CIFAR-100 data...")
    
    # Load datasets
    train_dataset, test_dataset, class_names = get_cifar100_datasets(
        data_dir=os.path.join(TestConfig.data_root, 'cifar-100')
    )
    
    # Create validation split
    train_dataset, val_dataset = create_validation_split(train_dataset, val_split=0.1)
    
    # Create subsets for faster testing
    train_indices = list(range(min(TestConfig.train_samples, len(train_dataset))))
    val_indices = list(range(min(TestConfig.val_samples, len(val_dataset))))
    test_indices = list(range(min(TestConfig.test_samples, len(test_dataset))))
    
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    test_subset = Subset(test_dataset, test_indices)
    
    # Convert to RGB+L streams
    print("Converting data to RGB + Brightness streams...")
    train_rgb, train_brightness, train_labels = process_dataset_to_streams(
        train_subset, desc="Processing training data"
    )
    val_rgb, val_brightness, val_labels = process_dataset_to_streams(
        val_subset, desc="Processing validation data"
    )
    test_rgb, test_brightness, test_labels = process_dataset_to_streams(
        test_subset, desc="Processing test data"
    )
    
    # Create dataloaders with augmentation
    print("Creating augmented dataloaders...")
    augmentation = CIFAR100Augmentation(**TestConfig.augmentation_config)
    mixup = MixUp(alpha=TestConfig.mixup_alpha)
    
    # Create augmented dataloaders
    train_loader_aug, val_loader = create_augmented_dataloaders(
        train_rgb, train_brightness, train_labels,
        val_rgb, val_brightness, val_labels,
        batch_size=TestConfig.batch_size,
        dataset="cifar100",
        augmentation_config=TestConfig.augmentation_config,
        mixup_alpha=TestConfig.mixup_alpha,
        num_workers=TestConfig.num_workers,
        pin_memory=TestConfig.device.type == 'cuda'
    )
    
    # Create test dataloader (no augmentation)
    test_loader = create_test_dataloader(
        test_rgb, test_brightness, test_labels,
        batch_size=TestConfig.batch_size,
        num_workers=TestConfig.num_workers,
        pin_memory=TestConfig.device.type == 'cuda'
    )
    
    # Create non-augmented dataloader for comparison
    train_dataset_no_aug = AugmentedMultiStreamDataset(
        train_rgb, train_brightness, train_labels,
        augmentation=None, mixup=None, train=True
    )
    
    train_loader_no_aug = DataLoader(
        train_dataset_no_aug,
        batch_size=TestConfig.batch_size,
        shuffle=True,
        num_workers=TestConfig.num_workers,
        pin_memory=TestConfig.device.type == 'cuda',
        drop_last=True
    )
    
    print(f"✅ Data preparation complete:")
    print(f"   Training samples: {len(train_labels)}")
    print(f"   Validation samples: {len(val_labels)}")
    print(f"   Test samples: {len(test_labels)}")
    print(f"   Augmentation: {augmentation.__class__.__name__}")
    print(f"   MixUp alpha: {TestConfig.mixup_alpha}")
    
    # Return all data
    return {
        'tensors': {
            'train_rgb': train_rgb,
            'train_brightness': train_brightness,
            'train_labels': train_labels,
            'val_rgb': val_rgb,
            'val_brightness': val_brightness,
            'val_labels': val_labels,
            'test_rgb': test_rgb,
            'test_brightness': test_brightness,
            'test_labels': test_labels
        },
        'loaders': {
            'train_aug': train_loader_aug,
            'train_no_aug': train_loader_no_aug,
            'val': val_loader,
            'test': test_loader
        },
        'class_names': class_names
    }


def create_models():
    """
    Create both test models.
    
    Returns:
        Tuple of (base_model, resnet_model)
    """
    print("🏗️ Creating models...")
    
    # Model dimensions for CIFAR-100
    input_channels_rgb = 3
    input_channels_brightness = 1  
    image_size = 32
    num_classes = 100
    
    # For dense model: flatten the image to 1D
    rgb_input_size = input_channels_rgb * image_size * image_size  # 3072
    brightness_input_size = input_channels_brightness * image_size * image_size  # 1024
    
    # Create BaseMultiChannelNetwork (Dense/FC model)
    base_model = BaseMultiChannelNetwork(
        color_input_size=rgb_input_size,
        brightness_input_size=brightness_input_size,
        hidden_sizes=[256, 128],
        num_classes=num_classes,
        use_shared_classifier=True,
        dropout=0.2,
        device=TestConfig.device
    )
    
    # Create MultiChannelResNetNetwork (CNN model)
    resnet_model = MultiChannelResNetNetwork(
        color_input_channels=input_channels_rgb,
        brightness_input_channels=input_channels_brightness,
        num_classes=num_classes,
        use_shared_classifier=True,
        activation='relu',
        device=TestConfig.device
    )
    
    # Count parameters
    base_params = sum(p.numel() for p in base_model.parameters())
    resnet_params = sum(p.numel() for p in resnet_model.parameters())
    
    print(f"✅ Models created:")
    print(f"   BaseMultiChannelNetwork: {base_params:,} parameters")
    print(f"   MultiChannelResNetNetwork: {resnet_params:,} parameters")
    print(f"   Device: {TestConfig.device}")
    
    return base_model, resnet_model


def train_model(model, train_loader, val_loader, use_augmentation, num_epochs=2):
    """
    Train a model and return training metrics.
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        use_augmentation: Whether this is training with augmentation
        num_epochs: Number of epochs to train
    
    Returns:
        Dictionary with training metrics
    """
    optimizer = optim.Adam(model.parameters(), lr=TestConfig.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    metrics = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'aug_status': 'with augmentation' if use_augmentation else 'without augmentation'
    }
    
    model_name = model.__class__.__name__
    aug_text = "w/ aug" if use_augmentation else "w/o aug"
    
    print(f"\n🏋️‍♀️ Training {model_name} {aug_text} for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, data in enumerate(train_bar):
            # Handle both regular data and MixUp data
            if use_augmentation and len(data) == 5:  # MixUp returns 5 items
                rgb, brightness, targets_a, targets_b, lam = data
                has_mixup = True
            else:
                rgb, brightness, targets = data
                has_mixup = False
            
            # Move to device
            rgb, brightness = rgb.to(TestConfig.device), brightness.to(TestConfig.device)
            
            if has_mixup:
                targets_a, targets_b = targets_a.to(TestConfig.device), targets_b.to(TestConfig.device)
            else:
                targets = targets.to(TestConfig.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Process inputs based on model type
            if isinstance(model, BaseMultiChannelNetwork):
                # BaseMultiChannelNetwork needs flattened inputs
                batch_size = rgb.size(0)
                rgb_flat = rgb.view(batch_size, -1)  # Flatten to (batch_size, channels*height*width)
                brightness_flat = brightness.view(batch_size, -1)  # Flatten
                outputs = model(rgb_flat, brightness_flat)
            else:
                # ResNet model can take the tensor directly
                outputs = model(rgb, brightness)
            
            # Calculate loss - handle MixUp case
            if has_mixup:
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                # For accuracy, we'll use the primary target
                _, predicted = outputs.max(1)
                train_total += targets_a.size(0)
                train_correct += (lam * predicted.eq(targets_a).sum().float() 
                                + (1 - lam) * predicted.eq(targets_b).sum().float()).item()
            else:
                loss = criterion(outputs, targets)
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update running loss
            train_loss += loss.item()
            
            # Update progress bar
            train_bar.set_postfix({
                'loss': train_loss/(batch_idx+1), 
                'acc': 100.*train_correct/train_total if train_total > 0 else 0
            })
        
        # Calculate epoch metrics
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total if train_total > 0 else 0
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_idx, (rgb, brightness, targets) in enumerate(val_loader):
                # Move to device
                rgb, brightness, targets = rgb.to(TestConfig.device), brightness.to(TestConfig.device), targets.to(TestConfig.device)
                
                # Process inputs based on model type
                if isinstance(model, BaseMultiChannelNetwork):
                    # BaseMultiChannelNetwork needs flattened inputs
                    batch_size = rgb.size(0)
                    rgb_flat = rgb.view(batch_size, -1)  # Flatten to (batch_size, channels*height*width)
                    brightness_flat = brightness.view(batch_size, -1)  # Flatten
                    outputs = model(rgb_flat, brightness_flat)
                else:
                    # ResNet model can take the tensor directly
                    outputs = model(rgb, brightness)
                
                # Calculate loss
                loss = criterion(outputs, targets)
                
                # Calculate metrics
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        # Calculate epoch metrics
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Store metrics
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        metrics['val_loss'].append(val_loss)
        metrics['val_acc'].append(val_acc)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    return metrics


def test_model(model, test_loader):
    """
    Test a model and return accuracy.
    
    Args:
        model: The model to test
        test_loader: DataLoader for test data
    
    Returns:
        Test accuracy
    """
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for rgb, brightness, targets in tqdm(test_loader, desc="Testing"):
            # Move to device
            rgb, brightness, targets = rgb.to(TestConfig.device), brightness.to(TestConfig.device), targets.to(TestConfig.device)
            
            # Process inputs based on model type
            if isinstance(model, BaseMultiChannelNetwork):
                # BaseMultiChannelNetwork needs flattened inputs
                batch_size = rgb.size(0)
                rgb_flat = rgb.view(batch_size, -1)  # Flatten to (batch_size, channels*height*width)
                brightness_flat = brightness.view(batch_size, -1)  # Flatten
                outputs = model(rgb_flat, brightness_flat)
            else:
                # ResNet model can take the tensor directly
                outputs = model(rgb, brightness)
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()
    
    test_acc = 100. * test_correct / test_total
    return test_acc


def run_augmentation_e2e_test():
    """Run the complete end-to-end test for augmentation pipeline."""
    print("\n" + "="*70)
    print("AUGMENTATION END-TO-END TEST".center(70))
    print("="*70)
    
    # Prepare data
    data = prepare_data()
    
    # Create models
    base_model, resnet_model = create_models()
    
    # Test BaseMultiChannelNetwork with and without augmentation
    base_aug_metrics = train_model(
        base_model, 
        data['loaders']['train_aug'], 
        data['loaders']['val'],
        use_augmentation=True,
        num_epochs=TestConfig.num_epochs
    )
    
    # Reset model for fair comparison
    base_model, _ = create_models()
    
    base_no_aug_metrics = train_model(
        base_model, 
        data['loaders']['train_no_aug'], 
        data['loaders']['val'],
        use_augmentation=False,
        num_epochs=TestConfig.num_epochs
    )
    
    # Test MultiChannelResNetNetwork with and without augmentation
    resnet_aug_metrics = train_model(
        resnet_model, 
        data['loaders']['train_aug'], 
        data['loaders']['val'],
        use_augmentation=True,
        num_epochs=TestConfig.num_epochs
    )
    
    # Reset model for fair comparison
    _, resnet_model = create_models()
    
    resnet_no_aug_metrics = train_model(
        resnet_model, 
        data['loaders']['train_no_aug'], 
        data['loaders']['val'],
        use_augmentation=False,
        num_epochs=TestConfig.num_epochs
    )
    
    # Evaluate final test accuracy
    print("\n📊 Final Test Evaluation:")
    
    # Reload models with best configs
    base_model, resnet_model = create_models()
    
    # Train with augmentation for test evaluation
    train_model(
        base_model, 
        data['loaders']['train_aug'], 
        data['loaders']['val'],
        use_augmentation=True,
        num_epochs=TestConfig.num_epochs
    )
    
    train_model(
        resnet_model, 
        data['loaders']['train_aug'], 
        data['loaders']['val'],
        use_augmentation=True,
        num_epochs=TestConfig.num_epochs
    )
    
    # Test on held-out test set
    base_test_acc = test_model(base_model, data['loaders']['test'])
    resnet_test_acc = test_model(resnet_model, data['loaders']['test'])
    
    print(f"BaseMultiChannelNetwork Test Accuracy: {base_test_acc:.2f}%")
    print(f"MultiChannelResNetNetwork Test Accuracy: {resnet_test_acc:.2f}%")
    
    # Print augmentation comparison
    print("\n📈 Augmentation Comparison:")
    print(f"BaseMultiChannelNetwork with augmentation - Final Val Acc: {base_aug_metrics['val_acc'][-1]:.2f}%")
    print(f"BaseMultiChannelNetwork without augmentation - Final Val Acc: {base_no_aug_metrics['val_acc'][-1]:.2f}%")
    print(f"MultiChannelResNetNetwork with augmentation - Final Val Acc: {resnet_aug_metrics['val_acc'][-1]:.2f}%")
    print(f"MultiChannelResNetNetwork without augmentation - Final Val Acc: {resnet_no_aug_metrics['val_acc'][-1]:.2f}%")
    
    # Comparison of training curves
    plot_training_comparison(
        [base_aug_metrics, base_no_aug_metrics], 
        "BaseMultiChannelNetwork Augmentation Comparison"
    )
    
    plot_training_comparison(
        [resnet_aug_metrics, resnet_no_aug_metrics], 
        "MultiChannelResNetNetwork Augmentation Comparison"
    )
    
    print("\n✅ Augmentation End-to-End Test Complete!")
    
    # Return test successful
    return True


def plot_training_comparison(metrics_list, title):
    """Plot training curves for comparison."""
    plt.figure(figsize=(12, 5))
    
    # Plot training accuracy
    plt.subplot(1, 2, 1)
    for metrics in metrics_list:
        plt.plot(metrics['train_acc'], label=f"Train {metrics['aug_status']}")
        plt.plot(metrics['val_acc'], label=f"Val {metrics['aug_status']}")
    
    plt.title(f"{title} - Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot training loss
    plt.subplot(1, 2, 2)
    for metrics in metrics_list:
        plt.plot(metrics['train_loss'], label=f"Train {metrics['aug_status']}")
        plt.plot(metrics['val_loss'], label=f"Val {metrics['aug_status']}")
    
    plt.title(f"{title} - Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    print(f"Saved comparison plot to {title.replace(' ', '_')}.png")


if __name__ == "__main__":
    # Run the test
    success = run_augmentation_e2e_test()
    sys.exit(0 if success else 1)
