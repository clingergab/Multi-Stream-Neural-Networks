#!/usr/bin/env python3
"""
Example script showing how to use StreamingDualChannelDataset for ImageNet training.

This demonstrates the complete setup for dual-channel ImageNet training including:
- Loading from multiple training folders
- Using validation truth file
- Setting up transforms
- Creating optimized dataloaders
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_utils.streaming_dual_channel_dataset import (
    create_imagenet_dual_channel_train_val_dataloaders,
    create_imagenet_dual_channel_test_dataloader,
    create_default_imagenet_transforms
)


def main():
    """Example usage of StreamingDualChannelDataset for ImageNet."""
    
    # Example paths - adjust these to your actual data structure
    train_folders = [
        "/content/drive/MyDrive/dataset/train_images_0",
        "/content/drive/MyDrive/dataset/train_images_1", 
        "/content/drive/MyDrive/dataset/train_images_2",
        "/content/drive/MyDrive/dataset/train_images_3",
        "/content/drive/MyDrive/dataset/train_images_4"
    ]
    
    val_folder = "/content/drive/MyDrive/dataset/val_images"
    test_folder = "/content/drive/MyDrive/dataset/test_images"
    truth_file = "/content/drive/MyDrive/dataset/ILSVRC2013_devkit/ILSVRC2012_validation_ground_truth.txt"
    
    # Create transforms
    train_transform, val_transform = create_default_imagenet_transforms(
        image_size=(224, 224)
    )
    
    print("🚀 Creating ImageNet dual-channel dataloaders...")
    
    # Create train and validation dataloaders
    train_loader, val_loader = create_imagenet_dual_channel_train_val_dataloaders(
        train_folders=train_folders,
        val_folder=val_folder,
        truth_file=truth_file,
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=32,
        image_size=(224, 224),
        num_workers=2,  # Adjust based on your system
        pin_memory=True,
        persistent_workers=True
    )
    
    # Create test dataloader (optional - test set may not have labels)
    test_loader = create_imagenet_dual_channel_test_dataloader(
        test_folder=test_folder,
        truth_file=None,  # Test set typically doesn't have public labels
        test_transform=val_transform,  # Use same transform as validation
        batch_size=32,
        num_workers=2
    )
    
    print(f"✅ Created dataloaders:")
    print(f"   Training: {len(train_loader)} batches")
    print(f"   Validation: {len(val_loader)} batches") 
    print(f"   Test: {len(test_loader)} batches")
    
    # Test loading a batch
    print("\n🔍 Testing batch loading...")
    
    for rgb_batch, brightness_batch, label_batch in train_loader:
        print(f"   RGB batch shape: {rgb_batch.shape}")
        print(f"   Brightness batch shape: {brightness_batch.shape}")
        print(f"   Label batch shape: {label_batch.shape}")
        print(f"   RGB range: [{rgb_batch.min():.3f}, {rgb_batch.max():.3f}]")
        print(f"   Brightness range: [{brightness_batch.min():.3f}, {brightness_batch.max():.3f}]")
        break
    
    print("\n✅ Dual-channel ImageNet dataloaders are working correctly!")
    
    # Example training loop structure
    print("\n📚 Example training loop structure:")
    print("""
    for epoch in range(num_epochs):
        model.train()
        for rgb_batch, brightness_batch, labels in train_loader:
            rgb_batch = rgb_batch.to(device)
            brightness_batch = brightness_batch.to(device)
            labels = labels.to(device)
            
            # Forward pass with dual channels
            outputs = model(rgb_batch, brightness_batch)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            for rgb_batch, brightness_batch, labels in val_loader:
                # ... validation logic
    """)


if __name__ == "__main__":
    main()
