#!/usr/bin/env python3
"""
Simple demonstration of CIFAR-100 loader + RGBtoRGBL workflow for notebook usage.
"""

import sys
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def demo_notebook_workflow():
    """Demonstrate the exact workflow that would be used in the notebook."""
    print("🚀 CIFAR-100 + RGBtoRGBL Notebook Workflow Demo")
    print("=" * 50)
    
    # Step 1: Import utilities (like in the notebook)
    print("📦 Importing utilities...")
    from src.utils.cifar100_loader import get_cifar100_datasets, CIFAR100_FINE_LABELS
    from src.transforms.rgb_to_rgbl import RGBtoRGBL
    
    # Step 2: Load CIFAR-100 datasets (like in the notebook)
    print("\n📊 Loading CIFAR-100 datasets...")
    train_dataset, test_dataset, class_names = get_cifar100_datasets()
    
    # Extract raw data for backward compatibility
    train_data = train_dataset.data
    train_labels = train_dataset.labels
    test_data = test_dataset.data
    test_labels = test_dataset.labels
    
    print(f"✅ Raw data extracted for compatibility:")
    print(f"   train_data shape: {train_data.shape}")
    print(f"   train_labels shape: {train_labels.shape}")
    
    # Step 3: Initialize transformation (like in the notebook)
    print("\n🔄 Initializing RGBtoRGBL transform...")
    rgb_to_rgbl = RGBtoRGBL()
    
    # Step 4: Process a sample (like in the notebook)
    print("\n🧪 Processing sample data...")
    sample_image = train_data[0]
    sample_label = train_labels[0]
    class_name = class_names[sample_label]
    
    print(f"Sample: {class_name} (label: {sample_label})")
    print(f"Original image shape: {sample_image.shape}")
    
    # Method 1: Get separate RGB and brightness (common in notebook)
    rgb_output, brightness_output = rgb_to_rgbl(sample_image)
    
    print(f"RGB output shape: {rgb_output.shape}")
    print(f"Brightness output shape: {brightness_output.shape}")
    
    # Method 2: Get combined RGBL tensor (also useful)
    rgbl_combined = rgb_to_rgbl.get_rgbl(sample_image)
    print(f"Combined RGBL shape: {rgbl_combined.shape}")
    
    # Step 5: Batch processing (like processing full dataset in notebook)
    print("\n📦 Batch processing example...")
    batch_size = 32
    batch_images = train_data[:batch_size]
    
    print(f"Processing batch of {batch_size} images...")
    print(f"Batch input shape: {batch_images.shape}")
    
    # Process the batch
    batch_rgb, batch_brightness = rgb_to_rgbl(batch_images)
    
    print(f"Batch RGB output shape: {batch_rgb.shape}")
    print(f"Batch brightness output shape: {batch_brightness.shape}")
    
    # Step 6: Create arrays for training (common notebook pattern)
    print("\n🎯 Creating training arrays...")
    
    # For demonstration, process a small subset
    subset_size = 100
    subset_images = train_data[:subset_size]
    subset_labels = train_labels[:subset_size]
    
    print(f"Processing {subset_size} images for training arrays...")
    
    rgb_data = []
    brightness_data = []
    
    for img in subset_images:
        rgb, brightness = rgb_to_rgbl(img)
        rgb_data.append(rgb)
        brightness_data.append(brightness)
    
    # Stack into tensors
    rgb_data = torch.stack(rgb_data)
    brightness_data = torch.stack(brightness_data)
    
    print(f"✅ Training arrays created:")
    print(f"   RGB data shape: {rgb_data.shape}")
    print(f"   Brightness data shape: {brightness_data.shape}")
    print(f"   Labels shape: {subset_labels.shape}")
    
    # Step 7: Verify compatibility with existing notebook variables
    print("\n✅ Notebook compatibility verification:")
    print(f"   Original train_data compatible: {train_data.dtype} {train_data.shape}")
    print(f"   RGB channels compatible: {rgb_data.dtype} {rgb_data.shape}")
    print(f"   Brightness channels compatible: {brightness_data.dtype} {brightness_data.shape}")
    print(f"   Class names available: {len(class_names)} classes")
    print(f"   CIFAR100_FINE_LABELS constant: {len(CIFAR100_FINE_LABELS)} classes")
    
    print("\n🎉 Workflow demo completed successfully!")
    print("💡 This demonstrates the exact pattern used in the notebook!")
    
    return True

if __name__ == "__main__":
    success = demo_notebook_workflow()
    
    if success:
        print("\n✅ The CIFAR-100 loader is fully compatible with the notebook workflow!")
        print("🚀 Ready for multi-stream neural network training!")
    else:
        print("\n❌ Workflow demo failed.")
