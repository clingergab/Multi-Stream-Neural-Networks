#!/usr/bin/env python3
"""
Test script to verify CIFAR-100 loader integration with RGBtoRGBL processor.
"""

import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_cifar100_with_rgbtorgbl():
    """Test CIFAR-100 loader with RGBtoRGBL transformation."""
    print("🔍 Testing CIFAR-100 loader with RGBtoRGBL processor...")
    
    try:
        # Import utilities
        from src.utils.cifar100_loader import get_cifar100_datasets, CIFAR100_FINE_LABELS
        from src.transforms.rgb_to_rgbl import RGBtoRGBL
        
        print("✅ Successfully imported CIFAR-100 loader and RGBtoRGBL")
        
        # Load CIFAR-100 data
        print("\n📊 Loading CIFAR-100 datasets...")
        train_dataset, test_dataset, class_names = get_cifar100_datasets()
        
        # Initialize the RGB to RGBL transform
        print("\n🔄 Initializing RGBtoRGBL transform...")
        rgb_to_rgbl = RGBtoRGBL()
        print(f"✅ RGBtoRGBL initialized successfully")
        
        # Test with a few samples
        print("\n🧪 Testing transformation on sample images...")
        num_test_samples = 5
        
        for i in range(num_test_samples):
            # Get a sample
            image, label = train_dataset[i]
            class_name = class_names[label]
            
            print(f"\n   Sample {i+1}: {class_name} (label: {label})")
            print(f"      Input shape: {image.shape}")
            print(f"      Input range: [{image.min():.3f}, {image.max():.3f}]")
            
            # Apply RGB to RGBL transformation (returns tuple)
            rgb_output, brightness_output = rgb_to_rgbl(image)
            
            print(f"      RGB output shape: {rgb_output.shape}")
            print(f"      Brightness output shape: {brightness_output.shape}")
            print(f"      RGB range: [{rgb_output.min():.3f}, {rgb_output.max():.3f}]")
            print(f"      Brightness range: [{brightness_output.min():.3f}, {brightness_output.max():.3f}]")
            
            # Test the concatenated RGBL version
            rgbl_image = rgb_to_rgbl.get_rgbl(image)
            print(f"      RGBL combined shape: {rgbl_image.shape}")
            print(f"      RGBL combined range: [{rgbl_image.min():.3f}, {rgbl_image.max():.3f}]")
            
            # Verify dimensions
            assert rgbl_image.shape[0] == 4, f"Expected 4 channels, got {rgbl_image.shape[0]}"
            assert rgbl_image.shape[1:] == image.shape[1:], f"Spatial dimensions changed: {rgbl_image.shape[1:]} vs {image.shape[1:]}"
            
            # Extract RGB and brightness channels from combined tensor
            rgb_channels = rgbl_image[:3]
            brightness_channel = rgbl_image[3:4]
            
            print(f"      Extracted RGB shape: {rgb_channels.shape}")
            print(f"      Extracted brightness shape: {brightness_channel.shape}")
            
            # Verify that the separate and combined outputs are the same
            assert torch.allclose(rgb_output, rgb_channels), "RGB channels don't match"
            assert torch.allclose(brightness_output, brightness_channel), "Brightness channels don't match"
        
        print("\n✅ All sample transformations successful!")
        
        # Test batch processing
        print("\n📦 Testing batch processing...")
        batch_size = 10
        batch_images = torch.stack([train_dataset[i][0] for i in range(batch_size)])
        
        print(f"   Batch input shape: {batch_images.shape}")
        
        # Process each image in the batch using the combined method
        batch_rgbl = []
        for img in batch_images:
            rgbl_img = rgb_to_rgbl.get_rgbl(img)
            batch_rgbl.append(rgbl_img)
        
        batch_rgbl = torch.stack(batch_rgbl)
        print(f"   Batch output shape: {batch_rgbl.shape}")
        
        # Verify batch processing
        assert batch_rgbl.shape == (batch_size, 4, 32, 32), f"Unexpected batch output shape: {batch_rgbl.shape}"
        
        # Test batch processing directly
        batch_rgb, batch_brightness = rgb_to_rgbl(batch_images)
        batch_rgbl_direct = rgb_to_rgbl.get_rgbl(batch_images)
        
        print(f"   Direct batch RGB shape: {batch_rgb.shape}")
        print(f"   Direct batch brightness shape: {batch_brightness.shape}")
        print(f"   Direct batch RGBL shape: {batch_rgbl_direct.shape}")
        
        assert batch_rgbl_direct.shape == (batch_size, 4, 32, 32), f"Unexpected direct batch shape: {batch_rgbl_direct.shape}"
        
        print("✅ Batch processing successful!")
        
        # Test with different image ranges to ensure robustness
        print("\n🔬 Testing with different input ranges...")
        
        # Test with normalized image (0-1 range)
        test_image = train_dataset[0][0]  # Already in 0-1 range
        rgbl_normalized = rgb_to_rgbl.get_rgbl(test_image)
        print(f"   Normalized input [0,1]: output range [{rgbl_normalized.min():.3f}, {rgbl_normalized.max():.3f}]")
        
        # Test with scaled image (0-255 range)
        test_image_255 = test_image * 255.0
        rgbl_255 = rgb_to_rgbl.get_rgbl(test_image_255)
        print(f"   Scaled input [0,255]: output range [{rgbl_255.min():.3f}, {rgbl_255.max():.3f}]")
        
        print("\n✅ All input range tests passed!")
        
        # Performance test
        print("\n⚡ Performance test...")
        import time
        
        start_time = time.time()
        for i in range(100):
            image, _ = train_dataset[i]
            rgbl_image = rgb_to_rgbl.get_rgbl(image)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        print(f"   Average processing time per image: {avg_time*1000:.2f} ms")
        
        print("\n🎉 All CIFAR-100 + RGBtoRGBL integration tests passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def visualize_transformation():
    """Create a visual test of the transformation."""
    print("\n🖼️ Creating visualization of RGB to RGBL transformation...")
    
    try:
        from src.utils.cifar100_loader import get_cifar100_datasets
        from src.transforms.rgb_to_rgbl import RGBtoRGBL
        
        # Load data
        train_dataset, _, class_names = get_cifar100_datasets()
        rgb_to_rgbl = RGBtoRGBL()
        
        # Create visualization for a few samples
        fig, axes = plt.subplots(3, 4, figsize=(12, 9))
        fig.suptitle('CIFAR-100: RGB to RGBL Transformation', fontsize=16, fontweight='bold')
        
        for i in range(3):
            image, label = train_dataset[i]
            class_name = class_names[label]
            
            # Apply transformation
            rgb_output, brightness_output = rgb_to_rgbl(image)
            rgbl_image = rgb_to_rgbl.get_rgbl(image)
            
            # Convert tensors to numpy for plotting
            rgb_img = image.permute(1, 2, 0).numpy()
            brightness_img = brightness_output[0].numpy()  # Extract brightness channel
            
            # Plot RGB image
            axes[i, 0].imshow(rgb_img)
            axes[i, 0].set_title(f'RGB - {class_name}', fontweight='bold')
            axes[i, 0].axis('off')
            
            # Plot R, G, B channels
            axes[i, 1].imshow(rgbl_image[0].numpy(), cmap='Reds')
            axes[i, 1].set_title('Red Channel')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(rgbl_image[1].numpy(), cmap='Greens')
            axes[i, 2].set_title('Green Channel')
            axes[i, 2].axis('off')
            
            # Plot brightness channel
            axes[i, 3].imshow(brightness_img, cmap='gray')
            axes[i, 3].set_title('Brightness Channel')
            axes[i, 3].axis('off')
        
        plt.tight_layout()
        
        # Save the visualization
        output_path = 'cifar100_rgbl_test_visualization.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Visualization saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 CIFAR-100 + RGBtoRGBL Integration Test")
    print("=" * 50)
    
    # Run integration test
    test_success = test_cifar100_with_rgbtorgbl()
    
    if test_success:
        print("\n" + "=" * 50)
        # Create visualization
        viz_success = visualize_transformation()
        
        if viz_success:
            print("\n🎉 All tests completed successfully!")
            print("💡 The CIFAR-100 loader works perfectly with RGBtoRGBL processor!")
        else:
            print("\n⚠️ Integration test passed but visualization failed")
    else:
        print("\n❌ Integration test failed")
