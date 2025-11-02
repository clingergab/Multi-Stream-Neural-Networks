"""
Accurate augmentation strength measurement.

This test measures ONLY appearance augmentations (color, blur, etc.)
by temporarily disabling spatial augmentations (flip, crop) to get
an accurate measurement of RGB vs Depth augmentation strength.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms

def measure_augmentation_strength():
    """
    Measure RGB vs Depth augmentation strength by manually applying
    ONLY appearance augmentations (no flip/crop) to get accurate ratio.
    """
    from src.data_utils.sunrgbd_dataset import SUNRGBDDataset

    # Load dataset to get raw images
    dataset = SUNRGBDDataset(train=False, target_size=(224, 224))  # Val to get unaugmented

    num_samples = 100
    rgb_diffs = []
    depth_diffs = []

    print(f"Measuring augmentation strength on {num_samples} samples...")
    print("(Applying ONLY appearance augs, no spatial transforms)\n")

    for i in range(num_samples):
        # Get original unaugmented sample
        rgb_orig, depth_orig, _ = dataset[i]

        # Denormalize to get back to [0, 255] range
        rgb_denorm = rgb_orig * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        rgb_denorm = torch.clamp(rgb_denorm * 255, 0, 255).byte()

        depth_denorm = depth_orig * torch.tensor([0.2197]).view(1, 1, 1) + torch.tensor([0.5027]).view(1, 1, 1)
        depth_denorm = torch.clamp(depth_denorm * 255, 0, 255).byte()

        # Convert to PIL for augmentation
        rgb_pil = Image.fromarray(rgb_denorm.permute(1, 2, 0).numpy(), mode='RGB')
        depth_pil = Image.fromarray(depth_denorm.squeeze().numpy(), mode='L')

        # Apply RGB augmentations (current settings)
        rgb_aug = rgb_pil

        # Color Jitter (43% prob) - UPDATED to match current settings
        if np.random.random() < 0.43:
            color_jitter = transforms.ColorJitter(
                brightness=0.37,
                contrast=0.37,
                saturation=0.37,
                hue=0.11
            )
            rgb_aug = color_jitter(rgb_aug)

        # Gaussian Blur (25% prob) - UPDATED to match current settings
        if np.random.random() < 0.25:
            kernel_size = int(np.random.choice([3, 5, 7]))
            sigma = float(np.random.uniform(0.1, 1.7))
            rgb_aug = transforms.functional.gaussian_blur(rgb_aug, kernel_size=kernel_size, sigma=sigma)

        # Grayscale (17% prob) - UPDATED to match current settings
        if np.random.random() < 0.17:
            rgb_aug = transforms.functional.to_grayscale(rgb_aug, num_output_channels=3)

        # Apply Depth augmentations (current settings)
        depth_aug_array = np.array(depth_pil, dtype=np.float32)

        # Brightness/Contrast + Noise (50% prob)
        if np.random.random() < 0.5:
            brightness_factor = np.random.uniform(0.75, 1.25)
            contrast_factor = np.random.uniform(0.75, 1.25)

            depth_aug_array = (depth_aug_array - 127.5) * contrast_factor + 127.5
            depth_aug_array = depth_aug_array * brightness_factor

            noise = np.random.normal(0, 15, depth_aug_array.shape)
            depth_aug_array = depth_aug_array + noise

            depth_aug_array = np.clip(depth_aug_array, 0, 255)

        depth_aug = Image.fromarray(depth_aug_array.astype(np.uint8), mode='L')

        # Convert back to tensors and normalize
        rgb_aug_tensor = transforms.functional.to_tensor(rgb_aug)
        rgb_aug_tensor = transforms.functional.normalize(
            rgb_aug_tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # Random Erasing (17% prob) - UPDATED to match current settings
        if np.random.random() < 0.17:
            erasing = transforms.RandomErasing(p=1.0, scale=(0.02, 0.10), ratio=(0.5, 2.0))
            rgb_aug_tensor = erasing(rgb_aug_tensor)

        depth_aug_tensor = transforms.functional.to_tensor(depth_aug)
        depth_aug_tensor = transforms.functional.normalize(
            depth_aug_tensor,
            mean=[0.5027],
            std=[0.2197]
        )

        # Random Erasing (10% prob) - post-normalization
        if np.random.random() < 0.10:
            erasing = transforms.RandomErasing(p=1.0, scale=(0.02, 0.1), ratio=(0.5, 2.0))
            depth_aug_tensor = erasing(depth_aug_tensor)

        # Compute MSE difference
        rgb_diff = torch.mean((rgb_aug_tensor - rgb_orig) ** 2).item()
        depth_diff = torch.mean((depth_aug_tensor - depth_orig) ** 2).item()

        rgb_diffs.append(rgb_diff)
        depth_diffs.append(depth_diff)

        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{num_samples} samples...")

    # Calculate statistics
    rgb_mean = np.mean(rgb_diffs)
    rgb_std = np.std(rgb_diffs)
    depth_mean = np.mean(depth_diffs)
    depth_std = np.std(depth_diffs)
    ratio = rgb_mean / depth_mean

    print("\n" + "="*80)
    print("ACCURATE AUGMENTATION STRENGTH MEASUREMENT")
    print("="*80)
    print(f"\nMean Squared Difference (excluding spatial transforms):")
    print(f"  RGB:   {rgb_mean:.4f} ± {rgb_std:.4f}")
    print(f"  Depth: {depth_mean:.4f} ± {depth_std:.4f}")
    print(f"  Ratio: {ratio:.2f}x")
    print()

    if ratio < 1.3:
        print(f"⚠️  Ratio is BELOW 1.5x target: {ratio:.2f}x")
        print(f"    Consider INCREASING RGB augmentation or DECREASING Depth augmentation")
    elif ratio > 1.7:
        print(f"⚠️  Ratio is ABOVE 1.5x target: {ratio:.2f}x")
        print(f"    Consider DECREASING RGB augmentation or INCREASING Depth augmentation")
    else:
        print(f"✓ Ratio is close to 1.5x target: {ratio:.2f}x")

    print("="*80)

    return ratio

if __name__ == '__main__':
    ratio = measure_augmentation_strength()
