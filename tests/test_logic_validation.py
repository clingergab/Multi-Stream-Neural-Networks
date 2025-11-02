"""
Validate the test logic by:
1. Testing denormalize->augment->normalize round-trip
2. Measuring augmentation with and without Random Erasing
3. Checking if MSE comparisons are valid across modalities
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms

def test_denormalization_roundtrip():
    """Test that denormalize->normalize is a valid round-trip"""
    print("\n" + "="*80)
    print("TEST 1: Denormalization Round-Trip")
    print("="*80)

    # Create a sample normalized tensor
    rgb_normalized = torch.randn(3, 224, 224) * 0.5  # Roughly normalized range

    # Denormalize
    rgb_denorm = rgb_normalized * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    rgb_denorm = torch.clamp(rgb_denorm * 255, 0, 255).byte()

    # Convert to PIL and back
    rgb_pil = Image.fromarray(rgb_denorm.permute(1, 2, 0).numpy(), mode='RGB')

    # Re-normalize
    rgb_renorm = transforms.functional.to_tensor(rgb_pil)
    rgb_renorm = transforms.functional.normalize(
        rgb_renorm,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # Check difference
    diff = torch.mean((rgb_renorm - rgb_normalized) ** 2).item()
    print(f"\nRound-trip MSE: {diff:.6f}")

    if diff < 0.001:
        print("✓ Round-trip is nearly lossless (MSE < 0.001)")
    else:
        print(f"⚠️  Round-trip has quantization error (MSE = {diff:.6f})")
        print("   This is expected due to uint8 quantization")

    return diff

def test_random_erasing_impact():
    """Test if Random Erasing dominates the measurement"""
    print("\n" + "="*80)
    print("TEST 2: Random Erasing Impact")
    print("="*80)

    from src.data_utils.sunrgbd_dataset import SUNRGBDDataset

    dataset = SUNRGBDDataset(train=False, target_size=(224, 224))

    num_samples = 50

    # Measure WITHOUT Random Erasing
    print("\nMeasuring WITHOUT Random Erasing...")
    rgb_diffs_no_erase = []
    depth_diffs_no_erase = []

    for i in range(num_samples):
        rgb_orig, depth_orig, _ = dataset[i]

        # Denormalize
        rgb_denorm = rgb_orig * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        rgb_denorm = torch.clamp(rgb_denorm * 255, 0, 255).byte()

        depth_denorm = depth_orig * torch.tensor([0.2197]).view(1, 1, 1) + torch.tensor([0.5027]).view(1, 1, 1)
        depth_denorm = torch.clamp(depth_denorm * 255, 0, 255).byte()

        rgb_pil = Image.fromarray(rgb_denorm.permute(1, 2, 0).numpy(), mode='RGB')
        depth_pil = Image.fromarray(depth_denorm.squeeze().numpy(), mode='L')

        rgb_aug = rgb_pil

        # Color Jitter (43%)
        if np.random.random() < 0.43:
            color_jitter = transforms.ColorJitter(brightness=0.37, contrast=0.37, saturation=0.37, hue=0.11)
            rgb_aug = color_jitter(rgb_aug)

        # Gaussian Blur (25%)
        if np.random.random() < 0.25:
            kernel_size = int(np.random.choice([3, 5, 7]))
            sigma = float(np.random.uniform(0.1, 1.7))
            rgb_aug = transforms.functional.gaussian_blur(rgb_aug, kernel_size=kernel_size, sigma=sigma)

        # Grayscale (17%)
        if np.random.random() < 0.17:
            rgb_aug = transforms.functional.to_grayscale(rgb_aug, num_output_channels=3)

        # Depth augmentation
        depth_aug_array = np.array(depth_pil, dtype=np.float32)

        if np.random.random() < 0.5:
            brightness_factor = np.random.uniform(0.75, 1.25)
            contrast_factor = np.random.uniform(0.75, 1.25)
            depth_aug_array = (depth_aug_array - 127.5) * contrast_factor + 127.5
            depth_aug_array = depth_aug_array * brightness_factor
            noise = np.random.normal(0, 15, depth_aug_array.shape)
            depth_aug_array = depth_aug_array + noise
            depth_aug_array = np.clip(depth_aug_array, 0, 255)

        depth_aug = Image.fromarray(depth_aug_array.astype(np.uint8), mode='L')

        # Normalize (NO ERASING)
        rgb_aug_tensor = transforms.functional.to_tensor(rgb_aug)
        rgb_aug_tensor = transforms.functional.normalize(rgb_aug_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        depth_aug_tensor = transforms.functional.to_tensor(depth_aug)
        depth_aug_tensor = transforms.functional.normalize(depth_aug_tensor, mean=[0.5027], std=[0.2197])

        # MSE
        rgb_diff = torch.mean((rgb_aug_tensor - rgb_orig) ** 2).item()
        depth_diff = torch.mean((depth_aug_tensor - depth_orig) ** 2).item()

        rgb_diffs_no_erase.append(rgb_diff)
        depth_diffs_no_erase.append(depth_diff)

    ratio_no_erase = np.mean(rgb_diffs_no_erase) / np.mean(depth_diffs_no_erase)

    # Measure WITH Random Erasing
    print("Measuring WITH Random Erasing...")
    rgb_diffs_with_erase = []
    depth_diffs_with_erase = []

    for i in range(num_samples):
        rgb_orig, depth_orig, _ = dataset[i]

        # Denormalize
        rgb_denorm = rgb_orig * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        rgb_denorm = torch.clamp(rgb_denorm * 255, 0, 255).byte()

        depth_denorm = depth_orig * torch.tensor([0.2197]).view(1, 1, 1) + torch.tensor([0.5027]).view(1, 1, 1)
        depth_denorm = torch.clamp(depth_denorm * 255, 0, 255).byte()

        rgb_pil = Image.fromarray(rgb_denorm.permute(1, 2, 0).numpy(), mode='RGB')
        depth_pil = Image.fromarray(depth_denorm.squeeze().numpy(), mode='L')

        rgb_aug = rgb_pil

        # Color Jitter (43%)
        if np.random.random() < 0.43:
            color_jitter = transforms.ColorJitter(brightness=0.37, contrast=0.37, saturation=0.37, hue=0.11)
            rgb_aug = color_jitter(rgb_aug)

        # Gaussian Blur (25%)
        if np.random.random() < 0.25:
            kernel_size = int(np.random.choice([3, 5, 7]))
            sigma = float(np.random.uniform(0.1, 1.7))
            rgb_aug = transforms.functional.gaussian_blur(rgb_aug, kernel_size=kernel_size, sigma=sigma)

        # Grayscale (17%)
        if np.random.random() < 0.17:
            rgb_aug = transforms.functional.to_grayscale(rgb_aug, num_output_channels=3)

        # Depth augmentation
        depth_aug_array = np.array(depth_pil, dtype=np.float32)

        if np.random.random() < 0.5:
            brightness_factor = np.random.uniform(0.75, 1.25)
            contrast_factor = np.random.uniform(0.75, 1.25)
            depth_aug_array = (depth_aug_array - 127.5) * contrast_factor + 127.5
            depth_aug_array = depth_aug_array * brightness_factor
            noise = np.random.normal(0, 15, depth_aug_array.shape)
            depth_aug_array = depth_aug_array + noise
            depth_aug_array = np.clip(depth_aug_array, 0, 255)

        depth_aug = Image.fromarray(depth_aug_array.astype(np.uint8), mode='L')

        # Normalize
        rgb_aug_tensor = transforms.functional.to_tensor(rgb_aug)
        rgb_aug_tensor = transforms.functional.normalize(rgb_aug_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        depth_aug_tensor = transforms.functional.to_tensor(depth_aug)
        depth_aug_tensor = transforms.functional.normalize(depth_aug_tensor, mean=[0.5027], std=[0.2197])

        # Random Erasing
        if np.random.random() < 0.17:
            erasing = transforms.RandomErasing(p=1.0, scale=(0.02, 0.10), ratio=(0.5, 2.0))
            rgb_aug_tensor = erasing(rgb_aug_tensor)

        if np.random.random() < 0.10:
            erasing = transforms.RandomErasing(p=1.0, scale=(0.02, 0.1), ratio=(0.5, 2.0))
            depth_aug_tensor = erasing(depth_aug_tensor)

        # MSE
        rgb_diff = torch.mean((rgb_aug_tensor - rgb_orig) ** 2).item()
        depth_diff = torch.mean((depth_aug_tensor - depth_orig) ** 2).item()

        rgb_diffs_with_erase.append(rgb_diff)
        depth_diffs_with_erase.append(depth_diff)

    ratio_with_erase = np.mean(rgb_diffs_with_erase) / np.mean(depth_diffs_with_erase)

    print(f"\nResults:")
    print(f"  WITHOUT Random Erasing:")
    print(f"    RGB:   {np.mean(rgb_diffs_no_erase):.4f} ± {np.std(rgb_diffs_no_erase):.4f}")
    print(f"    Depth: {np.mean(depth_diffs_no_erase):.4f} ± {np.std(depth_diffs_no_erase):.4f}")
    print(f"    Ratio: {ratio_no_erase:.2f}x")
    print()
    print(f"  WITH Random Erasing:")
    print(f"    RGB:   {np.mean(rgb_diffs_with_erase):.4f} ± {np.std(rgb_diffs_with_erase):.4f}")
    print(f"    Depth: {np.mean(depth_diffs_with_erase):.4f} ± {np.std(depth_diffs_with_erase):.4f}")
    print(f"    Ratio: {ratio_with_erase:.2f}x")
    print()

    ratio_change = abs(ratio_with_erase - ratio_no_erase) / ratio_no_erase * 100
    print(f"  Ratio change due to Random Erasing: {ratio_change:.1f}%")

    if ratio_change > 20:
        print("  ⚠️  Random Erasing significantly affects the measurement!")
    else:
        print("  ✓ Random Erasing has moderate impact")

    return ratio_no_erase, ratio_with_erase

if __name__ == '__main__':
    print("="*80)
    print("TEST LOGIC VALIDATION")
    print("="*80)

    # Test 1
    roundtrip_error = test_denormalization_roundtrip()

    # Test 2
    ratio_no_erase, ratio_with_erase = test_random_erasing_impact()

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✓ Round-trip quantization error: {roundtrip_error:.6f}")
    print(f"✓ Ratio without Random Erasing: {ratio_no_erase:.2f}x")
    print(f"✓ Ratio with Random Erasing: {ratio_with_erase:.2f}x")
    print("="*80)
