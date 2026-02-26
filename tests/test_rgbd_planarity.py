"""
Test RGB-D Planar Structure

Tests whether RGB-D data truly lies on 3D hyperplanes in 4D space.
This validates that orthogonal vectors are well-defined and consistent.

Tests:
1. Global hyperplane fit quality
2. Local hyperplane structure (per-image, per-scene)
3. Orthogonal vector consistency across samples
4. Residual analysis (how planar is the data?)
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.data_utils.sunrgbd_dataset import SUNRGBDDataset


def fit_hyperplane_svd(X):
    """
    Fit a 3D hyperplane to 4D data using SVD.

    Returns:
        basis: 3×4 matrix of basis vectors spanning the plane
        normal: 1×4 orthogonal vector (normal to the plane)
        fit_error: Mean squared distance from data to plane
    """
    # Center the data
    mean = X.mean(axis=0)
    X_centered = X - mean

    # SVD to find principal directions
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # First 3 right singular vectors span the hyperplane
    basis = Vt[:3, :]  # 3×4

    # 4th singular vector is orthogonal (normal to plane)
    normal = Vt[3, :]  # 1×4

    # Compute fit error (variance in orthogonal direction)
    fit_error = S[3]**2 / len(X)
    total_variance = np.sum(S**2) / len(X)
    fit_quality = 1 - (S[3]**2 / np.sum(S**2))

    return basis, normal, fit_error, fit_quality, S


def test_global_planarity():
    """Test 1: Global hyperplane fit across all data"""
    print("\n" + "="*80)
    print("TEST 1: Global Hyperplane Fit")
    print("="*80)

    dataset = SUNRGBDDataset(split='train', target_size=(224, 224))

    # Sample pixels from multiple images
    num_images = 100
    pixels_per_image = 1000

    print(f"\nSampling {pixels_per_image} pixels from {num_images} images...")

    samples = []
    for img_idx in range(min(num_images, len(dataset))):
        rgb, depth, _ = dataset[img_idx]

        # Denormalize
        rgb_denorm = rgb * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        rgb_denorm = torch.clamp(rgb_denorm, 0, 1)

        depth_denorm = depth * torch.tensor([0.2197]).view(1, 1, 1) + torch.tensor([0.5027]).view(1, 1, 1)
        depth_denorm = torch.clamp(depth_denorm, 0, 1)

        # Sample random pixels
        H, W = rgb.shape[1], rgb.shape[2]
        sample_indices = np.random.choice(H*W, size=pixels_per_image, replace=False)

        rgb_flat = rgb_denorm.reshape(3, -1)[:, sample_indices]
        depth_flat = depth_denorm.reshape(1, -1)[:, sample_indices]

        rgbd = torch.cat([rgb_flat, depth_flat], dim=0).T
        samples.append(rgbd.numpy())

    X = np.vstack(samples)
    print(f"Total samples: {X.shape}")

    # Fit global hyperplane
    print("\nFitting global 3D hyperplane to 4D data...")
    basis, normal, fit_error, fit_quality, S = fit_hyperplane_svd(X)

    print("\n" + "-"*80)
    print("HYPERPLANE FIT RESULTS:")
    print("-"*80)

    print(f"\nSingular values:")
    for i, sigma in enumerate(S):
        variance_pct = (sigma**2 / np.sum(S**2)) * 100
        print(f"  σ{i+1} = {sigma:10.4f}  ({variance_pct:6.2f}% variance)")

    print(f"\nFit quality: {fit_quality*100:.2f}%")
    print(f"  (percentage of variance explained by 3D hyperplane)")

    print(f"\nFit error (MSE): {fit_error:.6f}")
    print(f"  (mean squared distance from data to plane)")

    orthogonal_variance_pct = (S[3]**2 / np.sum(S**2)) * 100
    print(f"\nOrthogonal direction variance: {orthogonal_variance_pct:.4f}%")

    if fit_quality > 0.99:
        print("\n✅ Data lies VERY CLOSELY on a 3D hyperplane (>99% fit)")
    elif fit_quality > 0.95:
        print("\n✅ Data lies CLOSELY on a 3D hyperplane (>95% fit)")
    elif fit_quality > 0.90:
        print("\n⚠️  Data APPROXIMATELY lies on a 3D hyperplane (>90% fit)")
    else:
        print("\n❌ Data does NOT lie on a 3D hyperplane (<90% fit)")
        print("   → Structure is more complex (curved manifold?)")

    # Print orthogonal vector
    print("\n" + "-"*80)
    print("ORTHOGONAL VECTOR (normal to hyperplane):")
    print("-"*80)
    print(f"n = [{normal[0]:>7.4f}, {normal[1]:>7.4f}, {normal[2]:>7.4f}, {normal[3]:>7.4f}]")
    print(f"     (R,       G,       B,       D)")
    print(f"\n||n|| = {np.linalg.norm(normal):.6f} (should be 1.0)")

    return X, basis, normal, fit_quality


def test_local_planarity(dataset, num_images=20):
    """Test 2: Local hyperplane fit per image"""
    print("\n" + "="*80)
    print("TEST 2: Local Hyperplane Fit (Per-Image)")
    print("="*80)

    print(f"\nTesting planarity for {num_images} individual images...")

    fit_qualities = []
    orthogonal_vectors = []

    for img_idx in range(min(num_images, len(dataset))):
        rgb, depth, label = dataset[img_idx]

        # Denormalize
        rgb_denorm = rgb * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        rgb_denorm = torch.clamp(rgb_denorm, 0, 1)

        depth_denorm = depth * torch.tensor([0.2197]).view(1, 1, 1) + torch.tensor([0.5027]).view(1, 1, 1)
        depth_denorm = torch.clamp(depth_denorm, 0, 1)

        # Flatten all pixels
        H, W = rgb.shape[1], rgb.shape[2]
        rgb_flat = rgb_denorm.reshape(3, H*W).T
        depth_flat = depth_denorm.reshape(1, H*W).T

        X_img = np.concatenate([rgb_flat.numpy(), depth_flat.numpy()], axis=1)

        # Fit hyperplane to this image
        _, normal, _, fit_quality, _ = fit_hyperplane_svd(X_img)

        fit_qualities.append(fit_quality)
        orthogonal_vectors.append(normal)

    fit_qualities = np.array(fit_qualities)
    orthogonal_vectors = np.array(orthogonal_vectors)

    print("\n" + "-"*80)
    print("PER-IMAGE FIT QUALITY:")
    print("-"*80)
    print(f"Mean:   {fit_qualities.mean()*100:.2f}%")
    print(f"Std:    {fit_qualities.std()*100:.2f}%")
    print(f"Min:    {fit_qualities.min()*100:.2f}%")
    print(f"Max:    {fit_qualities.max()*100:.2f}%")
    print(f"Median: {np.median(fit_qualities)*100:.2f}%")

    if fit_qualities.mean() > 0.99:
        print("\n✅ Individual images lie VERY CLOSELY on 3D hyperplanes")
    elif fit_qualities.mean() > 0.95:
        print("\n✅ Individual images lie CLOSELY on 3D hyperplanes")
    else:
        print("\n⚠️  Individual images have more complex structure")

    # Analyze orthogonal vector consistency
    print("\n" + "-"*80)
    print("ORTHOGONAL VECTOR CONSISTENCY:")
    print("-"*80)

    # Compute pairwise angles between orthogonal vectors
    angles = []
    for i in range(len(orthogonal_vectors)):
        for j in range(i+1, len(orthogonal_vectors)):
            # Dot product (cosine of angle)
            cos_angle = np.abs(np.dot(orthogonal_vectors[i], orthogonal_vectors[j]))
            angle_deg = np.arccos(np.clip(cos_angle, 0, 1)) * 180 / np.pi
            angles.append(angle_deg)

    angles = np.array(angles)

    print(f"Pairwise angles between orthogonal vectors:")
    print(f"  Mean:   {angles.mean():.2f}°")
    print(f"  Std:    {angles.std():.2f}°")
    print(f"  Min:    {angles.min():.2f}°")
    print(f"  Max:    {angles.max():.2f}°")

    if angles.std() < 10:
        print("\n✅ Orthogonal vectors are VERY CONSISTENT across images (<10° std)")
    elif angles.std() < 30:
        print("\n✅ Orthogonal vectors are CONSISTENT across images (<30° std)")
    else:
        print("\n⚠️  Orthogonal vectors vary significantly across images")
        print("   → Different images have different orthogonal directions")

    # Plot distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Fit quality distribution
    axes[0].hist(fit_qualities * 100, bins=20, edgecolor='black', alpha=0.7)
    axes[0].axvline(fit_qualities.mean() * 100, color='red', linestyle='--', linewidth=2, label=f'Mean: {fit_qualities.mean()*100:.2f}%')
    axes[0].set_xlabel('Hyperplane Fit Quality (%)', fontsize=12)
    axes[0].set_ylabel('Number of Images', fontsize=12)
    axes[0].set_title('Per-Image Planarity', fontsize=14)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Plot 2: Orthogonal vector angle distribution
    axes[1].hist(angles, bins=20, edgecolor='black', alpha=0.7, color='orange')
    axes[1].axvline(angles.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {angles.mean():.2f}°')
    axes[1].set_xlabel('Angle Between Orthogonal Vectors (degrees)', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('Orthogonal Vector Consistency', fontsize=14)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('tests/rgbd_planarity_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved plot: tests/rgbd_planarity_analysis.png")
    plt.close()

    return fit_qualities, orthogonal_vectors, angles


def test_residual_analysis(X, basis, normal):
    """Test 3: Analyze residuals (distance from hyperplane)"""
    print("\n" + "="*80)
    print("TEST 3: Residual Analysis")
    print("="*80)

    # Center data
    mean = X.mean(axis=0)
    X_centered = X - mean

    # Project onto orthogonal direction
    residuals = X_centered @ normal

    print(f"\nResidual statistics (distance from hyperplane):")
    print(f"  Mean:   {residuals.mean():.6f} (should be ~0)")
    print(f"  Std:    {residuals.std():.6f}")
    print(f"  Min:    {residuals.min():.6f}")
    print(f"  Max:    {residuals.max():.6f}")
    print(f"  Range:  {residuals.max() - residuals.min():.6f}")

    # Compare to in-plane variance
    in_plane_projections = X_centered @ basis.T  # N × 3
    in_plane_variance = np.var(in_plane_projections)
    orthogonal_variance = np.var(residuals)

    ratio = orthogonal_variance / in_plane_variance

    print(f"\nVariance comparison:")
    print(f"  In-plane variance:      {in_plane_variance:.6f}")
    print(f"  Orthogonal variance:    {orthogonal_variance:.6f}")
    print(f"  Ratio (orth/in-plane):  {ratio:.6f} ({ratio*100:.2f}%)")

    if ratio < 0.01:
        print("\n✅ Orthogonal variance is < 1% of in-plane variance")
        print("   → Data is VERY PLANAR")
    elif ratio < 0.05:
        print("\n✅ Orthogonal variance is < 5% of in-plane variance")
        print("   → Data is PLANAR")
    else:
        print("\n⚠️  Orthogonal variance is significant compared to in-plane")
        print("   → Data has non-planar structure")

    # Plot residual distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Residual histogram
    axes[0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Hyperplane')
    axes[0].set_xlabel('Residual (distance from hyperplane)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution of Residuals', fontsize=14)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Plot 2: Q-Q plot to check if residuals are normally distributed
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot (Normality Check)', fontsize=14)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('tests/rgbd_residual_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved plot: tests/rgbd_residual_analysis.png")
    plt.close()

    return residuals, ratio


def test_scene_type_planarity():
    """Test 4: Does planarity vary by scene type?"""
    print("\n" + "="*80)
    print("TEST 4: Planarity by Scene Type")
    print("="*80)

    dataset = SUNRGBDDataset(split='train', target_size=(224, 224))

    # Group images by scene type
    scene_data = {}  # scene_label -> list of fit_qualities

    num_samples = 50
    print(f"\nAnalyzing planarity for {num_samples} images...")

    for img_idx in range(min(num_samples, len(dataset))):
        rgb, depth, label = dataset[img_idx]

        # Denormalize
        rgb_denorm = rgb * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        rgb_denorm = torch.clamp(rgb_denorm, 0, 1)

        depth_denorm = depth * torch.tensor([0.2197]).view(1, 1, 1) + torch.tensor([0.5027]).view(1, 1, 1)
        depth_denorm = torch.clamp(depth_denorm, 0, 1)

        # Flatten
        H, W = rgb.shape[1], rgb.shape[2]
        rgb_flat = rgb_denorm.reshape(3, H*W).T
        depth_flat = depth_denorm.reshape(1, H*W).T

        X_img = np.concatenate([rgb_flat.numpy(), depth_flat.numpy()], axis=1)

        # Fit hyperplane
        _, _, _, fit_quality, _ = fit_hyperplane_svd(X_img)

        # Get scene name
        scene_name = dataset.CLASS_NAMES[label]

        if scene_name not in scene_data:
            scene_data[scene_name] = []
        scene_data[scene_name].append(fit_quality)

    # Analyze per-scene
    print("\n" + "-"*80)
    print("PLANARITY BY SCENE TYPE:")
    print("-"*80)

    scene_stats = []
    for scene_name, qualities in sorted(scene_data.items()):
        if len(qualities) >= 3:  # Only show scenes with 3+ samples
            mean_quality = np.mean(qualities)
            std_quality = np.std(qualities)
            scene_stats.append((scene_name, mean_quality, std_quality, len(qualities)))

    # Sort by mean quality
    scene_stats.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'Scene Type':<20} {'Mean Fit':<12} {'Std':<10} {'Samples':<10}")
    print("-"*80)
    for scene_name, mean_q, std_q, count in scene_stats:
        print(f"{scene_name:<20} {mean_q*100:>6.2f}%      {std_q*100:>6.2f}%    {count:>3}")

    # Check if variance across scenes is significant
    all_means = [s[1] for s in scene_stats]
    scene_variance = np.var(all_means)

    print(f"\nVariance in fit quality across scene types: {scene_variance*100:.4f}%")

    if scene_variance < 0.0001:
        print("✅ Planarity is CONSISTENT across scene types")
    else:
        print("⚠️  Planarity varies by scene type")
        print("   → Different scenes have different structural complexity")

    return scene_data


def main():
    print("="*80)
    print("RGB-D PLANAR STRUCTURE ANALYSIS")
    print("="*80)
    print("\nGoal: Test if RGB-D data truly lies on 3D hyperplanes in 4D space")
    print("This validates that orthogonal vectors are well-defined and consistent.")

    # Load dataset
    dataset = SUNRGBDDataset(split='train', target_size=(224, 224))

    # Test 1: Global planarity
    X, basis, normal, global_fit_quality = test_global_planarity()

    # Test 2: Local planarity (per-image)
    fit_qualities, orthogonal_vectors, angles = test_local_planarity(dataset, num_images=50)

    # Test 3: Residual analysis
    residuals, variance_ratio = test_residual_analysis(X, basis, normal)

    # Test 4: Scene-type planarity
    scene_data = test_scene_type_planarity()

    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY: RGB-D PLANARITY")
    print("="*80)

    print(f"\nGlobal Fit Quality: {global_fit_quality*100:.2f}%")
    print(f"Per-Image Mean Fit: {fit_qualities.mean()*100:.2f}% ± {fit_qualities.std()*100:.2f}%")
    print(f"Orthogonal Vector Consistency: {angles.mean():.1f}° ± {angles.std():.1f}°")
    print(f"Variance Ratio (orth/in-plane): {variance_ratio*100:.4f}%")

    if global_fit_quality > 0.99 and variance_ratio < 0.01:
        print("\n✅ CONCLUSION: RGB-D data STRONGLY lies on a 3D hyperplane")
        print("✅ Orthogonal vector is well-defined and stable")
        print("✅ Safe to use as 3rd stream input!")
    elif global_fit_quality > 0.95 and variance_ratio < 0.05:
        print("\n✅ CONCLUSION: RGB-D data approximately lies on a 3D hyperplane")
        print("✅ Orthogonal vector is reasonably well-defined")
        print("✅ Can use as 3rd stream input with minor approximation")
    else:
        print("\n⚠️  CONCLUSION: RGB-D has more complex non-planar structure")
        print("⚠️  Orthogonal vector is an approximation")
        print("→ Consider using local/adaptive orthogonal vectors")

    print("\n" + "="*80)
    print("Generated Files:")
    print("  - tests/rgbd_planarity_analysis.png")
    print("  - tests/rgbd_residual_analysis.png")
    print("="*80)


if __name__ == '__main__':
    main()
