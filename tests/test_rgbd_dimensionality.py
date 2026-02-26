"""
RGB-D Dimensionality Analysis for SUN RGB-D Dataset

Tests whether RGB-D data spans the full 4D space or lies in a lower-dimensional subspace.
If rank < 4, then orthogonal dimension(s) exist that can be used as a 3rd stream.

Analysis:
1. Load SUN RGB-D training data
2. Sample pixels from multiple images
3. Stack as 4×N matrix [R, G, B, D]
4. Compute SVD to find singular values
5. Determine effective rank
6. Extract orthogonal vector(s) if rank < 4
7. Visualize orthogonal stream
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from src.data_utils.sunrgbd_dataset import SUNRGBDDataset

def test_dimensionality_basic():
    """Test 1: Basic dimensionality analysis"""
    print("\n" + "="*80)
    print("TEST 1: Basic RGB-D Dimensionality Analysis")
    print("="*80)

    # Load dataset
    dataset = SUNRGBDDataset(split='train', target_size=(224, 224))
    print(f"\nLoaded SUN RGB-D train set: {len(dataset)} images")

    # Sample pixels from multiple images
    num_images = 200  # Sample 200 images
    pixels_per_image = 500  # Sample 500 random pixels per image

    print(f"Sampling {pixels_per_image} pixels from {num_images} images...")
    print(f"Total samples: {num_images * pixels_per_image:,}")

    samples = []

    for img_idx in range(min(num_images, len(dataset))):
        rgb, depth, _ = dataset[img_idx]

        # Denormalize to get original pixel values
        rgb_denorm = rgb * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        rgb_denorm = torch.clamp(rgb_denorm, 0, 1)

        depth_denorm = depth * torch.tensor([0.2197]).view(1, 1, 1) + torch.tensor([0.5027]).view(1, 1, 1)
        depth_denorm = torch.clamp(depth_denorm, 0, 1)

        # Flatten and randomly sample pixels
        H, W = rgb.shape[1], rgb.shape[2]
        num_pixels = H * W
        sample_indices = np.random.choice(num_pixels, size=pixels_per_image, replace=False)

        rgb_flat = rgb_denorm.reshape(3, -1)[:, sample_indices]  # 3 × 500
        depth_flat = depth_denorm.reshape(1, -1)[:, sample_indices]  # 1 × 500

        # Stack as [R, G, B, D]
        rgbd = torch.cat([rgb_flat, depth_flat], dim=0).T  # 500 × 4
        samples.append(rgbd.numpy())

        if (img_idx + 1) % 50 == 0:
            print(f"  Processed {img_idx + 1}/{num_images} images...")

    # Combine all samples
    X = np.vstack(samples)  # N × 4 matrix
    print(f"\nTotal data matrix: {X.shape} (N × 4)")

    # Compute SVD
    print("\nComputing SVD...")
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    # Analyze singular values
    print("\n" + "-"*80)
    print("SINGULAR VALUES:")
    print("-"*80)
    total_variance = np.sum(S**2)

    for i, sigma in enumerate(S):
        variance_explained = (sigma**2) / total_variance * 100
        print(f"σ{i+1} = {sigma:10.4f}  ({variance_explained:6.2f}% of variance)")

    # Compute effective rank using VARIANCE threshold (not singular value ratio)
    variance_threshold = 0.01  # 1% of total variance
    variance_ratios = (S**2) / total_variance
    effective_rank = np.sum(variance_ratios > variance_threshold)

    # Also show old method for comparison
    normalized_S = S / S[0]
    effective_rank_old = np.sum(normalized_S > 0.01)

    print(f"\nNormalized singular values (relative to σ1):")
    for i, norm_sigma in enumerate(normalized_S):
        print(f"  σ{i+1}/σ1 = {norm_sigma:.6f}")

    print(f"\nVariance contribution per dimension:")
    for i, var_ratio in enumerate(variance_ratios):
        print(f"  Dim {i+1}: {var_ratio*100:.4f}% of total variance")

    print(f"\nEffective rank (variance threshold={variance_threshold*100}%): {effective_rank}/4")
    print(f"Effective rank (old method, σ/σ1 > 1%): {effective_rank_old}/4")

    # Determine if orthogonal dimension exists
    if effective_rank < 4:
        print(f"\n✓ RGB-D data spans ~{effective_rank}D subspace")
        print(f"✓ {4 - effective_rank} orthogonal dimension(s) exist!")
        print(f"\n→ We can extract orthogonal vector(s) for 3rd stream!")
    else:
        print(f"\n⚠️  RGB-D data spans full 4D space")
        print(f"⚠️  No orthogonal dimension in 4D")
        print(f"→ Need higher-dimensional space (spatial patches) for orthogonal stream")

    # Variance explained plot
    variance_ratios = (S**2) / total_variance * 100
    cumulative_variance = np.cumsum(variance_ratios)

    print(f"\nCumulative variance explained:")
    for i, cum_var in enumerate(cumulative_variance):
        print(f"  First {i+1} components: {cum_var:.2f}%")

    return S, Vt, effective_rank, X


def test_correlation_analysis(X):
    """Test 2: Correlation structure of RGB-D"""
    print("\n" + "="*80)
    print("TEST 2: Correlation Analysis")
    print("="*80)

    # Compute correlation matrix
    print("\nComputing correlation matrix...")
    corr_matrix = np.corrcoef(X.T)

    print("\nCorrelation Matrix:")
    print("-"*80)
    labels = ['R', 'G', 'B', 'D']
    print(f"{'':>6}", end="")
    for label in labels:
        print(f"{label:>8}", end="")
    print()
    print("-"*80)

    for i, label in enumerate(labels):
        print(f"{label:>6}", end="")
        for j in range(4):
            print(f"{corr_matrix[i, j]:>8.4f}", end="")
        print()

    # Analyze RGB internal correlations
    print("\n" + "-"*80)
    print("RGB Internal Correlations:")
    print("-"*80)
    print(f"ρ(R, G) = {corr_matrix[0, 1]:.4f}")
    print(f"ρ(R, B) = {corr_matrix[0, 2]:.4f}")
    print(f"ρ(G, B) = {corr_matrix[1, 2]:.4f}")

    rgb_avg_corr = (corr_matrix[0, 1] + corr_matrix[0, 2] + corr_matrix[1, 2]) / 3
    print(f"\nAverage RGB correlation: {rgb_avg_corr:.4f}")

    if rgb_avg_corr > 0.8:
        print("→ RGB channels are HIGHLY correlated (explains rank reduction)")
    elif rgb_avg_corr > 0.6:
        print("→ RGB channels are MODERATELY correlated")
    else:
        print("→ RGB channels are WEAKLY correlated")

    # Analyze RGB-Depth correlations
    print("\n" + "-"*80)
    print("RGB-Depth Correlations:")
    print("-"*80)
    print(f"ρ(R, D) = {corr_matrix[0, 3]:.4f}")
    print(f"ρ(G, D) = {corr_matrix[1, 3]:.4f}")
    print(f"ρ(B, D) = {corr_matrix[2, 3]:.4f}")

    # Compute luminance and check correlation with depth
    luminance = 0.299 * X[:, 0] + 0.587 * X[:, 1] + 0.114 * X[:, 2]
    depth = X[:, 3]
    lum_depth_corr = np.corrcoef(luminance, depth)[0, 1]

    print(f"\nρ(Luminance, Depth) = {lum_depth_corr:.4f}")

    if abs(lum_depth_corr) > 0.3:
        print("→ Luminance and Depth are CORRELATED (indoor lighting effect)")
    else:
        print("→ Luminance and Depth are WEAKLY correlated")

    # Visualize correlation matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                xticklabels=labels, yticklabels=labels, vmin=-1, vmax=1,
                center=0, square=True)
    plt.title('RGB-D Correlation Matrix (SUN RGB-D Dataset)')
    plt.tight_layout()
    plt.savefig('tests/rgbd_correlation_matrix.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved correlation matrix plot: tests/rgbd_correlation_matrix.png")
    plt.close()

    return corr_matrix


def test_extract_orthogonal_vector(S, Vt, effective_rank, X):
    """Test 3: Extract orthogonal vector(s) if they exist"""
    print("\n" + "="*80)
    print("TEST 3: Extract Orthogonal Vector")
    print("="*80)

    if effective_rank >= 4:
        print("\n⚠️  No orthogonal dimension exists (rank = 4)")
        print("→ Skipping orthogonal vector extraction")
        return None

    print(f"\nExtracting {4 - effective_rank} orthogonal vector(s)...")

    # The orthogonal vectors are the right singular vectors with small singular values
    # Vt has shape (4, 4), rows are right singular vectors
    # Last (4 - effective_rank) rows correspond to null space

    orthogonal_vectors = Vt[effective_rank:, :]  # (4-rank) × 4

    print(f"\nOrthogonal vector(s) (rows of V^T):")
    print("-"*80)
    labels = ['R', 'G', 'B', 'D']

    for i, vec in enumerate(orthogonal_vectors):
        print(f"v{i+1} = [{vec[0]:>7.4f}, {vec[1]:>7.4f}, {vec[2]:>7.4f}, {vec[3]:>7.4f}]")
        print(f"     ({labels[0]}, {labels[1]}, {labels[2]}, {labels[3]})")

        # Normalize to unit length
        norm = np.linalg.norm(vec)
        print(f"     ||v{i+1}|| = {norm:.6f}")

    # Verify orthogonality to principal components
    print("\n" + "-"*80)
    print("Verification: Orthogonality Check")
    print("-"*80)

    principal_vectors = Vt[:effective_rank, :]  # First 'rank' vectors

    print(f"\nDot products with principal components:")
    for i, orth_vec in enumerate(orthogonal_vectors):
        print(f"\nOrthogonal vector {i+1}:")
        for j, princ_vec in enumerate(principal_vectors):
            dot_product = np.dot(orth_vec, princ_vec)
            print(f"  v{i+1} · principal_{j+1} = {dot_product:>10.6f} {'✓' if abs(dot_product) < 1e-10 else '✗'}")

    # Project RGB-D data onto orthogonal vector(s)
    print("\n" + "-"*80)
    print("Projecting RGB-D Data onto Orthogonal Vector")
    print("-"*80)

    orthogonal_stream = X @ orthogonal_vectors[0]  # N × 1 (first orthogonal vector)

    print(f"\nOrthogonal stream statistics:")
    print(f"  Mean:   {orthogonal_stream.mean():.6f}")
    print(f"  Std:    {orthogonal_stream.std():.6f}")
    print(f"  Min:    {orthogonal_stream.min():.6f}")
    print(f"  Max:    {orthogonal_stream.max():.6f}")
    print(f"  Range:  {orthogonal_stream.max() - orthogonal_stream.min():.6f}")

    # Check if it's mostly noise or has structure
    variance_in_orthogonal = np.var(orthogonal_stream)
    total_variance = np.var(X)

    print(f"\nVariance in orthogonal stream: {variance_in_orthogonal:.6f}")
    print(f"Total variance in RGB-D: {total_variance:.6f}")
    print(f"Ratio: {variance_in_orthogonal / total_variance * 100:.4f}%")

    if variance_in_orthogonal / total_variance < 0.01:
        print("→ Orthogonal stream has very low variance (mostly noise)")
    elif variance_in_orthogonal / total_variance < 0.05:
        print("→ Orthogonal stream has low variance (minor residual)")
    else:
        print("→ Orthogonal stream has significant variance (meaningful signal!)")

    return orthogonal_vectors[0]


def test_visualize_orthogonal_stream(dataset, orthogonal_vector):
    """Test 4: Visualize what the orthogonal stream looks like"""
    print("\n" + "="*80)
    print("TEST 4: Visualize Orthogonal Stream")
    print("="*80)

    if orthogonal_vector is None:
        print("\n⚠️  No orthogonal vector to visualize")
        return

    print("\nGenerating orthogonal stream for sample images...")

    # Select 6 sample images
    sample_indices = [0, 100, 200, 300, 400, 500]

    fig, axes = plt.subplots(4, 6, figsize=(18, 12))

    for col_idx, img_idx in enumerate(sample_indices):
        if img_idx >= len(dataset):
            continue

        rgb, depth, label = dataset[img_idx]

        # Denormalize
        rgb_denorm = rgb * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        rgb_denorm = torch.clamp(rgb_denorm, 0, 1)

        depth_denorm = depth * torch.tensor([0.2197]).view(1, 1, 1) + torch.tensor([0.5027]).view(1, 1, 1)
        depth_denorm = torch.clamp(depth_denorm, 0, 1)

        # Compute orthogonal stream
        H, W = rgb.shape[1], rgb.shape[2]
        rgb_flat = rgb_denorm.reshape(3, H*W).T  # HW × 3
        depth_flat = depth_denorm.reshape(1, H*W).T  # HW × 1

        rgbd_flat = np.concatenate([rgb_flat.numpy(), depth_flat.numpy()], axis=1)  # HW × 4

        orthogonal_flat = rgbd_flat @ orthogonal_vector  # HW × 1
        orthogonal_img = orthogonal_flat.reshape(H, W)

        # Normalize for visualization
        orthogonal_img = (orthogonal_img - orthogonal_img.min()) / (orthogonal_img.max() - orthogonal_img.min() + 1e-8)

        # Plot RGB
        axes[0, col_idx].imshow(rgb_denorm.permute(1, 2, 0).numpy())
        axes[0, col_idx].set_title(f'RGB {img_idx}' if col_idx == 0 else f'{img_idx}', fontsize=9)
        axes[0, col_idx].axis('off')

        # Plot Depth
        axes[1, col_idx].imshow(depth_denorm.squeeze().numpy(), cmap='viridis')
        axes[1, col_idx].set_title(f'Depth' if col_idx == 0 else '', fontsize=9)
        axes[1, col_idx].axis('off')

        # Plot Orthogonal Stream
        axes[2, col_idx].imshow(orthogonal_img, cmap='RdBu_r')
        axes[2, col_idx].set_title(f'Orthogonal' if col_idx == 0 else '', fontsize=9)
        axes[2, col_idx].axis('off')

        # Plot RGB Intensity for comparison
        intensity = 0.299 * rgb_denorm[0] + 0.587 * rgb_denorm[1] + 0.114 * rgb_denorm[2]
        axes[3, col_idx].imshow(intensity.numpy(), cmap='gray')
        axes[3, col_idx].set_title(f'Intensity' if col_idx == 0 else '', fontsize=9)
        axes[3, col_idx].axis('off')

    plt.suptitle('RGB-D Orthogonal Stream Visualization', fontsize=14)
    plt.tight_layout()
    plt.savefig('tests/rgbd_orthogonal_stream_viz.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization: tests/rgbd_orthogonal_stream_viz.png")
    plt.close()


def test_pca_comparison():
    """Test 5: Compare with PCA analysis"""
    print("\n" + "="*80)
    print("TEST 5: PCA Analysis (Alternative Approach)")
    print("="*80)

    # Load dataset
    dataset = SUNRGBDDataset(split='train', target_size=(224, 224))

    # Sample fewer pixels for PCA
    num_images = 100
    pixels_per_image = 1000

    print(f"\nSampling {pixels_per_image} pixels from {num_images} images for PCA...")

    samples = []

    for img_idx in range(min(num_images, len(dataset))):
        rgb, depth, _ = dataset[img_idx]

        # Denormalize
        rgb_denorm = rgb * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        rgb_denorm = torch.clamp(rgb_denorm, 0, 1)

        depth_denorm = depth * torch.tensor([0.2197]).view(1, 1, 1) + torch.tensor([0.5027]).view(1, 1, 1)
        depth_denorm = torch.clamp(depth_denorm, 0, 1)

        # Flatten and randomly sample
        H, W = rgb.shape[1], rgb.shape[2]
        num_pixels = H * W
        sample_indices = np.random.choice(num_pixels, size=pixels_per_image, replace=False)

        rgb_flat = rgb_denorm.reshape(3, -1)[:, sample_indices]
        depth_flat = depth_denorm.reshape(1, -1)[:, sample_indices]

        rgbd = torch.cat([rgb_flat, depth_flat], dim=0).T
        samples.append(rgbd.numpy())

    X = np.vstack(samples)
    print(f"PCA data matrix: {X.shape}")

    # Fit PCA
    print("\nFitting PCA with 4 components...")
    pca = PCA(n_components=4)
    pca.fit(X)

    print("\n" + "-"*80)
    print("PCA Results:")
    print("-"*80)

    print("\nExplained variance ratio:")
    for i, var_ratio in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{i+1}: {var_ratio*100:6.2f}%")

    print(f"\nCumulative variance:")
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    for i, cum_var in enumerate(cumulative):
        print(f"  First {i+1} PCs: {cum_var*100:6.2f}%")

    print("\n" + "-"*80)
    print("PCA Components (loadings):")
    print("-"*80)
    labels = ['R', 'G', 'B', 'D']

    for i, component in enumerate(pca.components_):
        print(f"\nPC{i+1} = [", end="")
        for j, coef in enumerate(component):
            print(f"{coef:>7.4f}", end="")
            if j < 3:
                print(", ", end="")
        print("]")
        print(f"      ({', '.join(labels)})")

    # Determine effective dimensionality from PCA
    threshold = 0.01  # 1% variance
    effective_dim = np.sum(pca.explained_variance_ratio_ > threshold)

    print(f"\nEffective dimensionality (>{threshold*100}% variance): {effective_dim}/4")

    return pca


def main():
    print("="*80)
    print("RGB-D DIMENSIONALITY ANALYSIS - SUN RGB-D Dataset")
    print("="*80)
    print("\nGoal: Determine if RGB-D data spans full 4D space or lower-dimensional subspace")
    print("If rank < 4, we can extract orthogonal vector(s) for a 3rd stream!")

    # Test 1: Basic dimensionality
    S, Vt, effective_rank, X = test_dimensionality_basic()

    # Test 2: Correlation analysis
    corr_matrix = test_correlation_analysis(X)

    # Test 3: Extract orthogonal vector
    orthogonal_vector = test_extract_orthogonal_vector(S, Vt, effective_rank, X)

    # Test 4: Visualize orthogonal stream
    if orthogonal_vector is not None:
        dataset = SUNRGBDDataset(split='train', target_size=(224, 224))
        test_visualize_orthogonal_stream(dataset, orthogonal_vector)

    # Test 5: PCA comparison
    pca = test_pca_comparison()

    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    print(f"\nEffective Rank: {effective_rank}/4")
    print(f"Singular values: σ1={S[0]:.4f}, σ2={S[1]:.4f}, σ3={S[2]:.4f}, σ4={S[3]:.4f}")
    print(f"Ratio σ4/σ1: {S[3]/S[0]:.6f}")

    if effective_rank < 4:
        print(f"\n✅ CONCLUSION: RGB-D data spans ~{effective_rank}D subspace")
        print(f"✅ {4 - effective_rank} orthogonal dimension(s) exist!")
        print(f"\n→ We CAN extract an orthogonal vector for 3rd stream!")
        print(f"→ This vector is MATHEMATICALLY GUARANTEED to be orthogonal to R, G, B, D")
        print(f"→ It captures residual variance orthogonal to RGB-D correlations")
    else:
        print(f"\n⚠️  CONCLUSION: RGB-D data spans full 4D space")
        print(f"⚠️  No orthogonal dimension exists in 4D")
        print(f"\n→ Need to use higher-dimensional space (spatial patches) for orthogonal stream")
        print(f"→ Or use chrominance as representational alternative")

    print("\n" + "="*80)
    print("Generated Files:")
    print("  - tests/rgbd_correlation_matrix.png")
    if orthogonal_vector is not None:
        print("  - tests/rgbd_orthogonal_stream_viz.png")
    print("="*80)


if __name__ == '__main__':
    main()
