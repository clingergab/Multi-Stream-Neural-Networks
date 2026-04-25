"""Crop IoU diagnostic for two-view consistency training.

Samples N pairs of augmented views from SUNRGBDDataset(return_two_views=True),
inspects the spatial overlap between view A's crop and view B's crop, and
reports mean IoU + histogram. Run BEFORE kicking off a long consistency run
to confirm the two views aren't disjoint (which would make the KL invariance
target spurious).

Threshold (per the Phase 2 plan):
    - mean IoU >= 0.4
    - fewer than 20% of pairs with IoU < 0.2

Expected for the default RandomCrop(224) from 256x256 source: max offset is
32 px in each axis, so worst-case IoU is (224-32)^2 / (2*224^2 - (224-32)^2)
~= 0.58. Mean should sit around 0.7-0.8. Diagnostic is defensive — failing it
would indicate someone tightened crop scale unexpectedly.

We instrument _apply_spatial_aug by monkey-patching it to record the (i, j)
crop coordinates produced by `RandomCrop.get_params`. This avoids modifying
the dataset class for a one-off diagnostic.

Usage:
    python3 scripts/check_two_view_crop_iou.py \\
        --data-root data/sunrgbd_19_traintest \\
        --num-pairs 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F2

# Allow running as a top-level script from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_utils.sunrgbd_dataset import SUNRGBDDataset


def crop_iou(
    crop_a: tuple[int, int, int, int],
    crop_b: tuple[int, int, int, int],
) -> float:
    """IoU between two axis-aligned crops, each given as (i, j, h, w).

    i, j are top-left corner coords; h, w are crop height/width.
    """
    i_a, j_a, h_a, w_a = crop_a
    i_b, j_b, h_b, w_b = crop_b
    # Intersection
    i_inter_top = max(i_a, i_b)
    i_inter_bot = min(i_a + h_a, i_b + h_b)
    j_inter_left = max(j_a, j_b)
    j_inter_right = min(j_a + w_a, j_b + w_b)
    if i_inter_bot <= i_inter_top or j_inter_right <= j_inter_left:
        return 0.0
    inter = (i_inter_bot - i_inter_top) * (j_inter_right - j_inter_left)
    # Union
    area_a = h_a * w_a
    area_b = h_b * w_b
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def collect_crop_pairs(
    dataset: SUNRGBDDataset,
    num_pairs: int,
    seed: int,
) -> list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]]:
    """Run __getitem__ num_pairs times with monkey-patched _apply_spatial_aug
    that records (i, j, h, w) for each call. Returns a list of (crop_a, crop_b)
    tuples — two consecutive calls per __getitem__ invocation under
    return_two_views=True.
    """
    if not dataset.return_two_views:
        raise ValueError("dataset must be constructed with return_two_views=True")

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(dataset), size=num_pairs).tolist()

    recorded: list[tuple[int, int, int, int]] = []
    original_aug = dataset._apply_spatial_aug

    def patched_aug(rgb, depth):
        # Replicate the original logic but capture the crop coords.
        if np.random.random() < dataset._flip_p:
            rgb = F2.horizontal_flip(rgb)
            depth = F2.horizontal_flip(depth)
        i, j, h, w = v2.RandomCrop.get_params(
            rgb, output_size=(dataset.crop_size, dataset.crop_size)
        )
        recorded.append((int(i), int(j), int(h), int(w)))
        rgb = F2.crop(rgb, i, j, h, w)
        depth = F2.crop(depth, i, j, h, w)
        return rgb, depth

    dataset._apply_spatial_aug = patched_aug
    try:
        for idx in indices:
            _ = dataset[idx]  # triggers two _apply_spatial_aug calls
    finally:
        dataset._apply_spatial_aug = original_aug

    # Pair consecutive crops: (call 0, call 1) is (view A, view B) for sample 0,
    # (call 2, call 3) is (view A, view B) for sample 1, ...
    if len(recorded) != 2 * num_pairs:
        raise RuntimeError(
            f"Expected {2 * num_pairs} crop calls, got {len(recorded)}. "
            f"Two-view path may not have run as expected."
        )
    pairs = [(recorded[2 * k], recorded[2 * k + 1]) for k in range(num_pairs)]
    return pairs


def report(pairs, threshold_mean: float, threshold_low_frac: float, low_iou_cutoff: float):
    ious = np.array([crop_iou(a, b) for a, b in pairs])
    mean_iou = float(ious.mean())
    median_iou = float(np.median(ious))
    min_iou = float(ious.min())
    max_iou = float(ious.max())
    frac_low = float((ious < low_iou_cutoff).mean())

    print("=" * 60)
    print("Two-View Crop IoU Diagnostic")
    print("=" * 60)
    print(f"  Pairs sampled: {len(pairs)}")
    print(f"  Mean IoU:    {mean_iou:.3f}")
    print(f"  Median IoU:  {median_iou:.3f}")
    print(f"  Min IoU:     {min_iou:.3f}")
    print(f"  Max IoU:     {max_iou:.3f}")
    print(f"  Fraction with IoU < {low_iou_cutoff}: {frac_low:.1%}")

    # Histogram (10 bins from 0 to 1)
    print()
    print("  Histogram (10 bins, 0.0-1.0):")
    counts, edges = np.histogram(ious, bins=10, range=(0.0, 1.0))
    for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
        bar = '#' * int(40 * c / max(counts.max(), 1))
        print(f"    [{lo:.1f}, {hi:.1f}) | {c:4d} {bar}")

    print()
    pass_mean = mean_iou >= threshold_mean
    pass_low = frac_low < threshold_low_frac
    print(f"  Threshold mean IoU >= {threshold_mean}: {'PASS' if pass_mean else 'FAIL'}")
    print(f"  Threshold frac(IoU<{low_iou_cutoff}) < {threshold_low_frac}: {'PASS' if pass_low else 'FAIL'}")
    print("=" * 60)
    return pass_mean and pass_low


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=str, default="data/sunrgbd_19_traintest",
        help="Path to preprocessed SUN RGB-D dataset",
    )
    parser.add_argument(
        "--num-pairs", type=int, default=100,
        help="Number of view pairs to sample",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for sample selection",
    )
    parser.add_argument(
        "--threshold-mean", type=float, default=0.4,
        help="Minimum acceptable mean IoU",
    )
    parser.add_argument(
        "--threshold-low-frac", type=float, default=0.2,
        help="Maximum acceptable fraction of pairs with low IoU",
    )
    parser.add_argument(
        "--low-iou-cutoff", type=float, default=0.2,
        help="IoU below this counts as 'low' for the fraction threshold",
    )
    args = parser.parse_args()

    # Seed numpy and torch for reproducible aug draws inside the dataset
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = SUNRGBDDataset(
        data_root=args.data_root,
        split='train',
        return_two_views=True,
    )

    pairs = collect_crop_pairs(dataset, num_pairs=args.num_pairs, seed=args.seed)
    ok = report(
        pairs,
        threshold_mean=args.threshold_mean,
        threshold_low_frac=args.threshold_low_frac,
        low_iou_cutoff=args.low_iou_cutoff,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
