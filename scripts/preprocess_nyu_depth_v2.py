"""
Preprocess NYU Depth V2 dataset into the standard 10-category scene benchmark.

The 10 categories are the standard Silberman et al. 2012 set:
  9 named categories (bathroom, bedroom, bookstore, classroom, dining_room,
  home_office, kitchen, living_room, office) + 1 "others" category that
  bundles the remaining ~19 scene types. All 10 are trained and evaluated.

This script:
  1. Loads the official nyu_depth_v2_labeled.mat (HDF5 format, 1449 images).
  2. Extracts scene types and maps to the 10-category scheme.
  3. With --dry-run: prints distribution only (no files written).
  4. Without --dry-run: builds train/val/test tensor files at 256x256.

The 224x224 crop is applied at training time in the dataloader:
  - Train: RandomCrop(224)
  - Val/Test: CenterCrop(224)

Usage:
  # Check distribution first (no files written)
  python3 scripts/preprocess_nyu_depth_v2.py --dry-run

  # Build 3-way split (train 80% / val 20% / test official)
  python3 scripts/preprocess_nyu_depth_v2.py

  # Build 2-way split (all official train / test) for k-fold CV
  python3 scripts/preprocess_nyu_depth_v2.py --no-val-split

Output: data/nyu_depth_v2_10/ (or data/nyu_depth_v2_10_traintest/ with --no-val-split)
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict

import h5py
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
MAT_FILE_DEFAULT = "data/nyu_depth_v2_labeled.mat"
SPLIT_CSV = "data/nyu/nyu2_test.csv"
TARGET_SIZE = (256, 256)

# Silberman et al. 2012 — 9 named categories.
# Everything else maps to "others" (class index 9).
_STANDARD_9 = {
    "bathroom", "bedroom", "bookstore", "classroom", "dining_room",
    "home_office", "kitchen", "living_room", "office",
}

# Final 10-category list: 9 named (sorted) + "others" last.
NYU_CATEGORIES = sorted(_STANDARD_9) + ["others"]
_CAT_TO_IDX = {name: idx for idx, name in enumerate(NYU_CATEGORIES)}


# ---------------------------------------------------------------------------
# Scene name extraction
# ---------------------------------------------------------------------------

def _extract_scene_type(scene_name: str) -> str:
    """Extract scene type from a scene name like 'bedroom_0020' or 'nyu_office_0001'.

    Strips the trailing _NNNN[a-z]* suffix while preserving multi-word
    prefixes like 'dining_room', 'home_office', 'nyu_office'.
    """
    m = re.match(r"(.+?)_\d+[a-z]*$", scene_name)
    if m:
        return m.group(1)
    return scene_name


def _map_to_category(scene_type: str) -> str:
    """Map a raw scene type to one of the 10 standard categories."""
    if scene_type in _STANDARD_9:
        return scene_type
    return "others"


def load_scene_labels(h5f):
    """Extract per-sample scene types from the HDF5 scenes dataset.

    Returns:
        scene_types: list of str, length 1449 (raw scene types)
        mapped_categories: list of str, length 1449 (10-category mapped)
    """
    scenes_ds = h5f["scenes"]
    num_samples = scenes_ds.shape[1]  # shape is (1, N) for MATLAB row vector

    scene_types = []
    for i in range(num_samples):
        ref = scenes_ds[0, i]
        ascii_array = np.array(h5f[ref])
        scene_name = "".join(chr(int(c)) for c in ascii_array.flatten())
        scene_type = _extract_scene_type(scene_name)
        scene_types.append(scene_type)

    mapped = [_map_to_category(st) for st in scene_types]
    return scene_types, mapped


# ---------------------------------------------------------------------------
# Distribution printing
# ---------------------------------------------------------------------------

def print_distribution(scene_types, mapped_categories):
    """Print raw and mapped category distributions."""
    raw_counts = Counter(scene_types)
    mapped_counts = Counter(mapped_categories)

    print(f"\nRaw scene type distribution ({len(raw_counts)} unique):")
    print(f"  {'Scene Type':<22}  {'Count':>6}  {'Maps To'}")
    print(f"  {'-'*22}  {'-'*6}  {'-'*12}")
    for st in sorted(raw_counts):
        cat = _map_to_category(st)
        marker = "" if cat == st else f" -> {cat}"
        print(f"  {st:<22}  {raw_counts[st]:>6}{marker}")

    print(f"\n10-category distribution:")
    print(f"  {'Category':<22}  {'Count':>6}  {'Index':>5}")
    print(f"  {'-'*22}  {'-'*6}  {'-'*5}")
    for cat in NYU_CATEGORIES:
        print(f"  {cat:<22}  {mapped_counts.get(cat, 0):>6}  {_CAT_TO_IDX[cat]:>5}")
    total = sum(mapped_counts.values())
    print(f"  {'TOTAL':<22}  {total:>6}")

    # Sanity check
    assert total == len(scene_types), (
        f"Mapped total {total} != sample count {len(scene_types)}"
    )


# ---------------------------------------------------------------------------
# Official split
# ---------------------------------------------------------------------------

def load_official_split():
    """Load official 795/654 train/test split from nyu2_test.csv.

    Returns:
        train_indices: sorted list of 0-indexed train sample indices
        test_indices: sorted list of 0-indexed test sample indices
    """
    if not os.path.exists(SPLIT_CSV):
        raise FileNotFoundError(
            f"Split CSV not found: {SPLIT_CSV}\n"
            f"Expected nyu2_test.csv with lines like: "
            f"data/nyu2_test/00000_colors.png,data/nyu2_test/00000_depth.png"
        )

    test_indices = set()
    with open(SPLIT_CSV) as f:
        for line in f:
            m = re.match(r"data/nyu2_test/(\d+)_colors\.png", line.strip())
            if m:
                test_indices.add(int(m.group(1)))

    all_indices = set(range(1449))
    train_indices = sorted(all_indices - test_indices)
    test_indices = sorted(test_indices)

    print(f"Official split: {len(train_indices)} train, {len(test_indices)} test")
    assert len(train_indices) == 795, f"Expected 795 train, got {len(train_indices)}"
    assert len(test_indices) == 654, f"Expected 654 test, got {len(test_indices)}"

    return train_indices, test_indices


# ---------------------------------------------------------------------------
# Tensor building
# ---------------------------------------------------------------------------

def build_split(split_name, indices, mapped_categories, h5f, output_base):
    """Build RGB/depth tensors and labels for one split."""
    split_dir = os.path.join(output_base, split_name)
    os.makedirs(split_dir, exist_ok=True)

    N = len(indices)
    H, W = TARGET_SIZE
    print(f"\nBuilding {split_name} ({N} samples) ...")

    rgb_t = torch.empty(N, 3, H, W, dtype=torch.uint8)
    depth_t = torch.empty(N, 1, H, W, dtype=torch.uint16)
    labels = []

    images_ds = h5f["images"]
    depths_ds = h5f["depths"]

    for i, idx in enumerate(tqdm(indices)):
        # --- RGB ---
        rgb = np.array(images_ds[idx])
        assert rgb.shape == (3, 640, 480), (
            f"Sample {idx}: expected images shape (3, 640, 480), got {rgb.shape}"
        )
        rgb = np.transpose(rgb, (1, 2, 0))  # [640, 480, 3]
        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8)
        rgb_img = Image.fromarray(rgb, mode="RGB")
        rgb_img = TF.resize(rgb_img, TARGET_SIZE)
        rgb_arr = np.array(rgb_img, dtype=np.uint8)
        rgb_t[i] = torch.from_numpy(rgb_arr).permute(2, 0, 1)

        # --- Depth ---
        depth = np.array(depths_ds[idx])
        assert depth.shape == (640, 480), (
            f"Sample {idx}: expected depths shape (640, 480), got {depth.shape}"
        )
        # Convert meters -> millimeters
        depth_mm = depth * 1000.0
        assert depth_mm.max() < 65535, (
            f"Sample {idx}: depth exceeds uint16 range: max={depth_mm.max():.1f} mm"
        )
        depth_mm = np.clip(depth_mm, 0, 65535).astype(np.uint16)
        depth_img = Image.fromarray(depth_mm, mode="I;16")
        depth_img = TF.resize(
            depth_img, TARGET_SIZE,
            interpolation=TF.InterpolationMode.NEAREST,
        )
        depth_arr = np.array(depth_img, dtype=np.uint16)
        depth_t[i] = torch.from_numpy(depth_arr).unsqueeze(0)

        # --- Label ---
        cat = mapped_categories[idx]
        labels.append(_CAT_TO_IDX[cat])

    # Save
    rgb_out = os.path.join(split_dir, "rgb_tensors.pt")
    depth_out = os.path.join(split_dir, "depth_tensors.pt")
    torch.save(rgb_t, rgb_out)
    torch.save(depth_t, depth_out)

    with open(os.path.join(split_dir, "labels.txt"), "w") as f:
        f.write("\n".join(map(str, labels)) + "\n")

    print(f"  rgb_tensors.pt:   {tuple(rgb_t.shape)}  "
          f"{os.path.getsize(rgb_out)/1e6:.1f} MB")
    print(f"  depth_tensors.pt: {tuple(depth_t.shape)}  "
          f"{os.path.getsize(depth_out)/1e6:.1f} MB")
    return labels


# ---------------------------------------------------------------------------
# Split routing
# ---------------------------------------------------------------------------

def run_three_way(mapped_categories, h5f, output_base, val_ratio=0.2, seed=42):
    train_indices, test_indices = load_official_split()

    # Stratified sub-split of train -> train/val
    np.random.seed(seed)
    class_groups = defaultdict(list)
    for idx in train_indices:
        cls = _CAT_TO_IDX[mapped_categories[idx]]
        class_groups[cls].append(idx)

    train_sub, val_sub = [], []
    for cls in sorted(class_groups):
        idx_arr = np.array(class_groups[cls])
        np.random.shuffle(idx_arr)
        cut = int(len(idx_arr) * (1.0 - val_ratio))
        train_sub.extend(idx_arr[:cut].tolist())
        val_sub.extend(idx_arr[cut:].tolist())

    print(f"\n3-way split:")
    print(f"  Train: {len(train_sub)}  (80% of official train, stratified)")
    print(f"  Val:   {len(val_sub)}   (20% of official train)")
    print(f"  Test:  {len(test_indices)}  (official test set)")

    trl = build_split("train", train_sub, mapped_categories, h5f, output_base)
    vl = build_split("val", val_sub, mapped_categories, h5f, output_base)
    tel = build_split("test", test_indices, mapped_categories, h5f, output_base)
    _compute_and_save_norm_stats(output_base)
    _write_meta(output_base, len(train_sub), len(val_sub), len(test_indices),
                trl, vl, tel, val_ratio, seed)


def run_two_way(mapped_categories, h5f, output_base):
    train_indices, test_indices = load_official_split()

    print(f"\n2-way split (no val sub-split, for k-fold CV):")
    print(f"  Train: {len(train_indices)}  (all official train)")
    print(f"  Test:  {len(test_indices)}  (official test)")

    trl = build_split("train", train_indices, mapped_categories, h5f, output_base)
    tel = build_split("test", test_indices, mapped_categories, h5f, output_base)
    _compute_and_save_norm_stats(output_base)
    _write_meta(output_base, len(train_indices), 0, len(test_indices),
                trl, [], tel, None, None)


# ---------------------------------------------------------------------------
# Normalization statistics
# ---------------------------------------------------------------------------

def _compute_and_save_norm_stats(output_base):
    """Compute channel-wise mean/std from TRAINING split and save to norm_stats.json."""
    train_dir = os.path.join(output_base, "train")

    rgb_t = torch.load(os.path.join(train_dir, "rgb_tensors.pt"), weights_only=True)
    depth_t = torch.load(os.path.join(train_dir, "depth_tensors.pt"), weights_only=True)

    rgb_f = rgb_t.float() / 255.0
    depth_f = depth_t.float() / 1000.0

    rgb_mean = rgb_f.mean(dim=(0, 2, 3)).tolist()
    rgb_std = rgb_f.std(dim=(0, 2, 3)).tolist()

    valid_depth = depth_f[depth_t.to(torch.int32) > 0]
    depth_mean = [valid_depth.mean().item()]
    depth_std = [valid_depth.std().item()]

    stats = {
        "rgb_mean": rgb_mean,
        "rgb_std": rgb_std,
        "depth_mean": depth_mean,
        "depth_std": depth_std,
    }

    out_path = os.path.join(output_base, "norm_stats.json")
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nNormalization stats (from training split):")
    print(f"  RGB mean:   {rgb_mean}")
    print(f"  RGB std:    {rgb_std}")
    print(f"  Depth mean: {depth_mean}")
    print(f"  Depth std:  {depth_std}")
    print(f"  Saved to: {out_path}")

    return stats


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def _write_meta(output_base, n_tr, n_val, n_te,
                tr_labels, val_labels, te_labels, val_ratio, seed):
    n_cats = len(NYU_CATEGORIES)

    with open(os.path.join(output_base, "class_names.txt"), "w") as f:
        for i, name in enumerate(NYU_CATEGORIES):
            f.write(f"{i}: {name}\n")

    with open(os.path.join(output_base, "dataset_info.txt"), "w") as f:
        f.write(f"NYU Depth V2 {n_cats}-Category Scene Benchmark\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Categories: {n_cats} (9 named + others)\n")
        f.write("Split source: official 795/654 (from nyu2_test.csv)\n")
        f.write(f"Tensor size: {TARGET_SIZE}  (crop to 224x224 in dataloader)\n\n")
        if val_ratio is not None:
            f.write(f"Train/Val: {1-val_ratio:.0%}/{val_ratio:.0%} "
                    f"stratified (seed={seed})\n")
            f.write("Note: some classes have very few val samples "
                    "due to small dataset size.\n\n")
        else:
            f.write("No val sub-split (k-fold CV mode)\n\n")
        f.write(f"Train: {n_tr}\n")
        if n_val:
            f.write(f"Val:   {n_val}\n")
        f.write(f"Test:  {n_te}\n")
        f.write(f"Total: {n_tr + n_val + n_te}\n\n")
        f.write("Class names:\n")
        for i, name in enumerate(NYU_CATEGORIES):
            f.write(f"  {i:2d}: {name}\n")
        for split_name, lbls in [("Train", tr_labels),
                                  ("Val", val_labels),
                                  ("Test", te_labels)]:
            if not lbls:
                continue
            f.write(f"\n{split_name} distribution:\n")
            c = Counter(lbls)
            for i in range(n_cats):
                f.write(f"  {i:2d} {NYU_CATEGORIES[i]:22s}: {c.get(i, 0):5d}\n")

    print(f"\nDone. Output: {output_base}")
    print(f"class_names.txt lists the {n_cats} categories.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess NYU Depth V2 into standard 10-category "
                    "scene benchmark (9 named + others)."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print scene distribution only. No files written.",
    )
    parser.add_argument(
        "--no-val-split", action="store_true",
        help="Skip train/val sub-split (for k-fold CV).",
    )
    parser.add_argument(
        "--mat-file", type=str, default=MAT_FILE_DEFAULT,
        help=f"Path to nyu_depth_v2_labeled.mat (default: {MAT_FILE_DEFAULT}).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: data/nyu_depth_v2_10 or "
             "data/nyu_depth_v2_10_traintest).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.mat_file):
        print(f".mat file not found: {args.mat_file}")
        print("Downloading NYU Depth V2 labeled dataset (~2.8 GB) ...")
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "download_nyu_depth_v2",
            os.path.join(os.path.dirname(__file__), "download_nyu_depth_v2.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        download_nyu_depth_v2 = mod.download_nyu_depth_v2
        save_dir = os.path.dirname(args.mat_file) or "data"
        filename = os.path.basename(args.mat_file)
        download_nyu_depth_v2(save_dir=save_dir, filename=filename)

    print(f"Loading {args.mat_file} ...")
    with h5py.File(args.mat_file, "r") as h5f:
        # Step A: extract scene types
        scene_types, mapped_categories = load_scene_labels(h5f)
        print_distribution(scene_types, mapped_categories)

        if args.dry_run:
            print("\n--dry-run: stopping here. No files written.")
            return

        # Step C-F: build splits and metadata
        n_cats = len(NYU_CATEGORIES)
        if args.no_val_split:
            default_out = args.output_dir or f"data/nyu_depth_v2_{n_cats}_traintest"
            run_two_way(mapped_categories, h5f, default_out)
        else:
            default_out = args.output_dir or f"data/nyu_depth_v2_{n_cats}"
            run_three_way(mapped_categories, h5f, default_out)


if __name__ == "__main__":
    main()
