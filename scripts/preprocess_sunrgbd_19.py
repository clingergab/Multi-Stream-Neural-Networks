"""
Preprocess SUN RGB-D dataset into the standard 19-category benchmark.

The 19 categories are the standard "all classes with >80 images" set used by
all comparison papers (TRecgNet, CBCL baseline, Song et al., etc.).

This script:
  1. Scans all scene.txt files and counts samples per raw label.
  2. Keeps only categories with >80 total samples (yielding 19 categories).
  3. With --dry-run: stops after printing the distribution (no files written).
  4. Without --dry-run: builds train/val/test tensor files at 256x256.

The 224x224 crop is applied at training time in the dataloader:
  - Train: RandomCrop(224)
  - Val/Test: CenterCrop(224)

Usage:
  # Check distribution first (no files written)
  python preprocess_sunrgbd_19.py --dry-run

  # Build 3-way split (train 80% / val 20% / test official)
  python preprocess_sunrgbd_19.py

  # Build 2-way split (all official train / test) for k-fold CV
  python preprocess_sunrgbd_19.py --no-val-split

Output: data/sunrgbd_19/ (or data/sunrgbd_19_traintest/ with --no-val-split)
"""

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SUNRGBD_BASE = "data/sunrgbd/SUNRGBD"
TOOLBOX_PATH = "data/sunrgbd/SUNRGBDtoolbox"
SPLIT_FILE   = os.path.join(TOOLBOX_PATH, "traintestSUNRGBD/allsplit.mat")
MAT_PREFIX   = "/n/fs/sun3d/data/SUNRGBD/"

TARGET_SIZE  = (256, 256)   # tensors saved at 256x256; crop to 224 in dataloader

# The standard 19-category benchmark (Song et al. 2015, TRecgNet, CBCL).
# These are the 19 scene types used by all comparison papers.
# "office_kitchen" (82 samples) is excluded — it's not part of the standard set.
_STANDARD_19 = {
    "bathroom", "bedroom", "classroom", "computer_room", "conference_room",
    "corridor", "dining_area", "dining_room", "discussion_area",
    "furniture_store", "home_office", "kitchen", "lab", "lecture_theatre",
    "library", "living_room", "office", "rest_space", "study_space",
}

# Populated after build_category_list() is called
SUNRGBD_CATEGORIES = None
_CAT_TO_IDX        = None


def build_category_list(category_counts: Counter, min_samples: int = 80):
    """
    Determine the final category list from per-category counts.
    Prints the full distribution and flags which categories clear the threshold.
    Returns the sorted list of accepted category names.
    """
    global SUNRGBD_CATEGORIES, _CAT_TO_IDX

    print(f"\nPer-category sample counts (threshold: >{min_samples}):")
    print(f"  {'Category':<22}  {'Count':>6}  Status")
    print(f"  {'-'*22}  {'-'*6}  ------")

    accepted, rejected = [], []
    for cat in sorted(category_counts):
        n = category_counts[cat]
        ok = n > min_samples
        print(f"  {cat:<22}  {n:>6}  {'OK' if ok else 'SKIP'}")
        (accepted if ok else rejected).append(cat)

    if rejected:
        print(f"\n  Skipped (<={min_samples} samples): {rejected}")

    print(f"\n  Final category count: {len(accepted)}")

    SUNRGBD_CATEGORIES = sorted(accepted)
    _CAT_TO_IDX = {name: idx for idx, name in enumerate(SUNRGBD_CATEGORIES)}
    return SUNRGBD_CATEGORIES


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def find_rgb_depth(sample_dir: str):
    rgb_path = None
    for ext in ["*.jpg", "*.png"]:
        hits = list(Path(sample_dir, "image").glob(ext))
        if hits:
            rgb_path = str(hits[0])
            break

    depth_path = None
    for sub in ["depth_bfx", "depth"]:
        hits = list(Path(sample_dir, sub).glob("*.png"))
        if hits:
            depth_path = str(hits[0])
            break

    return rgb_path, depth_path


def load_official_split():
    if not os.path.exists(SPLIT_FILE):
        raise FileNotFoundError(
            f"Official split file not found: {SPLIT_FILE}\n"
            f"Ensure SUNRGBDtoolbox is at {TOOLBOX_PATH}"
        )
    mat = sio.loadmat(SPLIT_FILE)

    def _extract(arr):
        paths = set()
        for entry in arr.flatten():
            p = str(entry.flatten()[0]).replace(MAT_PREFIX, "").rstrip("/")
            paths.add(p)
        return paths

    train_paths = _extract(mat["alltrain"])
    test_paths  = _extract(mat["alltest"])
    print(f"Official split: {len(train_paths)} train paths, {len(test_paths)} test paths")
    return train_paths, test_paths


# ---------------------------------------------------------------------------
# Sample collection (two passes)
# ---------------------------------------------------------------------------

def collect_samples_pass1():
    """
    Scan all scene.txt files, count per-label totals.
    Returns (samples_raw, category_counts).
    samples_raw: list of (sample_dir, label, rgb_path, depth_path)
    label is the raw scene.txt value (lowercased), or None for skipped labels.
    """
    samples_raw = []
    category_counts = Counter()
    skipped_files = 0
    skipped_labels = 0

    print("Scanning SUN RGB-D dataset ...")
    for root, dirs, files in os.walk(SUNRGBD_BASE):
        if "scene.txt" not in files:
            continue
        try:
            label = open(os.path.join(root, "scene.txt")).read().strip().lower()

            rgb, depth = find_rgb_depth(root)
            if rgb is None or depth is None:
                skipped_files += 1
                continue

            if label not in _STANDARD_19:
                skipped_labels += 1
                continue

            category_counts[label] += 1
            samples_raw.append((root, label, rgb, depth))

        except Exception as e:
            print(f"  Error at {root}: {e}")

    print(f"\n  Total samples scanned:   {len(samples_raw)}")
    print(f"  Skipped (missing files): {skipped_files}")
    print(f"  Skipped (not in 19):     {skipped_labels}")

    return samples_raw, category_counts


def filter_samples(samples_raw):
    """Keep only samples whose label passed the >80 threshold."""
    return [
        (sd, label, _CAT_TO_IDX[label], rgb, depth)
        for (sd, label, rgb, depth) in samples_raw
        if label is not None and label in _CAT_TO_IDX
    ]


# ---------------------------------------------------------------------------
# Tensor building
# ---------------------------------------------------------------------------

def _read_rgb(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    img = TF.resize(img, TARGET_SIZE)
    arr = np.array(img, dtype=np.uint8)
    return torch.from_numpy(arr).permute(2, 0, 1)


def _read_depth(path: str) -> torch.Tensor:
    d = Image.open(path)
    if d.mode in ("I", "I;16", "I;16B"):
        arr = np.array(d, dtype=np.float32)
        arr = np.clip(arr / 65535.0 * 255.0, 0, 255).astype(np.uint8)
        d = Image.fromarray(arr, mode="L")
    else:
        d = d.convert("L")
    d = TF.resize(d, TARGET_SIZE)
    arr = np.array(d, dtype=np.uint8)
    return torch.from_numpy(arr).unsqueeze(0)


def build_split(split_name: str, split_samples: list, output_base: str):
    split_dir = os.path.join(output_base, split_name)
    os.makedirs(split_dir, exist_ok=True)

    N = len(split_samples)
    H, W = TARGET_SIZE
    print(f"\nBuilding {split_name} ({N} samples) ...")

    rgb_t   = torch.empty(N, 3, H, W, dtype=torch.uint8)
    depth_t = torch.empty(N, 1, H, W, dtype=torch.uint8)
    labels  = []

    for i, (_, label, cls, rgb_path, depth_path) in enumerate(tqdm(split_samples)):
        rgb_t[i]   = _read_rgb(rgb_path)
        depth_t[i] = _read_depth(depth_path)
        labels.append(cls)

    rgb_out   = os.path.join(split_dir, "rgb_tensors.pt")
    depth_out = os.path.join(split_dir, "depth_tensors.pt")
    torch.save(rgb_t,   rgb_out)
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

def _match_official(samples, train_paths, test_paths):
    path_to_sample = {
        os.path.relpath(s[0], SUNRGBD_BASE): s for s in samples
    }
    official_train, official_test = [], []
    skip_tr = skip_te = 0

    for p in train_paths:
        if p in path_to_sample:
            official_train.append(path_to_sample[p])
        else:
            skip_tr += 1

    for p in test_paths:
        if p in path_to_sample:
            official_test.append(path_to_sample[p])
        else:
            skip_te += 1

    n_cat = len(SUNRGBD_CATEGORIES)
    print(f"\nOfficial split matching ({n_cat}-class filtered):")
    print(f"  Train: {len(official_train)} matched  ({skip_tr} not in filtered set)")
    print(f"  Test:  {len(official_test)} matched  ({skip_te} not in filtered set)")
    return official_train, official_test


def run_three_way(samples, train_paths, test_paths, output_base, val_ratio=0.2, seed=42):
    official_train, official_test = _match_official(samples, train_paths, test_paths)

    np.random.seed(seed)
    class_groups = defaultdict(list)
    for i, s in enumerate(official_train):
        class_groups[s[2]].append(i)

    train_idx, val_idx = [], []
    for cls in sorted(class_groups):
        idx = np.array(class_groups[cls])
        np.random.shuffle(idx)
        cut = int(len(idx) * (1.0 - val_ratio))
        train_idx.extend(idx[:cut])
        val_idx.extend(idx[cut:])

    train_s = [official_train[i] for i in train_idx]
    val_s   = [official_train[i] for i in val_idx]
    test_s  = official_test

    print(f"\n3-way split:")
    print(f"  Train: {len(train_s)}  (80% of official train, stratified)")
    print(f"  Val:   {len(val_s)}   (20% of official train)")
    print(f"  Test:  {len(test_s)}  (official test set)")

    trl = build_split("train", train_s, output_base)
    vl  = build_split("val",   val_s,   output_base)
    tel = build_split("test",  test_s,  output_base)
    _compute_and_save_norm_stats(output_base)
    _write_meta(output_base, len(train_s), len(val_s), len(test_s),
                trl, vl, tel, val_ratio, seed)


def run_two_way(samples, train_paths, test_paths, output_base):
    official_train, official_test = _match_official(samples, train_paths, test_paths)

    print(f"\n2-way split (no val sub-split, for k-fold CV):")
    print(f"  Train: {len(official_train)}  (all official train)")
    print(f"  Test:  {len(official_test)}  (official test)")

    trl = build_split("train", official_train, output_base)
    tel = build_split("test",  official_test,  output_base)
    _compute_and_save_norm_stats(output_base)
    _write_meta(output_base, len(official_train), 0, len(official_test),
                trl, [], tel, None, None)


# ---------------------------------------------------------------------------
# Normalization statistics
# ---------------------------------------------------------------------------

def _compute_and_save_norm_stats(output_base: str):
    """Compute channel-wise mean/std from the TRAINING split tensors and save to norm_stats.json."""
    train_dir = os.path.join(output_base, "train")

    rgb_t = torch.load(os.path.join(train_dir, "rgb_tensors.pt"), weights_only=True)
    depth_t = torch.load(os.path.join(train_dir, "depth_tensors.pt"), weights_only=True)

    # Convert uint8 -> float32 [0, 1]
    rgb_f = rgb_t.float() / 255.0    # [N, 3, H, W]
    depth_f = depth_t.float() / 255.0  # [N, 1, H, W]

    # Per-channel mean and std across all pixels
    rgb_mean = rgb_f.mean(dim=(0, 2, 3)).tolist()
    rgb_std = rgb_f.std(dim=(0, 2, 3)).tolist()
    depth_mean = [depth_f.mean().item()]
    depth_std = [depth_f.std().item()]

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
    n_cats = len(SUNRGBD_CATEGORIES)

    with open(os.path.join(output_base, "class_names.txt"), "w") as f:
        for i, name in enumerate(SUNRGBD_CATEGORIES):
            f.write(f"{i}: {name}\n")

    with open(os.path.join(output_base, "dataset_info.txt"), "w") as f:
        f.write(f"SUN RGB-D {n_cats}-Category Standard Benchmark\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Categories: {n_cats} (all with >80 images)\n")
        f.write("Split source: SUNRGBDtoolbox allsplit.mat\n")
        f.write(f"Tensor size: {TARGET_SIZE}  (crop to 224x224 in dataloader)\n\n")
        if val_ratio is not None:
            f.write(f"Train/Val: {1-val_ratio:.0%}/{val_ratio:.0%} stratified (seed={seed})\n\n")
        else:
            f.write("No val sub-split (k-fold CV mode)\n\n")
        f.write(f"Train: {n_tr}\n")
        if n_val:
            f.write(f"Val:   {n_val}\n")
        f.write(f"Test:  {n_te}\n")
        f.write(f"Total: {n_tr + n_val + n_te}\n\n")
        f.write("Class names:\n")
        for i, name in enumerate(SUNRGBD_CATEGORIES):
            f.write(f"  {i:2d}: {name}\n")
        for split_name, lbls in [("Train", tr_labels), ("Val", val_labels), ("Test", te_labels)]:
            if not lbls:
                continue
            f.write(f"\n{split_name} distribution:\n")
            c = Counter(lbls)
            for i in range(n_cats):
                f.write(f"  {i:2d} {SUNRGBD_CATEGORIES[i]:22s}: {c.get(i, 0):5d}\n")

    print(f"\nDone. Output: {output_base}")
    print(f"class_names.txt lists the {n_cats} accepted categories.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess SUN RGB-D into standard N-category benchmark "
                    "(categories with >80 images)."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Scan and print category distribution only. No files written."
    )
    parser.add_argument(
        "--no-val-split", action="store_true",
        help="Skip train/val sub-split (for k-fold CV)."
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: data/sunrgbd_N or data/sunrgbd_N_trainval)."
    )
    parser.add_argument(
        "--min-samples", type=int, default=80,
        help="Minimum samples for a category to be included (default: 80)."
    )
    args = parser.parse_args()

    # Pass 1: scan + count
    samples_raw, category_counts = collect_samples_pass1()

    # Determine final category list (always printed)
    accepted = build_category_list(category_counts, min_samples=args.min_samples)

    if args.dry_run:
        print("\n--dry-run: stopping here. No files written.")
        print(f"Accepted categories ({len(accepted)}): {accepted}")
        return

    # Pass 2: filter
    samples = filter_samples(samples_raw)
    print(f"\nSamples after category filter: {len(samples)}")

    train_paths, test_paths = load_official_split()

    n = len(accepted)
    if args.no_val_split:
        default_out = args.output_dir or f"data/sunrgbd_{n}_traintest"
        run_two_way(samples, train_paths, test_paths, default_out)
    else:
        default_out = args.output_dir or f"data/sunrgbd_{n}"
        run_three_way(samples, train_paths, test_paths, default_out)


if __name__ == "__main__":
    main()