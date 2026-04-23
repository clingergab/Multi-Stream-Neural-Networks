"""
Analyze scene group structure in SUN RGB-D training set to quantify
potential data leakage from random train/val splits.

Scans the raw SUNRGBD directory, maps samples to the official training
split, extracts scene group IDs, and simulates both random and
group-aware splits to measure leakage.

Usage:
    python3 scripts/analyze_scene_groups.py
"""

import os
import re
from collections import Counter, defaultdict

import numpy as np
import scipy.io as sio
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

# ---------------------------------------------------------------------------
# Paths (same as preprocess_sunrgbd_19.py)
# ---------------------------------------------------------------------------
SUNRGBD_BASE = "data/sunrgbd/SUNRGBD"
SPLIT_FILE = "data/sunrgbd/SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat"
MAT_PREFIX = "/n/fs/sun3d/data/SUNRGBD/"

_STANDARD_19 = {
    "bathroom", "bedroom", "classroom", "computer_room", "conference_room",
    "corridor", "dining_area", "dining_room", "discussion_area",
    "furniture_store", "home_office", "kitchen", "lab", "lecture_theatre",
    "library", "living_room", "office", "rest_space", "study_space",
}


def extract_scene_group(rel_path):
    """Extract scene group ID from a sample's relative path within SUNRGBD_BASE.

    - xtion/sun3ddata/SCENE/...: group = xtion/sun3ddata/SCENE
    - kv2/kinect2data/SEQNUM_SESSION_rgbfFRAME-resize: group by session
    - kv1/*, realsense/*: each sample is its own group
    """
    parts = rel_path.split("/")

    if parts[0] == "xtion" and len(parts) >= 3:
        return "/".join(parts[:3])

    if parts[0] == "kv2" and len(parts) >= 2:
        dirname = parts[-1]
        session = re.sub(r"^\d+_", "", dirname)
        session = re.sub(r"_rgbf\d+-resize$", "", session)
        return f"kv2/kinect2data/{session}"

    return rel_path


def load_official_split():
    mat = sio.loadmat(SPLIT_FILE)

    def _extract(arr):
        paths = set()
        for entry in arr.flatten():
            p = str(entry.flatten()[0]).replace(MAT_PREFIX, "").rstrip("/")
            paths.add(p)
        return paths

    return _extract(mat["alltrain"]), _extract(mat["alltest"])


def collect_training_samples():
    """Walk SUNRGBD, filter to 19 classes + official train split."""
    train_paths, _ = load_official_split()

    samples = []
    for root, dirs, files in os.walk(SUNRGBD_BASE):
        if "scene.txt" not in files:
            continue
        label = open(os.path.join(root, "scene.txt")).read().strip().lower()
        if label not in _STANDARD_19:
            continue
        rel = os.path.relpath(root, SUNRGBD_BASE)
        if rel in train_paths:
            samples.append((rel, label))

    return samples


def main():
    print("=" * 70)
    print("SUN RGB-D Scene Group Analysis")
    print("=" * 70)

    samples = collect_training_samples()
    print(f"\nTotal training samples (19-class): {len(samples)}")

    # Extract scene groups
    groups = [extract_scene_group(rel) for rel, _ in samples]
    labels = [label for _, label in samples]

    group_counts = Counter(groups)
    unique_groups = len(group_counts)
    print(f"Unique scene groups: {unique_groups}")

    # Source breakdown
    source_counts = Counter()
    source_groups = defaultdict(set)
    for rel, _ in samples:
        source = rel.split("/")[0]
        source_counts[source] += 1
        source_groups[source].add(extract_scene_group(rel))

    print(f"\nBreakdown by source:")
    print(f"  {'Source':<12} {'Samples':>8} {'Groups':>8} {'Avg/Group':>10}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*10}")
    for src in sorted(source_counts):
        n = source_counts[src]
        g = len(source_groups[src])
        print(f"  {src:<12} {n:>8} {g:>8} {n/g:>10.1f}")

    # Distribution histogram
    sizes = list(group_counts.values())
    bins = [(1, 1), (2, 5), (6, 10), (11, 20), (21, 50), (51, float("inf"))]
    print(f"\nScene group size distribution:")
    print(f"  {'Size range':<14} {'Groups':>8} {'Samples':>10}")
    print(f"  {'-'*14} {'-'*8} {'-'*10}")
    for lo, hi in bins:
        matching = [s for s in sizes if lo <= s <= hi]
        label = f"{lo}" if lo == hi else f"{lo}-{int(hi) if hi != float('inf') else '∞'}"
        print(f"  {label:<14} {len(matching):>8} {sum(matching):>10}")

    multi = [(g, c) for g, c in group_counts.items() if c > 1]
    print(f"\nScene groups with >1 sample: {len(multi)} "
          f"({sum(c for _, c in multi)} samples, "
          f"{sum(c for _, c in multi)/len(samples)*100:.1f}% of training set)")

    # Top 20 largest groups
    print(f"\nTop 20 largest scene groups:")
    print(f"  {'Scene Group':<50} {'N':>4}  Classes")
    print(f"  {'-'*50} {'-'*4}  {'-'*30}")
    sorted_groups = sorted(group_counts.items(), key=lambda x: -x[1])[:20]
    group_to_labels = defaultdict(list)
    for (rel, label), g in zip(samples, groups):
        group_to_labels[g].append(label)
    for g, n in sorted_groups:
        cls_dist = Counter(group_to_labels[g])
        cls_str = ", ".join(f"{c}({n})" for c, n in cls_dist.most_common(5))
        print(f"  {g:<50} {n:>4}  {cls_str}")

    # --- Simulate leakage ---
    print("\n" + "=" * 70)
    print("LEAKAGE SIMULATION")
    print("=" * 70)

    cat_to_idx = {name: idx for idx, name in enumerate(sorted(_STANDARD_19))}
    label_indices = [cat_to_idx[l] for l in labels]
    indices = list(range(len(samples)))

    # Current approach: random stratified split
    seed = 42
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, random_state=seed, stratify=label_indices,
    )

    train_groups = set(groups[i] for i in train_idx)
    val_groups = set(groups[i] for i in val_idx)
    leaked = train_groups & val_groups
    val_leaked_count = sum(1 for i in val_idx if groups[i] in leaked)

    print(f"\nCurrent approach (random stratified 80/20, seed={seed}):")
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")
    print(f"  Scene groups in train: {len(train_groups)}")
    print(f"  Scene groups in val:   {len(val_groups)}")
    print(f"  Scene groups in BOTH:  {len(leaked)}  *** LEAKAGE ***")
    print(f"  Val samples from leaked groups: {val_leaked_count}/{len(val_idx)} "
          f"({val_leaked_count/len(val_idx)*100:.1f}%)")

    # Class distribution in val (current)
    val_class_counts = Counter(label_indices[i] for i in val_idx)
    idx_to_cat = {v: k for k, v in cat_to_idx.items()}
    print(f"\n  Val class distribution (current random split):")
    for i in range(19):
        name = idx_to_cat[i]
        n = val_class_counts.get(i, 0)
        print(f"    {name:<22} {n:>4}")

    # StratifiedGroupKFold approach
    label_arr = np.array(label_indices)
    group_arr = np.array(groups)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_idx_g, val_idx_g = next(sgkf.split(indices, label_arr, group_arr))

    train_groups_g = set(groups[i] for i in train_idx_g)
    val_groups_g = set(groups[i] for i in val_idx_g)
    leaked_g = train_groups_g & val_groups_g

    print(f"\nStratifiedGroupKFold (5-fold, fold 0, seed={seed}):")
    print(f"  Train: {len(train_idx_g)}, Val: {len(val_idx_g)}")
    print(f"  Scene groups in train: {len(train_groups_g)}")
    print(f"  Scene groups in val:   {len(val_groups_g)}")
    print(f"  Scene groups in BOTH:  {len(leaked_g)}  "
          f"{'*** LEAKAGE ***' if leaked_g else '✓ NO LEAKAGE'}")

    val_class_counts_g = Counter(label_arr[i] for i in val_idx_g)
    print(f"\n  Val class distribution (StratifiedGroupKFold):")
    for i in range(19):
        name = idx_to_cat[i]
        n = val_class_counts_g.get(i, 0)
        pct = n / len(val_idx_g) * 100 if len(val_idx_g) > 0 else 0
        print(f"    {name:<22} {n:>4}  ({pct:.1f}%)")

    missing = [idx_to_cat[i] for i in range(19) if val_class_counts_g.get(i, 0) == 0]
    if missing:
        print(f"\n  WARNING: Classes missing from val: {missing}")
    else:
        print(f"\n  All 19 classes represented in val ✓")


if __name__ == "__main__":
    main()
