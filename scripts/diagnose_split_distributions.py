"""Compare sensor / class / scene composition across HPO train, HPO val, and the
official test split for SUN RGB-D 19.

Reconstructs the same StratifiedGroupKFold split that the HPO uses (SEED=152)
so we can ask: is the HPO val distribution actually representative of the
official test distribution?

Outputs:
  - Console tables for sensor mix, class mix, class x sensor matrix
  - Scene-overlap stats (hpo_val <-> test, hpo_train <-> test, hpo_val <-> hpo_train)
  - Raw pixel statistics (mean RGB/depth) per split
  - CSV summary at reports/diagnostics/split_distributions.csv

Usage:
    python3 scripts/diagnose_split_distributions.py
    python3 scripts/diagnose_split_distributions.py --data-root data/sunrgbd_19_traintest --seed 152
"""

import argparse
import csv
import json
import os
from collections import Counter, defaultdict

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold


SENSORS = ("kv1", "kv2", "realsense", "xtion")


def load_split_meta(split_dir):
    """Return (labels, scene_groups, sample_paths) for a preprocessed split."""
    with open(os.path.join(split_dir, "labels.txt")) as f:
        labels = [int(x) for x in f.read().strip().splitlines()]
    with open(os.path.join(split_dir, "scene_groups.json")) as f:
        scene_groups = json.load(f)
    with open(os.path.join(split_dir, "sample_paths.json")) as f:
        sample_paths = json.load(f)
    if not (len(labels) == len(scene_groups) == len(sample_paths)):
        raise RuntimeError(
            f"Length mismatch in {split_dir}: labels={len(labels)}, "
            f"scene_groups={len(scene_groups)}, sample_paths={len(sample_paths)}"
        )
    return labels, scene_groups, sample_paths


def sensor_of(path):
    """First path component is the sensor prefix (kv1/kv2/realsense/xtion)."""
    head = path.split("/", 1)[0]
    return head if head in SENSORS else "other"


def class_names(data_root):
    with open(os.path.join(data_root, "class_names.txt")) as f:
        return [line.strip() for line in f if line.strip()]


def reconstruct_hpo_split(labels, scene_groups, seed, n_splits=5):
    """Reproduce cell 18 of colab_LiNet3_SUN_hype_tune_MD.ipynb."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, val_idx = next(sgkf.split(range(len(labels)), labels, scene_groups))
    return sorted(train_idx.tolist()), sorted(val_idx.tolist())


def distribution(values):
    """Counter with percentages."""
    c = Counter(values)
    total = sum(c.values())
    return c, {k: 100.0 * v / total for k, v in c.items()}


def print_sensor_table(label, items_by_split):
    """items_by_split: dict split -> list of sensors (one per sample)."""
    print(f"\n=== {label}: sensor distribution ===")
    header = f"  {'sensor':<12}" + "".join(f"{s:>18}" for s in items_by_split)
    print(header)
    all_keys = sorted({k for lst in items_by_split.values() for k in lst})
    totals = {split: len(lst) for split, lst in items_by_split.items()}
    for k in all_keys:
        row = f"  {k:<12}"
        for split, lst in items_by_split.items():
            n = lst.count(k)
            pct = 100.0 * n / totals[split] if totals[split] else 0.0
            row += f"{n:>8d} ({pct:5.1f}%)"
        print(row)
    row = f"  {'TOTAL':<12}"
    for split, lst in items_by_split.items():
        row += f"{len(lst):>8d} ({100.0:>5.1f}%)"
    print(row)


def print_class_table(label, class_lists_by_split, names):
    print(f"\n=== {label}: per-class counts and % ===")
    splits = list(class_lists_by_split.keys())
    header = f"  {'class':<18}" + "".join(f"{s:>18}" for s in splits)
    print(header)
    totals = {s: len(class_lists_by_split[s]) for s in splits}
    for c_idx, c_name in enumerate(names):
        row = f"  {c_name:<18}"
        for s in splits:
            n = class_lists_by_split[s].count(c_idx)
            pct = 100.0 * n / totals[s] if totals[s] else 0.0
            row += f"{n:>8d} ({pct:5.1f}%)"
        print(row)


def print_class_sensor_matrix(label, labels_seq, sensors_seq, names):
    print(f"\n=== {label}: class x sensor matrix (counts) ===")
    mat = defaultdict(lambda: Counter())
    for y, s in zip(labels_seq, sensors_seq):
        mat[y][s] += 1
    cols = SENSORS + ("other",)
    header = f"  {'class':<18}" + "".join(f"{c:>10}" for c in cols) + f"{'total':>10}"
    print(header)
    for c_idx, c_name in enumerate(names):
        row = f"  {c_name:<18}"
        total = 0
        for s in cols:
            n = mat[c_idx][s]
            row += f"{n:>10d}"
            total += n
        row += f"{total:>10d}"
        print(row)


def scene_overlap(scenes_a, scenes_b, label_a, label_b):
    sa, sb = set(scenes_a), set(scenes_b)
    inter = sa & sb
    samples_in_a_leaked = sum(1 for s in scenes_a if s in inter)
    samples_in_b_leaked = sum(1 for s in scenes_b if s in inter)
    print(f"\n=== Scene overlap: {label_a} <-> {label_b} ===")
    print(f"  unique scenes in {label_a}: {len(sa)}")
    print(f"  unique scenes in {label_b}: {len(sb)}")
    print(f"  shared scenes:            {len(inter)}")
    print(f"  samples in {label_a} whose scene appears in {label_b}: "
          f"{samples_in_a_leaked}/{len(scenes_a)} ({100*samples_in_a_leaked/max(1,len(scenes_a)):.1f}%)")
    print(f"  samples in {label_b} whose scene appears in {label_a}: "
          f"{samples_in_b_leaked}/{len(scenes_b)} ({100*samples_in_b_leaked/max(1,len(scenes_b)):.1f}%)")
    return {
        "shared_scenes": len(inter),
        f"{label_a}_samples_leaked": samples_in_a_leaked,
        f"{label_b}_samples_leaked": samples_in_b_leaked,
        "examples": sorted(inter)[:10],
    }


def pixel_stats(rgb_tensor, depth_tensor, indices):
    """Compute per-channel mean/std for the given indices. Works on mmap tensors.

    RGB tensor is uint8 [N,3,256,256] (mean/std reported on [0,255] scale).
    Depth tensor is uint16/int16 [N,1,256,256] (mm).
    Stats computed channelwise over all pixels.
    """
    idx = torch.as_tensor(indices, dtype=torch.long)
    rgb = rgb_tensor.index_select(0, idx).float()
    depth = depth_tensor.index_select(0, idx).float()
    rgb_mean = rgb.mean(dim=(0, 2, 3)).tolist()
    rgb_std = rgb.std(dim=(0, 2, 3)).tolist()
    depth_mean = depth.mean().item()
    depth_std = depth.std().item()
    return {
        "rgb_mean_0_255": rgb_mean,
        "rgb_std_0_255": rgb_std,
        "depth_mean_mm": depth_mean,
        "depth_std_mm": depth_std,
        "n_samples": len(indices),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data/sunrgbd_19_traintest")
    p.add_argument("--seed", type=int, default=152,
                   help="StratifiedGroupKFold seed used in HPO (cell 18)")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--out-dir", default="reports/diagnostics")
    p.add_argument("--pixel-stats", action="store_true",
                   help="Also compute raw pixel mean/std per split (loads tensors)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    names = class_names(args.data_root)
    print(f"Loaded {len(names)} class names from {args.data_root}")

    train_dir = os.path.join(args.data_root, "train")
    test_dir = os.path.join(args.data_root, "test")
    train_labels, train_scenes, train_paths = load_split_meta(train_dir)
    test_labels, test_scenes, test_paths = load_split_meta(test_dir)
    print(f"Train pool: {len(train_labels)} samples, {len(set(train_scenes))} unique scenes")
    print(f"Test set:   {len(test_labels)} samples, {len(set(test_scenes))} unique scenes")

    # Reproduce HPO split
    hpo_train_idx, hpo_val_idx = reconstruct_hpo_split(
        train_labels, train_scenes, seed=args.seed, n_splits=args.n_splits
    )
    print(f"\nReconstructed HPO split (seed={args.seed}, n_splits={args.n_splits}):")
    print(f"  HPO train: {len(hpo_train_idx)} samples")
    print(f"  HPO val:   {len(hpo_val_idx)} samples")

    # Per-sample sensor and scene for each logical split
    train_sensors = [sensor_of(p) for p in train_paths]
    test_sensors = [sensor_of(p) for p in test_paths]

    hpo_train_sensors = [train_sensors[i] for i in hpo_train_idx]
    hpo_val_sensors = [train_sensors[i] for i in hpo_val_idx]
    hpo_train_labels = [train_labels[i] for i in hpo_train_idx]
    hpo_val_labels = [train_labels[i] for i in hpo_val_idx]
    hpo_train_scenes = [train_scenes[i] for i in hpo_train_idx]
    hpo_val_scenes = [train_scenes[i] for i in hpo_val_idx]

    # 1. Sensor distribution across hpo_train / hpo_val / test
    print_sensor_table("All splits", {
        "hpo_train": hpo_train_sensors,
        "hpo_val": hpo_val_sensors,
        "official_test": test_sensors,
    })

    # 2. Class distribution
    print_class_table("All splits", {
        "hpo_train": hpo_train_labels,
        "hpo_val": hpo_val_labels,
        "official_test": test_labels,
    }, names)

    # 3. Class x sensor matrix for each split
    print_class_sensor_matrix("HPO val", hpo_val_labels, hpo_val_sensors, names)
    print_class_sensor_matrix("Official test", test_labels, test_sensors, names)

    # 4. Scene overlap analyses
    overlap_results = {}
    overlap_results["hpo_val_vs_hpo_train"] = scene_overlap(
        hpo_val_scenes, hpo_train_scenes, "hpo_val", "hpo_train"
    )
    overlap_results["hpo_val_vs_test"] = scene_overlap(
        hpo_val_scenes, test_scenes, "hpo_val", "official_test"
    )
    overlap_results["hpo_train_vs_test"] = scene_overlap(
        hpo_train_scenes, test_scenes, "hpo_train", "official_test"
    )

    # 5. Optional: pixel stats
    pixel_results = {}
    if args.pixel_stats:
        print("\nLoading tensors for pixel statistics (mmap)...")
        train_rgb = torch.load(os.path.join(train_dir, "rgb_tensors.pt"),
                               weights_only=True, mmap=True)
        train_depth = torch.load(os.path.join(train_dir, "depth_tensors.pt"),
                                 weights_only=True, mmap=True)
        test_rgb = torch.load(os.path.join(test_dir, "rgb_tensors.pt"),
                              weights_only=True, mmap=True)
        test_depth = torch.load(os.path.join(test_dir, "depth_tensors.pt"),
                                weights_only=True, mmap=True)

        print("Computing hpo_train pixel stats...")
        pixel_results["hpo_train"] = pixel_stats(train_rgb, train_depth, hpo_train_idx)
        print("Computing hpo_val pixel stats...")
        pixel_results["hpo_val"] = pixel_stats(train_rgb, train_depth, hpo_val_idx)
        print("Computing official_test pixel stats...")
        pixel_results["official_test"] = pixel_stats(
            test_rgb, test_depth, list(range(len(test_labels)))
        )

        print("\n=== Raw pixel statistics (channel-wise) ===")
        for split, stats in pixel_results.items():
            print(f"  {split}:")
            print(f"    rgb mean [R,G,B]: {stats['rgb_mean_0_255']}")
            print(f"    rgb std  [R,G,B]: {stats['rgb_std_0_255']}")
            print(f"    depth mean (mm):  {stats['depth_mean_mm']:.2f}")
            print(f"    depth std  (mm):  {stats['depth_std_mm']:.2f}")

    # 6. Save CSV summary
    csv_path = os.path.join(args.out_dir, "split_distributions.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "n_samples", "n_unique_scenes"] + list(SENSORS) + ["other"]
                   + [f"class_{n}" for n in names])
        for split_name, sensors_seq, labels_seq, scenes_seq in (
            ("hpo_train", hpo_train_sensors, hpo_train_labels, hpo_train_scenes),
            ("hpo_val", hpo_val_sensors, hpo_val_labels, hpo_val_scenes),
            ("official_test", test_sensors, test_labels, test_scenes),
        ):
            row = [split_name, len(labels_seq), len(set(scenes_seq))]
            row += [sensors_seq.count(s) for s in SENSORS]
            row.append(sensors_seq.count("other"))
            row += [labels_seq.count(i) for i in range(len(names))]
            w.writerow(row)

    # 7. Save full results as JSON for later plotting
    json_path = os.path.join(args.out_dir, "split_distributions.json")
    payload = {
        "seed": args.seed,
        "n_splits": args.n_splits,
        "sizes": {
            "hpo_train": len(hpo_train_idx),
            "hpo_val": len(hpo_val_idx),
            "official_test": len(test_labels),
        },
        "unique_scenes": {
            "hpo_train": len(set(hpo_train_scenes)),
            "hpo_val": len(set(hpo_val_scenes)),
            "official_test": len(set(test_scenes)),
        },
        "sensor_counts": {
            "hpo_train": dict(Counter(hpo_train_sensors)),
            "hpo_val": dict(Counter(hpo_val_sensors)),
            "official_test": dict(Counter(test_sensors)),
        },
        "class_counts": {
            "hpo_train": dict(Counter(hpo_train_labels)),
            "hpo_val": dict(Counter(hpo_val_labels)),
            "official_test": dict(Counter(test_labels)),
        },
        "scene_overlap": overlap_results,
        "pixel_stats": pixel_results,
        "class_names": names,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    print(f"\nWrote {csv_path}")
    print(f"Wrote {json_path}")
    print("Done.")


if __name__ == "__main__":
    main()
