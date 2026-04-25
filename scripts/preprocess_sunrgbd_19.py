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
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF
from tqdm import tqdm

# Make src/ importable when running from the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.data_utils.hha import compute_hha, read_extrinsics, read_intrinsics

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


def extract_scene_group(rel_path):
    """Extract scene group ID from a sample's relative path within SUNRGBD_BASE.

    Groups images from the same physical location so downstream code can do
    group-aware train/val splitting (preventing data leakage).

    - xtion/sun3ddata/SCENE/...: group = xtion/sun3ddata/SCENE
    - kv2/kinect2data/SEQNUM_SESSION_rgbfFRAME-resize: group by session
    - kv1/*, realsense/*: each sample is its own group (1:1)
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


def _read_depth_native(path: str) -> np.ndarray:
    """Read and unpack a SUN RGB-D depth PNG at native resolution.

    Returns ``[H, W]`` uint16 with values in millimeters; 0 marks missing.
    The 3-bit shift unpacking matches ``SUNRGBDtoolbox/read3dPoints.m``.
    """
    d = Image.open(path)
    arr = np.array(d, dtype=np.uint16)
    arr = np.bitwise_or(arr >> 3, arr << 13).astype(np.uint16)
    return arr


def _resize_depth_uint16(depth_native_mm: np.ndarray) -> torch.Tensor:
    """Nearest-neighbor resize a native uint16 depth array (mm) to TARGET_SIZE.

    Returns ``[1, H, W]`` torch.uint16 tensor.
    """
    d = Image.fromarray(depth_native_mm, mode="I;16")
    d = TF.resize(d, TARGET_SIZE, interpolation=TF.InterpolationMode.NEAREST)
    arr = np.array(d, dtype=np.uint16)
    return torch.from_numpy(arr).unsqueeze(0)


def _read_depth(path: str) -> torch.Tensor:
    """Backward-compat wrapper: native read + resize to TARGET_SIZE uint16."""
    return _resize_depth_uint16(_read_depth_native(path))


def _resize_hha_to_target(hha_native: np.ndarray) -> np.ndarray:
    """Nearest-neighbor resize a per-channel HHA float array to TARGET_SIZE.

    Preserves NaN (no interpolation across discontinuities).
    Input ``[3, H, W]`` float32; output ``[3, T_h, T_w]`` float32.
    """
    # F.interpolate doesn't support NaN well, but nearest mode is index-only,
    # so NaN is preserved exactly.
    t = torch.from_numpy(hha_native).unsqueeze(0)  # [1, 3, H, W]
    out = F.interpolate(t, size=TARGET_SIZE, mode="nearest")
    return out.squeeze(0).numpy()


def _sensor_from_relpath(rel_path: str) -> str:
    """Top-level sensor folder from a path relative to SUNRGBD_BASE."""
    parts = rel_path.split("/")
    return parts[0] if parts else "unknown"


def build_split(
    split_name: str,
    split_samples: list,
    output_base: str,
    *,
    hha_mode: str = "none",
    hha_dtype: str = "float16",
    failure_log: list = None,
):
    """Build one split's tensor files. ``hha_mode`` is one of:
        none      — only depth_tensors.pt (existing behavior)
        with-hha  — both depth_tensors.pt AND hha_tensors.pt
        hha-only  — only hha_tensors.pt (no depth_tensors.pt)
    """
    split_dir = os.path.join(output_base, split_name)
    os.makedirs(split_dir, exist_ok=True)

    N = len(split_samples)
    H, W = TARGET_SIZE
    write_depth = hha_mode in ("none", "with-hha")
    write_hha = hha_mode in ("with-hha", "hha-only")
    print(f"\nBuilding {split_name} ({N} samples), hha_mode={hha_mode} ...")

    rgb_t = torch.empty(N, 3, H, W, dtype=torch.uint8)
    depth_t = (
        torch.empty(N, 1, H, W, dtype=torch.uint16) if write_depth else None
    )
    hha_torch_dtype = (
        torch.float16 if hha_dtype == "float16" else torch.float32
    )
    hha_t = (
        torch.empty(N, 3, H, W, dtype=hha_torch_dtype) if write_hha else None
    )
    labels = []
    sensors = []

    # Cache intrinsics + extrinsics by scene_dir (many frames share a scene).
    intrinsics_cache: dict = {}
    extrinsics_cache: dict = {}

    def _get_K(scene_dir: str):
        if scene_dir not in intrinsics_cache:
            intrinsics_cache[scene_dir] = read_intrinsics(scene_dir)
        return intrinsics_cache[scene_dir]

    def _get_R(scene_dir: str):
        if scene_dir not in extrinsics_cache:
            extrinsics_cache[scene_dir] = read_extrinsics(scene_dir)
        return extrinsics_cache[scene_dir]

    for i, (sd, label, cls, rgb_path, depth_path) in enumerate(tqdm(split_samples)):
        rgb_t[i] = _read_rgb(rgb_path)
        depth_native_mm = _read_depth_native(depth_path)  # [H_n, W_n] uint16 mm

        if write_depth:
            depth_t[i] = _resize_depth_uint16(depth_native_mm)

        if write_hha:
            try:
                K = _get_K(sd)
                R = _get_R(sd)
                depth_m = depth_native_mm.astype(np.float32) / 1000.0
                hha_native = compute_hha(depth_m, K, R)  # [3, H_n, W_n] f32 with NaN
                hha_resized = _resize_hha_to_target(hha_native)  # [3, H, W] f32 with NaN
                hha_t[i] = torch.from_numpy(hha_resized).to(hha_torch_dtype)
                if np.isnan(hha_resized).all():
                    if failure_log is not None:
                        failure_log.append(os.path.relpath(sd, SUNRGBD_BASE))
            except Exception as exc:  # noqa: BLE001
                # compute_hha already returns all-NaN on internal failure, so
                # this branch only fires if intrinsics/extrinsics IO fails.
                logging.warning("HHA failed for %s: %s", sd, exc)
                hha_t[i] = torch.full(
                    (3, H, W), float("nan"), dtype=hha_torch_dtype
                )
                if failure_log is not None:
                    failure_log.append(os.path.relpath(sd, SUNRGBD_BASE))

        labels.append(cls)
        sensors.append(_sensor_from_relpath(os.path.relpath(sd, SUNRGBD_BASE)))

    rgb_out = os.path.join(split_dir, "rgb_tensors.pt")
    torch.save(rgb_t, rgb_out)

    if write_depth:
        depth_out = os.path.join(split_dir, "depth_tensors.pt")
        torch.save(depth_t, depth_out)

    if write_hha:
        hha_out = os.path.join(split_dir, "hha_tensors.pt")
        torch.save(hha_t, hha_out)

    with open(os.path.join(split_dir, "labels.txt"), "w") as f:
        f.write("\n".join(map(str, labels)) + "\n")

    # Save scene group IDs and sample paths for group-aware splitting
    sample_paths = [os.path.relpath(s[0], SUNRGBD_BASE) for s in split_samples]
    scene_groups = [extract_scene_group(p) for p in sample_paths]

    with open(os.path.join(split_dir, "scene_groups.json"), "w") as f:
        json.dump(scene_groups, f)
    with open(os.path.join(split_dir, "sample_paths.json"), "w") as f:
        json.dump(sample_paths, f)
    with open(os.path.join(split_dir, "sensors.json"), "w") as f:
        json.dump(sensors, f)

    print(f"  rgb_tensors.pt:   {tuple(rgb_t.shape)}  "
          f"{os.path.getsize(rgb_out)/1e6:.1f} MB")
    if write_depth:
        print(f"  depth_tensors.pt: {tuple(depth_t.shape)}  "
              f"{os.path.getsize(depth_out)/1e6:.1f} MB")
    if write_hha:
        print(f"  hha_tensors.pt:   {tuple(hha_t.shape)} {hha_dtype}  "
              f"{os.path.getsize(hha_out)/1e6:.1f} MB")
    print(f"  scene_groups.json: {len(set(scene_groups))} unique groups")
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


def run_three_way(
    samples, train_paths, test_paths, output_base,
    val_ratio=0.2, seed=42,
    *, hha_mode: str = "none", hha_dtype: str = "float16",
):
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

    failures: list = []
    kw = dict(hha_mode=hha_mode, hha_dtype=hha_dtype, failure_log=failures)
    trl = build_split("train", train_s, output_base, **kw)
    vl  = build_split("val",   val_s,   output_base, **kw)
    tel = build_split("test",  test_s,  output_base, **kw)
    _compute_and_save_norm_stats(output_base, hha_mode=hha_mode)
    n_total = len(train_s) + len(val_s) + len(test_s)
    _check_failure_rate(failures, n_total, hha_mode)
    _write_meta(
        output_base, len(train_s), len(val_s), len(test_s),
        trl, vl, tel, val_ratio, seed,
        hha_mode=hha_mode, hha_dtype=hha_dtype,
        failure_count=len(failures),
    )


def run_two_way(
    samples, train_paths, test_paths, output_base,
    *, hha_mode: str = "none", hha_dtype: str = "float16",
):
    official_train, official_test = _match_official(samples, train_paths, test_paths)

    print(f"\n2-way split (no val sub-split, for k-fold CV):")
    print(f"  Train: {len(official_train)}  (all official train)")
    print(f"  Test:  {len(official_test)}  (official test)")

    failures: list = []
    kw = dict(hha_mode=hha_mode, hha_dtype=hha_dtype, failure_log=failures)
    trl = build_split("train", official_train, output_base, **kw)
    tel = build_split("test",  official_test,  output_base, **kw)
    _compute_and_save_norm_stats(output_base, hha_mode=hha_mode)
    n_total = len(official_train) + len(official_test)
    _check_failure_rate(failures, n_total, hha_mode)
    _write_meta(
        output_base, len(official_train), 0, len(official_test),
        trl, [], tel, None, None,
        hha_mode=hha_mode, hha_dtype=hha_dtype,
        failure_count=len(failures),
    )


def _check_failure_rate(failures: list, n_total: int, hha_mode: str):
    if hha_mode == "none" or n_total == 0:
        return
    rate = len(failures) / n_total
    print(f"\nHHA computation: {len(failures)}/{n_total} failures ({rate*100:.2f}%).")
    if rate > 0.005:
        print("  ERROR: failure rate exceeds 0.5% — investigate before training.")
        for f in failures[:20]:
            print(f"    {f}")
        if len(failures) > 20:
            print(f"    ... and {len(failures) - 20} more")
        raise RuntimeError(f"HHA failure rate {rate*100:.2f}% > 0.5%")
    elif failures:
        print("  WARNING: some samples failed HHA computation; the corresponding "
              "rows are all-NaN in hha_tensors.pt and will be replaced with "
              "channel mean by the dataset.")


# ---------------------------------------------------------------------------
# Normalization statistics
# ---------------------------------------------------------------------------

def _compute_and_save_norm_stats(
    output_base: str, *, hha_mode: str = "none", batch_size: int = 256
):
    """Compute channel-wise mean/std from TRAINING split and save to norm_stats.json.

    Streams through the tensor files in chunks of ``batch_size`` samples to
    keep peak memory bounded. Quantiles are estimated from a reservoir
    sample of up to 5M valid pixels per channel.
    """
    train_dir = os.path.join(output_base, "train")

    # ---- RGB: chunked Welford on (uint8 / 255.0). ----
    rgb_t = torch.load(
        os.path.join(train_dir, "rgb_tensors.pt"), weights_only=True, mmap=True
    )
    N = rgb_t.shape[0]
    sum_c = torch.zeros(3, dtype=torch.float64)
    sumsq_c = torch.zeros(3, dtype=torch.float64)
    count = 0
    for i in range(0, N, batch_size):
        chunk = rgb_t[i:i + batch_size].to(torch.float32) / 255.0  # [B, 3, H, W]
        sum_c += chunk.sum(dim=(0, 2, 3)).double()
        sumsq_c += (chunk.double() ** 2).sum(dim=(0, 2, 3))
        count += chunk.shape[0] * chunk.shape[2] * chunk.shape[3]
    rgb_mean_t = sum_c / count
    # Unbiased variance (Bessel-corrected) to match torch.Tensor.std() default.
    rgb_var_t = (sumsq_c - count * rgb_mean_t ** 2) / max(1, count - 1)
    rgb_mean = rgb_mean_t.tolist()
    rgb_std = rgb_var_t.clamp_min(0).sqrt().tolist()
    del rgb_t

    stats = {
        "rgb_mean": rgb_mean,
        "rgb_std": rgb_std,
    }

    # ---- Raw depth (1-channel): chunked, exclude zero pixels. ----
    if hha_mode in ("none", "with-hha"):
        depth_t = torch.load(
            os.path.join(train_dir, "depth_tensors.pt"), weights_only=True, mmap=True
        )
        s = 0.0
        sq = 0.0
        n = 0
        for i in range(0, depth_t.shape[0], batch_size):
            chunk = depth_t[i:i + batch_size]  # uint16 [B, 1, H, W]
            valid_mask = chunk.to(torch.int32) > 0
            valid = (chunk.float() / 1000.0)[valid_mask]
            s += valid.double().sum().item()
            sq += (valid.double() ** 2).sum().item()
            n += valid.numel()
        depth_mean_v = s / n
        # Unbiased variance (Bessel-corrected) to match torch.Tensor.std().
        depth_var_v = max(0.0, (sq - n * depth_mean_v ** 2) / max(1, n - 1))
        stats["depth_mean"] = [depth_mean_v]
        stats["depth_std"] = [depth_var_v ** 0.5]
        del depth_t

    # ---- HHA (3-channel float16): chunked Welford on non-NaN pixels +
    # reservoir-sampled quantiles per channel.
    if hha_mode in ("with-hha", "hha-only"):
        hha_t = torch.load(
            os.path.join(train_dir, "hha_tensors.pt"), weights_only=True, mmap=True
        )  # [N, 3, H, W] float16
        n_total = hha_t.shape[0] * hha_t.shape[2] * hha_t.shape[3]  # per-channel
        c_sum = [0.0, 0.0, 0.0]
        c_sumsq = [0.0, 0.0, 0.0]
        c_n_finite = [0, 0, 0]
        c_min = [float("inf")] * 3
        c_max = [float("-inf")] * 3
        # Reservoir sample: up to RES_CAP per channel for quantile estimation.
        RES_CAP = 5_000_000
        c_reservoir = [
            torch.empty(0, dtype=torch.float32) for _ in range(3)
        ]
        rng = torch.Generator().manual_seed(152)
        for i in range(0, hha_t.shape[0], batch_size):
            chunk = hha_t[i:i + batch_size].to(torch.float32)  # [B, 3, H, W]
            for c in range(3):
                ch = chunk[:, c]
                finite_mask = torch.isfinite(ch)
                vals = ch[finite_mask]
                if vals.numel() == 0:
                    continue
                c_sum[c] += vals.double().sum().item()
                c_sumsq[c] += (vals.double() ** 2).sum().item()
                c_n_finite[c] += int(vals.numel())
                c_min[c] = min(c_min[c], float(vals.min().item()))
                c_max[c] = max(c_max[c], float(vals.max().item()))
                # Reservoir append + uniform downsample if over cap.
                c_reservoir[c] = torch.cat([c_reservoir[c], vals], dim=0)
                if c_reservoir[c].numel() > RES_CAP:
                    perm = torch.randperm(c_reservoir[c].numel(), generator=rng)[:RES_CAP]
                    c_reservoir[c] = c_reservoir[c][perm]
        hha_mean: list = []
        hha_std: list = []
        ranges = []
        for c in range(3):
            if c_n_finite[c] == 0:
                raise RuntimeError(f"HHA channel {c} has no finite values across train split")
            mu = c_sum[c] / c_n_finite[c]
            # Unbiased variance (Bessel-corrected) to match torch.Tensor.std().
            var = max(0.0, (c_sumsq[c] - c_n_finite[c] * mu ** 2) / max(1, c_n_finite[c] - 1))
            hha_mean.append(float(mu))
            hha_std.append(float(var ** 0.5))
            res = c_reservoir[c]
            ranges.append({
                "min": c_min[c],
                "max": c_max[c],
                "p1": float(torch.quantile(res, 0.01).item()),
                "p99": float(torch.quantile(res, 0.99).item()),
                "p99_9": float(torch.quantile(res, 0.999).item()),
                "nan_rate": float(1.0 - c_n_finite[c] / n_total),
            })
        stats["hha_mean"] = hha_mean
        stats["hha_std"] = hha_std
        stats["hha_ranges"] = ranges  # diagnostic; dataset doesn't read this
        del hha_t, c_reservoir

        # fp16-suitability range check.
        labels_ = ["disparity", "height_m", "angle_deg"]
        bad = False
        for c, (lab, r) in enumerate(zip(labels_, ranges)):
            extreme = (
                abs(r["max"]) > 1000
                or abs(r["min"]) > 1000
                or (lab == "height_m" and abs(r["p99_9"]) > 100)
                or (lab == "disparity" and r["p99_9"] > 100)
            )
            if extreme:
                print(
                    f"  WARNING: HHA channel {c} ({lab}) has extreme values "
                    f"(min={r['min']:.3f}, max={r['max']:.3f}, p99.9={r['p99_9']:.3f}). "
                    f"fp16 mantissa precision may be insufficient — re-run "
                    f"preprocessing with --hha-dtype float32 if loss/grads NaN."
                )
                bad = True
        if not bad:
            for c, (lab, r) in enumerate(zip(labels_, ranges)):
                print(
                    f"  HHA channel {c} ({lab}): "
                    f"min={r['min']:.3f}, max={r['max']:.3f}, "
                    f"p1={r['p1']:.3f}, p99={r['p99']:.3f}, "
                    f"p99.9={r['p99_9']:.3f}, nan={r['nan_rate']*100:.2f}%"
                )

    out_path = os.path.join(output_base, "norm_stats.json")
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nNormalization stats (from training split, hha_mode={hha_mode}):")
    print(f"  RGB mean:   {rgb_mean}")
    print(f"  RGB std:    {rgb_std}")
    if "depth_mean" in stats:
        print(f"  Depth mean: {stats['depth_mean']}")
        print(f"  Depth std:  {stats['depth_std']}")
    if "hha_mean" in stats:
        print(f"  HHA mean:   {stats['hha_mean']}")
        print(f"  HHA std:    {stats['hha_std']}")
    print(f"  Saved to: {out_path}")

    return stats


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def _per_sensor_hha_stats(output_base: str) -> str:
    """Return a formatted block of per-sensor HHA diagnostics for the train split.

    Streams the HHA tensor in chunks (mmap, per-sample float32 cast) so peak
    memory stays bounded even on the full ~5k-sample SUN RGB-D train split.
    """
    train_dir = os.path.join(output_base, "train")
    hha_path = os.path.join(train_dir, "hha_tensors.pt")
    sensors_path = os.path.join(train_dir, "sensors.json")
    if not (os.path.exists(hha_path) and os.path.exists(sensors_path)):
        return ""

    with open(sensors_path, "r") as f:
        sensors = json.load(f)
    hha = torch.load(hha_path, weights_only=True, mmap=True)  # [N,3,H,W] float16

    by_sensor: dict = defaultdict(list)
    for i, s in enumerate(sensors):
        by_sensor[s].append(i)

    H, W = hha.shape[2], hha.shape[3]
    out_lines = ["", "Per-sensor HHA diagnostics (train split):",
                 f"{'sensor':<12} {'count':>6}  "
                 f"{'disp_mean':>10} {'disp_std':>9} {'disp_nan%':>9}  "
                 f"{'hgt_mean':>9} {'hgt_std':>8} {'hgt_nan%':>9}  "
                 f"{'ang_mean':>9} {'ang_std':>8} {'ang_nan%':>9}"]
    for sensor, idxs in sorted(by_sensor.items()):
        sums = [0.0, 0.0, 0.0]
        sumsqs = [0.0, 0.0, 0.0]
        n_finite = [0, 0, 0]
        n_total_per_chan = len(idxs) * H * W
        # Process in batches of 64 samples for memory safety.
        for start in range(0, len(idxs), 64):
            chunk_idxs = idxs[start:start + 64]
            chunk = hha[chunk_idxs].to(torch.float32)  # [b, 3, H, W]
            for c in range(3):
                ch = chunk[:, c]
                finite_mask = torch.isfinite(ch)
                vals = ch[finite_mask]
                if vals.numel():
                    sums[c] += vals.double().sum().item()
                    sumsqs[c] += (vals.double() ** 2).sum().item()
                    n_finite[c] += int(vals.numel())
        cells = [f"{sensor:<12} {len(idxs):>6}"]
        for c in range(3):
            if n_finite[c]:
                mu = sums[c] / n_finite[c]
                # Unbiased variance (Bessel-corrected) to match torch.Tensor.std().
                var = max(0.0, (sumsqs[c] - n_finite[c] * mu ** 2) / max(1, n_finite[c] - 1))
                mean = float(mu)
                std = float(var ** 0.5)
            else:
                mean = float("nan")
                std = float("nan")
            nan_pct = 100.0 * (1.0 - n_finite[c] / n_total_per_chan)
            cells.append(f"  {mean:>9.3f} {std:>8.3f} {nan_pct:>8.2f}%")
        out_lines.append("".join(cells))
    return "\n".join(out_lines) + "\n"


def _write_meta(output_base, n_tr, n_val, n_te,
                tr_labels, val_labels, te_labels, val_ratio, seed,
                *, hha_mode: str = "none", hha_dtype: str = "float16",
                failure_count: int = 0):
    n_cats = len(SUNRGBD_CATEGORIES)

    with open(os.path.join(output_base, "class_names.txt"), "w") as f:
        for i, name in enumerate(SUNRGBD_CATEGORIES):
            f.write(f"{i}: {name}\n")

    sensor_block = _per_sensor_hha_stats(output_base) if hha_mode != "none" else ""

    with open(os.path.join(output_base, "dataset_info.txt"), "w") as f:
        f.write(f"SUN RGB-D {n_cats}-Category Standard Benchmark\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Categories: {n_cats} (all with >80 images)\n")
        f.write("Split source: SUNRGBDtoolbox allsplit.mat\n")
        f.write(f"Tensor size: {TARGET_SIZE}  (crop to 224x224 in dataloader)\n")
        f.write(f"HHA mode: {hha_mode} (dtype: {hha_dtype})\n")
        if hha_mode != "none":
            f.write(f"HHA computation failures: {failure_count}\n")
        f.write("\n")
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
        if sensor_block:
            f.write(sensor_block)

    print(f"\nDone. Output: {output_base}")
    print(f"class_names.txt lists the {n_cats} accepted categories.")
    if sensor_block:
        print(sensor_block)


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
        "--output-dir", "--output-base", dest="output_dir", type=str, default=None,
        help="Output directory (default: data/sunrgbd_N or data/sunrgbd_N_trainval). "
             "--output-base is an alias.",
    )
    parser.add_argument(
        "--min-samples", type=int, default=80,
        help="Minimum samples for a category to be included (default: 80)."
    )
    parser.add_argument(
        "--hha-mode", choices=("none", "with-hha", "hha-only"), default="none",
        help=("HHA computation mode. 'none' (default): only depth_tensors.pt. "
              "'with-hha': both depth and HHA. 'hha-only': only HHA "
              "(skip depth_tensors.pt)."),
    )
    parser.add_argument(
        "--hha-dtype", choices=("float16", "float32"), default="float16",
        help=("Storage dtype for hha_tensors.pt (default float16). Re-run "
              "with float32 if the range diagnostics warn about precision."),
    )
    args = parser.parse_args()

    # Pass 1: scan + count
    samples_raw, category_counts = collect_samples_pass1()

    # Determine final category list (always printed)
    accepted = build_category_list(category_counts, min_samples=args.min_samples)

    if args.dry_run:
        print("\n--dry-run: stopping here. No files written.")
        print(f"Accepted categories ({len(accepted)}): {accepted}")
        print(f"hha-mode: {args.hha_mode}  hha-dtype: {args.hha_dtype}")
        return

    # Pass 2: filter
    samples = filter_samples(samples_raw)
    print(f"\nSamples after category filter: {len(samples)}")

    train_paths, test_paths = load_official_split()

    n = len(accepted)
    if args.no_val_split:
        default_out = args.output_dir or f"data/sunrgbd_{n}_traintest"
        run_two_way(
            samples, train_paths, test_paths, default_out,
            hha_mode=args.hha_mode, hha_dtype=args.hha_dtype,
        )
    else:
        default_out = args.output_dir or f"data/sunrgbd_{n}"
        run_three_way(
            samples, train_paths, test_paths, default_out,
            hha_mode=args.hha_mode, hha_dtype=args.hha_dtype,
        )


if __name__ == "__main__":
    main()