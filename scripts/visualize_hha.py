"""Visualize HHA channels for verification.

Produces two diagnostic figures into ``<data_root>/_diagnostics/``:

  1. ``samples_grid.png`` — for each picked sample, a 5-column row:
        RGB | raw depth | disparity (1/m) | height (m) | angle (deg)
     Sample selection prioritises diversity across the four SUN RGB-D
     sensors and across scene types so you can spot per-sensor failures.

  2. ``histograms.png`` — per-channel histograms across (a sample of)
     the entire train split. The angle channel should look bimodal
     (peaks near 0 deg for horizontal surfaces and near 90 deg for
     vertical surfaces); height should have a sharp peak near 0 m
     (floor reference) and a broad mass between 1 and 3 m (walls,
     ceiling); disparity should be a unimodal hump.

Run from the repo root:

    python3 scripts/visualize_hha.py \
        --data-root data/sunrgbd_19_hha \
        --sunrgbd-base data/sunrgbd/SUNRGBD \
        --num-samples 6
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import torch
from PIL import Image

# Make src/ importable so we can reuse the depth-unpacking helper.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Reuse the depth read path from preprocessing (3-bit shift unpack + mm).
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "_pp", os.path.join(_REPO_ROOT, "scripts", "preprocess_sunrgbd_19.py")
)
_pp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pp)
_read_depth_native = _pp._read_depth_native


# ---------------------------------------------------------------------------
# Sample selection
# ---------------------------------------------------------------------------

def _pick_samples(
    sensors: list[str],
    labels: list[int],
    class_names: list[str],
    num_samples: int,
    seed: int = 152,
) -> list[int]:
    """Pick diverse samples: one per sensor first, then by varied scene type.

    Returns a list of train-split indices.
    """
    rng = np.random.default_rng(seed)
    picked: list[int] = []
    seen_sensors: set[str] = set()
    seen_scenes: set[int] = set()

    # Priority 1: at least one per sensor.
    sensor_to_idxs: dict[str, list[int]] = {}
    for i, s in enumerate(sensors):
        sensor_to_idxs.setdefault(s, []).append(i)
    for sensor in sorted(sensor_to_idxs):
        candidates = sensor_to_idxs[sensor]
        idx = int(rng.choice(candidates))
        picked.append(idx)
        seen_sensors.add(sensor)
        seen_scenes.add(labels[idx])
        if len(picked) >= num_samples:
            return picked[:num_samples]

    # Priority 2: fill remaining slots with new (sensor, scene) combos.
    all_pairs = [
        (i, sensors[i], labels[i])
        for i in range(len(sensors))
        if (sensors[i], labels[i]) not in {(sensors[p], labels[p]) for p in picked}
    ]
    rng.shuffle(all_pairs)
    for i, _s, _l in all_pairs:
        if labels[i] in seen_scenes:
            continue
        picked.append(i)
        seen_scenes.add(labels[i])
        if len(picked) >= num_samples:
            break

    # Fallback: top up with anything new.
    if len(picked) < num_samples:
        for i in range(len(sensors)):
            if i not in picked:
                picked.append(i)
                if len(picked) >= num_samples:
                    break

    return picked[:num_samples]


# ---------------------------------------------------------------------------
# Per-sample grid
# ---------------------------------------------------------------------------

def _load_raw_depth_m(sample_paths: list[str], idx: int, sunrgbd_base: str):
    """Return the original-resolution raw depth in meters, plus the resized
    256x256 version for visual alignment with the HHA tensor."""
    rel = sample_paths[idx]
    sample_dir = os.path.join(sunrgbd_base, rel)
    # Mirror find_rgb_depth from preprocess_sunrgbd_19.py.
    depth_path = None
    for sub in ["depth_bfx", "depth"]:
        d = os.path.join(sample_dir, sub)
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.endswith(".png"):
                    depth_path = os.path.join(d, f)
                    break
        if depth_path:
            break
    if depth_path is None:
        return None, None
    native_mm = _read_depth_native(depth_path)
    native_m = native_mm.astype(np.float32) / 1000.0
    # Resize to 256x256 (nearest) for direct visual alignment with HHA.
    pil = Image.fromarray(native_mm, mode="I;16")
    pil_256 = pil.resize((256, 256), Image.Resampling.NEAREST)
    resized_m = np.array(pil_256, dtype=np.uint16).astype(np.float32) / 1000.0
    return native_m, resized_m


def _imshow(ax, data, *, title, cmap, vmin=None, vmax=None, norm=None,
            mask_value=None, mask_color=(1.0, 0.0, 0.0, 0.8)):
    """imshow with optional masking of a sentinel value (rendered as red).

    Pass ``norm`` (e.g. ``TwoSlopeNorm``) to override vmin/vmax with a
    custom normalization (useful for diverging colormaps centered at a
    specific value).
    """
    arr = data.copy()
    if mask_value is not None:
        if np.isnan(mask_value):
            invalid = np.isnan(arr)
        else:
            invalid = (arr == mask_value) | np.isnan(arr)
    else:
        invalid = np.isnan(arr)
    if invalid.any():
        finite = arr[~invalid]
        fill = float(np.nanmin(finite)) if finite.size else 0.0
        arr = np.where(invalid, fill, arr)
    if norm is not None:
        im = ax.imshow(arr, cmap=cmap, norm=norm, interpolation="nearest")
    else:
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    if invalid.any():
        overlay = np.zeros((*arr.shape, 4))
        overlay[invalid] = mask_color
        ax.imshow(overlay, interpolation="nearest")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    return im


def render_samples_grid(
    data_root: str,
    sunrgbd_base: str,
    indices: list[int],
    out_path: str,
):
    train_dir = os.path.join(data_root, "train")
    rgb_t = torch.load(os.path.join(train_dir, "rgb_tensors.pt"),
                       weights_only=True, mmap=True)
    hha_t = torch.load(os.path.join(train_dir, "hha_tensors.pt"),
                       weights_only=True, mmap=True)
    with open(os.path.join(train_dir, "sample_paths.json")) as f:
        sample_paths = json.load(f)
    with open(os.path.join(train_dir, "sensors.json")) as f:
        sensors = json.load(f)
    with open(os.path.join(train_dir, "labels.txt")) as f:
        labels = [int(x) for x in f.read().split()]
    class_names = []
    with open(os.path.join(data_root, "class_names.txt")) as f:
        for line in f:
            class_names.append(line.strip().split(": ", 1)[1])

    n = len(indices)
    fig, axes = plt.subplots(n, 5, figsize=(15, 3.0 * n))
    if n == 1:
        axes = axes[None, :]

    for row, idx in enumerate(indices):
        rgb = rgb_t[idx].permute(1, 2, 0).numpy()  # [H, W, 3] uint8
        hha = hha_t[idx].to(torch.float32).numpy()  # [3, H, W] f32 with NaN

        # Raw depth (load from original SUN RGB-D dir).
        _, depth_resized = _load_raw_depth_m(sample_paths, idx, sunrgbd_base)
        if depth_resized is None:
            depth_for_show = np.zeros_like(hha[0])
            mask_val = 0.0
        else:
            depth_for_show = depth_resized.copy()
            depth_for_show[depth_for_show == 0] = np.nan
            mask_val = float("nan")

        # RGB
        ax = axes[row, 0]
        ax.imshow(rgb)
        sensor = sensors[idx] if idx < len(sensors) else "?"
        cls = class_names[labels[idx]] if labels[idx] < len(class_names) else "?"
        ax.set_title(f"#{idx}  {sensor} / {cls}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

        # Raw depth (m)
        d_finite = depth_for_show[~np.isnan(depth_for_show)] if depth_resized is not None else np.array([])
        d_vmax = float(np.percentile(d_finite, 99)) if d_finite.size else 8.0
        _imshow(axes[row, 1], depth_for_show,
                title=f"raw depth (m)\n[max p99={d_vmax:.2f}, red=missing]",
                cmap="viridis", vmin=0.0, vmax=d_vmax,
                mask_value=mask_val)

        # HHA channel 0: disparity (1/m), clamped 0-3 for visualization
        disp = hha[0]
        _imshow(axes[row, 2], disp,
                title="HHA[0] disparity (1/m)\n[bright=near, red=NaN]",
                cmap="viridis", vmin=0.0, vmax=3.0,
                mask_value=float("nan"))

        # HHA channel 1: height (m), diverging around 0 = floor reference
        height = hha[1]
        h_finite = height[~np.isnan(height)]
        h_lo = float(np.percentile(h_finite, 1)) if h_finite.size else -1.0
        h_hi = float(np.percentile(h_finite, 99)) if h_finite.size else 4.0
        h_max_abs = max(abs(h_lo), abs(h_hi), 1e-3)
        _imshow(axes[row, 3], height,
                title=f"HHA[1] height (m)\n[blue=below floor, red=above]",
                cmap="RdBu_r", vmin=-h_max_abs, vmax=h_max_abs,
                mask_value=float("nan"))

        # HHA channel 2: signed angle with gravity (deg) in [0, 180].
        # Diverging RdBu_r centered at 90 deg — the canonical choice for a
        # bipolar scalar field. Ceilings (0 deg) read as deep blue, vertical
        # walls (90 deg) as white/neutral, floors (180 deg) as deep red.
        # This makes the floor-vs-ceiling distinction visible (twilight is
        # cyclic and reads the two extremes as similar darks).
        angle = hha[2]
        norm = TwoSlopeNorm(vmin=0.0, vcenter=90.0, vmax=180.0)
        _imshow(axes[row, 4], angle,
                title=("HHA[2] angle vs gravity (deg)\n"
                       "[blue=ceiling (0), white=wall (90), red=floor (180)]"),
                cmap="RdBu_r", norm=norm,
                mask_value=float("nan"))

    fig.suptitle(
        "HHA verification — per-sample grid (train split)",
        fontsize=12, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"Wrote {out_path}")


# ---------------------------------------------------------------------------
# Aggregate histograms
# ---------------------------------------------------------------------------

def render_histograms(
    data_root: str,
    out_path: str,
    sample_pixels_per_channel: int = 5_000_000,
    chunk: int = 64,
):
    """Per-channel histograms across the train split.

    For each of (disparity, height, angle), reservoir-sample
    ``sample_pixels_per_channel`` valid pixels, then plot a histogram.
    Also overlays per-sensor histograms (subsampled smaller) to show
    sensor-specific drift.
    """
    train_dir = os.path.join(data_root, "train")
    hha_t = torch.load(os.path.join(train_dir, "hha_tensors.pt"),
                       weights_only=True, mmap=True)
    with open(os.path.join(train_dir, "sensors.json")) as f:
        sensors = json.load(f)
    sensor_set = sorted(set(sensors))
    sensor_to_idxs = {s: [i for i, x in enumerate(sensors) if x == s]
                      for s in sensor_set}

    # Reservoir samples per channel (overall) and per (channel, sensor).
    rng = torch.Generator().manual_seed(152)
    overall = [torch.empty(0, dtype=torch.float32) for _ in range(3)]
    per_sensor = {s: [torch.empty(0, dtype=torch.float32) for _ in range(3)]
                  for s in sensor_set}
    SENSOR_CAP = 1_000_000  # smaller per-sensor cap for plotting

    for start in range(0, hha_t.shape[0], chunk):
        end = min(hha_t.shape[0], start + chunk)
        ch_chunk = hha_t[start:end].to(torch.float32)
        chunk_sensors = sensors[start:end]
        for c in range(3):
            ch = ch_chunk[:, c]
            finite_mask = torch.isfinite(ch)
            vals = ch[finite_mask].view(-1)
            if vals.numel() == 0:
                continue
            overall[c] = torch.cat([overall[c], vals])
            if overall[c].numel() > sample_pixels_per_channel:
                perm = torch.randperm(overall[c].numel(), generator=rng)[
                    :sample_pixels_per_channel
                ]
                overall[c] = overall[c][perm]

            # Per-sensor: iterate samples in the chunk individually so we
            # know which sensor each row belongs to.
            for k, sname in enumerate(chunk_sensors):
                row = ch[k]
                rmask = torch.isfinite(row)
                rvals = row[rmask].view(-1)
                if rvals.numel() == 0:
                    continue
                buf = per_sensor[sname][c]
                per_sensor[sname][c] = torch.cat([buf, rvals])
                if per_sensor[sname][c].numel() > SENSOR_CAP:
                    p = torch.randperm(per_sensor[sname][c].numel(),
                                       generator=rng)[:SENSOR_CAP]
                    per_sensor[sname][c] = per_sensor[sname][c][p]

    # Plot.
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    titles = [
        "HHA[0] disparity (1/m)",
        "HHA[1] height (m)",
        "HHA[2] angle vs gravity (deg)  [0=ceiling, 90=wall, 180=floor]",
    ]
    ranges = [(0.0, 3.0), (-2.0, 5.0), (0.0, 180.0)]
    bins = 60

    for c in range(3):
        ax = axes[0, c]
        data = overall[c].numpy()
        ax.hist(data, bins=bins, range=ranges[c], color="C0",
                alpha=0.85, density=True)
        ax.set_title(f"{titles[c]}  (overall, n={data.size:,})", fontsize=10)
        ax.set_xlabel(titles[c])
        ax.set_ylabel("density")
        ax.set_xlim(*ranges[c])

        ax = axes[1, c]
        for s in sensor_set:
            d = per_sensor[s][c].numpy()
            if d.size == 0:
                continue
            ax.hist(d, bins=bins, range=ranges[c], alpha=0.45,
                    label=f"{s} (n={d.size:,})", density=True)
        ax.set_title(f"{titles[c]}  (per-sensor)", fontsize=10)
        ax.set_xlabel(titles[c])
        ax.set_ylabel("density")
        ax.set_xlim(*ranges[c])
        ax.legend(fontsize=7)

    # Mark expected canonical modes for the angle channel: 0=ceiling,
    # 90=wall, 180=floor. We expect a tri-modal distribution if the floor
    # and ceiling are well-separated in the data.
    for ax in [axes[0, 2], axes[1, 2]]:
        ax.axvline(0, color="black", linestyle=":", alpha=0.5)
        ax.axvline(90, color="black", linestyle=":", alpha=0.5)
        ax.axvline(180, color="black", linestyle=":", alpha=0.5)

    fig.suptitle(
        "HHA verification — per-channel histograms across train split",
        fontsize=12, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"Wrote {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True,
                   help="Preprocessed HHA dataset root (e.g. data/sunrgbd_19_hha)")
    p.add_argument("--sunrgbd-base", default="data/sunrgbd/SUNRGBD",
                   help="Original SUN RGB-D root (for raw-depth visualization).")
    p.add_argument("--num-samples", type=int, default=6,
                   help="Number of samples in the per-sample grid.")
    p.add_argument("--seed", type=int, default=152,
                   help="Seed for sample selection.")
    p.add_argument("--out-dir", default=None,
                   help="Output directory. Defaults to <data_root>/_diagnostics/")
    p.add_argument("--skip-grid", action="store_true",
                   help="Skip per-sample grid (only render histograms).")
    p.add_argument("--skip-hist", action="store_true",
                   help="Skip histograms (only render per-sample grid).")
    p.add_argument("--hist-pixels-per-channel", type=int, default=5_000_000,
                   help="Pixels reservoir-sampled per channel for histograms.")
    args = p.parse_args()

    out_dir = args.out_dir or os.path.join(args.data_root, "_diagnostics")
    os.makedirs(out_dir, exist_ok=True)

    if not args.skip_grid:
        train_dir = os.path.join(args.data_root, "train")
        with open(os.path.join(train_dir, "sensors.json")) as f:
            sensors = json.load(f)
        with open(os.path.join(train_dir, "labels.txt")) as f:
            labels = [int(x) for x in f.read().split()]
        class_names = []
        with open(os.path.join(args.data_root, "class_names.txt")) as f:
            for line in f:
                class_names.append(line.strip().split(": ", 1)[1])
        idxs = _pick_samples(sensors, labels, class_names,
                             num_samples=args.num_samples, seed=args.seed)
        print(f"Picked sample indices: {idxs}")
        for i in idxs:
            print(f"  #{i}: sensor={sensors[i]}  scene={class_names[labels[i]]}")
        render_samples_grid(args.data_root, args.sunrgbd_base, idxs,
                            os.path.join(out_dir, "samples_grid.png"))

    if not args.skip_hist:
        render_histograms(args.data_root,
                          os.path.join(out_dir, "histograms.png"),
                          sample_pixels_per_channel=args.hist_pixels_per_channel)


if __name__ == "__main__":
    main()
