"""
Weighted samplers for multi-stream (RGB-D) scene classification on datasets
with per-class sensor imbalance (e.g., SUN RGB-D).

Provides four variants with a single ``build_sampler(variant, ...)`` dispatch
so the choice is a one-line config knob in any notebook or script.

Variants:
    - ``class_only``: baseline inverse-class-frequency weighting (P(class) uniform).
    - ``v1``: uniform per-(class × sensor) weighting (P(sensor | class) uniform
      across represented sensors). Decorrelates class from sensor at the cost
      of aggressively upsampling rare cells (e.g., lab + kv1 with N=2 samples).
    - ``v2``: V1 + a cell-size threshold (sensors with < ``min_cell_n`` samples
      don't count as "represented") + a per-sample draws/epoch cap.
    - ``v3``: V1-style upsampling clipped to a cap, floored at the baseline
      class-only weights via ``torch.maximum``. Best of both — caps rare-cell
      memorization without downsampling well-populated cells below baseline.

All variants:
    * Use ONLY training statistics (no test-set leakage).
    * Normalize the returned ``sample_weights`` so ``sum == num_train``; each
      entry is thus interpretable as "expected draws per epoch per sample"
      (WeightedRandomSampler normalizes internally, so absolute scale doesn't
      change sampling behavior; normalization is for interpretable diagnostics).
    * Build and return a ``torch.utils.data.WeightedRandomSampler`` seeded by
      the provided ``seed`` via a dedicated ``torch.Generator``.
    * Print one grep-able key=value diagnostic line.
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from typing import Sequence

import torch
from torch.utils.data import WeightedRandomSampler


_SENSORS = ("kv1", "kv2", "realsense", "xtion")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sensor_of(path: str) -> str:
    """Return first path component — the sensor prefix (kv1/kv2/realsense/xtion)."""
    return path.split("/", 1)[0]


def _load_sensors_for_subset(
    data_root: str, train_indices: Sequence[int]
) -> list[str]:
    """Read train/sample_paths.json and return sensor per sample aligned with train_indices."""
    paths_file = os.path.join(data_root, "train", "sample_paths.json")
    with open(paths_file) as f:
        all_paths = json.load(f)
    return [_sensor_of(all_paths[i]) for i in train_indices]


def _make_sampler(sample_weights: torch.Tensor, seed: int) -> WeightedRandomSampler:
    g = torch.Generator().manual_seed(int(seed))
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=g,
    )


def _normalize_to_num_train(weights: torch.Tensor, num_train: int) -> torch.Tensor:
    """Scale weights so their sum equals num_train. Each weight then = expected draws/epoch."""
    total = weights.sum().item()
    if total <= 0:
        raise ValueError("sample weights summed to <= 0; check per-class/per-cell counts")
    return weights * (num_train / total)


# ---------------------------------------------------------------------------
# Variant implementations
# ---------------------------------------------------------------------------

def build_sampler_class_only(
    train_indices: Sequence[int],
    all_labels: Sequence[int],
    seed: int,
    verbose: bool = True,
) -> tuple[torch.Tensor, WeightedRandomSampler]:
    """Baseline inverse-class-frequency weighting.

    P(class) = 1/K uniform across classes. Each sample of class c gets weight
    ``num_train / (K * N_c)`` after normalization, i.e. ``num_train / N_c``
    raw then normalized to ``sum == num_train``.
    """
    subset_labels = [int(all_labels[i]) for i in train_indices]
    num_train = len(subset_labels)
    label_counts = Counter(subset_labels)
    class_weights = {y: num_train / n for y, n in label_counts.items()}

    sample_weights = torch.tensor(
        [class_weights[y] for y in subset_labels], dtype=torch.float32
    )
    sample_weights = _normalize_to_num_train(sample_weights, num_train)

    if verbose:
        max_d = sample_weights.max().item()
        min_d = sample_weights[sample_weights > 0].min().item()
        print(
            f"sampler=class_only variant built: n={num_train} "
            f"min_draws={min_d:.3f} max_draws={max_d:.3f} "
            f"ratio={max_d / min_d:.1f}"
        )
    return sample_weights, _make_sampler(sample_weights, seed)


def build_sampler_v1(
    train_indices: Sequence[int],
    all_labels: Sequence[int],
    data_root: str,
    seed: int,
    verbose: bool = True,
) -> tuple[torch.Tensor, WeightedRandomSampler]:
    """Uniform per-(class × sensor) weighting.

    For each class c with S_c represented sensors, each sample in cell (c, s)
    gets raw weight ``1 / (K * S_c * N_{c,s})``, then weights are normalized
    so sum equals num_train. Makes P(class) = 1/K AND P(sensor | class) = 1/S_c.

    Aggressively upsamples rare cells (e.g., lab + kv1 at N=2 samples can hit
    50× per-epoch draws). Use with caution; see V2 / V3 for safer variants.
    """
    subset_labels = [int(all_labels[i]) for i in train_indices]
    subset_sensors = _load_sensors_for_subset(data_root, train_indices)
    num_train = len(subset_labels)

    cell_positions: dict[tuple[int, str], list[int]] = defaultdict(list)
    for pos, (y, s) in enumerate(zip(subset_labels, subset_sensors)):
        cell_positions[(y, s)].append(pos)

    sensors_per_class: dict[int, set] = defaultdict(set)
    for (y, s) in cell_positions:
        sensors_per_class[y].add(s)

    num_classes = len(set(subset_labels))
    sample_weights = torch.zeros(num_train, dtype=torch.float32)
    for (y, s), pos_list in cell_positions.items():
        w = 1.0 / (num_classes * len(sensors_per_class[y]) * len(pos_list))
        for p in pos_list:
            sample_weights[p] = w
    sample_weights = _normalize_to_num_train(sample_weights, num_train)

    if verbose:
        max_d = sample_weights.max().item()
        min_d = sample_weights[sample_weights > 0].min().item()
        warn = " [WARN] max>5" if max_d > 5.0 else ""
        print(
            f"sampler=v1 variant built: n={num_train} "
            f"min_draws={min_d:.3f} max_draws={max_d:.3f} "
            f"ratio={max_d / min_d:.1f}{warn}"
        )
    return sample_weights, _make_sampler(sample_weights, seed)


def build_sampler_v2(
    train_indices: Sequence[int],
    all_labels: Sequence[int],
    data_root: str,
    seed: int,
    min_cell_n: int = 15,
    target_max_draws: float = 3.0,
    verbose: bool = True,
) -> tuple[torch.Tensor, WeightedRandomSampler]:
    """V1 + cell-size threshold + per-sample draws/epoch cap.

    Classes with ≤1 qualifying sensor (>= min_cell_n samples) fall back to
    class-only weighting. Non-qualifying cells in decorrelated classes are
    weighted as if they had min_cell_n samples (prevents small-cell over-
    upweighting while keeping them represented). Final per-sample weight is
    clamped to target_max_draws and renormalized.
    """
    subset_labels = [int(all_labels[i]) for i in train_indices]
    subset_sensors = _load_sensors_for_subset(data_root, train_indices)
    num_train = len(subset_labels)

    cell_positions: dict[tuple[int, str], list[int]] = defaultdict(list)
    for pos, (y, s) in enumerate(zip(subset_labels, subset_sensors)):
        cell_positions[(y, s)].append(pos)

    num_classes = len(set(subset_labels))
    sample_weights = torch.zeros(num_train, dtype=torch.float32)

    n_decorrelated = 0
    n_class_only = 0

    for y in set(subset_labels):
        cells = [
            (s, cell_positions[(y, s)])
            for s in _SENSORS
            if (y, s) in cell_positions
        ]
        qualifying = [(s, pl) for s, pl in cells if len(pl) >= min_cell_n]
        all_positions = [p for _, pl in cells for p in pl]

        if len(qualifying) <= 1:
            # Class-only weighting for this class
            w = 1.0 / (num_classes * len(all_positions))
            for p in all_positions:
                sample_weights[p] = w
            n_class_only += 1
        else:
            S_c = len(qualifying)
            for s, pos_list in qualifying:
                w = 1.0 / (num_classes * S_c * len(pos_list))
                for p in pos_list:
                    sample_weights[p] = w
            # Non-qualifying cells: effective_n = max(actual, min_cell_n)
            for s, pos_list in cells:
                if len(pos_list) >= min_cell_n:
                    continue
                effective_n = max(len(pos_list), min_cell_n)
                w = 1.0 / (num_classes * S_c * effective_n)
                for p in pos_list:
                    sample_weights[p] = w
            n_decorrelated += 1

    # First normalization to num_train so draws/ep is interpretable BEFORE clip
    sample_weights = _normalize_to_num_train(sample_weights, num_train)
    n_over_cap = int((sample_weights > target_max_draws).sum().item())

    # Clip per-sample draws/epoch and renormalize
    sample_weights = sample_weights.clamp_max(target_max_draws)
    sample_weights = _normalize_to_num_train(sample_weights, num_train)

    if verbose:
        max_d = sample_weights.max().item()
        print(
            f"sampler=v2 variant built: n={num_train} "
            f"decorrelated={n_decorrelated} class_only={n_class_only} "
            f"clipped={n_over_cap} max_draws={max_d:.3f} "
            f"min_cell_n={min_cell_n} target_max_draws={target_max_draws}"
        )
    return sample_weights, _make_sampler(sample_weights, seed)


def build_sampler_v3(
    train_indices: Sequence[int],
    all_labels: Sequence[int],
    data_root: str,
    seed: int,
    cap: float = 5.0,
    verbose: bool = True,
) -> tuple[torch.Tensor, WeightedRandomSampler]:
    """Asymmetric clipping: V1 upsampling capped, baseline class-only as a floor.

    Procedure (all weights normalized to sum=num_train before combining):
        v1_capped = V1 weights clamped to `cap`
        sample_weights = torch.maximum(v1_capped, baseline)
        sample_weights = sample_weights * num_train / sample_weights.sum()

    The pre-renormalization `max` ensures no sample weight falls below its
    baseline class-only weight, protecting well-populated cells (e.g.,
    furniture_store + kv2) that V1 would have downsampled. The cap prevents
    rare-cell memorization. Post-renormalization shrinks everything by a
    factor k ≤ 1 (since the max-combined sum >= num_train); ratios relative
    to baseline are preserved.
    """
    subset_labels = [int(all_labels[i]) for i in train_indices]
    num_train = len(subset_labels)

    # V1 weights (fully computed, normalized)
    v1_weights, _ = build_sampler_v1(
        train_indices, all_labels, data_root, seed, verbose=False
    )

    # Baseline class-only weights (normalized)
    baseline_weights, _ = build_sampler_class_only(
        train_indices, all_labels, seed, verbose=False
    )

    # Asymmetric: clip V1's overshoots, floor at baseline via max
    v1_capped = v1_weights.clamp_max(cap)
    sample_weights = torch.maximum(v1_capped, baseline_weights)

    # Bookkeeping before renormalization
    n_upsampled = int((v1_capped > baseline_weights).sum().item())
    n_baseline_floored = int((baseline_weights >= v1_capped).sum().item())
    sum_before_renorm = sample_weights.sum().item()
    k = num_train / sum_before_renorm  # renormalization factor

    sample_weights = sample_weights * k  # renormalize to sum==num_train

    if verbose:
        max_d = sample_weights.max().item()
        print(
            f"sampler=v3 variant built: n={num_train} cap={cap} "
            f"upsampled={n_upsampled} baseline_floored={n_baseline_floored} "
            f"k={k:.3f} max_draws={max_d:.3f}"
        )
    return sample_weights, _make_sampler(sample_weights, seed)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def build_sampler(
    variant: str,
    train_indices: Sequence[int],
    all_labels: Sequence[int],
    data_root: str,
    seed: int,
    *,
    v2_min_cell_n: int = 15,
    v2_target_max_draws: float = 3.0,
    v3_cap: float = 5.0,
    verbose: bool = True,
) -> tuple[torch.Tensor, WeightedRandomSampler]:
    """Dispatch to the requested sampler variant.

    Args:
        variant: One of 'class_only', 'v1', 'v2', 'v3'.
        train_indices: Sample positions in the full train pool to include.
        all_labels: Class labels for the full train pool (indexed by raw position).
        data_root: Directory containing ``train/sample_paths.json`` (needed by
            variants v1/v2/v3 to extract per-sample sensors).
        seed: Seed for the sampler's internal generator.
        v2_min_cell_n, v2_target_max_draws: V2-specific knobs.
        v3_cap: V3-specific cap on V1 upsampling.
        verbose: Print a grep-able key=value diagnostic line.

    Returns:
        (sample_weights, train_sampler) — weights normalized so sum==num_train
        (each entry is expected draws/epoch for that sample) and a
        WeightedRandomSampler seeded accordingly.
    """
    v = variant.lower()
    if v == "class_only":
        return build_sampler_class_only(train_indices, all_labels, seed, verbose=verbose)
    if v == "v1":
        return build_sampler_v1(train_indices, all_labels, data_root, seed, verbose=verbose)
    if v == "v2":
        return build_sampler_v2(
            train_indices, all_labels, data_root, seed,
            min_cell_n=v2_min_cell_n, target_max_draws=v2_target_max_draws,
            verbose=verbose,
        )
    if v == "v3":
        return build_sampler_v3(
            train_indices, all_labels, data_root, seed,
            cap=v3_cap, verbose=verbose,
        )
    raise ValueError(
        f"Unknown sampler variant {variant!r}. "
        f"Expected one of: 'class_only', 'v1', 'v2', 'v3'."
    )
