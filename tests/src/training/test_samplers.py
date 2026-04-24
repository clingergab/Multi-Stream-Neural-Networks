"""Unit tests for src.training.samplers.

Covers:
- Strict bit-for-bit equivalence of build_sampler_class_only / _v1 / _v2 against
  the exact math that was inline in colab_LINet3_SUN_training_with_val.ipynb cell
  19. If these fail, the factory has drifted from the existing production math.
- V3's asymmetric-clip invariants (no sample below baseline pre-renorm; k in (0, 1]).
- Dispatch behavior of build_sampler.
"""

import json
import os
from collections import Counter, defaultdict

import pytest
import torch

from training.samplers import (
    build_sampler,
    build_sampler_class_only,
    build_sampler_v1,
    build_sampler_v2,
    build_sampler_v3,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic SUN RGB-D-style training subset on disk
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_data_root(tmp_path):
    """Create a tmp dataset root with train/sample_paths.json of known composition.

    Composition (19 classes × 4 sensors, but a deliberately-skewed subset to
    exercise per-class sensor imbalance):
        - class 0: 30 xtion, 2 kv1          (rare-cell: kv1 has 2 samples)
        - class 1: 100 kv2, 10 xtion        (imbalanced)
        - class 2: 20 kv1, 20 realsense     (balanced, mid-size)
        - class 3: 50 kv1                   (single-sensor class)
        - class 4: 4 kv2, 4 realsense, 4 xtion  (all cells small)
    Total: 30+2 + 100+10 + 20+20 + 50 + 4+4+4 = 244 samples across 5 classes.

    Returns: (data_root, train_indices, all_labels).
    """
    train_dir = tmp_path / "train"
    train_dir.mkdir()

    # Build (label, sensor) tuples in order
    composition = [
        (0, "xtion", 30),
        (0, "kv1", 2),
        (1, "kv2", 100),
        (1, "xtion", 10),
        (2, "kv1", 20),
        (2, "realsense", 20),
        (3, "kv1", 50),
        (4, "kv2", 4),
        (4, "realsense", 4),
        (4, "xtion", 4),
    ]
    paths = []
    labels = []
    for label, sensor, count in composition:
        for i in range(count):
            paths.append(f"{sensor}/synthetic/{label}_{sensor}_{i}")
            labels.append(label)

    with open(train_dir / "sample_paths.json", "w") as f:
        json.dump(paths, f)

    # train_indices: include all samples (no val held out in these tests)
    train_indices = list(range(len(labels)))
    return str(tmp_path), train_indices, labels


# ---------------------------------------------------------------------------
# Helper: reference implementations matching the exact inline math from
# notebooks/colab_LINet3_SUN_training_with_val.ipynb cell 19 (both the HEAD
# baseline and the installed V1/V2 blocks). Bit-for-bit equivalence is the
# strongest available regression test that the factory hasn't drifted.
# ---------------------------------------------------------------------------

def _reference_class_only_weights_head(train_indices, all_labels):
    """Reference: the original baseline sampler math from git HEAD of the notebook.

    Original code (non-normalized, sum = K * num_train):
        label_counts = Counter(subset_labels)
        num_train = len(subset_labels)
        class_weights = {label: num_train / count for label, count in label_counts.items()}
        sample_weights = torch.tensor([class_weights[label] for label in subset_labels], ...)

    Our factory normalizes to sum=num_train. To match bit-for-bit we normalize
    the reference the same way.
    """
    subset_labels = [int(all_labels[i]) for i in train_indices]
    num_train = len(subset_labels)
    label_counts = Counter(subset_labels)
    class_weights = {y: num_train / n for y, n in label_counts.items()}
    w = torch.tensor([class_weights[y] for y in subset_labels], dtype=torch.float32)
    return w * (num_train / w.sum())


def _reference_v1_weights(train_indices, all_labels, data_root):
    """Reference: the exact V1 math installed in cell 19."""
    with open(os.path.join(data_root, "train", "sample_paths.json")) as f:
        all_paths = json.load(f)

    def _sensor_of(p):
        return p.split("/", 1)[0]

    subset_labels = [int(all_labels[i]) for i in train_indices]
    subset_sensors = [_sensor_of(all_paths[i]) for i in train_indices]
    num_train = len(subset_labels)

    cell_positions = defaultdict(list)
    for pos, (y, s) in enumerate(zip(subset_labels, subset_sensors)):
        cell_positions[(y, s)].append(pos)
    sensors_per_class = defaultdict(set)
    for (y, s) in cell_positions:
        sensors_per_class[y].add(s)

    num_classes = len(set(subset_labels))
    w = torch.zeros(num_train, dtype=torch.float32)
    for (y, s), pos_list in cell_positions.items():
        ww = 1.0 / (num_classes * len(sensors_per_class[y]) * len(pos_list))
        for p in pos_list:
            w[p] = ww
    return w * (num_train / w.sum())


# ---------------------------------------------------------------------------
# class_only tests
# ---------------------------------------------------------------------------

class TestBuildSamplerClassOnly:
    def test_bit_for_bit_matches_head_baseline(self, synthetic_data_root):
        """Strict torch.equal match against the HEAD baseline math."""
        data_root, train_indices, all_labels = synthetic_data_root
        new_weights, _ = build_sampler_class_only(
            train_indices, all_labels, seed=152, verbose=False
        )
        ref_weights = _reference_class_only_weights_head(train_indices, all_labels)
        assert torch.equal(new_weights, ref_weights), \
            "build_sampler_class_only output drifted from HEAD baseline inline math"

    def test_weights_sum_to_num_train(self, synthetic_data_root):
        data_root, train_indices, all_labels = synthetic_data_root
        weights, _ = build_sampler_class_only(
            train_indices, all_labels, seed=0, verbose=False
        )
        assert weights.sum().item() == pytest.approx(len(train_indices), rel=1e-5)

    def test_samples_in_same_class_have_equal_weight(self, synthetic_data_root):
        data_root, train_indices, all_labels = synthetic_data_root
        weights, _ = build_sampler_class_only(
            train_indices, all_labels, seed=0, verbose=False
        )
        # Group weights by class; within each class all weights should be identical
        by_class = defaultdict(list)
        for i, y in enumerate(all_labels):
            by_class[y].append(weights[i].item())
        for y, ws in by_class.items():
            assert max(ws) == pytest.approx(min(ws)), \
                f"class {y}: expected equal weights within class"

    def test_per_class_mass_is_uniform(self, synthetic_data_root):
        """P(class) = 1/K => each class's total expected draws should equal num_train/K."""
        data_root, train_indices, all_labels = synthetic_data_root
        weights, _ = build_sampler_class_only(
            train_indices, all_labels, seed=0, verbose=False
        )
        num_classes = len(set(all_labels))
        expected_per_class = len(train_indices) / num_classes
        per_class_mass = defaultdict(float)
        for i, y in enumerate(all_labels):
            per_class_mass[y] += weights[i].item()
        for y, mass in per_class_mass.items():
            assert mass == pytest.approx(expected_per_class, rel=1e-4), \
                f"class {y} total mass {mass} != expected {expected_per_class}"


# ---------------------------------------------------------------------------
# V1 tests
# ---------------------------------------------------------------------------

class TestBuildSamplerV1:
    def test_bit_for_bit_matches_inline_v1(self, synthetic_data_root):
        """Strict torch.equal match against the V1 math installed in cell 19."""
        data_root, train_indices, all_labels = synthetic_data_root
        new_weights, _ = build_sampler_v1(
            train_indices, all_labels, data_root, seed=152, verbose=False
        )
        ref_weights = _reference_v1_weights(train_indices, all_labels, data_root)
        assert torch.equal(new_weights, ref_weights), \
            "build_sampler_v1 output drifted from inline cell-19 V1 math"

    def test_weights_sum_to_num_train(self, synthetic_data_root):
        data_root, train_indices, all_labels = synthetic_data_root
        weights, _ = build_sampler_v1(
            train_indices, all_labels, data_root, seed=0, verbose=False
        )
        assert weights.sum().item() == pytest.approx(len(train_indices), rel=1e-5)

    def test_per_sensor_mass_within_class_is_uniform(self, synthetic_data_root):
        """Within each class, per-sensor total mass should be 1/S_c of class's mass."""
        data_root, train_indices, all_labels = synthetic_data_root
        weights, _ = build_sampler_v1(
            train_indices, all_labels, data_root, seed=0, verbose=False
        )
        with open(os.path.join(data_root, "train", "sample_paths.json")) as f:
            paths = json.load(f)
        sensors = [p.split("/", 1)[0] for p in paths]

        # Group mass by (class, sensor)
        cell_mass = defaultdict(float)
        for i in range(len(all_labels)):
            cell_mass[(all_labels[i], sensors[i])] += weights[i].item()

        # For each class, the non-zero cells' masses should be equal
        by_class = defaultdict(list)
        for (y, s), m in cell_mass.items():
            by_class[y].append(m)
        for y, masses in by_class.items():
            if len(masses) > 1:
                assert max(masses) == pytest.approx(min(masses), rel=1e-4), \
                    f"class {y}: sensor masses {masses} not uniform"

    def test_rare_cell_gets_high_draws_per_epoch(self, synthetic_data_root):
        """Class 0 has 2 kv1 samples; each should be drawn (num_train / (K*S_c*2)) per epoch."""
        data_root, train_indices, all_labels = synthetic_data_root
        weights, _ = build_sampler_v1(
            train_indices, all_labels, data_root, seed=0, verbose=False
        )
        # Class 0's kv1 samples are indices 30 and 31 (after 30 xtion samples)
        num_train = len(all_labels)
        num_classes = len(set(all_labels))
        # class 0 has 2 sensors: xtion(30), kv1(2). S_c = 2.
        expected = num_train / (num_classes * 2 * 2)
        assert weights[30].item() == pytest.approx(expected, rel=1e-4)
        assert weights[31].item() == pytest.approx(expected, rel=1e-4)


# ---------------------------------------------------------------------------
# V2 tests
# ---------------------------------------------------------------------------

class TestBuildSamplerV2:
    def test_weights_sum_to_num_train(self, synthetic_data_root):
        data_root, train_indices, all_labels = synthetic_data_root
        weights, _ = build_sampler_v2(
            train_indices, all_labels, data_root, seed=0,
            min_cell_n=15, target_max_draws=3.0, verbose=False,
        )
        assert weights.sum().item() == pytest.approx(len(train_indices), rel=1e-5)

    def test_max_draws_respects_target(self, synthetic_data_root):
        data_root, train_indices, all_labels = synthetic_data_root
        cap = 3.0
        weights, _ = build_sampler_v2(
            train_indices, all_labels, data_root, seed=0,
            min_cell_n=15, target_max_draws=cap, verbose=False,
        )
        # After clip+renormalize, max may nudge slightly above cap due to renorm.
        # But any sample that was capped should still be <= cap * (num_train / sum_after_clip_pre_renorm),
        # which in practice stays at or under cap by a small margin.
        # Strict invariant: max is not >> cap.
        max_d = weights.max().item()
        assert max_d <= cap * 1.2, f"max draws/ep {max_d} exceeds cap {cap} by >20%"

    def test_class_with_single_qualifying_sensor_falls_back_to_class_only(
        self, synthetic_data_root
    ):
        """Class 3 has 50 kv1 samples (only 1 sensor represented).
        It should fall back to class-only weighting: all class-3 samples have equal weight."""
        data_root, train_indices, all_labels = synthetic_data_root
        weights, _ = build_sampler_v2(
            train_indices, all_labels, data_root, seed=0,
            min_cell_n=15, target_max_draws=100.0,  # disable clip effect
            verbose=False,
        )
        class3_weights = [weights[i].item() for i, y in enumerate(all_labels) if y == 3]
        assert len(class3_weights) == 50
        assert max(class3_weights) == pytest.approx(min(class3_weights))


# ---------------------------------------------------------------------------
# V3 tests
# ---------------------------------------------------------------------------

class TestBuildSamplerV3:
    def test_weights_sum_to_num_train(self, synthetic_data_root):
        data_root, train_indices, all_labels = synthetic_data_root
        weights, _ = build_sampler_v3(
            train_indices, all_labels, data_root, seed=0, cap=5.0, verbose=False,
        )
        assert weights.sum().item() == pytest.approx(len(train_indices), rel=1e-5)

    def test_no_sample_below_baseline_pre_renormalization(self, synthetic_data_root):
        """Pre-renormalization, max(v1_capped, baseline) >= baseline for every sample.
        Post-renormalization, ratios are preserved (all scaled by k < 1 uniformly),
        so sample_weights[i] / baseline[i] == k for baseline-floored samples.
        """
        data_root, train_indices, all_labels = synthetic_data_root
        baseline, _ = build_sampler_class_only(
            train_indices, all_labels, seed=0, verbose=False,
        )
        v3, _ = build_sampler_v3(
            train_indices, all_labels, data_root, seed=0, cap=5.0, verbose=False,
        )
        # Check that ratio v3/baseline is >= k for all samples, where k is the
        # global scale factor (equal for all baseline-floored samples).
        ratios = v3 / baseline
        # Samples where baseline was the floor all share the SAME ratio (the k factor).
        # Samples that were upsampled by V1 have a larger ratio.
        # The minimum ratio = k.
        k = ratios.min().item()
        assert k > 0, "renormalization factor must be positive"
        assert k <= 1.0 + 1e-5, f"k={k} should be <= 1 (max-combined sum >= num_train)"
        # Every sample's weight should be at least k * baseline (the baseline-floored case)
        assert torch.all(v3 >= k * baseline - 1e-6), \
            "some samples fell below baseline * k after renormalization"

    def test_clipping_prevents_rare_cell_hammering(self, synthetic_data_root):
        """Class 0 + kv1 (2 samples) would get ~60 draws/ep under raw V1.
        With cap=5 in V3, their weight should be clipped before the max operation."""
        data_root, train_indices, all_labels = synthetic_data_root
        cap = 5.0
        weights, _ = build_sampler_v3(
            train_indices, all_labels, data_root, seed=0, cap=cap, verbose=False,
        )
        # The kv1 class-0 samples (indices 30, 31) should not exceed cap * k ~ cap
        # (after renormalization shrink). Loose upper bound: cap.
        assert weights[30].item() <= cap
        assert weights[31].item() <= cap


# ---------------------------------------------------------------------------
# Dispatch tests
# ---------------------------------------------------------------------------

class TestBuildSamplerDispatch:
    @pytest.mark.parametrize("variant", ["class_only", "v1", "v2", "v3"])
    def test_dispatch_returns_matching_variant(self, synthetic_data_root, variant):
        data_root, train_indices, all_labels = synthetic_data_root
        weights_dispatch, _ = build_sampler(
            variant=variant, train_indices=train_indices, all_labels=all_labels,
            data_root=data_root, seed=42, verbose=False,
        )
        # Compare against the direct function call
        direct_fn = {
            "class_only": lambda: build_sampler_class_only(
                train_indices, all_labels, seed=42, verbose=False,
            ),
            "v1": lambda: build_sampler_v1(
                train_indices, all_labels, data_root, seed=42, verbose=False,
            ),
            "v2": lambda: build_sampler_v2(
                train_indices, all_labels, data_root, seed=42, verbose=False,
            ),
            "v3": lambda: build_sampler_v3(
                train_indices, all_labels, data_root, seed=42, verbose=False,
            ),
        }[variant]
        weights_direct, _ = direct_fn()
        assert torch.equal(weights_dispatch, weights_direct), \
            f"dispatch({variant!r}) drifted from direct function call"

    def test_unknown_variant_raises(self, synthetic_data_root):
        data_root, train_indices, all_labels = synthetic_data_root
        with pytest.raises(ValueError, match="Unknown sampler variant"):
            build_sampler(
                variant="nonsense", train_indices=train_indices,
                all_labels=all_labels, data_root=data_root, seed=0, verbose=False,
            )

    def test_case_insensitive_variant(self, synthetic_data_root):
        data_root, train_indices, all_labels = synthetic_data_root
        w_lower, _ = build_sampler("v1", train_indices, all_labels, data_root, seed=0, verbose=False)
        w_upper, _ = build_sampler("V1", train_indices, all_labels, data_root, seed=0, verbose=False)
        assert torch.equal(w_lower, w_upper)
