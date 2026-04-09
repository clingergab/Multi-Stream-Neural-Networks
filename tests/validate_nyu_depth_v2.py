"""
Comprehensive validation of the preprocessed NYU Depth V2 10-category dataset.

Validates file completeness, tensor shapes/dtypes, labels, split correctness,
depth sanity, normalization stats, class distribution, and category mapping.
"""

import json
import os
import re
import sys
from collections import Counter

import h5py
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "nyu_depth_v2_10_traintest")
MAT_FILE = os.path.join(PROJECT_ROOT, "data", "nyu_depth_v2_labeled.mat")
# The CSV may have been moved after preprocessing; search common locations
SPLIT_CSV_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "data", "nyu", "nyu2_test.csv"),
    os.path.expanduser("~/Downloads/nyu_data/data/nyu2_test.csv"),
    os.path.expanduser("~/Downloads/nyu_data 2/data/nyu2_test.csv"),
]

TARGET_SIZE = (256, 256)

_STANDARD_9 = {
    "bathroom", "bedroom", "bookstore", "classroom", "dining_room",
    "home_office", "kitchen", "living_room", "office",
}
NYU_CATEGORIES = sorted(_STANDARD_9) + ["others"]
_CAT_TO_IDX = {name: idx for idx, name in enumerate(NYU_CATEGORIES)}

# ---------------------------------------------------------------------------
# Test tracking
# ---------------------------------------------------------------------------
_passed = 0
_failed = 0
_errors = []


def check(name, condition, detail=""):
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  [PASS] {name}")
    else:
        _failed += 1
        msg = f"  [FAIL] {name}"
        if detail:
            msg += f"  -- {detail}"
        print(msg)
        _errors.append(f"{name}: {detail}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_split_csv():
    for p in SPLIT_CSV_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


def parse_test_indices(csv_path):
    test_indices = set()
    with open(csv_path) as f:
        for line in f:
            m = re.match(r"data/nyu2_test/(\d+)_colors\.png", line.strip())
            if m:
                test_indices.add(int(m.group(1)))
    return sorted(test_indices)


def _extract_scene_type(scene_name):
    m = re.match(r"(.+?)_\d+[a-z]*$", scene_name)
    if m:
        return m.group(1)
    return scene_name


def _map_to_category(scene_type):
    if scene_type in _STANDARD_9:
        return scene_type
    return "others"


def load_scene_labels_from_mat(h5f):
    scenes_ds = h5f["scenes"]
    num_samples = scenes_ds.shape[1]
    scene_types = []
    for i in range(num_samples):
        ref = scenes_ds[0, i]
        ascii_array = np.array(h5f[ref])
        scene_name = "".join(chr(int(c)) for c in ascii_array.flatten())
        scene_type = _extract_scene_type(scene_name)
        scene_types.append(scene_type)
    mapped = [_map_to_category(st) for st in scene_types]
    return scene_types, mapped


# ===========================================================================
# VALIDATION CHECKS
# ===========================================================================

def validate_file_completeness():
    print("\n1. FILE COMPLETENESS")
    expected_files = [
        "train/rgb_tensors.pt",
        "train/depth_tensors.pt",
        "train/labels.txt",
        "test/rgb_tensors.pt",
        "test/depth_tensors.pt",
        "test/labels.txt",
        "class_names.txt",
        "dataset_info.txt",
        "norm_stats.json",
    ]
    for f in expected_files:
        path = os.path.join(DATA_ROOT, f)
        check(f"File exists: {f}", os.path.exists(path), f"Missing: {path}")


def validate_tensor_shapes_dtypes():
    print("\n2. TENSOR SHAPES & DTYPES")
    expected = {"train": 795, "test": 654}

    for split, n_expected in expected.items():
        rgb = torch.load(os.path.join(DATA_ROOT, split, "rgb_tensors.pt"),
                         weights_only=True)
        depth = torch.load(os.path.join(DATA_ROOT, split, "depth_tensors.pt"),
                           weights_only=True)

        check(f"{split} RGB shape",
              rgb.shape == (n_expected, 3, 256, 256),
              f"got {tuple(rgb.shape)}, expected ({n_expected}, 3, 256, 256)")

        check(f"{split} RGB dtype",
              rgb.dtype == torch.uint8,
              f"got {rgb.dtype}, expected uint8")

        check(f"{split} depth shape",
              depth.shape == (n_expected, 1, 256, 256),
              f"got {tuple(depth.shape)}, expected ({n_expected}, 1, 256, 256)")

        check(f"{split} depth dtype",
              depth.dtype == torch.uint16,
              f"got {depth.dtype}, expected uint16")


def validate_labels():
    print("\n3. LABEL VALIDATION")
    expected_n = {"train": 795, "test": 654}

    for split, n_expected in expected_n.items():
        labels_path = os.path.join(DATA_ROOT, split, "labels.txt")
        with open(labels_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        labels = [int(l) for l in lines]

        check(f"{split} labels count",
              len(labels) == n_expected,
              f"got {len(labels)}, expected {n_expected}")

        all_valid = all(0 <= l <= 9 for l in labels)
        check(f"{split} labels in [0, 9]",
              all_valid,
              f"out-of-range labels found")

        classes_present = set(labels)
        check(f"{split} all 10 classes represented",
              classes_present == set(range(10)),
              f"missing classes: {set(range(10)) - classes_present}")

    # class_names.txt
    cn_path = os.path.join(DATA_ROOT, "class_names.txt")
    with open(cn_path) as f:
        cn_lines = [l.strip() for l in f if l.strip()]

    check("class_names.txt has 10 entries",
          len(cn_lines) == 10,
          f"got {len(cn_lines)}")

    format_ok = True
    for line in cn_lines:
        if not re.match(r"^\d+: \w+$", line):
            format_ok = False
            break
    check("class_names.txt format 'idx: name'",
          format_ok,
          f"bad format in: {line if not format_ok else ''}")


def validate_split(h5f):
    print("\n4. SPLIT VALIDATION (no data leakage)")

    csv_path = find_split_csv()
    if csv_path is None:
        check("nyu2_test.csv found", False,
              "Could not find nyu2_test.csv in any expected location")
        return

    check("nyu2_test.csv found", True, csv_path)

    test_indices_csv = parse_test_indices(csv_path)
    check("CSV has 654 test indices",
          len(test_indices_csv) == 654,
          f"got {len(test_indices_csv)}")

    all_indices = set(range(1449))
    train_indices_csv = sorted(all_indices - set(test_indices_csv))
    check("Train indices = complement (795)",
          len(train_indices_csv) == 795,
          f"got {len(train_indices_csv)}")

    # Verify no overlap
    overlap = set(train_indices_csv) & set(test_indices_csv)
    check("No overlap between train and test indices",
          len(overlap) == 0,
          f"{len(overlap)} overlapping indices")

    # Verify union is complete
    union = set(train_indices_csv) | set(test_indices_csv)
    check("Union covers all 1449 samples",
          union == all_indices,
          f"union has {len(union)} samples")

    # Pixel-level comparison for random sample of ~20 images
    print("  Verifying pixel-level match for 20 random samples...")
    images_ds = h5f["images"]
    rgb_train = torch.load(os.path.join(DATA_ROOT, "train", "rgb_tensors.pt"),
                           weights_only=True)
    rgb_test = torch.load(os.path.join(DATA_ROOT, "test", "rgb_tensors.pt"),
                          weights_only=True)

    rng = np.random.RandomState(42)
    # Sample 10 from train, 10 from test
    train_sample = rng.choice(len(train_indices_csv), 10, replace=False)
    test_sample = rng.choice(len(test_indices_csv), 10, replace=False)

    all_match = True
    for local_idx in train_sample:
        mat_idx = train_indices_csv[local_idx]
        rgb_mat = np.array(images_ds[mat_idx])
        rgb_mat = np.transpose(rgb_mat, (1, 2, 0))
        if rgb_mat.dtype != np.uint8:
            rgb_mat = rgb_mat.astype(np.uint8)
        rgb_img = Image.fromarray(rgb_mat, mode="RGB")
        rgb_img = TF.resize(rgb_img, TARGET_SIZE)
        rgb_ref = torch.from_numpy(np.array(rgb_img, dtype=np.uint8)).permute(2, 0, 1)
        if not torch.equal(rgb_train[local_idx], rgb_ref):
            all_match = False
            break

    for local_idx in test_sample:
        mat_idx = test_indices_csv[local_idx]
        rgb_mat = np.array(images_ds[mat_idx])
        rgb_mat = np.transpose(rgb_mat, (1, 2, 0))
        if rgb_mat.dtype != np.uint8:
            rgb_mat = rgb_mat.astype(np.uint8)
        rgb_img = Image.fromarray(rgb_mat, mode="RGB")
        rgb_img = TF.resize(rgb_img, TARGET_SIZE)
        rgb_ref = torch.from_numpy(np.array(rgb_img, dtype=np.uint8)).permute(2, 0, 1)
        if not torch.equal(rgb_test[local_idx], rgb_ref):
            all_match = False
            break

    check("Pixel-level RGB match for 20 random samples", all_match)


def validate_depth_sanity(h5f):
    print("\n5. DEPTH SANITY")

    for split in ["train", "test"]:
        depth = torch.load(os.path.join(DATA_ROOT, split, "depth_tensors.pt"),
                           weights_only=True)
        depth_i32 = depth.to(torch.int32)
        depth_f = depth_i32.float()

        # No NaN/Inf (uint16 can't have them, but check float cast)
        check(f"{split} depth no NaN", not torch.isnan(depth_f).any())
        check(f"{split} depth no Inf", not torch.isinf(depth_f).any())

        # Range check (mm): valid non-zero should mostly be 500-10000
        nonzero_mask = depth_i32 > 0
        nonzero_vals = depth_f[nonzero_mask]
        d_min = nonzero_vals.min().item()
        d_max = nonzero_vals.max().item()

        check(f"{split} depth nonzero min >= 1 mm",
              d_min >= 1,
              f"min = {d_min}")
        check(f"{split} depth nonzero max <= 65535 mm",
              d_max <= 65535,
              f"max = {d_max}")

        # Most non-zero depth should be in plausible indoor range
        plausible = ((nonzero_vals >= 500) & (nonzero_vals <= 10000)).float().mean().item()
        check(f"{split} depth >80% in 500-10000mm range",
              plausible > 0.80,
              f"only {plausible*100:.1f}% in range")

        # NYU Depth V2 labeled.mat contains inpainted (hole-filled) depth,
        # so there should be NO zero-depth pixels (unlike raw Kinect data).
        zero_frac = (depth_i32 == 0).float().mean().item()
        check(f"{split} depth=0 pixels absent (inpainted depth)",
              zero_frac == 0.0,
              f"zero fraction = {zero_frac*100:.2f}% (expected 0%)")

    # Verify a few samples match .mat source after meters->mm conversion and resize
    print("  Verifying depth samples match .mat source...")
    csv_path = find_split_csv()
    if csv_path is None:
        check("Depth source match (skipped, no CSV)", False, "no CSV found")
        return

    test_indices_csv = parse_test_indices(csv_path)
    train_indices_csv = sorted(set(range(1449)) - set(test_indices_csv))

    depths_ds = h5f["depths"]
    depth_train = torch.load(os.path.join(DATA_ROOT, "train", "depth_tensors.pt"),
                             weights_only=True)

    rng = np.random.RandomState(123)
    sample_idxs = rng.choice(len(train_indices_csv), 5, replace=False)

    all_depth_match = True
    for local_idx in sample_idxs:
        mat_idx = train_indices_csv[local_idx]
        depth_mat = np.array(depths_ds[mat_idx])  # (640, 480) float, meters
        depth_mm = depth_mat * 1000.0
        depth_mm = np.clip(depth_mm, 0, 65535).astype(np.uint16)
        depth_img = Image.fromarray(depth_mm, mode="I;16")
        depth_img = TF.resize(depth_img, TARGET_SIZE,
                              interpolation=TF.InterpolationMode.NEAREST)
        depth_ref = torch.from_numpy(np.array(depth_img, dtype=np.uint16)).unsqueeze(0)
        stored = depth_train[local_idx]
        if not torch.equal(stored.to(torch.int32), depth_ref.to(torch.int32)):
            all_depth_match = False
            diff = (stored.to(torch.int32) - depth_ref.to(torch.int32)).abs()
            print(f"    Sample {local_idx} (mat {mat_idx}): max diff = {diff.max().item()}")

    check("Depth pixel-level match for 5 train samples", all_depth_match)


def validate_norm_stats():
    print("\n6. NORMALIZATION STATS VALIDATION")

    with open(os.path.join(DATA_ROOT, "norm_stats.json")) as f:
        saved_stats = json.load(f)

    # Recompute from train tensors
    rgb_t = torch.load(os.path.join(DATA_ROOT, "train", "rgb_tensors.pt"),
                       weights_only=True)
    depth_t = torch.load(os.path.join(DATA_ROOT, "train", "depth_tensors.pt"),
                         weights_only=True)

    rgb_f = rgb_t.float() / 255.0
    rgb_mean = rgb_f.mean(dim=(0, 2, 3)).tolist()
    rgb_std = rgb_f.std(dim=(0, 2, 3)).tolist()

    depth_f = depth_t.float() / 1000.0
    depth_i32 = depth_t.to(torch.int32)
    valid_depth = depth_f[depth_i32 > 0]
    depth_mean = [valid_depth.mean().item()]
    depth_std = [valid_depth.std().item()]

    # Compare to 6 decimal places
    for i, (comp, saved) in enumerate(zip(rgb_mean, saved_stats["rgb_mean"])):
        match = abs(comp - saved) < 1e-6
        check(f"RGB mean[{i}] match (6 dp)",
              match,
              f"computed={comp:.8f}, saved={saved:.8f}, diff={abs(comp-saved):.2e}")

    for i, (comp, saved) in enumerate(zip(rgb_std, saved_stats["rgb_std"])):
        match = abs(comp - saved) < 1e-6
        check(f"RGB std[{i}] match (6 dp)",
              match,
              f"computed={comp:.8f}, saved={saved:.8f}, diff={abs(comp-saved):.2e}")

    for i, (comp, saved) in enumerate(zip(depth_mean, saved_stats["depth_mean"])):
        match = abs(comp - saved) < 1e-6
        check(f"Depth mean[{i}] match (6 dp)",
              match,
              f"computed={comp:.8f}, saved={saved:.8f}, diff={abs(comp-saved):.2e}")

    for i, (comp, saved) in enumerate(zip(depth_std, saved_stats["depth_std"])):
        match = abs(comp - saved) < 1e-6
        check(f"Depth std[{i}] match (6 dp)",
              match,
              f"computed={comp:.8f}, saved={saved:.8f}, diff={abs(comp-saved):.2e}")

    # Verify stats are from train only (not test) -- recompute with test data
    # and show they differ
    rgb_test = torch.load(os.path.join(DATA_ROOT, "test", "rgb_tensors.pt"),
                          weights_only=True)
    rgb_all = torch.cat([rgb_t, rgb_test], dim=0).float() / 255.0
    rgb_mean_all = rgb_all.mean(dim=(0, 2, 3)).tolist()
    differs = any(abs(a - b) > 1e-4 for a, b in zip(rgb_mean_all, rgb_mean))
    check("Stats differ from train+test (train-only verified)", differs,
          "Stats match train+test combined -- may not be train-only!")


def validate_class_distribution():
    print("\n7. CLASS DISTRIBUTION")

    all_labels = []
    for split in ["train", "test"]:
        labels_path = os.path.join(DATA_ROOT, split, "labels.txt")
        with open(labels_path) as f:
            labels = [int(l.strip()) for l in f if l.strip()]
        all_labels.extend(labels)

        counts = Counter(labels)
        print(f"  {split} per-class counts:")
        for i in range(10):
            print(f"    {i:2d} {NYU_CATEGORIES[i]:22s}: {counts.get(i, 0)}")

    total = len(all_labels)
    check("Total samples = 1449",
          total == 1449,
          f"got {total}")

    # Verify "others" (idx 9) has samples
    train_labels = []
    with open(os.path.join(DATA_ROOT, "train", "labels.txt")) as f:
        train_labels = [int(l.strip()) for l in f if l.strip()]
    test_labels = []
    with open(os.path.join(DATA_ROOT, "test", "labels.txt")) as f:
        test_labels = [int(l.strip()) for l in f if l.strip()]

    train_others = sum(1 for l in train_labels if l == 9)
    test_others = sum(1 for l in test_labels if l == 9)
    check("'others' class has train samples",
          train_others > 0,
          f"train others count = {train_others}")
    check("'others' class has test samples",
          test_others > 0,
          f"test others count = {test_others}")


def validate_category_mapping(h5f):
    print("\n8. CATEGORY MAPPING VERIFICATION")

    scene_types, mapped = load_scene_labels_from_mat(h5f)

    # Check all 9 standard categories map to themselves
    for cat in sorted(_STANDARD_9):
        idx = _CAT_TO_IDX[cat]
        check(f"'{cat}' maps to class {idx}",
              _map_to_category(cat) == cat)

    # Check non-standard types map to "others"
    non_standard = set(scene_types) - _STANDARD_9
    all_to_others = all(_map_to_category(st) == "others" for st in non_standard)
    check(f"All {len(non_standard)} non-standard types map to 'others'",
          all_to_others)

    # Verify mapping consistency between train/test
    csv_path = find_split_csv()
    if csv_path is None:
        check("Category mapping consistency (skipped, no CSV)", False)
        return

    test_indices = parse_test_indices(csv_path)
    train_indices = sorted(set(range(1449)) - set(test_indices))

    train_labels_path = os.path.join(DATA_ROOT, "train", "labels.txt")
    test_labels_path = os.path.join(DATA_ROOT, "test", "labels.txt")
    with open(train_labels_path) as f:
        train_labels = [int(l.strip()) for l in f if l.strip()]
    with open(test_labels_path) as f:
        test_labels = [int(l.strip()) for l in f if l.strip()]

    # Check that stored labels match what we'd compute from .mat scene types
    train_consistent = True
    for i, mat_idx in enumerate(train_indices):
        expected_label = _CAT_TO_IDX[mapped[mat_idx]]
        if train_labels[i] != expected_label:
            train_consistent = False
            print(f"    Train mismatch at i={i}, mat_idx={mat_idx}: "
                  f"stored={train_labels[i]}, expected={expected_label} "
                  f"(scene={scene_types[mat_idx]})")
            break

    check("Train labels match .mat scene mapping", train_consistent)

    test_consistent = True
    for i, mat_idx in enumerate(test_indices):
        expected_label = _CAT_TO_IDX[mapped[mat_idx]]
        if test_labels[i] != expected_label:
            test_consistent = False
            print(f"    Test mismatch at i={i}, mat_idx={mat_idx}: "
                  f"stored={test_labels[i]}, expected={expected_label} "
                  f"(scene={scene_types[mat_idx]})")
            break

    check("Test labels match .mat scene mapping", test_consistent)

    # Print non-standard scene types that went to "others"
    others_types = Counter(st for st in scene_types if st not in _STANDARD_9)
    print(f"  Scene types mapped to 'others' ({len(others_types)} unique):")
    for st, cnt in others_types.most_common():
        print(f"    {st:<22s}: {cnt}")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("=" * 60)
    print("NYU Depth V2 10-Category Dataset Validation")
    print("=" * 60)
    print(f"Data root: {DATA_ROOT}")
    print(f"MAT file:  {MAT_FILE}")

    if not os.path.exists(DATA_ROOT):
        print(f"\nFATAL: Data root does not exist: {DATA_ROOT}")
        sys.exit(1)

    if not os.path.exists(MAT_FILE):
        print(f"\nFATAL: MAT file does not exist: {MAT_FILE}")
        sys.exit(1)

    # Run all validations
    validate_file_completeness()
    validate_tensor_shapes_dtypes()
    validate_labels()

    with h5py.File(MAT_FILE, "r") as h5f:
        validate_split(h5f)
        validate_depth_sanity(h5f)

    validate_norm_stats()
    validate_class_distribution()

    with h5py.File(MAT_FILE, "r") as h5f:
        validate_category_mapping(h5f)

    # Summary
    print("\n" + "=" * 60)
    print(f"VALIDATION SUMMARY: {_passed} passed, {_failed} failed")
    print("=" * 60)
    if _errors:
        print("\nFailed checks:")
        for e in _errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("\nAll checks passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
