# Development Plan: OmniPretrain Dataset & Dataloader

## 1. Objective & Scope

**What we are building:** A new dataset class `OmniPretrainDataset` and factory function `get_omnipretrain_dataloaders()` for loading per-sample tensor files from an ImageNet-style folder layout. This dataset will be used for pretraining LINet with ~90 object categories of paired RGB + depth data.

**Why:** The existing `SUNRGBDDataset` loads monolithic tensor files from a pre-split directory structure. The OmniPretrain dataset uses per-sample `.pt` files in class folders, requires train/val splitting at runtime, has uint16 millimeter depth (not uint8 normalized), and needs additional depth augmentations (scale jitter, random hole dropout) not present in SUNRGBD.

**Out of scope:**
- Modifying `SUNRGBDDataset` or its factory function
- GPU augmentation pipeline changes (this dataset follows the same `normalize` flag pattern)
- Preprocessing script to create the dataset on disk

**IMPORTANT — depth normalization units:** The preprocessing script MUST compute depth stats in meters (i.e., `valid_depth.float() / 1000.0` before computing mean/std), NOT in the 0–1 range. The `norm_stats.json` `depth_mean` and `depth_std` values must be in meters to match the `/1000.0` conversion in `__getitem__`. If stats are computed as `/65535.0` instead, sentinel replacement will use a near-zero value (~0.076) instead of the correct mean depth (~2.5m), silently corrupting training.

**In scope (clarification):**
- New OmniPretrain-specific augmentation constants (scale jitter range, hole dropout parameters) will be added to `augmentation_config.py` for consistency with the existing shared constant pattern.

## 2. Architecture & Design Decisions

### High-level approach
Create a single new file `src/data_utils/omnipretrain_dataset.py` containing:
1. `OmniPretrainDataset(Dataset)` -- the dataset class
2. `get_omnipretrain_dataloaders()` -- the factory function
3. Helper functions for class name/norm stats loading (reused from sunrgbd pattern but with format flexibility)

Update `src/data_utils/__init__.py` to export the new symbols.

### Key Design Decisions

**Decision 1: Reuse `_load_norm_stats` but create new `_load_class_names` with flexible parsing.**
Rationale: The requirements state class_names.txt must handle both `"bathroom"` and `"0: bathroom"` formats. The SUNRGBD version only handles the `"0: bathroom"` format. We write a new `_load_class_names` in the new module that handles both.
Rejected alternative: Import and modify the SUNRGBD version -- would break SUNRGBD's strict format expectations.

**Decision 2: File discovery at `__init__` time, storing list of `(rgb_path, depth_path, label)` tuples.**
Rationale: Walking directories once at init is cheap (~90 folders, thousands of files). Storing full paths avoids re-computation in `__getitem__`. Pairing validation happens once, failing fast on data integrity issues.
Rejected alternative: Lazy discovery (walk on first access) -- adds complexity without benefit since we need the full list for train/val splitting anyway.

**Decision 3: Train/val split via `sklearn.model_selection.train_test_split` with stratification.**
Rationale: Requirements specify 90/10 stratified split with reproducibility. sklearn provides this out of the box. The split operates on file path indices, not loading any tensors.
Rejected alternative: Manual stratified split implementation -- error-prone, sklearn is already a project dependency.

**Decision 4: Depth augmentation constants in `augmentation_config.py`.**
Rationale: Scale jitter range (0.9-1.1) and hole dropout parameters are augmentation concepts that belong alongside the other `BASE_*` constants. Placing them in `augmentation_config.py` maintains the single-source-of-truth pattern for all augmentation parameters and allows future datasets to reuse them.
Rejected alternative: Module-level constants in `omnipretrain_dataset.py` -- breaks the established pattern where all `BASE_*` augmentation constants live in one file.

**Decision 5: Depth 0-sentinel handling as mask-restore pattern.**
Rationale: Requirements are explicit about this pattern. Before augmentation, capture `zero_mask = (depth == 0.0)`. After all depth augmentations (scale jitter, brightness/contrast/noise, hole dropout), restore with `depth[zero_mask] = 0.0`. Then replace 0s with depth_mean before normalization. This ensures original missing pixels are treated correctly.
Rejected alternative: Using NaN as sentinel -- torch operations propagate NaN unpredictably, would require masking everywhere.

**Decision 6: `split` parameter accepts only `'train'` or `'val'`, with split computed internally by factory.**
Rationale: Unlike SUNRGBD which has pre-split directories on disk, this dataset splits at runtime. The factory computes indices for train/val and passes them to the dataset constructor. The dataset class accepts an explicit `indices` parameter to select which samples to use.
Rejected alternative: Having the dataset class itself do the split -- violates single responsibility, makes it impossible to ensure train and val use the same split without coordination.

**Decision 7: `_load_norm_stats` duplication is intentional.**
Rationale: Both `sunrgbd_dataset.py` and `omnipretrain_dataset.py` define their own `_load_norm_stats`. This keeps each module self-contained with zero coupling. Extracting a shared utility is not worth the added import dependency for a 10-line function with only 2 consumers.

**Decision 8: `_WorkerInitFn` duplication is intentional.**
Rationale: Same reasoning as Decision 7. The class is trivial (5 lines), has only 2 consumers, and keeping it local avoids cross-module coupling.

**Decision 9: Use raw tensor ops for zero_mask flip/crop instead of `F2` functions.**
Rationale: `F2.horizontal_flip` and `F2.crop` on a `torch.bool` tensor is NOT guaranteed to work across torchvision versions. Using `zero_mask.flip(-1)` and `zero_mask[:, i:i+h, j:j+w]` are pure tensor operations that always work on any dtype.
Rejected alternative: Using `F2.horizontal_flip(zero_mask)` and `F2.crop(zero_mask, ...)` -- may silently fail or error on bool tensors depending on torchvision version.

**Decision 10: Stratified split with fallback to non-stratified.**
Rationale: `train_test_split` with `stratify=` raises `ValueError` when any class has fewer than 2 samples. Rather than crashing, catch the error and fall back to a non-stratified random split with a warning. This is defensive coding for edge cases in real data.
Rejected alternative: Always requiring >= 2 samples per class -- too restrictive, would require users to curate data more carefully.

## 3. Implementation Details

### File 1: `src/data_utils/omnipretrain_dataset.py` (NEW)

This is the main implementation file. Order of implementation within the file:

1. Imports (including `warnings` for stratified split fallback)
2. `_load_class_names()` helper
3. `_load_norm_stats()` helper
4. `_discover_samples()` helper
5. `OmniPretrainDataset` class
6. `_WorkerInitFn` class (same pattern as SUNRGBD, plus `torch.manual_seed`)
7. `get_omnipretrain_dataloaders()` factory (with `stratified` parameter)

### File 2: `src/data_utils/__init__.py` (MODIFY)

Add imports for the new dataset and factory.

### File 3: `tests/src/data_utils/test_omnipretrain_dataset.py` (NEW)

Full test suite.

### File 4: `src/training/augmentation_config.py` (MODIFY)

Add new constants for hole dropout and depth scale jitter alongside existing `BASE_*` constants.

---

### Implementation order:
1. `augmentation_config.py` -- add new constants (no dependencies)
2. `omnipretrain_dataset.py` -- main implementation (depends on step 1)
3. `__init__.py` -- export new symbols (depends on step 2)
4. `test_omnipretrain_dataset.py` -- tests (depends on steps 1-3)

## 4. Code Snippets & Interface Contracts

### 4.1 New constants in `augmentation_config.py`

```python
# Depth scale jitter (OmniPretrain-specific, value multiply not spatial)
BASE_DEPTH_SCALE_JITTER_P = 0.50
BASE_DEPTH_SCALE_MIN = 0.9
BASE_DEPTH_SCALE_MAX = 1.1

# Depth random hole dropout (simulates sensor missing data)
BASE_HOLE_DROPOUT_P = 0.30
BASE_HOLE_DROPOUT_NUM_MIN = 3     # minimum number of rectangular holes
BASE_HOLE_DROPOUT_NUM_MAX = 8     # maximum number of rectangular holes
BASE_HOLE_DROPOUT_SIZE_MIN = 5    # minimum side length in pixels
BASE_HOLE_DROPOUT_SIZE_MAX = 20   # maximum side length in pixels
```

### 4.2 `omnipretrain_dataset.py` -- Full implementation

```python
"""
OmniPretrain Dataset Loader for LINet pretraining.

Loads per-sample tensor files from an ImageNet-style folder layout with
paired RGB and depth data. Class folders contain individual .pt files.

Tensors are stored at 256x256. At load time:
  - Train: RandomCrop(crop_size) + horizontal flip + augmentations
  - Val: CenterCrop(crop_size)
"""

import json
import os
import random
import re
import warnings
from collections import Counter

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F2

from src.training.augmentation_config import (
    # Probability baselines
    BASE_FLIP_P,
    BASE_COLOR_JITTER_P,
    BASE_BLUR_P,
    BASE_GRAYSCALE_P,
    BASE_RGB_ERASING_P,
    BASE_DEPTH_AUG_P,
    BASE_DEPTH_ERASING_P,
    BASE_DEPTH_SCALE_JITTER_P,
    BASE_DEPTH_SCALE_MIN,
    BASE_DEPTH_SCALE_MAX,
    BASE_HOLE_DROPOUT_P,
    BASE_HOLE_DROPOUT_NUM_MIN,
    BASE_HOLE_DROPOUT_NUM_MAX,
    BASE_HOLE_DROPOUT_SIZE_MIN,
    BASE_HOLE_DROPOUT_SIZE_MAX,
    # Magnitude baselines
    BASE_BRIGHTNESS,
    BASE_CONTRAST,
    BASE_SATURATION,
    BASE_HUE,
    BASE_BLUR_SIGMA_MIN,
    BASE_BLUR_SIGMA_MAX,
    BASE_ERASING_SCALE_MIN,
    BASE_ERASING_SCALE_MAX,
    BASE_ERASING_RATIO_MIN,
    BASE_ERASING_RATIO_MAX,
    BASE_DEPTH_BRIGHTNESS,
    BASE_DEPTH_CONTRAST,
    BASE_DEPTH_NOISE_STD,
    # Caps
    MAX_PROBABILITY,
    MAX_BRIGHTNESS,
    MAX_CONTRAST,
    MAX_SATURATION,
    MAX_HUE,
    MAX_BLUR_SIGMA,
    MAX_DEPTH_BRIGHTNESS,
    MAX_DEPTH_CONTRAST,
    MAX_DEPTH_NOISE_STD,
    MAX_ERASING_SCALE,
)


def _load_class_names(data_root: str) -> list[str]:
    """Load class names from class_names.txt in data_root.

    Handles both formats:
      - Plain: 'bathroom'
      - Indexed: '0: bathroom'

    Args:
        data_root: Root directory containing class_names.txt.

    Returns:
        List of class name strings in order.

    Raises:
        FileNotFoundError: If class_names.txt does not exist.
    """
    path = os.path.join(data_root, 'class_names.txt')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"class_names.txt not found in {data_root}. "
            f"Run the preprocessing script first."
        )
    names = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Handle "0: bathroom" format
            if ': ' in line and line.split(': ', 1)[0].strip().isdigit():
                names.append(line.split(': ', 1)[1])
            else:
                names.append(line)
    return names


def _load_norm_stats(data_root: str) -> dict:
    """Load normalization statistics from norm_stats.json in data_root.

    The JSON file must contain four keys with the following types:
      - ``rgb_mean``: ``list[float]`` (3 elements, channel means)
      - ``rgb_std``:  ``list[float]`` (3 elements, channel stds)
      - ``depth_mean``: ``list[float]`` (1 element, mean depth in meters)
      - ``depth_std``:  ``list[float]`` (1 element, depth std in meters)

    Args:
        data_root: Root directory containing norm_stats.json.

    Returns:
        Dict with keys: rgb_mean, rgb_std, depth_mean, depth_std.
        Values are lists of floats as described above.

    Raises:
        FileNotFoundError: If norm_stats.json does not exist.
    """
    path = os.path.join(data_root, 'norm_stats.json')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"norm_stats.json not found in {data_root}. "
            f"Run the preprocessing script with stats computation first."
        )
    with open(path, 'r') as f:
        return json.load(f)


def _discover_samples(
    data_root: str,
    class_names: list[str],
) -> list[tuple[str, str, int]]:
    """Walk class folders and discover paired RGB/depth sample files.

    Args:
        data_root: Root directory with class subfolders.
        class_names: Canonical class list from class_names.txt.

    Returns:
        List of (rgb_path, depth_path, label) tuples sorted by
        (class_name, rgb_filename) for deterministic ordering.

    Raises:
        ValueError: If unpaired rgb/depth files are found.
    """
    class_to_label = {name: i for i, name in enumerate(class_names)}
    samples = []

    for folder_name in sorted(os.listdir(data_root)):
        folder_path = os.path.join(data_root, folder_name)
        if not os.path.isdir(folder_path):
            continue
        if folder_name not in class_to_label:
            # Skip folders not in class_names.txt (e.g., metadata dirs)
            continue

        label = class_to_label[folder_name]

        # Collect RGB and depth files separately
        rgb_files = {}
        depth_files = {}

        for fname in os.listdir(folder_path):
            if not fname.endswith('.pt'):
                continue
            if fname.endswith('_rgb.pt'):
                stem = fname[:-len('_rgb.pt')]
                rgb_files[stem] = os.path.join(folder_path, fname)
            elif fname.endswith('_depth.pt'):
                stem = fname[:-len('_depth.pt')]
                depth_files[stem] = os.path.join(folder_path, fname)

        # Validate pairing
        rgb_only = set(rgb_files.keys()) - set(depth_files.keys())
        depth_only = set(depth_files.keys()) - set(rgb_files.keys())
        if rgb_only or depth_only:
            msg_parts = []
            if rgb_only:
                msg_parts.append(
                    f"RGB without depth: {sorted(rgb_only)[:5]}"
                )
            if depth_only:
                msg_parts.append(
                    f"Depth without RGB: {sorted(depth_only)[:5]}"
                )
            raise ValueError(
                f"Unpaired files in {folder_path}: {'; '.join(msg_parts)}"
            )

        for stem in sorted(rgb_files.keys()):
            samples.append((rgb_files[stem], depth_files[stem], label))

    return samples


class OmniPretrainDataset(Dataset):
    """
    OmniPretrain dataset for LINet pretraining.

    Loads per-sample .pt tensor files from an ImageNet-style class folder layout.
    Each sample has paired *_rgb.pt and *_depth.pt files.

    Class names are loaded from class_names.txt in data_root.
    Normalization stats are loaded from norm_stats.json in data_root.

    Tensors are stored at 256x256. crop_size controls the output:
      - Train: RandomCrop(crop_size) from 256x256
      - Val: CenterCrop(crop_size) from 256x256

    Directory structure:
        data_root/
            class_names.txt
            norm_stats.json
            chair/
                obj_000001_f000_rgb.pt
                obj_000001_f000_depth.pt
                ...
            sofa/
                ...
    """

    VALID_SPLITS = ('train', 'val')

    def __init__(
        self,
        data_root: str,
        split: str,
        samples: list[tuple[str, str, int]],
        class_names: list[str],
        norm_stats: dict,
        crop_size: int = 224,
        normalize: bool = True,
        rgb_aug_prob: float = 1.0,
        rgb_aug_mag: float = 1.0,
        depth_aug_prob: float = 1.0,
        depth_aug_mag: float = 1.0,
    ):
        """
        Args:
            data_root: Root directory of the dataset.
            split: One of 'train' or 'val'.
            samples: List of (rgb_path, depth_path, label) tuples.
            class_names: List of class name strings.
            norm_stats: Dict with rgb_mean, rgb_std, depth_mean, depth_std.
            crop_size: Output crop size (224).
            normalize: If True, normalize in __getitem__.
            rgb_aug_prob: Scales probability of RGB augmentations.
            rgb_aug_mag: Scales magnitude of RGB augmentations.
            depth_aug_prob: Scales probability of depth augmentations.
            depth_aug_mag: Scales magnitude of depth augmentations.
        """
        if split not in self.VALID_SPLITS:
            raise ValueError(
                f"split must be one of {self.VALID_SPLITS}, got '{split}'"
            )

        self.data_root = data_root
        self.split = split
        self.crop_size = crop_size
        self.normalize = normalize

        self.CLASS_NAMES = class_names
        self.num_classes = len(class_names)
        self._norm_stats = norm_stats

        self.samples = samples
        self.labels = [s[2] for s in samples]
        self.num_samples = len(samples)

        # Store augmentation scaling parameters
        self.rgb_aug_prob = rgb_aug_prob
        self.rgb_aug_mag = rgb_aug_mag
        self.depth_aug_prob = depth_aug_prob
        self.depth_aug_mag = depth_aug_mag

        # Pre-compute scaled augmentation values
        self._compute_scaled_aug_values()

        # Log augmentation config if scaling is applied
        if split == 'train' and any(
            p != 1.0
            for p in [rgb_aug_prob, rgb_aug_mag, depth_aug_prob, depth_aug_mag]
        ):
            self._log_augmentation_config()

    def __len__(self) -> int:
        return self.num_samples

    def _compute_scaled_aug_values(self):
        """Pre-compute scaled augmentation values based on aug_prob and aug_mag."""
        sync_prob = (self.rgb_aug_prob + self.depth_aug_prob) / 2

        # === SYNCHRONIZED (flip) ===
        self._flip_p = min(BASE_FLIP_P * sync_prob, MAX_PROBABILITY)

        # === RGB-ONLY ===
        self._color_jitter_p = min(
            BASE_COLOR_JITTER_P * self.rgb_aug_prob, MAX_PROBABILITY
        )
        self._brightness = min(BASE_BRIGHTNESS * self.rgb_aug_mag, MAX_BRIGHTNESS)
        self._contrast = min(BASE_CONTRAST * self.rgb_aug_mag, MAX_CONTRAST)
        self._saturation = min(BASE_SATURATION * self.rgb_aug_mag, MAX_SATURATION)
        self._hue = min(BASE_HUE * self.rgb_aug_mag, MAX_HUE)

        self._blur_p = min(BASE_BLUR_P * self.rgb_aug_prob, MAX_PROBABILITY)
        self._blur_sigma_min = BASE_BLUR_SIGMA_MIN
        self._blur_sigma_max = min(
            BASE_BLUR_SIGMA_MAX * self.rgb_aug_mag, MAX_BLUR_SIGMA
        )

        self._grayscale_p = min(
            BASE_GRAYSCALE_P * self.rgb_aug_prob, MAX_PROBABILITY
        )

        self._rgb_erasing_p = min(
            BASE_RGB_ERASING_P * self.rgb_aug_prob, MAX_PROBABILITY
        )
        self._rgb_erasing_scale_min = BASE_ERASING_SCALE_MIN
        self._rgb_erasing_scale_max = min(
            BASE_ERASING_SCALE_MAX * self.rgb_aug_mag, MAX_ERASING_SCALE
        )

        # === DEPTH-ONLY ===
        self._depth_aug_p = min(
            BASE_DEPTH_AUG_P * self.depth_aug_prob, MAX_PROBABILITY
        )
        self._depth_brightness = min(
            BASE_DEPTH_BRIGHTNESS * self.depth_aug_mag, MAX_DEPTH_BRIGHTNESS
        )
        self._depth_contrast = min(
            BASE_DEPTH_CONTRAST * self.depth_aug_mag, MAX_DEPTH_CONTRAST
        )
        self._depth_noise_std = min(
            BASE_DEPTH_NOISE_STD * self.depth_aug_mag, MAX_DEPTH_NOISE_STD
        )

        self._depth_erasing_p = min(
            BASE_DEPTH_ERASING_P * self.depth_aug_prob, MAX_PROBABILITY
        )
        self._depth_erasing_scale_min = BASE_ERASING_SCALE_MIN
        self._depth_erasing_scale_max = min(
            BASE_ERASING_SCALE_MAX * self.depth_aug_mag, MAX_ERASING_SCALE
        )

        # === DEPTH SCALE JITTER ===
        self._depth_scale_jitter_p = min(
            BASE_DEPTH_SCALE_JITTER_P * self.depth_aug_prob, MAX_PROBABILITY
        )

        # === DEPTH HOLE DROPOUT ===
        self._hole_dropout_p = min(
            BASE_HOLE_DROPOUT_P * self.depth_aug_prob, MAX_PROBABILITY
        )

        # === PRE-CREATE REUSABLE TRANSFORM INSTANCES ===
        self._color_jitter_transform = v2.ColorJitter(
            brightness=self._brightness,
            contrast=self._contrast,
            saturation=self._saturation,
            hue=self._hue,
        )
        self._rgb_erasing_transform = v2.RandomErasing(
            p=1.0,
            scale=(self._rgb_erasing_scale_min, self._rgb_erasing_scale_max),
            ratio=(BASE_ERASING_RATIO_MIN, BASE_ERASING_RATIO_MAX),
        )
        self._depth_erasing_transform = v2.RandomErasing(
            p=1.0,
            scale=(self._depth_erasing_scale_min, self._depth_erasing_scale_max),
            ratio=(BASE_ERASING_RATIO_MIN, BASE_ERASING_RATIO_MAX),
        )

    def _log_augmentation_config(self):
        """Log computed augmentation values when scaling is applied."""
        print(f"\nOmniPretrain augmentation scaling applied:")
        print(f"  RGB:   prob={self.rgb_aug_prob:.2f}, mag={self.rgb_aug_mag:.2f}")
        print(f"  Depth: prob={self.depth_aug_prob:.2f}, mag={self.depth_aug_mag:.2f}")
        print(f"  Computed values:")
        print(f"    [Sync]  Flip prob: {BASE_FLIP_P:.2f} -> {self._flip_p:.3f}")
        print(f"    [RGB]   ColorJitter prob: {BASE_COLOR_JITTER_P:.2f} -> {self._color_jitter_p:.3f}")
        print(f"    [RGB]   Brightness: +/-{BASE_BRIGHTNESS:.2f} -> +/-{self._brightness:.3f}")
        print(f"    [RGB]   Blur prob: {BASE_BLUR_P:.2f} -> {self._blur_p:.3f}")
        print(f"    [RGB]   Grayscale prob: {BASE_GRAYSCALE_P:.2f} -> {self._grayscale_p:.3f}")
        print(f"    [RGB]   Erasing prob: {BASE_RGB_ERASING_P:.2f} -> {self._rgb_erasing_p:.3f}")
        print(f"    [Depth] Aug prob: {BASE_DEPTH_AUG_P:.2f} -> {self._depth_aug_p:.3f}")
        print(f"    [Depth] Brightness: +/-{BASE_DEPTH_BRIGHTNESS:.2f} -> +/-{self._depth_brightness:.3f}")
        print(f"    [Depth] Noise std: {BASE_DEPTH_NOISE_STD:.3f} -> {self._depth_noise_std:.3f}")
        print(f"    [Depth] Erasing prob: {BASE_DEPTH_ERASING_P:.2f} -> {self._depth_erasing_p:.3f}")
        print(f"    [Depth] Scale jitter prob: {BASE_DEPTH_SCALE_JITTER_P:.2f} -> {self._depth_scale_jitter_p:.3f}")
        print(f"    [Depth] Hole dropout prob: {BASE_HOLE_DROPOUT_P:.2f} -> {self._hole_dropout_p:.3f}")

    def _apply_hole_dropout(self, depth: torch.Tensor) -> torch.Tensor:
        """Apply random rectangular hole dropout to depth tensor.

        Zeros out multiple small rectangular patches to simulate sensor
        missing data. Applied BEFORE normalization (sets pixels to 0 = sentinel).

        Modifies depth in-place and returns it.

        Args:
            depth: float32 tensor [1, H, W] in meters.

        Returns:
            The same depth tensor (modified in-place) with random rectangular
            regions set to 0.0.
        """
        _, h, w = depth.shape
        num_holes = np.random.randint(
            BASE_HOLE_DROPOUT_NUM_MIN, BASE_HOLE_DROPOUT_NUM_MAX + 1
        )
        for _ in range(num_holes):
            hole_h = np.random.randint(
                BASE_HOLE_DROPOUT_SIZE_MIN, BASE_HOLE_DROPOUT_SIZE_MAX + 1
            )
            hole_w = np.random.randint(
                BASE_HOLE_DROPOUT_SIZE_MIN, BASE_HOLE_DROPOUT_SIZE_MAX + 1
            )
            top = np.random.randint(0, max(1, h - hole_h + 1))
            left = np.random.randint(0, max(1, w - hole_w + 1))
            depth[:, top:top + hole_h, left:left + hole_w] = 0.0
        return depth

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Load and return a single sample.

        Returns:
            rgb: float32 [3, crop_size, crop_size]
            depth: float32 [1, crop_size, crop_size]
            label: int (0 to num_classes-1)
        """
        rgb_path, depth_path, label = self.samples[idx]

        # Load per-sample tensors
        rgb = torch.load(rgb_path, weights_only=True)      # uint8 [3, 256, 256]
        depth = torch.load(depth_path, weights_only=True)   # uint16 [1, 256, 256]

        # Validate loaded tensors
        assert rgb.shape[0] == 3 and rgb.ndim == 3, (
            f"Bad RGB shape {rgb.shape} at {rgb_path}"
        )
        assert depth.shape[0] == 1 and depth.ndim == 3, (
            f"Bad depth shape {depth.shape} at {depth_path}"
        )

        # Convert depth: uint16 mm -> float32 meters
        depth = depth.float() / 1000.0

        # ==================== TRAINING AUGMENTATION ====================
        if self.split == 'train':
            # Capture 0-sentinel mask BEFORE any augmentation
            zero_mask = (depth == 0.0)

            # 1. Synchronized Random Horizontal Flip
            if np.random.random() < self._flip_p:
                rgb = F2.horizontal_flip(rgb)
                depth = F2.horizontal_flip(depth)
                zero_mask = zero_mask.flip(-1)  # raw tensor op for bool safety

            # 2. Synchronized RandomCrop (256 -> crop_size)
            i, j, h, w = v2.RandomCrop.get_params(
                rgb, output_size=(self.crop_size, self.crop_size)
            )
            rgb = F2.crop(rgb, i, j, h, w)
            depth = F2.crop(depth, i, j, h, w)
            zero_mask = zero_mask[:, i:i+h, j:j+w]  # raw tensor op for bool safety

            # 3-5. RGB-Only Appearance Augmentation
            if self.normalize:
                # 3. Color Jitter
                if np.random.random() < self._color_jitter_p:
                    rgb = self._color_jitter_transform(rgb)

                # 4. Gaussian Blur
                if np.random.random() < self._blur_p:
                    kernel_size = int(np.random.choice([3, 5, 7]))
                    sigma = float(
                        np.random.uniform(
                            self._blur_sigma_min, self._blur_sigma_max
                        )
                    )
                    rgb = F2.gaussian_blur(rgb, kernel_size=kernel_size, sigma=sigma)

                # 5. Grayscale
                if np.random.random() < self._grayscale_p:
                    rgb = F2.rgb_to_grayscale(rgb, num_output_channels=3)

            # 6. Depth Scale Jitter (value multiply, NOT spatial resize)
            # Skip when normalize=False (GPU augmentation mode handles these)
            if self.normalize and np.random.random() < self._depth_scale_jitter_p:
                scale_factor = np.random.uniform(
                    BASE_DEPTH_SCALE_MIN, BASE_DEPTH_SCALE_MAX
                )
                depth = depth * scale_factor

            # 7. Depth Appearance Augmentation (brightness/contrast/noise)
            # Skip when normalize=False (GPU augmentation mode)
            if self.normalize and np.random.random() < self._depth_aug_p:
                # Normalize valid pixels to [0, 1]
                d_min = depth[~zero_mask].min() if (~zero_mask).any() else 0.0
                d_max = depth[~zero_mask].max() if (~zero_mask).any() else 1.0
                d_range = d_max - d_min

                depth_01 = torch.zeros_like(depth)  # zeros for masked pixels
                if d_range > 1e-6:
                    depth_01[~zero_mask] = (depth[~zero_mask] - d_min) / d_range

                brightness_factor = np.random.uniform(
                    1.0 - self._depth_brightness,
                    1.0 + self._depth_brightness,
                )
                contrast_factor = np.random.uniform(
                    1.0 - self._depth_contrast,
                    1.0 + self._depth_contrast,
                )

                # Contrast around 0.5, then brightness
                depth_01 = (depth_01 - 0.5) * contrast_factor + 0.5
                depth_01 = depth_01 * brightness_factor

                # Add Gaussian noise
                depth_01 = depth_01 + torch.randn_like(depth_01) * self._depth_noise_std

                # Map back, only for valid pixels
                if d_range > 1e-6:
                    depth[~zero_mask] = depth_01[~zero_mask].clamp(0.0, 1.0) * d_range + d_min
                # Zero-mask pixels remain unchanged (still 0.0 from scale jitter or original)

            # 8. Random Hole Dropout (BEFORE normalization, sets to 0 sentinel)
            if np.random.random() < self._hole_dropout_p:
                depth = self._apply_hole_dropout(depth)
                # Update zero_mask to include new holes
                zero_mask = zero_mask | (depth == 0.0)

            # Restore original 0-sentinel pixels
            depth[zero_mask] = 0.0

        else:
            # Val: CenterCrop (256 -> crop_size)
            rgb = F2.center_crop(rgb, (self.crop_size, self.crop_size))
            depth = F2.center_crop(depth, (self.crop_size, self.crop_size))
            # Compute zero_mask AFTER crop -- we only need to replace zeros
            # that survived into the cropped region.
            zero_mask = (depth == 0.0)

        # ==================== TO FLOAT32 ====================
        if rgb.dtype == torch.uint8:
            rgb = rgb.float() / 255.0

        # depth is already float32 in meters

        # ==================== SENTINEL REPLACEMENT ====================
        # Replace 0-sentinel with depth_mean so normalization maps them to ~0
        depth_mean_val = self._norm_stats['depth_mean'][0]
        depth[zero_mask] = depth_mean_val

        # ==================== NORMALIZATION ====================
        if self.normalize:
            rgb = F2.normalize(
                rgb,
                mean=self._norm_stats['rgb_mean'],
                std=self._norm_stats['rgb_std'],
            )
            depth = F2.normalize(
                depth,
                mean=self._norm_stats['depth_mean'],
                std=self._norm_stats['depth_std'],
            )

            # Post-normalization Random Erasing (CPU mode only)
            if self.split == 'train':
                if np.random.random() < self._rgb_erasing_p:
                    rgb = self._rgb_erasing_transform(rgb)
                if np.random.random() < self._depth_erasing_p:
                    depth = self._depth_erasing_transform(depth)

        return rgb, depth, label

    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for weighted loss (inverse frequency).

        Returns:
            torch.Tensor: Shape [num_classes], dtype float32.
        """
        label_counts = Counter(self.labels)
        weights = torch.zeros(self.num_classes)
        total = len(self.labels)

        for class_idx in range(self.num_classes):
            count = label_counts.get(class_idx, 0)
            if count > 0:
                weights[class_idx] = total / (self.num_classes * count)
            else:
                weights[class_idx] = 0.0

        return weights

    def get_sample_weights(self) -> torch.Tensor:
        """Calculate per-sample weights for WeightedRandomSampler.

        Each sample gets weight = 1 / (count of its class). This ensures
        balanced sampling across classes.

        Returns:
            torch.Tensor: Shape [num_samples], dtype float64.
        """
        label_counts = Counter(self.labels)
        sample_weights = torch.tensor(
            [1.0 / label_counts[label] for label in self.labels],
            dtype=torch.float64,
        )
        return sample_weights

    def get_class_distribution(self) -> dict[str, dict[str, float]]:
        """Get class distribution statistics.

        Returns:
            Dict mapping class name to {'count': int, 'percentage': float}.
        """
        label_counts = Counter(self.labels)
        distribution = {}
        for class_idx in range(self.num_classes):
            count = label_counts.get(class_idx, 0)
            percentage = (count / self.num_samples * 100) if self.num_samples > 0 else 0.0
            distribution[self.CLASS_NAMES[class_idx]] = {
                'count': count,
                'percentage': percentage,
            }
        return distribution

    def get_norm_stats(self) -> dict:
        """Return the normalization statistics dict loaded from norm_stats.json."""
        return self._norm_stats


class _WorkerInitFn:
    """Callable for DataLoader worker initialization (picklable)."""

    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, worker_id: int):
        worker_seed = self.seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)


def get_omnipretrain_dataloaders(
    data_root: str = 'data/sparse_omni_256',
    batch_size: int = 32,
    num_workers: int = 4,
    crop_size: int = 224,
    use_class_weights: bool = False,
    seed: int = 42,
    val_fraction: float = 0.1,
    normalize: bool = True,
    stratified: bool = True,
    rgb_aug_prob: float = 1.0,
    rgb_aug_mag: float = 1.0,
    depth_aug_prob: float = 1.0,
    depth_aug_mag: float = 1.0,
) -> tuple:
    """Create train and val dataloaders for OmniPretrain dataset.

    Performs a stratified 90/10 train/val split seeded for reproducibility.
    Uses WeightedRandomSampler for balanced training when stratified=True.

    Args:
        data_root: Root directory of the dataset.
        batch_size: Batch size.
        num_workers: Number of dataloader workers.
        crop_size: Output crop size.
        use_class_weights: If True, return class_weights as fourth element.
        seed: Random seed for split and reproducible loading.
        val_fraction: Fraction of data for validation (default 0.1).
        normalize: If True, normalize in __getitem__.
        stratified: If True, use stratified split and WeightedRandomSampler.
            If False, use non-stratified split and shuffle=True for training.
        rgb_aug_prob: Probability scaling for RGB augmentations.
        rgb_aug_mag: Magnitude scaling for RGB augmentations.
        depth_aug_prob: Probability scaling for depth augmentations.
        depth_aug_mag: Magnitude scaling for depth augmentations.

    Returns:
        (train_loader, val_loader, num_classes) if use_class_weights is False.
        (train_loader, val_loader, num_classes, class_weights) if True.
    """
    # Load metadata
    class_names = _load_class_names(data_root)
    norm_stats = _load_norm_stats(data_root)

    # Discover all samples
    all_samples = _discover_samples(data_root, class_names)
    if len(all_samples) == 0:
        raise ValueError(f"No samples found in {data_root}")

    all_labels = [s[2] for s in all_samples]

    # Stratified train/val split with fallback
    if stratified:
        try:
            train_indices, val_indices = train_test_split(
                list(range(len(all_samples))),
                test_size=val_fraction,
                random_state=seed,
                stratify=all_labels,
            )
        except ValueError:
            # Fallback: non-stratified split (class has < 2 samples)
            warnings.warn(
                "Stratified split failed (class with < 2 samples). "
                "Falling back to non-stratified random split.",
                UserWarning,
            )
            train_indices, val_indices = train_test_split(
                list(range(len(all_samples))),
                test_size=val_fraction,
                random_state=seed,
            )
    else:
        train_indices, val_indices = train_test_split(
            list(range(len(all_samples))),
            test_size=val_fraction,
            random_state=seed,
        )

    train_samples = [all_samples[i] for i in train_indices]
    val_samples = [all_samples[i] for i in val_indices]

    # Create datasets
    train_dataset = OmniPretrainDataset(
        data_root=data_root,
        split='train',
        samples=train_samples,
        class_names=class_names,
        norm_stats=norm_stats,
        crop_size=crop_size,
        normalize=normalize,
        rgb_aug_prob=rgb_aug_prob,
        rgb_aug_mag=rgb_aug_mag,
        depth_aug_prob=depth_aug_prob,
        depth_aug_mag=depth_aug_mag,
    )
    val_dataset = OmniPretrainDataset(
        data_root=data_root,
        split='val',
        samples=val_samples,
        class_names=class_names,
        norm_stats=norm_stats,
        crop_size=crop_size,
        normalize=normalize,
    )

    num_classes = len(class_names)

    # Setup reproducibility
    worker_init_fn = _WorkerInitFn(seed)

    # Weighted sampling for training (class imbalance) or simple shuffle
    if stratified:
        sample_weights = train_dataset.get_sample_weights()
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
        train_shuffle = False
    else:
        train_sampler = None
        train_shuffle = True

    # Handle num_workers=0 prefetch_factor footgun
    train_prefetch = 4 if num_workers > 0 else None
    val_prefetch = 2 if num_workers > 0 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        prefetch_factor=train_prefetch,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        worker_init_fn=worker_init_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=val_prefetch,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        worker_init_fn=worker_init_fn,
    )

    sampling_mode = "weighted" if stratified else "shuffle"
    print(f"\nOmniPretrain Dataset:")
    print(f"  Total samples: {len(all_samples)}")
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print(f"  Classes: {num_classes}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sampling: {sampling_mode}")

    if use_class_weights:
        class_weights = train_dataset.get_class_weights()
        print(f"  Class weights computed (inverse frequency)")
        return train_loader, val_loader, num_classes, class_weights

    return train_loader, val_loader, num_classes
```

### 4.3 Changes to `src/data_utils/__init__.py`

Add after the existing imports:

```python
# OmniPretrain dataset for LINet pretraining
from .omnipretrain_dataset import (
    OmniPretrainDataset,
    get_omnipretrain_dataloaders,
)
```

And add to `__all__`:

```python
    # OmniPretrain dataset
    'OmniPretrainDataset',
    'get_omnipretrain_dataloaders',
```

### 4.4 Changes to `src/training/augmentation_config.py`

Add after `BASE_DEPTH_ERASING_P = 0.10` (line 38):

```python
BASE_DEPTH_SCALE_JITTER_P = 0.50
```

Add after `BASE_DEPTH_NOISE_STD = 0.059` (line 70):

```python
# Depth scale jitter (value multiply, not spatial resize)
BASE_DEPTH_SCALE_MIN = 0.9
BASE_DEPTH_SCALE_MAX = 1.1

# Depth random hole dropout (simulates sensor missing data)
BASE_HOLE_DROPOUT_P = 0.30
BASE_HOLE_DROPOUT_NUM_MIN = 3
BASE_HOLE_DROPOUT_NUM_MAX = 8
BASE_HOLE_DROPOUT_SIZE_MIN = 5    # pixels
BASE_HOLE_DROPOUT_SIZE_MAX = 20   # pixels
```

### 4.5 Interface Contracts Table

| Function/Method | Signature | Returns | Raises |
|---|---|---|---|
| `_load_class_names(data_root: str)` | Single string arg | `list[str]` -- ordered class names | `FileNotFoundError` if file missing |
| `_load_norm_stats(data_root: str)` | Single string arg | `dict` with keys: `rgb_mean: list[float]` (3 elements), `rgb_std: list[float]` (3 elements), `depth_mean: list[float]` (1 element, in meters), `depth_std: list[float]` (1 element, in meters) | `FileNotFoundError` if file missing |
| `_discover_samples(data_root: str, class_names: list[str])` | Root dir + class list | `list[tuple[str, str, int]]` -- (rgb_path, depth_path, label), sorted deterministically | `ValueError` if unpaired files found |
| `OmniPretrainDataset.__init__(self, data_root: str, split: str, samples: list[tuple[str, str, int]], class_names: list[str], norm_stats: dict, crop_size: int = 224, normalize: bool = True, rgb_aug_prob: float = 1.0, rgb_aug_mag: float = 1.0, depth_aug_prob: float = 1.0, depth_aug_mag: float = 1.0)` | See args | `None` | `ValueError` if split invalid |
| `OmniPretrainDataset.__len__(self)` | No args | `int` | -- |
| `OmniPretrainDataset.__getitem__(self, idx: int)` | Integer index | `tuple[torch.Tensor, torch.Tensor, int]` -- (rgb float32 [3,224,224], depth float32 [1,224,224], label int) | `AssertionError` if loaded tensor has bad shape |
| `OmniPretrainDataset.get_class_weights(self)` | No args | `torch.Tensor` shape `[num_classes]`, dtype `float32` | -- |
| `OmniPretrainDataset.get_sample_weights(self)` | No args | `torch.Tensor` shape `[num_samples]`, dtype `float64` | -- |
| `OmniPretrainDataset.get_class_distribution(self)` | No args | `dict[str, dict[str, float]]` -- `{class_name: {'count': int, 'percentage': float}}` | -- |
| `OmniPretrainDataset.get_norm_stats(self)` | No args | `dict` -- same dict from norm_stats.json (see `_load_norm_stats` for key/value types) | -- |
| `OmniPretrainDataset._apply_hole_dropout(self, depth: torch.Tensor)` | float32 [1,H,W] | `torch.Tensor` same shape, some regions zeroed. **Modifies input in-place.** | -- |
| `_WorkerInitFn.__init__(self, seed: int)` | Integer seed | `None` | -- |
| `_WorkerInitFn.__call__(self, worker_id: int)` | Integer worker id. Seeds `np.random`, `random`, and `torch.manual_seed`. | `None` | -- |
| `get_omnipretrain_dataloaders(data_root: str = 'data/sparse_omni_256', batch_size: int = 32, num_workers: int = 4, crop_size: int = 224, use_class_weights: bool = False, seed: int = 42, val_fraction: float = 0.1, normalize: bool = True, stratified: bool = True, rgb_aug_prob: float = 1.0, rgb_aug_mag: float = 1.0, depth_aug_prob: float = 1.0, depth_aug_mag: float = 1.0)` | See args | `tuple[DataLoader, DataLoader, int]` or `tuple[DataLoader, DataLoader, int, torch.Tensor]` if `use_class_weights=True` | `ValueError` if no samples found; `FileNotFoundError` if metadata files missing |

## 5. Testing Strategy

### Test file: `tests/src/data_utils/test_omnipretrain_dataset.py`

All tests use `tmp_path` pytest fixture to create a temporary dataset directory with synthetic data.

#### Fixtures and Helpers

```python
import json
import os
import warnings

import numpy as np
import pytest
import torch

from src.data_utils.omnipretrain_dataset import (
    OmniPretrainDataset,
    _discover_samples,
    _load_class_names,
    _load_norm_stats,
    get_omnipretrain_dataloaders,
)


def _make_dataset(fake_dataset, split='train', **kwargs):
    """Module-level helper to construct dataset from fake_dataset fixture."""
    data_root, class_names, norm_stats = fake_dataset
    samples = _discover_samples(str(data_root), class_names)
    return OmniPretrainDataset(
        data_root=str(data_root),
        split=split,
        samples=samples,
        class_names=class_names,
        norm_stats=norm_stats,
        **kwargs,
    )


@pytest.fixture
def fake_dataset(tmp_path):
    """Create a minimal fake OmniPretrain dataset on disk.

    Creates 3 classes with 10 samples each (30 total).
    Includes some depth pixels set to 0 (missing data sentinel).
    """
    class_names = ['chair', 'sofa', 'table']

    # class_names.txt
    with open(tmp_path / 'class_names.txt', 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")

    # norm_stats.json
    norm_stats = {
        'rgb_mean': [0.485, 0.456, 0.406],
        'rgb_std': [0.229, 0.224, 0.225],
        'depth_mean': [2.5],
        'depth_std': [1.2],
    }
    with open(tmp_path / 'norm_stats.json', 'w') as f:
        json.dump(norm_stats, f)

    # Create class folders with paired .pt files
    for cls_idx, cls_name in enumerate(class_names):
        cls_dir = tmp_path / cls_name
        cls_dir.mkdir()
        for i in range(10):
            rgb = torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8)
            depth = torch.randint(0, 10000, (1, 256, 256), dtype=torch.uint16)
            # Set some pixels to 0 (missing data)
            depth[0, :5, :5] = 0
            torch.save(rgb, cls_dir / f'obj_{cls_idx:03d}_f{i:03d}_rgb.pt')
            torch.save(depth, cls_dir / f'obj_{cls_idx:03d}_f{i:03d}_depth.pt')

    return tmp_path, class_names, norm_stats


@pytest.fixture
def fake_dataset_indexed_classnames(tmp_path):
    """Like fake_dataset but with '0: chair' format in class_names.txt."""
    class_names = ['chair', 'sofa', 'table']
    with open(tmp_path / 'class_names.txt', 'w') as f:
        for i, name in enumerate(class_names):
            f.write(f"{i}: {name}\n")

    norm_stats = {
        'rgb_mean': [0.485, 0.456, 0.406],
        'rgb_std': [0.229, 0.224, 0.225],
        'depth_mean': [2.5],
        'depth_std': [1.2],
    }
    with open(tmp_path / 'norm_stats.json', 'w') as f:
        json.dump(norm_stats, f)

    for cls_idx, cls_name in enumerate(class_names):
        cls_dir = tmp_path / cls_name
        cls_dir.mkdir()
        for i in range(4):
            rgb = torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8)
            depth = torch.randint(100, 5000, (1, 256, 256), dtype=torch.uint16)
            torch.save(rgb, cls_dir / f'obj_{cls_idx:03d}_f{i:03d}_rgb.pt')
            torch.save(depth, cls_dir / f'obj_{cls_idx:03d}_f{i:03d}_depth.pt')

    return tmp_path, class_names
```

#### Test Functions

```python
class TestLoadClassNames:
    """Tests for _load_class_names helper."""

    def test_plain_format(self, fake_dataset):
        """Plain 'chair' format lines are parsed correctly."""
        data_root, expected_names, _ = fake_dataset
        names = _load_class_names(str(data_root))
        assert names == expected_names

    def test_indexed_format(self, fake_dataset_indexed_classnames):
        """'0: chair' format lines are parsed correctly."""
        data_root, expected_names = fake_dataset_indexed_classnames
        names = _load_class_names(str(data_root))
        assert names == expected_names

    def test_missing_file_raises(self, tmp_path):
        """FileNotFoundError raised when class_names.txt missing."""
        with pytest.raises(FileNotFoundError, match="class_names.txt"):
            _load_class_names(str(tmp_path))

    def test_empty_lines_skipped(self, tmp_path):
        """Empty lines in class_names.txt are skipped."""
        with open(tmp_path / 'class_names.txt', 'w') as f:
            f.write("chair\n\nsofa\n\n")
        names = _load_class_names(str(tmp_path))
        assert names == ['chair', 'sofa']


class TestLoadNormStats:
    """Tests for _load_norm_stats helper."""

    def test_loads_correctly(self, fake_dataset):
        """Stats loaded match what was written."""
        data_root, _, expected_stats = fake_dataset
        stats = _load_norm_stats(str(data_root))
        assert stats == expected_stats

    def test_missing_file_raises(self, tmp_path):
        """FileNotFoundError raised when norm_stats.json missing."""
        with pytest.raises(FileNotFoundError, match="norm_stats.json"):
            _load_norm_stats(str(tmp_path))

    def test_value_types(self, fake_dataset):
        """Verify norm_stats values are lists of floats with correct lengths."""
        data_root, _, _ = fake_dataset
        stats = _load_norm_stats(str(data_root))
        assert isinstance(stats['rgb_mean'], list) and len(stats['rgb_mean']) == 3
        assert isinstance(stats['rgb_std'], list) and len(stats['rgb_std']) == 3
        assert isinstance(stats['depth_mean'], list) and len(stats['depth_mean']) == 1
        assert isinstance(stats['depth_std'], list) and len(stats['depth_std']) == 1
        for key in stats:
            assert all(isinstance(v, float) for v in stats[key])


class TestDiscoverSamples:
    """Tests for _discover_samples helper."""

    def test_discovers_all_paired_samples(self, fake_dataset):
        """All 30 samples (3 classes x 10) discovered with correct labels."""
        data_root, class_names, _ = fake_dataset
        samples = _discover_samples(str(data_root), class_names)
        assert len(samples) == 30
        labels = [s[2] for s in samples]
        assert labels.count(0) == 10  # chair
        assert labels.count(1) == 10  # sofa
        assert labels.count(2) == 10  # table

    def test_each_sample_has_valid_paths(self, fake_dataset):
        """Each (rgb_path, depth_path, label) has existing files."""
        data_root, class_names, _ = fake_dataset
        samples = _discover_samples(str(data_root), class_names)
        for rgb_path, depth_path, label in samples:
            assert os.path.exists(rgb_path)
            assert os.path.exists(depth_path)
            assert 0 <= label < len(class_names)

    def test_unpaired_rgb_raises(self, fake_dataset):
        """ValueError raised when RGB file has no matching depth."""
        data_root, class_names, _ = fake_dataset
        # Add orphan RGB file
        orphan = data_root / 'chair' / 'orphan_f000_rgb.pt'
        torch.save(torch.zeros(3, 256, 256, dtype=torch.uint8), orphan)
        with pytest.raises(ValueError, match="Unpaired files"):
            _discover_samples(str(data_root), class_names)

    def test_unpaired_depth_raises(self, fake_dataset):
        """ValueError raised when depth file has no matching RGB."""
        data_root, class_names, _ = fake_dataset
        orphan = data_root / 'chair' / 'orphan_f000_depth.pt'
        torch.save(torch.zeros(1, 256, 256, dtype=torch.uint16), orphan)
        with pytest.raises(ValueError, match="Unpaired files"):
            _discover_samples(str(data_root), class_names)

    def test_unknown_folder_skipped(self, fake_dataset):
        """Folders not in class_names.txt are skipped silently."""
        data_root, class_names, _ = fake_dataset
        extra = data_root / 'unknown_class'
        extra.mkdir()
        torch.save(torch.zeros(3, 256, 256, dtype=torch.uint8), extra / 'x_f000_rgb.pt')
        torch.save(torch.zeros(1, 256, 256, dtype=torch.uint16), extra / 'x_f000_depth.pt')
        samples = _discover_samples(str(data_root), class_names)
        assert len(samples) == 30  # unchanged

    def test_deterministic_ordering(self, fake_dataset):
        """Two calls return identical ordering."""
        data_root, class_names, _ = fake_dataset
        s1 = _discover_samples(str(data_root), class_names)
        s2 = _discover_samples(str(data_root), class_names)
        assert s1 == s2


class TestOmniPretrainDataset:
    """Tests for OmniPretrainDataset class."""

    def test_init_train(self, fake_dataset):
        """Train dataset initializes with correct counts."""
        ds = _make_dataset(fake_dataset, split='train')
        assert len(ds) == 30
        assert ds.num_classes == 3
        assert ds.split == 'train'

    def test_init_val(self, fake_dataset):
        """Val dataset initializes correctly."""
        ds = _make_dataset(fake_dataset, split='val')
        assert ds.split == 'val'

    def test_invalid_split_raises(self, fake_dataset):
        """ValueError on invalid split name."""
        with pytest.raises(ValueError, match="split must be one of"):
            _make_dataset(fake_dataset, split='test')

    def test_getitem_shapes_train(self, fake_dataset):
        """__getitem__ returns correct shapes and types for train."""
        ds = _make_dataset(fake_dataset, split='train', crop_size=224)
        rgb, depth, label = ds[0]
        assert rgb.shape == (3, 224, 224)
        assert depth.shape == (1, 224, 224)
        assert rgb.dtype == torch.float32
        assert depth.dtype == torch.float32
        assert isinstance(label, int)
        assert 0 <= label < 3

    def test_getitem_shapes_val(self, fake_dataset):
        """__getitem__ returns correct shapes and types for val."""
        ds = _make_dataset(fake_dataset, split='val', crop_size=224)
        rgb, depth, label = ds[0]
        assert rgb.shape == (3, 224, 224)
        assert depth.shape == (1, 224, 224)
        assert rgb.dtype == torch.float32
        assert depth.dtype == torch.float32

    def test_depth_conversion_to_meters(self, fake_dataset):
        """Depth uint16 mm is converted to float32 meters, not /65535 normalized."""
        ds = _make_dataset(fake_dataset, split='val', normalize=False)
        rgb, depth, label = ds[0]
        # Original is randint(0, 10000) mm = 0-10 meters
        # If conversion were /65535 (wrong), max would be <= 0.153
        # If conversion is /1000 (correct), max should be in 0.5-10.0 range
        assert depth.max() > 0.5, (
            f"depth.max()={depth.max():.4f} — likely /65535 instead of /1000"
        )
        assert depth.max() <= 10.0

    def test_depth_mm_conversion_numeric(self, tmp_path):
        """Verify specific mm value converts to correct meters value."""
        class_names = ['chair']
        depth_mean = 2.5
        norm_stats = {
            'rgb_mean': [0.5, 0.5, 0.5],
            'rgb_std': [0.2, 0.2, 0.2],
            'depth_mean': [depth_mean],
            'depth_std': [1.0],
        }

        with open(tmp_path / 'class_names.txt', 'w') as f:
            f.write("chair\n")
        with open(tmp_path / 'norm_stats.json', 'w') as f:
            json.dump(norm_stats, f)

        cls_dir = tmp_path / 'chair'
        cls_dir.mkdir()
        rgb = torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8)
        # All pixels set to 3000mm
        depth = torch.full((1, 256, 256), 3000, dtype=torch.uint16)
        torch.save(rgb, cls_dir / 'obj_000_f000_rgb.pt')
        torch.save(depth, cls_dir / 'obj_000_f000_depth.pt')

        samples = _discover_samples(str(tmp_path), class_names)
        ds = OmniPretrainDataset(
            data_root=str(tmp_path),
            split='val',
            samples=samples,
            class_names=class_names,
            norm_stats=norm_stats,
            normalize=False,
        )
        _, out_depth, _ = ds[0]
        # CenterCrop 256->224 starts at (16, 16), center pixel (128,128) -> (112,112)
        # All non-zero pixels = 3000mm = 3.0m, no sentinel replacement needed
        assert out_depth[0, 112, 112].item() == pytest.approx(3.0)

    def test_sentinel_replaced_with_mean(self, tmp_path):
        """0-sentinel pixels at known positions are replaced with depth_mean.

        Creates a depth tensor with zeros at center position [0, 128, 128].
        On the val path with normalize=False, CenterCrop from 256->224
        crops starting at offset (16, 16). So pixel [0, 128, 128] in the
        original maps to [0, 128-16, 128-16] = [0, 112, 112] in the crop.
        That pixel should equal depth_mean after sentinel replacement.
        """
        class_names = ['chair']
        depth_mean = 2.5
        norm_stats = {
            'rgb_mean': [0.5, 0.5, 0.5],
            'rgb_std': [0.2, 0.2, 0.2],
            'depth_mean': [depth_mean],
            'depth_std': [1.0],
        }

        with open(tmp_path / 'class_names.txt', 'w') as f:
            f.write("chair\n")
        with open(tmp_path / 'norm_stats.json', 'w') as f:
            json.dump(norm_stats, f)

        cls_dir = tmp_path / 'chair'
        cls_dir.mkdir()
        rgb = torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8)
        # All non-zero depth except at center
        depth = torch.full((1, 256, 256), 5000, dtype=torch.uint16)
        depth[0, 128, 128] = 0  # sentinel at known position
        torch.save(rgb, cls_dir / 'obj_000_f000_rgb.pt')
        torch.save(depth, cls_dir / 'obj_000_f000_depth.pt')

        samples = _discover_samples(str(tmp_path), class_names)
        ds = OmniPretrainDataset(
            data_root=str(tmp_path),
            split='val',
            samples=samples,
            class_names=class_names,
            norm_stats=norm_stats,
            normalize=False,
        )
        _, out_depth, _ = ds[0]
        # CenterCrop 256->224 starts at (16, 16), so original (128, 128) -> (112, 112)
        assert out_depth[0, 112, 112] == pytest.approx(depth_mean)

    def test_normalized_output_range(self, fake_dataset):
        """With normalize=True, output is not in [0,1] range (standardized)."""
        ds = _make_dataset(fake_dataset, split='val', normalize=True)
        rgb, depth, label = ds[0]
        # Normalized data should have values outside [0,1]
        # (mean-subtracted, std-divided)
        assert rgb.min() < 0.0 or rgb.max() > 1.0

    def test_unnormalized_rgb_range(self, fake_dataset):
        """With normalize=False, RGB is in [0,1] range."""
        ds = _make_dataset(fake_dataset, split='val', normalize=False)
        rgb, depth, label = ds[0]
        assert rgb.min() >= 0.0
        assert rgb.max() <= 1.0

    def test_labels_are_list_of_int(self, fake_dataset):
        """self.labels is list[int]."""
        ds = _make_dataset(fake_dataset, split='train')
        assert isinstance(ds.labels, list)
        assert all(isinstance(l, int) for l in ds.labels)

    def test_get_class_weights_shape(self, fake_dataset):
        """get_class_weights returns [num_classes] float32 tensor."""
        ds = _make_dataset(fake_dataset, split='train')
        weights = ds.get_class_weights()
        assert weights.shape == (3,)
        assert weights.dtype == torch.float32
        # With equal class distribution, weights should be ~1.0
        assert torch.allclose(weights, torch.ones(3), atol=0.01)

    def test_get_sample_weights_shape(self, fake_dataset):
        """get_sample_weights returns [num_samples] float64 tensor."""
        ds = _make_dataset(fake_dataset, split='train')
        weights = ds.get_sample_weights()
        assert weights.shape == (30,)
        assert weights.dtype == torch.float64
        assert (weights > 0).all()

    def test_get_class_distribution(self, fake_dataset):
        """get_class_distribution returns correct counts."""
        ds = _make_dataset(fake_dataset, split='train')
        dist = ds.get_class_distribution()
        assert set(dist.keys()) == {'chair', 'sofa', 'table'}
        for name in ['chair', 'sofa', 'table']:
            assert dist[name]['count'] == 10
            assert abs(dist[name]['percentage'] - 100.0 / 3) < 0.1

    def test_get_norm_stats(self, fake_dataset):
        """get_norm_stats returns the loaded dict."""
        data_root, _, norm_stats = fake_dataset
        ds = _make_dataset(fake_dataset, split='train')
        assert ds.get_norm_stats() == norm_stats

    def test_zero_mask_pixel_correspondence_train(self, fake_dataset):
        """Sentinel pixels are replaced with depth_mean even on train path.

        Uses crop_size=256 to skip RandomCrop, ensuring sentinel pixels
        at known positions survive into the output.
        """
        data_root, class_names, norm_stats = fake_dataset
        samples = _discover_samples(str(data_root), class_names)
        ds = OmniPretrainDataset(
            data_root=str(data_root),
            split='train',
            samples=samples,
            class_names=class_names,
            norm_stats=norm_stats,
            crop_size=256,
            normalize=False,
        )
        depth_mean = norm_stats['depth_mean'][0]

        # Run multiple iterations to exercise augmentation paths
        np.random.seed(12345)
        for _ in range(5):
            _, out_depth, _ = ds[0]
            # The fixture has zeros in top-left 5x5. With crop_size=256
            # no cropping happens, so those pixels always survive.
            # After sentinel replacement, they should equal depth_mean.
            assert out_depth[0, 0, 0].item() == pytest.approx(depth_mean)
            assert out_depth[0, 4, 4].item() == pytest.approx(depth_mean)

    def test_scale_jitter_shape_invariance(self, fake_dataset):
        """Depth scale jitter changes values but not spatial dimensions.

        Scale jitter is a value multiply (not spatial resize), so shape
        must remain [1, crop_size, crop_size] regardless.
        """
        ds = _make_dataset(fake_dataset, split='train', normalize=False)
        np.random.seed(42)
        for _ in range(10):
            _, depth, _ = ds[0]
            assert depth.shape == (1, 224, 224)


class TestHoleDropout:
    """Tests for _apply_hole_dropout method."""

    def test_creates_zero_regions(self, fake_dataset):
        """Hole dropout sets some pixels to 0."""
        ds = _make_dataset(fake_dataset, split='train')
        depth = torch.ones(1, 224, 224, dtype=torch.float32) * 2.5
        result = ds._apply_hole_dropout(depth)
        assert (result == 0.0).any()

    def test_preserves_shape_dtype(self, fake_dataset):
        """Output has same shape and dtype as input."""
        ds = _make_dataset(fake_dataset, split='train')
        depth = torch.ones(1, 224, 224, dtype=torch.float32) * 2.5
        result = ds._apply_hole_dropout(depth)
        assert result.shape == (1, 224, 224)
        assert result.dtype == torch.float32

    def test_modifies_in_place(self, fake_dataset):
        """_apply_hole_dropout modifies the input tensor in-place."""
        ds = _make_dataset(fake_dataset, split='train')
        depth = torch.ones(1, 224, 224, dtype=torch.float32) * 2.5
        result = ds._apply_hole_dropout(depth)
        assert result is depth  # same object


class TestGetOmnipretrainDataloaders:
    """Tests for get_omnipretrain_dataloaders factory."""

    def test_returns_three_elements(self, fake_dataset):
        """Factory returns (train_loader, val_loader, num_classes)."""
        data_root, _, _ = fake_dataset
        result = get_omnipretrain_dataloaders(
            data_root=str(data_root),
            batch_size=4,
            num_workers=0,
            seed=42,
        )
        assert len(result) == 3
        train_loader, val_loader, num_classes = result
        assert isinstance(num_classes, int)
        assert num_classes == 3

    def test_returns_four_elements_with_class_weights(self, fake_dataset):
        """Factory returns 4-tuple when use_class_weights=True."""
        data_root, _, _ = fake_dataset
        result = get_omnipretrain_dataloaders(
            data_root=str(data_root),
            batch_size=4,
            num_workers=0,
            seed=42,
            use_class_weights=True,
        )
        assert len(result) == 4
        _, _, num_classes, class_weights = result
        assert class_weights.shape == (num_classes,)

    def test_batch_shapes(self, fake_dataset):
        """Batches from loader have correct shapes."""
        data_root, _, _ = fake_dataset
        train_loader, val_loader, num_classes = get_omnipretrain_dataloaders(
            data_root=str(data_root),
            batch_size=4,
            num_workers=0,
            seed=42,
        )
        rgb, depth, labels = next(iter(train_loader))
        assert rgb.shape == (4, 3, 224, 224)
        assert depth.shape == (4, 1, 224, 224)
        assert labels.shape == (4,)

    def test_val_no_augmentation_deterministic(self, fake_dataset):
        """Val loader produces identical outputs on repeated iteration."""
        data_root, _, _ = fake_dataset
        _, val_loader, _ = get_omnipretrain_dataloaders(
            data_root=str(data_root),
            batch_size=4,
            num_workers=0,
            seed=42,
        )
        batch1 = next(iter(val_loader))
        batch2 = next(iter(val_loader))
        assert torch.equal(batch1[0], batch2[0])  # rgb identical
        assert torch.equal(batch1[1], batch2[1])  # depth identical

    def test_stratified_split(self, fake_dataset):
        """Train and val both contain samples from all classes."""
        data_root, _, _ = fake_dataset
        train_loader, val_loader, _ = get_omnipretrain_dataloaders(
            data_root=str(data_root),
            batch_size=30,  # all at once
            num_workers=0,
            seed=42,
        )
        # Collect all train labels
        train_labels = set()
        for _, _, labels in train_loader:
            train_labels.update(labels.tolist())
        assert len(train_labels) == 3

        val_labels = set()
        for _, _, labels in val_loader:
            val_labels.update(labels.tolist())
        assert len(val_labels) == 3

    def test_num_workers_zero_no_crash(self, fake_dataset):
        """num_workers=0 works without prefetch_factor error."""
        data_root, _, _ = fake_dataset
        train_loader, val_loader, _ = get_omnipretrain_dataloaders(
            data_root=str(data_root),
            batch_size=4,
            num_workers=0,
            seed=42,
        )
        # Just iterate once to verify no crash
        next(iter(train_loader))
        next(iter(val_loader))

    def test_reproducible_split(self, fake_dataset):
        """Same seed produces same split."""
        data_root, _, _ = fake_dataset
        _, val1, _ = get_omnipretrain_dataloaders(
            data_root=str(data_root), batch_size=30, num_workers=0, seed=42
        )
        _, val2, _ = get_omnipretrain_dataloaders(
            data_root=str(data_root), batch_size=30, num_workers=0, seed=42
        )
        b1 = next(iter(val1))
        b2 = next(iter(val2))
        assert torch.equal(b1[0], b2[0])
        assert torch.equal(b1[2], b2[2])

    def test_empty_dataset_raises(self, tmp_path):
        """ValueError raised when no samples found."""
        with open(tmp_path / 'class_names.txt', 'w') as f:
            f.write("chair\n")
        with open(tmp_path / 'norm_stats.json', 'w') as f:
            json.dump({
                'rgb_mean': [0.5, 0.5, 0.5], 'rgb_std': [0.2, 0.2, 0.2],
                'depth_mean': [2.0], 'depth_std': [1.0],
            }, f)
        with pytest.raises(ValueError, match="No samples found"):
            get_omnipretrain_dataloaders(data_root=str(tmp_path), num_workers=0)

    def test_single_sample_class_does_not_crash(self, tmp_path):
        """Class with 1 sample falls back to non-stratified split."""
        class_names = ['chair', 'sofa', 'table']
        norm_stats = {
            'rgb_mean': [0.5, 0.5, 0.5],
            'rgb_std': [0.2, 0.2, 0.2],
            'depth_mean': [2.5],
            'depth_std': [1.0],
        }

        with open(tmp_path / 'class_names.txt', 'w') as f:
            for name in class_names:
                f.write(f"{name}\n")
        with open(tmp_path / 'norm_stats.json', 'w') as f:
            json.dump(norm_stats, f)

        # chair: 1 sample, sofa: 10 samples, table: 10 samples
        for cls_idx, cls_name in enumerate(class_names):
            cls_dir = tmp_path / cls_name
            cls_dir.mkdir()
            count = 1 if cls_name == 'chair' else 10
            for i in range(count):
                rgb = torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8)
                depth = torch.randint(100, 5000, (1, 256, 256), dtype=torch.uint16)
                torch.save(rgb, cls_dir / f'obj_{cls_idx:03d}_f{i:03d}_rgb.pt')
                torch.save(depth, cls_dir / f'obj_{cls_idx:03d}_f{i:03d}_depth.pt')

        # Should succeed with a warning about fallback to non-stratified split
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = get_omnipretrain_dataloaders(
                data_root=str(tmp_path),
                batch_size=4,
                num_workers=0,
                seed=42,
            )
            # Check that we got a warning about the fallback
            stratified_warnings = [
                x for x in w
                if "Stratified split failed" in str(x.message)
            ]
            assert len(stratified_warnings) == 1

        assert len(result) == 3
        train_loader, val_loader, num_classes = result
        assert num_classes == 3

    def test_factory_stratified_false(self, fake_dataset):
        """stratified=False uses shuffle instead of WeightedRandomSampler."""
        data_root, _, _ = fake_dataset
        result = get_omnipretrain_dataloaders(
            data_root=str(data_root),
            batch_size=4,
            num_workers=0,
            seed=42,
            stratified=False,
        )
        assert len(result) == 3
        train_loader, val_loader, num_classes = result
        assert num_classes == 3
        # Verify train_loader does NOT have a sampler (uses shuffle instead)
        assert train_loader.sampler.__class__.__name__ != 'WeightedRandomSampler'
        # Should still produce valid batches
        rgb, depth, labels = next(iter(train_loader))
        assert rgb.shape == (4, 3, 224, 224)
```

## 6. Evaluation Criteria

### Acceptance Criteria Checklist

1. `_load_class_names` parses both `"chair"` and `"0: chair"` formats correctly
2. `_load_class_names` raises `FileNotFoundError` when file is missing
3. `_load_norm_stats` loads JSON and returns correct dict with documented types (`rgb_mean: list[float]` 3 elements, `rgb_std: list[float]` 3 elements, `depth_mean: list[float]` 1 element in meters, `depth_std: list[float]` 1 element in meters)
4. `_discover_samples` finds all paired files and returns sorted (rgb_path, depth_path, label) tuples
5. `_discover_samples` raises `ValueError` on unpaired files
6. `_discover_samples` ignores folders not in class_names.txt
7. `OmniPretrainDataset.__getitem__` returns `(float32 [3,224,224], float32 [1,224,224], int)`
8. Depth is converted from uint16 mm to float32 meters
9. 0-sentinel pixels are preserved through augmentation and replaced with depth_mean before normalization
10. Train augmentations include: synchronized flip, random crop, color jitter, blur, grayscale, depth scale jitter, depth appearance aug, hole dropout, post-norm erasing
11. Val uses CenterCrop only, no augmentations
12. Synchronized flip applies identically to RGB and depth
13. Depth scale jitter is a value multiply (not spatial resize) -- shape is invariant
14. Hole dropout zeros out rectangular patches before normalization, modifies depth in-place
15. `get_sample_weights()` returns `float64` tensor of shape `[num_samples]`
16. `get_class_weights()` returns `float32` tensor of shape `[num_classes]`
17. Factory returns `(train_loader, val_loader, num_classes)` -- num_classes is third element
18. Factory returns 4-tuple with class_weights when `use_class_weights=True`
19. `num_workers=0` works without crash (prefetch_factor=None)
20. Train/val split is stratified and reproducible with seed
21. All imports at top of file, grouped correctly (stdlib / third-party / local)
22. `__init__.py` exports new symbols
23. OmniPretrain-specific augmentation constants live in `augmentation_config.py`
24. All tests pass
25. `zero_mask` flip uses `tensor.flip(-1)` not `F2.horizontal_flip` (cross-version bool safety)
26. `zero_mask` crop uses raw tensor slicing not `F2.crop` (cross-version bool safety)
27. Loaded tensors validated with shape assertions after `torch.load`
28. Depth appearance augmentation explicitly handles zero-masked pixels (only transforms `~zero_mask` pixels)
29. Stratified split has try/except fallback to non-stratified for rare classes (< 2 samples)
30. `_WorkerInitFn` seeds `np.random`, `random`, AND `torch.manual_seed`
31. `stratified=False` factory parameter uses `shuffle=True` instead of `WeightedRandomSampler`
32. `WeightedRandomSampler` generator is created locally (not shared)

### Performance expectations
- `__init__` (including file discovery): < 5 seconds for ~90 categories, ~50K files
- `__getitem__`: < 50ms per sample (dominated by torch.load I/O)
- No memory leak from per-sample file loading

## 7. Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| `torch.load` per sample is slow compared to monolithic tensors | Training throughput limited by I/O | Use `num_workers >= 4` to overlap loading with compute. If too slow, add a future caching layer or convert to WebDataset format. |
| Large number of small files causes filesystem strain | Slow `_discover_samples` or `__getitem__` | `_discover_samples` uses `os.listdir` (fast). Per-file `torch.load` is the bottleneck, mitigated by workers. |
| `sklearn` not installed | Import error at module load | sklearn is a standard ML dependency; add to requirements if missing. Check with `pip list`. |
| Stratified split fails with very rare classes (< 2 samples) | `train_test_split` raises error | Wrapped in try/except. Falls back to non-stratified split with `warnings.warn()`. Test `test_single_sample_class_does_not_crash` verifies this. |
| Depth appearance augmentation produces negative depth values | Nonsensical depth | The per-sample [0,1] normalization + clamp(0,1) + map-back ensures depth stays in valid range. Only valid (non-zero-mask) pixels are transformed. |
| Hole dropout produces too many missing pixels | Training signal degraded | Parameters are conservative (3-8 holes of 5-20px each on 224x224 = 0.3-7% area). Gated by probability. |
| zero_mask not flipped/cropped with depth | Sentinel restoration applies to wrong pixels | zero_mask uses raw tensor ops (`flip(-1)`, slice indexing) that are guaranteed to work on bool tensors across all torchvision versions. |
| `F2.horizontal_flip` / `F2.crop` breaks on bool tensors in future torchvision | Silent data corruption or crash | Avoided entirely by using `tensor.flip(-1)` and `tensor[:, i:i+h, j:j+w]` for zero_mask. |
| Loaded tensor has unexpected shape (data corruption) | Silent wrong results or crash | Assertions after `torch.load` validate RGB is `[3, H, W]` and depth is `[1, H, W]`, failing fast with clear error message. |
| `torch.Generator` shared between sampler and other uses | Non-reproducible sampling | Generator is created locally inline inside `WeightedRandomSampler` constructor, not shared. |

### Rollback strategy
- Single new file (`omnipretrain_dataset.py`) + minimal changes to `__init__.py` and `augmentation_config.py`
- Revert is a single `git revert` of the commit
- No existing functionality is modified

---

## Design Decision Notes (Reviewer Q&A)

**Q: Where should the new augmentation constants live?**
A: In `augmentation_config.py`, consistent with the existing `BASE_*` pattern. This is reflected in sections 1, 3, and 4.

**Q: Why duplicate `_load_norm_stats` instead of sharing?**
A: Intentional. Keeps each dataset module self-contained with zero coupling. Not worth a shared utility for a 10-line function with only 2 consumers.

## Gap Analysis Fixes Applied

This plan incorporates 8 fixes from gap analysis review:

1. **FIX 1 (CRITICAL)**: `zero_mask` flip/crop now uses raw tensor ops (`flip(-1)`, slice `[:, i:i+h, j:j+w]`) instead of `F2.horizontal_flip`/`F2.crop` which are not guaranteed to work on `torch.bool` tensors across torchvision versions.

2. **FIX 2**: Added `assert` statements after `torch.load` to validate RGB shape is `[3, H, W]` and depth shape is `[1, H, W]`. Does not assert specific dtype since `torch` may not support `uint16` natively and `.float() / 1000.0` works on any integer dtype.

3. **FIX 3**: Depth appearance augmentation now explicitly handles zero-mask pixels. `depth_01` is initialized to `torch.zeros_like(depth)`, only `~zero_mask` pixels are normalized to [0,1]. After augmentation, only `~zero_mask` pixels are mapped back. Zero-mask pixels remain at 0.0 throughout.

4. **FIX 4**: Stratified split wrapped in `try/except ValueError` with fallback to non-stratified `train_test_split`. Emits `warnings.warn()` on fallback. `warnings` added to stdlib imports.

5. **FIX 5**: `_WorkerInitFn.__call__` now also calls `torch.manual_seed(worker_seed)` in addition to `np.random.seed` and `random.seed`.

6. **FIX 6**: Added `stratified: bool = True` parameter to `get_omnipretrain_dataloaders`. When `False`, uses `shuffle=True` instead of `WeightedRandomSampler`. The `torch.Generator` for `WeightedRandomSampler` is created locally inline (not shared).

7. **FIX 7**: Added three new test functions: `test_depth_mm_conversion_numeric` (verifies 3000mm -> 3.0m), `test_single_sample_class_does_not_crash` (verifies fallback warning), `test_factory_stratified_false` (verifies shuffle mode).

8. **FIX 8**: `test_zero_mask_pixel_correspondence_train` now uses `crop_size=256` to guarantee sentinel pixels survive (no cropping), and asserts specific pixel values equal `depth_mean`.

---

**Relevant file paths:**
- `/Users/gclinger/Documents/projects/Multi-Stream-Neural-Networks/docs/plans/2026-03-20-omnipretrain-dataloader-plan.md` -- the plan to be updated
- `/Users/gclinger/Documents/projects/Multi-Stream-Neural-Networks/src/data_utils/omnipretrain_dataset.py` -- new file to create
- `/Users/gclinger/Documents/projects/Multi-Stream-Neural-Networks/src/data_utils/__init__.py` -- to modify with new exports
- `/Users/gclinger/Documents/projects/Multi-Stream-Neural-Networks/src/training/augmentation_config.py` -- to modify with new constants
- `/Users/gclinger/Documents/projects/Multi-Stream-Neural-Networks/tests/src/data_utils/test_omnipretrain_dataset.py` -- new test file to create

I do not have a file-writing tool available in this environment. The complete updated plan is provided above in full -- please save it to `/Users/gclinger/Documents/projects/Multi-Stream-Neural-Networks/docs/plans/2026-03-20-omnipretrain-dataloader-plan.md`, replacing the existing content.