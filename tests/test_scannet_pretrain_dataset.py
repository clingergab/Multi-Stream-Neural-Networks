"""Synthetic-data tests for ScanNetPretrainDataset.

These tests construct a tiny tempdir-based fake preprocessed ScanNet dir
with a few hand-crafted samples to exercise the dataset class, INCLUDING
the new ``use_hha=True`` path. They do NOT depend on the real ScanNet
preprocessed tarball -- that lives only on Drive and the real-data shape +
range contract is enforced by the preprocess notebook's validation gate
(see Phase 3.6 in the plan), not pytest.
"""

import json
import os
import tempfile

import numpy as np
import pytest
import torch

from src.data_utils.scannet_pretrain_dataset import (
    ScanNetPretrainDataset,
    _discover_samples,
    _load_class_names,
    _load_norm_stats,
    get_scannet_pretrain_dataloaders,
)


CLASSES = ["bathroom", "bedroom"]
H = W = 256
CROP = 224


def _write_sample(folder: str, scene_id: str, frame_idx: int, *, hha: bool):
    rgb = (np.random.rand(3, H, W) * 255).astype(np.uint8)
    torch.save(torch.from_numpy(rgb), os.path.join(folder, f"{scene_id}_f{frame_idx:05d}_rgb.pt"))
    if hha:
        # 3-channel HHA float16 in canonical ranges (with some NaN sprinkled).
        h = np.empty((3, H, W), dtype=np.float32)
        h[0] = np.random.uniform(0.1, 1.5, size=(H, W))    # disparity
        h[1] = np.random.uniform(-0.1, 2.5, size=(H, W))   # height
        h[2] = np.random.uniform(0.0, 180.0, size=(H, W))  # angle
        # Sprinkle 5% NaN
        nan_mask = np.random.rand(H, W) < 0.05
        h[:, nan_mask] = np.nan
        torch.save(torch.from_numpy(h.astype(np.float16)),
                   os.path.join(folder, f"{scene_id}_f{frame_idx:05d}_hha.pt"))
    else:
        depth = (np.random.uniform(500, 5000, size=(1, H, W))).astype(np.uint16)
        # Sprinkle 5% missing pixels (zero sentinel)
        missing = np.random.rand(H, W) < 0.05
        depth[0, missing] = 0
        torch.save(torch.from_numpy(depth),
                   os.path.join(folder, f"{scene_id}_f{frame_idx:05d}_depth.pt"))


def _build_fake_dataset(root: str, *, hha: bool):
    os.makedirs(root, exist_ok=True)
    # class_names.txt
    with open(os.path.join(root, "class_names.txt"), "w") as f:
        for i, c in enumerate(CLASSES):
            f.write(f"{i}: {c}\n")
    # norm_stats.json
    if hha:
        norm_stats = {
            "rgb_mean": [0.5, 0.5, 0.5],
            "rgb_std": [0.25, 0.25, 0.25],
            "hha_mean": [0.5, 0.7, 60.0],
            "hha_std": [0.3, 0.6, 30.0],
        }
    else:
        norm_stats = {
            "rgb_mean": [0.5, 0.5, 0.5],
            "rgb_std": [0.25, 0.25, 0.25],
            "depth_mean": [2.5],
            "depth_std": [1.5],
        }
    with open(os.path.join(root, "norm_stats.json"), "w") as f:
        json.dump(norm_stats, f)
    # train/<class>/samples
    for split in ["train", "val"]:
        for cls_idx, cls in enumerate(CLASSES):
            folder = os.path.join(root, split, cls)
            os.makedirs(folder, exist_ok=True)
            for fi in range(2):  # 2 samples per class per split
                _write_sample(folder, f"scene{cls_idx:04d}_{fi:02d}", 0, hha=hha)


def _construct(root: str, split: str, *, hha: bool):
    class_names = _load_class_names(root)
    norm_stats = _load_norm_stats(root)
    samples = _discover_samples(os.path.join(root, split), class_names, use_hha=hha)
    return ScanNetPretrainDataset(
        data_root=root,
        split=split,
        samples=samples,
        class_names=class_names,
        norm_stats=norm_stats,
        crop_size=CROP,
        normalize=True,
        use_hha=hha,
    )


# ---------------------------------------------------------------------------
# Raw-depth path (existing behavior, regression coverage)
# ---------------------------------------------------------------------------

def test_raw_depth_shape_contract():
    np.random.seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        _build_fake_dataset(tmp, hha=False)
        ds = _construct(tmp, "train", hha=False)
        assert len(ds) == 4  # 2 classes x 2 samples
        rgb, depth, label = ds[0]
        assert rgb.shape == (3, CROP, CROP)
        assert depth.shape == (1, CROP, CROP)
        assert rgb.dtype == torch.float32
        assert depth.dtype == torch.float32
        assert not torch.isnan(rgb).any()
        assert not torch.isnan(depth).any()
        assert label in (0, 1)


# ---------------------------------------------------------------------------
# HHA path (new behavior)
# ---------------------------------------------------------------------------

def test_hha_shape_contract():
    """use_hha=True returns (3, crop, crop) for the depth slot."""
    np.random.seed(1)
    with tempfile.TemporaryDirectory() as tmp:
        _build_fake_dataset(tmp, hha=True)
        ds = _construct(tmp, "train", hha=True)
        assert len(ds) == 4
        rgb, hha, label = ds[0]
        assert rgb.shape == (3, CROP, CROP)
        assert hha.shape == (3, CROP, CROP)  # <-- 3 channels, not 1
        assert rgb.dtype == torch.float32
        assert hha.dtype == torch.float32


def test_hha_no_nan_survives_normalization():
    """NaN sentinel pixels must be replaced with per-channel mean before the
    model ever sees them (whether from on-disk NaN or from hole-dropout)."""
    np.random.seed(2)
    with tempfile.TemporaryDirectory() as tmp:
        _build_fake_dataset(tmp, hha=True)
        ds = _construct(tmp, "train", hha=True)
        # Iterate enough samples (with random aug) that hole dropout should fire at least once.
        for _ in range(8):
            for i in range(len(ds)):
                _, hha, _ = ds[i]
                assert not torch.isnan(hha).any(), \
                    "NaN survived sentinel-replacement / normalization"


def test_hha_value_range_after_normalization():
    """Catches silent statistics bugs (wrong norm_stats, stale tarball,
    Welford precision issues) — values out of the expected normalized range
    even though shapes are fine."""
    np.random.seed(3)
    with tempfile.TemporaryDirectory() as tmp:
        _build_fake_dataset(tmp, hha=True)
        ds = _construct(tmp, "train", hha=True)
        all_hha = []
        for i in range(len(ds)):
            _, hha, _ = ds[i]
            all_hha.append(hha)
        stacked = torch.stack(all_hha, dim=0)
        # 99th-percentile-abs should sit comfortably below 5 after Z-score
        # normalization with appropriate norm_stats.
        q99 = stacked.abs().quantile(0.99).item()
        assert q99 < 5.0, (
            f"HHA normalized values out of range: 99th-percentile abs={q99:.3f}"
        )


def test_hha_val_split_no_augmentation():
    """val split should center-crop and normalize, no random aug."""
    np.random.seed(4)
    with tempfile.TemporaryDirectory() as tmp:
        _build_fake_dataset(tmp, hha=True)
        ds = _construct(tmp, "val", hha=True)
        # Run twice — should be deterministic since no augmentation runs.
        out_a = ds[0]
        out_b = ds[0]
        torch.testing.assert_close(out_a[0], out_b[0])
        torch.testing.assert_close(out_a[1], out_b[1])


def test_hha_dataloader_iterates():
    """Smoke test the high-level helper too — it threads use_hha through."""
    np.random.seed(5)
    with tempfile.TemporaryDirectory() as tmp:
        _build_fake_dataset(tmp, hha=True)
        train_loader, val_loader, num_classes = get_scannet_pretrain_dataloaders(
            data_root=tmp,
            batch_size=2,
            num_workers=0,
            crop_size=CROP,
            balanced_sampling=False,
            use_hha=True,
        )
        assert num_classes == 2
        rgb, hha, labels = next(iter(train_loader))
        assert rgb.shape == (2, 3, CROP, CROP)
        assert hha.shape == (2, 3, CROP, CROP)
        assert not torch.isnan(rgb).any()
        assert not torch.isnan(hha).any()
        assert labels.shape == (2,)


def test_hha_missing_norm_stats_raises():
    """Constructing with use_hha=True but only depth-side norm_stats raises clearly."""
    np.random.seed(6)
    # Norm stats with only the depth-modality keys (no hha_mean/std).
    raw_depth_norm_stats = {
        "rgb_mean": [0.5, 0.5, 0.5],
        "rgb_std": [0.25, 0.25, 0.25],
        "depth_mean": [2.5],
        "depth_std": [1.5],
    }
    # Empty samples list is fine for the constructor — the check fires before
    # any sample is loaded.
    with pytest.raises(ValueError, match="hha_mean"):
        ScanNetPretrainDataset(
            data_root="/tmp/unused",
            split="train",
            samples=[],
            class_names=CLASSES,
            norm_stats=raw_depth_norm_stats,
            use_hha=True,
        )
