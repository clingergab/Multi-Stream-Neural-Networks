"""Tests for OmniPretrainDataset and get_omnipretrain_dataloaders."""

import json
import os

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


def _create_split_samples(root, class_names, split, objs_per_class, frames_per_obj,
                           depth_low=0, depth_high=10000, zero_sentinel=True):
    """Helper to create .pt sample files in root/<split>/<class>/."""
    split_dir = root / split
    for cls_idx, cls_name in enumerate(class_names):
        cls_dir = split_dir / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)
        for obj_idx in range(objs_per_class):
            for frame_idx in range(frames_per_obj):
                rgb = torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8)
                depth = torch.randint(depth_low, depth_high, (1, 256, 256), dtype=torch.uint16)
                if zero_sentinel:
                    depth[0, :5, :5] = 0
                torch.save(rgb, cls_dir / f'obj_{cls_idx:03d}_{obj_idx:03d}_f{frame_idx:03d}_rgb.pt')
                torch.save(depth, cls_dir / f'obj_{cls_idx:03d}_{obj_idx:03d}_f{frame_idx:03d}_depth.pt')


def _make_dataset(fake_dataset, split='train', **kwargs):
    """Module-level helper to construct dataset from fake_dataset fixture."""
    data_root, class_names, norm_stats = fake_dataset
    split_dir = os.path.join(str(data_root), split)
    samples = _discover_samples(split_dir, class_names)
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
    """Create a minimal fake OmniPretrain dataset with train/val split.

    Creates 3 classes:
      - train: 4 objects x 2 frames = 8 samples per class (24 total)
      - val:   1 object  x 2 frames = 2 samples per class (6 total)
    Includes some depth pixels set to 0 (missing data sentinel).
    """
    class_names = ['chair', 'sofa', 'table']

    # class_names.txt at root
    with open(tmp_path / 'class_names.txt', 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")

    # norm_stats.json at root
    norm_stats = {
        'rgb_mean': [0.485, 0.456, 0.406],
        'rgb_std': [0.229, 0.224, 0.225],
        'depth_mean': [2.5],
        'depth_std': [1.2],
    }
    with open(tmp_path / 'norm_stats.json', 'w') as f:
        json.dump(norm_stats, f)

    # Create train split: 4 objects x 2 frames per class
    _create_split_samples(tmp_path, class_names, 'train',
                           objs_per_class=4, frames_per_obj=2)
    # Create val split: 1 object x 2 frames per class
    _create_split_samples(tmp_path, class_names, 'val',
                           objs_per_class=1, frames_per_obj=2)

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

    _create_split_samples(tmp_path, class_names, 'train',
                           objs_per_class=2, frames_per_obj=2,
                           depth_low=100, depth_high=5000, zero_sentinel=False)
    _create_split_samples(tmp_path, class_names, 'val',
                           objs_per_class=1, frames_per_obj=2,
                           depth_low=100, depth_high=5000, zero_sentinel=False)

    return tmp_path, class_names


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
        """All 24 train samples (3 classes x 4 objs x 2 frames) discovered."""
        data_root, class_names, _ = fake_dataset
        train_dir = str(data_root / 'train')
        samples = _discover_samples(train_dir, class_names)
        assert len(samples) == 24
        labels = [s[2] for s in samples]
        assert labels.count(0) == 8  # chair
        assert labels.count(1) == 8  # sofa
        assert labels.count(2) == 8  # table

    def test_val_split_discovered_separately(self, fake_dataset):
        """Val samples discovered from val/ subdirectory."""
        data_root, class_names, _ = fake_dataset
        val_dir = str(data_root / 'val')
        samples = _discover_samples(val_dir, class_names)
        assert len(samples) == 6  # 3 classes x 1 obj x 2 frames

    def test_each_sample_has_valid_paths(self, fake_dataset):
        """Each (rgb_path, depth_path, label) has existing files."""
        data_root, class_names, _ = fake_dataset
        samples = _discover_samples(str(data_root / 'train'), class_names)
        for rgb_path, depth_path, label in samples:
            assert os.path.exists(rgb_path)
            assert os.path.exists(depth_path)
            assert 0 <= label < len(class_names)

    def test_unpaired_rgb_raises(self, fake_dataset):
        """ValueError raised when RGB file has no matching depth."""
        data_root, class_names, _ = fake_dataset
        orphan = data_root / 'train' / 'chair' / 'orphan_f000_rgb.pt'
        torch.save(torch.zeros(3, 256, 256, dtype=torch.uint8), orphan)
        with pytest.raises(ValueError, match="Unpaired files"):
            _discover_samples(str(data_root / 'train'), class_names)

    def test_unpaired_depth_raises(self, fake_dataset):
        """ValueError raised when depth file has no matching RGB."""
        data_root, class_names, _ = fake_dataset
        orphan = data_root / 'train' / 'chair' / 'orphan_f000_depth.pt'
        torch.save(torch.zeros(1, 256, 256, dtype=torch.uint16), orphan)
        with pytest.raises(ValueError, match="Unpaired files"):
            _discover_samples(str(data_root / 'train'), class_names)

    def test_unknown_folder_skipped(self, fake_dataset):
        """Folders not in class_names.txt are skipped silently."""
        data_root, class_names, _ = fake_dataset
        extra = data_root / 'train' / 'unknown_class'
        extra.mkdir()
        torch.save(torch.zeros(3, 256, 256, dtype=torch.uint8), extra / 'x_f000_rgb.pt')
        torch.save(torch.zeros(1, 256, 256, dtype=torch.uint16), extra / 'x_f000_depth.pt')
        samples = _discover_samples(str(data_root / 'train'), class_names)
        assert len(samples) == 24  # unchanged

    def test_deterministic_ordering(self, fake_dataset):
        """Two calls return identical ordering."""
        data_root, class_names, _ = fake_dataset
        train_dir = str(data_root / 'train')
        s1 = _discover_samples(train_dir, class_names)
        s2 = _discover_samples(train_dir, class_names)
        assert s1 == s2


class TestOmniPretrainDataset:
    """Tests for OmniPretrainDataset class."""

    def test_init_train(self, fake_dataset):
        """Train dataset initializes with correct counts."""
        ds = _make_dataset(fake_dataset, split='train')
        assert len(ds) == 24
        assert ds.num_classes == 3
        assert ds.split == 'train'

    def test_init_val(self, fake_dataset):
        """Val dataset initializes correctly."""
        ds = _make_dataset(fake_dataset, split='val')
        assert ds.split == 'val'
        assert len(ds) == 6

    def test_invalid_split_raises(self, fake_dataset):
        """ValueError on invalid split name."""
        data_root, class_names, norm_stats = fake_dataset
        samples = _discover_samples(str(data_root / 'train'), class_names)
        with pytest.raises(ValueError, match="split must be one of"):
            OmniPretrainDataset(
                data_root=str(data_root),
                split='test',
                samples=samples,
                class_names=class_names,
                norm_stats=norm_stats,
            )

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

        cls_dir = tmp_path / 'train' / 'chair'
        cls_dir.mkdir(parents=True)
        rgb = torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8)
        # All pixels set to 3000mm
        depth = torch.full((1, 256, 256), 3000, dtype=torch.uint16)
        torch.save(rgb, cls_dir / 'obj_000_f000_rgb.pt')
        torch.save(depth, cls_dir / 'obj_000_f000_depth.pt')

        samples = _discover_samples(str(tmp_path / 'train'), class_names)
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

        cls_dir = tmp_path / 'train' / 'chair'
        cls_dir.mkdir(parents=True)
        rgb = torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8)
        # All non-zero depth except at center
        depth = torch.full((1, 256, 256), 5000, dtype=torch.uint16)
        depth[0, 128, 128] = 0  # sentinel at known position
        torch.save(rgb, cls_dir / 'obj_000_f000_rgb.pt')
        torch.save(depth, cls_dir / 'obj_000_f000_depth.pt')

        samples = _discover_samples(str(tmp_path / 'train'), class_names)
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
        assert weights.shape == (24,)
        assert weights.dtype == torch.float64
        assert (weights > 0).all()

    def test_get_class_distribution(self, fake_dataset):
        """get_class_distribution returns correct counts."""
        ds = _make_dataset(fake_dataset, split='train')
        dist = ds.get_class_distribution()
        assert set(dist.keys()) == {'chair', 'sofa', 'table'}
        for name in ['chair', 'sofa', 'table']:
            assert dist[name]['count'] == 8
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
        samples = _discover_samples(str(data_root / 'train'), class_names)
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
        # Disable flip so sentinel position is deterministic
        ds._flip_p = 0.0

        # Run multiple iterations to exercise augmentation paths
        np.random.seed(12345)
        for _ in range(5):
            _, out_depth, _ = ds[0]
            # The fixture has zeros in top-left 5x5. With crop_size=256
            # no cropping happens and flip disabled, so those pixels
            # always survive at their original positions.
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

    def test_both_splits_have_all_classes(self, fake_dataset):
        """Train and val both contain samples from all classes."""
        data_root, _, _ = fake_dataset
        train_loader, val_loader, _ = get_omnipretrain_dataloaders(
            data_root=str(data_root),
            batch_size=30,  # all at once
            num_workers=0,
            seed=42,
        )
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

    def test_reproducible_loading(self, fake_dataset):
        """Same seed produces same val outputs."""
        data_root, _, _ = fake_dataset
        _, val1, _ = get_omnipretrain_dataloaders(
            data_root=str(data_root), batch_size=6, num_workers=0, seed=42
        )
        _, val2, _ = get_omnipretrain_dataloaders(
            data_root=str(data_root), batch_size=6, num_workers=0, seed=42
        )
        b1 = next(iter(val1))
        b2 = next(iter(val2))
        assert torch.equal(b1[0], b2[0])
        assert torch.equal(b1[2], b2[2])

    def test_missing_split_dirs_raises(self, tmp_path):
        """FileNotFoundError raised when train/val dirs missing."""
        with open(tmp_path / 'class_names.txt', 'w') as f:
            f.write("chair\n")
        with open(tmp_path / 'norm_stats.json', 'w') as f:
            json.dump({
                'rgb_mean': [0.5, 0.5, 0.5], 'rgb_std': [0.2, 0.2, 0.2],
                'depth_mean': [2.0], 'depth_std': [1.0],
            }, f)
        with pytest.raises(FileNotFoundError, match="train/ and val/"):
            get_omnipretrain_dataloaders(data_root=str(tmp_path), num_workers=0)

    def test_empty_train_raises(self, tmp_path):
        """ValueError raised when train dir has no samples."""
        with open(tmp_path / 'class_names.txt', 'w') as f:
            f.write("chair\n")
        with open(tmp_path / 'norm_stats.json', 'w') as f:
            json.dump({
                'rgb_mean': [0.5, 0.5, 0.5], 'rgb_std': [0.2, 0.2, 0.2],
                'depth_mean': [2.0], 'depth_std': [1.0],
            }, f)
        (tmp_path / 'train').mkdir()
        (tmp_path / 'val').mkdir()
        with pytest.raises(ValueError, match="No training samples"):
            get_omnipretrain_dataloaders(data_root=str(tmp_path), num_workers=0)

    def test_balanced_sampling_false(self, fake_dataset):
        """balanced_sampling=False uses shuffle instead of WeightedRandomSampler."""
        data_root, _, _ = fake_dataset
        result = get_omnipretrain_dataloaders(
            data_root=str(data_root),
            batch_size=4,
            num_workers=0,
            seed=42,
            balanced_sampling=False,
        )
        assert len(result) == 3
        train_loader, val_loader, num_classes = result
        assert num_classes == 3
        # Verify train_loader does NOT have a weighted sampler
        assert train_loader.sampler.__class__.__name__ != 'WeightedRandomSampler'
        # Should still produce valid batches
        rgb, depth, labels = next(iter(train_loader))
        assert rgb.shape == (4, 3, 224, 224)
