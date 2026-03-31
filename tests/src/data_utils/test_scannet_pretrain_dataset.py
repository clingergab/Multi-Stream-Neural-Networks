"""Tests for ScanNetPretrainDataset and get_scannet_pretrain_dataloaders."""

import json
import os

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


def _create_split_samples(root, class_names, split, scenes_per_class, frames_per_scene,
                           depth_low=0, depth_high=10000, zero_sentinel=True):
    """Helper to create .pt sample files in root/<split>/<class>/."""
    split_dir = root / split
    for cls_idx, cls_name in enumerate(class_names):
        cls_dir = split_dir / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)
        for scene_idx in range(scenes_per_class):
            for frame_idx in range(frames_per_scene):
                rgb = torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8)
                depth = torch.randint(depth_low, depth_high, (1, 256, 256), dtype=torch.uint16)
                if zero_sentinel:
                    depth[0, :5, :5] = 0
                torch.save(rgb, cls_dir / f'scene{cls_idx:04d}_{scene_idx:02d}_f{frame_idx:03d}_rgb.pt')
                torch.save(depth, cls_dir / f'scene{cls_idx:04d}_{scene_idx:02d}_f{frame_idx:03d}_depth.pt')


def _make_dataset(fake_dataset, split='train', **kwargs):
    """Module-level helper to construct dataset from fake_dataset fixture."""
    data_root, class_names, norm_stats = fake_dataset
    split_dir = os.path.join(str(data_root), split)
    samples = _discover_samples(split_dir, class_names)
    return ScanNetPretrainDataset(
        data_root=str(data_root),
        split=split,
        samples=samples,
        class_names=class_names,
        norm_stats=norm_stats,
        **kwargs,
    )


@pytest.fixture
def fake_dataset(tmp_path):
    """Create a minimal fake ScanNet dataset with train/val split.

    Creates 3 classes:
      - train: 4 scenes x 2 frames = 8 samples per class (24 total)
      - val:   1 scene  x 2 frames = 2 samples per class (6 total)
    Includes some depth pixels set to 0 (missing data sentinel).
    """
    class_names = ['bathroom', 'kitchen', 'office']

    with open(tmp_path / 'class_names.txt', 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")

    norm_stats = {
        'rgb_mean': [0.485, 0.456, 0.406],
        'rgb_std': [0.229, 0.224, 0.225],
        'depth_mean': [2.5],
        'depth_std': [1.2],
    }
    with open(tmp_path / 'norm_stats.json', 'w') as f:
        json.dump(norm_stats, f)

    _create_split_samples(tmp_path, class_names, 'train',
                           scenes_per_class=4, frames_per_scene=2)
    _create_split_samples(tmp_path, class_names, 'val',
                           scenes_per_class=1, frames_per_scene=2)

    return tmp_path, class_names, norm_stats


@pytest.fixture
def fake_dataset_indexed_classnames(tmp_path):
    """Like fake_dataset but with '0: bathroom' format in class_names.txt."""
    class_names = ['bathroom', 'kitchen', 'office']
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
                           scenes_per_class=2, frames_per_scene=2,
                           depth_low=100, depth_high=5000, zero_sentinel=False)
    _create_split_samples(tmp_path, class_names, 'val',
                           scenes_per_class=1, frames_per_scene=2,
                           depth_low=100, depth_high=5000, zero_sentinel=False)

    return tmp_path, class_names


class TestLoadClassNames:
    """Tests for _load_class_names helper."""

    def test_plain_format(self, fake_dataset):
        data_root, expected_names, _ = fake_dataset
        names = _load_class_names(str(data_root))
        assert names == expected_names

    def test_indexed_format(self, fake_dataset_indexed_classnames):
        data_root, expected_names = fake_dataset_indexed_classnames
        names = _load_class_names(str(data_root))
        assert names == expected_names

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="class_names.txt"):
            _load_class_names(str(tmp_path))

    def test_empty_lines_skipped(self, tmp_path):
        with open(tmp_path / 'class_names.txt', 'w') as f:
            f.write("bathroom\n\nkitchen\n\n")
        names = _load_class_names(str(tmp_path))
        assert names == ['bathroom', 'kitchen']


class TestLoadNormStats:
    """Tests for _load_norm_stats helper."""

    def test_loads_correctly(self, fake_dataset):
        data_root, _, expected_stats = fake_dataset
        stats = _load_norm_stats(str(data_root))
        assert stats == expected_stats

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="norm_stats.json"):
            _load_norm_stats(str(tmp_path))

    def test_value_types(self, fake_dataset):
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
        data_root, class_names, _ = fake_dataset
        train_dir = str(data_root / 'train')
        samples = _discover_samples(train_dir, class_names)
        assert len(samples) == 24
        labels = [s[2] for s in samples]
        assert labels.count(0) == 8
        assert labels.count(1) == 8
        assert labels.count(2) == 8

    def test_val_split_discovered_separately(self, fake_dataset):
        data_root, class_names, _ = fake_dataset
        val_dir = str(data_root / 'val')
        samples = _discover_samples(val_dir, class_names)
        assert len(samples) == 6

    def test_each_sample_has_valid_paths(self, fake_dataset):
        data_root, class_names, _ = fake_dataset
        samples = _discover_samples(str(data_root / 'train'), class_names)
        for rgb_path, depth_path, label in samples:
            assert os.path.exists(rgb_path)
            assert os.path.exists(depth_path)
            assert 0 <= label < len(class_names)

    def test_unpaired_rgb_raises(self, fake_dataset):
        data_root, class_names, _ = fake_dataset
        orphan = data_root / 'train' / 'bathroom' / 'orphan_f000_rgb.pt'
        torch.save(torch.zeros(3, 256, 256, dtype=torch.uint8), orphan)
        with pytest.raises(ValueError, match="Unpaired files"):
            _discover_samples(str(data_root / 'train'), class_names)

    def test_unpaired_depth_raises(self, fake_dataset):
        data_root, class_names, _ = fake_dataset
        orphan = data_root / 'train' / 'bathroom' / 'orphan_f000_depth.pt'
        torch.save(torch.zeros(1, 256, 256, dtype=torch.uint16), orphan)
        with pytest.raises(ValueError, match="Unpaired files"):
            _discover_samples(str(data_root / 'train'), class_names)

    def test_unknown_folder_skipped(self, fake_dataset):
        data_root, class_names, _ = fake_dataset
        extra = data_root / 'train' / 'unknown_class'
        extra.mkdir()
        torch.save(torch.zeros(3, 256, 256, dtype=torch.uint8), extra / 'x_f000_rgb.pt')
        torch.save(torch.zeros(1, 256, 256, dtype=torch.uint16), extra / 'x_f000_depth.pt')
        samples = _discover_samples(str(data_root / 'train'), class_names)
        assert len(samples) == 24

    def test_deterministic_ordering(self, fake_dataset):
        data_root, class_names, _ = fake_dataset
        train_dir = str(data_root / 'train')
        s1 = _discover_samples(train_dir, class_names)
        s2 = _discover_samples(train_dir, class_names)
        assert s1 == s2


class TestScanNetPretrainDataset:
    """Tests for ScanNetPretrainDataset class."""

    def test_init_train(self, fake_dataset):
        ds = _make_dataset(fake_dataset, split='train')
        assert len(ds) == 24
        assert ds.num_classes == 3
        assert ds.split == 'train'

    def test_init_val(self, fake_dataset):
        ds = _make_dataset(fake_dataset, split='val')
        assert ds.split == 'val'
        assert len(ds) == 6

    def test_invalid_split_raises(self, fake_dataset):
        data_root, class_names, norm_stats = fake_dataset
        samples = _discover_samples(str(data_root / 'train'), class_names)
        with pytest.raises(ValueError, match="split must be one of"):
            ScanNetPretrainDataset(
                data_root=str(data_root),
                split='test',
                samples=samples,
                class_names=class_names,
                norm_stats=norm_stats,
            )

    def test_getitem_shapes_train(self, fake_dataset):
        ds = _make_dataset(fake_dataset, split='train', crop_size=224)
        rgb, depth, label = ds[0]
        assert rgb.shape == (3, 224, 224)
        assert depth.shape == (1, 224, 224)
        assert rgb.dtype == torch.float32
        assert depth.dtype == torch.float32
        assert isinstance(label, int)
        assert 0 <= label < 3

    def test_getitem_shapes_val(self, fake_dataset):
        ds = _make_dataset(fake_dataset, split='val', crop_size=224)
        rgb, depth, label = ds[0]
        assert rgb.shape == (3, 224, 224)
        assert depth.shape == (1, 224, 224)
        assert rgb.dtype == torch.float32
        assert depth.dtype == torch.float32

    def test_depth_conversion_to_meters(self, fake_dataset):
        ds = _make_dataset(fake_dataset, split='val', normalize=False)
        rgb, depth, label = ds[0]
        assert depth.max() > 0.5, (
            f"depth.max()={depth.max():.4f} — likely /65535 instead of /1000"
        )
        assert depth.max() <= 10.0

    def test_sentinel_replaced_with_mean(self, tmp_path):
        class_names = ['bathroom']
        depth_mean = 2.5
        norm_stats = {
            'rgb_mean': [0.5, 0.5, 0.5],
            'rgb_std': [0.2, 0.2, 0.2],
            'depth_mean': [depth_mean],
            'depth_std': [1.0],
        }

        with open(tmp_path / 'class_names.txt', 'w') as f:
            f.write("bathroom\n")
        with open(tmp_path / 'norm_stats.json', 'w') as f:
            json.dump(norm_stats, f)

        cls_dir = tmp_path / 'train' / 'bathroom'
        cls_dir.mkdir(parents=True)
        rgb = torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8)
        depth = torch.full((1, 256, 256), 5000, dtype=torch.uint16)
        depth[0, 128, 128] = 0
        torch.save(rgb, cls_dir / 'scene0000_00_f000_rgb.pt')
        torch.save(depth, cls_dir / 'scene0000_00_f000_depth.pt')

        samples = _discover_samples(str(tmp_path / 'train'), class_names)
        ds = ScanNetPretrainDataset(
            data_root=str(tmp_path),
            split='val',
            samples=samples,
            class_names=class_names,
            norm_stats=norm_stats,
            normalize=False,
        )
        _, out_depth, _ = ds[0]
        assert out_depth[0, 112, 112] == pytest.approx(depth_mean)

    def test_normalized_output_range(self, fake_dataset):
        ds = _make_dataset(fake_dataset, split='val', normalize=True)
        rgb, depth, label = ds[0]
        assert rgb.min() < 0.0 or rgb.max() > 1.0

    def test_unnormalized_rgb_range(self, fake_dataset):
        ds = _make_dataset(fake_dataset, split='val', normalize=False)
        rgb, depth, label = ds[0]
        assert rgb.min() >= 0.0
        assert rgb.max() <= 1.0

    def test_labels_are_list_of_int(self, fake_dataset):
        ds = _make_dataset(fake_dataset, split='train')
        assert isinstance(ds.labels, list)
        assert all(isinstance(l, int) for l in ds.labels)

    def test_get_class_weights_shape(self, fake_dataset):
        ds = _make_dataset(fake_dataset, split='train')
        weights = ds.get_class_weights()
        assert weights.shape == (3,)
        assert weights.dtype == torch.float32
        assert torch.allclose(weights, torch.ones(3), atol=0.01)

    def test_get_sample_weights_shape(self, fake_dataset):
        ds = _make_dataset(fake_dataset, split='train')
        weights = ds.get_sample_weights()
        assert weights.shape == (24,)
        assert weights.dtype == torch.float64
        assert (weights > 0).all()

    def test_get_class_distribution(self, fake_dataset):
        ds = _make_dataset(fake_dataset, split='train')
        dist = ds.get_class_distribution()
        assert set(dist.keys()) == {'bathroom', 'kitchen', 'office'}
        for name in ['bathroom', 'kitchen', 'office']:
            assert dist[name]['count'] == 8
            assert isinstance(dist[name]['count'], int)
            assert isinstance(dist[name]['percentage'], float)
            assert abs(dist[name]['percentage'] - 100.0 / 3) < 0.1

    def test_get_norm_stats(self, fake_dataset):
        data_root, _, norm_stats = fake_dataset
        ds = _make_dataset(fake_dataset, split='train')
        assert ds.get_norm_stats() == norm_stats

    def test_val_deterministic(self, fake_dataset):
        """Two calls with same idx return identical tensors (val, no augmentation)."""
        ds = _make_dataset(fake_dataset, split='val')
        rgb1, depth1, label1 = ds[0]
        rgb2, depth2, label2 = ds[0]
        assert torch.equal(rgb1, rgb2)
        assert torch.equal(depth1, depth2)
        assert label1 == label2


class TestSynchronizedErasing:
    """Tests for synchronized random erasing between RGB and depth."""

    def test_erasing_shares_box_coordinates(self, tmp_path):
        """When both modalities are erased, the erased region is identical.

        Uses depth values far from depth_mean to avoid sentinel-replacement
        zeros confusing the detection (normalized sentinel pixels become 0.0
        which would be indistinguishable from erased pixels).
        """
        class_names = ['bathroom']
        # Use depth_mean=2.5; create depth far from 2500mm to avoid
        # normalized values near 0.0
        norm_stats = {
            'rgb_mean': [0.485, 0.456, 0.406],
            'rgb_std': [0.229, 0.224, 0.225],
            'depth_mean': [2.5],
            'depth_std': [1.2],
        }
        with open(tmp_path / 'class_names.txt', 'w') as f:
            f.write("bathroom\n")
        with open(tmp_path / 'norm_stats.json', 'w') as f:
            json.dump(norm_stats, f)

        cls_dir = tmp_path / 'train' / 'bathroom'
        cls_dir.mkdir(parents=True)
        for i in range(5):
            # depth_mean=2.5m=2500mm; use 4000-5000mm so normalized values
            # are far from 0.0 and won't be confused with erased pixels
            rgb = torch.randint(50, 200, (3, 256, 256), dtype=torch.uint8)
            depth = torch.randint(4000, 5000, (1, 256, 256), dtype=torch.uint16)
            torch.save(rgb, cls_dir / f'scene0000_{i:02d}_f000_rgb.pt')
            torch.save(depth, cls_dir / f'scene0000_{i:02d}_f000_depth.pt')

        samples = _discover_samples(str(tmp_path / 'train'), class_names)
        ds = ScanNetPretrainDataset(
            data_root=str(tmp_path), split='train', samples=samples,
            class_names=class_names, norm_stats=norm_stats, normalize=True,
        )
        ds._rgb_erasing_p = 1.0
        ds._depth_erasing_p = 1.0
        ds._flip_p = 0.0
        ds._color_jitter_p = 0.0
        ds._blur_p = 0.0
        ds._grayscale_p = 0.0
        ds._depth_aug_p = 0.0
        ds._depth_scale_jitter_p = 0.0
        ds._hole_dropout_p = 0.0

        matches = 0
        trials = 20
        for trial_idx in range(trials):
            np.random.seed(trial_idx * 7)
            rgb, depth, _ = ds[trial_idx % len(ds)]
            # Erased region is filled with 0.0
            rgb_erased = (rgb == 0.0).all(dim=0)  # [H, W]
            depth_erased = (depth[0] == 0.0)  # [H, W]
            if rgb_erased.any() and depth_erased.any():
                if torch.equal(rgb_erased, depth_erased):
                    matches += 1

        assert matches > trials * 0.5, (
            f"Only {matches}/{trials} trials had matching erase regions"
        )


class TestHoleDropout:
    """Tests for _apply_hole_dropout method."""

    def test_creates_zero_regions(self, fake_dataset):
        ds = _make_dataset(fake_dataset, split='train')
        depth = torch.ones(1, 224, 224, dtype=torch.float32) * 2.5
        result = ds._apply_hole_dropout(depth)
        assert (result == 0.0).any()

    def test_preserves_shape_dtype(self, fake_dataset):
        ds = _make_dataset(fake_dataset, split='train')
        depth = torch.ones(1, 224, 224, dtype=torch.float32) * 2.5
        result = ds._apply_hole_dropout(depth)
        assert result.shape == (1, 224, 224)
        assert result.dtype == torch.float32

    def test_modifies_in_place(self, fake_dataset):
        ds = _make_dataset(fake_dataset, split='train')
        depth = torch.ones(1, 224, 224, dtype=torch.float32) * 2.5
        result = ds._apply_hole_dropout(depth)
        assert result is depth


class TestGetScannetPretrainDataloaders:
    """Tests for get_scannet_pretrain_dataloaders factory."""

    def test_returns_three_elements(self, fake_dataset):
        data_root, _, _ = fake_dataset
        result = get_scannet_pretrain_dataloaders(
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
        data_root, _, _ = fake_dataset
        result = get_scannet_pretrain_dataloaders(
            data_root=str(data_root),
            batch_size=4,
            num_workers=0,
            seed=42,
            use_class_weights=True,
        )
        assert len(result) == 4
        _, _, num_classes, class_weights = result
        assert class_weights.shape == (num_classes,)
        assert class_weights.dtype == torch.float32

    def test_batch_shapes(self, fake_dataset):
        data_root, _, _ = fake_dataset
        train_loader, val_loader, num_classes = get_scannet_pretrain_dataloaders(
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
        data_root, _, _ = fake_dataset
        _, val_loader, _ = get_scannet_pretrain_dataloaders(
            data_root=str(data_root),
            batch_size=4,
            num_workers=0,
            seed=42,
        )
        batch1 = next(iter(val_loader))
        batch2 = next(iter(val_loader))
        assert torch.equal(batch1[0], batch2[0])
        assert torch.equal(batch1[1], batch2[1])

    def test_num_workers_zero_no_crash(self, fake_dataset):
        data_root, _, _ = fake_dataset
        train_loader, val_loader, _ = get_scannet_pretrain_dataloaders(
            data_root=str(data_root),
            batch_size=4,
            num_workers=0,
            seed=42,
        )
        next(iter(train_loader))
        next(iter(val_loader))

    def test_num_workers_one_val_no_crash(self, fake_dataset):
        """num_workers=1 -> val gets 0 workers; must not crash (bug fix)."""
        data_root, _, _ = fake_dataset
        train_loader, val_loader, _ = get_scannet_pretrain_dataloaders(
            data_root=str(data_root),
            batch_size=4,
            num_workers=1,
            seed=42,
        )
        next(iter(val_loader))

    def test_missing_split_dirs_raises(self, tmp_path):
        with open(tmp_path / 'class_names.txt', 'w') as f:
            f.write("bathroom\n")
        with open(tmp_path / 'norm_stats.json', 'w') as f:
            json.dump({
                'rgb_mean': [0.5, 0.5, 0.5], 'rgb_std': [0.2, 0.2, 0.2],
                'depth_mean': [2.0], 'depth_std': [1.0],
            }, f)
        with pytest.raises(FileNotFoundError, match="train/ and val/"):
            get_scannet_pretrain_dataloaders(data_root=str(tmp_path), num_workers=0)

    def test_empty_train_raises(self, tmp_path):
        with open(tmp_path / 'class_names.txt', 'w') as f:
            f.write("bathroom\n")
        with open(tmp_path / 'norm_stats.json', 'w') as f:
            json.dump({
                'rgb_mean': [0.5, 0.5, 0.5], 'rgb_std': [0.2, 0.2, 0.2],
                'depth_mean': [2.0], 'depth_std': [1.0],
            }, f)
        (tmp_path / 'train').mkdir()
        (tmp_path / 'val').mkdir()
        with pytest.raises(ValueError, match="No training samples"):
            get_scannet_pretrain_dataloaders(data_root=str(tmp_path), num_workers=0)

    def test_balanced_sampling_false(self, fake_dataset):
        data_root, _, _ = fake_dataset
        result = get_scannet_pretrain_dataloaders(
            data_root=str(data_root),
            batch_size=4,
            num_workers=0,
            seed=42,
            balanced_sampling=False,
        )
        assert len(result) == 3
        train_loader, val_loader, num_classes = result
        assert num_classes == 3
        assert train_loader.sampler.__class__.__name__ != 'WeightedRandomSampler'
        rgb, depth, labels = next(iter(train_loader))
        assert rgb.shape == (4, 3, 224, 224)

    def test_reproducible_loading(self, fake_dataset):
        data_root, _, _ = fake_dataset
        _, val1, _ = get_scannet_pretrain_dataloaders(
            data_root=str(data_root), batch_size=6, num_workers=0, seed=42
        )
        _, val2, _ = get_scannet_pretrain_dataloaders(
            data_root=str(data_root), batch_size=6, num_workers=0, seed=42
        )
        b1 = next(iter(val1))
        b2 = next(iter(val2))
        assert torch.equal(b1[0], b2[0])
        assert torch.equal(b1[2], b2[2])
