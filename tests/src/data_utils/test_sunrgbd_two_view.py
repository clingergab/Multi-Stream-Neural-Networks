"""Unit tests for SUNRGBDDataset return_two_views=True path."""

import os

import pytest
import torch
from torch.utils.data import DataLoader

from data_utils.sunrgbd_dataset import SUNRGBDDataset


DATA_ROOT = 'data/sunrgbd_19_traintest'

pytestmark = pytest.mark.skipif(
    not os.path.exists(DATA_ROOT),
    reason=f"SUN RGB-D 19-cat dataset not present at {DATA_ROOT}",
)


class TestReturnTwoViewsBackwardCompat:
    def test_default_returns_3tuple_train(self):
        ds = SUNRGBDDataset(data_root=DATA_ROOT, split='train')
        out = ds[0]
        assert len(out) == 3
        rgb, depth, label = out
        assert rgb.shape == (3, 224, 224)
        assert depth.shape == (1, 224, 224)
        assert isinstance(label, int)

    def test_default_returns_3tuple_val(self):
        # 'test' split exists in this dataset (no separate val)
        ds = SUNRGBDDataset(data_root=DATA_ROOT, split='test')
        out = ds[0]
        assert len(out) == 3

    def test_explicit_false_returns_3tuple(self):
        ds = SUNRGBDDataset(data_root=DATA_ROOT, split='train', return_two_views=False)
        assert len(ds[0]) == 3


class TestReturnTwoViewsTrue:
    def test_returns_5tuple(self):
        ds = SUNRGBDDataset(data_root=DATA_ROOT, split='train', return_two_views=True)
        out = ds[0]
        assert len(out) == 5
        rgb_a, depth_a, rgb_b, depth_b, label = out
        assert rgb_a.shape == (3, 224, 224)
        assert depth_a.shape == (1, 224, 224)
        assert rgb_b.shape == (3, 224, 224)
        assert depth_b.shape == (1, 224, 224)
        assert isinstance(label, int)

    def test_dtypes_match_single_view(self):
        ds_single = SUNRGBDDataset(data_root=DATA_ROOT, split='train')
        ds_two = SUNRGBDDataset(data_root=DATA_ROOT, split='train', return_two_views=True)
        rgb_s, depth_s, _ = ds_single[0]
        rgb_a, depth_a, rgb_b, depth_b, _ = ds_two[0]
        assert rgb_a.dtype == rgb_s.dtype
        assert depth_a.dtype == depth_s.dtype
        assert rgb_b.dtype == rgb_s.dtype
        assert depth_b.dtype == depth_s.dtype

    def test_two_views_are_independent(self):
        """Two views must come from independent stochastic-augmentation calls
        — they should differ on at least one stream (statistically nearly
        always; with default aug probabilities of 1.0 each view runs the full
        pipeline with fresh random draws)."""
        ds = SUNRGBDDataset(data_root=DATA_ROOT, split='train', return_two_views=True)
        # Sample several to be robust against rare exact matches
        any_differ = False
        for idx in [0, 7, 19, 42, 100]:
            rgb_a, depth_a, rgb_b, depth_b, _ = ds[idx]
            if not torch.equal(rgb_a, rgb_b) or not torch.equal(depth_a, depth_b):
                any_differ = True
                break
        assert any_differ, "Two views should differ on at least one stream/sample"

    def test_label_matches_single_view(self):
        """Both views share the same label as the single-view path for the
        same idx (label is deterministic per idx)."""
        ds_single = SUNRGBDDataset(data_root=DATA_ROOT, split='train')
        ds_two = SUNRGBDDataset(data_root=DATA_ROOT, split='train', return_two_views=True)
        for idx in [0, 1, 2, 100]:
            _, _, label_single = ds_single[idx]
            _, _, _, _, label_two = ds_two[idx]
            assert label_single == label_two


class TestReturnTwoViewsValidation:
    def test_val_split_raises(self):
        # No 'val' split in this dataset; check 'test' which is the held-out split
        with pytest.raises(ValueError, match="return_two_views=True is only valid for split='train'"):
            SUNRGBDDataset(data_root=DATA_ROOT, split='test', return_two_views=True)


class TestReturnTwoViewsDataLoader:
    def test_default_collate_5tuple(self):
        """PyTorch's default collate should stack the 5-tuple into 5 batched
        tensors without a custom collate function."""
        ds = SUNRGBDDataset(data_root=DATA_ROOT, split='train', return_two_views=True)
        loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
        batch = next(iter(loader))
        assert len(batch) == 5
        rgb_a, depth_a, rgb_b, depth_b, labels = batch
        assert rgb_a.shape == (4, 3, 224, 224)
        assert depth_a.shape == (4, 1, 224, 224)
        assert rgb_b.shape == (4, 3, 224, 224)
        assert depth_b.shape == (4, 1, 224, 224)
        assert labels.shape == (4,)
