"""End-to-end integration tests for LiNet3 fit() with use_consistency=True.

Mirrors test_sam_mixup_integration.py: tiny model + tiny synthetic two-view
data, 1-2 epochs, just enough to catch silent failures across the three
forward branches (SAM / AMP-disabled / standard). Stays under ~30s on CPU.
"""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.linear_integration.li_net3 import li_resnet18


@pytest.fixture
def tiny_model():
    model = li_resnet18(
        num_classes=4,
        stream_input_channels=[3, 1],
        width_multiplier=0.25,
        device="cpu",
        use_amp=False,  # Stay deterministic on CPU
    )
    model.compile(
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        scheduler=None,
        loss="cross_entropy",
        label_smoothing=0.0,
        gpu_augmentation=False,
    )
    return model


@pytest.fixture
def two_view_loaders():
    """Train loader yields 5-tuples (rgb_a, depth_a, rgb_b, depth_b, label).
    Val loader yields 3-tuples (single-view) — matches real two-view training
    config where val_loader uses a single-view dataset."""
    torch.manual_seed(0)
    n = 24
    rgb_a = torch.rand(n, 3, 32, 32)
    depth_a = torch.rand(n, 1, 32, 32)
    rgb_b = torch.rand(n, 3, 32, 32)
    depth_b = torch.rand(n, 1, 32, 32)
    labels = torch.randint(0, 4, (n,))
    train_ds = TensorDataset(rgb_a, depth_a, rgb_b, depth_b, labels)

    rgb_v = torch.rand(n, 3, 32, 32)
    depth_v = torch.rand(n, 1, 32, 32)
    labels_v = torch.randint(0, 4, (n,))
    val_ds = TensorDataset(rgb_v, depth_v, labels_v)

    return (
        DataLoader(train_ds, batch_size=8, shuffle=True),
        DataLoader(val_ds, batch_size=8, shuffle=False),
    )


@pytest.fixture
def single_view_loaders():
    """For tests that mistakenly enable use_consistency without two-view data."""
    torch.manual_seed(0)
    n = 24
    rgb = torch.rand(n, 3, 32, 32)
    depth = torch.rand(n, 1, 32, 32)
    labels = torch.randint(0, 4, (n,))
    train_ds = TensorDataset(rgb, depth, labels)
    return DataLoader(train_ds, batch_size=8, shuffle=True)


class TestFitWithConsistency:
    def test_consistency_alone_runs_2_epochs(self, tiny_model, two_view_loaders):
        """use_consistency=True with no other flags executes for 2 epochs and
        moves weights, populates history['consistency_loss']."""
        train_loader, val_loader = two_view_loaders
        w0 = {n: p.detach().clone() for n, p in tiny_model.named_parameters()}
        history = tiny_model.fit(
            train_loader=train_loader, val_loader=val_loader,
            epochs=2, verbose=False,
            use_mixup=False, use_sam=False,
            use_consistency=True,
            consistency_weight=1.0,
            consistency_temperature=2.0,
        )
        assert "consistency_loss" in history
        assert len(history["consistency_loss"]) == 2
        assert all(v >= 0.0 for v in history["consistency_loss"])
        # Weights moved
        changed = any(
            not torch.equal(p, w0[n]) for n, p in tiny_model.named_parameters()
        )
        assert changed

    def test_consistency_with_symmetric_mixup_runs(self, tiny_model, two_view_loaders):
        """The production config: consistency + mixup together, with the
        symmetric-mixup path (shared λ + perm across both views)."""
        train_loader, val_loader = two_view_loaders
        history = tiny_model.fit(
            train_loader=train_loader, val_loader=val_loader,
            epochs=2, verbose=False,
            use_mixup=True, mixup_alpha=0.2,
            use_consistency=True,
            consistency_weight=1.0,
            consistency_temperature=2.0,
        )
        assert "consistency_loss" in history
        assert len(history["consistency_loss"]) == 2

    def test_consistency_with_sam_runs(self, tiny_model, two_view_loaders):
        """SAM × consistency = 4× compute (2 SAM passes × 2-view stack).
        1 epoch only to keep the test cheap."""
        train_loader, val_loader = two_view_loaders
        history = tiny_model.fit(
            train_loader=train_loader, val_loader=val_loader,
            epochs=1, verbose=False,
            use_mixup=False,
            use_sam=True, sam_rho=0.05,
            use_consistency=True,
            consistency_weight=1.0,
            consistency_temperature=2.0,
        )
        assert "consistency_loss" in history
        assert len(history["consistency_loss"]) == 1

    def test_consistency_with_sam_and_mixup_runs(self, tiny_model, two_view_loaders):
        """Full stack: SAM + mixup + consistency. 1 epoch."""
        train_loader, val_loader = two_view_loaders
        history = tiny_model.fit(
            train_loader=train_loader, val_loader=val_loader,
            epochs=1, verbose=False,
            use_mixup=True, mixup_alpha=0.2,
            use_sam=True, sam_rho=0.05,
            use_consistency=True,
            consistency_weight=1.0,
            consistency_temperature=2.0,
        )
        assert "consistency_loss" in history

    def test_consistency_with_cbbn_raises(self, tiny_model, two_view_loaders):
        """CBBN's per-class label stash is incompatible with the stacked 2B
        forward; fit() must reject the combo with a clear message."""
        train_loader, val_loader = two_view_loaders
        with pytest.raises(ValueError, match="incompatible"):
            tiny_model.fit(
                train_loader=train_loader, val_loader=val_loader,
                epochs=1, verbose=False,
                use_consistency=True,
                use_class_balanced_bn=True,
            )

    def test_consistency_with_single_view_loader_raises(self, tiny_model, single_view_loaders):
        """User error: use_consistency=True but the loader only yields 3-tuples
        (forgot return_two_views=True). Should fail loudly with a hint."""
        train_loader = single_view_loaders
        with pytest.raises(RuntimeError, match="return_two_views=True"):
            tiny_model.fit(
                train_loader=train_loader,
                epochs=1, verbose=False,
                use_consistency=True,
            )

    def test_consistency_weight_zero_disables_kl_term(self, tiny_model, two_view_loaders):
        """When consistency_weight=0, KL is not added to the loss but the
        two-view forward still runs; consistency_loss history records 0.0."""
        train_loader, val_loader = two_view_loaders
        history = tiny_model.fit(
            train_loader=train_loader, val_loader=val_loader,
            epochs=1, verbose=False,
            use_consistency=True,
            consistency_weight=0.0,
        )
        # consistency_loss not accumulated when weight is 0 (the if-guard
        # short-circuits the kl computation)
        assert history["consistency_loss"] == [0.0]

    def test_validation_unaffected_by_consistency(self, tiny_model, two_view_loaders):
        """Val loader is single-view; _validate path must work unchanged."""
        train_loader, val_loader = two_view_loaders
        history = tiny_model.fit(
            train_loader=train_loader, val_loader=val_loader,
            epochs=1, verbose=False,
            use_consistency=True,
        )
        assert len(history["val_loss"]) == 1
        assert history["val_loss"][0] >= 0.0
        assert len(history["val_accuracy"]) == 1
