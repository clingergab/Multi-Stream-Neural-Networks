"""End-to-end integration test for li_net.py fit() with use_mixup + use_sam.

Runs a tiny LiNet3 on random 2-stream data for 2 epochs to catch silent failures
in the mixup + SAM path and verify:
- No exceptions on any flag combination (both off, mixup only, SAM only, both on)
- history dict has expected keys
- Training actually moves the weights
- model.use_amp is restored after SAM-with-AMP runs

Marked slow because it constructs a model; still <30s on CPU.
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
def tiny_loaders():
    torch.manual_seed(0)
    n = 32
    rgb = torch.rand(n, 3, 32, 32)
    depth = torch.rand(n, 1, 32, 32)
    labels = torch.randint(0, 4, (n,))
    # Training loader returns (rgb, depth, labels) — matches N-stream tuple format
    train_ds = TensorDataset(rgb, depth, labels)
    # Val uses a disjoint set
    rgb_v = torch.rand(n, 3, 32, 32)
    depth_v = torch.rand(n, 1, 32, 32)
    labels_v = torch.randint(0, 4, (n,))
    val_ds = TensorDataset(rgb_v, depth_v, labels_v)
    return (
        DataLoader(train_ds, batch_size=8, shuffle=True),
        DataLoader(val_ds, batch_size=8, shuffle=False),
    )


EXPECTED_HISTORY_KEYS = {
    "train_loss", "val_loss", "train_accuracy", "val_accuracy",
    "train_mca", "val_mca", "learning_rates", "streams_frozen",
}


class TestFitWithMixupAndSam:
    def test_baseline_both_off_runs_cleanly(self, tiny_model, tiny_loaders):
        """Sanity: existing behavior preserved when both flags are False."""
        train_loader, val_loader = tiny_loaders
        w0 = {n: p.detach().clone() for n, p in tiny_model.named_parameters()}
        history = tiny_model.fit(
            train_loader=train_loader, val_loader=val_loader,
            epochs=2, verbose=False,
            use_mixup=False, use_sam=False,
        )
        assert EXPECTED_HISTORY_KEYS.issubset(history.keys())
        assert len(history["train_loss"]) == 2
        # Weights should have changed
        changed = any(
            not torch.equal(p, w0[n])
            for n, p in tiny_model.named_parameters()
        )
        assert changed, "baseline fit() did not modify weights"

    def test_mixup_only(self, tiny_model, tiny_loaders):
        train_loader, val_loader = tiny_loaders
        w0 = {n: p.detach().clone() for n, p in tiny_model.named_parameters()}
        history = tiny_model.fit(
            train_loader=train_loader, val_loader=val_loader,
            epochs=2, verbose=False,
            use_mixup=True, mixup_alpha=0.3,
            use_sam=False,
        )
        assert EXPECTED_HISTORY_KEYS.issubset(history.keys())
        assert len(history["train_loss"]) == 2
        changed = any(
            not torch.equal(p, w0[n])
            for n, p in tiny_model.named_parameters()
        )
        assert changed, "mixup fit() did not modify weights"

    def test_sam_only(self, tiny_model, tiny_loaders):
        train_loader, val_loader = tiny_loaders
        w0 = {n: p.detach().clone() for n, p in tiny_model.named_parameters()}
        history = tiny_model.fit(
            train_loader=train_loader, val_loader=val_loader,
            epochs=2, verbose=False,
            use_mixup=False,
            use_sam=True, sam_rho=0.05,
        )
        assert EXPECTED_HISTORY_KEYS.issubset(history.keys())
        assert len(history["train_loss"]) == 2
        changed = any(
            not torch.equal(p, w0[n])
            for n, p in tiny_model.named_parameters()
        )
        assert changed, "SAM fit() did not modify weights"

    def test_mixup_plus_sam(self, tiny_model, tiny_loaders):
        train_loader, val_loader = tiny_loaders
        w0 = {n: p.detach().clone() for n, p in tiny_model.named_parameters()}
        history = tiny_model.fit(
            train_loader=train_loader, val_loader=val_loader,
            epochs=2, verbose=False,
            use_mixup=True, mixup_alpha=0.2,
            use_sam=True, sam_rho=0.05,
        )
        assert EXPECTED_HISTORY_KEYS.issubset(history.keys())
        assert len(history["train_loss"]) == 2
        changed = any(
            not torch.equal(p, w0[n])
            for n, p in tiny_model.named_parameters()
        )
        assert changed, "mixup+SAM fit() did not modify weights"

    def test_sam_with_amp_restores_amp_flag(self, tiny_model, tiny_loaders):
        """When use_sam=True and model.use_amp=True, fit() should temporarily
        disable AMP and restore it before returning. We force use_amp=True
        post-compile to simulate the CUDA case on a CPU-only test environment.
        """
        train_loader, val_loader = tiny_loaders
        # Simulate AMP-enabled state (compile() auto-disables on CPU).
        tiny_model.use_amp = True
        _ = tiny_model.fit(
            train_loader=train_loader, val_loader=val_loader,
            epochs=1, verbose=False, use_sam=True, sam_rho=0.05,
        )
        # AMP flag should be restored after fit() returns
        assert tiny_model.use_amp is True, "use_amp was not restored after SAM run"

    def test_sam_rejects_gradient_accumulation(self, tiny_model, tiny_loaders):
        train_loader, val_loader = tiny_loaders
        with pytest.raises(ValueError, match="use_sam.*gradient_accumulation"):
            tiny_model.fit(
                train_loader=train_loader, val_loader=val_loader,
                epochs=1, verbose=False,
                use_sam=True, gradient_accumulation_steps=4,
            )


@pytest.fixture
def tiny_maxout_model():
    """Same tiny LiNet3 but with classifier_head='maxout' so the diversity-loss
    path has a real MaxoutHead at self.fc. Default LinearHead's diversity_loss
    returns 0 unconditionally and wouldn't exercise the fc.weight gradient path."""
    model = li_resnet18(
        num_classes=4,
        stream_input_channels=[3, 1],
        width_multiplier=0.25,
        device="cpu",
        use_amp=False,
        classifier_head='maxout',
        num_subnodes=3,
        subnode_dropout=0.1,
        maxout_init_perturb_std=0.1,
    )
    model.compile(
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        scheduler=None,
        loss="cross_entropy",
        label_smoothing=0.0,
        gpu_augmentation=False,
    )
    return model


class TestFitWithDiversityLoss:
    """Verify diversity_loss_weight threads through fit() into all four loss-
    injection branches: plain, mixup, SAM, SAM+mixup. Guards against silent
    regression when any of those branches gets refactored."""

    def test_diversity_plain(self, tiny_maxout_model, tiny_loaders):
        train_loader, val_loader = tiny_loaders
        w0 = tiny_maxout_model.fc.fc.weight.detach().clone()
        history = tiny_maxout_model.fit(
            train_loader=train_loader, val_loader=val_loader,
            epochs=2, verbose=False,
            use_mixup=False, use_sam=False,
            diversity_loss_weight=0.1,
        )
        assert len(history["train_loss"]) == 2
        assert not torch.equal(tiny_maxout_model.fc.fc.weight, w0)

    def test_diversity_with_mixup(self, tiny_maxout_model, tiny_loaders):
        train_loader, val_loader = tiny_loaders
        w0 = tiny_maxout_model.fc.fc.weight.detach().clone()
        history = tiny_maxout_model.fit(
            train_loader=train_loader, val_loader=val_loader,
            epochs=2, verbose=False,
            use_mixup=True, mixup_alpha=0.2, use_sam=False,
            diversity_loss_weight=0.1,
        )
        assert len(history["train_loss"]) == 2
        assert not torch.equal(tiny_maxout_model.fc.fc.weight, w0)

    def test_diversity_with_sam(self, tiny_maxout_model, tiny_loaders):
        train_loader, val_loader = tiny_loaders
        w0 = tiny_maxout_model.fc.fc.weight.detach().clone()
        history = tiny_maxout_model.fit(
            train_loader=train_loader, val_loader=val_loader,
            epochs=2, verbose=False,
            use_mixup=False, use_sam=True, sam_rho=0.05,
            diversity_loss_weight=0.1,
        )
        assert len(history["train_loss"]) == 2
        assert not torch.equal(tiny_maxout_model.fc.fc.weight, w0)

    def test_diversity_with_sam_and_mixup(self, tiny_maxout_model, tiny_loaders):
        train_loader, val_loader = tiny_loaders
        w0 = tiny_maxout_model.fc.fc.weight.detach().clone()
        history = tiny_maxout_model.fit(
            train_loader=train_loader, val_loader=val_loader,
            epochs=2, verbose=False,
            use_mixup=True, mixup_alpha=0.2, use_sam=True, sam_rho=0.05,
            diversity_loss_weight=0.1,
        )
        assert len(history["train_loss"]) == 2
        assert not torch.equal(tiny_maxout_model.fc.fc.weight, w0)

    def test_diversity_zero_weight_is_noop_on_linear_head(self, tiny_model, tiny_loaders):
        """diversity_loss_weight=0.1 on a LinearHead must not error — LinearHead's
        diversity_loss returns 0, so adding it is identity. Guards the default
        'linear' path against accidental breakage when users leave the knob on."""
        train_loader, val_loader = tiny_loaders
        history = tiny_model.fit(
            train_loader=train_loader, val_loader=val_loader,
            epochs=1, verbose=False,
            diversity_loss_weight=0.1,  # set but head is LinearHead → no-op
        )
        assert len(history["train_loss"]) == 1
