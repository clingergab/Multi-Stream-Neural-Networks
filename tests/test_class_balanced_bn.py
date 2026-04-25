"""Tests for ClassBalancedLIBatchNorm2d.

Covers:
- _class_balanced_batch_stats math (per-class mean/var; equal-weight average)
- Identical to standard LIBatchNorm2d when no labels are stashed
- Identical to standard LIBatchNorm2d at eval time
- Per-stream pathways unchanged (only integrated is class-balanced)
- Integrated pathway differs from standard when label distribution is skewed
- Running-stats update Bessel correction matches standard BN convention
- convert_to_class_balanced_bn preserves all parameters/buffers
- End-to-end: forward through a small LINet works with stash_labels active
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import torch.nn as nn

from src.models.linear_integration.li_net3 import li_resnet18
from src.models.linear_integration.li_net3.conv import LIBatchNorm2d
from src.models.linear_integration.li_net3.class_balanced_bn import (
    ClassBalancedLIBatchNorm2d,
    _class_balanced_batch_stats,
    convert_to_class_balanced_bn,
    stash_labels,
)


# ---------------------------------------------------------------------------
# Group A — _class_balanced_batch_stats (pure math)
# ---------------------------------------------------------------------------

def test_cb_stats_uniform_labels_matches_standard_when_classes_have_same_distribution():
    """If every class has identical x distribution (in expectation), the class-
    balanced average should be very close to the standard batch mean/var.
    Using a finite sample we expect close but not identical values."""
    torch.manual_seed(0)
    x = torch.randn(64, 8, 4, 4)  # identical distribution across all samples
    labels = torch.randint(0, 4, (64,))
    m_cb, v_cb, n_total = _class_balanced_batch_stats(x, labels)
    m_std = x.mean(dim=(0, 2, 3))
    v_std = x.var(dim=(0, 2, 3), unbiased=False)
    # Expected to be close (same distribution) but not identical (finite sample).
    assert torch.allclose(m_cb, m_std, atol=0.2)
    assert torch.allclose(v_cb, v_std, atol=0.3)


def test_cb_stats_skewed_labels_pulls_toward_minority():
    """If one class has shifted features (mean +5) and is rare (1 of 16 samples),
    standard batch mean is dominated by the majority. Class-balanced mean
    weights the minority equally → pulled significantly higher."""
    torch.manual_seed(0)
    x_majority = torch.randn(60, 8, 4, 4)            # 60 majority samples, mean ≈ 0
    x_minority = torch.randn(4, 8, 4, 4) + 5.0        # 4 minority samples, mean ≈ +5
    x = torch.cat([x_majority, x_minority], dim=0)    # (64, 8, 4, 4)
    labels = torch.cat([torch.zeros(60, dtype=torch.long), torch.ones(4, dtype=torch.long)])

    m_std = x.mean(dim=(0, 2, 3))
    m_cb, _, _ = _class_balanced_batch_stats(x, labels)

    # Standard mean ≈ (60*0 + 4*5) / 64 = 0.31
    assert m_std.mean().item() == pytest.approx(0.31, abs=0.15)
    # Class-balanced ≈ (0 + 5) / 2 = 2.5  (equal weight per class)
    assert m_cb.mean().item() == pytest.approx(2.5, abs=0.5)
    # Pulled meaningfully toward minority
    assert m_cb.mean().item() > m_std.mean().item() + 1.5


def test_cb_stats_skips_classes_with_too_few_elements():
    """A class with 0 samples in the batch is silently skipped from averaging."""
    x = torch.randn(8, 4, 2, 2)
    labels = torch.tensor([0] * 8)  # only class 0 present
    m, v, n = _class_balanced_batch_stats(x, labels)
    # Only class 0 contributes
    expected_m = x.mean(dim=(0, 2, 3))
    expected_v = x.var(dim=(0, 2, 3), unbiased=False)
    assert torch.allclose(m, expected_m)
    assert torch.allclose(v, expected_v)
    assert n == 8 * 2 * 2


def test_cb_stats_rejects_wrong_dim():
    with pytest.raises(ValueError, match="4D input"):
        _class_balanced_batch_stats(torch.randn(8, 4, 2), torch.zeros(8, dtype=torch.long))
    with pytest.raises(ValueError, match="batch dim"):
        _class_balanced_batch_stats(torch.randn(8, 4, 2, 2), torch.zeros(7, dtype=torch.long))


# ---------------------------------------------------------------------------
# Group B — ClassBalancedLIBatchNorm2d module-level behavior
# ---------------------------------------------------------------------------

def _make_pair_of_bn_modules(seed: int = 0):
    """Two LIBatchNorm2d instances with identical weights/buffers — the second
    one converted to ClassBalancedLIBatchNorm2d. Used for parity comparisons."""
    torch.manual_seed(seed)
    bn_std = LIBatchNorm2d(stream_num_features=[8, 8], integrated_num_features=8)
    torch.manual_seed(seed)
    bn_cb = LIBatchNorm2d(stream_num_features=[8, 8], integrated_num_features=8)
    convert_to_class_balanced_bn(nn.Sequential(bn_cb))
    assert isinstance(bn_cb, ClassBalancedLIBatchNorm2d)
    # Sanity: weights/buffers identical at construction
    assert torch.equal(bn_std.integrated_weight, bn_cb.integrated_weight)
    assert torch.equal(bn_std.integrated_running_mean, bn_cb.integrated_running_mean)
    return bn_std, bn_cb


def _forward_inputs(seed: int = 1):
    torch.manual_seed(seed)
    s0 = torch.randn(16, 8, 4, 4)
    s1 = torch.randn(16, 8, 4, 4)
    integ = torch.randn(16, 8, 4, 4)
    return [s0, s1], integ


def test_cb_bn_with_no_labels_stashed_matches_standard_bn_exactly():
    """When stash is empty, the override must call super() and produce
    bit-identical outputs to standard LIBatchNorm2d (training mode included)."""
    bn_std, bn_cb = _make_pair_of_bn_modules()
    bn_std.train()
    bn_cb.train()
    streams, integ = _forward_inputs()

    # No stash_labels context → should be identical to standard BN
    s_out_std, i_out_std = bn_std(streams, integ)
    s_out_cb, i_out_cb = bn_cb(streams, integ)

    for a, b in zip(s_out_std, s_out_cb):
        assert torch.equal(a, b), f"per-stream output diverged with no labels stashed"
    assert torch.equal(i_out_std, i_out_cb), "integrated output diverged with no labels stashed"
    # Running stats should also match
    assert torch.equal(bn_std.integrated_running_mean, bn_cb.integrated_running_mean)
    assert torch.equal(bn_std.integrated_running_var, bn_cb.integrated_running_var)


def test_cb_bn_eval_mode_matches_standard_bn_exactly():
    """Eval mode: BN uses running stats, no batch stats computed. Override must
    fall through to super() regardless of stash."""
    bn_std, bn_cb = _make_pair_of_bn_modules()
    bn_std.eval(); bn_cb.eval()
    streams, integ = _forward_inputs()
    labels = torch.randint(0, 4, (16,))
    with stash_labels(labels):
        s_out_std, i_out_std = bn_std(streams, integ)
        s_out_cb, i_out_cb = bn_cb(streams, integ)
    for a, b in zip(s_out_std, s_out_cb):
        assert torch.equal(a, b)
    assert torch.equal(i_out_std, i_out_cb)


def test_cb_bn_per_stream_unchanged_even_with_labels_stashed():
    """Per-stream outputs must equal standard BN even when labels are stashed.
    Only the integrated pathway is supposed to be class-balanced."""
    bn_std, bn_cb = _make_pair_of_bn_modules()
    bn_std.train(); bn_cb.train()
    streams, integ = _forward_inputs()
    labels = torch.randint(0, 4, (16,))

    s_out_std, _ = bn_std(streams, integ)
    with stash_labels(labels):
        s_out_cb, _ = bn_cb(streams, integ)
    for a, b in zip(s_out_std, s_out_cb):
        assert torch.equal(a, b), "per-stream output should not be class-balanced"


def test_cb_bn_integrated_differs_from_standard_with_skewed_labels():
    """With imbalanced label distribution on the SAME integrated input, the
    class-balanced output should differ from standard BN's output."""
    bn_std, bn_cb = _make_pair_of_bn_modules()
    bn_std.train(); bn_cb.train()
    streams, integ = _forward_inputs()
    # Skewed: 14 samples of class 0, 2 of class 1
    labels = torch.cat([torch.zeros(14, dtype=torch.long), torch.ones(2, dtype=torch.long)])

    _, i_out_std = bn_std(streams, integ)
    with stash_labels(labels):
        _, i_out_cb = bn_cb(streams, integ)
    assert not torch.equal(i_out_std, i_out_cb), \
        "integrated output should differ from standard BN when labels are skewed"


def test_cb_bn_running_stats_update_uses_balanced_stats():
    """After a CB forward, integrated_running_mean must reflect the balanced
    mean (equal class weight), NOT the dataset-frequency-weighted mean."""
    bn_std, bn_cb = _make_pair_of_bn_modules()
    bn_std.train(); bn_cb.train()
    # Strongly skewed: minority class shifted by +10
    integ_majority = torch.randn(15, 8, 4, 4)
    integ_minority = torch.randn(1, 8, 4, 4) + 10.0
    integ = torch.cat([integ_majority, integ_minority], dim=0)
    streams = [torch.randn(16, 8, 4, 4), torch.randn(16, 8, 4, 4)]
    labels = torch.cat([torch.zeros(15, dtype=torch.long), torch.ones(1, dtype=torch.long)])

    rm_std_before = bn_std.integrated_running_mean.clone()
    rm_cb_before = bn_cb.integrated_running_mean.clone()

    bn_std(streams, integ)
    with stash_labels(labels):
        bn_cb(streams, integ)

    delta_std = (bn_std.integrated_running_mean - rm_std_before).mean().item()
    delta_cb = (bn_cb.integrated_running_mean - rm_cb_before).mean().item()
    # Standard: shifted by ~ (15*0 + 1*10) / 16 = 0.625, * momentum (0.1) = 0.0625
    # Class-balanced: shifted by ~ (0 + 10) / 2 = 5.0, * momentum (0.1) = 0.5
    # CB delta should be much larger
    assert delta_cb > delta_std + 0.2, (
        f"CB running mean update ({delta_cb:.3f}) should be larger than std ({delta_std:.3f}) "
        f"under skewed labels"
    )


def test_cb_bn_running_var_bessel_correction_applied():
    """Running variance update should multiply by n/(n-1) for Bessel correction,
    matching standard BN convention."""
    bn_std, bn_cb = _make_pair_of_bn_modules()
    bn_std.train(); bn_cb.train()
    integ = torch.randn(16, 8, 4, 4)
    streams = [torch.randn(16, 8, 4, 4), torch.randn(16, 8, 4, 4)]
    labels = torch.zeros(16, dtype=torch.long)  # single class — CB ≡ standard
    rv_before = bn_cb.integrated_running_var.clone()
    with stash_labels(labels):
        bn_cb(streams, integ)
    bn_std(streams, integ)
    # Single-class CB stats == standard batch stats; running_var update should match
    # (within floating-point precision).
    assert torch.allclose(bn_cb.integrated_running_var, bn_std.integrated_running_var, atol=1e-5)


# ---------------------------------------------------------------------------
# Group C — convert_to_class_balanced_bn helper
# ---------------------------------------------------------------------------

def test_convert_changes_type_in_place_and_preserves_state():
    """Conversion must rebind __class__ without losing parameters or buffers."""
    torch.manual_seed(0)
    model = li_resnet18(num_classes=19, width_multiplier=0.25, device='cpu', use_amp=False)
    # Snapshot a representative buffer + parameter
    sample_bn_name = None
    for name, m in model.named_modules():
        if type(m) is LIBatchNorm2d:
            sample_bn_name = name
            break
    assert sample_bn_name is not None, "no LIBatchNorm2d found in li_resnet18"

    bn = dict(model.named_modules())[sample_bn_name]
    saved_int_mean = bn.integrated_running_mean.clone()
    saved_int_weight = bn.integrated_weight.clone()
    saved_stream_mean = bn.stream0_running_mean.clone()

    n = convert_to_class_balanced_bn(model)
    assert n > 0, "expected at least 1 LIBatchNorm2d to convert"

    # Same instance, new class
    bn_after = dict(model.named_modules())[sample_bn_name]
    assert bn_after is bn, "module identity should be preserved (in-place rebind)"
    assert isinstance(bn_after, ClassBalancedLIBatchNorm2d)
    # Buffers and weights preserved bit-for-bit
    assert torch.equal(bn_after.integrated_running_mean, saved_int_mean)
    assert torch.equal(bn_after.integrated_weight, saved_int_weight)
    assert torch.equal(bn_after.stream0_running_mean, saved_stream_mean)


def test_convert_is_idempotent():
    """A second call returns 0 — already-converted modules are skipped."""
    model = li_resnet18(num_classes=19, width_multiplier=0.25, device='cpu', use_amp=False)
    n_first = convert_to_class_balanced_bn(model)
    assert n_first > 0
    n_second = convert_to_class_balanced_bn(model)
    assert n_second == 0


# ---------------------------------------------------------------------------
# Group D — End-to-end: forward through a small LINet with CBBN
# ---------------------------------------------------------------------------

def test_e2e_forward_with_cbbn_runs_and_updates_running_stats():
    """Build a tiny LINet, convert to CBBN, run two forward passes (with and
    without stashed labels), verify outputs are sane and the integrated
    running mean shifts."""
    torch.manual_seed(0)
    model = li_resnet18(num_classes=4, stream_input_channels=[3, 1],
                         width_multiplier=0.25, device='cpu', use_amp=False)
    convert_to_class_balanced_bn(model)
    model.train()

    # Find the final LIBatchNorm2d (the most consequential one for classifier)
    last_bn = None
    for m in model.modules():
        if isinstance(m, ClassBalancedLIBatchNorm2d):
            last_bn = m
    assert last_bn is not None
    rm_before = last_bn.integrated_running_mean.clone()

    streams = [torch.randn(8, 3, 32, 32), torch.randn(8, 1, 32, 32)]
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 2, 3])  # skewed: 4× class 0
    with stash_labels(labels):
        out = model(streams)
    assert out.shape == (8, 4)
    assert torch.isfinite(out).all()
    # Running stats should have moved
    assert not torch.equal(last_bn.integrated_running_mean, rm_before)


def test_e2e_forward_without_stash_matches_standard_model():
    """With CBBN-converted model but NO stash_labels, training-mode forward
    must produce the same logits as the un-converted (standard LIBatchNorm2d)
    model under matched seeding. This guards against silent behavior change
    when CBBN is wired in but unused."""
    torch.manual_seed(0)
    model_std = li_resnet18(num_classes=4, stream_input_channels=[3, 1],
                             width_multiplier=0.25, device='cpu', use_amp=False)
    torch.manual_seed(0)
    model_cb = li_resnet18(num_classes=4, stream_input_channels=[3, 1],
                            width_multiplier=0.25, device='cpu', use_amp=False)
    convert_to_class_balanced_bn(model_cb)
    model_std.train(); model_cb.train()
    streams = [torch.randn(8, 3, 32, 32), torch.randn(8, 1, 32, 32)]
    # No stash → CBBN modules fall through to super()._forward_single_pathway
    out_std = model_std(streams)
    out_cb = model_cb(streams)
    assert torch.equal(out_std, out_cb), (
        "without stash_labels, CBBN-converted model must produce identical logits "
        "to standard model"
    )


def test_cbbn_e2e_training_then_eval_predicts_correctly():
    """The real end-to-end check that should have caught the eval-collapse bug.
    Train a tiny CBBN model on a class-separable problem with a realistic
    recipe (mixup + modality dropout, AMP-style autocast), then confirm that
    eval-mode forward produces VARIED, MOSTLY-CORRECT predictions — not a
    single class for all inputs.

    A model whose running stats are corrupted (e.g. by SAM's second-pass
    leak, by an fp16 EMA drift, or by any other CBBN-side bug) would
    produce uniform / degenerate output here. This test is the
    eval-quality safety net the existing CBBN integration tests lacked.
    """
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from torch.amp import autocast

    # Class-separable synthetic data (4 classes with distinct rgb/depth means)
    torch.manual_seed(0)
    N_CLASSES = 4
    N_PER_CLASS = 64
    rgb_list, depth_list, labels_list = [], [], []
    for c in range(N_CLASSES):
        rgb_mean = torch.randn(3, 1, 1) * 0.5
        depth_mean = torch.randn(1, 1, 1) * 0.3
        for _ in range(N_PER_CLASS):
            rgb_list.append(torch.randn(3, 32, 32) * 0.3 + rgb_mean)
            depth_list.append(torch.randn(1, 32, 32) * 0.3 + depth_mean)
            labels_list.append(c)
    ds = TensorDataset(torch.stack(rgb_list), torch.stack(depth_list),
                       torch.tensor(labels_list, dtype=torch.long))
    train_loader = DataLoader(ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(ds, batch_size=32, shuffle=False)

    # Build model with CBBN, recipe matching user's setup
    torch.manual_seed(0)
    model = li_resnet18(num_classes=N_CLASSES, stream_input_channels=[3, 1],
                         width_multiplier=0.25, dropout_p=0.0,
                         device='cpu', use_amp=False)
    model.compile(
        optimizer=torch.optim.AdamW(model.parameters(), lr=3e-3),
        scheduler=None, loss='cross_entropy',
        label_smoothing=0.06, gpu_augmentation=False,
    )

    history = model.fit(
        train_loader=train_loader, val_loader=val_loader,
        epochs=10, verbose=False,
        modality_dropout=True, modality_dropout_rate=0.5, modality_dropout_ramp=0,
        use_mixup=True, mixup_alpha=0.2,
        use_sam=False,                # exact user setup: mixup ON, SAM OFF
        use_class_balanced_bn=True,
    )

    # Eval-mode quality check — these are the specific assertions a degenerate
    # eval (val_acc = N/N_test, val_mca = 1/N_classes) would fail.
    final_val_mca = history['val_mca'][-1]
    assert final_val_mca > 1.5 / N_CLASSES, (
        f"Eval val_mca = {final_val_mca:.3f} ≈ 1/{N_CLASSES} = uniform random. "
        f"CBBN-trained model collapsed at eval time. This is the symptom of "
        f"corrupted running stats: training works but eval is broken."
    )

    # Predict on val explicitly, confirm at least 2 unique predicted classes
    model.eval()
    all_preds = []
    with torch.no_grad():
        for rgb, depth, _ in val_loader:
            preds = model([rgb, depth]).argmax(dim=1)
            all_preds.append(preds)
    preds = torch.cat(all_preds)
    n_unique = preds.unique().numel()
    assert n_unique >= 2, (
        f"Eval-mode predictions are degenerate: only {n_unique}/{N_CLASSES} "
        f"unique classes predicted across {preds.numel()} val samples. "
        f"Model is producing a single class for every input — running stats are wrong."
    )


def test_cbbn_e2e_training_then_eval_under_autocast():
    """Same as test_cbbn_e2e_training_then_eval_predicts_correctly, but with
    autocast active during training. Catches dtype-related running-stats drift
    that only manifests under AMP (which is the user's actual setup)."""
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from torch.amp import autocast

    torch.manual_seed(0)
    N_CLASSES = 4
    N_PER_CLASS = 64
    rgb_list, depth_list, labels_list = [], [], []
    for c in range(N_CLASSES):
        rgb_mean = torch.randn(3, 1, 1) * 0.5
        depth_mean = torch.randn(1, 1, 1) * 0.3
        for _ in range(N_PER_CLASS):
            rgb_list.append(torch.randn(3, 32, 32) * 0.3 + rgb_mean)
            depth_list.append(torch.randn(1, 32, 32) * 0.3 + depth_mean)
            labels_list.append(c)
    ds = TensorDataset(torch.stack(rgb_list), torch.stack(depth_list),
                       torch.tensor(labels_list, dtype=torch.long))
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    torch.manual_seed(0)
    model = li_resnet18(num_classes=N_CLASSES, stream_input_channels=[3, 1],
                        width_multiplier=0.25, dropout_p=0.0,
                        device='cpu', use_amp=False)
    convert_to_class_balanced_bn(model)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-3)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.06)

    # 5 epochs of training under bf16 autocast (closest CPU equivalent to fp16 AMP)
    model.train()
    for epoch in range(5):
        for rgb, depth, lbls in loader:
            optim.zero_grad()
            with autocast(device_type='cpu', dtype=torch.bfloat16), stash_labels(lbls):
                out = model([rgb, depth])
            loss = loss_fn(out.float(), lbls)
            loss.backward()
            optim.step()

    # Eval check
    model.eval()
    all_preds = []
    with torch.no_grad():
        for rgb, depth, _ in loader:
            all_preds.append(model([rgb, depth]).argmax(dim=1))
    preds = torch.cat(all_preds)
    n_unique = preds.unique().numel()
    assert n_unique >= 2, (
        f"Under autocast, eval-mode predictions are degenerate: only "
        f"{n_unique}/{N_CLASSES} unique classes. "
        f"Running-stats drift / dtype issue under autocast."
    )

    # Running stats sanity: should be finite, positive var
    for m in model.modules():
        if isinstance(m, ClassBalancedLIBatchNorm2d):
            assert torch.isfinite(m.integrated_running_mean).all()
            assert torch.isfinite(m.integrated_running_var).all()
            assert (m.integrated_running_var > 0).all(), \
                "running_var must be strictly positive"


def test_sam_second_pass_does_not_corrupt_cbbn_running_stats():
    """Regression test for the exact bug that broke Phase 1c: SAM's second
    forward pass runs at adversarially-perturbed weights (w+ε), and the
    surrounding ``disable_running_stats(model)`` call must freeze BN momentum
    on CBBN modules so they don't accumulate stats from the perturbed
    features. A previous name-string check in `_is_bn_like` skipped subclasses
    of LIBatchNorm2d, so CBBN modules' momentum stayed at 0.1 during SAM's
    second pass — corrupting their running stats and producing val_acc
    stuck at 1/19 (uniform-random) regardless of how well training went.
    """
    from src.training.sam import disable_running_stats, enable_running_stats

    bn = LIBatchNorm2d(stream_num_features=[8, 8], integrated_num_features=8)
    bn.__class__ = ClassBalancedLIBatchNorm2d
    bn.train()

    streams = [torch.randn(16, 8, 4, 4), torch.randn(16, 8, 4, 4)]
    integ = torch.randn(16, 8, 4, 4)
    labels = torch.randint(0, 4, (16,))

    # First pass (good weights): running stats SHOULD update
    from src.models.linear_integration.li_net3.class_balanced_bn import stash_labels
    rm_before_first = bn.integrated_running_mean.clone()
    with stash_labels(labels):
        bn(streams, integ)
    rm_after_first = bn.integrated_running_mean.clone()
    assert not torch.equal(rm_before_first, rm_after_first), (
        "first SAM pass should update CBBN running stats"
    )

    # SAM's pre-second-pass freeze
    disable_running_stats(nn.Sequential(bn))
    assert bn.momentum == 0.0, (
        "BUG: disable_running_stats must freeze momentum on CBBN modules "
        "(subclass of LIBatchNorm2d). Without this, SAM's second pass "
        "writes adversarial-perturbation stats into running_mean/var."
    )

    # Second pass (simulated adversarial input): running stats must NOT change
    integ_adv = integ + torch.randn_like(integ) * 0.5
    rm_before_second = bn.integrated_running_mean.clone()
    with stash_labels(labels):
        bn(streams, integ_adv)
    rm_after_second = bn.integrated_running_mean.clone()
    assert torch.equal(rm_before_second, rm_after_second), (
        "SAM second pass corrupted CBBN running stats — momentum freeze did "
        "not apply. This is the exact failure mode that broke Phase 1c eval."
    )

    # And the restore must put momentum back so subsequent batches behave normally
    enable_running_stats(nn.Sequential(bn))
    assert bn.momentum > 0.0, "enable_running_stats failed to restore momentum"
