"""Tests for the pluggable classifier heads on LINet3.

Grouped:
    Group A — head-module tests (LinearHead, MaxoutHead in isolation)
    Group B — LINet integration tests
    Group C — behavioral test (slow)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import torch.nn as nn

from src.models.linear_integration.li_net3 import LinearHead, MaxoutHead, li_resnet18


# ---------------------------------------------------------------------------
# Group A — head-module tests
# ---------------------------------------------------------------------------

def test_linear_head_shape():
    """LinearHead returns (B, num_classes) and wraps an nn.Linear of matching shape."""
    head = LinearHead(384, 19)
    x = torch.randn(4, 384)
    out = head(x)
    assert out.shape == (4, 19), out.shape
    assert head.fc.weight.shape == (19, 384), head.fc.weight.shape
    assert head.num_classes == 19
    assert head.feat_dim == 384


def test_maxout_head_shape():
    """MaxoutHead returns (B, num_classes) (post-max), inner Linear is (C*K, feat_dim)."""
    head = MaxoutHead(384, 19, num_subnodes=3)
    x = torch.randn(4, 384)
    out = head(x)
    assert out.shape == (4, 19), out.shape  # NOT (4, 19, 3)
    assert head.fc.weight.shape == (57, 384), head.fc.weight.shape
    assert head.num_subnodes == 3
    assert head.num_classes == 19
    assert head.feat_dim == 384


def test_maxout_head_rejects_zero_subnodes():
    """MaxoutHead with num_subnodes=0 raises AssertionError."""
    with pytest.raises(AssertionError, match="num_subnodes"):
        MaxoutHead(384, 19, num_subnodes=0)


def test_maxout_k1_math_equals_linear():
    """MaxoutHead's view+amax over a singleton K dimension is mathematically identity.

    With matched fc weights, MaxoutHead(K=1) and LinearHead must produce bit-equal
    forward output. Copying weights here is correct — this is a forward-math test,
    not an equivalence-of-construction test.
    """
    linear = LinearHead(384, 19)
    maxout_k1 = MaxoutHead(384, 19, num_subnodes=1)
    maxout_k1.fc.weight.data.copy_(linear.fc.weight.data)
    maxout_k1.fc.bias.data.copy_(linear.fc.bias.data)
    x = torch.randn(8, 384)
    assert torch.equal(linear(x), maxout_k1(x))


# ---------------------------------------------------------------------------
# Group B — LINet integration tests
# ---------------------------------------------------------------------------

def _build_small(**kwargs):
    """Construct a small LINet for fast tests."""
    return li_resnet18(
        num_classes=19,
        width_multiplier=0.75,
        dropout_p=0.0,
        device='cpu',
        **kwargs,
    )


def test_default_is_linear_head():
    """Default LINet construction uses LinearHead and (B, num_classes) output."""
    model = _build_small()
    # Attribute name set by LINet (not `classifier_head`, which would collide
    # with the head module instance at `self.fc`).
    assert model.classifier_head_type == 'linear'
    assert isinstance(model.fc, LinearHead), type(model.fc)
    assert model.fc.fc.weight.shape == (19, 384), model.fc.fc.weight.shape

    streams = [torch.randn(2, 3, 224, 224), torch.randn(2, 1, 224, 224)]
    out = model(streams)
    assert out.shape == (2, 19), out.shape


def test_unknown_head_raises():
    """Unknown classifier_head value raises ValueError mentioning the bad string and valid options."""
    with pytest.raises(ValueError) as exc_info:
        _build_small(classifier_head='unknown')
    msg = str(exc_info.value)
    assert 'unknown' in msg, msg
    assert 'linear' in msg and 'maxout' in msg, msg


def test_maxout_k1_strict_construction_equals_linear():
    """Strict construction equality — no weight copying.

    Seed RNG, build linear default and maxout-K=1 independently, assert
    torch.equal on fc weights, on a backbone weight (catches upstream RNG
    drift), and on forward output. Proves maxout-K=1 consumes the same RNG
    amount as the linear path so weights are naturally identical.
    """
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    m_linear = _build_small()  # defaults: classifier_head='linear'

    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    m_maxout_k1 = _build_small(classifier_head='maxout', num_subnodes=1)

    # Both heads' inner Linear should have identical weights via RNG consumption order
    assert torch.equal(m_linear.fc.fc.weight, m_maxout_k1.fc.fc.weight)
    assert torch.equal(m_linear.fc.fc.bias, m_maxout_k1.fc.fc.bias)
    # Backbone weight equality confirms no upstream RNG drift
    assert torch.equal(m_linear.conv1.stream_weights[0], m_maxout_k1.conv1.stream_weights[0])

    # Forward equality on identical input
    m_linear.eval(); m_maxout_k1.eval()
    torch.manual_seed(123)
    streams = [torch.randn(2, 3, 224, 224), torch.randn(2, 1, 224, 224)]
    with torch.no_grad():
        out_linear = m_linear(streams)
        out_maxout = m_maxout_k1(streams)
    assert torch.equal(out_linear, out_maxout), \
        f"forward mismatch: max abs diff = {(out_linear - out_maxout).abs().max()}"


# ---------------------------------------------------------------------------
# Group C — behavioral test (slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_maxout_k3_argmax_specialization():
    """MaxoutHead specializes its sub-nodes on a synthetic XOR-like 2-mode-per-class task.

    Setup: 384-dim features sampled from XOR-like Gaussian clusters where each
    class has two modes that are NOT linearly separable from another class's
    modes using a single hyperplane. K=1 cannot solve it; K=2+ can.

    Precondition: maxout's loss must drop below half of linear's loss after
    training. If this fails, the synthetic setup is wrong (clusters too close,
    feature dim too high, training too short) — fix the setup, do not relax
    the assertion.

    Behavioral assertion: at inference, the two clusters of the same class
    route to different sub-nodes >70% of the time.
    """
    torch.manual_seed(42)
    feat_dim = 384
    num_classes = 4
    num_subnodes = 3
    samples_per_cluster = 256

    # XOR-like cluster centers in two leading dims; rest are noise.
    # Class A: (+1, +1) and (-1, -1). Class B: (+1, -1) and (-1, +1).
    # Class C: (+1, +0) and (-1, +0). Class D: (+0, +1) and (+0, -1).
    centers = {
        0: [(+1.0, +1.0), (-1.0, -1.0)],
        1: [(+1.0, -1.0), (-1.0, +1.0)],
        2: [(+1.0,  0.0), (-1.0,  0.0)],
        3: [( 0.0, +1.0), ( 0.0, -1.0)],
    }
    cluster_scale = 3.0  # spread between modes
    noise_scale = 0.3

    feats, labels, cluster_ids = [], [], []
    for c, two_centers in centers.items():
        for cluster_idx, (cx, cy) in enumerate(two_centers):
            f = torch.randn(samples_per_cluster, feat_dim) * noise_scale
            f[:, 0] += cluster_scale * cx
            f[:, 1] += cluster_scale * cy
            feats.append(f)
            labels.append(torch.full((samples_per_cluster,), c, dtype=torch.long))
            cluster_ids.append(torch.full((samples_per_cluster,), cluster_idx, dtype=torch.long))
    X = torch.cat(feats, 0)
    y = torch.cat(labels, 0)
    cid = torch.cat(cluster_ids, 0)

    perm = torch.randperm(X.size(0))
    X, y, cid = X[perm], y[perm], cid[perm]

    def train_head(head, steps=400, lr=0.05):
        opt = torch.optim.Adam(head.parameters(), lr=lr)
        ce = nn.CrossEntropyLoss()
        head.train()
        last = None
        for step in range(steps):
            idx = torch.randint(0, X.size(0), (128,))
            logits = head(X[idx])
            loss = ce(logits, y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            last = loss.item()
        return last

    torch.manual_seed(0)
    linear = LinearHead(feat_dim, num_classes)
    torch.manual_seed(0)
    maxout = MaxoutHead(feat_dim, num_classes, num_subnodes=num_subnodes)

    loss_linear = train_head(linear)
    loss_maxout = train_head(maxout)

    # Precondition: maxout must clearly outperform linear on this XOR-like task.
    assert loss_maxout < 0.5 * loss_linear, (
        f"Setup precondition failed: maxout loss {loss_maxout:.3f} not < "
        f"0.5 * linear loss {loss_linear:.3f}. The synthetic data is not "
        f"separable enough to demonstrate specialization — fix the setup."
    )

    # Behavioral assertion: per class, samples from the two modes route to
    # different sub-nodes more often than a random K-way head would. With K=3
    # and no specialization, two random samples agree by chance ~1/K = 33% of
    # the time → disagreement baseline ~67%. Specialization should clearly
    # exceed that. The plan's bar is >70% cross-cluster disagreement averaged
    # across classes.
    maxout.eval()
    per_class_disagreement = []
    with torch.no_grad():
        for c in range(num_classes):
            mask_c = (y == c)
            mask_c0 = mask_c & (cid == 0)
            mask_c1 = mask_c & (cid == 1)
            pre_max_c0 = maxout.fc(X[mask_c0]).view(-1, num_classes, num_subnodes)
            pre_max_c1 = maxout.fc(X[mask_c1]).view(-1, num_classes, num_subnodes)
            sub_c0 = pre_max_c0[:, c, :].argmax(dim=1)
            sub_c1 = pre_max_c1[:, c, :].argmax(dim=1)
            # Cross-cluster disagreement: P(sub-node-of-c0-sample != sub-node-of-c1-sample)
            # under independent draws from each cluster's argmax distribution.
            n0 = torch.bincount(sub_c0, minlength=num_subnodes).float() / sub_c0.numel()
            n1 = torch.bincount(sub_c1, minlength=num_subnodes).float() / sub_c1.numel()
            agree_prob = float((n0 * n1).sum())  # P(same sub-node)
            per_class_disagreement.append(1.0 - agree_prob)

    mean_disagreement = sum(per_class_disagreement) / num_classes
    assert mean_disagreement >= 0.7, (
        f"Sub-node specialization failed: mean cross-cluster disagreement "
        f"{mean_disagreement:.3f} < 0.7. Per-class: {per_class_disagreement}"
    )
