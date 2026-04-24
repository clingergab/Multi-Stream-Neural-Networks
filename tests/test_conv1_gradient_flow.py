"""
Tests for gradient flow through LINet3's first layer (conv1) using synthetic data.

LINet3 is a multi-stream neural network where RGB and depth streams are processed
independently and integrated via 1x1 convolutions. The first layer (conv1) creates
the integrated stream from scratch. These tests verify that gradients properly reach
conv1's stream_weights and integration_from_streams parameters, including under
modality dropout and skip_stream_tail conditions.

Tests:
1. conv1 autograd graph integrity (gradients reach conv1 stream_weights)
2. conv1 gradients with modality dropout (various blanking patterns)
3. skip_stream_tail gradient flow (layer4[-1] still gets gradients)
4. gradient cliff structure (gradient magnitude table across all layers)
"""

import sys
import os

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.linear_integration.li_net3 import li_resnet18


# ---------------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------------
DEVICE = 'cpu'
NUM_CLASSES = 10
BATCH_SIZE = 4
IMAGE_SIZE = 224


def _make_synthetic_batch():
    """Create a synthetic batch of RGB + depth inputs and targets."""
    rgb = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
    depth = torch.randn(BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE)
    targets = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))
    return rgb, depth, targets


def _assert_grad_healthy(grad, name):
    """Assert that a gradient tensor is not None, not all zeros, and has no NaN/Inf."""
    assert grad is not None, f"{name}.grad is None!"
    assert grad.abs().sum() > 0, f"{name}.grad is all zeros!"
    assert not torch.isnan(grad).any(), f"{name}.grad has NaN!"
    assert not torch.isinf(grad).any(), f"{name}.grad has Inf!"


# ---------------------------------------------------------------------------
# Phase 1a: Autograd graph integrity
# ---------------------------------------------------------------------------
def test_conv1_autograd_graph_integrity():
    """Verify gradients reach conv1 stream_weights through integration path."""
    model = li_resnet18(num_classes=NUM_CLASSES, stream_input_channels=[3, 1], device=DEVICE)
    model.train()

    rgb, depth, targets = _make_synthetic_batch()

    logits = model([rgb, depth])
    loss = F.cross_entropy(logits, targets)
    loss.backward()

    # conv1 stream weights must have non-None, non-zero gradients
    for i in range(2):
        _assert_grad_healthy(
            model.conv1.stream_weights[i].grad,
            f"conv1.stream_weights[{i}]",
        )

    # integration_from_streams at conv1 must also have gradients
    for i in range(2):
        _assert_grad_healthy(
            model.conv1.integration_from_streams[i].grad,
            f"conv1.integration_from_streams[{i}]",
        )


# ---------------------------------------------------------------------------
# Phase 1b: Gradients with modality dropout
# ---------------------------------------------------------------------------
def test_conv1_gradients_with_modality_dropout():
    """Test forward+backward with blanked_mask active (partial blanking)."""
    model = li_resnet18(num_classes=NUM_CLASSES, stream_input_channels=[3, 1], device=DEVICE)
    model.train()

    rgb, depth, targets = _make_synthetic_batch()

    # Blank RGB for first 2 samples, depth never blanked
    blanked_mask = {
        0: torch.tensor([True, True, False, False]),
        1: torch.tensor([False, False, False, False]),
    }

    logits = model([rgb, depth], blanked_mask=blanked_mask)
    loss = F.cross_entropy(logits, targets)
    loss.backward()

    for i in range(2):
        _assert_grad_healthy(
            model.conv1.stream_weights[i].grad,
            f"conv1.stream_weights[{i}] (partial blank)",
        )


def test_conv1_gradients_with_both_streams_blanked():
    """Test that gradients survive when both streams are blanked for some samples."""
    model = li_resnet18(num_classes=NUM_CLASSES, stream_input_channels=[3, 1], device=DEVICE)
    model.train()

    rgb, depth, targets = _make_synthetic_batch()

    # Both streams blanked for sample 0, only RGB blanked for sample 1,
    # only depth blanked for sample 2, neither blanked for sample 3
    blanked_mask = {
        0: torch.tensor([True, True, False, False]),
        1: torch.tensor([True, False, True, False]),
    }

    logits = model([rgb, depth], blanked_mask=blanked_mask)
    loss = F.cross_entropy(logits, targets)
    loss.backward()

    for i in range(2):
        _assert_grad_healthy(
            model.conv1.stream_weights[i].grad,
            f"conv1.stream_weights[{i}] (both blanked)",
        )


def test_conv1_gradients_with_alternating_blanking():
    """Test with alternating blanking pattern across streams."""
    model = li_resnet18(num_classes=NUM_CLASSES, stream_input_channels=[3, 1], device=DEVICE)
    model.train()

    rgb, depth, targets = _make_synthetic_batch()

    # Alternating: when RGB is blanked, depth is not, and vice versa
    blanked_mask = {
        0: torch.tensor([True, False, True, False]),
        1: torch.tensor([False, True, False, True]),
    }

    logits = model([rgb, depth], blanked_mask=blanked_mask)
    loss = F.cross_entropy(logits, targets)
    loss.backward()

    for i in range(2):
        _assert_grad_healthy(
            model.conv1.stream_weights[i].grad,
            f"conv1.stream_weights[{i}] (alternating blank)",
        )


# ---------------------------------------------------------------------------
# Phase 2b: skip_stream_tail gradient flow
# ---------------------------------------------------------------------------
def test_skip_stream_tail_gradient_flow():
    """Verify that layer4[-1] stream weights get gradients with skip_stream_tail=True."""
    model = li_resnet18(num_classes=NUM_CLASSES, stream_input_channels=[3, 1], device=DEVICE)
    model.train()

    assert model.layer4[-1].skip_stream_tail is True, (
        "Expected layer4[-1].skip_stream_tail to be True"
    )

    rgb, depth, targets = _make_synthetic_batch()

    logits = model([rgb, depth])
    loss = F.cross_entropy(logits, targets)
    loss.backward()

    # Check layer4[-1].conv1 and conv2 stream weights have gradients
    for conv_name in ("conv1", "conv2"):
        conv = getattr(model.layer4[-1], conv_name)
        for i in range(2):
            _assert_grad_healthy(
                conv.stream_weights[i].grad,
                f"layer4[-1].{conv_name}.stream_weights[{i}]",
            )


# ---------------------------------------------------------------------------
# Phase 3a: Gradient cliff structure
# ---------------------------------------------------------------------------
def test_gradient_cliff_structure():
    """Build gradient magnitude table across layers and verify no None gradients."""
    model = li_resnet18(num_classes=NUM_CLASSES, stream_input_channels=[3, 1], device=DEVICE)
    model.train()

    rgb, depth, targets = _make_synthetic_batch()

    logits = model([rgb, depth])
    loss = F.cross_entropy(logits, targets)
    loss.backward()

    layers = [
        ("conv1", model.conv1),
        ("layer1.0.conv1", model.layer1[0].conv1),
        ("layer2.0.conv1", model.layer2[0].conv1),
        ("layer3.0.conv1", model.layer3[0].conv1),
        ("layer4.0.conv1", model.layer4[0].conv1),
    ]

    print(
        f"\n{'Layer':<20} | {'stream_wt grad':>15} | "
        f"{'integ_from_str grad':>20} | {'integrated_wt grad':>20}"
    )
    print("-" * 85)

    for name, conv in layers:
        sw_grad = conv.stream_weights[0].grad
        assert sw_grad is not None, f"{name}.stream_weights[0].grad is None!"
        sw_val = sw_grad.abs().mean().item()

        ifs_grad = conv.integration_from_streams[0].grad
        assert ifs_grad is not None, f"{name}.integration_from_streams[0].grad is None!"
        ifs_val = ifs_grad.abs().mean().item()

        if conv.integrated_weight.numel() > 0:
            iw_grad = conv.integrated_weight.grad
            assert iw_grad is not None, f"{name}.integrated_weight.grad is None!"
            iw_val = iw_grad.abs().mean().item()
            iw_str = f"{iw_val:.2e}"
        else:
            iw_str = "N/A (empty)"

        print(f"{name:<20} | {sw_val:>15.2e} | {ifs_val:>20.2e} | {iw_str:>20}")

        # All stream_weights grads must be > 0 and not NaN
        assert sw_val > 0, f"{name}.stream_weights[0].grad is all zeros!"
        assert not torch.isnan(sw_grad).any(), f"{name}.stream_weights[0].grad has NaN!"

    # Also print fc — head module wraps the inner nn.Linear at .fc.fc
    fc_grad = model.fc.fc.weight.grad
    assert fc_grad is not None, "fc.fc.weight.grad is None!"
    fc_val = fc_grad.abs().mean().item()
    print(f"{'fc.fc.weight':<20} | {fc_val:>15.2e} | {'N/A':>20} | {'N/A':>20}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
ALL_TESTS = [
    test_conv1_autograd_graph_integrity,
    test_conv1_gradients_with_modality_dropout,
    test_conv1_gradients_with_both_streams_blanked,
    test_conv1_gradients_with_alternating_blanking,
    test_skip_stream_tail_gradient_flow,
    test_gradient_cliff_structure,
]


def run_all():
    """Run all tests and print results."""
    print("=" * 85)
    print("TEST SUITE: conv1 gradient flow through LINet3")
    print("=" * 85)

    passed = 0
    failed = 0

    for test_fn in ALL_TESTS:
        name = test_fn.__name__
        try:
            test_fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1

    print("-" * 85)
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 85)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    run_all()
