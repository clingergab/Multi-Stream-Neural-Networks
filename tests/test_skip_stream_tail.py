"""
Tests for skip_stream_tail optimization on LINet's last residual block.

When skip_stream_tail=True on the last block of layer4, stream BN+ReLU+residual
are skipped since stream outputs are discarded after layer4. Only the integrated
pathway gets BN + residual + ReLU before classification.

Tests:
1. Flag placement — only layer4[-1] has skip_stream_tail=True
2. Forward pass produces correct output shape (eval and train)
3. Backward pass completes without error
4. forward_stream() still works on layer4[-1] (separate code path)
5. Stream BN stats are NOT updated on the last block during training
6. Integrated BN stats ARE updated on the last block during training
7. Works with downsample (layers=[1,1,1,1] edge case)
8. Works with Bottleneck blocks
9. forward_integrated() on LIBatchNorm2d works correctly
"""

import sys
import os

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.linear_integration.li_net3.li_net import LINet
from src.models.linear_integration.li_net3.blocks import LIBasicBlock, LIBottleneck
from src.models.linear_integration.li_net3.conv import LIBatchNorm2d


def create_model(layers=(2, 2, 2, 2), width_multiplier=0.25, block=LIBasicBlock):
    """Create a small LINet for testing."""
    return LINet(
        block=block,
        layers=list(layers),
        num_classes=5,
        stream_input_channels=[3, 1],
        device='cpu',
        use_amp=False,
        width_multiplier=width_multiplier,
    )


def create_inputs(batch_size=2, size=32):
    """Create small synthetic inputs."""
    return [torch.randn(batch_size, 3, size, size),
            torch.randn(batch_size, 1, size, size)]


def test_flag_placement():
    """Only layer4[-1] should have skip_stream_tail=True."""
    model = create_model()
    for name in ['layer1', 'layer2', 'layer3']:
        layer = getattr(model, name)
        for i, block in enumerate(layer):
            assert block.skip_stream_tail == False, \
                f"{name}[{i}].skip_stream_tail should be False"

    # layer4: all blocks False except the last
    for i, block in enumerate(model.layer4[:-1]):
        assert block.skip_stream_tail == False, \
            f"layer4[{i}].skip_stream_tail should be False"
    assert model.layer4[-1].skip_stream_tail == True, \
        "layer4[-1].skip_stream_tail should be True"
    print("PASS: test_flag_placement")


def test_forward_eval():
    """Forward pass in eval mode produces correct output shape."""
    model = create_model()
    model.eval()
    out = model(create_inputs())
    assert out.shape == (2, 5), f"Expected (2, 5), got {out.shape}"
    print("PASS: test_forward_eval")


def test_forward_train():
    """Forward pass in train mode produces correct output shape."""
    model = create_model()
    model.train()
    out = model(create_inputs())
    assert out.shape == (2, 5), f"Expected (2, 5), got {out.shape}"
    print("PASS: test_forward_train")


def test_backward():
    """Backward pass completes without error."""
    model = create_model()
    model.train()
    out = model(create_inputs())
    out.sum().backward()
    # Verify gradients exist on stream conv weights (integration path still works)
    last_block = model.layer4[-1]
    for i, w in enumerate(last_block.conv2.stream_weights):
        assert w.grad is not None, f"stream_weight[{i}] should have grad"
        assert w.grad.abs().sum() > 0, f"stream_weight[{i}] grad should be non-zero"
    print("PASS: test_backward")


def test_forward_stream_still_works():
    """forward_stream() on layer4[-1] uses a separate code path and should work."""
    model = create_model()
    model.eval()
    # Get appropriate input shape for layer4 blocks
    w = 0.25
    channels = int(512 * w)  # 128 at 0.25x
    stream_x = torch.randn(1, channels, 4, 4)
    out = model.layer4[-1].forward_stream(0, stream_x)
    assert out.shape == stream_x.shape, \
        f"Expected {stream_x.shape}, got {out.shape}"
    print("PASS: test_forward_stream_still_works")


def test_stream_bn_stats_not_updated():
    """Stream BN running stats on layer4[-1] should NOT update during training."""
    model = create_model()
    last_block = model.layer4[-1]

    # Record stream BN running stats before training
    stream0_mean_before = last_block.bn2.stream0_running_mean.clone()
    stream0_var_before = last_block.bn2.stream0_running_var.clone()

    # Run a few forward passes in train mode
    model.train()
    for _ in range(3):
        model(create_inputs())

    stream0_mean_after = last_block.bn2.stream0_running_mean
    stream0_var_after = last_block.bn2.stream0_running_var

    assert torch.equal(stream0_mean_before, stream0_mean_after), \
        "Stream BN running_mean should not change on last block"
    assert torch.equal(stream0_var_before, stream0_var_after), \
        "Stream BN running_var should not change on last block"
    print("PASS: test_stream_bn_stats_not_updated")


def test_integrated_bn_stats_updated():
    """Integrated BN running stats on layer4[-1] SHOULD update during training."""
    model = create_model()
    last_block = model.layer4[-1]

    integrated_mean_before = last_block.bn2.integrated_running_mean.clone()

    model.train()
    for _ in range(3):
        model(create_inputs())

    integrated_mean_after = last_block.bn2.integrated_running_mean

    assert not torch.equal(integrated_mean_before, integrated_mean_after), \
        "Integrated BN running_mean should change on last block"
    print("PASS: test_integrated_bn_stats_updated")


def test_with_downsample():
    """layers=[1,1,1,1] means layer4[-1] IS layer4[0] and has a downsample."""
    model = create_model(layers=(1, 1, 1, 1))
    last_block = model.layer4[-1]

    assert last_block.downsample is not None, \
        "Single-block layer4 should have downsample"
    assert last_block.skip_stream_tail == True

    # Forward + backward should still work
    model.train()
    out = model(create_inputs())
    assert out.shape == (2, 5)
    out.sum().backward()
    print("PASS: test_with_downsample")


def test_with_bottleneck():
    """skip_stream_tail works with LIBottleneck blocks."""
    model = create_model(block=LIBottleneck)
    assert model.layer4[-1].skip_stream_tail == True

    model.train()
    out = model(create_inputs())
    assert out.shape == (2, 5)
    out.sum().backward()
    print("PASS: test_with_bottleneck")


def test_forward_integrated_method():
    """LIBatchNorm2d.forward_integrated() processes only integrated pathway."""
    bn = LIBatchNorm2d([16, 16], 16)
    bn.train()

    integrated_input = torch.randn(2, 16, 8, 8)
    out = bn.forward_integrated(integrated_input)

    assert out.shape == integrated_input.shape, \
        f"Expected {integrated_input.shape}, got {out.shape}"

    # With apply_relu
    out_relu = bn.forward_integrated(integrated_input, apply_relu=True)
    assert (out_relu >= 0).all(), "apply_relu=True should produce non-negative output"

    # Stream stats should not change (we didn't pass streams)
    stream0_mean = bn.stream0_running_mean.clone()
    bn.forward_integrated(torch.randn(2, 16, 8, 8))
    assert torch.equal(stream0_mean, bn.stream0_running_mean), \
        "forward_integrated should not update stream running stats"

    # Integrated stats should change
    integrated_mean_before = bn.integrated_running_mean.clone()
    bn.forward_integrated(torch.randn(4, 16, 8, 8))
    assert not torch.equal(integrated_mean_before, bn.integrated_running_mean), \
        "forward_integrated should update integrated running stats"
    print("PASS: test_forward_integrated_method")


if __name__ == '__main__':
    test_flag_placement()
    test_forward_eval()
    test_forward_train()
    test_backward()
    test_forward_stream_still_works()
    test_stream_bn_stats_not_updated()
    test_integrated_bn_stats_updated()
    test_with_downsample()
    test_with_bottleneck()
    test_forward_integrated_method()
    print("\n" + "=" * 40)
    print("ALL TESTS PASSED")
    print("=" * 40)
