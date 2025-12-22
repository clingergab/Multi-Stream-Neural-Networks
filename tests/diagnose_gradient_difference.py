"""
Deep dive into why gradients differ between LINet3 and Original.

Forward pass is identical, but gradients differ by ~1e-5.
This means the backward pass has a bug.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import sys
sys.path.insert(0, '/Users/gclinger/Documents/projects/Multi-Stream-Neural-Networks')

from src.models.linear_integration.li_net3.li_net import LINet as LINet3
from src.models.linear_integration.li_net3.blocks import LIBasicBlock as LIBasicBlock3
from src.models.linear_integration.li_net3.conv import LIConv2d as LIConv2d3

from src.models.linear_integration.li_net import LINet as LINetOriginal
from src.models.linear_integration.blocks import LIBasicBlock as LIBasicBlockOriginal
from src.models.linear_integration.conv import LIConv2d as LIConv2dOriginal

SEED = 42


def test_conv2d_gradient_isolated():
    """Test just the conv layers in isolation to find the gradient bug."""
    print("=" * 80)
    print("Testing LIConv2d Gradient in Isolation")
    print("=" * 80)

    torch.manual_seed(SEED)

    # Create Original conv
    conv_orig = LIConv2dOriginal(
        stream1_in_channels=3,
        stream1_out_channels=64,
        stream2_in_channels=1,
        stream2_out_channels=64,
        integrated_in_channels=0,
        integrated_out_channels=64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False
    )

    torch.manual_seed(SEED)

    # Create LINet3 conv
    conv_linet3 = LIConv2d3(
        stream_in_channels=[3, 1],
        stream_out_channels=[64, 64],
        integrated_in_channels=0,
        integrated_out_channels=64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False
    )

    # Create identical inputs
    torch.manual_seed(SEED)
    rgb = torch.randn(4, 3, 32, 32, requires_grad=True)
    depth = torch.randn(4, 1, 32, 32, requires_grad=True)

    # Clone for the other model
    rgb_clone = rgb.detach().clone().requires_grad_(True)
    depth_clone = depth.detach().clone().requires_grad_(True)

    # Forward pass - Original
    s1_orig, s2_orig, int_orig = conv_orig(rgb, depth, None)

    # Forward pass - LINet3
    streams_li3, int_li3 = conv_linet3([rgb_clone, depth_clone], None)

    print("\n--- Forward Pass Comparison ---")
    print(f"stream1 diff: {(s1_orig - streams_li3[0]).abs().max().item():.2e}")
    print(f"stream2 diff: {(s2_orig - streams_li3[1]).abs().max().item():.2e}")
    print(f"integrated diff: {(int_orig - int_li3).abs().max().item():.2e}")

    # Create identical loss
    loss_orig = int_orig.sum() + s1_orig.sum() + s2_orig.sum()
    loss_li3 = int_li3.sum() + streams_li3[0].sum() + streams_li3[1].sum()

    print(f"\nloss_orig: {loss_orig.item():.6f}")
    print(f"loss_li3:  {loss_li3.item():.6f}")

    # Backward pass
    loss_orig.backward()
    loss_li3.backward()

    # Compare gradients
    print("\n--- Gradient Comparison ---")

    # Stream1 weight
    grad_diff_s1 = (conv_orig.stream1_weight.grad - conv_linet3.stream_weights[0].grad).abs()
    print(f"\nstream1_weight grad diff:")
    print(f"  Max:  {grad_diff_s1.max().item():.2e}")
    print(f"  Mean: {grad_diff_s1.mean().item():.2e}")

    # Stream2 weight
    grad_diff_s2 = (conv_orig.stream2_weight.grad - conv_linet3.stream_weights[1].grad).abs()
    print(f"\nstream2_weight grad diff:")
    print(f"  Max:  {grad_diff_s2.max().item():.2e}")
    print(f"  Mean: {grad_diff_s2.mean().item():.2e}")

    # Integration weights
    grad_diff_int1 = (conv_orig.integration_from_stream1.grad - conv_linet3.integration_from_streams[0].grad).abs()
    print(f"\nintegration_from_stream1 grad diff:")
    print(f"  Max:  {grad_diff_int1.max().item():.2e}")
    print(f"  Mean: {grad_diff_int1.mean().item():.2e}")

    grad_diff_int2 = (conv_orig.integration_from_stream2.grad - conv_linet3.integration_from_streams[1].grad).abs()
    print(f"\nintegration_from_stream2 grad diff:")
    print(f"  Max:  {grad_diff_int2.max().item():.2e}")
    print(f"  Mean: {grad_diff_int2.mean().item():.2e}")

    # Input gradients
    input_grad_diff = (rgb.grad - rgb_clone.grad).abs()
    print(f"\nInput (rgb) grad diff:")
    print(f"  Max:  {input_grad_diff.max().item():.2e}")
    print(f"  Mean: {input_grad_diff.mean().item():.2e}")


def compare_forward_implementations():
    """Compare the actual forward implementations line by line."""
    print("\n" + "=" * 80)
    print("Comparing Forward Implementations")
    print("=" * 80)

    torch.manual_seed(SEED)

    # Create simple test tensors
    rgb = torch.randn(2, 3, 16, 16)
    depth = torch.randn(2, 1, 16, 16)

    # Create weights manually
    stream1_weight = torch.randn(64, 3, 7, 7)
    stream2_weight = torch.randn(64, 1, 7, 7)
    int_from_s1 = torch.randn(64, 64, 1, 1)
    int_from_s2 = torch.randn(64, 64, 1, 1)
    integrated_weight = torch.randn(64, 0, 1, 1)  # Empty for first layer

    # ========== ORIGINAL LOGIC ==========
    print("\n--- Original Forward Logic (from conv.py) ---")

    # Stream convolutions
    stream1_out = F.conv2d(rgb, stream1_weight, None, stride=2, padding=3)
    stream2_out = F.conv2d(depth, stream2_weight, None, stride=2, padding=3)

    print(f"stream1_out: {stream1_out.shape}, mean={stream1_out.mean():.6f}")
    print(f"stream2_out: {stream2_out.shape}, mean={stream2_out.mean():.6f}")

    # Integration (Original uses stream_out which may include bias)
    integrated_from_s1 = F.conv2d(stream1_out, int_from_s1, None, stride=1, padding=0)
    integrated_from_s2 = F.conv2d(stream2_out, int_from_s2, None, stride=1, padding=0)

    # No previous integrated for first layer
    integrated_orig = integrated_from_s1 + integrated_from_s2

    print(f"integrated: {integrated_orig.shape}, mean={integrated_orig.mean():.6f}")

    # ========== LINET3 LOGIC ==========
    print("\n--- LINet3 Forward Logic (from li_net3/conv.py) ---")

    # Stream convolutions - SAME as original
    stream0_raw = F.conv2d(rgb, stream1_weight, None, stride=2, padding=3)
    stream1_raw = F.conv2d(depth, stream2_weight, None, stride=2, padding=3)

    print(f"stream0_raw: {stream0_raw.shape}, mean={stream0_raw.mean():.6f}")
    print(f"stream1_raw: {stream1_raw.shape}, mean={stream1_raw.mean():.6f}")

    # Integration (LINet3 uses stream_out_raw which NEVER has bias)
    integrated_from_s0 = F.conv2d(stream0_raw, int_from_s1, None, stride=1, padding=0)
    integrated_from_s1_li3 = F.conv2d(stream1_raw, int_from_s2, None, stride=1, padding=0)

    integrated_li3 = integrated_from_s0 + integrated_from_s1_li3

    print(f"integrated: {integrated_li3.shape}, mean={integrated_li3.mean():.6f}")

    # Compare
    print("\n--- Comparison ---")
    diff = (integrated_orig - integrated_li3).abs()
    print(f"Integrated diff: max={diff.max():.2e}, mean={diff.mean():.2e}")

    if diff.max() < 1e-6:
        print("✅ Forward logic is identical")
    else:
        print("⚠️  Forward logic differs!")


def test_backward_with_gradient_check():
    """Use torch.autograd.gradcheck to find gradient issues."""
    print("\n" + "=" * 80)
    print("Backward Pass Analysis")
    print("=" * 80)

    torch.manual_seed(SEED)

    # Create a minimal reproduction
    class OriginalStyleConv(nn.Module):
        def __init__(self):
            super().__init__()
            self.stream1_weight = nn.Parameter(torch.randn(8, 3, 3, 3) * 0.1)
            self.stream2_weight = nn.Parameter(torch.randn(8, 1, 3, 3) * 0.1)
            self.integration_from_stream1 = nn.Parameter(torch.randn(8, 8, 1, 1) * 0.1)
            self.integration_from_stream2 = nn.Parameter(torch.randn(8, 8, 1, 1) * 0.1)

        def forward(self, stream1_input, stream2_input):
            # Stream conv
            s1_out = F.conv2d(stream1_input, self.stream1_weight, None, padding=1)
            s2_out = F.conv2d(stream2_input, self.stream2_weight, None, padding=1)

            # Integration (uses s1_out, s2_out)
            int_from_s1 = F.conv2d(s1_out, self.integration_from_stream1, None)
            int_from_s2 = F.conv2d(s2_out, self.integration_from_stream2, None)

            integrated = int_from_s1 + int_from_s2

            return s1_out, s2_out, integrated

    class LINet3StyleConv(nn.Module):
        def __init__(self):
            super().__init__()
            self.stream_weights = nn.ParameterList([
                nn.Parameter(torch.randn(8, 3, 3, 3) * 0.1),
                nn.Parameter(torch.randn(8, 1, 3, 3) * 0.1),
            ])
            self.integration_from_streams = nn.ParameterList([
                nn.Parameter(torch.randn(8, 8, 1, 1) * 0.1),
                nn.Parameter(torch.randn(8, 8, 1, 1) * 0.1),
            ])

        def forward(self, stream_inputs):
            # Stream conv
            stream_outputs = []
            stream_outputs_raw = []

            for i, (stream_input, stream_weight) in enumerate(zip(stream_inputs, self.stream_weights)):
                s_out = F.conv2d(stream_input, stream_weight, None, padding=1)
                stream_outputs_raw.append(s_out)
                stream_outputs.append(s_out)  # No bias, so same as raw

            # Integration (uses raw outputs)
            integrated_parts = []
            for s_raw, int_weight in zip(stream_outputs_raw, self.integration_from_streams):
                int_part = F.conv2d(s_raw, int_weight, None)
                integrated_parts.append(int_part)

            integrated = sum(integrated_parts)

            return stream_outputs, integrated

    # Create both with identical weights
    torch.manual_seed(SEED)
    orig = OriginalStyleConv()

    torch.manual_seed(SEED)
    li3 = LINet3StyleConv()

    # Copy weights to ensure they're identical
    li3.stream_weights[0].data.copy_(orig.stream1_weight.data)
    li3.stream_weights[1].data.copy_(orig.stream2_weight.data)
    li3.integration_from_streams[0].data.copy_(orig.integration_from_stream1.data)
    li3.integration_from_streams[1].data.copy_(orig.integration_from_stream2.data)

    # Create inputs
    torch.manual_seed(SEED)
    rgb = torch.randn(2, 3, 8, 8, requires_grad=True)
    depth = torch.randn(2, 1, 8, 8, requires_grad=True)

    rgb_clone = rgb.detach().clone().requires_grad_(True)
    depth_clone = depth.detach().clone().requires_grad_(True)

    # Forward
    s1_o, s2_o, int_o = orig(rgb, depth)
    streams_l, int_l = li3([rgb_clone, depth_clone])

    print("\n--- Forward Comparison (Minimal Models) ---")
    print(f"stream1 diff: {(s1_o - streams_l[0]).abs().max():.2e}")
    print(f"stream2 diff: {(s2_o - streams_l[1]).abs().max():.2e}")
    print(f"integrated diff: {(int_o - int_l).abs().max():.2e}")

    # Backward with same gradient
    grad_integrated = torch.ones_like(int_o)

    int_o.backward(grad_integrated)
    int_l.backward(grad_integrated)

    print("\n--- Gradient Comparison (Minimal Models) ---")
    print(f"stream1_weight grad diff: {(orig.stream1_weight.grad - li3.stream_weights[0].grad).abs().max():.2e}")
    print(f"stream2_weight grad diff: {(orig.stream2_weight.grad - li3.stream_weights[1].grad).abs().max():.2e}")
    print(f"int_from_s1 grad diff: {(orig.integration_from_stream1.grad - li3.integration_from_streams[0].grad).abs().max():.2e}")
    print(f"int_from_s2 grad diff: {(orig.integration_from_stream2.grad - li3.integration_from_streams[1].grad).abs().max():.2e}")
    print(f"rgb grad diff: {(rgb.grad - rgb_clone.grad).abs().max():.2e}")
    print(f"depth grad diff: {(depth.grad - depth_clone.grad).abs().max():.2e}")

    # Check if minimal models have identical gradients
    if (orig.stream1_weight.grad - li3.stream_weights[0].grad).abs().max() < 1e-6:
        print("\n✅ Minimal models have identical gradients!")
        print("   The difference must be in the actual LIConv2d implementation.")
    else:
        print("\n⚠️  Even minimal models have gradient differences!")
        print("   The logic itself is different.")


def inspect_actual_conv_forward():
    """Inspect the actual LIConv2d._conv_forward implementations."""
    print("\n" + "=" * 80)
    print("Inspecting Actual LIConv2d Forward Implementations")
    print("=" * 80)

    # Read and compare the forward methods
    import inspect

    print("\n--- Original LIConv2d._conv_forward signature ---")
    print(inspect.signature(LIConv2dOriginal._conv_forward))

    print("\n--- LINet3 LIConv2d._conv_forward signature ---")
    print(inspect.signature(LIConv2d3._conv_forward))

    # Check if they have different argument handling
    orig_args = inspect.signature(LIConv2dOriginal._conv_forward).parameters
    li3_args = inspect.signature(LIConv2d3._conv_forward).parameters

    print(f"\nOriginal args: {list(orig_args.keys())}")
    print(f"LINet3 args:   {list(li3_args.keys())}")


def test_bn_gradient_flow():
    """Check if BatchNorm affects gradient flow differently."""
    print("\n" + "=" * 80)
    print("Testing BatchNorm Gradient Flow")
    print("=" * 80)

    # The gradient difference might come from BN layers, not conv layers
    # Let's test full model gradient flow

    torch.manual_seed(SEED)
    model_orig = LINetOriginal(
        block=LIBasicBlockOriginal,
        layers=[2, 2, 2, 2],
        num_classes=10,
        stream1_input_channels=3,
        stream2_input_channels=1,
        device='cpu'
    ).to('cpu')

    torch.manual_seed(SEED)
    model_linet3 = LINet3(
        block=LIBasicBlock3,
        layers=[2, 2, 2, 2],
        num_classes=10,
        stream_input_channels=[3, 1],
        device='cpu'
    ).to('cpu')

    # Set to eval mode to disable BN running stats updates
    model_orig.eval()
    model_linet3.eval()

    torch.manual_seed(SEED)
    rgb = torch.randn(4, 3, 32, 32, requires_grad=True)
    depth = torch.randn(4, 1, 32, 32, requires_grad=True)
    targets = torch.randint(0, 10, (4,))

    rgb_clone = rgb.detach().clone().requires_grad_(True)
    depth_clone = depth.detach().clone().requires_grad_(True)

    # Forward in eval mode
    out_orig = model_orig(rgb, depth)
    out_li3 = model_linet3([rgb_clone, depth_clone])

    print(f"\n--- Eval Mode Forward ---")
    print(f"Output diff: {(out_orig - out_li3).abs().max():.2e}")

    # Backward
    loss_orig = F.cross_entropy(out_orig, targets)
    loss_li3 = F.cross_entropy(out_li3, targets)

    loss_orig.backward()
    loss_li3.backward()

    # Compare conv1 gradients
    grad_diff = (model_orig.conv1.stream1_weight.grad - model_linet3.conv1.stream_weights[0].grad).abs()
    print(f"\n--- Eval Mode Gradients ---")
    print(f"conv1.stream1_weight grad diff: max={grad_diff.max():.2e}, mean={grad_diff.mean():.2e}")

    if grad_diff.max() < 1e-5:
        print("\n✅ In eval mode, gradients are identical!")
        print("   The difference in train mode comes from BatchNorm running stats.")
    else:
        print(f"\n⚠️  Gradients still differ in eval mode: {grad_diff.max():.2e}")


def trace_full_gradient_chain():
    """Trace gradients through the full network to find where they diverge."""
    print("\n" + "=" * 80)
    print("Tracing Full Gradient Chain")
    print("=" * 80)

    torch.manual_seed(SEED)
    model_orig = LINetOriginal(
        block=LIBasicBlockOriginal,
        layers=[2, 2, 2, 2],
        num_classes=10,
        stream1_input_channels=3,
        stream2_input_channels=1,
        device='cpu'
    ).to('cpu')

    torch.manual_seed(SEED)
    model_linet3 = LINet3(
        block=LIBasicBlock3,
        layers=[2, 2, 2, 2],
        num_classes=10,
        stream_input_channels=[3, 1],
        device='cpu'
    ).to('cpu')

    model_orig.train()
    model_linet3.train()

    torch.manual_seed(SEED)
    rgb = torch.randn(4, 3, 32, 32)
    depth = torch.randn(4, 1, 32, 32)
    targets = torch.randint(0, 10, (4,))

    # Forward
    out_orig = model_orig(rgb, depth)
    out_li3 = model_linet3([rgb, depth])

    # Backward
    F.cross_entropy(out_orig, targets).backward()
    F.cross_entropy(out_li3, targets).backward()

    # Compare gradients at each layer
    print("\n--- Gradient Comparison by Layer ---")

    layers = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']

    for layer_name in layers:
        layer_orig = getattr(model_orig, layer_name)
        layer_li3 = getattr(model_linet3, layer_name)

        orig_grad_norm = 0
        li3_grad_norm = 0
        max_diff = 0

        for (n1, p1), (n2, p2) in zip(layer_orig.named_parameters(), layer_li3.named_parameters()):
            if p1.grad is not None and p2.grad is not None:
                orig_grad_norm += p1.grad.norm().item()
                li3_grad_norm += p2.grad.norm().item()
                diff = (p1.grad - p2.grad).abs().max().item()
                if diff > max_diff:
                    max_diff = diff

        rel_diff = abs(orig_grad_norm - li3_grad_norm) / max(orig_grad_norm, 1e-8)
        flag = " ⚠️" if max_diff > 1e-4 else ""
        print(f"{layer_name}: orig_norm={orig_grad_norm:.4f}, li3_norm={li3_grad_norm:.4f}, max_diff={max_diff:.2e}{flag}")


if __name__ == "__main__":
    print("\n🔍 Deep Dive: Why Do Gradients Differ?\n")

    test_conv2d_gradient_isolated()
    compare_forward_implementations()
    test_backward_with_gradient_check()
    inspect_actual_conv_forward()
    test_bn_gradient_flow()
    trace_full_gradient_chain()
