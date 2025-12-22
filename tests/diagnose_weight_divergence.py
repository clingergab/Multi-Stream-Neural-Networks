"""
Diagnose weight divergence between LINet3 and Original after training steps.

This investigates why weights diverge starting from step 0, which suggests
a real bug in the forward/backward pass.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

import sys
sys.path.insert(0, '/Users/gclinger/Documents/projects/Multi-Stream-Neural-Networks')

# Import both models
from src.models.linear_integration.li_net3.li_net import LINet as LINet3
from src.models.linear_integration.li_net3.blocks import LIBasicBlock as LIBasicBlock3

from src.models.linear_integration.li_net import LINet as LINetOriginal
from src.models.linear_integration.blocks import LIBasicBlock as LIBasicBlockOriginal

SEED = 42
torch.manual_seed(SEED)


def create_models():
    """Create both models with identical seeds."""
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

    return model_orig, model_linet3


def create_inputs(seed=SEED):
    """Create identical inputs for both models."""
    torch.manual_seed(seed)
    rgb = torch.randn(4, 3, 32, 32, device='cpu')
    depth = torch.randn(4, 1, 32, 32, device='cpu')
    targets = torch.randint(0, 10, (4,), device='cpu')
    return rgb, depth, targets


def map_linet3_to_orig_name(name):
    """Map LINet3 parameter name to Original parameter name."""
    # stream_weights.0 -> stream1_weight
    # stream_weights.1 -> stream2_weight
    # stream_biases.0 -> stream1_bias
    # stream_biases.1 -> stream2_bias
    # integration_from_streams.0 -> integration_from_stream1
    # integration_from_streams.1 -> integration_from_stream2

    name = name.replace('stream_weights.0', 'stream1_weight')
    name = name.replace('stream_weights.1', 'stream2_weight')
    name = name.replace('stream_biases.0', 'stream1_bias')
    name = name.replace('stream_biases.1', 'stream2_bias')
    name = name.replace('integration_from_streams.0', 'integration_from_stream1')
    name = name.replace('integration_from_streams.1', 'integration_from_stream2')

    # BatchNorm: stream0_running_mean -> stream1_running_mean
    name = name.replace('stream0_running_mean', 'stream1_running_mean')
    name = name.replace('stream1_running_mean', 'stream2_running_mean') if 'stream1_running' in name and 'stream0' not in name else name
    name = name.replace('stream0_running_var', 'stream1_running_var')
    name = name.replace('stream1_running_var', 'stream2_running_var') if 'stream1_running' in name and 'stream0' not in name else name

    # Fix the above - be more careful
    if 'stream_weights.0' in name or 'stream0' in name:
        pass  # Already handled

    return name


def compare_initial_weights():
    """Compare initial weights between models."""
    print("=" * 80)
    print("Step 0: Comparing Initial Weights")
    print("=" * 80)

    model_orig, model_linet3 = create_models()

    # Build name mapping
    orig_params = dict(model_orig.named_parameters())
    linet3_params = dict(model_linet3.named_parameters())

    print(f"\nOriginal params: {len(orig_params)}")
    print(f"LINet3 params: {len(linet3_params)}")

    # Check conv1 specifically
    print("\n--- conv1 Weight Comparison ---")

    # Original
    orig_s1 = model_orig.conv1.stream1_weight
    orig_s2 = model_orig.conv1.stream2_weight
    orig_int_s1 = model_orig.conv1.integration_from_stream1
    orig_int_s2 = model_orig.conv1.integration_from_stream2

    # LINet3
    li3_s0 = model_linet3.conv1.stream_weights[0]
    li3_s1 = model_linet3.conv1.stream_weights[1]
    li3_int_s0 = model_linet3.conv1.integration_from_streams[0]
    li3_int_s1 = model_linet3.conv1.integration_from_streams[1]

    print(f"\nstream1_weight vs stream_weights[0]:")
    print(f"  Shapes: {orig_s1.shape} vs {li3_s0.shape}")
    if orig_s1.shape == li3_s0.shape:
        diff = (orig_s1 - li3_s0).abs().max().item()
        print(f"  Max diff: {diff:.2e}")
    else:
        print(f"  Shape mismatch!")

    print(f"\nstream2_weight vs stream_weights[1]:")
    print(f"  Shapes: {orig_s2.shape} vs {li3_s1.shape}")
    if orig_s2.shape == li3_s1.shape:
        diff = (orig_s2 - li3_s1).abs().max().item()
        print(f"  Max diff: {diff:.2e}")
    else:
        print(f"  Shape mismatch!")

    print(f"\nintegration_from_stream1 vs integration_from_streams[0]:")
    print(f"  Shapes: {orig_int_s1.shape} vs {li3_int_s0.shape}")
    if orig_int_s1.shape == li3_int_s0.shape:
        diff = (orig_int_s1 - li3_int_s0).abs().max().item()
        print(f"  Max diff: {diff:.2e}")
    else:
        print(f"  Shape mismatch!")

    print(f"\nintegration_from_stream2 vs integration_from_streams[1]:")
    print(f"  Shapes: {orig_int_s2.shape} vs {li3_int_s1.shape}")
    if orig_int_s2.shape == li3_int_s1.shape:
        diff = (orig_int_s2 - li3_int_s1).abs().max().item()
        print(f"  Max diff: {diff:.2e}")
    else:
        print(f"  Shape mismatch!")

    return model_orig, model_linet3


def compare_forward_pass():
    """Compare forward pass outputs."""
    print("\n" + "=" * 80)
    print("Step 1: Comparing Forward Pass Outputs")
    print("=" * 80)

    model_orig, model_linet3 = create_models()
    model_orig.eval()
    model_linet3.eval()

    rgb, depth, targets = create_inputs()

    with torch.no_grad():
        # Original expects (stream1, stream2)
        out_orig = model_orig(rgb, depth)

        # LINet3 expects [stream0, stream1]
        out_linet3 = model_linet3([rgb, depth])

    print(f"\nOriginal output: mean={out_orig.mean():.6f}, std={out_orig.std():.6f}")
    print(f"LINet3 output:   mean={out_linet3.mean():.6f}, std={out_linet3.std():.6f}")

    diff = (out_orig - out_linet3).abs()
    print(f"\nOutput difference:")
    print(f"  Max:  {diff.max().item():.2e}")
    print(f"  Mean: {diff.mean().item():.2e}")

    if diff.max().item() < 1e-5:
        print("\n✅ Forward pass outputs are nearly identical")
    else:
        print("\n⚠️  Forward pass outputs differ significantly!")

    return out_orig, out_linet3


def compare_conv1_outputs():
    """Compare conv1 outputs in detail."""
    print("\n" + "=" * 80)
    print("Step 2: Comparing conv1 Outputs in Detail")
    print("=" * 80)

    model_orig, model_linet3 = create_models()
    model_orig.eval()
    model_linet3.eval()

    rgb, depth, targets = create_inputs()

    # Capture conv1 outputs
    orig_outputs = {}
    linet3_outputs = {}

    def orig_hook(module, input, output):
        s1, s2, integrated = output
        orig_outputs['stream1'] = s1.detach()
        orig_outputs['stream2'] = s2.detach()
        orig_outputs['integrated'] = integrated.detach() if integrated is not None else None

    def linet3_hook(module, input, output):
        streams, integrated = output
        linet3_outputs['stream0'] = streams[0].detach()
        linet3_outputs['stream1'] = streams[1].detach()
        linet3_outputs['integrated'] = integrated.detach() if integrated is not None else None

    model_orig.conv1.register_forward_hook(orig_hook)
    model_linet3.conv1.register_forward_hook(linet3_hook)

    with torch.no_grad():
        model_orig(rgb, depth)
        model_linet3([rgb, depth])

    print("\n--- Stream Outputs ---")
    print(f"Original stream1 vs LINet3 stream0:")
    diff_s1 = (orig_outputs['stream1'] - linet3_outputs['stream0']).abs()
    print(f"  Max diff:  {diff_s1.max().item():.2e}")
    print(f"  Mean diff: {diff_s1.mean().item():.2e}")

    print(f"\nOriginal stream2 vs LINet3 stream1:")
    diff_s2 = (orig_outputs['stream2'] - linet3_outputs['stream1']).abs()
    print(f"  Max diff:  {diff_s2.max().item():.2e}")
    print(f"  Mean diff: {diff_s2.mean().item():.2e}")

    print(f"\n--- Integrated Outputs ---")
    if orig_outputs['integrated'] is not None and linet3_outputs['integrated'] is not None:
        diff_int = (orig_outputs['integrated'] - linet3_outputs['integrated']).abs()
        print(f"Max diff:  {diff_int.max().item():.2e}")
        print(f"Mean diff: {diff_int.mean().item():.2e}")

        if diff_int.max().item() > 1e-5:
            print("\n⚠️  INTEGRATED OUTPUTS DIFFER!")
            print("This is the key finding - integration produces different results.")

            # Check what's being integrated
            print("\n--- Debugging Integration ---")
            print("Original integrates: stream1_out (with bias), stream2_out (with bias)")
            print("LINet3 integrates:   stream_out_raw (NO bias)")

            # With bias=False, these should be identical
            # Let's check if bias is actually False
            print(f"\nOriginal conv1 stream1_bias: {model_orig.conv1.stream1_bias}")
            print(f"LINet3 conv1 stream_biases: {model_linet3.conv1.stream_biases}")
    else:
        print("One or both integrated outputs are None")


def compare_gradients():
    """Compare gradients after one backward pass."""
    print("\n" + "=" * 80)
    print("Step 3: Comparing Gradients After Backward")
    print("=" * 80)

    model_orig, model_linet3 = create_models()
    model_orig.train()
    model_linet3.train()

    rgb, depth, targets = create_inputs()

    # Forward
    out_orig = model_orig(rgb, depth)
    out_linet3 = model_linet3([rgb, depth])

    # Loss
    loss_orig = F.cross_entropy(out_orig, targets)
    loss_linet3 = F.cross_entropy(out_linet3, targets)

    print(f"\nLoss Original: {loss_orig.item():.6f}")
    print(f"Loss LINet3:   {loss_linet3.item():.6f}")

    # Backward
    loss_orig.backward()
    loss_linet3.backward()

    # Compare gradients for conv1
    print("\n--- conv1 Gradient Comparison ---")

    # Stream weights
    orig_grad_s1 = model_orig.conv1.stream1_weight.grad
    li3_grad_s0 = model_linet3.conv1.stream_weights[0].grad

    if orig_grad_s1 is not None and li3_grad_s0 is not None:
        diff = (orig_grad_s1 - li3_grad_s0).abs()
        print(f"\nstream1_weight grad vs stream_weights[0] grad:")
        print(f"  Max diff:  {diff.max().item():.2e}")
        print(f"  Mean diff: {diff.mean().item():.2e}")

        if diff.max().item() > 1e-5:
            print("  ⚠️  Gradients differ significantly!")

    orig_grad_s2 = model_orig.conv1.stream2_weight.grad
    li3_grad_s1 = model_linet3.conv1.stream_weights[1].grad

    if orig_grad_s2 is not None and li3_grad_s1 is not None:
        diff = (orig_grad_s2 - li3_grad_s1).abs()
        print(f"\nstream2_weight grad vs stream_weights[1] grad:")
        print(f"  Max diff:  {diff.max().item():.2e}")
        print(f"  Mean diff: {diff.mean().item():.2e}")

        if diff.max().item() > 1e-5:
            print("  ⚠️  Gradients differ significantly!")

    # Integration weights
    orig_grad_int1 = model_orig.conv1.integration_from_stream1.grad
    li3_grad_int0 = model_linet3.conv1.integration_from_streams[0].grad

    if orig_grad_int1 is not None and li3_grad_int0 is not None:
        diff = (orig_grad_int1 - li3_grad_int0).abs()
        print(f"\nintegration_from_stream1 grad vs integration_from_streams[0] grad:")
        print(f"  Max diff:  {diff.max().item():.2e}")
        print(f"  Mean diff: {diff.mean().item():.2e}")

        if diff.max().item() > 1e-5:
            print("  ⚠️  Integration gradients differ!")


def trace_integration_step():
    """Trace exactly what happens in the integration step."""
    print("\n" + "=" * 80)
    print("Step 4: Tracing Integration Step in Detail")
    print("=" * 80)

    model_orig, model_linet3 = create_models()
    model_orig.eval()
    model_linet3.eval()

    rgb, depth, targets = create_inputs()

    # Get conv1 layers
    conv1_orig = model_orig.conv1
    conv1_linet3 = model_linet3.conv1

    with torch.no_grad():
        # ========== ORIGINAL ==========
        print("\n--- Original Integration Logic ---")

        # Stream1 conv
        stream1_out = F.conv2d(
            rgb, conv1_orig.stream1_weight, conv1_orig.stream1_bias,
            conv1_orig.stride, conv1_orig.padding, conv1_orig.dilation, conv1_orig.groups
        )
        print(f"stream1_out (with bias): mean={stream1_out.mean():.6f}, std={stream1_out.std():.6f}")

        # Stream2 conv
        stream2_out = F.conv2d(
            depth, conv1_orig.stream2_weight, conv1_orig.stream2_bias,
            conv1_orig.stride, conv1_orig.padding, conv1_orig.dilation, conv1_orig.groups
        )
        print(f"stream2_out (with bias): mean={stream2_out.mean():.6f}, std={stream2_out.std():.6f}")

        # Integration (uses stream_out which includes bias)
        integrated_from_s1_orig = F.conv2d(
            stream1_out, conv1_orig.integration_from_stream1, None,
            stride=1, padding=0
        )
        integrated_from_s2_orig = F.conv2d(
            stream2_out, conv1_orig.integration_from_stream2, None,
            stride=1, padding=0
        )
        integrated_orig = integrated_from_s1_orig + integrated_from_s2_orig
        print(f"integrated (from biased): mean={integrated_orig.mean():.6f}, std={integrated_orig.std():.6f}")

        # ========== LINET3 ==========
        print("\n--- LINet3 Integration Logic ---")

        # Stream0 conv (raw, no bias)
        stream0_raw = F.conv2d(
            rgb, conv1_linet3.stream_weights[0], None,  # No bias!
            conv1_linet3.stride, conv1_linet3.padding, conv1_linet3.dilation, conv1_linet3.groups
        )
        print(f"stream0_raw (no bias):   mean={stream0_raw.mean():.6f}, std={stream0_raw.std():.6f}")

        # Stream1 conv (raw, no bias)
        stream1_raw = F.conv2d(
            depth, conv1_linet3.stream_weights[1], None,  # No bias!
            conv1_linet3.stride, conv1_linet3.padding, conv1_linet3.dilation, conv1_linet3.groups
        )
        print(f"stream1_raw (no bias):   mean={stream1_raw.mean():.6f}, std={stream1_raw.std():.6f}")

        # Integration (uses raw outputs)
        integrated_from_s0_li3 = F.conv2d(
            stream0_raw, conv1_linet3.integration_from_streams[0], None,
            stride=1, padding=0
        )
        integrated_from_s1_li3 = F.conv2d(
            stream1_raw, conv1_linet3.integration_from_streams[1], None,
            stride=1, padding=0
        )
        integrated_linet3 = integrated_from_s0_li3 + integrated_from_s1_li3
        print(f"integrated (from raw):   mean={integrated_linet3.mean():.6f}, std={integrated_linet3.std():.6f}")

        # ========== COMPARISON ==========
        print("\n--- Comparison ---")

        # Check if biases are actually None
        print(f"\nOriginal stream1_bias: {conv1_orig.stream1_bias}")
        print(f"LINet3 stream_biases: {conv1_linet3.stream_biases}")

        # If biases are None, the difference should be 0
        if conv1_orig.stream1_bias is None and conv1_linet3.stream_biases is None:
            print("\n✓ Both have bias=None, so stream outputs should be identical")
            diff_s1 = (stream1_out - stream0_raw).abs()
            print(f"stream1_out vs stream0_raw diff: max={diff_s1.max():.2e}, mean={diff_s1.mean():.2e}")

            diff_int = (integrated_orig - integrated_linet3).abs()
            print(f"integrated diff: max={diff_int.max():.2e}, mean={diff_int.mean():.2e}")

            if diff_int.max().item() > 1e-5:
                print("\n🚨 BUG FOUND: Integration outputs differ even though bias=None!")
                print("   The difference must be in the integration weights themselves.")

                # Check integration weight differences
                diff_w1 = (conv1_orig.integration_from_stream1 - conv1_linet3.integration_from_streams[0]).abs()
                diff_w2 = (conv1_orig.integration_from_stream2 - conv1_linet3.integration_from_streams[1]).abs()
                print(f"\n   integration_from_stream1 vs integration_from_streams[0]:")
                print(f"     Max diff: {diff_w1.max():.2e}")
                print(f"   integration_from_stream2 vs integration_from_streams[1]:")
                print(f"     Max diff: {diff_w2.max():.2e}")


def check_weight_initialization_order():
    """Check if weights are initialized in the same order."""
    print("\n" + "=" * 80)
    print("Step 5: Checking Weight Initialization Order")
    print("=" * 80)

    # Create with same seed and check first few random values
    torch.manual_seed(SEED)

    print("\nCreating Original model...")
    model_orig = LINetOriginal(
        block=LIBasicBlockOriginal,
        layers=[2, 2, 2, 2],
        num_classes=10,
        stream1_input_channels=3,
        stream2_input_channels=1,
        device='cpu'
    ).to('cpu')

    # Get first parameter values
    orig_first_vals = []
    for name, param in list(model_orig.named_parameters())[:5]:
        orig_first_vals.append((name, param.data.flatten()[:5].tolist()))

    # Reset seed and create LINet3
    torch.manual_seed(SEED)

    print("Creating LINet3 model...")
    model_linet3 = LINet3(
        block=LIBasicBlock3,
        layers=[2, 2, 2, 2],
        num_classes=10,
        stream_input_channels=[3, 1],
        device='cpu'
    ).to('cpu')

    li3_first_vals = []
    for name, param in list(model_linet3.named_parameters())[:5]:
        li3_first_vals.append((name, param.data.flatten()[:5].tolist()))

    print("\n--- First 5 Parameters (first 5 values each) ---")
    print("\nOriginal:")
    for name, vals in orig_first_vals:
        print(f"  {name}: {vals}")

    print("\nLINet3:")
    for name, vals in li3_first_vals:
        print(f"  {name}: {vals}")

    # Check if same parameter names appear in same order
    print("\n--- Parameter Name Order ---")
    orig_names = [n for n, _ in model_orig.named_parameters()]
    li3_names = [n for n, _ in model_linet3.named_parameters()]

    print(f"\nOriginal first 10 param names:")
    for n in orig_names[:10]:
        print(f"  {n}")

    print(f"\nLINet3 first 10 param names:")
    for n in li3_names[:10]:
        print(f"  {n}")


def test_identical_weights_init():
    """Force identical weights and test forward pass."""
    print("\n" + "=" * 80)
    print("Step 6: Force Identical Weights and Test")
    print("=" * 80)

    model_orig, model_linet3 = create_models()

    # Copy weights from Original to LINet3
    print("\nCopying weights from Original to LINet3...")

    # conv1
    model_linet3.conv1.stream_weights[0].data.copy_(model_orig.conv1.stream1_weight.data)
    model_linet3.conv1.stream_weights[1].data.copy_(model_orig.conv1.stream2_weight.data)
    model_linet3.conv1.integration_from_streams[0].data.copy_(model_orig.conv1.integration_from_stream1.data)
    model_linet3.conv1.integration_from_streams[1].data.copy_(model_orig.conv1.integration_from_stream2.data)
    model_linet3.conv1.integrated_weight.data.copy_(model_orig.conv1.integrated_weight.data)

    # Verify copy
    diff = (model_linet3.conv1.stream_weights[0] - model_orig.conv1.stream1_weight).abs().max()
    print(f"conv1.stream_weights[0] copy verified: diff={diff:.2e}")

    diff = (model_linet3.conv1.integration_from_streams[0] - model_orig.conv1.integration_from_stream1).abs().max()
    print(f"conv1.integration_from_streams[0] copy verified: diff={diff:.2e}")

    # Now test forward pass with identical weights
    model_orig.eval()
    model_linet3.eval()

    rgb, depth, targets = create_inputs()

    with torch.no_grad():
        out_orig = model_orig(rgb, depth)
        out_linet3 = model_linet3([rgb, depth])

    diff = (out_orig - out_linet3).abs()
    print(f"\nWith COPIED weights, output diff:")
    print(f"  Max:  {diff.max().item():.2e}")
    print(f"  Mean: {diff.mean().item():.2e}")

    if diff.max().item() > 1e-5:
        print("\n🚨 Outputs STILL differ even with identical weights!")
        print("   This means the forward pass logic itself is different.")
    else:
        print("\n✅ Outputs are identical when weights are copied.")
        print("   The difference was in weight initialization order/values.")


if __name__ == "__main__":
    print("\n🔍 Diagnosing Weight Divergence Between LINet3 and Original...\n")

    compare_initial_weights()
    compare_forward_pass()
    compare_conv1_outputs()
    compare_gradients()
    trace_integration_step()
    check_weight_initialization_order()
    test_identical_weights_init()

    print("\n" + "=" * 80)
    print("DIAGNOSIS COMPLETE")
    print("=" * 80)
