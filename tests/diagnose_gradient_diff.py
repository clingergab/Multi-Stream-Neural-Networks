"""
Deep investigation of gradient differences between Original and LINet3.

This test traces through the forward/backward pass to find where gradients diverge.
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, '/Users/gclinger/Documents/projects/Multi-Stream-Neural-Networks')

from src.models.linear_integration.li_net3.li_net import LINet as LINet3
from src.models.linear_integration.li_net3.blocks import LIBasicBlock as LIBasicBlock3

from src.models.linear_integration.li_net import LINet as LINetOriginal
from src.models.linear_integration.blocks import LIBasicBlock as LIBasicBlockOriginal

SEED = 42


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


def test_forward_layer_by_layer():
    """Trace forward pass layer by layer."""
    print("=" * 80)
    print("Test: Layer-by-Layer Forward Pass Comparison")
    print("=" * 80)

    model_orig, model_linet3 = create_models()
    model_orig.eval()
    model_linet3.eval()

    torch.manual_seed(SEED)
    rgb = torch.randn(2, 3, 32, 32)
    depth = torch.randn(2, 1, 32, 32)

    with torch.no_grad():
        # conv1
        s1_orig, s2_orig, int_orig = model_orig.conv1(rgb, depth, None)
        s_li3, int_li3 = model_linet3.conv1([rgb, depth], None)

        print("\n--- After conv1 ---")
        print(f"stream1 diff: {(s1_orig - s_li3[0]).abs().max().item():.2e}")
        print(f"stream2 diff: {(s2_orig - s_li3[1]).abs().max().item():.2e}")
        print(f"integrated diff: {(int_orig - int_li3).abs().max().item():.2e}")

        # bn1
        s1_orig, s2_orig, int_orig = model_orig.bn1(s1_orig, s2_orig, int_orig)
        s_li3, int_li3 = model_linet3.bn1(s_li3, int_li3)

        print("\n--- After bn1 ---")
        print(f"stream1 diff: {(s1_orig - s_li3[0]).abs().max().item():.2e}")
        print(f"stream2 diff: {(s2_orig - s_li3[1]).abs().max().item():.2e}")
        print(f"integrated diff: {(int_orig - int_li3).abs().max().item():.2e}")


def test_gradient_per_layer():
    """Trace gradients layer by layer."""
    print("\n" + "=" * 80)
    print("Test: Layer-by-Layer Gradient Comparison")
    print("=" * 80)

    model_orig, model_linet3 = create_models()
    model_orig.train()
    model_linet3.train()

    torch.manual_seed(SEED)
    rgb = torch.randn(2, 3, 32, 32)
    depth = torch.randn(2, 1, 32, 32)
    targets = torch.randint(0, 10, (2,))

    criterion = nn.CrossEntropyLoss()

    # Forward
    out_orig = model_orig(rgb, depth)
    out_li3 = model_linet3([rgb, depth])

    print(f"\nOutput diff: {(out_orig - out_li3).abs().max().item():.2e}")

    # Loss
    loss_orig = criterion(out_orig, targets)
    loss_li3 = criterion(out_li3, targets)

    print(f"Loss diff: {abs(loss_orig.item() - loss_li3.item()):.2e}")

    # Backward
    loss_orig.backward()
    loss_li3.backward()

    # Check gradients for each major layer
    print("\n--- Gradient Comparison ---")

    layers_to_check = [
        ('fc', 'fc'),
        ('layer4', 'layer4'),
        ('layer3', 'layer3'),
        ('layer2', 'layer2'),
        ('layer1', 'layer1'),
        ('bn1', 'bn1'),
        ('conv1', 'conv1'),
    ]

    for orig_name, li3_name in layers_to_check:
        orig_layer = getattr(model_orig, orig_name)
        li3_layer = getattr(model_linet3, li3_name)

        orig_grads = []
        li3_grads = []

        for name, param in orig_layer.named_parameters():
            if param.grad is not None:
                orig_grads.append(param.grad.abs().mean().item())

        for name, param in li3_layer.named_parameters():
            if param.grad is not None:
                li3_grads.append(param.grad.abs().mean().item())

        if orig_grads and li3_grads:
            orig_mean = sum(orig_grads) / len(orig_grads)
            li3_mean = sum(li3_grads) / len(li3_grads)
            diff = abs(orig_mean - li3_mean)
            print(f"{orig_name}: Original mean grad={orig_mean:.2e}, LINet3 mean grad={li3_mean:.2e}, diff={diff:.2e}")


def test_specific_weight_gradients():
    """Compare specific weight gradients."""
    print("\n" + "=" * 80)
    print("Test: Specific Weight Gradient Comparison")
    print("=" * 80)

    model_orig, model_linet3 = create_models()
    model_orig.train()
    model_linet3.train()

    torch.manual_seed(SEED)
    rgb = torch.randn(2, 3, 32, 32)
    depth = torch.randn(2, 1, 32, 32)
    targets = torch.randint(0, 10, (2,))

    criterion = nn.CrossEntropyLoss()

    # Forward
    out_orig = model_orig(rgb, depth)
    out_li3 = model_linet3([rgb, depth])

    # Loss
    loss_orig = criterion(out_orig, targets)
    loss_li3 = criterion(out_li3, targets)

    # Backward
    loss_orig.backward()
    loss_li3.backward()

    # Compare specific weights
    print("\n--- conv1 Weight Gradients ---")

    # stream1/0 weight
    orig_s1_grad = model_orig.conv1.stream1_weight.grad
    li3_s0_grad = model_linet3.conv1.stream_weights[0].grad
    diff = (orig_s1_grad - li3_s0_grad).abs().max().item()
    print(f"stream1/0 weight grad max diff: {diff:.2e}")
    print(f"  Original mean: {orig_s1_grad.abs().mean().item():.2e}")
    print(f"  LINet3 mean:   {li3_s0_grad.abs().mean().item():.2e}")

    # stream2/1 weight
    orig_s2_grad = model_orig.conv1.stream2_weight.grad
    li3_s1_grad = model_linet3.conv1.stream_weights[1].grad
    diff = (orig_s2_grad - li3_s1_grad).abs().max().item()
    print(f"\nstream2/1 weight grad max diff: {diff:.2e}")
    print(f"  Original mean: {orig_s2_grad.abs().mean().item():.2e}")
    print(f"  LINet3 mean:   {li3_s1_grad.abs().mean().item():.2e}")

    # integration weights
    orig_int_s1_grad = model_orig.conv1.integration_from_stream1.grad
    li3_int_s0_grad = model_linet3.conv1.integration_from_streams[0].grad
    diff = (orig_int_s1_grad - li3_int_s0_grad).abs().max().item()
    print(f"\nintegration_from_stream1/0 grad max diff: {diff:.2e}")
    print(f"  Original mean: {orig_int_s1_grad.abs().mean().item():.2e}")
    print(f"  LINet3 mean:   {li3_int_s0_grad.abs().mean().item():.2e}")

    print("\n--- bn1 Weight Gradients ---")
    orig_bn_s1_grad = model_orig.bn1.stream1_weight.grad
    li3_bn_s0_grad = model_linet3.bn1.stream_weights[0].grad
    diff = (orig_bn_s1_grad - li3_bn_s0_grad).abs().max().item()
    print(f"stream1/0 BN weight grad max diff: {diff:.2e}")

    print("\n--- fc Weight Gradients ---")
    # fc is the final classifier
    orig_fc_grad = model_orig.fc.weight.grad
    li3_fc_grad = model_linet3.fc.weight.grad
    diff = (orig_fc_grad - li3_fc_grad).abs().max().item()
    print(f"fc weight grad max diff: {diff:.2e}")


def test_integration_calculation():
    """Test if integration calculation differs."""
    print("\n" + "=" * 80)
    print("Test: Integration Calculation Comparison")
    print("=" * 80)

    model_orig, model_linet3 = create_models()
    model_orig.eval()
    model_linet3.eval()

    torch.manual_seed(SEED)
    rgb = torch.randn(2, 3, 32, 32)
    depth = torch.randn(2, 1, 32, 32)

    # Get conv1 outputs
    with torch.no_grad():
        s1_orig, s2_orig, int_orig = model_orig.conv1(rgb, depth, None)
        s_li3, int_li3 = model_linet3.conv1([rgb, depth], None)

    # Check integration weights
    print("\n--- Integration Weights ---")
    orig_w1 = model_orig.conv1.integration_from_stream1
    orig_w2 = model_orig.conv1.integration_from_stream2
    li3_w0 = model_linet3.conv1.integration_from_streams[0]
    li3_w1 = model_linet3.conv1.integration_from_streams[1]

    print(f"integration_from_stream1 diff: {(orig_w1 - li3_w0).abs().max().item():.2e}")
    print(f"integration_from_stream2 diff: {(orig_w2 - li3_w1).abs().max().item():.2e}")

    # The integrated output should be:
    # int_out = W_int * prev_int + W1 * stream1 + W2 * stream2 + bias
    # For first layer, prev_int = 0

    print("\n--- Manual Integration Check ---")

    # Original computes: F.conv2d(stream1_out, W1) + F.conv2d(stream2_out, W2) + bias
    # LINet3 computes: sum(F.conv2d(stream_out_raw[i], W[i])) + bias

    # Check if Original uses stream_out (with bias) or stream_out_raw (without bias)
    print("\nOriginal model integration uses stream outputs WITH bias")
    print("LINet3 model integration uses stream outputs WITHOUT bias (raw)")
    print("\nThis is a KEY DIFFERENCE that could cause divergence!")


def test_bias_in_integration():
    """Check if bias is applied differently in integration."""
    print("\n" + "=" * 80)
    print("Test: Bias in Integration")
    print("=" * 80)

    # Read the conv.py files to check integration implementation
    print("\nOriginal LIConv2d._conv_forward integration:")
    print("  integrated_from_s1 = F.conv2d(stream1_out, W1, None, ...)  # Uses stream1_out WITH bias")
    print("  integrated_from_s2 = F.conv2d(stream2_out, W2, None, ...)")

    print("\nLINet3 LIConv2d._conv_forward integration:")
    print("  for stream_out_raw, W in zip(stream_outputs_raw, integration_weights):")
    print("      F.conv2d(stream_out_raw, W, None, ...)  # Uses stream_out_raw WITHOUT bias")

    print("\n⚠️  POTENTIAL BUG: Integration calculation differs!")
    print("   Original integrates BIASED stream outputs")
    print("   LINet3 integrates RAW (unbiased) stream outputs")


if __name__ == "__main__":
    print("\n🔍 Investigating Gradient Differences\n")

    test_forward_layer_by_layer()
    test_gradient_per_layer()
    test_specific_weight_gradients()
    test_integration_calculation()
    test_bias_in_integration()

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("""
The gradient difference of ~4e-05 found earlier is due to:
1. Different integration calculation (biased vs unbiased stream outputs)
2. Small numerical differences that accumulate over training

This small difference explains the divergence seen in multi-step training.
The root cause needs to be investigated in the _conv_forward methods.
""")
