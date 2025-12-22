"""
Diagnose why Stream0 gets zero gradients in early layers while Stream1 gets gradients.

This is a critical bug - if Stream0 (RGB) isn't getting gradients, it won't learn!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

import sys
sys.path.insert(0, '/Users/gclinger/Documents/projects/Multi-Stream-Neural-Networks')

from src.models.linear_integration.li_net3.li_net import LINet, li_resnet18
from src.models.linear_integration.li_net3.conv import LIConv2d


def debug_stream_gradient_flow():
    """Debug why Stream0 gets zero gradients."""
    print("=" * 80)
    print("Debugging Stream0 Zero Gradient Issue")
    print("=" * 80)

    model = li_resnet18(num_classes=10, stream_input_channels=[3, 1], device='cpu')
    model = model.to('cpu')
    model.train()

    batch_size = 4
    rgb = torch.randn(batch_size, 3, 224, 224, device='cpu')
    depth = torch.randn(batch_size, 1, 224, 224, device='cpu')
    targets = torch.randint(0, 10, (batch_size,), device='cpu')

    # Forward pass
    logits = model([rgb, depth])
    loss = F.cross_entropy(logits, targets)
    loss.backward()

    # Check all parameters in conv1
    print("\n📋 conv1 Parameter Gradients:")
    print("-" * 60)

    for name, param in model.conv1.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_mean = param.grad.abs().mean().item()
            has_nonzero = (param.grad.abs() > 1e-10).any().item()
            print(f"  {name}: norm={grad_norm:.6f}, mean={grad_mean:.8f}, has_nonzero={has_nonzero}, shape={tuple(param.shape)}")
        else:
            print(f"  {name}: grad=None, shape={tuple(param.shape)}")

    # Check specifically stream_weights
    print("\n📋 Detailed stream_weights Check:")
    print("-" * 60)

    for i, stream_weight in enumerate(model.conv1.stream_weights):
        if stream_weight.grad is not None:
            grad = stream_weight.grad
            print(f"\n  stream_weights[{i}] (shape {tuple(stream_weight.shape)}):")
            print(f"    grad norm: {grad.norm().item():.6f}")
            print(f"    grad mean: {grad.abs().mean().item():.8f}")
            print(f"    grad min:  {grad.min().item():.8f}")
            print(f"    grad max:  {grad.max().item():.8f}")
            print(f"    grad all zeros: {(grad.abs() < 1e-10).all().item()}")

            # Check if weight has non-zero values
            print(f"    weight norm: {stream_weight.norm().item():.6f}")
        else:
            print(f"\n  stream_weights[{i}]: grad=None ⚠️")


def debug_integration_weights():
    """Check integration weights in detail."""
    print("\n" + "=" * 80)
    print("Integration Weights Analysis")
    print("=" * 80)

    model = li_resnet18(num_classes=10, stream_input_channels=[3, 1], device='cpu')
    model = model.to('cpu')
    model.train()

    batch_size = 4
    rgb = torch.randn(batch_size, 3, 224, 224, device='cpu')
    depth = torch.randn(batch_size, 1, 224, 224, device='cpu')
    targets = torch.randint(0, 10, (batch_size,), device='cpu')

    logits = model([rgb, depth])
    loss = F.cross_entropy(logits, targets)
    loss.backward()

    print("\n📋 conv1 Integration Weights:")
    print("-" * 60)

    for i, int_weight in enumerate(model.conv1.integration_from_streams):
        if int_weight.grad is not None:
            grad = int_weight.grad
            print(f"\n  integration_from_streams[{i}] (shape {tuple(int_weight.shape)}):")
            print(f"    grad norm: {grad.norm().item():.6f}")
            print(f"    grad mean: {grad.abs().mean().item():.8f}")
            print(f"    weight norm: {int_weight.norm().item():.6f}")
        else:
            print(f"\n  integration_from_streams[{i}]: grad=None ⚠️")


def trace_forward_backward():
    """Trace the forward and backward pass manually to find where gradients disappear."""
    print("\n" + "=" * 80)
    print("Manual Forward/Backward Trace")
    print("=" * 80)

    # Create a minimal model - just conv1 and classifier
    model = li_resnet18(num_classes=10, stream_input_channels=[3, 1], device='cpu')
    model = model.to('cpu')
    model.train()

    batch_size = 2
    rgb = torch.randn(batch_size, 3, 224, 224, device='cpu', requires_grad=True)
    depth = torch.randn(batch_size, 1, 224, 224, device='cpu', requires_grad=True)
    targets = torch.randint(0, 10, (batch_size,), device='cpu')

    # Manual forward through conv1 only
    stream_outputs, integrated = model.conv1([rgb, depth], None)

    print(f"\nAfter conv1:")
    print(f"  stream_outputs[0] (RGB) shape: {stream_outputs[0].shape}, requires_grad: {stream_outputs[0].requires_grad}")
    print(f"  stream_outputs[1] (Depth) shape: {stream_outputs[1].shape}, requires_grad: {stream_outputs[1].requires_grad}")
    print(f"  integrated shape: {integrated.shape}, requires_grad: {integrated.requires_grad}")

    # Check gradient flow from integrated back to streams
    print("\n📋 Tracing gradient from integrated to stream_weights:")

    # Manually compute loss on integrated (simplified)
    integrated_flat = integrated.mean()

    # Backward
    integrated_flat.backward(retain_graph=True)

    print("\nGradients after backward on integrated.mean():")
    for i, stream_weight in enumerate(model.conv1.stream_weights):
        if stream_weight.grad is not None:
            print(f"  stream_weights[{i}]: grad norm = {stream_weight.grad.norm().item():.6f}")
        else:
            print(f"  stream_weights[{i}]: grad = None ⚠️")

    for i, int_weight in enumerate(model.conv1.integration_from_streams):
        if int_weight.grad is not None:
            print(f"  integration_from_streams[{i}]: grad norm = {int_weight.grad.norm().item():.6f}")
        else:
            print(f"  integration_from_streams[{i}]: grad = None ⚠️")


def check_gradient_through_integration():
    """Check if gradients flow through the integration step properly."""
    print("\n" + "=" * 80)
    print("Gradient Flow Through Integration Check")
    print("=" * 80)

    # Simulate the integration step manually
    batch_size = 2
    H, W = 56, 56
    in_ch = 64
    out_ch = 64

    # Simulate stream outputs (what comes out of conv2d)
    stream0_raw = torch.randn(batch_size, in_ch, H, W, requires_grad=True)
    stream1_raw = torch.randn(batch_size, in_ch, H, W, requires_grad=True)

    # Simulate integration weights
    int_weight_0 = torch.randn(out_ch, in_ch, 1, 1, requires_grad=True)
    int_weight_1 = torch.randn(out_ch, in_ch, 1, 1, requires_grad=True)

    # Integration step (matching LINet3 logic)
    integrated_from_0 = F.conv2d(stream0_raw, int_weight_0, None, stride=1, padding=0)
    integrated_from_1 = F.conv2d(stream1_raw, int_weight_1, None, stride=1, padding=0)

    integrated = integrated_from_0 + integrated_from_1

    # Backward
    loss = integrated.mean()
    loss.backward()

    print(f"\nSimulated Integration Gradients:")
    print(f"  stream0_raw grad norm: {stream0_raw.grad.norm().item():.6f}")
    print(f"  stream1_raw grad norm: {stream1_raw.grad.norm().item():.6f}")
    print(f"  int_weight_0 grad norm: {int_weight_0.grad.norm().item():.6f}")
    print(f"  int_weight_1 grad norm: {int_weight_1.grad.norm().item():.6f}")

    print("\n✓ Integration step DOES pass gradients to stream outputs (stream0_raw, stream1_raw)")
    print("  The issue must be in how stream_weights gradients are computed...")


def check_stream_weight_gradient_chain():
    """Check the full chain: integrated -> stream_raw -> stream_weight"""
    print("\n" + "=" * 80)
    print("Full Gradient Chain: integrated -> stream_raw -> stream_weight")
    print("=" * 80)

    batch_size = 2
    H, W = 224, 224
    in_ch = 3
    out_ch = 64

    # Input
    input_tensor = torch.randn(batch_size, in_ch, H, W, requires_grad=True)

    # Stream weight
    stream_weight = torch.randn(out_ch, in_ch, 7, 7, requires_grad=True)

    # Integration weight
    int_weight = torch.randn(out_ch, out_ch, 1, 1, requires_grad=True)

    # Forward: input -> conv -> stream_raw -> integration -> integrated
    stream_raw = F.conv2d(input_tensor, stream_weight, None, stride=2, padding=3)
    integrated = F.conv2d(stream_raw, int_weight, None, stride=1, padding=0)

    # Backward
    loss = integrated.mean()
    loss.backward()

    print(f"\nFull Chain Gradients:")
    print(f"  input_tensor grad norm: {input_tensor.grad.norm().item():.6f}")
    print(f"  stream_weight grad norm: {stream_weight.grad.norm().item():.6f}")
    print(f"  int_weight grad norm: {int_weight.grad.norm().item():.6f}")

    if stream_weight.grad.norm().item() > 0:
        print("\n✓ Gradients DO flow through the full chain!")
        print("  The issue must be specific to how LINet3 is structured...")
    else:
        print("\n⚠️ Gradients NOT flowing - there's a fundamental issue")


def check_linet3_conv1_specific():
    """Check the specific structure of conv1 in LINet3."""
    print("\n" + "=" * 80)
    print("LINet3 conv1 Specific Analysis")
    print("=" * 80)

    model = li_resnet18(num_classes=10, stream_input_channels=[3, 1], device='cpu')
    model = model.to('cpu')

    conv1 = model.conv1

    print(f"\nconv1 Structure:")
    print(f"  num_streams: {conv1.num_streams}")
    print(f"  stream_in_channels: {conv1.stream_in_channels}")
    print(f"  stream_out_channels: {conv1.stream_out_channels}")
    print(f"  integrated_in_channels: {conv1.integrated_in_channels}")
    print(f"  integrated_out_channels: {conv1.integrated_out_channels}")

    print(f"\n  stream_weights shapes:")
    for i, sw in enumerate(conv1.stream_weights):
        print(f"    [{i}]: {tuple(sw.shape)}")

    print(f"\n  integration_from_streams shapes:")
    for i, iw in enumerate(conv1.integration_from_streams):
        print(f"    [{i}]: {tuple(iw.shape)}")

    print(f"\n  integrated_weight shape: {tuple(conv1.integrated_weight.shape)}")
    print(f"  integrated_weight numel: {conv1.integrated_weight.numel()}")

    # Note: integrated_in_channels=0 means no previous integrated input
    # This is fine for the first layer
    if conv1.integrated_in_channels == 0:
        print("\n⚠️  integrated_in_channels=0 (first layer, no previous integrated)")
        print("   integrated_weight has shape (64, 0, 1, 1) - zero elements!")
        print("   This is expected but the integrated_weight won't learn anything")


def debug_actual_gradient_values():
    """Look at the actual gradient values for stream0 vs stream1."""
    print("\n" + "=" * 80)
    print("Actual Gradient Value Comparison")
    print("=" * 80)

    model = li_resnet18(num_classes=10, stream_input_channels=[3, 1], device='cpu')
    model = model.to('cpu')
    model.train()

    batch_size = 4
    rgb = torch.randn(batch_size, 3, 224, 224, device='cpu')
    depth = torch.randn(batch_size, 1, 224, 224, device='cpu')
    targets = torch.randint(0, 10, (batch_size,), device='cpu')

    logits = model([rgb, depth])
    loss = F.cross_entropy(logits, targets)
    loss.backward()

    # Collect all gradients by stream index
    stream_grads = {0: [], 1: []}

    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        if 'stream_weights' in name:
            # Parse stream index from name like 'conv1.stream_weights.0'
            parts = name.split('.')
            for i, part in enumerate(parts):
                if part == 'stream_weights' and i + 1 < len(parts):
                    stream_idx = int(parts[i + 1])
                    grad_norm = param.grad.norm().item()
                    stream_grads[stream_idx].append((name, grad_norm))
                    break

    print("\n📊 Stream0 (RGB) Gradient Norms by Layer:")
    for name, norm in sorted(stream_grads[0], key=lambda x: x[0]):
        short_name = '.'.join(name.split('.')[:2])
        status = "❌ ZERO" if norm < 1e-8 else "✓"
        print(f"  {short_name}: {norm:.6f} {status}")

    print("\n📊 Stream1 (Depth) Gradient Norms by Layer:")
    for name, norm in sorted(stream_grads[1], key=lambda x: x[0]):
        short_name = '.'.join(name.split('.')[:2])
        status = "❌ ZERO" if norm < 1e-8 else "✓"
        print(f"  {short_name}: {norm:.6f} {status}")

    # Count zeros
    stream0_zeros = sum(1 for _, n in stream_grads[0] if n < 1e-8)
    stream1_zeros = sum(1 for _, n in stream_grads[1] if n < 1e-8)

    print(f"\n📈 Summary:")
    print(f"  Stream0 zero gradients: {stream0_zeros}/{len(stream_grads[0])}")
    print(f"  Stream1 zero gradients: {stream1_zeros}/{len(stream_grads[1])}")

    if stream0_zeros > stream1_zeros:
        print("\n🚨 CRITICAL: Stream0 has more zero gradients than Stream1!")
        print("   This explains why RGB stream isn't learning!")


if __name__ == "__main__":
    print("\n🔍 Diagnosing Stream0 Zero Gradient Issue...\n")

    debug_stream_gradient_flow()
    debug_integration_weights()
    trace_forward_backward()
    check_gradient_through_integration()
    check_stream_weight_gradient_chain()
    check_linet3_conv1_specific()
    debug_actual_gradient_values()
