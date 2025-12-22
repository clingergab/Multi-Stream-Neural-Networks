"""
Comprehensive BatchNorm divergence diagnosis between LINet3 and Original.

This script:
1. Compares BN running stats after each forward pass
2. Checks for bugs in BN initialization/calling order
3. Tests with track_running_stats=False to verify the theory
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

import sys
sys.path.insert(0, '/Users/gclinger/Documents/projects/Multi-Stream-Neural-Networks')

from src.models.linear_integration.li_net3.li_net import LINet as LINet3
from src.models.linear_integration.li_net3.blocks import LIBasicBlock as LIBasicBlock3
from src.models.linear_integration.li_net3.conv import LIBatchNorm2d as LIBatchNorm2d3

from src.models.linear_integration.li_net import LINet as LINetOriginal
from src.models.linear_integration.blocks import LIBasicBlock as LIBasicBlockOriginal
from src.models.linear_integration.conv import LIBatchNorm2d as LIBatchNorm2dOriginal

SEED = 42


def create_models(seed=SEED):
    """Create both models with identical seeds."""
    torch.manual_seed(seed)
    model_orig = LINetOriginal(
        block=LIBasicBlockOriginal,
        layers=[2, 2, 2, 2],
        num_classes=10,
        stream1_input_channels=3,
        stream2_input_channels=1,
        device='cpu'
    ).to('cpu')

    torch.manual_seed(seed)
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


# ============================================================================
# Test 1: Compare BN running stats step by step
# ============================================================================
def test_bn_running_stats_per_step():
    """Compare BN running stats after EACH forward pass."""
    print("=" * 80)
    print("Test 1: BN Running Stats Per Forward Pass")
    print("=" * 80)

    model_orig, model_linet3 = create_models()
    model_orig.train()
    model_linet3.train()

    def get_bn1_stats(model, model_type):
        """Get bn1 running stats."""
        bn1 = model.bn1
        if model_type == "original":
            return {
                'stream1_mean': bn1.stream1_running_mean.clone(),
                'stream1_var': bn1.stream1_running_var.clone(),
                'stream2_mean': bn1.stream2_running_mean.clone(),
                'stream2_var': bn1.stream2_running_var.clone(),
                'integrated_mean': bn1.integrated_running_mean.clone() if bn1.integrated_running_mean is not None else None,
                'integrated_var': bn1.integrated_running_var.clone() if bn1.integrated_running_var is not None else None,
            }
        else:  # linet3
            stats = {}
            # LINet3 uses stream{i}_running_mean attribute naming
            for i in range(bn1.num_streams):
                rm = getattr(bn1, f'stream{i}_running_mean')
                rv = getattr(bn1, f'stream{i}_running_var')
                stats[f'stream{i}_mean'] = rm.clone() if rm is not None else None
                stats[f'stream{i}_var'] = rv.clone() if rv is not None else None
            if bn1.integrated_running_mean is not None:
                stats['integrated_mean'] = bn1.integrated_running_mean.clone()
                stats['integrated_var'] = bn1.integrated_running_var.clone()
            return stats

    # Get initial stats
    print("\n--- Initial BN1 Running Stats ---")
    orig_stats_0 = get_bn1_stats(model_orig, "original")
    li3_stats_0 = get_bn1_stats(model_linet3, "linet3")

    print(f"Original stream1_mean[0]: {orig_stats_0['stream1_mean'][0]:.6f}")
    print(f"LINet3  stream0_mean[0]:  {li3_stats_0['stream0_mean'][0]:.6f}")

    # Forward pass 1
    rgb, depth, targets = create_inputs(SEED)

    with torch.no_grad():
        _ = model_orig(rgb, depth)
        _ = model_linet3([rgb, depth])

    orig_stats_1 = get_bn1_stats(model_orig, "original")
    li3_stats_1 = get_bn1_stats(model_linet3, "linet3")

    print("\n--- After 1 Forward Pass ---")
    print(f"Original stream1_mean[0]: {orig_stats_1['stream1_mean'][0]:.6f}")
    print(f"LINet3  stream0_mean[0]:  {li3_stats_1['stream0_mean'][0]:.6f}")

    diff_mean = (orig_stats_1['stream1_mean'] - li3_stats_1['stream0_mean']).abs().max().item()
    diff_var = (orig_stats_1['stream1_var'] - li3_stats_1['stream0_var']).abs().max().item()
    print(f"Stream1/0 mean diff: {diff_mean:.2e}")
    print(f"Stream1/0 var diff:  {diff_var:.2e}")

    # Forward pass 2 (same input)
    with torch.no_grad():
        _ = model_orig(rgb, depth)
        _ = model_linet3([rgb, depth])

    orig_stats_2 = get_bn1_stats(model_orig, "original")
    li3_stats_2 = get_bn1_stats(model_linet3, "linet3")

    print("\n--- After 2 Forward Passes ---")
    diff_mean = (orig_stats_2['stream1_mean'] - li3_stats_2['stream0_mean']).abs().max().item()
    diff_var = (orig_stats_2['stream1_var'] - li3_stats_2['stream0_var']).abs().max().item()
    print(f"Stream1/0 mean diff: {diff_mean:.2e}")
    print(f"Stream1/0 var diff:  {diff_var:.2e}")

    # Forward pass 3 (different input)
    rgb2, depth2, _ = create_inputs(SEED + 1)
    with torch.no_grad():
        _ = model_orig(rgb2, depth2)
        _ = model_linet3([rgb2, depth2])

    orig_stats_3 = get_bn1_stats(model_orig, "original")
    li3_stats_3 = get_bn1_stats(model_linet3, "linet3")

    print("\n--- After 3 Forward Passes (new input) ---")
    diff_mean = (orig_stats_3['stream1_mean'] - li3_stats_3['stream0_mean']).abs().max().item()
    diff_var = (orig_stats_3['stream1_var'] - li3_stats_3['stream0_var']).abs().max().item()
    print(f"Stream1/0 mean diff: {diff_mean:.2e}")
    print(f"Stream1/0 var diff:  {diff_var:.2e}")

    if diff_mean < 1e-6 and diff_var < 1e-6:
        print("\n✅ BN running stats are identical!")
    else:
        print("\n⚠️  BN running stats diverge!")


# ============================================================================
# Test 2: Check BN initialization and num_batches_tracked
# ============================================================================
def test_bn_initialization():
    """Check BN initialization details."""
    print("\n" + "=" * 80)
    print("Test 2: BN Initialization Check")
    print("=" * 80)

    model_orig, model_linet3 = create_models()

    # Check bn1 initialization
    bn1_orig = model_orig.bn1
    bn1_li3 = model_linet3.bn1

    print("\n--- Original bn1 ---")
    print(f"  stream1_running_mean init: {bn1_orig.stream1_running_mean.mean():.6f}")
    print(f"  stream1_running_var init:  {bn1_orig.stream1_running_var.mean():.6f}")
    print(f"  num_batches_tracked: {bn1_orig.num_batches_tracked}")

    print("\n--- LINet3 bn1 ---")
    stream0_mean = getattr(bn1_li3, 'stream0_running_mean')
    stream0_var = getattr(bn1_li3, 'stream0_running_var')
    print(f"  stream0_running_mean init: {stream0_mean.mean():.6f}")
    print(f"  stream0_running_var init:  {stream0_var.mean():.6f}")
    print(f"  num_batches_tracked: {bn1_li3.num_batches_tracked}")

    # Check momentum
    print("\n--- Momentum ---")
    print(f"Original momentum: {bn1_orig.momentum}")
    print(f"LINet3 momentum:   {bn1_li3.momentum}")

    # Check eps
    print("\n--- Epsilon ---")
    print(f"Original eps: {bn1_orig.eps}")
    print(f"LINet3 eps:   {bn1_li3.eps}")


# ============================================================================
# Test 3: Check BN forward method differences
# ============================================================================
def test_bn_forward_method():
    """Check if BN forward methods are called in the same order."""
    print("\n" + "=" * 80)
    print("Test 3: BN Forward Method Tracing")
    print("=" * 80)

    model_orig, model_linet3 = create_models()
    model_orig.train()
    model_linet3.train()

    # Track BN calls
    orig_bn_calls = []
    li3_bn_calls = []

    def make_bn_hook(call_list, name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                call_list.append((name, 'tuple', len(output)))
            else:
                call_list.append((name, 'tensor', output.shape))
        return hook

    # Register hooks on all BN layers
    for name, module in model_orig.named_modules():
        if 'bn' in name.lower() and hasattr(module, 'stream1_running_mean'):
            module.register_forward_hook(make_bn_hook(orig_bn_calls, name))

    for name, module in model_linet3.named_modules():
        if 'bn' in name.lower() and hasattr(module, 'stream0_running_mean'):
            module.register_forward_hook(make_bn_hook(li3_bn_calls, name))

    rgb, depth, _ = create_inputs()

    with torch.no_grad():
        _ = model_orig(rgb, depth)
        _ = model_linet3([rgb, depth])

    print(f"\nOriginal BN layer calls: {len(orig_bn_calls)}")
    for call in orig_bn_calls[:5]:
        print(f"  {call}")

    print(f"\nLINet3 BN layer calls: {len(li3_bn_calls)}")
    for call in li3_bn_calls[:5]:
        print(f"  {call}")


# ============================================================================
# Test 4: Test with track_running_stats=False
# ============================================================================
def test_without_running_stats():
    """Test if gradients match when track_running_stats=False."""
    print("\n" + "=" * 80)
    print("Test 4: Training Without Running Stats (track_running_stats=False)")
    print("=" * 80)

    # We need to create models and then disable track_running_stats
    model_orig, model_linet3 = create_models()

    # Disable track_running_stats on all BN layers
    def disable_running_stats(model):
        for module in model.modules():
            if hasattr(module, 'track_running_stats'):
                module.track_running_stats = False
                # Also set running_mean/var to None to ensure they're not used
                if hasattr(module, 'running_mean'):
                    module.running_mean = None
                if hasattr(module, 'running_var'):
                    module.running_var = None
            # For multi-stream BN (Original 2-stream)
            if hasattr(module, 'stream1_running_mean'):
                module.stream1_running_mean = None
                module.stream1_running_var = None
                module.stream2_running_mean = None
                module.stream2_running_var = None
            # For multi-stream BN (LINet3 N-stream)
            if hasattr(module, 'stream0_running_mean'):
                for i in range(getattr(module, 'num_streams', 2)):
                    setattr(module, f'stream{i}_running_mean', None)
                    setattr(module, f'stream{i}_running_var', None)
            if hasattr(module, 'integrated_running_mean'):
                module.integrated_running_mean = None
                module.integrated_running_var = None

    disable_running_stats(model_orig)
    disable_running_stats(model_linet3)

    model_orig.train()
    model_linet3.train()

    rgb, depth, targets = create_inputs()

    # Forward
    out_orig = model_orig(rgb, depth)
    out_li3 = model_linet3([rgb, depth])

    print(f"\n--- Forward Pass (no running stats) ---")
    print(f"Output diff: {(out_orig - out_li3).abs().max():.2e}")

    # Backward
    loss_orig = F.cross_entropy(out_orig, targets)
    loss_li3 = F.cross_entropy(out_li3, targets)

    loss_orig.backward()
    loss_li3.backward()

    # Compare gradients with proper name mapping
    print(f"\n--- Gradient Comparison (no running stats) ---")

    # Build parameter dict for LINet3 with mapped names
    def map_li3_name_to_orig(li3_name):
        """Map LINet3 parameter name to Original parameter name."""
        return (li3_name
                .replace('stream_weights.0', 'stream1_weight')
                .replace('stream_weights.1', 'stream2_weight')
                .replace('stream_biases.0', 'stream1_bias')
                .replace('stream_biases.1', 'stream2_bias')
                .replace('integration_from_streams.0', 'integration_from_stream1')
                .replace('integration_from_streams.1', 'integration_from_stream2'))

    li3_params = {map_li3_name_to_orig(n): p for n, p in model_linet3.named_parameters()}

    grad_diffs = []
    for orig_name, orig_param in model_orig.named_parameters():
        if orig_param.grad is None:
            continue
        if orig_name in li3_params:
            li3_param = li3_params[orig_name]
            if li3_param.grad is not None and orig_param.shape == li3_param.shape:
                diff = (orig_param.grad - li3_param.grad).abs().max().item()
                grad_diffs.append((orig_name, diff))

    # Show layers with largest differences
    grad_diffs.sort(key=lambda x: x[1], reverse=True)
    print(f"\nTop 10 gradient differences (properly mapped):")
    for name, diff in grad_diffs[:10]:
        flag = " ⚠️" if diff > 1e-5 else ""
        print(f"  {name}: {diff:.2e}{flag}")

    max_diff = max(d for _, d in grad_diffs) if grad_diffs else 0
    if max_diff < 1e-5:
        print(f"\n✅ Gradients are identical when track_running_stats=False!")
        print("   This confirms the issue is with BN running stats.")
    else:
        print(f"\n⚠️  Gradients still differ (max: {max_diff:.2e})")
        print("   The issue may be elsewhere.")


# ============================================================================
# Test 5: Compare actual BN forward implementations
# ============================================================================
def test_bn_forward_implementations():
    """Compare the actual BN forward code."""
    print("\n" + "=" * 80)
    print("Test 5: BN Forward Implementation Comparison")
    print("=" * 80)

    # Create isolated BN layers
    torch.manual_seed(SEED)
    bn_orig = LIBatchNorm2dOriginal(
        stream1_num_features=64,
        stream2_num_features=64,
        integrated_num_features=64
    )

    torch.manual_seed(SEED)
    bn_li3 = LIBatchNorm2d3(
        stream_num_features=[64, 64],
        integrated_num_features=64
    )

    bn_orig.train()
    bn_li3.train()

    # Create test inputs
    torch.manual_seed(SEED)
    s1 = torch.randn(4, 64, 8, 8)
    s2 = torch.randn(4, 64, 8, 8)
    integrated = torch.randn(4, 64, 8, 8)

    # Forward pass
    out1_orig, out2_orig, int_orig = bn_orig(s1, s2, integrated)
    (out_streams_li3, int_li3) = bn_li3([s1, s2], integrated)

    print("\n--- Isolated BN Forward Comparison ---")
    print(f"Stream1 output diff: {(out1_orig - out_streams_li3[0]).abs().max():.2e}")
    print(f"Stream2 output diff: {(out2_orig - out_streams_li3[1]).abs().max():.2e}")
    print(f"Integrated output diff: {(int_orig - int_li3).abs().max():.2e}")

    # Check running stats after forward
    print("\n--- Running Stats After Forward ---")
    print(f"Original stream1_running_mean[0]: {bn_orig.stream1_running_mean[0]:.6f}")
    li3_stream0_mean = getattr(bn_li3, 'stream0_running_mean')
    print(f"LINet3  stream0_running_mean[0]:  {li3_stream0_mean[0]:.6f}")

    diff = (bn_orig.stream1_running_mean - li3_stream0_mean).abs().max().item()
    print(f"Running mean diff: {diff:.2e}")

    if diff < 1e-6:
        print("\n✅ Isolated BN layers produce identical results!")
    else:
        print("\n⚠️  Isolated BN layers differ!")


# ============================================================================
# Test 6: Check if parameter ordering affects anything
# ============================================================================
def test_parameter_ordering():
    """Check if parameter ordering in named_parameters() differs."""
    print("\n" + "=" * 80)
    print("Test 6: Parameter Ordering Check")
    print("=" * 80)

    model_orig, model_linet3 = create_models()

    orig_names = [n for n, _ in model_orig.named_parameters()]
    li3_names = [n for n, _ in model_linet3.named_parameters()]

    print(f"\nOriginal has {len(orig_names)} parameters")
    print(f"LINet3 has {len(li3_names)} parameters")

    # Check first 20 parameter names
    print("\n--- First 20 Parameter Names ---")
    print(f"{'Idx':<5} {'Original':<45} {'LINet3':<45}")
    print("-" * 95)

    for i in range(min(20, len(orig_names), len(li3_names))):
        o = orig_names[i]
        l = li3_names[i]
        match = "✓" if o.replace('stream1', 'stream_weights.0').replace('stream2', 'stream_weights.1') in l or o == l else "✗"
        print(f"{i:<5} {o:<45} {l:<45} {match}")


# ============================================================================
# Test 7: Detailed gradient comparison in train mode
# ============================================================================
def test_train_mode_gradients_detailed():
    """Detailed gradient comparison in train mode."""
    print("\n" + "=" * 80)
    print("Test 7: Detailed Train Mode Gradient Analysis")
    print("=" * 80)

    model_orig, model_linet3 = create_models()
    model_orig.train()
    model_linet3.train()

    rgb, depth, targets = create_inputs()

    out_orig = model_orig(rgb, depth)
    out_li3 = model_linet3([rgb, depth])

    loss_orig = F.cross_entropy(out_orig, targets)
    loss_li3 = F.cross_entropy(out_li3, targets)

    loss_orig.backward()
    loss_li3.backward()

    # Compare gradients by layer type
    conv_diffs = []
    bn_diffs = []
    integration_diffs = []
    other_diffs = []

    # Map LINet3 names to Original names
    def map_name(li3_name):
        return (li3_name
                .replace('stream_weights.0', 'stream1_weight')
                .replace('stream_weights.1', 'stream2_weight')
                .replace('stream_biases.0', 'stream1_bias')
                .replace('stream_biases.1', 'stream2_bias')
                .replace('integration_from_streams.0', 'integration_from_stream1')
                .replace('integration_from_streams.1', 'integration_from_stream2')
                .replace('stream_running_means.0', 'stream1_running_mean')
                .replace('stream_running_means.1', 'stream2_running_mean'))

    li3_params = {map_name(n): p for n, p in model_linet3.named_parameters()}

    for orig_name, orig_param in model_orig.named_parameters():
        if orig_param.grad is None:
            continue

        if orig_name in li3_params:
            li3_param = li3_params[orig_name]
            if li3_param.grad is not None and orig_param.shape == li3_param.shape:
                diff = (orig_param.grad - li3_param.grad).abs().max().item()

                if 'conv' in orig_name and 'bn' not in orig_name:
                    conv_diffs.append((orig_name, diff))
                elif 'bn' in orig_name:
                    bn_diffs.append((orig_name, diff))
                elif 'integration' in orig_name:
                    integration_diffs.append((orig_name, diff))
                else:
                    other_diffs.append((orig_name, diff))

    print("\n--- Gradient Differences by Category ---")

    print(f"\nConvolution layers ({len(conv_diffs)} matched):")
    conv_diffs.sort(key=lambda x: x[1], reverse=True)
    for name, diff in conv_diffs[:5]:
        print(f"  {name}: {diff:.2e}")

    print(f"\nBatchNorm layers ({len(bn_diffs)} matched):")
    bn_diffs.sort(key=lambda x: x[1], reverse=True)
    for name, diff in bn_diffs[:5]:
        print(f"  {name}: {diff:.2e}")

    print(f"\nIntegration layers ({len(integration_diffs)} matched):")
    integration_diffs.sort(key=lambda x: x[1], reverse=True)
    for name, diff in integration_diffs[:5]:
        print(f"  {name}: {diff:.2e}")

    print(f"\nOther layers ({len(other_diffs)} matched):")
    other_diffs.sort(key=lambda x: x[1], reverse=True)
    for name, diff in other_diffs[:5]:
        print(f"  {name}: {diff:.2e}")

    # Summary
    all_diffs = conv_diffs + bn_diffs + integration_diffs + other_diffs
    if all_diffs:
        max_diff = max(d for _, d in all_diffs)
        avg_diff = sum(d for _, d in all_diffs) / len(all_diffs)
        print(f"\n--- Summary ---")
        print(f"Max gradient diff: {max_diff:.2e}")
        print(f"Avg gradient diff: {avg_diff:.2e}")


# ============================================================================
# Test 8: Check if using eval mode fixes everything
# ============================================================================
def test_eval_mode_complete():
    """Complete test in eval mode to verify baseline."""
    print("\n" + "=" * 80)
    print("Test 8: Complete Eval Mode Test")
    print("=" * 80)

    model_orig, model_linet3 = create_models()
    model_orig.eval()
    model_linet3.eval()

    rgb, depth, targets = create_inputs()

    # Forward
    out_orig = model_orig(rgb, depth)
    out_li3 = model_linet3([rgb, depth])

    print(f"\n--- Eval Mode Forward ---")
    print(f"Output diff: {(out_orig - out_li3).abs().max():.2e}")

    # Backward
    out_orig.sum().backward()
    out_li3.sum().backward()

    # Compare all gradients with proper name mapping
    def map_li3_name_to_orig(li3_name):
        """Map LINet3 parameter name to Original parameter name."""
        return (li3_name
                .replace('stream_weights.0', 'stream1_weight')
                .replace('stream_weights.1', 'stream2_weight')
                .replace('stream_biases.0', 'stream1_bias')
                .replace('stream_biases.1', 'stream2_bias')
                .replace('integration_from_streams.0', 'integration_from_stream1')
                .replace('integration_from_streams.1', 'integration_from_stream2'))

    li3_params = {map_li3_name_to_orig(n): p for n, p in model_linet3.named_parameters()}

    max_diff = 0
    matched = 0
    for orig_name, orig_param in model_orig.named_parameters():
        if orig_param.grad is None:
            continue
        if orig_name in li3_params:
            li3_param = li3_params[orig_name]
            if li3_param.grad is not None and orig_param.shape == li3_param.shape:
                diff = (orig_param.grad - li3_param.grad).abs().max().item()
                max_diff = max(max_diff, diff)
                matched += 1

    print(f"Max gradient diff (eval mode, {matched} params matched): {max_diff:.2e}")

    if max_diff < 1e-6:
        print("\n✅ Eval mode: Forward and backward are identical!")
    else:
        print(f"\n⚠️  Eval mode shows differences: {max_diff:.2e}")


if __name__ == "__main__":
    print("\n🔍 Comprehensive BatchNorm Divergence Diagnosis\n")

    test_bn_running_stats_per_step()
    test_bn_initialization()
    test_bn_forward_method()
    test_without_running_stats()
    test_bn_forward_implementations()
    test_parameter_ordering()
    test_train_mode_gradients_detailed()
    test_eval_mode_complete()

    print("\n" + "=" * 80)
    print("DIAGNOSIS COMPLETE")
    print("=" * 80)
    print("""
Summary:
- If Test 4 (track_running_stats=False) shows identical gradients,
  the issue is definitively in BatchNorm running stats updates.
- If Test 5 (isolated BN) shows differences, the BN implementation itself differs.
- If Test 8 (eval mode) is identical, the forward pass logic is correct.

Recommended fixes:
1. Ensure BN momentum is identical
2. Ensure BN is initialized identically
3. Check if BN forward methods update stats in same order
4. Consider using SyncBatchNorm or disabling running stats during debugging
""")
