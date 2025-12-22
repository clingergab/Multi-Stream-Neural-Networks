"""
Test to verify BN running stats get corrupted during stream_monitoring in LINet3.

When stream_monitoring=True, LINet3's _forward_stream_pathway fills other streams with zeros
and runs a full forward pass. This corrupts BN running stats because F.batch_norm updates
the stats with zero-valued inputs.

This test:
1. Creates LINet3 model
2. Runs forward pass (BN stats updated with real data)
3. Simulates stream_monitoring by calling _forward_stream_pathway
4. Shows BN stats get corrupted (different from Original behavior)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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


def create_inputs():
    """Create identical inputs for both models."""
    torch.manual_seed(SEED)
    rgb = torch.randn(4, 3, 32, 32, device='cpu')
    depth = torch.randn(4, 1, 32, 32, device='cpu')
    targets = torch.randint(0, 10, (4,), device='cpu')
    return rgb, depth, targets


def test_bn_corruption_during_monitoring():
    """Test that stream monitoring is side-effect free (no BN stats changes)."""
    print("=" * 80)
    print("Testing Stream Monitoring is Side-Effect Free")
    print("=" * 80)

    model_orig, model_linet3 = create_models()
    model_orig.train()
    model_linet3.train()

    rgb, depth, _ = create_inputs()

    def get_bn1_stream1_stats(model, model_type):
        """Get stream1/0 running stats from bn1."""
        if model_type == "original":
            return (
                model.bn1.stream1_running_mean.clone(),
                model.bn1.stream1_running_var.clone()
            )
        else:
            return (
                getattr(model.bn1, 'stream0_running_mean').clone(),
                getattr(model.bn1, 'stream0_running_var').clone()
            )

    def get_bn1_stream2_stats(model, model_type):
        """Get stream2/1 running stats from bn1."""
        if model_type == "original":
            return (
                model.bn1.stream2_running_mean.clone(),
                model.bn1.stream2_running_var.clone()
            )
        else:
            return (
                getattr(model.bn1, 'stream1_running_mean').clone(),
                getattr(model.bn1, 'stream1_running_var').clone()
            )

    # Step 1: Initial state (should be 0/1)
    print("\n--- Initial BN Stats ---")
    orig_s1_mean, orig_s1_var = get_bn1_stream1_stats(model_orig, "original")
    li3_s0_mean, li3_s0_var = get_bn1_stream1_stats(model_linet3, "linet3")
    print(f"Original stream1_running_mean[0]: {orig_s1_mean[0]:.6f}")
    print(f"LINet3  stream0_running_mean[0]:  {li3_s0_mean[0]:.6f}")

    # Step 2: Main forward pass (both should update identically)
    print("\n--- After Main Forward Pass ---")
    with torch.no_grad():
        _ = model_orig(rgb, depth)
        _ = model_linet3([rgb, depth])

    orig_s1_mean_after_main, orig_s1_var_after_main = get_bn1_stream1_stats(model_orig, "original")
    li3_s0_mean_after_main, li3_s0_var_after_main = get_bn1_stream1_stats(model_linet3, "linet3")

    orig_s2_mean_after_main, orig_s2_var_after_main = get_bn1_stream2_stats(model_orig, "original")
    li3_s1_mean_after_main, li3_s1_var_after_main = get_bn1_stream2_stats(model_linet3, "linet3")

    diff_mean = (orig_s1_mean_after_main - li3_s0_mean_after_main).abs().max().item()
    print(f"Stream1/0 mean diff after main forward: {diff_mean:.2e}")
    print(f"Original stream1_mean[0]: {orig_s1_mean_after_main[0]:.6f}")
    print(f"LINet3  stream0_mean[0]:  {li3_s0_mean_after_main[0]:.6f}")

    # Step 3: Simulate stream monitoring - forward stream1/0 pathway
    # IMPORTANT: Stream monitoring should use eval mode to be side-effect free
    print("\n--- After Stream1/0 Pathway Forward (stream_monitoring with eval mode) ---")

    # Switch to eval mode before monitoring (this is what the training loop now does)
    model_orig.eval()
    model_linet3.eval()

    # Original uses dedicated forward_stream1
    _ = model_orig._forward_stream1_pathway(rgb)

    # LINet3 uses _forward_stream_pathway
    _ = model_linet3._forward_stream_pathway(0, rgb)

    # Restore training mode
    model_orig.train()
    model_linet3.train()

    orig_s1_mean_after_s1, _ = get_bn1_stream1_stats(model_orig, "original")
    li3_s0_mean_after_s0, _ = get_bn1_stream1_stats(model_linet3, "linet3")

    orig_s2_mean_after_s1, _ = get_bn1_stream2_stats(model_orig, "original")
    li3_s1_mean_after_s0, _ = get_bn1_stream2_stats(model_linet3, "linet3")

    print(f"Original stream1_mean[0]: {orig_s1_mean_after_s1[0]:.6f}")
    print(f"LINet3  stream0_mean[0]:  {li3_s0_mean_after_s0[0]:.6f}")

    # Key check: Did stream2/1 stats change?
    print("\n--- CRITICAL: Stream2/1 Stats After Stream1/0 Pathway ---")
    print(f"Original stream2_mean[0] BEFORE: {orig_s2_mean_after_main[0]:.6f}")
    print(f"Original stream2_mean[0] AFTER:  {orig_s2_mean_after_s1[0]:.6f}")
    print(f"  -> Change: {(orig_s2_mean_after_s1[0] - orig_s2_mean_after_main[0]).abs():.6f}")

    print(f"\nLINet3  stream1_mean[0] BEFORE: {li3_s1_mean_after_main[0]:.6f}")
    print(f"LINet3  stream1_mean[0] AFTER:  {li3_s1_mean_after_s0[0]:.6f}")
    print(f"  -> Change: {(li3_s1_mean_after_s0[0] - li3_s1_mean_after_main[0]).abs():.6f}")

    # Diagnosis
    orig_s2_changed = (orig_s2_mean_after_s1 - orig_s2_mean_after_main).abs().max().item() > 1e-8
    li3_s1_changed = (li3_s1_mean_after_s0 - li3_s1_mean_after_main).abs().max().item() > 1e-8

    print("\n" + "=" * 80)
    print("DIAGNOSIS:")
    print("=" * 80)

    if orig_s2_changed:
        print("  ⚠️  Original stream2 stats changed (unexpected)")
    else:
        print("  ✅ Original stream2 stats unchanged (correct - only stream1 updated)")

    if li3_s1_changed:
        print("  🚨 LINet3 stream1 stats CHANGED (CORRUPTED by zero inputs!)")
        print("\n  This is the BUG causing poor training performance!")
        print("  LINet3's _forward_stream_pathway updates ALL streams' BN stats,")
        print("  including streams that receive zero inputs.")
    else:
        print("  ✅ LINet3 stream1 stats unchanged (would be correct)")

    # Step 4: Forward stream2/1 pathway and check stream1/0 corruption
    print("\n\n--- After Stream2/1 Pathway Forward (stream_monitoring with eval mode) ---")

    model_orig.eval()
    model_linet3.eval()

    _ = model_orig._forward_stream2_pathway(depth)
    _ = model_linet3._forward_stream_pathway(1, depth)

    model_orig.train()
    model_linet3.train()

    orig_s1_mean_after_s2, _ = get_bn1_stream1_stats(model_orig, "original")
    li3_s0_mean_after_s1, _ = get_bn1_stream1_stats(model_linet3, "linet3")

    print(f"Original stream1_mean[0] BEFORE stream2: {orig_s1_mean_after_s1[0]:.6f}")
    print(f"Original stream1_mean[0] AFTER stream2:  {orig_s1_mean_after_s2[0]:.6f}")
    print(f"  -> Change: {(orig_s1_mean_after_s2[0] - orig_s1_mean_after_s1[0]).abs():.6f}")

    print(f"\nLINet3  stream0_mean[0] BEFORE stream1: {li3_s0_mean_after_s0[0]:.6f}")
    print(f"LINet3  stream0_mean[0] AFTER stream1:  {li3_s0_mean_after_s1[0]:.6f}")
    print(f"  -> Change: {(li3_s0_mean_after_s1[0] - li3_s0_mean_after_s0[0]).abs():.6f}")

    orig_s1_changed = (orig_s1_mean_after_s2 - orig_s1_mean_after_s1).abs().max().item() > 1e-8
    li3_s0_changed = (li3_s0_mean_after_s1 - li3_s0_mean_after_s0).abs().max().item() > 1e-8

    if orig_s1_changed:
        print("\n  ⚠️  Original stream1 stats changed when forwarding stream2 (unexpected)")
    else:
        print("\n  ✅ Original stream1 stats unchanged (correct)")

    if li3_s0_changed:
        print("  🚨 LINet3 stream0 stats CHANGED when forwarding stream1 (CORRUPTED!)")
    else:
        print("  ✅ LINet3 stream0 stats unchanged (would be correct)")


def test_magnitude_of_corruption():
    """Verify that stream monitoring with eval mode has zero side effects."""
    print("\n\n" + "=" * 80)
    print("Verifying Stream Monitoring is Side-Effect Free")
    print("=" * 80)

    model_orig, model_linet3 = create_models()
    model_orig.train()
    model_linet3.train()

    rgb, depth, _ = create_inputs()

    # Main forward pass
    with torch.no_grad():
        _ = model_orig(rgb, depth)
        _ = model_linet3([rgb, depth])

    # Get initial stats
    orig_s1_init = model_orig.bn1.stream1_running_mean.clone()
    li3_s0_init = getattr(model_linet3.bn1, 'stream0_running_mean').clone()

    print("\nSimulating 10 batches with stream_monitoring=True (using eval mode)...")
    print("(Each batch: 1 main forward + 2 stream pathway forwards in eval mode)\n")

    for i in range(10):
        # Simulate a training batch with stream_monitoring
        # Main forward (in training mode - this SHOULD update stats)
        _ = model_orig(rgb, depth)
        _ = model_linet3([rgb, depth])

        # Stream monitoring forwards (in eval mode - this should NOT update stats)
        model_orig.eval()
        model_linet3.eval()

        _ = model_orig._forward_stream1_pathway(rgb)
        _ = model_orig._forward_stream2_pathway(depth)

        _ = model_linet3._forward_stream_pathway(0, rgb)
        _ = model_linet3._forward_stream_pathway(1, depth)

        model_orig.train()
        model_linet3.train()

    # Compare final stats
    orig_s1_final = model_orig.bn1.stream1_running_mean.clone()
    li3_s0_final = getattr(model_linet3.bn1, 'stream0_running_mean').clone()

    diff_after_10 = (orig_s1_final - li3_s0_final).abs().max().item()

    print(f"After 10 batches with stream_monitoring:")
    print(f"  Stream1/0 running_mean diff: {diff_after_10:.6f}")

    # The Original and LINet3 should diverge significantly if LINet3 has the bug
    if diff_after_10 > 0.01:
        print(f"\n  🚨 SIGNIFICANT DIVERGENCE: {diff_after_10:.4f}")
        print("  This confirms BN stats corruption in LINet3 during stream monitoring!")
    else:
        print(f"\n  ✅ Stats remain similar (diff: {diff_after_10:.6f})")


if __name__ == "__main__":
    print("\n🔍 Testing BN Corruption During Stream Monitoring\n")

    test_bn_corruption_during_monitoring()
    test_magnitude_of_corruption()

    print("\n\n" + "=" * 80)
    print("RECOMMENDATIONS:")
    print("=" * 80)
    print("""
1. Add forward_stream{i} methods to LINet3's LIConv2d and LIBatchNorm2d
   that only process and update stats for the specified stream.

2. Or, wrap stream monitoring forward passes in eval() mode to prevent
   BN stats updates (but this would affect auxiliary classifier training).

3. Or, add an option to disable running stats updates during stream
   pathway forwards.

The recommended fix is option 1: implement dedicated forward_stream methods
that mirror the Original model's behavior.
""")
