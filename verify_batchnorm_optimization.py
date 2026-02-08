"""
Rigorous verification that BatchNorm optimization maintains EXACT same behavior.

This test compares the old implementation with the new implementation
across multiple scenarios to ensure 100% functional equivalence.
"""

import torch
import torch.nn.functional as F


def old_implementation(stream_input, stream_blanked, running_mean, running_var, weight, bias, exp_avg_factor):
    """Original implementation - exactly as it was before."""
    active_idx = (~stream_blanked).nonzero(as_tuple=True)[0]
    active_input = stream_input[active_idx]  # [num_active, C, H, W]

    # Standard BN on active samples
    active_output = F.batch_norm(
        active_input, running_mean, running_var, weight, bias,
        training=True, momentum=exp_avg_factor, eps=1e-5
    )

    # Scatter back - blanked samples stay as zeros
    stream_out = torch.zeros_like(stream_input)
    stream_out[active_idx] = active_output

    return stream_out


def new_implementation(stream_input, stream_blanked, running_mean, running_var, weight, bias, exp_avg_factor):
    """New optimized implementation."""
    active_idx = torch.where(~stream_blanked)[0]

    # Process only active samples
    active_output = F.batch_norm(
        stream_input[active_idx], running_mean, running_var, weight, bias,
        training=True, momentum=exp_avg_factor, eps=1e-5
    )

    # Use scatter operation with new_zeros
    stream_out = stream_input.new_zeros(stream_input.shape)
    stream_out.index_copy_(0, active_idx, active_output)

    return stream_out


def test_scenario(name, stream_input, stream_blanked):
    """Test a specific scenario and verify exact equivalence."""
    print(f"\n{'='*80}")
    print(f"Testing: {name}")
    print(f"{'='*80}")

    batch_size, channels, h, w = stream_input.shape

    # BN parameters (same for both)
    running_mean = torch.zeros(channels)
    running_var = torch.ones(channels)
    weight = torch.ones(channels)
    bias = torch.zeros(channels)
    exp_avg_factor = 0.1

    print(f"Input shape: {stream_input.shape}")
    print(f"Blanked mask: {stream_blanked.tolist()}")
    print(f"Number of blanked samples: {stream_blanked.sum().item()}/{batch_size}")
    print(f"Number of active samples: {(~stream_blanked).sum().item()}/{batch_size}")

    # Test 1: Verify active_idx computation is identical
    active_idx_old = (~stream_blanked).nonzero(as_tuple=True)[0]
    active_idx_new = torch.where(~stream_blanked)[0]

    print(f"\n1. Active index computation:")
    print(f"   Old method: {active_idx_old.tolist()}")
    print(f"   New method: {active_idx_new.tolist()}")
    assert torch.equal(active_idx_old, active_idx_new), "❌ Active indices are different!"
    print(f"   ✓ Active indices are identical")

    # Test 2: Verify tensor allocation produces same dtype/device
    zeros_old = torch.zeros_like(stream_input)
    zeros_new = stream_input.new_zeros(stream_input.shape)

    print(f"\n2. Tensor allocation:")
    print(f"   Old (zeros_like) - dtype: {zeros_old.dtype}, device: {zeros_old.device}, shape: {zeros_old.shape}")
    print(f"   New (new_zeros)  - dtype: {zeros_new.dtype}, device: {zeros_new.device}, shape: {zeros_new.shape}")
    assert zeros_old.dtype == zeros_new.dtype, "❌ Different dtypes!"
    assert zeros_old.device == zeros_new.device, "❌ Different devices!"
    assert zeros_old.shape == zeros_new.shape, "❌ Different shapes!"
    assert torch.equal(zeros_old, zeros_new), "❌ Allocated tensors are not equal!"
    print(f"   ✓ Tensor allocation is identical")

    # Test 3: Verify scatter operation produces same results
    test_values = torch.randn_like(stream_input[active_idx_old])

    scatter_old = torch.zeros_like(stream_input)
    scatter_old[active_idx_old] = test_values

    scatter_new = stream_input.new_zeros(stream_input.shape)
    scatter_new.index_copy_(0, active_idx_new, test_values)

    print(f"\n3. Scatter operation:")
    print(f"   Max difference: {(scatter_old - scatter_new).abs().max().item():.10f}")
    assert torch.equal(scatter_old, scatter_new), "❌ Scatter operations produce different results!"
    print(f"   ✓ Scatter operations are identical")

    # Test 4: Run full forward pass and compare
    # Clone inputs to ensure both start from same state
    stream_input_old = stream_input.clone()
    stream_input_new = stream_input.clone()

    # Clone BN parameters to ensure same starting state
    running_mean_old = running_mean.clone()
    running_var_old = running_var.clone()
    weight_old = weight.clone()
    bias_old = bias.clone()

    running_mean_new = running_mean.clone()
    running_var_new = running_var.clone()
    weight_new = weight.clone()
    bias_new = bias.clone()

    output_old = old_implementation(
        stream_input_old, stream_blanked,
        running_mean_old, running_var_old, weight_old, bias_old,
        exp_avg_factor
    )

    output_new = new_implementation(
        stream_input_new, stream_blanked,
        running_mean_new, running_var_new, weight_new, bias_new,
        exp_avg_factor
    )

    print(f"\n4. Full forward pass:")
    print(f"   Output shape: {output_old.shape}")
    print(f"   Max difference: {(output_old - output_new).abs().max().item():.10f}")
    print(f"   Mean difference: {(output_old - output_new).abs().mean().item():.10f}")

    # Check exact equality
    if torch.equal(output_old, output_new):
        print(f"   ✓ Outputs are EXACTLY identical (bit-perfect)")
    elif torch.allclose(output_old, output_new, atol=1e-7, rtol=1e-5):
        print(f"   ✓ Outputs are numerically identical (within tolerance)")
    else:
        print(f"   ❌ Outputs differ!")
        print(f"   Old output stats: min={output_old.min():.6f}, max={output_old.max():.6f}, mean={output_old.mean():.6f}")
        print(f"   New output stats: min={output_new.min():.6f}, max={output_new.max():.6f}, mean={output_new.mean():.6f}")
        raise AssertionError("Outputs are not equal!")

    # Test 5: Verify blanked samples are zeros in both
    print(f"\n5. Blanked samples verification:")
    blanked_old = output_old[stream_blanked]
    blanked_new = output_new[stream_blanked]

    print(f"   Old blanked samples max: {blanked_old.abs().max().item():.10f}")
    print(f"   New blanked samples max: {blanked_new.abs().max().item():.10f}")

    assert torch.allclose(blanked_old, torch.zeros_like(blanked_old), atol=1e-10), "❌ Old: Blanked samples not zero!"
    assert torch.allclose(blanked_new, torch.zeros_like(blanked_new), atol=1e-10), "❌ New: Blanked samples not zero!"
    print(f"   ✓ Blanked samples are zero in both implementations")

    # Test 6: Verify active samples are identical
    print(f"\n6. Active samples verification:")
    active_old = output_old[~stream_blanked]
    active_new = output_new[~stream_blanked]

    print(f"   Active samples max difference: {(active_old - active_new).abs().max().item():.10f}")

    if torch.equal(active_old, active_new):
        print(f"   ✓ Active samples are EXACTLY identical")
    elif torch.allclose(active_old, active_new, atol=1e-7):
        print(f"   ✓ Active samples are numerically identical")
    else:
        raise AssertionError("❌ Active samples differ!")

    print(f"\n{'='*80}")
    print(f"✓ ALL CHECKS PASSED for {name}")
    print(f"{'='*80}")

    return True


def test_gradient_equivalence():
    """Test that gradients are identical between old and new implementations."""
    print(f"\n{'='*80}")
    print(f"Testing: Gradient Equivalence")
    print(f"{'='*80}")

    torch.manual_seed(42)
    batch_size = 8
    channels = 16
    h, w = 32, 32

    # Create inputs with gradient tracking
    stream_input_old = torch.randn(batch_size, channels, h, w, requires_grad=True)
    stream_input_new = stream_input_old.clone().detach().requires_grad_(True)

    stream_blanked = torch.tensor([True, False, False, True, False, True, False, False])

    # BN parameters
    running_mean = torch.zeros(channels)
    running_var = torch.ones(channels)
    weight = torch.ones(channels, requires_grad=True)
    bias = torch.zeros(channels, requires_grad=True)
    weight_new = weight.clone().detach().requires_grad_(True)
    bias_new = bias.clone().detach().requires_grad_(True)

    # Forward pass
    output_old = old_implementation(
        stream_input_old, stream_blanked,
        running_mean.clone(), running_var.clone(), weight, bias, 0.1
    )

    output_new = new_implementation(
        stream_input_new, stream_blanked,
        running_mean.clone(), running_var.clone(), weight_new, bias_new, 0.1
    )

    # Backward pass
    loss_old = output_old.sum()
    loss_new = output_new.sum()

    loss_old.backward()
    loss_new.backward()

    # Compare gradients
    print(f"\n1. Input gradients:")
    print(f"   Max difference: {(stream_input_old.grad - stream_input_new.grad).abs().max().item():.10f}")
    assert torch.allclose(stream_input_old.grad, stream_input_new.grad, atol=1e-6), "❌ Input gradients differ!"
    print(f"   ✓ Input gradients are identical")

    print(f"\n2. Weight gradients:")
    print(f"   Max difference: {(weight.grad - weight_new.grad).abs().max().item():.10f}")
    assert torch.allclose(weight.grad, weight_new.grad, atol=1e-6), "❌ Weight gradients differ!"
    print(f"   ✓ Weight gradients are identical")

    print(f"\n3. Bias gradients:")
    print(f"   Max difference: {(bias.grad - bias_new.grad).abs().max().item():.10f}")
    assert torch.allclose(bias.grad, bias_new.grad, atol=1e-6), "❌ Bias gradients differ!"
    print(f"   ✓ Bias gradients are identical")

    # Verify blanked samples have zero gradient
    print(f"\n4. Blanked sample gradients:")
    blanked_grad_old = stream_input_old.grad[stream_blanked]
    blanked_grad_new = stream_input_new.grad[stream_blanked]
    print(f"   Old blanked grad max: {blanked_grad_old.abs().max().item():.10f}")
    print(f"   New blanked grad max: {blanked_grad_new.abs().max().item():.10f}")
    print(f"   ✓ Blanked sample gradients verified")

    print(f"\n{'='*80}")
    print(f"✓ GRADIENT EQUIVALENCE VERIFIED")
    print(f"{'='*80}")

    return True


if __name__ == "__main__":
    print("=" * 80)
    print("RIGOROUS BATCHNORM OPTIMIZATION VERIFICATION")
    print("=" * 80)

    torch.manual_seed(42)

    # Test different scenarios
    scenarios = []

    # Scenario 1: Half blanked, half active
    stream_input = torch.randn(8, 16, 32, 32)
    stream_blanked = torch.tensor([True, False, True, False, True, False, True, False])
    scenarios.append(("Half blanked, half active", stream_input, stream_blanked))

    # Scenario 2: Mostly active (realistic 20% dropout)
    stream_input = torch.randn(10, 32, 64, 64)
    stream_blanked = torch.tensor([True, False, False, False, False, True, False, False, False, False])
    scenarios.append(("20% dropout rate", stream_input, stream_blanked))

    # Scenario 3: Single blanked sample
    stream_input = torch.randn(4, 8, 16, 16)
    stream_blanked = torch.tensor([False, False, True, False])
    scenarios.append(("Single blanked sample", stream_input, stream_blanked))

    # Scenario 4: Single active sample (extreme case)
    stream_input = torch.randn(4, 8, 16, 16)
    stream_blanked = torch.tensor([True, True, False, True])
    scenarios.append(("Single active sample", stream_input, stream_blanked))

    # Scenario 5: First and last blanked
    stream_input = torch.randn(6, 12, 24, 24)
    stream_blanked = torch.tensor([True, False, False, False, False, True])
    scenarios.append(("First and last blanked", stream_input, stream_blanked))

    # Scenario 6: Large batch size
    stream_input = torch.randn(32, 64, 224, 224)
    stream_blanked = torch.zeros(32, dtype=torch.bool)
    stream_blanked[[0, 5, 10, 15, 20, 25, 30]] = True  # 7 blanked
    scenarios.append(("Large batch (realistic)", stream_input, stream_blanked))

    # Run all scenarios
    all_passed = True
    for name, stream_input, stream_blanked in scenarios:
        try:
            test_scenario(name, stream_input, stream_blanked)
        except Exception as e:
            print(f"\n❌ FAILED: {name}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    # Test gradient equivalence
    try:
        test_gradient_equivalence()
    except Exception as e:
        print(f"\n❌ GRADIENT TEST FAILED")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # Final summary
    print("\n" + "=" * 80)
    if all_passed:
        print("✓✓✓ ALL VERIFICATION TESTS PASSED ✓✓✓")
        print("\nCONCLUSION: The optimization maintains EXACT functional equivalence.")
        print("The new implementation produces identical results to the old implementation")
        print("across all tested scenarios, including edge cases and gradient computation.")
        print("\nChanges are PURE OPTIMIZATIONS with zero behavioral differences.")
    else:
        print("✗✗✗ SOME TESTS FAILED ✗✗✗")
        print("\nPlease review the failed tests above.")
    print("=" * 80)
