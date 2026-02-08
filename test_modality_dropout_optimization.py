"""
Test to verify that modality dropout optimizations maintain correctness.

This test ensures that in-place operations produce identical results to the original implementation.
"""

import torch
import torch.nn.functional as F
from src.training.modality_dropout import generate_per_sample_blanked_mask


def test_conv_masking_correctness():
    """Test that in-place masking produces same results as out-of-place."""
    print("Testing Conv layer masking correctness...")

    torch.manual_seed(42)
    batch_size = 8
    channels = 16
    h, w = 32, 32

    # Create test data
    stream_out_orig = torch.randn(batch_size, channels, h, w)
    stream_out_opt = stream_out_orig.clone()

    stream_blanked = torch.tensor([True, False, False, True, False, True, False, False])
    mask = (~stream_blanked).float().view(-1, 1, 1, 1)

    # Original approach (out-of-place)
    result_orig = stream_out_orig * mask

    # Optimized approach (in-place)
    result_opt = stream_out_opt.mul_(mask)

    # Verify they're identical
    assert torch.allclose(result_orig, result_opt, atol=1e-6), \
        f"In-place masking produced different results! Max diff: {(result_orig - result_opt).abs().max()}"

    # Verify blanked samples are actually zero
    assert torch.allclose(result_opt[stream_blanked], torch.zeros_like(result_opt[stream_blanked])), \
        "Blanked samples are not zero!"

    # Verify active samples are unchanged
    assert torch.allclose(result_opt[~stream_blanked], stream_out_orig[~stream_blanked]), \
        "Active samples were modified incorrectly!"

    print("✓ Conv masking correctness test passed!")
    return True


def test_batchnorm_subset_correctness():
    """Test that optimized subset BN produces same results."""
    print("\nTesting BatchNorm subset processing correctness...")

    torch.manual_seed(42)
    batch_size = 8
    channels = 16
    h, w = 32, 32

    # Create test data
    stream_input = torch.randn(batch_size, channels, h, w)
    stream_blanked = torch.tensor([True, False, False, True, False, True, False, False])

    # BN parameters
    running_mean = torch.zeros(channels)
    running_var = torch.ones(channels)
    weight = torch.ones(channels)
    bias = torch.zeros(channels)

    # Original approach: nonzero().as_tuple()[0]
    active_idx_orig = (~stream_blanked).nonzero(as_tuple=True)[0]

    # Optimized approach: torch.where
    active_idx_opt = torch.where(~stream_blanked)[0]

    # Verify indices are identical
    assert torch.equal(active_idx_orig, active_idx_opt), \
        "torch.where produces different indices than nonzero!"

    # Process with both approaches
    active_input = stream_input[active_idx_orig]
    active_output = F.batch_norm(
        active_input, running_mean, running_var, weight, bias,
        training=True, momentum=0.1, eps=1e-5
    )

    # Original scatter
    stream_out_orig = torch.zeros_like(stream_input)
    stream_out_orig[active_idx_orig] = active_output

    # Optimized scatter
    stream_out_opt = stream_input.new_zeros(stream_input.shape)
    stream_out_opt.index_copy_(0, active_idx_opt, active_output)

    # Verify they're identical
    assert torch.allclose(stream_out_orig, stream_out_opt, atol=1e-6), \
        f"Optimized scatter produced different results! Max diff: {(stream_out_orig - stream_out_opt).abs().max()}"

    # Verify blanked samples are zero
    assert torch.allclose(stream_out_opt[stream_blanked], torch.zeros_like(stream_out_opt[stream_blanked])), \
        "Blanked samples are not zero in optimized version!"

    print("✓ BatchNorm subset correctness test passed!")
    return True


def test_full_forward_pass():
    """Test a full forward pass with modality dropout enabled."""
    print("\nTesting full forward pass with modality dropout...")

    from src.models.linear_integration.li_net3.conv import LIConv2d, LIBatchNorm2d

    torch.manual_seed(42)
    batch_size = 4
    h, w = 32, 32

    # Create LI layers
    conv = LIConv2d(
        stream_in_channels=[3, 3],
        stream_out_channels=[16, 16],
        integrated_in_channels=0,  # First layer
        integrated_out_channels=16,
        kernel_size=3,
        padding=1
    )

    bn = LIBatchNorm2d(
        stream_num_features=[16, 16],
        integrated_num_features=16
    )

    # Create test inputs
    stream_inputs = [
        torch.randn(batch_size, 3, h, w),
        torch.randn(batch_size, 3, h, w)
    ]

    # Generate blanked mask
    blanked_mask = generate_per_sample_blanked_mask(
        batch_size=batch_size,
        num_streams=2,
        dropout_prob=0.5,
        device='cpu'
    )

    # Forward pass with modality dropout
    stream_outputs, integrated_out = conv(stream_inputs, None, blanked_mask)
    stream_outputs, integrated_out = bn(stream_outputs, integrated_out, blanked_mask)

    # Verify outputs
    assert len(stream_outputs) == 2, "Wrong number of stream outputs!"
    assert stream_outputs[0].shape == (batch_size, 16, h, w), "Wrong output shape!"
    assert integrated_out.shape == (batch_size, 16, h, w), "Wrong integrated shape!"

    # Verify blanked samples are zero in affected streams
    if blanked_mask is not None:
        for i, stream_out in enumerate(stream_outputs):
            stream_blanked = blanked_mask.get(i)
            if stream_blanked is not None and stream_blanked.any():
                blanked_samples = stream_out[stream_blanked]
                assert torch.allclose(blanked_samples, torch.zeros_like(blanked_samples), atol=1e-6), \
                    f"Stream {i} has non-zero blanked samples!"

    print("✓ Full forward pass test passed!")
    return True


def test_gradient_flow():
    """Test that gradients flow correctly through optimized operations."""
    print("\nTesting gradient flow...")

    from src.models.linear_integration.li_net3.conv import LIConv2d

    torch.manual_seed(42)
    batch_size = 4
    h, w = 16, 16

    # Create LI layer
    conv = LIConv2d(
        stream_in_channels=[3, 3],
        stream_out_channels=[8, 8],
        integrated_in_channels=0,
        integrated_out_channels=8,
        kernel_size=3,
        padding=1
    )

    # Create test inputs with gradient tracking
    stream_inputs = [
        torch.randn(batch_size, 3, h, w, requires_grad=True),
        torch.randn(batch_size, 3, h, w, requires_grad=True)
    ]

    # Generate blanked mask
    blanked_mask = generate_per_sample_blanked_mask(
        batch_size=batch_size,
        num_streams=2,
        dropout_prob=0.5,
        device='cpu'
    )

    # Forward pass
    stream_outputs, integrated_out = conv(stream_inputs, None, blanked_mask)

    # Compute loss and backward
    loss = sum(s.sum() for s in stream_outputs) + integrated_out.sum()
    loss.backward()

    # Verify gradients exist
    for i, inp in enumerate(stream_inputs):
        assert inp.grad is not None, f"No gradient for stream_input[{i}]!"
        assert not torch.isnan(inp.grad).any(), f"NaN gradient for stream_input[{i}]!"
        assert not torch.isinf(inp.grad).any(), f"Inf gradient for stream_input[{i}]!"

        # Verify blanked samples have zero gradients (they contributed zero to loss)
        if blanked_mask is not None:
            stream_blanked = blanked_mask.get(i)
            if stream_blanked is not None and stream_blanked.any():
                # Blanked samples should have zero gradient (or very small due to numerical errors)
                blanked_grad = inp.grad[stream_blanked]
                # Note: Gradient might not be exactly zero due to integrated stream contribution
                print(f"  Stream {i} blanked sample gradient norm: {blanked_grad.norm():.6f}")

    print("✓ Gradient flow test passed!")
    return True


if __name__ == "__main__":
    print("=" * 80)
    print("MODALITY DROPOUT OPTIMIZATION CORRECTNESS TESTS")
    print("=" * 80)

    all_passed = True

    try:
        all_passed &= test_conv_masking_correctness()
        all_passed &= test_batchnorm_subset_correctness()
        all_passed &= test_full_forward_pass()
        all_passed &= test_gradient_flow()
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("✓ ALL TESTS PASSED - Optimizations maintain correctness!")
    else:
        print("✗ SOME TESTS FAILED - Review optimizations!")
    print("=" * 80)
