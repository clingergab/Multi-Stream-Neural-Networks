"""
Comprehensive tests for modality dropout implementation.

Verifies:
1. Mask generation - never both streams blanked for same sample
2. Conv masking - blanked samples have zero stream_out_raw
3. BN subset - running stats only include active samples
4. Zero propagation - zeros propagate through ReLU, MaxPool
5. Gradient test - blanked stream weights get zero gradient
6. Integration weights gradient - receive gradients from non-blanked samples
7. Validation no dropout - _validate() doesn't apply dropout
8. History tracking - modality_dropout_prob populated correctly
9. Stream monitoring isolation - blanked samples excluded from metrics
10. Single-stream eval - evaluate() with blanked_streams parameter
"""

import torch
import torch.nn as nn
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.modality_dropout import (
    get_modality_dropout_prob,
    generate_per_sample_blanked_mask
)
from src.models.linear_integration.li_net3.conv import LIConv2d, LIBatchNorm2d
from src.models.linear_integration.li_net3.container import LIReLU, LISequential
from src.models.linear_integration.li_net3.pooling import LIMaxPool2d, LIAdaptiveAvgPool2d
from src.models.linear_integration.li_net3.blocks import LIBasicBlock


def test_mask_generation():
    """Test 1: Verify mask generation never blanks both streams for same sample."""
    print("\n" + "=" * 60)
    print("TEST 1: Mask Generation")
    print("=" * 60)

    device = torch.device('cpu')
    num_tests = 100
    batch_size = 64
    num_streams = 2
    dropout_prob = 0.5  # High prob to ensure we get blanking

    violations = 0
    total_samples = 0
    total_blanked = 0

    for _ in range(num_tests):
        mask = generate_per_sample_blanked_mask(
            batch_size=batch_size,
            num_streams=num_streams,
            dropout_prob=dropout_prob,
            device=device
        )

        if mask is not None:
            # Check that no sample has both streams blanked
            both_blanked = mask[0] & mask[1]
            violations += both_blanked.sum().item()
            total_samples += batch_size
            total_blanked += mask[0].sum().item() + mask[1].sum().item()

    # Verify we actually got some blanking (sanity check that test is meaningful)
    blanking_rate = total_blanked / (total_samples * 2) if total_samples > 0 else 0
    print(f"  Tested {num_tests} batches, {total_samples} total samples")
    print(f"  Blanking rate: {blanking_rate:.2%} (expected ~{dropout_prob/2:.2%} per stream)")
    print(f"  Violations (both streams blanked): {violations}")

    # Sanity check: we should have some blanking happening
    if blanking_rate < 0.1:
        print("  ⚠️ WARNING: Very low blanking rate - test may not be meaningful")

    if violations == 0:
        print("  ✅ PASSED: Never blanks both streams for same sample")
        return True
    else:
        print("  ❌ FAILED: Found samples with both streams blanked")
        return False


def test_dropout_prob_schedule():
    """Test get_modality_dropout_prob schedule.

    New schedule (fixed): epoch at start_epoch gets (1/ramp_epochs * final_rate),
    ramping linearly to final_rate over ramp_epochs epochs.

    Example with start=10, ramp=20, final=0.2:
      - Epoch 10: 1/20 * 0.2 = 0.01 (1%)
      - Epoch 11: 2/20 * 0.2 = 0.02 (2%)
      - ...
      - Epoch 29: 20/20 * 0.2 = 0.2 (20%)
      - Epoch 30+: 0.2 (20%)
    """
    print("\n" + "=" * 60)
    print("TEST 1b: Dropout Probability Schedule")
    print("=" * 60)

    # Test with start_epoch=10, ramp_epochs=20, final_rate=0.2
    start, ramp, final = 10, 20, 0.2

    # Before start - should be 0
    prob = get_modality_dropout_prob(5, start, ramp, final)
    assert prob == 0.0, f"Expected 0.0 before start, got {prob}"
    print(f"  Epoch 5 (before start): {prob:.4f} ✓")

    # At start - should be 1/ramp_epochs * final = 0.01
    prob = get_modality_dropout_prob(10, start, ramp, final)
    expected = final / ramp  # 0.2 / 20 = 0.01
    assert abs(prob - expected) < 1e-6, f"Expected {expected}, got {prob}"
    print(f"  Epoch 10 (at start): {prob:.4f} ✓")

    # Midway through ramp - epoch 19 is 10 epochs in, so 10/20 * 0.2 = 0.1
    # Wait, with new formula: epochs_since_start=9, (9+1)/20 * 0.2 = 0.1
    prob = get_modality_dropout_prob(19, start, ramp, final)
    expected = 0.1  # (19-10+1)/20 * 0.2 = 10/20 * 0.2 = 0.1
    assert abs(prob - expected) < 1e-6, f"Expected {expected}, got {prob}"
    print(f"  Epoch 19 (mid-ramp): {prob:.4f} ✓")

    # End of ramp - epoch 29 is last ramp epoch: (29-10+1)/20 * 0.2 = 20/20 * 0.2 = 0.2
    prob = get_modality_dropout_prob(29, start, ramp, final)
    assert abs(prob - final) < 1e-6, f"Expected {final}, got {prob}"
    print(f"  Epoch 29 (end of ramp): {prob:.4f} ✓")

    # After ramp - should be final
    prob = get_modality_dropout_prob(50, start, ramp, final)
    assert prob == final, f"Expected {final}, got {prob}"
    print(f"  Epoch 50 (after ramp): {prob:.4f} ✓")

    print("  ✅ PASSED: Dropout probability schedule works correctly")


def test_conv_masking():
    """Test 2: Verify blanked samples have zero output after conv."""
    print("\n" + "=" * 60)
    print("TEST 2: Conv Masking")
    print("=" * 60)

    device = torch.device('cpu')
    batch_size = 8

    # Create LIConv2d
    conv = LIConv2d(
        stream_in_channels=[3, 3],
        stream_out_channels=[16, 16],
        integrated_in_channels=16,
        integrated_out_channels=16,
        kernel_size=3,
        padding=1
    )

    # Create inputs with KNOWN non-zero values
    stream_inputs = [
        torch.ones(batch_size, 3, 32, 32) * 2.0,  # All 2s
        torch.ones(batch_size, 3, 32, 32) * 3.0   # All 3s
    ]
    integrated_input = torch.randn(batch_size, 16, 32, 32)

    # First, verify WITHOUT masking that outputs are non-zero
    stream_outputs_no_mask, _ = conv(stream_inputs, integrated_input, None)
    s0_no_mask_nonzero = stream_outputs_no_mask[0].abs().sum() > 0
    s1_no_mask_nonzero = stream_outputs_no_mask[1].abs().sum() > 0
    print(f"  Without mask - Stream 0 has output: {s0_no_mask_nonzero.item()}")
    print(f"  Without mask - Stream 1 has output: {s1_no_mask_nonzero.item()}")

    if not (s0_no_mask_nonzero and s1_no_mask_nonzero):
        print("  ❌ FAILED: Conv produces no output even without masking!")
        return False

    # Now test WITH masking
    # Create mask: blank stream 0 for samples 0,2,4 and stream 1 for samples 1,3
    blanked_mask = {
        0: torch.tensor([True, False, True, False, True, False, False, False]),
        1: torch.tensor([False, True, False, True, False, False, False, False])
    }

    # Forward pass with mask
    stream_outputs, integrated_output = conv(stream_inputs, integrated_input, blanked_mask)

    # Check that blanked samples are zeros
    stream0_blanked_samples = stream_outputs[0][[0, 2, 4]]
    stream1_blanked_samples = stream_outputs[1][[1, 3]]

    stream0_zeros = (stream0_blanked_samples.abs().sum() == 0).item()
    stream1_zeros = (stream1_blanked_samples.abs().sum() == 0).item()

    # Check that non-blanked samples are NOT zeros
    stream0_active_samples = stream_outputs[0][[1, 3, 5, 6, 7]]
    stream1_active_samples = stream_outputs[1][[0, 2, 4, 5, 6, 7]]

    stream0_active_nonzero = (stream0_active_samples.abs().sum() > 0).item()
    stream1_active_nonzero = (stream1_active_samples.abs().sum() > 0).item()

    print(f"  With mask - Stream 0 blanked samples (0,2,4) are zeros: {stream0_zeros}")
    print(f"  With mask - Stream 1 blanked samples (1,3) are zeros: {stream1_zeros}")
    print(f"  With mask - Stream 0 active samples are non-zero: {stream0_active_nonzero}")
    print(f"  With mask - Stream 1 active samples are non-zero: {stream1_active_nonzero}")

    if stream0_zeros and stream1_zeros and stream0_active_nonzero and stream1_active_nonzero:
        print("  ✅ PASSED: Conv correctly masks blanked samples to zero")
        return True
    else:
        print("  ❌ FAILED: Conv masking not working correctly")
        return False


def test_bn_subset():
    """Test 3: Verify BN running stats only include active samples."""
    print("\n" + "=" * 60)
    print("TEST 3: BN Subset (Running Stats)")
    print("=" * 60)

    device = torch.device('cpu')
    batch_size = 8
    channels = 4  # Small for easier verification

    # Create BN layer
    bn = LIBatchNorm2d([channels, channels], channels)
    bn.train()

    # Create inputs where stream 0 active samples have VERY DIFFERENT values
    # than if we included zeros
    stream_inputs = [
        torch.ones(batch_size, channels, 4, 4) * 10.0,  # All 10s for stream 0
        torch.ones(batch_size, channels, 4, 4) * 5.0    # All 5s for stream 1
    ]

    # Blank stream 0 for samples 0,1,2,3 (half the batch)
    # If BN includes blanked samples (zeros), mean would be ~5.0
    # If BN only includes active samples (all 10s), mean would be ~10.0
    blanked_mask = {
        0: torch.tensor([True, True, True, True, False, False, False, False]),
        1: torch.tensor([False, False, False, False, False, False, False, False])
    }

    # Zero out the blanked samples (simulating what conv does)
    stream_inputs[0][[0, 1, 2, 3]] = 0

    # Forward through BN with mask
    stream_outputs, _ = bn(stream_inputs, None, blanked_mask)

    # Check 1: Blanked samples should remain zeros
    blanked_output = stream_outputs[0][[0, 1, 2, 3]]
    blanked_is_zero = (blanked_output.abs().sum() == 0).item()
    print(f"  Blanked samples remain zero after BN: {blanked_is_zero}")

    # Check 2: Running mean should reflect only active samples (should be ~10, not ~5)
    # After one batch with momentum=0.1: running_mean = 0.9*0 + 0.1*batch_mean
    # If batch_mean is computed from active samples only: batch_mean ≈ 10
    # If batch_mean includes zeros: batch_mean ≈ 5
    running_mean = bn.stream0_running_mean.mean().item()
    print(f"  Stream 0 running mean: {running_mean:.4f}")
    print(f"  (If only active samples: ~1.0, if including zeros: ~0.5)")

    # The running mean should be closer to 1.0 (10 * 0.1) than 0.5 (5 * 0.1)
    # because BN should only process active samples
    correct_mean = abs(running_mean - 1.0) < abs(running_mean - 0.5)
    print(f"  Running mean closer to expected (1.0): {correct_mean}")

    # Check 3: Active samples should be normalized (mean ~0, std ~1)
    active_output = stream_outputs[0][[4, 5, 6, 7]]
    active_mean = active_output.mean().item()
    active_std = active_output.std().item()
    print(f"  Active samples output mean: {active_mean:.4f} (should be ~0)")
    print(f"  Active samples output std: {active_std:.4f} (should be ~1)")

    # Mean should be close to 0 for normalized data
    normalized_correctly = abs(active_mean) < 0.1

    if blanked_is_zero and correct_mean and normalized_correctly:
        print("  ✅ PASSED: BN correctly handles blanked samples")
        return True
    else:
        print("  ❌ FAILED: BN not handling blanked samples correctly")
        return False


def test_zero_propagation():
    """Test 4: Verify zeros propagate through ReLU, MaxPool, AvgPool."""
    print("\n" + "=" * 60)
    print("TEST 4: Zero Propagation")
    print("=" * 60)

    batch_size = 4
    channels = 16

    # Create modules
    relu = LIReLU()
    maxpool = LIMaxPool2d(kernel_size=2, stride=2)
    avgpool = LIAdaptiveAvgPool2d((1, 1))

    # Create inputs with KNOWN non-zero values for active samples
    stream_inputs = [
        torch.ones(batch_size, channels, 8, 8) * 5.0,  # All 5s
        torch.ones(batch_size, channels, 8, 8) * 3.0   # All 3s
    ]

    # Zero out samples 0 and 2 in stream 0
    stream_inputs[0][[0, 2]] = 0

    blanked_mask = {
        0: torch.tensor([True, False, True, False]),
        1: torch.tensor([False, False, False, False])
    }

    # Verify inputs are set up correctly
    assert stream_inputs[0][[0, 2]].abs().sum() == 0, "Setup error: blanked inputs not zero"
    assert stream_inputs[0][[1, 3]].abs().sum() > 0, "Setup error: active inputs are zero"

    # Test ReLU
    relu_out, _ = relu(stream_inputs, None, blanked_mask)
    relu_blanked_zero = (relu_out[0][[0, 2]].abs().sum() == 0).item()
    relu_active_nonzero = (relu_out[0][[1, 3]].abs().sum() > 0).item()

    # Test MaxPool
    maxpool_out, _ = maxpool(stream_inputs, None, blanked_mask)
    maxpool_blanked_zero = (maxpool_out[0][[0, 2]].abs().sum() == 0).item()
    maxpool_active_nonzero = (maxpool_out[0][[1, 3]].abs().sum() > 0).item()

    # Test AvgPool
    avgpool_out, _ = avgpool(stream_inputs, None, blanked_mask)
    avgpool_blanked_zero = (avgpool_out[0][[0, 2]].abs().sum() == 0).item()
    avgpool_active_nonzero = (avgpool_out[0][[1, 3]].abs().sum() > 0).item()

    print(f"  ReLU: blanked→zero={relu_blanked_zero}, active→nonzero={relu_active_nonzero}")
    print(f"  MaxPool: blanked→zero={maxpool_blanked_zero}, active→nonzero={maxpool_active_nonzero}")
    print(f"  AvgPool: blanked→zero={avgpool_blanked_zero}, active→nonzero={avgpool_active_nonzero}")

    all_correct = all([
        relu_blanked_zero, relu_active_nonzero,
        maxpool_blanked_zero, maxpool_active_nonzero,
        avgpool_blanked_zero, avgpool_active_nonzero
    ])

    if all_correct:
        print("  ✅ PASSED: Zeros propagate correctly through all layers")
        return True
    else:
        print("  ❌ FAILED: Zero propagation broken")
        return False


def test_gradient_isolation():
    """Test 5: Verify blanked stream weights get zero gradient for blanked samples."""
    print("\n" + "=" * 60)
    print("TEST 5: Gradient Isolation")
    print("=" * 60)

    device = torch.device('cpu')
    batch_size = 4

    # Create a simple LIConv2d
    conv = LIConv2d(
        stream_in_channels=[3, 3],
        stream_out_channels=[8, 8],
        integrated_in_channels=8,
        integrated_out_channels=8,
        kernel_size=3,
        padding=1
    )

    # First, verify that WITHOUT masking, both streams get gradients
    stream_inputs_1 = [
        torch.ones(batch_size, 3, 8, 8, requires_grad=True) * 2.0,
        torch.ones(batch_size, 3, 8, 8, requires_grad=True) * 3.0
    ]
    integrated_input_1 = torch.randn(batch_size, 8, 8, 8, requires_grad=True)

    conv.zero_grad()
    stream_outputs_no_mask, _ = conv(stream_inputs_1, integrated_input_1, None)
    loss_no_mask = stream_outputs_no_mask[0].sum() + stream_outputs_no_mask[1].sum()
    loss_no_mask.backward()

    s0_grad_no_mask = conv.stream_weights[0].grad.abs().sum().item()
    s1_grad_no_mask = conv.stream_weights[1].grad.abs().sum().item()
    print(f"  Without mask - Stream 0 gradient: {s0_grad_no_mask:.4f}")
    print(f"  Without mask - Stream 1 gradient: {s1_grad_no_mask:.4f}")

    if s0_grad_no_mask == 0 or s1_grad_no_mask == 0:
        print("  ❌ FAILED: No gradients even without masking!")
        return False

    # Now test WITH masking - blank ALL samples for stream 0
    # Create NEW inputs for fresh computation graph
    stream_inputs_2 = [
        torch.ones(batch_size, 3, 8, 8, requires_grad=True) * 2.0,
        torch.ones(batch_size, 3, 8, 8, requires_grad=True) * 3.0
    ]
    integrated_input_2 = torch.randn(batch_size, 8, 8, 8, requires_grad=True)

    conv.zero_grad()
    blanked_mask = {
        0: torch.tensor([True, True, True, True]),  # All blanked
        1: torch.tensor([False, False, False, False])  # None blanked
    }

    stream_outputs, integrated_output = conv(stream_inputs_2, integrated_input_2, blanked_mask)

    # Loss depends on both stream outputs
    loss = stream_outputs[0].sum() + stream_outputs[1].sum()
    loss.backward()

    # Stream 0 weight should have zero gradient (all samples blanked, output is zeros)
    stream0_grad = conv.stream_weights[0].grad
    stream0_grad_sum = stream0_grad.abs().sum().item()

    # Stream 1 weight should have non-zero gradient
    stream1_grad = conv.stream_weights[1].grad
    stream1_grad_sum = stream1_grad.abs().sum().item()

    print(f"  With mask (stream 0 all blanked):")
    print(f"    Stream 0 weight gradient sum: {stream0_grad_sum:.6f} (should be 0)")
    print(f"    Stream 1 weight gradient sum: {stream1_grad_sum:.6f} (should be >0)")

    if stream0_grad_sum == 0 and stream1_grad_sum > 0:
        print("  ✅ PASSED: Gradients correctly isolated")
        return True
    else:
        print("  ❌ FAILED: Gradient isolation not working")
        return False


def test_integration_weights_gradient():
    """Test 6: Verify integration weights receive gradients from non-blanked samples."""
    print("\n" + "=" * 60)
    print("TEST 6: Integration Weights Gradient")
    print("=" * 60)

    device = torch.device('cpu')
    batch_size = 4

    # Create LIConv2d
    conv = LIConv2d(
        stream_in_channels=[3, 3],
        stream_out_channels=[8, 8],
        integrated_in_channels=8,
        integrated_out_channels=8,
        kernel_size=3,
        padding=1
    )

    # Create inputs
    stream_inputs = [
        torch.randn(batch_size, 3, 8, 8, requires_grad=True),
        torch.randn(batch_size, 3, 8, 8, requires_grad=True)
    ]
    integrated_input = torch.randn(batch_size, 8, 8, 8, requires_grad=True)

    # Blank stream 0 for half the samples
    blanked_mask = {
        0: torch.tensor([True, True, False, False]),  # Half blanked
        1: torch.tensor([False, False, False, False])  # None blanked
    }

    # Forward pass
    stream_outputs, integrated_output = conv(stream_inputs, integrated_input, blanked_mask)

    # Loss on integrated output
    loss = integrated_output.sum()
    loss.backward()

    # Integration weights should have gradients
    int_weight_0 = conv.integration_from_streams[0].grad
    int_weight_1 = conv.integration_from_streams[1].grad

    int_grad_0_sum = int_weight_0.abs().sum().item() if int_weight_0 is not None else 0
    int_grad_1_sum = int_weight_1.abs().sum().item() if int_weight_1 is not None else 0

    print(f"  Integration weight 0 gradient sum: {int_grad_0_sum:.6f}")
    print(f"  Integration weight 1 gradient sum: {int_grad_1_sum:.6f}")

    # Both should have gradients (stream 0 from non-blanked samples, stream 1 from all)
    # Note: stream 0 integration gets gradient from samples 2,3 (non-blanked)
    if int_grad_0_sum > 0 and int_grad_1_sum > 0:
        print("  ✅ PASSED: Integration weights receive gradients from non-blanked samples")
        return True
    else:
        print("  ❌ FAILED: Integration weights not receiving proper gradients")
        return False


def test_blanked_no_integration_contribution():
    """Test 7: Verify blanked streams don't contribute to integration."""
    print("\n" + "=" * 60)
    print("TEST 7: Blanked Streams Don't Contribute to Integration")
    print("=" * 60)

    device = torch.device('cpu')
    batch_size = 2

    # Create LIConv2d
    conv = LIConv2d(
        stream_in_channels=[3, 3],
        stream_out_channels=[8, 8],
        integrated_in_channels=8,
        integrated_out_channels=8,
        kernel_size=3,
        padding=1
    )

    # Create inputs with VERY DIFFERENT values for the two streams
    stream_inputs = [
        torch.ones(batch_size, 3, 8, 8) * 100.0,  # Stream 0: large values
        torch.ones(batch_size, 3, 8, 8) * 1.0     # Stream 1: small values
    ]
    integrated_input = torch.zeros(batch_size, 8, 8, 8)  # Start with zeros

    # Test 1: No blanking - integrated should reflect both streams
    _, integrated_no_blank = conv(stream_inputs, integrated_input, None)
    int_magnitude_no_blank = integrated_no_blank.abs().mean().item()

    # Test 2: Blank stream 0 (the large one) - integrated should be smaller
    blanked_mask = {
        0: torch.tensor([True, True]),  # Blank stream 0 (large values)
        1: torch.tensor([False, False])  # Keep stream 1 (small values)
    }
    _, integrated_with_blank = conv(stream_inputs, integrated_input, blanked_mask)
    int_magnitude_with_blank = integrated_with_blank.abs().mean().item()

    print(f"  Integrated magnitude (no blanking): {int_magnitude_no_blank:.4f}")
    print(f"  Integrated magnitude (stream 0 blanked): {int_magnitude_with_blank:.4f}")

    # When we blank the large stream, integrated output should be significantly smaller
    ratio = int_magnitude_with_blank / int_magnitude_no_blank if int_magnitude_no_blank > 0 else 0
    print(f"  Ratio (with/without): {ratio:.4f}")

    # The blanked version should be much smaller (stream 0 was 100x larger)
    if ratio < 0.5:  # At least 50% reduction
        print("  ✅ PASSED: Blanked streams don't contribute to integration")
        return True
    else:
        print("  ❌ FAILED: Blanked streams still contributing to integration")
        return False


def test_history_tracking():
    """Test 8: Verify history['modality_dropout_prob'] schedule values."""
    print("\n" + "=" * 60)
    print("TEST 8: History Tracking")
    print("=" * 60)

    # Test the schedule calculation
    start_epoch = 5
    ramp_epochs = 10
    final_rate = 0.2
    total_epochs = 20

    expected_probs = []
    for epoch in range(total_epochs):
        prob = get_modality_dropout_prob(epoch, start_epoch, ramp_epochs, final_rate)
        expected_probs.append(prob)

    print(f"  Schedule (start={start_epoch}, ramp={ramp_epochs}, final={final_rate}):")
    print(f"  Epochs 0-4: {expected_probs[0:5]}")
    print(f"  Epochs 5-9: {[f'{p:.2f}' for p in expected_probs[5:10]]}")
    print(f"  Epochs 10-14: {[f'{p:.2f}' for p in expected_probs[10:15]]}")
    print(f"  Epochs 15-19: {[f'{p:.2f}' for p in expected_probs[15:20]]}")

    # Verify schedule
    # Before start: all zeros
    before_start_ok = all(p == 0.0 for p in expected_probs[:5])
    # At start: 0
    at_start_ok = expected_probs[5] == 0.0
    # During ramp: increasing
    ramp_probs = expected_probs[5:15]
    ramp_increasing = ramp_probs == sorted(ramp_probs)
    # After ramp: final_rate
    after_ramp_ok = all(p == final_rate for p in expected_probs[15:])

    print(f"  Before start (epochs 0-4) all zero: {before_start_ok}")
    print(f"  At start (epoch 5) is zero: {at_start_ok}")
    print(f"  During ramp is increasing: {ramp_increasing}")
    print(f"  After ramp equals final_rate: {after_ramp_ok}")

    if before_start_ok and at_start_ok and ramp_increasing and after_ramp_ok:
        print("  ✅ PASSED: History tracking schedule is correct")
        return True
    else:
        print("  ❌ FAILED: History tracking schedule incorrect")
        return False


def test_residual_connection_with_blanking():
    """Test 9: Verify residual connections work correctly with blanking."""
    print("\n" + "=" * 60)
    print("TEST 9: Residual Connection with Blanking")
    print("=" * 60)

    batch_size = 4

    # Create a BasicBlock with downsample (to test both paths)
    block = LIBasicBlock(
        stream_inplanes=[64, 64],
        stream_planes=[64, 64],
        integrated_inplanes=64,
        integrated_planes=64,
    )

    # Create inputs - blanked samples should ALREADY be zeros from previous layer
    stream_inputs = [
        torch.randn(batch_size, 64, 8, 8),
        torch.randn(batch_size, 64, 8, 8)
    ]
    integrated_input = torch.randn(batch_size, 64, 8, 8)

    # Blank stream 0 for samples 0,1
    blanked_mask = {
        0: torch.tensor([True, True, False, False]),
        1: torch.tensor([False, False, False, False])
    }

    # Zero out blanked samples in input
    stream_inputs[0][[0, 1]] = 0

    # Store original input values for active samples
    s0_active_input = stream_inputs[0][[2, 3]].clone()

    # Forward pass
    stream_outputs, integrated_output = block(stream_inputs, integrated_input, blanked_mask)

    # Check 1: Blanked samples should be zeros (0 + 0 = 0 in residual)
    s0_blanked_output = stream_outputs[0][[0, 1]]
    blanked_is_zero = (s0_blanked_output.abs().sum() == 0).item()

    # Check 2: Active samples should be non-zero (input + residual != 0)
    s0_active_output = stream_outputs[0][[2, 3]]
    active_is_nonzero = (s0_active_output.abs().sum() > 0).item()

    # Check 3: Integrated output should be non-zero for all samples
    # (it receives contributions from non-blanked stream 0 samples AND all stream 1 samples)
    int_nonzero = (integrated_output.abs().sum() > 0).item()

    print(f"  Blanked samples (0,1) output is zero: {blanked_is_zero}")
    print(f"  Active samples (2,3) output is non-zero: {active_is_nonzero}")
    print(f"  Integrated output is non-zero: {int_nonzero}")

    if blanked_is_zero and active_is_nonzero and int_nonzero:
        print("  ✅ PASSED: Residual connections work correctly with blanking")
        return True
    else:
        print("  ❌ FAILED: Residual connection issue with blanking")
        return False


def test_single_stream_eval():
    """Test 10: Verify full batch blanking works (for single-stream eval)."""
    print("\n" + "=" * 60)
    print("TEST 10: Single-Stream Evaluation (Full Batch Blanking)")
    print("=" * 60)

    device = torch.device('cpu')
    batch_size = 4

    # Create a conv layer
    conv = LIConv2d(
        stream_in_channels=[3, 3],
        stream_out_channels=[16, 16],
        integrated_in_channels=16,
        integrated_out_channels=16,
        kernel_size=3,
        padding=1
    )

    # Create inputs with different values for each stream
    stream_inputs = [
        torch.ones(batch_size, 3, 8, 8) * 10.0,  # Stream 0
        torch.ones(batch_size, 3, 8, 8) * 5.0    # Stream 1
    ]
    integrated_input = torch.zeros(batch_size, 16, 8, 8)

    # Simulate single-stream eval: blank stream 0 for ALL samples
    blanked_mask = {
        0: torch.ones(batch_size, dtype=torch.bool),   # All blanked
        1: torch.zeros(batch_size, dtype=torch.bool)   # None blanked
    }

    stream_outputs, integrated_output = conv(stream_inputs, integrated_input, blanked_mask)

    # Stream 0 should be ALL zeros
    s0_all_zero = (stream_outputs[0].abs().sum() == 0).item()
    # Stream 1 should be ALL non-zero
    s1_all_nonzero = (stream_outputs[1].abs().sum() > 0).item()
    # Integrated should still work (from stream 1 only)
    int_nonzero = (integrated_output.abs().sum() > 0).item()

    print(f"  Stream 0 (all blanked) is all zeros: {s0_all_zero}")
    print(f"  Stream 1 (none blanked) is non-zero: {s1_all_nonzero}")
    print(f"  Integrated output is non-zero: {int_nonzero}")

    if s0_all_zero and s1_all_nonzero and int_nonzero:
        print("  ✅ PASSED: Full batch blanking works for single-stream eval")
        return True
    else:
        print("  ❌ FAILED: Full batch blanking not working")
        return False


def test_full_forward_pass():
    """Test 11: Full forward pass through LIBasicBlock with masking."""
    print("\n" + "=" * 60)
    print("TEST 11: Full Forward Pass (LIBasicBlock)")
    print("=" * 60)

    batch_size = 4

    # Create a BasicBlock
    block = LIBasicBlock(
        stream_inplanes=[64, 64],
        stream_planes=[64, 64],
        integrated_inplanes=64,
        integrated_planes=64,
    )

    # Create inputs - blanked samples should ALREADY be zeros
    # (simulating what previous layers would output)
    stream_inputs = [
        torch.randn(batch_size, 64, 8, 8),
        torch.randn(batch_size, 64, 8, 8)
    ]
    integrated_input = torch.randn(batch_size, 64, 8, 8)

    # Blank stream 0 for samples 0,1 and stream 1 for sample 2
    blanked_mask = {
        0: torch.tensor([True, True, False, False]),
        1: torch.tensor([False, False, True, False])
    }

    # Zero out the blanked samples in input (simulating previous layer output)
    stream_inputs[0][[0, 1]] = 0
    stream_inputs[1][[2]] = 0

    # Forward pass
    stream_outputs, integrated_output = block(stream_inputs, integrated_input, blanked_mask)

    # Check outputs
    # Stream 0, samples 0,1 should be zeros
    s0_blanked_zeros = (stream_outputs[0][[0, 1]].abs().sum() == 0).item()
    # Stream 1, sample 2 should be zeros
    s1_blanked_zeros = (stream_outputs[1][[2]].abs().sum() == 0).item()
    # Other samples should be non-zero
    s0_active_nonzero = (stream_outputs[0][[2, 3]].abs().sum() > 0).item()
    s1_active_nonzero = (stream_outputs[1][[0, 1, 3]].abs().sum() > 0).item()
    # Integrated should be non-zero for all samples
    int_nonzero = (integrated_output.abs().sum() > 0).item()

    print(f"  Stream 0 blanked (0,1) are zeros: {s0_blanked_zeros}")
    print(f"  Stream 1 blanked (2) is zeros: {s1_blanked_zeros}")
    print(f"  Stream 0 active (2,3) are non-zero: {s0_active_nonzero}")
    print(f"  Stream 1 active (0,1,3) are non-zero: {s1_active_nonzero}")
    print(f"  Integrated output is non-zero: {int_nonzero}")

    if all([s0_blanked_zeros, s1_blanked_zeros, s0_active_nonzero, s1_active_nonzero, int_nonzero]):
        print("  ✅ PASSED: Full forward pass handles masking correctly")
        return True
    else:
        print("  ❌ FAILED: Full forward pass masking issue")
        return False


def test_mask_not_applied_when_none():
    """Test 12: Verify no masking happens when blanked_mask is None."""
    print("\n" + "=" * 60)
    print("TEST 12: No Masking When blanked_mask=None")
    print("=" * 60)

    batch_size = 4

    conv = LIConv2d(
        stream_in_channels=[3, 3],
        stream_out_channels=[16, 16],
        integrated_in_channels=16,
        integrated_out_channels=16,
        kernel_size=3,
        padding=1
    )

    # Create identical inputs
    stream_inputs = [
        torch.randn(batch_size, 3, 8, 8),
        torch.randn(batch_size, 3, 8, 8)
    ]
    integrated_input = torch.randn(batch_size, 16, 8, 8)

    # Run with blanked_mask=None
    stream_outputs, integrated_output = conv(stream_inputs, integrated_input, None)

    # ALL outputs should be non-zero
    s0_nonzero = (stream_outputs[0].abs().sum() > 0).item()
    s1_nonzero = (stream_outputs[1].abs().sum() > 0).item()
    int_nonzero = (integrated_output.abs().sum() > 0).item()

    print(f"  Stream 0 output non-zero: {s0_nonzero}")
    print(f"  Stream 1 output non-zero: {s1_nonzero}")
    print(f"  Integrated output non-zero: {int_nonzero}")

    if s0_nonzero and s1_nonzero and int_nonzero:
        print("  ✅ PASSED: No masking when blanked_mask=None")
        return True
    else:
        print("  ❌ FAILED: Something zeroed when it shouldn't")
        return False


def test_validate_no_modality_dropout():
    """Test 13: Verify _validate() doesn't apply modality dropout by default."""
    print("\n" + "=" * 60)
    print("TEST 13: Validation No Modality Dropout")
    print("=" * 60)

    # Check that _validate doesn't have modality_dropout_prob parameter
    # and that it only applies blanking when blanked_streams is explicitly passed
    from src.models.linear_integration.li_net3.li_net import LINet
    import inspect

    # Check _validate signature
    sig = inspect.signature(LINet._validate)
    params = list(sig.parameters.keys())

    has_dropout_prob = 'modality_dropout_prob' in params
    has_blanked_streams = 'blanked_streams' in params

    print(f"  _validate parameters: {params}")
    print(f"  Has modality_dropout_prob: {has_dropout_prob} (should be False)")
    print(f"  Has blanked_streams: {has_blanked_streams} (should be True, for explicit single-stream eval)")

    # Check implementation - blanked_mask should only be created when blanked_streams is passed
    source = inspect.getsource(LINet._validate)

    # The key check: blanked_mask is only created when blanked_streams is truthy
    creates_mask_conditionally = 'if blanked_streams:' in source or 'if blanked_streams' in source
    print(f"  Creates blanked_mask only when blanked_streams passed: {creates_mask_conditionally}")

    if not has_dropout_prob and has_blanked_streams and creates_mask_conditionally:
        print("  ✅ PASSED: _validate() doesn't apply modality dropout by default")
        return True
    else:
        print("  ❌ FAILED: _validate() may incorrectly apply dropout")
        return False


def test_stream_monitoring_with_blanking():
    """Test 14: Verify stream monitoring excludes blanked samples from metrics."""
    print("\n" + "=" * 60)
    print("TEST 14: Stream Monitoring Isolation")
    print("=" * 60)

    from src.models.linear_integration.li_net3.li_net import LINet
    import inspect

    # Check _train_epoch for proper handling of blanked samples in stream monitoring
    source = inspect.getsource(LINet._train_epoch)

    # Key patterns that must exist for proper isolation:
    checks = {
        'stream_blanked variable': 'stream_blanked' in source,
        'active_idx for non-blanked': 'active_idx' in source,
        'stream_train_active counter': 'stream_train_active' in source,
        'conditional aux loss': 'aux_loss' in source and 'active_idx' in source,
    }

    print("  Checking _train_epoch for stream monitoring isolation:")
    all_passed = True
    for name, found in checks.items():
        status = "✓" if found else "✗"
        print(f"    {status} {name}: {found}")
        if not found:
            all_passed = False

    # Also verify the logic handles the case where some samples are blanked
    handles_partial_blanking = '(~stream_blanked)' in source or 'stream_blanked.any()' in source

    print(f"    {'✓' if handles_partial_blanking else '✗'} Handles partial blanking: {handles_partial_blanking}")

    if all_passed and handles_partial_blanking:
        print("  ✅ PASSED: Stream monitoring correctly isolates blanked samples")
        return True
    else:
        print("  ❌ FAILED: Stream monitoring may include blanked samples")
        return False


def test_full_model_with_modality_dropout():
    """Test 15: Integration test - create model and verify modality dropout parameters exist."""
    print("\n" + "=" * 60)
    print("TEST 15: Full Model Integration")
    print("=" * 60)

    from src.models.linear_integration.li_net3.li_net import LINet
    import inspect

    # Check that fit() has all modality dropout parameters
    sig = inspect.signature(LINet.fit)
    params = list(sig.parameters.keys())

    required_params = [
        'modality_dropout',
        'modality_dropout_start',
        'modality_dropout_ramp',
        'modality_dropout_rate',
    ]

    print("  Checking fit() has modality dropout parameters:")
    all_present = True
    for param in required_params:
        present = param in params
        status = "✓" if present else "✗"
        print(f"    {status} {param}: {present}")
        if not present:
            all_present = False

    # Check that forward() accepts blanked_mask
    forward_sig = inspect.signature(LINet.forward)
    forward_params = list(forward_sig.parameters.keys())
    has_blanked_mask = 'blanked_mask' in forward_params

    print(f"  forward() accepts blanked_mask: {has_blanked_mask}")

    # Check _train_epoch calls generate_per_sample_blanked_mask
    train_source = inspect.getsource(LINet._train_epoch)
    calls_mask_gen = 'generate_per_sample_blanked_mask' in train_source

    print(f"  _train_epoch calls generate_per_sample_blanked_mask: {calls_mask_gen}")

    if all_present and has_blanked_mask and calls_mask_gen:
        print("  ✅ PASSED: Full model integration complete")
        return True
    else:
        print("  ❌ FAILED: Model missing modality dropout integration")
        return False


def test_evaluate_with_blanked_streams():
    """Test 16: Verify evaluate() properly handles blanked_streams parameter."""
    print("\n" + "=" * 60)
    print("TEST 16: Evaluate with blanked_streams")
    print("=" * 60)

    from src.models.linear_integration.li_net3.li_net import LINet
    import inspect

    # Check evaluate() signature
    sig = inspect.signature(LINet.evaluate)
    params = list(sig.parameters.keys())

    has_blanked_streams = 'blanked_streams' in params
    print(f"  evaluate() has blanked_streams parameter: {has_blanked_streams}")

    # Check that evaluate passes blanked_streams to _validate
    eval_source = inspect.getsource(LINet.evaluate)
    passes_to_validate = 'blanked_streams=blanked_streams' in eval_source

    print(f"  evaluate() passes blanked_streams to _validate: {passes_to_validate}")

    # Check that _validate converts blanked_streams to blanked_mask
    validate_source = inspect.getsource(LINet._validate)

    # Key patterns for proper conversion
    creates_mask = 'blanked_mask = {' in validate_source or 'blanked_mask = {\n' in validate_source
    uses_ones_for_blanked = 'torch.ones' in validate_source and 'blanked_streams' in validate_source
    uses_zeros_for_active = 'torch.zeros' in validate_source

    print(f"  _validate creates blanked_mask dict: {creates_mask}")
    print(f"  Uses torch.ones for blanked streams: {uses_ones_for_blanked}")
    print(f"  Uses torch.zeros for active streams: {uses_zeros_for_active}")

    # Check that forward is called with blanked_mask
    calls_forward_with_mask = 'blanked_mask=blanked_mask' in validate_source

    print(f"  Calls forward with blanked_mask: {calls_forward_with_mask}")

    if all([has_blanked_streams, passes_to_validate, creates_mask, calls_forward_with_mask]):
        print("  ✅ PASSED: evaluate() properly handles blanked_streams")
        return True
    else:
        print("  ❌ FAILED: evaluate() blanked_streams handling incomplete")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("MODALITY DROPOUT VERIFICATION TESTS")
    print("=" * 60)

    tests = [
        ("1. Mask Generation", test_mask_generation),
        ("1b. Dropout Prob Schedule", test_dropout_prob_schedule),
        ("2. Conv Masking", test_conv_masking),
        ("3. BN Subset", test_bn_subset),
        ("4. Zero Propagation", test_zero_propagation),
        ("5. Gradient Isolation", test_gradient_isolation),
        ("6. Integration Weights Gradient", test_integration_weights_gradient),
        ("7. Blanked No Integration", test_blanked_no_integration_contribution),
        ("8. History Tracking", test_history_tracking),
        ("9. Residual with Blanking", test_residual_connection_with_blanking),
        ("10. Single-Stream Conv", test_single_stream_eval),
        ("11. Full Forward Pass", test_full_forward_pass),
        ("12. No Masking When None", test_mask_not_applied_when_none),
        ("13. Validate No Dropout", test_validate_no_modality_dropout),
        ("14. Stream Monitoring Isolation", test_stream_monitoring_with_blanking),
        ("15. Full Model Integration", test_full_model_with_modality_dropout),
        ("16. Evaluate blanked_streams", test_evaluate_with_blanked_streams),
    ]

    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"\n  ❌ EXCEPTION in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✅" if result else "❌"
        print(f"  {status} {name}")

    print(f"\n  Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n  🎉 ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n  ⚠️ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
