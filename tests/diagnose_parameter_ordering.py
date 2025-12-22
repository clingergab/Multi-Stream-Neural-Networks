"""
Diagnostic test to analyze parameter ordering differences between Original LINet and LINet3.

The user expected parameter order: stream1, stream2, ..., integrated
This test checks if both models follow this convention.
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


def test_conv_parameter_ordering():
    """Test parameter ordering in LIConv2d layers."""
    print("=" * 80)
    print("Test: LIConv2d Parameter Ordering")
    print("=" * 80)

    model_orig, model_linet3 = create_models()

    # Check conv1 parameters
    print("\n--- Original Model conv1 Parameters ---")
    orig_params = list(model_orig.conv1.named_parameters())
    for name, param in orig_params:
        print(f"  {name}: {param.shape}")

    print("\n--- LINet3 Model conv1 Parameters ---")
    li3_params = list(model_linet3.conv1.named_parameters())
    for name, param in li3_params:
        print(f"  {name}: {param.shape}")

    # Analyze ordering
    print("\n--- Parameter Ordering Analysis ---")

    orig_names = [n for n, _ in orig_params]
    li3_names = [n for n, _ in li3_params]

    print(f"\nOriginal order: {orig_names}")
    print(f"LINet3 order:   {li3_names}")

    # Check if Original follows stream1 -> stream2 -> integrated
    print("\n--- Expected Order Check ---")
    print("Expected: stream1_*, stream2_*, integrated_* (or stream_weights.0, stream_weights.1, integrated_*)")

    # Check Original
    orig_first_stream = None
    for name in orig_names:
        if 'stream1' in name:
            orig_first_stream = 'stream1'
            break
        elif 'stream2' in name:
            orig_first_stream = 'stream2'
            break
        elif 'integrated' in name:
            orig_first_stream = 'integrated'
            break
    print(f"\nOriginal model first stream in order: {orig_first_stream}")

    # Check LINet3
    li3_first_stream = None
    for name in li3_names:
        if 'stream_weights.0' in name or 'stream0' in name:
            li3_first_stream = 'stream0'
            break
        elif 'stream_weights.1' in name or 'stream1' in name:
            li3_first_stream = 'stream1'
            break
        elif 'integrated' in name:
            li3_first_stream = 'integrated'
            break
    print(f"LINet3 model first stream in order: {li3_first_stream}")

    if orig_first_stream == 'stream1' and li3_first_stream == 'integrated':
        print("\n⚠️  ORDERING MISMATCH: Original starts with stream1, LINet3 starts with integrated!")
    elif orig_first_stream == li3_first_stream:
        print("\n✅ Ordering matches!")


def test_bn_parameter_ordering():
    """Test parameter ordering in LIBatchNorm2d layers."""
    print("\n" + "=" * 80)
    print("Test: LIBatchNorm2d Parameter Ordering")
    print("=" * 80)

    model_orig, model_linet3 = create_models()

    # Check bn1 parameters
    print("\n--- Original Model bn1 Parameters ---")
    orig_params = list(model_orig.bn1.named_parameters())
    for name, param in orig_params:
        print(f"  {name}: {param.shape}")

    print("\n--- LINet3 Model bn1 Parameters ---")
    li3_params = list(model_linet3.bn1.named_parameters())
    for name, param in li3_params:
        print(f"  {name}: {param.shape}")

    # Analyze ordering
    print("\n--- Parameter Ordering Analysis ---")

    orig_names = [n for n, _ in orig_params]
    li3_names = [n for n, _ in li3_params]

    print(f"\nOriginal order: {orig_names}")
    print(f"LINet3 order:   {li3_names}")


def test_bn_buffer_ordering():
    """Test buffer ordering in LIBatchNorm2d layers."""
    print("\n" + "=" * 80)
    print("Test: LIBatchNorm2d Buffer Ordering")
    print("=" * 80)

    model_orig, model_linet3 = create_models()

    # Check bn1 buffers
    print("\n--- Original Model bn1 Buffers ---")
    orig_buffers = list(model_orig.bn1.named_buffers())
    for name, buf in orig_buffers:
        print(f"  {name}: {buf.shape if hasattr(buf, 'shape') else buf}")

    print("\n--- LINet3 Model bn1 Buffers ---")
    li3_buffers = list(model_linet3.bn1.named_buffers())
    for name, buf in li3_buffers:
        print(f"  {name}: {buf.shape if hasattr(buf, 'shape') else buf}")

    # Analyze ordering
    print("\n--- Buffer Ordering Analysis ---")

    orig_names = [n for n, _ in orig_buffers]
    li3_names = [n for n, _ in li3_buffers]

    print(f"\nOriginal order: {orig_names}")
    print(f"LINet3 order:   {li3_names}")


def test_parameter_value_correspondence():
    """Test if parameter values correspond correctly after initialization."""
    print("\n" + "=" * 80)
    print("Test: Parameter Value Correspondence")
    print("=" * 80)

    model_orig, model_linet3 = create_models()

    # Compare conv1 weights
    print("\n--- conv1 Weight Comparison ---")

    # Original: stream1_weight, stream2_weight, integrated_weight
    # LINet3: stream_weights[0], stream_weights[1], integrated_weight

    orig_s1_weight = model_orig.conv1.stream1_weight
    orig_s2_weight = model_orig.conv1.stream2_weight
    orig_int_weight = model_orig.conv1.integrated_weight

    li3_s0_weight = model_linet3.conv1.stream_weights[0]
    li3_s1_weight = model_linet3.conv1.stream_weights[1]
    li3_int_weight = model_linet3.conv1.integrated_weight

    print(f"Original stream1_weight shape: {orig_s1_weight.shape}")
    print(f"LINet3 stream_weights[0] shape: {li3_s0_weight.shape}")
    print(f"  Shapes match: {orig_s1_weight.shape == li3_s0_weight.shape}")

    if orig_s1_weight.shape == li3_s0_weight.shape:
        diff = (orig_s1_weight - li3_s0_weight).abs().max().item()
        print(f"  Max diff: {diff:.2e}")
        if diff < 1e-6:
            print(f"  ✅ Values match!")
        else:
            print(f"  ⚠️  Values differ!")

    print(f"\nOriginal stream2_weight shape: {orig_s2_weight.shape}")
    print(f"LINet3 stream_weights[1] shape: {li3_s1_weight.shape}")
    print(f"  Shapes match: {orig_s2_weight.shape == li3_s1_weight.shape}")

    if orig_s2_weight.shape == li3_s1_weight.shape:
        diff = (orig_s2_weight - li3_s1_weight).abs().max().item()
        print(f"  Max diff: {diff:.2e}")
        if diff < 1e-6:
            print(f"  ✅ Values match!")
        else:
            print(f"  ⚠️  Values differ!")

    print(f"\nOriginal integrated_weight shape: {orig_int_weight.shape}")
    print(f"LINet3 integrated_weight shape: {li3_int_weight.shape}")
    print(f"  Shapes match: {orig_int_weight.shape == li3_int_weight.shape}")

    if orig_int_weight.shape == li3_int_weight.shape:
        if orig_int_weight.numel() > 0:
            diff = (orig_int_weight - li3_int_weight).abs().max().item()
            print(f"  Max diff: {diff:.2e}")
        else:
            print(f"  (Empty tensor - first layer has no integrated input)")


def test_integration_weights():
    """Test integration weight naming and ordering."""
    print("\n" + "=" * 80)
    print("Test: Integration Weight Naming")
    print("=" * 80)

    model_orig, model_linet3 = create_models()

    # Original integration weights
    print("\n--- Original Integration Weights ---")
    print(f"integration_from_stream1: {model_orig.conv1.integration_from_stream1.shape}")
    print(f"integration_from_stream2: {model_orig.conv1.integration_from_stream2.shape}")

    # LINet3 integration weights
    print("\n--- LINet3 Integration Weights ---")
    for i, w in enumerate(model_linet3.conv1.integration_from_streams):
        print(f"integration_from_streams[{i}]: {w.shape}")


def test_state_dict_key_mapping():
    """Test state_dict key mapping between models."""
    print("\n" + "=" * 80)
    print("Test: State Dict Key Mapping")
    print("=" * 80)

    model_orig, model_linet3 = create_models()

    orig_state = model_orig.state_dict()
    li3_state = model_linet3.state_dict()

    # Find conv1 keys
    print("\n--- Original conv1 State Dict Keys ---")
    orig_conv1_keys = sorted([k for k in orig_state.keys() if k.startswith('conv1.')])
    for k in orig_conv1_keys:
        print(f"  {k}: {orig_state[k].shape}")

    print("\n--- LINet3 conv1 State Dict Keys ---")
    li3_conv1_keys = sorted([k for k in li3_state.keys() if k.startswith('conv1.')])
    for k in li3_conv1_keys:
        print(f"  {k}: {li3_state[k].shape}")

    # Build mapping
    print("\n--- Key Mapping ---")
    mapping = {
        'conv1.stream1_weight': 'conv1.stream_weights.0',
        'conv1.stream2_weight': 'conv1.stream_weights.1',
        'conv1.integrated_weight': 'conv1.integrated_weight',
        'conv1.stream1_bias': 'conv1.stream_biases.0',
        'conv1.stream2_bias': 'conv1.stream_biases.1',
        'conv1.integrated_bias': 'conv1.integrated_bias',
        'conv1.integration_from_stream1': 'conv1.integration_from_streams.0',
        'conv1.integration_from_stream2': 'conv1.integration_from_streams.1',
    }

    for orig_key, li3_key in mapping.items():
        orig_shape = orig_state.get(orig_key, {})
        li3_shape = li3_state.get(li3_key, {})
        orig_shape_str = str(orig_shape.shape) if hasattr(orig_shape, 'shape') else 'NOT FOUND'
        li3_shape_str = str(li3_shape.shape) if hasattr(li3_shape, 'shape') else 'NOT FOUND'
        match = '✅' if orig_shape_str == li3_shape_str else '❌'
        print(f"  {match} {orig_key} -> {li3_key}")
        print(f"      Original: {orig_shape_str}, LINet3: {li3_shape_str}")


def test_forward_pass_outputs():
    """Test if forward pass outputs match between models."""
    print("\n" + "=" * 80)
    print("Test: Forward Pass Output Comparison")
    print("=" * 80)

    model_orig, model_linet3 = create_models()
    model_orig.eval()
    model_linet3.eval()

    torch.manual_seed(SEED)
    rgb = torch.randn(2, 3, 32, 32)
    depth = torch.randn(2, 1, 32, 32)

    with torch.no_grad():
        out_orig = model_orig(rgb, depth)
        out_li3 = model_linet3([rgb, depth])

    diff = (out_orig - out_li3).abs().max().item()
    print(f"\nOutput difference: {diff:.6f}")

    if diff < 1e-5:
        print("✅ Outputs match!")
    else:
        print("⚠️  Outputs differ!")

        # Trace through layers to find where divergence starts
        print("\n--- Layer-by-layer Trace ---")

        # conv1
        with torch.no_grad():
            s1_orig, s2_orig, int_orig = model_orig.conv1(rgb, depth, None)
            s_li3, int_li3 = model_linet3.conv1([rgb, depth], None)

        diff_s1 = (s1_orig - s_li3[0]).abs().max().item()
        diff_s2 = (s2_orig - s_li3[1]).abs().max().item()
        diff_int = (int_orig - int_li3).abs().max().item()

        print(f"After conv1:")
        print(f"  stream1 diff: {diff_s1:.2e}")
        print(f"  stream2 diff: {diff_s2:.2e}")
        print(f"  integrated diff: {diff_int:.2e}")


if __name__ == "__main__":
    print("\n🔍 Diagnosing Parameter Ordering Differences\n")

    test_conv_parameter_ordering()
    test_bn_parameter_ordering()
    test_bn_buffer_ordering()
    test_parameter_value_correspondence()
    test_integration_weights()
    test_state_dict_key_mapping()
    test_forward_pass_outputs()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Key Findings:

1. PARAMETER NAMING CONVENTION:
   - Original: stream1_*, stream2_*, integrated_* (explicit stream naming)
   - LINet3: stream_weights[i], stream_biases[i], integrated_* (indexed ParameterList)

2. EXPECTED ORDERING (per user):
   - stream1, stream2, ..., integrated

3. ACTUAL ORDERING:
   - Original: stream1_weight, stream2_weight, integrated_weight,
               integration_from_stream1, integration_from_stream2
   - LINet3: Check if ParameterList maintains order or if integrated comes first

4. IMPLICATIONS:
   - PyTorch ParameterList is ordered, but the state_dict key format differs
   - Optimizer parameter groups will have different orderings
   - State dict loading between models would require key remapping
""")
