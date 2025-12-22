"""
Diagnostic script to identify instability issues in LINet models.

This script can diagnose LINet3 (N-stream), original 2-stream, and Soma variants.

Usage:
    python tests/diagnose_soma_instability.py              # Diagnose li_net3 vs original (default)
    python tests/diagnose_soma_instability.py li_net3      # Diagnose LINet3 vs original
    python tests/diagnose_soma_instability.py original     # Diagnose original 2-stream vs LINet3
    python tests/diagnose_soma_instability.py soma         # Diagnose Soma vs LINet3

Checks:
1. Forward pass activation statistics (mean, std, min, max)
2. Gradient magnitudes
3. Weight statistics
4. Comparison between stream outputs vs integrated outputs
5. GroupNorm vs BatchNorm behavior
6. Scale propagation through network
"""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict

# Import all models for comparison
from src.models.linear_integration.li_net3.li_net import LINet as LINet3
from src.models.linear_integration.li_net3.blocks import LIBasicBlock as LIBasicBlock3

from src.models.linear_integration.li_net3_soma.li_net_soma import LINet as LINetSoma
from src.models.linear_integration.li_net3_soma.blocks import LIBasicBlock as LIBasicBlockSoma

from src.models.linear_integration.li_net import LINet as LINetOriginal
from src.models.linear_integration.blocks import LIBasicBlock as LIBasicBlockOriginal

# Parse command line argument for model selection
DIAGNOSE_MODEL = "li_net3"   # Default - diagnose li_net3
COMPARE_MODEL = "original"   # Compare against original 2-stream
if len(sys.argv) > 1:
    arg = sys.argv[1].lower()
    if arg in ["li_net3", "linet3"]:
        DIAGNOSE_MODEL = "li_net3"
        COMPARE_MODEL = "original"
    elif arg in ["original", "2stream", "2-stream"]:
        DIAGNOSE_MODEL = "original"
        COMPARE_MODEL = "li_net3"
    elif arg in ["soma"]:
        DIAGNOSE_MODEL = "soma"
        COMPARE_MODEL = "li_net3"
    else:
        print(f"Unknown model: {sys.argv[1]}. Use 'li_net3', 'original', or 'soma'")
        sys.exit(1)

# ============================================================================
# Configuration
# ============================================================================
DEVICE = 'cpu'
NUM_CLASSES = 10
IMAGE_SIZE = 32
BATCH_SIZE = 8
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# ============================================================================
# Helper functions
# ============================================================================

def get_model_and_block(model_type=None):
    """Get the model class and block class based on model type."""
    if model_type is None:
        model_type = DIAGNOSE_MODEL

    if model_type == "soma":
        return LINetSoma, LIBasicBlockSoma, "LINet Soma (GroupNorm on integrated)"
    elif model_type == "original":
        return LINetOriginal, LIBasicBlockOriginal, "Original 2-stream (integrates biased)"
    else:  # li_net3
        return LINet3, LIBasicBlock3, "LINet3 (BatchNorm, integrates raw)"


def get_activation_stats(tensor, name=""):
    """Get statistics for a tensor."""
    if tensor is None:
        return {"name": name, "status": "None"}

    with torch.no_grad():
        return {
            "name": name,
            "shape": list(tensor.shape),
            "mean": tensor.mean().item(),
            "std": tensor.std().item(),
            "min": tensor.min().item(),
            "max": tensor.max().item(),
            "abs_mean": tensor.abs().mean().item(),
            "has_nan": torch.isnan(tensor).any().item(),
            "has_inf": torch.isinf(tensor).any().item(),
        }


def print_stats(stats, indent=0):
    """Pretty print statistics."""
    prefix = "  " * indent
    if stats.get("status") == "None":
        print(f"{prefix}{stats['name']}: None")
        return

    print(f"{prefix}{stats['name']}:")
    print(f"{prefix}  shape: {stats['shape']}")
    print(f"{prefix}  mean: {stats['mean']:.6f}, std: {stats['std']:.6f}")
    print(f"{prefix}  min: {stats['min']:.6f}, max: {stats['max']:.6f}")
    print(f"{prefix}  abs_mean: {stats['abs_mean']:.6f}")
    if stats['has_nan'] or stats['has_inf']:
        print(f"{prefix}  ⚠️  has_nan: {stats['has_nan']}, has_inf: {stats['has_inf']}")


class ActivationHook:
    """Hook to capture activations during forward pass."""

    def __init__(self):
        self.activations = {}
        self.handles = []

    def register(self, model, layer_names=None):
        """Register hooks on named modules."""
        for name, module in model.named_modules():
            if layer_names is None or any(ln in name for ln in layer_names):
                handle = module.register_forward_hook(self._make_hook(name))
                self.handles.append(handle)

    def _make_hook(self, name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                # Handle multi-output modules (LI layers)
                for i, out in enumerate(output):
                    if isinstance(out, list):
                        for j, o in enumerate(out):
                            if o is not None:
                                self.activations[f"{name}_out{i}_stream{j}"] = o.detach()
                    elif out is not None:
                        self.activations[f"{name}_out{i}"] = out.detach()
            elif output is not None:
                self.activations[name] = output.detach()
        return hook

    def clear(self):
        self.activations = {}

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def create_dummy_input(batch_size, stream_channels, image_size, device='cpu'):
    """Create dummy input for the model."""
    return [torch.randn(batch_size, ch, image_size, image_size, device=device) for ch in stream_channels]


# ============================================================================
# Helper to create model with appropriate API
# ============================================================================
def create_model(model_type, seed=SEED):
    """Create a model of the given type with appropriate parameters."""
    torch.manual_seed(seed)

    if model_type == "original":
        model = LINetOriginal(
            block=LIBasicBlockOriginal,
            layers=[2, 2, 2, 2],
            num_classes=NUM_CLASSES,
            stream1_input_channels=3,
            stream2_input_channels=1,
        )
    elif model_type == "soma":
        model = LINetSoma(
            block=LIBasicBlockSoma,
            layers=[2, 2, 2, 2],
            num_classes=NUM_CLASSES,
            stream_input_channels=[3, 1],
        )
    else:  # li_net3
        model = LINet3(
            block=LIBasicBlock3,
            layers=[2, 2, 2, 2],
            num_classes=NUM_CLASSES,
            stream_input_channels=[3, 1],
        )

    return model.to('cpu')


def create_model_input(model_type, seed=SEED):
    """Create input in the format expected by the model type."""
    torch.manual_seed(seed)
    if model_type == "original":
        # Original expects two separate tensors
        stream1 = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE, device='cpu')
        stream2 = torch.randn(BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE, device='cpu')
        return (stream1, stream2)
    else:
        # N-stream models (li_net3, soma) expect a list
        return create_dummy_input(BATCH_SIZE, [3, 1], IMAGE_SIZE, device='cpu')


def forward_model(model, inputs, model_type):
    """Forward pass with correct input format for model type."""
    if model_type == "original":
        return model(inputs[0], inputs[1])
    else:
        return model(inputs)


# ============================================================================
# Test 1: Compare Forward Pass Activations (Both Models)
# ============================================================================
def test_forward_activations():
    print("\n" + "=" * 80)
    print(f"TEST 1: Forward Pass Activation Statistics ({DIAGNOSE_MODEL.upper()} vs {COMPARE_MODEL.upper()})")
    print("=" * 80)

    # Create both models
    model1 = create_model(DIAGNOSE_MODEL)
    model1.eval()

    model2 = create_model(COMPARE_MODEL)
    model2.eval()

    # Create inputs for both
    inputs1 = create_model_input(DIAGNOSE_MODEL)
    inputs2 = create_model_input(COMPARE_MODEL)

    # Register hooks
    hook1 = ActivationHook()
    hook2 = ActivationHook()

    # Register on key layers
    key_layers = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']
    hook1.register(model1, key_layers)
    hook2.register(model2, key_layers)

    # Forward pass
    with torch.no_grad():
        out1 = forward_model(model1, inputs1, DIAGNOSE_MODEL)
        out2 = forward_model(model2, inputs2, COMPARE_MODEL)

    # Get model names
    _, _, name1 = get_model_and_block(DIAGNOSE_MODEL)
    _, _, name2 = get_model_and_block(COMPARE_MODEL)

    # Compare outputs
    print("\n--- Model Outputs ---")
    print(f"{DIAGNOSE_MODEL}: mean={out1.mean():.6f}, std={out1.std():.6f}")
    print(f"{COMPARE_MODEL}: mean={out2.mean():.6f}, std={out2.std():.6f}")

    output_scale_diff = abs(out1.std().item() - out2.std().item()) / max(out1.std().item(), 0.001)
    if output_scale_diff > 0.1:
        print(f"\n⚠️  Output scale difference: {output_scale_diff*100:.1f}%")
    else:
        print(f"\n✅  Output scales are similar (diff: {output_scale_diff*100:.1f}%)")

    # Compare key activations
    print("\n--- Key Layer Activations ---")

    # Find matching keys
    keys1 = set(hook1.activations.keys())
    keys2 = set(hook2.activations.keys())

    print(f"\n{DIAGNOSE_MODEL} layers captured: {len(keys1)}")
    print(f"{COMPARE_MODEL} layers captured: {len(keys2)}")

    # Compare common layers
    common_keys = sorted(keys1 & keys2)
    print(f"\nComparing {len(common_keys)} common layers:")

    diff_count = 0
    for key in common_keys:
        act1 = hook1.activations[key]
        act2 = hook2.activations[key]

        stats1 = get_activation_stats(act1, f"{DIAGNOSE_MODEL}.{key}")
        stats2 = get_activation_stats(act2, f"{COMPARE_MODEL}.{key}")

        scale_diff = abs(stats1['std'] - stats2['std']) / max(stats1['std'], 0.001)

        if scale_diff > 0.5:  # More than 50% difference
            diff_count += 1
            if diff_count <= 10:  # Show first 10
                print(f"\n⚠️  SCALE DIFFERENCE in {key}:")
                print(f"   {DIAGNOSE_MODEL}: std={stats1['std']:.6f}")
                print(f"   {COMPARE_MODEL}: std={stats2['std']:.6f}")
                print(f"   Diff:   {scale_diff*100:.1f}%")

    if diff_count > 10:
        print(f"\n... and {diff_count - 10} more layers with >50% scale difference")

    if diff_count == 0:
        print("\n✅  No significant scale differences detected in common layers")

    hook1.remove()
    hook2.remove()


# ============================================================================
# Test 2: Gradient Analysis
# ============================================================================
def test_gradient_flow():
    print("\n" + "=" * 80)
    print(f"TEST 2: Gradient Flow Analysis ({DIAGNOSE_MODEL.upper()})")
    print("=" * 80)

    _, _, model_name = get_model_and_block()

    # Create model using helper
    model = create_model(DIAGNOSE_MODEL)
    model.train()

    # Create input and target
    inputs = create_model_input(DIAGNOSE_MODEL)
    torch.manual_seed(SEED)
    targets = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))

    # Forward + backward
    criterion = nn.CrossEntropyLoss()
    outputs = forward_model(model, inputs, DIAGNOSE_MODEL)
    loss = criterion(outputs, targets)
    loss.backward()

    print(f"\nModel: {model_name}")
    print(f"Loss: {loss.item():.6f}")

    # Analyze gradients by component
    grad_stats = defaultdict(list)

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad

            # Categorize by component
            if 'stream_weights' in name or 'stream_biases' in name:
                category = 'stream_bn'
            elif 'integration' in name:
                category = 'integration'
            elif 'integrated' in name:
                category = 'integrated'
            elif 'stream' in name:
                category = 'stream_conv'
            elif 'groupnorm' in name:
                category = 'groupnorm'
            else:
                category = 'other'

            grad_stats[category].append({
                'name': name,
                'mean': grad.abs().mean().item(),
                'max': grad.abs().max().item(),
                'has_nan': torch.isnan(grad).any().item(),
                'has_inf': torch.isinf(grad).any().item(),
            })

    print("\n--- Gradient Statistics by Component ---")
    for category, stats_list in sorted(grad_stats.items()):
        means = [s['mean'] for s in stats_list]
        maxes = [s['max'] for s in stats_list]
        has_issues = any(s['has_nan'] or s['has_inf'] for s in stats_list)

        print(f"\n{category} ({len(stats_list)} params):")
        print(f"  grad mean: {np.mean(means):.2e} (range: {np.min(means):.2e} - {np.max(means):.2e})")
        print(f"  grad max:  {np.mean(maxes):.2e} (range: {np.min(maxes):.2e} - {np.max(maxes):.2e})")
        if has_issues:
            print(f"  ⚠️  Contains NaN or Inf gradients!")

    # Check for gradient vanishing/explosion
    print("\n--- Potential Issues ---")

    all_means = []
    for stats_list in grad_stats.values():
        all_means.extend([s['mean'] for s in stats_list])

    if min(all_means) < 1e-10:
        print("⚠️  Very small gradients detected (potential vanishing)")
    else:
        print("✅  No vanishing gradients detected")
    if max(all_means) > 1e3:
        print("⚠️  Very large gradients detected (potential explosion)")
    else:
        print("✅  No exploding gradients detected")


# ============================================================================
# Test 3: Integration Signal Analysis
# ============================================================================
def test_integration_signals():
    print("\n" + "=" * 80)
    print(f"TEST 3: Integration Signal Analysis ({DIAGNOSE_MODEL.upper()})")
    print("=" * 80)

    _, _, model_name = get_model_and_block()

    # Create model using helper
    model = create_model(DIAGNOSE_MODEL)
    model.eval()

    # Create input
    inputs = create_model_input(DIAGNOSE_MODEL)

    # Capture intermediate values from first conv layer
    conv1 = model.conv1

    print(f"\nModel: {model_name}")

    with torch.no_grad():
        print("\n--- First Conv Layer (conv1) ---")

        # Handle different model types
        if DIAGNOSE_MODEL == "original":
            # Original 2-stream model has different structure
            stream_weights_list = [conv1.stream1_weight, conv1.stream2_weight]
            stream_biases_list = [
                conv1.stream1_bias if hasattr(conv1, 'stream1_bias') and conv1.stream1_bias is not None else None,
                conv1.stream2_bias if hasattr(conv1, 'stream2_bias') and conv1.stream2_bias is not None else None
            ]
            stream_inputs = list(inputs)  # tuple to list
        else:
            # N-stream models (li_net3, soma)
            stream_weights_list = list(conv1.stream_weights)
            stream_biases_list = list(conv1.stream_biases) if conv1.stream_biases is not None else [None] * conv1.num_streams
            stream_inputs = inputs

        # Process streams manually to see raw vs biased
        stream_outputs = []
        stream_outputs_raw = []

        for i, (stream_input, stream_weight, stream_bias) in enumerate(
            zip(stream_inputs, stream_weights_list, stream_biases_list)
        ):
            # Raw output (no bias)
            stream_out_raw = torch.nn.functional.conv2d(
                stream_input, stream_weight, None,
                conv1.stride, conv1.padding, conv1.dilation, conv1.groups
            )

            # Biased output
            if stream_bias is not None:
                stream_out = stream_out_raw + stream_bias.view(1, -1, 1, 1)
            else:
                stream_out = stream_out_raw

            stream_outputs_raw.append(stream_out_raw)
            stream_outputs.append(stream_out)

            print(f"\nStream {i}:")
            print_stats(get_activation_stats(stream_out_raw, "raw (no bias)"), indent=1)
            print_stats(get_activation_stats(stream_out, "biased"), indent=1)

            if stream_bias is not None:
                bias_stats = get_activation_stats(stream_bias, "bias values")
                print(f"    bias mean: {bias_stats['mean']:.6f}, std: {bias_stats['std']:.6f}")

        # Check what gets integrated
        print("\n--- Integration Inputs ---")
        if DIAGNOSE_MODEL == "original":
            print("Note: Original 2-stream integrates BIASED outputs")
            for i, biased in enumerate(stream_outputs):
                print(f"\nIntegrated from stream {i}:")
                print_stats(get_activation_stats(biased, "biased signal"), indent=1)
        else:
            print("Note: LINet3 and Soma integrate RAW (no bias) outputs")
            for i, raw in enumerate(stream_outputs_raw):
                print(f"\nIntegrated from stream {i}:")
                print_stats(get_activation_stats(raw, "raw signal"), indent=1)


# ============================================================================
# Test 4: Normalization Comparison
# ============================================================================
def test_norm_comparison():
    print("\n" + "=" * 80)
    print("TEST 4: GroupNorm(1) vs BatchNorm Behavior")
    print("=" * 80)

    # Create test input
    torch.manual_seed(SEED)
    test_input = torch.randn(BATCH_SIZE, 64, 8, 8)  # Typical feature map

    # Create both norm layers
    bn = nn.BatchNorm2d(64)
    gn = nn.GroupNorm(1, 64)  # GroupNorm(1) = LayerNorm over channels

    # Eval mode comparison
    bn.eval()
    gn.eval()

    with torch.no_grad():
        bn_out = bn(test_input)
        gn_out = gn(test_input)

    print("\n--- Eval Mode Outputs ---")
    print_stats(get_activation_stats(bn_out, "BatchNorm output"))
    print_stats(get_activation_stats(gn_out, "GroupNorm(1) output"))

    # Training mode comparison
    bn.train()
    gn.train()

    bn_out_train = bn(test_input)
    gn_out_train = gn(test_input)

    print("\n--- Training Mode Outputs ---")
    print_stats(get_activation_stats(bn_out_train, "BatchNorm output"))
    print_stats(get_activation_stats(gn_out_train, "GroupNorm(1) output"))

    # Per-sample variance in GroupNorm
    print("\n--- Per-Sample Variance (GroupNorm characteristic) ---")
    for i in range(min(3, BATCH_SIZE)):
        sample_std = gn_out_train[i].std().item()
        print(f"  Sample {i}: std = {sample_std:.6f}")

    print("\nNote: GroupNorm normalizes per-sample, so each sample has similar stats.")
    print("BatchNorm normalizes across batch, so individual samples may vary more.")


# ============================================================================
# Test 5: Training Step Simulation
# ============================================================================
def test_training_step():
    print("\n" + "=" * 80)
    print(f"TEST 5: Training Step Simulation ({DIAGNOSE_MODEL.upper()})")
    print("=" * 80)

    _, _, model_name = get_model_and_block()

    # Create model using helper
    model = create_model(DIAGNOSE_MODEL)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Track loss over several steps
    losses = []
    grad_norms = []

    print(f"\nModel: {model_name}")
    print("\n--- Training Steps ---")
    for step in range(10):
        inputs = create_model_input(DIAGNOSE_MODEL, seed=SEED + step)
        torch.manual_seed(SEED + step)
        targets = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))

        optimizer.zero_grad()
        outputs = forward_model(model, inputs, DIAGNOSE_MODEL)
        loss = criterion(outputs, targets)
        loss.backward()

        # Compute gradient norm
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        optimizer.step()

        losses.append(loss.item())
        grad_norms.append(total_norm)

        print(f"Step {step}: loss={loss.item():.4f}, grad_norm={total_norm:.4f}")

    print("\n--- Stability Analysis ---")
    loss_std = np.std(losses)
    grad_std = np.std(grad_norms)

    print(f"Loss: mean={np.mean(losses):.4f}, std={loss_std:.4f}")
    print(f"Grad norm: mean={np.mean(grad_norms):.4f}, std={grad_std:.4f}")

    if loss_std > np.mean(losses) * 0.5:
        print("⚠️  High loss variance - training may be unstable")
    else:
        print("✅  Loss variance looks reasonable")

    if grad_std > np.mean(grad_norms) * 0.5:
        print("⚠️  High gradient variance - training may be unstable")
    else:
        print("✅  Gradient variance looks reasonable")

    if max(grad_norms) > 100:
        print("⚠️  Very large gradients detected - consider gradient clipping")
    else:
        print("✅  Gradient magnitudes look reasonable")


# ============================================================================
# Test 6: Scale Analysis Through Network
# ============================================================================
def test_scale_propagation():
    print("\n" + "=" * 80)
    print(f"TEST 6: Signal Scale Propagation Through Network ({DIAGNOSE_MODEL.upper()})")
    print("=" * 80)

    _, _, model_name = get_model_and_block()

    # Create model using helper
    model = create_model(DIAGNOSE_MODEL)
    model.eval()

    # Hook to capture outputs at each major stage
    activations = {}

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                if DIAGNOSE_MODEL == "original":
                    # Original returns (stream1, stream2, integrated)
                    stream1, stream2, integrated = output
                    activations[f"{name}_stream0"] = stream1.detach()
                    activations[f"{name}_stream1"] = stream2.detach()
                    if integrated is not None:
                        activations[f"{name}_integrated"] = integrated.detach()
                else:
                    # N-stream returns (list_of_streams, integrated)
                    streams, integrated = output
                    for i, s in enumerate(streams):
                        activations[f"{name}_stream{i}"] = s.detach()
                    if integrated is not None:
                        activations[f"{name}_integrated"] = integrated.detach()
            else:
                activations[name] = output.detach()
        return hook

    # Register hooks on key layers
    handles = []
    handles.append(model.conv1.register_forward_hook(make_hook("conv1")))
    handles.append(model.bn1.register_forward_hook(make_hook("bn1")))
    handles.append(model.layer1.register_forward_hook(make_hook("layer1")))
    handles.append(model.layer2.register_forward_hook(make_hook("layer2")))
    handles.append(model.layer3.register_forward_hook(make_hook("layer3")))
    handles.append(model.layer4.register_forward_hook(make_hook("layer4")))

    # Forward pass
    inputs = create_model_input(DIAGNOSE_MODEL)

    with torch.no_grad():
        output = forward_model(model, inputs, DIAGNOSE_MODEL)

    print(f"\nModel: {model_name}")
    print(f"Output: mean={output.mean():.6f}, std={output.std():.6f}")

    # Analyze scale propagation
    print("\n--- Signal Scales Through Network ---")
    print(f"{'Layer':<25} {'Stream0 std':<15} {'Stream1 std':<15} {'Integrated std':<15}")
    print("-" * 70)

    for layer in ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4']:
        s0_key = f"{layer}_stream0"
        s1_key = f"{layer}_stream1"
        int_key = f"{layer}_integrated"

        s0_std = activations.get(s0_key, torch.zeros(1)).std().item() if s0_key in activations else 0
        s1_std = activations.get(s1_key, torch.zeros(1)).std().item() if s1_key in activations else 0
        int_std = activations.get(int_key, torch.zeros(1)).std().item() if int_key in activations else 0

        # Flag significant differences
        flag = ""
        if int_std > 0 and s0_std > 0:
            ratio = int_std / s0_std
            if ratio > 2 or ratio < 0.5:
                flag = " ⚠️"

        print(f"{layer:<25} {s0_std:<15.4f} {s1_std:<15.4f} {int_std:<15.4f}{flag}")

    # Cleanup
    for h in handles:
        h.remove()


# ============================================================================
# Test 7: Weight Distribution Comparison
# ============================================================================
def test_weight_distributions():
    print("\n" + "=" * 80)
    print(f"TEST 7: Weight Distribution Comparison ({DIAGNOSE_MODEL.upper()} vs {COMPARE_MODEL.upper()})")
    print("=" * 80)

    model1 = create_model(DIAGNOSE_MODEL)
    model2 = create_model(COMPARE_MODEL)

    # Collect weight statistics by category
    def get_weight_stats(model, model_type):
        stats = defaultdict(list)
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                category = 'conv'
                if 'bn' in name or 'norm' in name:
                    category = 'norm'
                elif 'fc' in name:
                    category = 'fc'
                elif 'integration' in name:
                    category = 'integration'

                stats[category].append({
                    'name': name,
                    'mean': param.data.mean().item(),
                    'std': param.data.std().item(),
                    'norm': param.data.norm().item(),
                    'shape': list(param.shape),
                })
        return stats

    stats1 = get_weight_stats(model1, DIAGNOSE_MODEL)
    stats2 = get_weight_stats(model2, COMPARE_MODEL)

    print(f"\n--- Weight Statistics by Category ---")
    print(f"{'Category':<15} {'Model':<12} {'Count':<8} {'Mean Std':<12} {'Mean Norm':<12}")
    print("-" * 60)

    for category in sorted(set(stats1.keys()) | set(stats2.keys())):
        if category in stats1:
            avg_std = np.mean([s['std'] for s in stats1[category]])
            avg_norm = np.mean([s['norm'] for s in stats1[category]])
            print(f"{category:<15} {DIAGNOSE_MODEL:<12} {len(stats1[category]):<8} {avg_std:<12.6f} {avg_norm:<12.4f}")

        if category in stats2:
            avg_std = np.mean([s['std'] for s in stats2[category]])
            avg_norm = np.mean([s['norm'] for s in stats2[category]])
            print(f"{'':<15} {COMPARE_MODEL:<12} {len(stats2[category]):<8} {avg_std:<12.6f} {avg_norm:<12.4f}")


# ============================================================================
# Test 8: Integration Weight Analysis
# ============================================================================
def test_integration_weights():
    print("\n" + "=" * 80)
    print(f"TEST 8: Integration Weight Analysis ({DIAGNOSE_MODEL.upper()} vs {COMPARE_MODEL.upper()})")
    print("=" * 80)

    model1 = create_model(DIAGNOSE_MODEL)
    model2 = create_model(COMPARE_MODEL)

    def analyze_integration_weights(model, model_name):
        integration_params = []
        for name, param in model.named_parameters():
            if 'integration' in name.lower():
                integration_params.append({
                    'name': name,
                    'mean': param.data.mean().item(),
                    'std': param.data.std().item(),
                    'min': param.data.min().item(),
                    'max': param.data.max().item(),
                    'norm': param.data.norm().item(),
                    'shape': list(param.shape),
                    'numel': param.numel(),
                })
        return integration_params

    print(f"\n--- {DIAGNOSE_MODEL.upper()} Integration Weights ---")
    params1 = analyze_integration_weights(model1, DIAGNOSE_MODEL)
    if params1:
        total_norm = sum(p['norm'] for p in params1)
        print(f"Total integration params: {len(params1)}")
        print(f"Total integration weight norm: {total_norm:.4f}")
        print(f"\nPer-layer breakdown:")
        for p in params1[:10]:  # First 10
            print(f"  {p['name']}: std={p['std']:.6f}, norm={p['norm']:.4f}")
    else:
        print("No integration weights found")

    print(f"\n--- {COMPARE_MODEL.upper()} Integration Weights ---")
    params2 = analyze_integration_weights(model2, COMPARE_MODEL)
    if params2:
        total_norm = sum(p['norm'] for p in params2)
        print(f"Total integration params: {len(params2)}")
        print(f"Total integration weight norm: {total_norm:.4f}")
        print(f"\nPer-layer breakdown:")
        for p in params2[:10]:  # First 10
            print(f"  {p['name']}: std={p['std']:.6f}, norm={p['norm']:.4f}")
    else:
        print("No integration weights found")


# ============================================================================
# Test 9: Batch Normalization Running Stats Comparison
# ============================================================================
def test_bn_running_stats():
    print("\n" + "=" * 80)
    print(f"TEST 9: BatchNorm Running Stats After Training ({DIAGNOSE_MODEL.upper()} vs {COMPARE_MODEL.upper()})")
    print("=" * 80)

    # Create models and run a few training steps
    model1 = create_model(DIAGNOSE_MODEL)
    model2 = create_model(COMPARE_MODEL)

    model1.train()
    model2.train()

    criterion = nn.CrossEntropyLoss()

    # Run 20 training steps
    for step in range(20):
        inputs1 = create_model_input(DIAGNOSE_MODEL, seed=SEED + step)
        inputs2 = create_model_input(COMPARE_MODEL, seed=SEED + step)
        torch.manual_seed(SEED + step)
        targets = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))

        out1 = forward_model(model1, inputs1, DIAGNOSE_MODEL)
        out2 = forward_model(model2, inputs2, COMPARE_MODEL)

        loss1 = criterion(out1, targets)
        loss2 = criterion(out2, targets)

        loss1.backward()
        loss2.backward()

    # Collect BN running stats
    def get_bn_stats(model):
        stats = []
        for name, module in model.named_modules():
            if hasattr(module, 'running_mean') and module.running_mean is not None:
                # Check for multi-stream BN
                if hasattr(module, 'stream_running_means'):
                    # N-stream BatchNorm
                    for i, (rm, rv) in enumerate(zip(module.stream_running_means, module.stream_running_vars)):
                        stats.append({
                            'name': f"{name}.stream{i}",
                            'mean_of_means': rm.mean().item(),
                            'std_of_means': rm.std().item(),
                            'mean_of_vars': rv.mean().item(),
                            'std_of_vars': rv.std().item(),
                        })
                    if hasattr(module, 'integrated_running_mean') and module.integrated_running_mean is not None:
                        stats.append({
                            'name': f"{name}.integrated",
                            'mean_of_means': module.integrated_running_mean.mean().item(),
                            'std_of_means': module.integrated_running_mean.std().item(),
                            'mean_of_vars': module.integrated_running_var.mean().item(),
                            'std_of_vars': module.integrated_running_var.std().item(),
                        })
                elif hasattr(module, 'stream1_running_mean'):
                    # 2-stream BatchNorm
                    stats.append({
                        'name': f"{name}.stream1",
                        'mean_of_means': module.stream1_running_mean.mean().item(),
                        'std_of_means': module.stream1_running_mean.std().item(),
                        'mean_of_vars': module.stream1_running_var.mean().item(),
                        'std_of_vars': module.stream1_running_var.std().item(),
                    })
                    stats.append({
                        'name': f"{name}.stream2",
                        'mean_of_means': module.stream2_running_mean.mean().item(),
                        'std_of_means': module.stream2_running_mean.std().item(),
                        'mean_of_vars': module.stream2_running_var.mean().item(),
                        'std_of_vars': module.stream2_running_var.std().item(),
                    })
                    if hasattr(module, 'integrated_running_mean') and module.integrated_running_mean is not None:
                        stats.append({
                            'name': f"{name}.integrated",
                            'mean_of_means': module.integrated_running_mean.mean().item(),
                            'std_of_means': module.integrated_running_mean.std().item(),
                            'mean_of_vars': module.integrated_running_var.mean().item(),
                            'std_of_vars': module.integrated_running_var.std().item(),
                        })
                else:
                    # Standard BatchNorm
                    stats.append({
                        'name': name,
                        'mean_of_means': module.running_mean.mean().item(),
                        'std_of_means': module.running_mean.std().item(),
                        'mean_of_vars': module.running_var.mean().item(),
                        'std_of_vars': module.running_var.std().item(),
                    })
        return stats

    stats1 = get_bn_stats(model1)
    stats2 = get_bn_stats(model2)

    print(f"\n--- {DIAGNOSE_MODEL.upper()} BN Running Stats (after 20 steps) ---")
    print(f"{'Layer':<40} {'Mean(means)':<12} {'Mean(vars)':<12}")
    print("-" * 65)
    for s in stats1[:15]:  # First 15
        print(f"{s['name']:<40} {s['mean_of_means']:<12.6f} {s['mean_of_vars']:<12.6f}")

    print(f"\n--- {COMPARE_MODEL.upper()} BN Running Stats (after 20 steps) ---")
    print(f"{'Layer':<40} {'Mean(means)':<12} {'Mean(vars)':<12}")
    print("-" * 65)
    for s in stats2[:15]:  # First 15
        print(f"{s['name']:<40} {s['mean_of_means']:<12.6f} {s['mean_of_vars']:<12.6f}")


# ============================================================================
# Test 10: Gradient Flow Per Layer (Detailed)
# ============================================================================
def test_detailed_gradient_flow():
    print("\n" + "=" * 80)
    print(f"TEST 10: Detailed Gradient Flow Per Layer ({DIAGNOSE_MODEL.upper()} vs {COMPARE_MODEL.upper()})")
    print("=" * 80)

    model1 = create_model(DIAGNOSE_MODEL)
    model2 = create_model(COMPARE_MODEL)

    model1.train()
    model2.train()

    inputs1 = create_model_input(DIAGNOSE_MODEL)
    inputs2 = create_model_input(COMPARE_MODEL)
    torch.manual_seed(SEED)
    targets = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))

    criterion = nn.CrossEntropyLoss()

    out1 = forward_model(model1, inputs1, DIAGNOSE_MODEL)
    out2 = forward_model(model2, inputs2, COMPARE_MODEL)

    loss1 = criterion(out1, targets)
    loss2 = criterion(out2, targets)

    loss1.backward()
    loss2.backward()

    def get_layer_grads(model):
        layer_grads = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                # Extract layer name
                parts = name.split('.')
                if len(parts) >= 2:
                    layer = '.'.join(parts[:2])
                else:
                    layer = parts[0]

                if layer not in layer_grads:
                    layer_grads[layer] = []
                layer_grads[layer].append(param.grad.abs().mean().item())
        return {k: np.mean(v) for k, v in layer_grads.items()}

    grads1 = get_layer_grads(model1)
    grads2 = get_layer_grads(model2)

    print(f"\n--- Mean Gradient Magnitude Per Layer ---")
    print(f"{'Layer':<25} {DIAGNOSE_MODEL:<15} {COMPARE_MODEL:<15} {'Ratio':<10}")
    print("-" * 70)

    all_layers = sorted(set(grads1.keys()) | set(grads2.keys()))
    for layer in all_layers:
        g1 = grads1.get(layer, 0)
        g2 = grads2.get(layer, 0)
        ratio = g1 / g2 if g2 > 0 else float('inf')
        flag = " ⚠️" if ratio > 2 or ratio < 0.5 else ""
        print(f"{layer:<25} {g1:<15.6f} {g2:<15.6f} {ratio:<10.2f}{flag}")


# ============================================================================
# Test 11: Feature Map Statistics at Each Layer
# ============================================================================
def test_feature_map_stats():
    print("\n" + "=" * 80)
    print(f"TEST 11: Feature Map Statistics ({DIAGNOSE_MODEL.upper()} vs {COMPARE_MODEL.upper()})")
    print("=" * 80)

    model1 = create_model(DIAGNOSE_MODEL)
    model2 = create_model(COMPARE_MODEL)

    model1.eval()
    model2.eval()

    # Collect feature map stats with hooks
    stats1 = {}
    stats2 = {}

    def make_stats_hook(stats_dict, model_type):
        def hook(module, input, output):
            name = None
            for n, m in (model1 if model_type == DIAGNOSE_MODEL else model2).named_modules():
                if m is module:
                    name = n
                    break
            if name is None:
                return

            if isinstance(output, tuple):
                if model_type == "original":
                    s1, s2, integrated = output
                    outputs = [('stream0', s1), ('stream1', s2), ('integrated', integrated)]
                else:
                    streams, integrated = output
                    outputs = [(f'stream{i}', s) for i, s in enumerate(streams)]
                    outputs.append(('integrated', integrated))

                for suffix, out in outputs:
                    if out is not None:
                        key = f"{name}.{suffix}"
                        stats_dict[key] = {
                            'mean': out.mean().item(),
                            'std': out.std().item(),
                            'min': out.min().item(),
                            'max': out.max().item(),
                            'sparsity': (out == 0).float().mean().item(),
                        }
            else:
                stats_dict[name] = {
                    'mean': output.mean().item(),
                    'std': output.std().item(),
                    'min': output.min().item(),
                    'max': output.max().item(),
                    'sparsity': (output == 0).float().mean().item(),
                }
        return hook

    # Register hooks on key layers
    target_layers = ['conv1', 'bn1', 'relu', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']
    handles1 = []
    handles2 = []

    for name, module in model1.named_modules():
        if any(t in name for t in target_layers):
            handles1.append(module.register_forward_hook(make_stats_hook(stats1, DIAGNOSE_MODEL)))

    for name, module in model2.named_modules():
        if any(t in name for t in target_layers):
            handles2.append(module.register_forward_hook(make_stats_hook(stats2, COMPARE_MODEL)))

    # Forward pass
    inputs1 = create_model_input(DIAGNOSE_MODEL)
    inputs2 = create_model_input(COMPARE_MODEL)

    with torch.no_grad():
        forward_model(model1, inputs1, DIAGNOSE_MODEL)
        forward_model(model2, inputs2, COMPARE_MODEL)

    # Cleanup
    for h in handles1 + handles2:
        h.remove()

    # Print comparison for integrated stream
    print(f"\n--- Integrated Stream Feature Maps ---")
    print(f"{'Layer':<40} {DIAGNOSE_MODEL+' std':<12} {COMPARE_MODEL+' std':<12} {'Diff %':<10}")
    print("-" * 75)

    for key in sorted(stats1.keys()):
        if 'integrated' in key:
            s1 = stats1[key]
            # Find matching key in stats2
            key2 = key
            if key2 in stats2:
                s2 = stats2[key2]
                diff = abs(s1['std'] - s2['std']) / max(s1['std'], 0.001) * 100
                flag = " ⚠️" if diff > 50 else ""
                print(f"{key:<40} {s1['std']:<12.6f} {s2['std']:<12.6f} {diff:<10.1f}{flag}")


# ============================================================================
# Test 12: Numerical Precision / Reproducibility
# ============================================================================
def test_reproducibility():
    print("\n" + "=" * 80)
    print(f"TEST 12: Numerical Reproducibility ({DIAGNOSE_MODEL.upper()})")
    print("=" * 80)

    # Run the same forward pass twice and check outputs are identical
    model = create_model(DIAGNOSE_MODEL)
    model.eval()

    inputs = create_model_input(DIAGNOSE_MODEL)

    with torch.no_grad():
        out1 = forward_model(model, inputs, DIAGNOSE_MODEL)
        out2 = forward_model(model, inputs, DIAGNOSE_MODEL)

    diff = (out1 - out2).abs().max().item()
    print(f"\nMax difference between two identical forward passes: {diff:.2e}")

    if diff == 0:
        print("✅  Perfect reproducibility")
    elif diff < 1e-6:
        print("✅  Acceptable numerical precision")
    else:
        print("⚠️  Unexpected numerical differences detected")

    # Check that different seeds produce different outputs
    inputs_diff = create_model_input(DIAGNOSE_MODEL, seed=SEED + 100)
    with torch.no_grad():
        out3 = forward_model(model, inputs_diff, DIAGNOSE_MODEL)

    diff_seeds = (out1 - out3).abs().mean().item()
    print(f"Mean difference with different input: {diff_seeds:.6f}")
    if diff_seeds > 0.01:
        print("✅  Model responds to different inputs")
    else:
        print("⚠️  Model may not be responding to input changes")


# ============================================================================
# Test 13: Loss Landscape Smoothness
# ============================================================================
def test_loss_landscape():
    print("\n" + "=" * 80)
    print(f"TEST 13: Loss Landscape Smoothness ({DIAGNOSE_MODEL.upper()} vs {COMPARE_MODEL.upper()})")
    print("=" * 80)

    model1 = create_model(DIAGNOSE_MODEL)
    model2 = create_model(COMPARE_MODEL)

    model1.eval()
    model2.eval()

    criterion = nn.CrossEntropyLoss()

    # Compute loss for multiple random inputs
    losses1 = []
    losses2 = []

    for i in range(20):
        inputs1 = create_model_input(DIAGNOSE_MODEL, seed=SEED + i)
        inputs2 = create_model_input(COMPARE_MODEL, seed=SEED + i)
        torch.manual_seed(SEED + i)
        targets = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))

        with torch.no_grad():
            out1 = forward_model(model1, inputs1, DIAGNOSE_MODEL)
            out2 = forward_model(model2, inputs2, COMPARE_MODEL)

            losses1.append(criterion(out1, targets).item())
            losses2.append(criterion(out2, targets).item())

    print(f"\n--- Loss Statistics (20 random batches) ---")
    print(f"{'Model':<15} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print("-" * 55)
    print(f"{DIAGNOSE_MODEL:<15} {np.mean(losses1):<12.4f} {np.std(losses1):<12.4f} {np.min(losses1):<12.4f} {np.max(losses1):<12.4f}")
    print(f"{COMPARE_MODEL:<15} {np.mean(losses2):<12.4f} {np.std(losses2):<12.4f} {np.min(losses2):<12.4f} {np.max(losses2):<12.4f}")

    # Check for concerning variance
    if np.std(losses1) > np.mean(losses1) * 0.5:
        print(f"\n⚠️  {DIAGNOSE_MODEL} shows high loss variance")
    else:
        print(f"\n✅  {DIAGNOSE_MODEL} loss variance is reasonable")

    if np.std(losses2) > np.mean(losses2) * 0.5:
        print(f"⚠️  {COMPARE_MODEL} shows high loss variance")
    else:
        print(f"✅  {COMPARE_MODEL} loss variance is reasonable")


# ============================================================================
# Test 14: Parameter Count Comparison
# ============================================================================
def test_parameter_count():
    print("\n" + "=" * 80)
    print(f"TEST 14: Parameter Count Comparison")
    print("=" * 80)

    model1 = create_model(DIAGNOSE_MODEL)
    model2 = create_model(COMPARE_MODEL)

    def count_params(model):
        total = 0
        trainable = 0
        by_category = defaultdict(int)

        for name, param in model.named_parameters():
            n = param.numel()
            total += n
            if param.requires_grad:
                trainable += n

            # Categorize
            if 'integration' in name:
                by_category['integration'] += n
            elif 'stream' in name and 'weight' in name:
                by_category['stream_conv'] += n
            elif 'integrated' in name:
                by_category['integrated'] += n
            elif 'bn' in name or 'norm' in name:
                by_category['normalization'] += n
            elif 'fc' in name:
                by_category['classifier'] += n
            else:
                by_category['other'] += n

        return total, trainable, dict(by_category)

    total1, trainable1, cats1 = count_params(model1)
    total2, trainable2, cats2 = count_params(model2)

    print(f"\n--- Total Parameters ---")
    print(f"{DIAGNOSE_MODEL}: {total1:,} ({trainable1:,} trainable)")
    print(f"{COMPARE_MODEL}: {total2:,} ({trainable2:,} trainable)")
    print(f"Difference: {total1 - total2:,} ({(total1-total2)/total2*100:.1f}%)")

    print(f"\n--- Parameters by Category ---")
    print(f"{'Category':<20} {DIAGNOSE_MODEL:<15} {COMPARE_MODEL:<15} {'Diff':<10}")
    print("-" * 60)

    all_cats = sorted(set(cats1.keys()) | set(cats2.keys()))
    for cat in all_cats:
        c1 = cats1.get(cat, 0)
        c2 = cats2.get(cat, 0)
        diff = c1 - c2
        print(f"{cat:<20} {c1:<15,} {c2:<15,} {diff:+,}")


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    ModelClass, BlockClass, model_name = get_model_and_block()

    print("=" * 80)
    print(f"DIAGNOSTIC SCRIPT: {model_name}")
    print("=" * 80)

    try:
        test_forward_activations()
        test_gradient_flow()
        test_integration_signals()
        test_norm_comparison()
        test_training_step()
        test_scale_propagation()
        test_weight_distributions()
        test_integration_weights()
        test_bn_running_stats()
        test_detailed_gradient_flow()
        test_feature_map_stats()
        test_reproducibility()
        test_loss_landscape()
        test_parameter_count()

        print("\n" + "=" * 80)
        print("DIAGNOSTICS COMPLETE")
        print("=" * 80)
        print("\nReview the output above for:")
        print("  - ⚠️  warnings indicating potential issues")
        print("  - Scale mismatches between streams and integrated")
        print("  - Gradient magnitude differences between components")
        print("  - NaN or Inf values")
        print("  - Parameter count discrepancies")
        print("  - BN running stats drift")

    except Exception as e:
        print(f"\n❌ Error during diagnostics: {e}")
        import traceback
        traceback.print_exc()
