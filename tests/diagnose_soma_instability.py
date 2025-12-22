"""
Diagnostic script to identify instability issues in LINet models.

This script can diagnose both LINet3 (BatchNorm) and LINet Soma (GroupNorm) to identify
potential causes of training instability.

Usage:
    python tests/diagnose_soma_instability.py          # Diagnose Soma (default)
    python tests/diagnose_soma_instability.py soma     # Diagnose Soma
    python tests/diagnose_soma_instability.py li_net3  # Diagnose LINet3

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

# Import both models for comparison
from src.models.linear_integration.li_net3.li_net import LINet as LINet3
from src.models.linear_integration.li_net3.blocks import LIBasicBlock as LIBasicBlock3

from src.models.linear_integration.li_net3_soma.li_net_soma import LINet as LINetSoma
from src.models.linear_integration.li_net3_soma.blocks import LIBasicBlock as LIBasicBlockSoma

# Parse command line argument for model selection
DIAGNOSE_MODEL = "soma"  # Default
if len(sys.argv) > 1:
    if sys.argv[1].lower() in ["soma", "li_net3", "linet3"]:
        DIAGNOSE_MODEL = sys.argv[1].lower()
        if DIAGNOSE_MODEL == "linet3":
            DIAGNOSE_MODEL = "li_net3"
    else:
        print(f"Unknown model: {sys.argv[1]}. Use 'soma' or 'li_net3'")
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

def get_model_and_block():
    """Get the model class and block class based on DIAGNOSE_MODEL."""
    if DIAGNOSE_MODEL == "soma":
        return LINetSoma, LIBasicBlockSoma, "LINet Soma (GroupNorm on integrated)"
    else:
        return LINet3, LIBasicBlock3, "LINet3 (BatchNorm on integrated)"


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
# Test 1: Compare Forward Pass Activations (Both Models)
# ============================================================================
def test_forward_activations():
    print("\n" + "=" * 80)
    print("TEST 1: Forward Pass Activation Statistics (Comparing Both Models)")
    print("=" * 80)

    stream_channels = [3, 1]  # RGB + Depth

    # Create models - force CPU to avoid device issues
    torch.manual_seed(SEED)
    model_li3 = LINet3(
        block=LIBasicBlock3,
        layers=[2, 2, 2, 2],
        num_classes=NUM_CLASSES,
        stream_input_channels=stream_channels,
    )
    model_li3 = model_li3.to('cpu')
    model_li3.eval()

    torch.manual_seed(SEED)
    model_soma = LINetSoma(
        block=LIBasicBlockSoma,
        layers=[2, 2, 2, 2],
        num_classes=NUM_CLASSES,
        stream_input_channels=stream_channels,
    )
    model_soma = model_soma.to('cpu')
    model_soma.eval()

    # Create same input for both
    torch.manual_seed(SEED)
    inputs = create_dummy_input(BATCH_SIZE, stream_channels, IMAGE_SIZE, device='cpu')

    # Register hooks
    hook_li3 = ActivationHook()
    hook_soma = ActivationHook()

    # Register on key layers
    key_layers = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']
    hook_li3.register(model_li3, key_layers)
    hook_soma.register(model_soma, key_layers)

    # Forward pass
    with torch.no_grad():
        out_li3 = model_li3(inputs)
        out_soma = model_soma(inputs)

    # Compare outputs
    print("\n--- Model Outputs ---")
    print(f"LINet3 output: mean={out_li3.mean():.6f}, std={out_li3.std():.6f}")
    print(f"Soma output:   mean={out_soma.mean():.6f}, std={out_soma.std():.6f}")

    # Compare key activations
    print("\n--- Key Layer Activations ---")

    # Find matching keys
    li3_keys = set(hook_li3.activations.keys())
    soma_keys = set(hook_soma.activations.keys())

    print(f"\nLINet3 layers captured: {len(li3_keys)}")
    print(f"Soma layers captured: {len(soma_keys)}")

    # Compare common layers
    common_keys = sorted(li3_keys & soma_keys)
    print(f"\nComparing {len(common_keys)} common layers:")

    for key in common_keys[:10]:  # First 10
        li3_act = hook_li3.activations[key]
        soma_act = hook_soma.activations[key]

        li3_stats = get_activation_stats(li3_act, f"LINet3.{key}")
        soma_stats = get_activation_stats(soma_act, f"Soma.{key}")

        scale_diff = abs(li3_stats['std'] - soma_stats['std']) / max(li3_stats['std'], 0.001)

        if scale_diff > 0.5:  # More than 50% difference
            print(f"\n⚠️  SCALE DIFFERENCE in {key}:")
            print(f"   LINet3: std={li3_stats['std']:.6f}")
            print(f"   Soma:   std={soma_stats['std']:.6f}")
            print(f"   Diff:   {scale_diff*100:.1f}%")

    hook_li3.remove()
    hook_soma.remove()


# ============================================================================
# Test 2: Gradient Analysis
# ============================================================================
def test_gradient_flow():
    print("\n" + "=" * 80)
    print(f"TEST 2: Gradient Flow Analysis ({DIAGNOSE_MODEL.upper()})")
    print("=" * 80)

    stream_channels = [3, 1]
    ModelClass, BlockClass, model_name = get_model_and_block()

    # Create model
    torch.manual_seed(SEED)
    model = ModelClass(
        block=BlockClass,
        layers=[2, 2, 2, 2],
        num_classes=NUM_CLASSES,
        stream_input_channels=stream_channels,
    )
    model = model.to('cpu')
    model.train()

    # Create input and target
    torch.manual_seed(SEED)
    inputs = create_dummy_input(BATCH_SIZE, stream_channels, IMAGE_SIZE, device='cpu')
    targets = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))

    # Forward + backward
    criterion = nn.CrossEntropyLoss()
    outputs = model(inputs)
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

    stream_channels = [3, 1]
    ModelClass, BlockClass, model_name = get_model_and_block()

    # Create model
    torch.manual_seed(SEED)
    model = ModelClass(
        block=BlockClass,
        layers=[2, 2, 2, 2],
        num_classes=NUM_CLASSES,
        stream_input_channels=stream_channels,
    )
    model = model.to('cpu')
    model.eval()

    # Create input
    torch.manual_seed(SEED)
    inputs = create_dummy_input(BATCH_SIZE, stream_channels, IMAGE_SIZE, device='cpu')

    # Capture intermediate values from first conv layer
    conv1 = model.conv1

    # Manually run forward to capture raw vs biased
    stream_inputs = inputs
    integrated_input = None

    print(f"\nModel: {model_name}")

    with torch.no_grad():
        # Access internal state of LIConv2d
        stream_weights_list = list(conv1.stream_weights)
        integration_from_streams_weights_list = list(conv1.integration_from_streams)
        stream_biases_list = list(conv1.stream_biases) if conv1.stream_biases is not None else [None] * conv1.num_streams

        print("\n--- First Conv Layer (conv1) ---")

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
        print(f"Note: Both LINet3 and Soma integrate RAW (no bias) outputs")

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

    stream_channels = [3, 1]
    ModelClass, BlockClass, model_name = get_model_and_block()

    torch.manual_seed(SEED)
    model = ModelClass(
        block=BlockClass,
        layers=[2, 2, 2, 2],
        num_classes=NUM_CLASSES,
        stream_input_channels=stream_channels,
    )
    model = model.to('cpu')
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Track loss over several steps
    losses = []
    grad_norms = []

    print(f"\nModel: {model_name}")
    print("\n--- Training Steps ---")
    for step in range(10):
        torch.manual_seed(SEED + step)
        inputs = create_dummy_input(BATCH_SIZE, stream_channels, IMAGE_SIZE, device='cpu')
        targets = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))

        optimizer.zero_grad()
        outputs = model(inputs)
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

    stream_channels = [3, 1]
    ModelClass, BlockClass, model_name = get_model_and_block()

    torch.manual_seed(SEED)
    model = ModelClass(
        block=BlockClass,
        layers=[2, 2, 2, 2],
        num_classes=NUM_CLASSES,
        stream_input_channels=stream_channels,
    )
    model = model.to('cpu')
    model.eval()

    # Hook to capture outputs at each major stage
    activations = {}

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                # For LI layers, capture stream outputs and integrated
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
    torch.manual_seed(SEED)
    inputs = create_dummy_input(BATCH_SIZE, stream_channels, IMAGE_SIZE, device='cpu')

    with torch.no_grad():
        output = model(inputs)

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

        print("\n" + "=" * 80)
        print("DIAGNOSTICS COMPLETE")
        print("=" * 80)
        print("\nReview the output above for:")
        print("  - ⚠️  warnings indicating potential issues")
        print("  - Scale mismatches between streams and integrated")
        print("  - Gradient magnitude differences between components")
        print("  - NaN or Inf values")

    except Exception as e:
        print(f"\n❌ Error during diagnostics: {e}")
        import traceback
        traceback.print_exc()
