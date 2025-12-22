"""
Diagnostic test for LINet3 gradient flow issues.

This script diagnoses why stream pathways aren't learning while
the integrated stream is volatile during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

# Add project root to path
import sys
sys.path.insert(0, '/Users/gclinger/Documents/projects/Multi-Stream-Neural-Networks')

from src.models.linear_integration.li_net3.li_net import LINet, li_resnet18
from src.models.linear_integration.li_net3.conv import LIConv2d
from src.models.linear_integration.li_net3.blocks import LIBasicBlock

def analyze_gradient_flow():
    """Analyze gradient flow from loss back to stream parameters."""
    print("=" * 80)
    print("LINet3 Gradient Flow Analysis")
    print("=" * 80)

    # Create model on CPU explicitly
    model = li_resnet18(num_classes=10, stream_input_channels=[3, 1], device='cpu')
    model = model.to('cpu')
    model.train()

    # Create synthetic inputs on CPU
    batch_size = 4
    rgb = torch.randn(batch_size, 3, 224, 224, device='cpu')
    depth = torch.randn(batch_size, 1, 224, 224, device='cpu')
    targets = torch.randint(0, 10, (batch_size,), device='cpu')

    # Forward pass
    logits = model([rgb, depth])
    loss = F.cross_entropy(logits, targets)

    # Backward pass
    loss.backward()

    # Collect gradient statistics by parameter type
    stream_grads = defaultdict(list)
    integration_grads = []
    integrated_grads = []
    classifier_grads = []

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_mean = param.grad.abs().mean().item()
            grad_max = param.grad.abs().max().item()

            if 'stream_weights' in name or 'stream_biases' in name:
                # Extract stream index
                for i in range(model.num_streams):
                    if f'.{i}.' in name or f'.{i}' in name:
                        stream_grads[f'stream{i}'].append((name, grad_norm, grad_mean, grad_max))
                        break
            elif 'integration_from_streams' in name:
                integration_grads.append((name, grad_norm, grad_mean, grad_max))
            elif 'integrated_weight' in name or 'integrated_bias' in name:
                integrated_grads.append((name, grad_norm, grad_mean, grad_max))
            elif 'fc.' in name:
                classifier_grads.append((name, grad_norm, grad_mean, grad_max))

    # Report results
    print("\n📊 Gradient Statistics:")
    print("-" * 60)

    # Stream pathway gradients
    for stream_name, grads in sorted(stream_grads.items()):
        total_norm = sum(g[1] for g in grads)
        avg_mean = sum(g[2] for g in grads) / len(grads) if grads else 0
        print(f"\n{stream_name.upper()} Pathway:")
        print(f"  Total gradient norm: {total_norm:.6f}")
        print(f"  Average gradient mean: {avg_mean:.6f}")
        print(f"  Number of parameters with gradients: {len(grads)}")
        if grads:
            # Show top 3 by gradient norm
            sorted_grads = sorted(grads, key=lambda x: x[1], reverse=True)[:3]
            print(f"  Top gradients:")
            for name, norm, mean, max_val in sorted_grads:
                short_name = name.split('.')[-2] + '.' + name.split('.')[-1]
                print(f"    {short_name}: norm={norm:.6f}, mean={mean:.6f}")

    # Integration weights gradients
    print(f"\nINTEGRATION Weights (from_streams):")
    total_norm = sum(g[1] for g in integration_grads)
    avg_mean = sum(g[2] for g in integration_grads) / len(integration_grads) if integration_grads else 0
    print(f"  Total gradient norm: {total_norm:.6f}")
    print(f"  Average gradient mean: {avg_mean:.6f}")
    print(f"  Number of parameters: {len(integration_grads)}")

    # Integrated pathway gradients
    print(f"\nINTEGRATED Pathway (previous layer weights):")
    total_norm = sum(g[1] for g in integrated_grads)
    avg_mean = sum(g[2] for g in integrated_grads) / len(integrated_grads) if integrated_grads else 0
    print(f"  Total gradient norm: {total_norm:.6f}")
    print(f"  Average gradient mean: {avg_mean:.6f}")
    print(f"  Number of parameters: {len(integrated_grads)}")

    # Classifier gradients
    print(f"\nCLASSIFIER (fc):")
    total_norm = sum(g[1] for g in classifier_grads)
    avg_mean = sum(g[2] for g in classifier_grads) / len(classifier_grads) if classifier_grads else 0
    print(f"  Total gradient norm: {total_norm:.6f}")
    print(f"  Average gradient mean: {avg_mean:.6f}")

    # KEY INSIGHT: Compare gradient magnitudes
    print("\n" + "=" * 80)
    print("KEY INSIGHT: Gradient Ratio Analysis")
    print("=" * 80)

    stream0_norm = sum(g[1] for g in stream_grads.get('stream0', []))
    stream1_norm = sum(g[1] for g in stream_grads.get('stream1', []))
    integration_norm = sum(g[1] for g in integration_grads)

    if stream0_norm > 0:
        print(f"\nIntegration/Stream0 gradient ratio: {integration_norm/stream0_norm:.2f}x")
    if stream1_norm > 0:
        print(f"Integration/Stream1 gradient ratio: {integration_norm/stream1_norm:.2f}x")

    print("\n⚠️  If integration gradients are much larger than stream gradients,")
    print("   the integration weights will update much faster, causing volatility")
    print("   while streams appear to not learn.")


def analyze_gradient_flow_per_layer():
    """Analyze gradient flow at each layer."""
    print("\n" + "=" * 80)
    print("Per-Layer Gradient Analysis")
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

    # Analyze by layer
    layers_to_check = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']

    for layer_name in layers_to_check:
        layer = getattr(model, layer_name, None)
        if layer is None:
            continue

        stream_grad_norms = defaultdict(float)
        integration_grad_norm = 0.0
        integrated_grad_norm = 0.0

        for name, param in layer.named_parameters():
            if param.grad is None:
                continue
            grad_norm = param.grad.norm().item()

            if 'stream_weights' in name:
                if '.0.' in name or '.0' == name.split('.')[-2]:
                    stream_grad_norms['stream0'] += grad_norm
                else:
                    stream_grad_norms['stream1'] += grad_norm
            elif 'integration_from_streams' in name:
                integration_grad_norm += grad_norm
            elif 'integrated_weight' in name:
                integrated_grad_norm += grad_norm

        print(f"\n{layer_name}:")
        print(f"  Stream0 grad norm:     {stream_grad_norms.get('stream0', 0):.6f}")
        print(f"  Stream1 grad norm:     {stream_grad_norms.get('stream1', 0):.6f}")
        print(f"  Integration grad norm: {integration_grad_norm:.6f}")
        print(f"  Integrated grad norm:  {integrated_grad_norm:.6f}")

        # Ratio analysis
        total_stream = stream_grad_norms.get('stream0', 0) + stream_grad_norms.get('stream1', 0)
        if total_stream > 0 and integration_grad_norm > 0:
            ratio = integration_grad_norm / total_stream
            if ratio > 10:
                print(f"  ⚠️  Integration/Streams ratio: {ratio:.1f}x (HIGH - integration dominates!)")
            elif ratio > 2:
                print(f"  ⚡ Integration/Streams ratio: {ratio:.1f}x (moderate)")
            else:
                print(f"  ✓ Integration/Streams ratio: {ratio:.1f}x (balanced)")


def simulate_training_dynamics():
    """Simulate several training steps to see weight change patterns."""
    print("\n" + "=" * 80)
    print("Training Dynamics Simulation (10 steps)")
    print("=" * 80)

    model = li_resnet18(num_classes=10, stream_input_channels=[3, 1], device='cpu')
    model = model.to('cpu')
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Record initial weights
    initial_stream_weights = {}
    initial_integration_weights = {}
    initial_integrated_weights = {}

    for name, param in model.named_parameters():
        if 'stream_weights' in name:
            initial_stream_weights[name] = param.data.clone()
        elif 'integration_from_streams' in name:
            initial_integration_weights[name] = param.data.clone()
        elif 'integrated_weight' in name:
            initial_integrated_weights[name] = param.data.clone()

    # Training loop
    for step in range(10):
        batch_size = 4
        rgb = torch.randn(batch_size, 3, 224, 224, device='cpu')
        depth = torch.randn(batch_size, 1, 224, 224, device='cpu')
        targets = torch.randint(0, 10, (batch_size,), device='cpu')

        optimizer.zero_grad()
        logits = model([rgb, depth])
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()

    # Compute weight changes
    stream_changes = []
    integration_changes = []
    integrated_changes = []

    for name, param in model.named_parameters():
        if name in initial_stream_weights:
            change = (param.data - initial_stream_weights[name]).norm().item()
            stream_changes.append((name, change))
        elif name in initial_integration_weights:
            change = (param.data - initial_integration_weights[name]).norm().item()
            integration_changes.append((name, change))
        elif name in initial_integrated_weights:
            change = (param.data - initial_integrated_weights[name]).norm().item()
            integrated_changes.append((name, change))

    # Report
    print("\nWeight Changes After 10 Steps:")
    print("-" * 60)

    total_stream_change = sum(c[1] for c in stream_changes)
    total_integration_change = sum(c[1] for c in integration_changes)
    total_integrated_change = sum(c[1] for c in integrated_changes)

    print(f"\nStream Weights Total Change:      {total_stream_change:.6f}")
    print(f"Integration Weights Total Change: {total_integration_change:.6f}")
    print(f"Integrated Weights Total Change:  {total_integrated_change:.6f}")

    if total_stream_change > 0:
        ratio = total_integration_change / total_stream_change
        print(f"\nIntegration/Stream change ratio: {ratio:.2f}x")

        if ratio > 10:
            print("\n🚨 DIAGNOSIS: Integration weights are changing MUCH faster than stream weights!")
            print("   This explains why streams appear flat while integrated is volatile.")
            print("\n   POTENTIAL FIXES:")
            print("   1. Use layer-wise learning rates (lower LR for integration weights)")
            print("   2. Normalize integration weights or use weight decay")
            print("   3. Initialize integration weights smaller")
            print("   4. Add auxiliary losses for stream pathways")
        elif ratio > 2:
            print("\n⚠️  Integration weights changing faster than stream weights (moderate)")
        else:
            print("\n✓ Weight changes are relatively balanced")


def check_integration_weight_initialization():
    """Check how integration weights are initialized vs stream weights."""
    print("\n" + "=" * 80)
    print("Weight Initialization Analysis")
    print("=" * 80)

    model = li_resnet18(num_classes=10, stream_input_channels=[3, 1], device='cpu')
    model = model.to('cpu')

    stream_weight_stats = []
    integration_weight_stats = []

    for name, param in model.named_parameters():
        if 'stream_weights' in name:
            stream_weight_stats.append({
                'name': name,
                'mean': param.data.mean().item(),
                'std': param.data.std().item(),
                'norm': param.data.norm().item(),
                'shape': tuple(param.shape)
            })
        elif 'integration_from_streams' in name:
            integration_weight_stats.append({
                'name': name,
                'mean': param.data.mean().item(),
                'std': param.data.std().item(),
                'norm': param.data.norm().item(),
                'shape': tuple(param.shape)
            })

    print("\nStream Weights (sample):")
    for stat in stream_weight_stats[:3]:
        print(f"  {stat['name'].split('.')[-2]}: std={stat['std']:.4f}, norm={stat['norm']:.4f}, shape={stat['shape']}")

    print("\nIntegration Weights (sample):")
    for stat in integration_weight_stats[:3]:
        print(f"  {stat['name'].split('.')[-2]}: std={stat['std']:.4f}, norm={stat['norm']:.4f}, shape={stat['shape']}")

    # Compare fan-in
    print("\n📐 Fan-in Analysis:")
    if stream_weight_stats and integration_weight_stats:
        # Stream weights: (out_ch, in_ch, kH, kW)
        sample_stream = stream_weight_stats[0]
        stream_fan_in = sample_stream['shape'][1] * sample_stream['shape'][2] * sample_stream['shape'][3]

        # Integration weights: (out_ch, in_ch, 1, 1)
        sample_integration = integration_weight_stats[0]
        integration_fan_in = sample_integration['shape'][1]

        print(f"  Stream fan-in (conv1):      {stream_fan_in} (in_ch × kH × kW)")
        print(f"  Integration fan-in (conv1): {integration_fan_in} (just in_ch)")

        if stream_fan_in > integration_fan_in:
            print(f"\n⚠️  Integration has {stream_fan_in/integration_fan_in:.1f}x smaller fan-in")
            print("   Kaiming init produces LARGER weights for integration!")
            print("   This can cause integration to dominate early in training.")


def analyze_forward_pass_magnitudes():
    """Analyze activation magnitudes at different points."""
    print("\n" + "=" * 80)
    print("Forward Pass Activation Analysis")
    print("=" * 80)

    model = li_resnet18(num_classes=10, stream_input_channels=[3, 1], device='cpu')
    model = model.to('cpu')
    model.eval()

    batch_size = 4
    rgb = torch.randn(batch_size, 3, 224, 224, device='cpu')
    depth = torch.randn(batch_size, 1, 224, 224, device='cpu')

    # Hook to capture activations
    activations = {}

    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                stream_outputs, integrated = output
                activations[f'{name}_streams'] = [s.detach() for s in stream_outputs]
                activations[f'{name}_integrated'] = integrated.detach()
            else:
                activations[name] = output.detach()
        return hook

    # Register hooks
    model.conv1.register_forward_hook(hook_fn('conv1'))
    model.layer1.register_forward_hook(hook_fn('layer1'))
    model.layer2.register_forward_hook(hook_fn('layer2'))
    model.layer3.register_forward_hook(hook_fn('layer3'))
    model.layer4.register_forward_hook(hook_fn('layer4'))

    # Forward pass
    with torch.no_grad():
        logits = model([rgb, depth])

    # Analyze
    print("\nActivation Magnitudes (mean absolute value):")
    print("-" * 60)

    for layer in ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']:
        streams = activations.get(f'{layer}_streams')
        integrated = activations.get(f'{layer}_integrated')

        if streams and integrated is not None:
            stream0_mag = streams[0].abs().mean().item()
            stream1_mag = streams[1].abs().mean().item()
            integrated_mag = integrated.abs().mean().item()

            print(f"\n{layer}:")
            print(f"  Stream0 magnitude:    {stream0_mag:.4f}")
            print(f"  Stream1 magnitude:    {stream1_mag:.4f}")
            print(f"  Integrated magnitude: {integrated_mag:.4f}")

            if integrated_mag > 2 * max(stream0_mag, stream1_mag):
                print(f"  ⚠️  Integrated is {integrated_mag/max(stream0_mag, stream1_mag):.1f}x larger than streams!")


if __name__ == "__main__":
    print("\n🔍 Running LINet3 Gradient Flow Diagnostics...\n")

    analyze_gradient_flow()
    analyze_gradient_flow_per_layer()
    simulate_training_dynamics()
    check_integration_weight_initialization()
    analyze_forward_pass_magnitudes()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
The key issue with LINet3 training may be:

1. GRADIENT IMBALANCE: Integration weights (1x1 convs) have smaller fan-in
   than stream weights (3x3 or 7x7 convs). With Kaiming init, this means
   integration weights start LARGER and get LARGER gradients.

2. INTEGRATION DOMINATES: Since integrated stream is the only one used for
   classification, gradients flow primarily through integration weights.
   Stream weights only receive gradients indirectly through integration.

3. POTENTIAL FIXES:
   a) Use smaller initialization for integration weights
   b) Use differential learning rates (lower for integration)
   c) Add weight decay specifically to integration weights
   d) Add auxiliary losses for stream pathways
   e) Use gradient scaling/normalization

Run this script to see specific numbers for your model!
""")
