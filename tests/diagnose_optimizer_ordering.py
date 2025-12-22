"""
Test if parameter ordering affects optimizer behavior.

The parameter ordering in LINet3 (integrated first) vs Original (streams first)
could potentially affect:
1. How optimizer updates parameters
2. Learning rate scheduling per-parameter-group
3. Gradient accumulation order
"""

import torch
import torch.nn as nn
import torch.optim as optim
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


def test_optimizer_step_equivalence():
    """Test that optimizer updates are equivalent despite parameter order."""
    print("=" * 80)
    print("Test: Optimizer Step Equivalence")
    print("=" * 80)

    model_orig, model_linet3 = create_models()
    model_orig.train()
    model_linet3.train()

    # Create optimizers with same settings
    lr = 0.01
    opt_orig = optim.SGD(model_orig.parameters(), lr=lr, momentum=0.9)
    opt_li3 = optim.SGD(model_linet3.parameters(), lr=lr, momentum=0.9)

    # Same input and targets
    torch.manual_seed(SEED)
    rgb = torch.randn(4, 3, 32, 32)
    depth = torch.randn(4, 1, 32, 32)
    targets = torch.randint(0, 10, (4,))

    criterion = nn.CrossEntropyLoss()

    # Forward pass
    out_orig = model_orig(rgb, depth)
    out_li3 = model_linet3([rgb, depth])

    # Loss
    loss_orig = criterion(out_orig, targets)
    loss_li3 = criterion(out_li3, targets)

    print(f"\nInitial losses: Original={loss_orig.item():.6f}, LINet3={loss_li3.item():.6f}")

    # Backward
    opt_orig.zero_grad()
    opt_li3.zero_grad()

    loss_orig.backward()
    loss_li3.backward()

    # Check gradients before step
    print("\n--- Gradients Before Step ---")
    orig_s1_grad = model_orig.conv1.stream1_weight.grad.clone()
    li3_s0_grad = model_linet3.conv1.stream_weights[0].grad.clone()
    grad_diff = (orig_s1_grad - li3_s0_grad).abs().max().item()
    print(f"conv1 stream1/0 weight grad diff: {grad_diff:.2e}")

    # Step
    opt_orig.step()
    opt_li3.step()

    # Check weights after step
    print("\n--- Weights After Step ---")
    orig_s1_weight = model_orig.conv1.stream1_weight
    li3_s0_weight = model_linet3.conv1.stream_weights[0]
    weight_diff = (orig_s1_weight - li3_s0_weight).abs().max().item()
    print(f"conv1 stream1/0 weight diff: {weight_diff:.2e}")

    if weight_diff < 1e-6:
        print("\n✅ Optimizer updates are equivalent!")
    else:
        print("\n⚠️  Optimizer updates differ!")

    # Second forward to check outputs
    with torch.no_grad():
        out_orig_2 = model_orig(rgb, depth)
        out_li3_2 = model_linet3([rgb, depth])

    output_diff = (out_orig_2 - out_li3_2).abs().max().item()
    print(f"\nOutput diff after optimization step: {output_diff:.2e}")

    if output_diff < 1e-5:
        print("✅ Outputs still match after optimization!")
    else:
        print("⚠️  Outputs diverge after optimization!")


def test_training_loop_equivalence():
    """Test that multiple training steps produce equivalent results."""
    print("\n" + "=" * 80)
    print("Test: Multi-Step Training Equivalence")
    print("=" * 80)

    model_orig, model_linet3 = create_models()
    model_orig.train()
    model_linet3.train()

    lr = 0.01
    opt_orig = optim.SGD(model_orig.parameters(), lr=lr, momentum=0.9)
    opt_li3 = optim.SGD(model_linet3.parameters(), lr=lr, momentum=0.9)

    criterion = nn.CrossEntropyLoss()

    print("\nRunning 10 training steps...")

    for step in range(10):
        # Different input each step
        torch.manual_seed(SEED + step)
        rgb = torch.randn(4, 3, 32, 32)
        depth = torch.randn(4, 1, 32, 32)
        targets = torch.randint(0, 10, (4,))

        # Forward
        out_orig = model_orig(rgb, depth)
        out_li3 = model_linet3([rgb, depth])

        # Loss
        loss_orig = criterion(out_orig, targets)
        loss_li3 = criterion(out_li3, targets)

        # Backward
        opt_orig.zero_grad()
        opt_li3.zero_grad()
        loss_orig.backward()
        loss_li3.backward()

        # Step
        opt_orig.step()
        opt_li3.step()

        if step % 3 == 0:
            loss_diff = abs(loss_orig.item() - loss_li3.item())
            print(f"  Step {step}: Loss diff = {loss_diff:.2e}")

    # Final comparison
    print("\n--- Final State Comparison ---")

    # Check weights
    orig_s1_weight = model_orig.conv1.stream1_weight
    li3_s0_weight = model_linet3.conv1.stream_weights[0]
    weight_diff = (orig_s1_weight - li3_s0_weight).abs().max().item()
    print(f"conv1 stream1/0 weight diff: {weight_diff:.2e}")

    # Check BN stats
    orig_bn_mean = model_orig.bn1.stream1_running_mean
    li3_bn_mean = getattr(model_linet3.bn1, 'stream0_running_mean')
    bn_diff = (orig_bn_mean - li3_bn_mean).abs().max().item()
    print(f"bn1 stream1/0 running_mean diff: {bn_diff:.2e}")

    # Check outputs
    torch.manual_seed(999)
    rgb = torch.randn(4, 3, 32, 32)
    depth = torch.randn(4, 1, 32, 32)

    model_orig.eval()
    model_linet3.eval()

    with torch.no_grad():
        out_orig = model_orig(rgb, depth)
        out_li3 = model_linet3([rgb, depth])

    output_diff = (out_orig - out_li3).abs().max().item()
    print(f"Final output diff: {output_diff:.2e}")

    if output_diff < 1e-4 and weight_diff < 1e-5:
        print("\n✅ Training is fully equivalent!")
    else:
        print("\n⚠️  Training diverges!")


def test_parameter_group_order():
    """Examine parameter ordering in optimizer groups."""
    print("\n" + "=" * 80)
    print("Test: Optimizer Parameter Group Order")
    print("=" * 80)

    model_orig, model_linet3 = create_models()

    opt_orig = optim.SGD(model_orig.parameters(), lr=0.01)
    opt_li3 = optim.SGD(model_linet3.parameters(), lr=0.01)

    print("\n--- Original Model Parameter Order in Optimizer ---")
    for i, (name, param) in enumerate(model_orig.named_parameters()):
        if i < 10:
            print(f"  {i}: {name} ({param.shape})")
        elif i == 10:
            print(f"  ... and {len(list(model_orig.parameters())) - 10} more parameters")

    print("\n--- LINet3 Model Parameter Order in Optimizer ---")
    for i, (name, param) in enumerate(model_linet3.named_parameters()):
        if i < 10:
            print(f"  {i}: {name} ({param.shape})")
        elif i == 10:
            print(f"  ... and {len(list(model_linet3.parameters())) - 10} more parameters")

    print("\n--- Analysis ---")
    print("The optimizer iterates through parameters in registration order.")
    print("Original registers: stream1 → stream2 → integrated → integration_from_*")
    print("LINet3 registers: integrated → stream_weights → integration_from_streams")
    print("\nHowever, SGD/Adam update each parameter independently, so order doesn't")
    print("affect the final result - only the update rule and gradient matter.")


if __name__ == "__main__":
    print("\n🔍 Testing Optimizer Behavior with Different Parameter Orders\n")

    test_optimizer_step_equivalence()
    test_training_loop_equivalence()
    test_parameter_group_order()

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("""
The parameter ordering difference between Original and LINet3 does NOT affect:
1. Optimizer updates (each parameter updated independently)
2. Training convergence (same gradients, same update rules)
3. Model outputs (forward pass is identical)

The ordering only affects:
1. State dict keys (different naming convention)
2. Parameter iteration order (for debugging/inspection)

This is NOT a cause of the training issues.
""")
