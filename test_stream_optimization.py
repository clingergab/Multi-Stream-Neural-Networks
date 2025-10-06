"""
Test stream-specific optimization in MCResNet.

Verifies that:
1. Parameters are correctly separated by stream
2. Different learning rates are applied to different streams
3. Different weight decay is applied to different streams
4. Optimizer parameter groups work correctly
"""

import torch
from src.models.multi_channel import mc_resnet18

def test_stream_specific_optimization():
    """Test that stream-specific optimization works correctly."""

    print("Testing Stream-Specific Optimization")
    print("=" * 70)

    # Test parameters
    num_classes = 27
    batch_size = 4
    img_size = 224

    # Create model
    model = mc_resnet18(
        num_classes=num_classes,
        stream1_input_channels=3,  # RGB
        stream2_input_channels=1,  # Depth
        fusion_type='weighted',
        device='cpu'
    )

    print("\n1. Parameter Separation Test")
    print("-" * 70)

    # Count parameters by stream
    stream1_params = []
    stream2_params = []
    shared_params = []

    for name, param in model.named_parameters():
        if 'stream1' in name:
            stream1_params.append((name, param))
        elif 'stream2' in name:
            stream2_params.append((name, param))
        else:
            shared_params.append((name, param))

    print(f"Stream1 parameters: {len(stream1_params)} tensors, {sum(p.numel() for _, p in stream1_params):,} values")
    print(f"Stream2 parameters: {len(stream2_params)} tensors, {sum(p.numel() for _, p in stream2_params):,} values")
    print(f"Shared parameters: {len(shared_params)} tensors, {sum(p.numel() for _, p in shared_params):,} values")

    print(f"\n✓ Sample stream1 params: {stream1_params[0][0]}, {stream1_params[1][0]}")
    print(f"✓ Sample stream2 params: {stream2_params[0][0]}, {stream2_params[1][0]}")
    print(f"✓ Sample shared params: {shared_params[0][0]}, {shared_params[1][0]}")

    print("\n2. Standard Optimization (No Stream-Specific)")
    print("-" * 70)

    model.compile(
        optimizer='adamw',
        learning_rate=1e-3,
        weight_decay=1e-2
    )

    # Check optimizer has single parameter group
    assert len(model.optimizer.param_groups) == 1, "Should have 1 param group for standard optimization"
    print(f"✓ Parameter groups: {len(model.optimizer.param_groups)}")
    print(f"✓ Learning rate: {model.optimizer.param_groups[0]['lr']}")
    print(f"✓ Weight decay: {model.optimizer.param_groups[0]['weight_decay']}")

    print("\n3. Stream-Specific Learning Rates")
    print("-" * 70)

    # Compile with stream-specific learning rates
    stream1_lr = 2e-4
    stream2_lr = 5e-4
    base_lr = 1e-3

    model.compile(
        optimizer='adamw',
        learning_rate=base_lr,
        weight_decay=1e-2,
        stream1_lr=stream1_lr,
        stream2_lr=stream2_lr
    )

    # Check optimizer has multiple parameter groups
    n_groups = len(model.optimizer.param_groups)
    print(f"✓ Parameter groups: {n_groups}")
    assert n_groups == 3, f"Should have 3 param groups (stream1, stream2, shared), got {n_groups}"

    # Verify learning rates
    lrs = [pg['lr'] for pg in model.optimizer.param_groups]
    print(f"✓ Learning rates: {lrs}")
    assert stream1_lr in lrs, f"Stream1 LR {stream1_lr} not found in {lrs}"
    assert stream2_lr in lrs, f"Stream2 LR {stream2_lr} not found in {lrs}"
    assert base_lr in lrs, f"Base LR {base_lr} not found in {lrs}"

    print("\n4. Stream-Specific Weight Decay")
    print("-" * 70)

    # Compile with stream-specific weight decay
    stream1_wd = 1e-3
    stream2_wd = 5e-2
    base_wd = 1e-2

    model.compile(
        optimizer='adamw',
        learning_rate=1e-3,
        weight_decay=base_wd,
        stream1_weight_decay=stream1_wd,
        stream2_weight_decay=stream2_wd
    )

    # Verify weight decays
    wds = [pg['weight_decay'] for pg in model.optimizer.param_groups]
    print(f"✓ Weight decays: {wds}")
    assert stream1_wd in wds, f"Stream1 WD {stream1_wd} not found in {wds}"
    assert stream2_wd in wds, f"Stream2 WD {stream2_wd} not found in {wds}"
    assert base_wd in wds, f"Base WD {base_wd} not found in {wds}"

    print("\n5. Combined Stream-Specific Optimization")
    print("-" * 70)

    # Compile with both LR and WD stream-specific
    model.compile(
        optimizer='adamw',
        learning_rate=1e-3,
        weight_decay=2e-2,
        stream1_lr=1e-4,
        stream2_lr=5e-4,
        stream1_weight_decay=1e-3,
        stream2_weight_decay=4e-2
    )

    print(f"✓ Parameter groups: {len(model.optimizer.param_groups)}")
    for i, pg in enumerate(model.optimizer.param_groups):
        param_count = sum(p.numel() for p in pg['params'])
        print(f"  Group {i}: lr={pg['lr']:.1e}, wd={pg['weight_decay']:.1e}, params={param_count:,}")

    print("\n6. Training Step Test")
    print("-" * 70)

    # Create sample data
    stream1_input = torch.randn(batch_size, 3, img_size, img_size)
    stream2_input = torch.randn(batch_size, 1, img_size, img_size)
    targets = torch.randint(0, num_classes, (batch_size,))

    # Set up loss
    model.compile(
        optimizer='adamw',
        learning_rate=1e-3,
        weight_decay=2e-2,
        stream1_lr=2e-4,  # Lower LR for stream1 (RGB)
        stream2_lr=5e-4,  # Higher LR for stream2 (depth)
        stream1_weight_decay=1e-3,  # Lower WD for stream1
        stream2_weight_decay=4e-2,  # Higher WD for stream2
        loss='cross_entropy'
    )

    # Training step
    model.train()

    # Get initial weights
    initial_stream1_weight = model.conv1.stream1_weight.clone()
    initial_stream2_weight = model.conv1.stream2_weight.clone()

    # Forward pass
    output = model(stream1_input, stream2_input)
    loss = model.criterion(output, targets)

    # Backward pass
    model.optimizer.zero_grad()
    loss.backward()
    model.optimizer.step()

    # Check weights changed
    stream1_changed = not torch.allclose(model.conv1.stream1_weight, initial_stream1_weight)
    stream2_changed = not torch.allclose(model.conv1.stream2_weight, initial_stream2_weight)

    assert stream1_changed, "Stream1 weights should have changed"
    assert stream2_changed, "Stream2 weights should have changed"

    print(f"✓ Stream1 weights updated")
    print(f"✓ Stream2 weights updated")
    print(f"✓ Loss: {loss.item():.4f}")

    print("\n7. Optimizer State Verification")
    print("-" * 70)

    # Check optimizer state dict
    state_dict = model.optimizer.state_dict()
    print(f"✓ Optimizer state has {len(state_dict['param_groups'])} parameter groups")

    for i, pg in enumerate(state_dict['param_groups']):
        print(f"  Group {i}: lr={pg['lr']:.1e}, weight_decay={pg['weight_decay']:.1e}")

    print("\n" + "=" * 70)
    print("✅ All stream-specific optimization tests passed!")
    print("=" * 70)

    # Summary
    print("\nSummary:")
    print("-" * 70)
    print("Stream-specific optimization allows you to:")
    print("  • Set different learning rates for stream1 and stream2 pathways")
    print("  • Set different weight decay for stream1 and stream2 pathways")
    print("  • Boost weaker stream (RGB) with higher LR")
    print("  • Regularize stronger stream (depth) with higher weight decay")
    print("\nUsage:")
    print("  model.compile(")
    print("      optimizer='adamw',")
    print("      learning_rate=1e-3,      # Base LR for shared params")
    print("      weight_decay=2e-2,       # Base WD for shared params")
    print("      stream1_lr=2e-4,         # RGB pathway LR")
    print("      stream2_lr=5e-4,         # Depth pathway LR")
    print("      stream1_weight_decay=1e-3,  # RGB pathway WD")
    print("      stream2_weight_decay=4e-2   # Depth pathway WD")
    print("  )")

if __name__ == "__main__":
    test_stream_specific_optimization()
