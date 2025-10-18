"""
Debug script to investigate NaN losses in ResNet50 training.
Tests for numerical instability, gradient explosion, and AMP issues.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import numpy as np
from src.models.linear_integration.li_net import li_resnet50


def check_for_nans(model, name=""):
    """Check model for NaN values in parameters and gradients."""
    has_nan = False
    for param_name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"  ❌ NaN in {name} parameter: {param_name}")
            has_nan = True
        if param.grad is not None and torch.isnan(param.grad).any():
            print(f"  ❌ NaN in {name} gradient: {param_name}")
            has_nan = True
    return has_nan


def test_resnet50_numerical_stability():
    """Test ResNet50 for numerical stability issues."""
    print("=" * 80)
    print("RESNET50 NUMERICAL STABILITY TEST")
    print("=" * 80)

    # Create model
    print("\n1. Creating ResNet50 model...")
    model = li_resnet50(
        num_classes=15,
        stream1_input_channels=3,
        stream2_input_channels=1,
        dropout_p=0.5,
        device='cpu',  # Use CPU for easier debugging
        use_amp=False  # Disable AMP for now
    )
    model.train()
    print("   ✅ Model created")

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-5, weight_decay=2e-4)
    criterion = nn.CrossEntropyLoss()

    # Create synthetic data (similar to SUN RGB-D)
    print("\n2. Creating synthetic test data...")
    batch_size = 8  # Small batch for testing
    rgb = torch.randn(batch_size, 3, 416, 544)
    depth = torch.randn(batch_size, 1, 416, 544)
    labels = torch.randint(0, 15, (batch_size,))

    print(f"   RGB shape: {rgb.shape}, range: [{rgb.min():.3f}, {rgb.max():.3f}]")
    print(f"   Depth shape: {depth.shape}, range: [{depth.min():.3f}, {depth.max():.3f}]")
    print(f"   Labels: {labels.tolist()}")

    # Test forward pass
    print("\n3. Testing forward pass...")
    try:
        outputs = model(rgb, depth)
        print(f"   ✅ Forward pass successful")
        print(f"   Output shape: {outputs.shape}")
        print(f"   Output range: [{outputs.min():.3f}, {outputs.max():.3f}]")
        print(f"   Output has NaN: {torch.isnan(outputs).any()}")
        print(f"   Output has Inf: {torch.isinf(outputs).any()}")

        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
            print("\n   ❌ FOUND NaN/Inf IN FORWARD PASS!")
            return False

    except Exception as e:
        print(f"   ❌ Forward pass failed: {e}")
        return False

    # Test backward pass
    print("\n4. Testing backward pass...")
    try:
        loss = criterion(outputs, labels)
        print(f"   Loss: {loss.item():.6f}")

        if torch.isnan(loss):
            print("   ❌ Loss is NaN!")
            return False

        optimizer.zero_grad()
        loss.backward()

        # Check gradients
        has_nan = check_for_nans(model, "after first backward")
        if has_nan:
            print("\n   ❌ FOUND NaN IN GRADIENTS!")
            return False
        else:
            print("   ✅ No NaN in gradients")

        # Check gradient magnitudes
        max_grad = 0
        for param in model.parameters():
            if param.grad is not None:
                max_grad = max(max_grad, param.grad.abs().max().item())
        print(f"   Max gradient magnitude: {max_grad:.6f}")

        if max_grad > 1000:
            print(f"   ⚠️  WARNING: Very large gradients detected!")

        optimizer.step()
        print("   ✅ Backward pass and optimizer step successful")

    except Exception as e:
        print(f"   ❌ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test multiple iterations
    print("\n5. Testing 100 training iterations...")
    for i in range(100):
        # Generate new random data each iteration
        rgb = torch.randn(batch_size, 3, 416, 544)
        depth = torch.randn(batch_size, 1, 416, 544)
        labels = torch.randint(0, 15, (batch_size,))

        outputs = model(rgb, depth)
        loss = criterion(outputs, labels)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n   ❌ NaN/Inf at iteration {i+1}!")
            print(f"   Loss: {loss.item()}")
            return False

        optimizer.zero_grad()
        loss.backward()

        # Check for NaN in gradients
        has_nan = False
        for param in model.parameters():
            if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                has_nan = True
                break

        if has_nan:
            print(f"\n   ❌ NaN/Inf in gradients at iteration {i+1}!")
            return False

        optimizer.step()

        if (i + 1) % 20 == 0:
            print(f"   Iteration {i+1}/100: loss={loss.item():.6f}")

    print("\n   ✅ All 100 iterations completed without NaN!")

    return True


def test_with_real_data_distribution():
    """Test with data distribution similar to actual training."""
    print("\n" + "=" * 80)
    print("TESTING WITH REALISTIC DATA DISTRIBUTION")
    print("=" * 80)

    print("\nCreating model...")
    model = li_resnet50(
        num_classes=15,
        stream1_input_channels=3,
        stream2_input_channels=1,
        dropout_p=0.5,
        device='cpu',
        use_amp=False
    )
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-5, weight_decay=2e-4)
    criterion = nn.CrossEntropyLoss()

    batch_size = 8

    # Use ImageNet normalization statistics (similar to actual data)
    print("\nUsing ImageNet-normalized data distribution...")
    for i in range(50):
        # RGB: ImageNet normalized
        rgb = torch.randn(batch_size, 3, 416, 544) * 0.225 + 0.456  # Approximate ImageNet stats

        # Depth: normalized to [0, 1] range
        depth = torch.rand(batch_size, 1, 416, 544)

        labels = torch.randint(0, 15, (batch_size,))

        outputs = model(rgb, depth)
        loss = criterion(outputs, labels)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n❌ NaN/Inf at iteration {i+1} with realistic data!")
            print(f"Loss: {loss.item()}")
            print(f"RGB range: [{rgb.min():.3f}, {rgb.max():.3f}]")
            print(f"Depth range: [{depth.min():.3f}, {depth.max():.3f}]")
            print(f"Output range: [{outputs.min():.3f}, {outputs.max():.3f}]")
            return False

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print(f"Iteration {i+1}/50: loss={loss.item():.6f}")

    print("\n✅ All iterations completed successfully with realistic data!")
    return True


def test_with_amp():
    """Test with Automatic Mixed Precision (AMP)."""
    if not torch.cuda.is_available():
        print("\n⚠️  Skipping AMP test (no CUDA available)")
        return True

    print("\n" + "=" * 80)
    print("TESTING WITH AUTOMATIC MIXED PRECISION (AMP)")
    print("=" * 80)

    print("\nCreating model on CUDA...")
    model = li_resnet50(
        num_classes=15,
        stream1_input_channels=3,
        stream2_input_channels=1,
        dropout_p=0.5,
        device='cuda',
        use_amp=True
    )
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-5, weight_decay=2e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    batch_size = 8

    print("\nRunning 50 iterations with AMP...")
    for i in range(50):
        rgb = (torch.randn(batch_size, 3, 416, 544) * 0.225 + 0.456).cuda()
        depth = torch.rand(batch_size, 1, 416, 544).cuda()
        labels = torch.randint(0, 15, (batch_size,)).cuda()

        with torch.cuda.amp.autocast():
            outputs = model(rgb, depth)
            loss = criterion(outputs, labels)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n❌ NaN/Inf at iteration {i+1} with AMP!")
            print(f"Loss: {loss.item()}")
            return False

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if (i + 1) % 10 == 0:
            print(f"Iteration {i+1}/50: loss={loss.item():.6f}")

    print("\n✅ All AMP iterations completed successfully!")
    return True


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("RESNET50 NaN DEBUGGING TEST SUITE")
    print("=" * 80)

    all_passed = True

    # Test 1: Basic numerical stability
    if not test_resnet50_numerical_stability():
        print("\n❌ Basic numerical stability test FAILED")
        all_passed = False

    # Test 2: Realistic data distribution
    if not test_with_real_data_distribution():
        print("\n❌ Realistic data test FAILED")
        all_passed = False

    # Test 3: AMP
    if not test_with_amp():
        print("\n❌ AMP test FAILED")
        all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("✅ ALL TESTS PASSED - No NaN issues detected!")
        print("\nConclusion: ResNet50 architecture is numerically stable.")
        print("NaN issue likely caused by:")
        print("  1. Corrupted/invalid data in actual dataset")
        print("  2. Extreme augmentation producing invalid values")
        print("  3. Learning rate scheduler causing LR spike")
        print("  4. Batch with all-same labels causing division by zero")
    else:
        print("❌ SOME TESTS FAILED - NaN issues detected!")
        print("\nInvestigate the failing test above for root cause.")
    print("=" * 80)
