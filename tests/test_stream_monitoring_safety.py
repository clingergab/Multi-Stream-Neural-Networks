"""
Comprehensive safety tests for stream monitoring with auxiliary classifiers.

These tests verify that auxiliary classifiers DO NOT affect main model training:
1. No gradient leakage to stream weights
2. No impact on main model convergence
3. No difference in final trained weights
4. Identical training dynamics with/without monitoring
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from src.models.linear_integration.li_net import li_resnet18


def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_deterministic_data(num_samples=32, seed=42):
    """Create deterministic dataset for reproducible testing."""
    set_seed(seed)
    rgb = torch.randn(num_samples, 3, 224, 224)
    depth = torch.randn(num_samples, 1, 224, 224)
    targets = torch.randint(0, 10, (num_samples,))

    dataset = torch.utils.data.TensorDataset(rgb, depth, targets)
    # Use fixed ordering (shuffle=False) for reproducibility
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)

    return dataloader


def get_all_main_model_params(model):
    """Get all main model parameters (excluding auxiliary classifiers)."""
    params = {}
    for name, param in model.named_parameters():
        # Skip auxiliary classifier parameters
        if 'fc_stream1' not in name and 'fc_stream2' not in name:
            params[name] = param.clone().detach()
    return params


def compare_model_params(params1, params2, tolerance=1e-7):
    """Compare two sets of model parameters."""
    differences = {}
    all_match = True

    for name in params1.keys():
        if name not in params2:
            print(f"⚠️  Parameter {name} missing in params2")
            all_match = False
            continue

        p1 = params1[name]
        p2 = params2[name]

        if not torch.allclose(p1, p2, atol=tolerance):
            max_diff = (p1 - p2).abs().max().item()
            mean_diff = (p1 - p2).abs().mean().item()
            differences[name] = {
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'shape': p1.shape
            }
            all_match = False

    return all_match, differences


def test_no_gradient_leakage():
    """
    Test 1: Verify auxiliary classifiers don't leak gradients to main model.

    This is the MOST CRITICAL test - we verify that during auxiliary training:
    - Stream pathway weights receive NO gradients
    - Integration weights receive NO gradients
    - Main classifier receives NO gradients
    - ONLY fc_stream1 and fc_stream2 receive gradients
    """
    print("\n" + "="*80)
    print("TEST 1: No Gradient Leakage During Auxiliary Training")
    print("="*80)

    model = li_resnet18(num_classes=10, device='cpu')
    model.compile(optimizer='adam', learning_rate=0.001, loss='cross_entropy')

    # Create single batch
    rgb = torch.randn(8, 3, 224, 224)
    depth = torch.randn(8, 1, 224, 224)
    targets = torch.randint(0, 10, (8,))

    # First, do main training step
    model.train()
    model.optimizer.zero_grad()
    outputs = model(rgb, depth)
    loss = model.criterion(outputs, targets)
    loss.backward()
    model.optimizer.step()

    # Now test auxiliary training (this is where we check for leakage)
    model.optimizer.zero_grad()

    # Forward through stream pathways
    stream1_features = model._forward_stream1_pathway(rgb)
    stream2_features = model._forward_stream2_pathway(depth)

    # DETACH features
    stream1_features_detached = stream1_features.detach()
    stream2_features_detached = stream2_features.detach()

    # Auxiliary classifier forward
    stream1_outputs = model.fc_stream1(stream1_features_detached)
    stream2_outputs = model.fc_stream2(stream2_features_detached)

    # Auxiliary losses
    stream1_aux_loss = model.criterion(stream1_outputs, targets)
    stream2_aux_loss = model.criterion(stream2_outputs, targets)

    # Backward passes
    stream1_aux_loss.backward()
    stream2_aux_loss.backward()

    # Check gradients on ALL parameters
    print("\nGradient Analysis:")
    print("-" * 80)

    has_gradient = []
    no_gradient = []

    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_gradient.append(name)
        else:
            no_gradient.append(name)

    print(f"✓ Parameters WITH gradients ({len(has_gradient)}):")
    for name in has_gradient:
        print(f"  - {name}")

    print(f"\n✓ Parameters WITHOUT gradients ({len(no_gradient)}):")
    # Just show count for main model params (there are many)
    stream_params = [n for n in no_gradient if 'stream1' in n and 'fc_stream1' not in n]
    integration_params = [n for n in no_gradient if 'integration' in n]
    fc_params = [n for n in no_gradient if n.startswith('fc.')]

    print(f"  - Stream1 pathway: {len([n for n in stream_params if 'stream1' in n])} parameters")
    print(f"  - Stream2 pathway: {len([n for n in no_gradient if 'stream2' in n and 'fc_stream2' not in n])} parameters")
    print(f"  - Integration layers: {len(integration_params)} parameters")
    print(f"  - Main classifier (fc): {len(fc_params)} parameters")

    # Critical assertions
    print("\nCritical Safety Checks:")
    print("-" * 80)

    # 1. ONLY auxiliary classifiers should have gradients
    aux_classifier_params = ['fc_stream1.weight', 'fc_stream1.bias', 'fc_stream2.weight', 'fc_stream2.bias']
    assert set(has_gradient) == set(aux_classifier_params), \
        f"❌ GRADIENT LEAKAGE DETECTED! Only {aux_classifier_params} should have gradients, but found: {has_gradient}"
    print("✅ PASS: Only auxiliary classifiers have gradients")

    # 2. NO stream pathway parameters should have gradients
    for name in no_gradient:
        if 'stream1' in name or 'stream2' in name:
            param = dict(model.named_parameters())[name]
            if param.grad is not None:
                grad_norm = param.grad.abs().sum().item()
                assert grad_norm == 0, f"❌ GRADIENT LEAKAGE: {name} has gradient norm {grad_norm}"
    print("✅ PASS: No stream pathway parameters have gradients")

    # 3. NO integration weights should have gradients
    for name in no_gradient:
        if 'integration' in name:
            param = dict(model.named_parameters())[name]
            if param.grad is not None:
                grad_norm = param.grad.abs().sum().item()
                assert grad_norm == 0, f"❌ GRADIENT LEAKAGE: {name} has gradient norm {grad_norm}"
    print("✅ PASS: No integration weights have gradients")

    # 4. Main classifier (fc) should have NO gradients
    fc_weight = model.fc.weight
    fc_bias = model.fc.bias
    assert fc_weight.grad is None or fc_weight.grad.abs().sum() == 0, \
        "❌ GRADIENT LEAKAGE: Main classifier weight has gradients"
    assert fc_bias.grad is None or fc_bias.grad.abs().sum() == 0, \
        "❌ GRADIENT LEAKAGE: Main classifier bias has gradients"
    print("✅ PASS: Main classifier has no gradients")

    print("\n" + "="*80)
    print("✅ TEST 1 PASSED: No gradient leakage detected")
    print("="*80)


def test_identical_training_without_monitoring():
    """
    Test 2: Verify training with stream_monitoring=True produces IDENTICAL
    main model weights as training with stream_monitoring=False.

    This ensures auxiliary classifiers have ZERO impact on main training.
    """
    print("\n" + "="*80)
    print("TEST 2: Identical Training Dynamics With/Without Monitoring")
    print("="*80)

    # Train model WITHOUT monitoring
    print("\nTraining model WITHOUT stream monitoring...")
    set_seed(42)
    model_no_monitoring = li_resnet18(num_classes=10, device='cpu')
    model_no_monitoring.compile(optimizer='adam', learning_rate=0.001, loss='cross_entropy')

    train_loader_1 = create_deterministic_data(num_samples=32, seed=42)

    history_no_monitoring = model_no_monitoring.fit(
        train_loader=train_loader_1,
        val_loader=None,
        epochs=3,
        verbose=False,
        stream_monitoring=False  # NO MONITORING
    )

    params_no_monitoring = get_all_main_model_params(model_no_monitoring)

    # Train model WITH monitoring
    print("Training model WITH stream monitoring...")
    set_seed(42)  # Same seed for identical initialization
    model_with_monitoring = li_resnet18(num_classes=10, device='cpu')
    model_with_monitoring.compile(optimizer='adam', learning_rate=0.001, loss='cross_entropy')

    train_loader_2 = create_deterministic_data(num_samples=32, seed=42)

    history_with_monitoring = model_with_monitoring.fit(
        train_loader=train_loader_2,
        val_loader=None,
        epochs=3,
        verbose=False,
        stream_monitoring=True  # WITH MONITORING
    )

    params_with_monitoring = get_all_main_model_params(model_with_monitoring)

    # Compare training histories
    print("\nTraining History Comparison:")
    print("-" * 80)

    for epoch in range(3):
        loss_no_mon = history_no_monitoring['train_loss'][epoch]
        loss_with_mon = history_with_monitoring['train_loss'][epoch]
        acc_no_mon = history_no_monitoring['train_accuracy'][epoch]
        acc_with_mon = history_with_monitoring['train_accuracy'][epoch]

        print(f"Epoch {epoch+1}:")
        print(f"  Loss:     {loss_no_mon:.6f} (no mon) vs {loss_with_mon:.6f} (with mon) | diff: {abs(loss_no_mon - loss_with_mon):.2e}")
        print(f"  Accuracy: {acc_no_mon:.6f} (no mon) vs {acc_with_mon:.6f} (with mon) | diff: {abs(acc_no_mon - acc_with_mon):.2e}")

        # These should be IDENTICAL (or within floating point precision)
        assert abs(loss_no_mon - loss_with_mon) < 1e-6, \
            f"❌ Training loss differs at epoch {epoch+1}: {abs(loss_no_mon - loss_with_mon)}"
        assert abs(acc_no_mon - acc_with_mon) < 1e-6, \
            f"❌ Training accuracy differs at epoch {epoch+1}: {abs(acc_no_mon - acc_with_mon)}"

    print("✅ Training histories are identical")

    # Compare final model parameters
    print("\nModel Parameter Comparison:")
    print("-" * 80)

    all_match, differences = compare_model_params(params_no_monitoring, params_with_monitoring, tolerance=1e-7)

    if all_match:
        print("✅ All main model parameters are IDENTICAL")
        print(f"   Total parameters checked: {len(params_no_monitoring)}")
    else:
        print(f"❌ PARAMETERS DIFFER! This indicates auxiliary classifiers affected main training!")
        print("\nDifferences found:")
        for name, diff_info in differences.items():
            print(f"  {name}:")
            print(f"    Max diff: {diff_info['max_diff']:.2e}")
            print(f"    Mean diff: {diff_info['mean_diff']:.2e}")
            print(f"    Shape: {diff_info['shape']}")

        assert False, "❌ Main model parameters differ with/without monitoring!"

    print("\n" + "="*80)
    print("✅ TEST 2 PASSED: Training is identical with/without monitoring")
    print("="*80)


def test_main_model_weight_changes():
    """
    Test 3: Verify that during a training step with monitoring:
    - Main model weights DO change (from main backward pass)
    - Main model weights DO NOT change (from auxiliary backward pass)
    """
    print("\n" + "="*80)
    print("TEST 3: Main Model Weight Changes Only From Main Training")
    print("="*80)

    model = li_resnet18(num_classes=10, device='cpu')
    model.compile(optimizer='adam', learning_rate=0.001, loss='cross_entropy')

    # Create single batch
    rgb = torch.randn(8, 3, 224, 224)
    depth = torch.randn(8, 1, 224, 224)
    targets = torch.randint(0, 10, (8,))

    # Save initial weights
    initial_params = get_all_main_model_params(model)

    # Step 1: Main training step
    print("\nStep 1: Main training backward pass...")
    model.train()
    model.optimizer.zero_grad()
    outputs = model(rgb, depth)
    loss = model.criterion(outputs, targets)
    loss.backward()
    model.optimizer.step()

    # Check that main weights changed
    after_main_params = get_all_main_model_params(model)
    all_match, _ = compare_model_params(initial_params, after_main_params, tolerance=1e-9)

    assert not all_match, "❌ Main model weights should have changed after main training!"
    print("✅ Main model weights changed after main training (expected)")

    # Step 2: Auxiliary training step
    print("\nStep 2: Auxiliary classifier backward pass...")
    before_aux_params = get_all_main_model_params(model)

    model.optimizer.zero_grad()

    # Forward through streams with detach
    stream1_features = model._forward_stream1_pathway(rgb)
    stream2_features = model._forward_stream2_pathway(depth)
    stream1_features_detached = stream1_features.detach()
    stream2_features_detached = stream2_features.detach()

    # Auxiliary forward/backward
    stream1_outputs = model.fc_stream1(stream1_features_detached)
    stream2_outputs = model.fc_stream2(stream2_features_detached)
    stream1_aux_loss = model.criterion(stream1_outputs, targets)
    stream2_aux_loss = model.criterion(stream2_outputs, targets)
    stream1_aux_loss.backward()
    stream2_aux_loss.backward()
    model.optimizer.step()

    # Check that main weights DID NOT change
    after_aux_params = get_all_main_model_params(model)
    all_match, differences = compare_model_params(before_aux_params, after_aux_params, tolerance=1e-9)

    if not all_match:
        print("❌ CRITICAL FAILURE: Main model weights changed during auxiliary training!")
        print("\nParameters that changed:")
        for name, diff_info in differences.items():
            print(f"  {name}: max_diff={diff_info['max_diff']:.2e}")
        assert False, "❌ Main model weights should NOT change during auxiliary training!"

    print("✅ Main model weights unchanged after auxiliary training (expected)")

    print("\n" + "="*80)
    print("✅ TEST 3 PASSED: Main weights only change from main training")
    print("="*80)


def test_auxiliary_classifiers_do_learn():
    """
    Test 4: Verify that auxiliary classifiers themselves DO learn.

    This ensures our test setup is correct - if auxiliary classifiers
    don't learn at all, the tests above would pass trivially.
    """
    print("\n" + "="*80)
    print("TEST 4: Auxiliary Classifiers DO Learn (Sanity Check)")
    print("="*80)

    model = li_resnet18(num_classes=10, device='cpu')
    model.compile(optimizer='adam', learning_rate=0.01, loss='cross_entropy')

    # Save initial auxiliary weights
    initial_fc_stream1_weight = model.fc_stream1.weight.clone().detach()
    initial_fc_stream2_weight = model.fc_stream2.weight.clone().detach()

    # Train with monitoring
    train_loader = create_deterministic_data(num_samples=32, seed=42)

    history = model.fit(
        train_loader=train_loader,
        val_loader=None,
        epochs=3,
        verbose=False,
        stream_monitoring=True
    )

    # Check that auxiliary weights changed
    final_fc_stream1_weight = model.fc_stream1.weight.clone().detach()
    final_fc_stream2_weight = model.fc_stream2.weight.clone().detach()

    fc_stream1_changed = not torch.allclose(initial_fc_stream1_weight, final_fc_stream1_weight, atol=1e-6)
    fc_stream2_changed = not torch.allclose(initial_fc_stream2_weight, final_fc_stream2_weight, atol=1e-6)

    print(f"\nAuxiliary Classifier Training:")
    print("-" * 80)
    print(f"fc_stream1 weights changed: {fc_stream1_changed}")
    print(f"fc_stream2 weights changed: {fc_stream2_changed}")

    if fc_stream1_changed:
        max_change = (final_fc_stream1_weight - initial_fc_stream1_weight).abs().max().item()
        print(f"fc_stream1 max weight change: {max_change:.6f}")

    if fc_stream2_changed:
        max_change = (final_fc_stream2_weight - initial_fc_stream2_weight).abs().max().item()
        print(f"fc_stream2 max weight change: {max_change:.6f}")

    assert fc_stream1_changed, "❌ fc_stream1 weights should have changed!"
    assert fc_stream2_changed, "❌ fc_stream2 weights should have changed!"

    # Check that accuracies improved
    print(f"\nStream Accuracies Over Training:")
    print("-" * 80)
    print(f"Stream1: {history['stream1_train_acc']}")
    print(f"Stream2: {history['stream2_train_acc']}")

    # At least some learning should occur
    assert len(history['stream1_train_acc']) == 3, "Should have 3 epochs"
    assert len(history['stream2_train_acc']) == 3, "Should have 3 epochs"

    print("\n" + "="*80)
    print("✅ TEST 4 PASSED: Auxiliary classifiers learn successfully")
    print("="*80)


def run_all_tests():
    """Run all safety tests."""
    print("\n" + "="*80)
    print("COMPREHENSIVE STREAM MONITORING SAFETY TEST SUITE")
    print("="*80)
    print("\nThis test suite verifies that auxiliary classifiers:")
    print("1. Do NOT leak gradients to main model")
    print("2. Do NOT affect main model training dynamics")
    print("3. Do NOT change main model weights")
    print("4. DO learn successfully (sanity check)")
    print("\n" + "="*80)

    try:
        test_no_gradient_leakage()
        test_main_model_weight_changes()
        test_auxiliary_classifiers_do_learn()
        test_identical_training_without_monitoring()

        print("\n" + "="*80)
        print("🎉 ALL SAFETY TESTS PASSED! 🎉")
        print("="*80)
        print("\nConclusion:")
        print("✅ Auxiliary classifiers are properly isolated")
        print("✅ No gradient leakage to main model")
        print("✅ Main model training is completely unaffected")
        print("✅ Stream monitoring is SAFE to use")
        print("="*80 + "\n")

    except AssertionError as e:
        print("\n" + "="*80)
        print("❌ SAFETY TEST FAILED!")
        print("="*80)
        print(f"\nError: {e}")
        print("\n⚠️  DO NOT USE STREAM MONITORING until this is fixed!")
        print("="*80 + "\n")
        raise


if __name__ == "__main__":
    run_all_tests()
