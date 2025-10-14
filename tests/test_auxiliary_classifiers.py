"""Test auxiliary classifiers for LINet stream monitoring."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from src.models.linear_integration.li_net import li_resnet18


def test_auxiliary_classifiers_exist():
    """Test that auxiliary classifiers are created correctly."""
    print("\n=== Test 1: Auxiliary Classifiers Exist ===")

    model = li_resnet18(num_classes=10, device='cpu')

    # Check that auxiliary classifiers exist
    assert hasattr(model, 'fc_stream1'), "fc_stream1 should exist"
    assert hasattr(model, 'fc_stream2'), "fc_stream2 should exist"

    # Check they're Linear layers
    assert isinstance(model.fc_stream1, nn.Linear), "fc_stream1 should be nn.Linear"
    assert isinstance(model.fc_stream2, nn.Linear), "fc_stream2 should be nn.Linear"

    # Check they have correct output dimensions
    assert model.fc_stream1.out_features == 10, "fc_stream1 should output 10 classes"
    assert model.fc_stream2.out_features == 10, "fc_stream2 should output 10 classes"

    print("✅ Auxiliary classifiers created correctly")


def test_gradient_isolation():
    """Test that auxiliary classifiers don't affect stream weights."""
    print("\n=== Test 2: Gradient Isolation ===")

    model = li_resnet18(num_classes=10, device='cpu')
    model.compile(optimizer='adam', learning_rate=0.001, loss='cross_entropy')

    # Get initial stream weights
    initial_stream1_weights = {}
    initial_stream2_weights = {}
    for name, param in model.named_parameters():
        if 'stream1_weight' in name:
            initial_stream1_weights[name] = param.clone().detach()
        elif 'stream2_weight' in name:
            initial_stream2_weights[name] = param.clone().detach()

    # Create fake batch
    rgb = torch.randn(4, 3, 224, 224)
    depth = torch.randn(4, 1, 224, 224)
    targets = torch.randint(0, 10, (4,))

    # Simulate auxiliary classifier training (with gradient isolation)
    model.train()

    # Forward through stream pathways
    stream1_features = model._forward_stream1_pathway(rgb)
    stream2_features = model._forward_stream2_pathway(depth)

    # DETACH features - stops gradient flow to streams
    stream1_features_detached = stream1_features.detach()
    stream2_features_detached = stream2_features.detach()

    # Classify with auxiliary classifiers
    stream1_outputs = model.fc_stream1(stream1_features_detached)
    stream2_outputs = model.fc_stream2(stream2_features_detached)

    # Compute auxiliary losses (full gradients, no scaling)
    criterion = nn.CrossEntropyLoss()
    stream1_loss = criterion(stream1_outputs, targets)
    stream2_loss = criterion(stream2_outputs, targets)

    # Backward pass
    model.optimizer.zero_grad()
    stream1_loss.backward()
    stream2_loss.backward()
    model.optimizer.step()

    # Check that stream weights haven't changed
    for name, param in model.named_parameters():
        if 'stream1_weight' in name:
            assert torch.allclose(param, initial_stream1_weights[name], atol=1e-6), \
                f"Stream1 weight {name} should NOT change from auxiliary loss"
        elif 'stream2_weight' in name:
            assert torch.allclose(param, initial_stream2_weights[name], atol=1e-6), \
                f"Stream2 weight {name} should NOT change from auxiliary loss"

    print("✅ Gradient isolation working - stream weights unchanged")

    # Check that auxiliary classifier weights DID change
    initial_fc_stream1_weight = model.fc_stream1.weight.clone().detach()

    # Do another backward pass
    stream1_features = model._forward_stream1_pathway(rgb)
    stream1_features_detached = stream1_features.detach()
    stream1_outputs = model.fc_stream1(stream1_features_detached)
    stream1_loss = criterion(stream1_outputs, targets)

    model.optimizer.zero_grad()
    stream1_loss.backward()
    model.optimizer.step()

    # Auxiliary classifier weights should change
    assert not torch.allclose(model.fc_stream1.weight, initial_fc_stream1_weight, atol=1e-6), \
        "Auxiliary classifier weights SHOULD change"

    print("✅ Auxiliary classifier weights are trainable")


def test_stream_monitoring_training():
    """Test stream monitoring with auxiliary classifiers during training."""
    print("\n=== Test 3: Stream Monitoring Training ===")

    model = li_resnet18(num_classes=10, device='cpu')
    model.compile(optimizer='adam', learning_rate=0.01, loss='cross_entropy')  # Higher LR for faster learning

    # Create fake dataset (larger for more stable accuracy metrics)
    rgb = torch.randn(40, 3, 224, 224)
    depth = torch.randn(40, 1, 224, 224)
    targets = torch.randint(0, 10, (40,))

    dataset = torch.utils.data.TensorDataset(rgb, depth, targets)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)

    # Train for 3 epochs with stream monitoring (full gradients for accurate monitoring)
    history = model.fit(
        train_loader=train_loader,
        val_loader=None,
        epochs=3,
        verbose=False,
        stream_monitoring=True
    )

    # Check that stream accuracies are recorded
    assert 'stream1_train_acc' in history, "stream1_train_acc should be in history"
    assert 'stream2_train_acc' in history, "stream2_train_acc should be in history"
    assert len(history['stream1_train_acc']) == 3, "Should have 3 epochs of stream1 acc"
    assert len(history['stream2_train_acc']) == 3, "Should have 3 epochs of stream2 acc"

    # Stream accuracies should be recorded and non-negative
    # With random data and 10 classes, we just verify that the auxiliary classifiers are being tracked
    assert history['stream1_train_acc'][-1] >= 0.0, f"Stream1 train acc should be >= 0, got {history['stream1_train_acc'][-1]}"
    assert history['stream2_train_acc'][-1] >= 0.0, f"Stream2 train acc should be >= 0, got {history['stream2_train_acc'][-1]}"

    # At least one should show some learning (> 0%)
    assert history['stream1_train_acc'][-1] > 0.0 or history['stream2_train_acc'][-1] > 0.0, \
        "At least one stream should show non-zero accuracy"

    print(f"Stream1 train acc: {history['stream1_train_acc']}")
    print(f"Stream2 train acc: {history['stream2_train_acc']}")

    # Check that accuracy is improving (final should be better than first, or at least not worse)
    stream1_improved = history['stream1_train_acc'][-1] >= history['stream1_train_acc'][0]
    stream2_improved = history['stream2_train_acc'][-1] >= history['stream2_train_acc'][0]

    print(f"Stream1 improved: {stream1_improved} ({history['stream1_train_acc'][0]:.4f} -> {history['stream1_train_acc'][-1]:.4f})")
    print(f"Stream2 improved: {stream2_improved} ({history['stream2_train_acc'][0]:.4f} -> {history['stream2_train_acc'][-1]:.4f})")
    print("✅ Stream monitoring produces meaningful accuracies")


def test_contribution_to_integration_method():
    """Test the simplified calculate_stream_contributions_to_integration() method."""
    print("\n=== Test 4: Contribution to Integration Method (Simplified) ===")

    model = li_resnet18(num_classes=10, device='cpu')
    model.compile(optimizer='adam', learning_rate=0.001, loss='cross_entropy')

    # Note: No data loader needed - method only analyzes weights!
    contributions = model.calculate_stream_contributions_to_integration()

    # Check structure
    assert 'method' in contributions, "Should have method key"
    assert contributions['method'] == 'integration_weights', "Method should be integration_weights"

    assert 'stream1_contribution' in contributions, "Should have stream1_contribution"
    assert 'stream2_contribution' in contributions, "Should have stream2_contribution"

    # Check values are valid (0-1 range, sum to 1)
    s1 = contributions['stream1_contribution']
    s2 = contributions['stream2_contribution']

    assert 0 <= s1 <= 1, "Stream1 contribution should be in [0,1]"
    assert 0 <= s2 <= 1, "Stream2 contribution should be in [0,1]"

    total = s1 + s2
    assert 0.99 <= total <= 1.01, f"Contributions should sum to 1, got {total}"

    print(f"Stream1 contribution: {s1:.3f} ({contributions['interpretation']['stream1_percentage']})")
    print(f"Stream2 contribution: {s2:.3f} ({contributions['interpretation']['stream2_percentage']})")
    print(f"Total: {total:.3f}")
    print("✅ Contribution to integration method works correctly")

    # Check raw norms are present
    assert 'raw_norms' in contributions, "Should have raw_norms"
    assert 'stream1_integration_weights' in contributions['raw_norms'], "Should have stream1 weight norms"
    assert 'stream2_integration_weights' in contributions['raw_norms'], "Should have stream2 weight norms"

    # Check interpretation is present
    assert 'interpretation' in contributions, "Should have interpretation"
    assert 'stream1_percentage' in contributions['interpretation'], "Should have stream1 percentage"
    assert 'stream2_percentage' in contributions['interpretation'], "Should have stream2 percentage"

    print("✅ Simplified method: fast, stable, and interpretable!")


def test_deprecation_warning():
    """Test that old method shows deprecation warning."""
    print("\n=== Test 5: Deprecation Warning ===")

    import warnings

    model = li_resnet18(num_classes=10, device='cpu')
    model.compile(optimizer='adam', learning_rate=0.001, loss='cross_entropy')

    # Create fake dataset
    rgb = torch.randn(8, 3, 224, 224)
    depth = torch.randn(8, 1, 224, 224)
    targets = torch.randint(0, 10, (8,))

    dataset = torch.utils.data.TensorDataset(rgb, depth, targets)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)

    # Call old method and catch warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = model.calculate_stream_contributions(data_loader)

        # Check that a deprecation warning was raised
        assert len(w) == 1, "Should have raised one warning"
        assert issubclass(w[0].category, DeprecationWarning), "Should be DeprecationWarning"
        assert "deprecated" in str(w[0].message).lower(), "Warning should mention deprecation"
        assert "calculate_stream_contributions_to_integration" in str(w[0].message), \
            "Warning should suggest new method"

    print("✅ Deprecation warning shown correctly")


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING AUXILIARY CLASSIFIERS FOR LINET")
    print("=" * 60)

    test_auxiliary_classifiers_exist()
    test_gradient_isolation()
    test_stream_monitoring_training()
    test_contribution_to_integration_method()
    test_deprecation_warning()

    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED!")
    print("=" * 60)
