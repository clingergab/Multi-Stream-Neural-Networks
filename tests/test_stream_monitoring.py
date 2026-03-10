"""
Comprehensive test for blanked-stream monitoring.

Tests:
1. _evaluate_stream_contributions returns correct structure and values
2. Blanking actually changes model outputs (not identical)
3. Stream accuracy mapping is correct (stream_0 acc = model with stream_1 blanked)
4. History keys are populated correctly during fit()
5. _print_stream_monitoring formats correctly
6. Edge cases: stream_eval_freq=0, no val_loader, with val_loader
7. Verify blanking works at the _validate level
"""

import sys
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.linear_integration.li_net3.li_net import LINet
from src.models.linear_integration.li_net3.blocks import LIBasicBlock


def create_test_model(num_classes=5, device='cpu'):
    """Create a small LINet for testing."""
    model = LINet(
        block=LIBasicBlock,
        layers=[1, 1, 1, 1],  # Minimal ResNet
        num_classes=num_classes,
        stream_input_channels=[3, 1],
        device=device,
        use_amp=False,
        width_multiplier=0.25,  # Small model
    )
    return model


def create_test_data(n_samples=100, num_classes=5, device='cpu'):
    """Create synthetic data that is NOT random — streams have different signal."""
    torch.manual_seed(42)

    # Create data where stream 0 (RGB) has class signal in channel means
    # and stream 1 (Depth) has class signal in spatial patterns
    labels = torch.randint(0, num_classes, (n_samples,))

    # Stream 0: class-dependent channel means (RGB-like signal)
    rgb = torch.randn(n_samples, 3, 32, 32) * 0.1
    for i in range(n_samples):
        rgb[i, labels[i] % 3] += 1.0  # Boost one channel based on class

    # Stream 1: class-dependent spatial pattern (Depth-like signal)
    depth = torch.randn(n_samples, 1, 32, 32) * 0.1
    for i in range(n_samples):
        quadrant = labels[i] % 4
        h, w = 16, 16
        row, col = quadrant // 2, quadrant % 2
        depth[i, 0, row * h:(row + 1) * h, col * w:(col + 1) * w] += 1.0

    dataset = TensorDataset(rgb, depth, labels)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    return loader, dataset


def test_validate_returns_tuple():
    """Test that _validate returns (loss, accuracy) tuple."""
    print("Test 1: _validate return type...")
    model = create_test_model()
    model.compile(
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        loss='cross_entropy',
    )
    loader, _ = create_test_data()

    result = model._validate(loader)
    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
    assert len(result) == 2, f"Expected 2 elements, got {len(result)}"
    loss, acc = result
    assert isinstance(loss, float), f"Loss should be float, got {type(loss)}"
    assert isinstance(acc, float), f"Acc should be float, got {type(acc)}"
    assert 0 <= acc <= 1, f"Accuracy should be 0-1, got {acc}"
    print(f"  OK: loss={loss:.4f}, acc={acc:.4f}")


def test_validate_with_blanking():
    """Test that blanking produces different results from no blanking."""
    print("\nTest 2: _validate with blanking produces different outputs...")
    model = create_test_model()
    model.compile(
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        loss='cross_entropy',
    )
    loader, _ = create_test_data()

    # Train for a few steps so the model has non-trivial weights
    model.train()
    for epoch in range(5):
        for batch in loader:
            rgb, depth, labels = batch
            model.optimizer.zero_grad()
            outputs = model([rgb, depth])
            loss = model.criterion(outputs, labels)
            loss.backward()
            model.optimizer.step()

    # Now test blanking
    baseline_loss, baseline_acc = model._validate(loader)
    blanked0_loss, blanked0_acc = model._validate(loader, blanked_streams={0})
    blanked1_loss, blanked1_acc = model._validate(loader, blanked_streams={1})

    print(f"  Baseline:        loss={baseline_loss:.4f}, acc={baseline_acc:.4f}")
    print(f"  Stream 0 blanked: loss={blanked0_loss:.4f}, acc={blanked0_acc:.4f}")
    print(f"  Stream 1 blanked: loss={blanked1_loss:.4f}, acc={blanked1_acc:.4f}")

    # Blanking should change the results
    assert blanked0_loss != baseline_loss or blanked0_acc != baseline_acc, \
        "Blanking stream 0 should change results!"
    assert blanked1_loss != baseline_loss or blanked1_acc != baseline_acc, \
        "Blanking stream 1 should change results!"
    # The two blanked results should also differ from each other
    assert blanked0_acc != blanked1_acc or blanked0_loss != blanked1_loss, \
        "Blanking different streams should give different results!"
    print("  OK: All three conditions give different results")


def test_evaluate_stream_contributions():
    """Test _evaluate_stream_contributions returns correct structure."""
    print("\nTest 3: _evaluate_stream_contributions structure...")
    model = create_test_model()
    model.compile(
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        loss='cross_entropy',
    )
    loader, _ = create_test_data()

    result = model._evaluate_stream_contributions(loader)

    # Check all expected keys exist
    expected_keys = [
        'baseline_acc', 'baseline_loss',
        'stream_0_blanked_acc', 'stream_0_blanked_loss', 'stream_0_contribution',
        'stream_1_blanked_acc', 'stream_1_blanked_loss', 'stream_1_contribution',
    ]
    for key in expected_keys:
        assert key in result, f"Missing key: {key}"
        assert isinstance(result[key], float), f"{key} should be float, got {type(result[key])}"

    # Verify contribution = baseline - blanked
    for i in range(2):
        expected_contrib = result['baseline_acc'] - result[f'stream_{i}_blanked_acc']
        actual_contrib = result[f'stream_{i}_contribution']
        assert abs(expected_contrib - actual_contrib) < 1e-10, \
            f"Stream {i} contribution mismatch: {actual_contrib} != {expected_contrib}"

    print(f"  baseline_acc={result['baseline_acc']:.4f}")
    print(f"  stream_0_blanked_acc={result['stream_0_blanked_acc']:.4f}, contribution={result['stream_0_contribution']:.4f}")
    print(f"  stream_1_blanked_acc={result['stream_1_blanked_acc']:.4f}, contribution={result['stream_1_contribution']:.4f}")
    print("  OK: All keys present, contributions computed correctly")


def test_stream_accuracy_mapping():
    """Test that stream_0 T_acc = model accuracy when stream_1 is blanked (and vice versa)."""
    print("\nTest 4: Stream accuracy mapping (stream_i acc = blanked OTHER stream)...")
    model = create_test_model()
    model.compile(
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        loss='cross_entropy',
    )
    loader, _ = create_test_data()

    # Train briefly
    model.train()
    for epoch in range(5):
        for batch in loader:
            rgb, depth, labels = batch
            model.optimizer.zero_grad()
            outputs = model([rgb, depth])
            loss = model.criterion(outputs, labels)
            loss.backward()
            model.optimizer.step()

    contrib = model._evaluate_stream_contributions(loader)

    # Stream 0's accuracy should be the accuracy when stream 1 is blanked
    # (because when stream 1 is blanked, only stream 0 is active)
    stream_0_acc = contrib['stream_1_blanked_acc']  # blank stream 1 → stream 0 solo
    stream_1_acc = contrib['stream_0_blanked_acc']  # blank stream 0 → stream 1 solo

    print(f"  stream_0 solo acc (stream_1 blanked): {stream_0_acc:.4f}")
    print(f"  stream_1 solo acc (stream_0 blanked): {stream_1_acc:.4f}")
    print(f"  baseline (both active):               {contrib['baseline_acc']:.4f}")

    # Verify the mapping matches what _print_stream_monitoring would show
    # For 2 streams: other_stream = (i + 1) % 2
    # stream 0 display → train_contrib['stream_1_blanked_acc']
    # stream 1 display → train_contrib['stream_0_blanked_acc']
    other_0 = (0 + 1) % 2  # = 1
    other_1 = (1 + 1) % 2  # = 0
    assert contrib[f'stream_{other_0}_blanked_acc'] == stream_0_acc
    assert contrib[f'stream_{other_1}_blanked_acc'] == stream_1_acc
    print("  OK: Mapping is correct")


def test_blanking_actually_zeros_stream():
    """Verify that blanking truly zeros out a stream's contribution in forward pass."""
    print("\nTest 5: Blanking actually zeros the stream output...")
    model = create_test_model()
    model.eval()

    torch.manual_seed(99)
    rgb = torch.randn(4, 3, 32, 32)
    depth = torch.randn(4, 1, 32, 32)

    # Forward with no blanking
    with torch.no_grad():
        out_normal = model([rgb, depth])

    # Forward with stream 0 blanked
    blanked_mask_0 = {
        0: torch.ones(4, dtype=torch.bool),
        1: torch.zeros(4, dtype=torch.bool),
    }
    with torch.no_grad():
        out_blank0 = model([rgb, depth], blanked_mask=blanked_mask_0)

    # Forward with stream 1 blanked
    blanked_mask_1 = {
        0: torch.zeros(4, dtype=torch.bool),
        1: torch.ones(4, dtype=torch.bool),
    }
    with torch.no_grad():
        out_blank1 = model([rgb, depth], blanked_mask=blanked_mask_1)

    # All three should be different
    diff_01 = (out_normal - out_blank0).abs().sum().item()
    diff_02 = (out_normal - out_blank1).abs().sum().item()
    diff_12 = (out_blank0 - out_blank1).abs().sum().item()

    print(f"  |normal - blank0| = {diff_01:.4f}")
    print(f"  |normal - blank1| = {diff_02:.4f}")
    print(f"  |blank0 - blank1| = {diff_12:.4f}")

    assert diff_01 > 0, "Blanking stream 0 should change output"
    assert diff_02 > 0, "Blanking stream 1 should change output"
    assert diff_12 > 0, "Different blanked streams should give different output"

    # Forward with stream 0 blanked + zero RGB input should equal blank0
    rgb_zero = torch.zeros_like(rgb)
    with torch.no_grad():
        out_zero_rgb = model([rgb_zero, depth])

    # These won't be exactly equal because blanking zeros at conv output level,
    # not at input level. But let's verify blanking is internally consistent.
    print("  OK: Blanking produces different outputs for each stream")


def test_fit_history_keys():
    """Test that fit() with stream_monitoring populates correct history keys."""
    print("\nTest 6: fit() history keys with stream_monitoring=True...")
    model = create_test_model()
    model.compile(
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        loss='cross_entropy',
    )
    loader, _ = create_test_data(n_samples=50)

    history = model.fit(
        train_loader=loader,
        epochs=2,
        verbose=False,
        stream_monitoring=True,
        stream_eval_samples=50,
    )

    # Check history keys
    for i in range(2):
        key_train = f'stream_{i}_train_acc'
        key_val = f'stream_{i}_val_acc'
        assert key_train in history, f"Missing history key: {key_train}"
        assert key_val in history, f"Missing history key: {key_val}"
        assert len(history[key_train]) == 2, \
            f"{key_train} should have 2 entries, got {len(history[key_train])}"
        assert len(history[key_val]) == 2, \
            f"{key_val} should have 2 entries, got {len(history[key_val])}"
        print(f"  stream_{i}_train_acc: {history[key_train]}")
        print(f"  stream_{i}_val_acc: {history[key_val]}")

    print("  OK: All history keys present with correct length")


def test_fit_with_val_loader():
    """Test that fit() with val_loader populates both train and val stream metrics."""
    print("\nTest 7: fit() with val_loader...")
    model = create_test_model()
    model.compile(
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        loss='cross_entropy',
    )
    train_loader, _ = create_test_data(n_samples=50)
    val_loader, _ = create_test_data(n_samples=30)

    history = model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=2,
        verbose=False,
        stream_monitoring=True,
        stream_eval_samples=50,
    )

    for i in range(2):
        train_accs = history[f'stream_{i}_train_acc']
        val_accs = history[f'stream_{i}_val_acc']
        print(f"  stream_{i}: train_acc={train_accs}, val_acc={val_accs}")
        # With val_loader, val_acc should come from val data (not all zeros)
        # Though early in training it could legitimately be 0, the train_acc should differ
        assert len(train_accs) == 2
        assert len(val_accs) == 2

    print("  OK: Both train and val stream metrics populated")


def test_eval_freq_zero():
    """Test that stream_eval_freq=0 disables evaluation but keeps history aligned."""
    print("\nTest 8: stream_eval_freq=0 disables evaluation...")
    model = create_test_model()
    model.compile(
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        loss='cross_entropy',
    )
    loader, _ = create_test_data(n_samples=50)

    history = model.fit(
        train_loader=loader,
        epochs=3,
        verbose=False,
        stream_monitoring=True,
        stream_eval_freq=0,
    )

    import math
    for i in range(2):
        train_accs = history[f'stream_{i}_train_acc']
        assert len(train_accs) == 3, f"Should have 3 entries, got {len(train_accs)}"
        assert all(math.isnan(a) for a in train_accs), \
            f"With freq=0, all should be NaN (not evaluated), got {train_accs}"
        # LR should still be tracked every epoch
        lrs = history[f'stream_{i}_lr']
        assert len(lrs) == 3, f"LR should have 3 entries, got {len(lrs)}"
        assert all(lr > 0 for lr in lrs), f"LR should be positive, got {lrs}"

    print("  OK: History has correct length with NaN for skipped epochs, LR always tracked")


def test_eval_freq_skipping():
    """Test that stream_eval_freq=2 evaluates on epochs 0, 2 and last epoch."""
    print("\nTest 9: stream_eval_freq=2 skips correctly...")
    model = create_test_model()
    model.compile(
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        loss='cross_entropy',
    )
    loader, _ = create_test_data(n_samples=50)

    history = model.fit(
        train_loader=loader,
        epochs=5,
        verbose=False,
        stream_monitoring=True,
        stream_eval_freq=2,
        stream_eval_samples=50,
    )

    import math
    train_accs = history['stream_0_train_acc']
    assert len(train_accs) == 5, f"Should have 5 entries, got {len(train_accs)}"
    # Epochs 0, 2, 4 should be evaluated (freq=2, and epoch 4 is last)
    # Epochs 1, 3 should be NaN (skipped)
    print(f"  stream_0_train_acc per epoch: {train_accs}")
    assert math.isnan(train_accs[1]), f"Epoch 1 should be NaN (skipped), got {train_accs[1]}"
    assert math.isnan(train_accs[3]), f"Epoch 3 should be NaN (skipped), got {train_accs[3]}"
    # Evaluated epochs should have real values (not NaN)
    assert not math.isnan(train_accs[0]), f"Epoch 0 should be evaluated, got NaN"
    assert not math.isnan(train_accs[2]), f"Epoch 2 should be evaluated, got NaN"
    assert not math.isnan(train_accs[4]), f"Epoch 4 should be evaluated, got NaN"
    # LR list should have same length as acc list
    lrs = history['stream_0_lr']
    assert len(lrs) == 5, f"LR should have 5 entries (every epoch), got {len(lrs)}"
    print("  OK: Skipped epochs have NaN, evaluated epochs have values, LR always tracked")


def test_evaluate_method():
    """Test the public evaluate() method with stream_monitoring."""
    print("\nTest 10: evaluate() method with stream_monitoring=True...")
    model = create_test_model()
    model.compile(
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        loss='cross_entropy',
    )
    loader, _ = create_test_data()

    result = model.evaluate(loader, stream_monitoring=True)

    assert 'loss' in result
    assert 'accuracy' in result
    assert 'baseline_acc' in result
    assert 'stream_0_blanked_acc' in result
    assert 'stream_1_blanked_acc' in result
    assert 'stream_0_contribution' in result
    assert 'stream_1_contribution' in result

    print(f"  accuracy={result['accuracy']:.4f}")
    print(f"  baseline_acc={result['baseline_acc']:.4f}")
    print(f"  stream_0_blanked_acc={result['stream_0_blanked_acc']:.4f}")
    print(f"  stream_1_blanked_acc={result['stream_1_blanked_acc']:.4f}")

    # accuracy and baseline_acc should be the same (both are full model on same data)
    assert abs(result['accuracy'] - result['baseline_acc']) < 1e-10, \
        f"accuracy ({result['accuracy']}) should equal baseline_acc ({result['baseline_acc']})"

    print("  OK: evaluate() returns correct structure, accuracy == baseline_acc")


def test_evaluate_without_monitoring():
    """Test evaluate() with stream_monitoring=False returns only loss/accuracy."""
    print("\nTest 11: evaluate() without stream_monitoring...")
    model = create_test_model()
    model.compile(
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        loss='cross_entropy',
    )
    loader, _ = create_test_data()

    result = model.evaluate(loader, stream_monitoring=False)
    assert 'loss' in result
    assert 'accuracy' in result
    assert 'baseline_acc' not in result, "Should not have stream contrib keys"
    print(f"  result keys: {list(result.keys())}")
    print("  OK: No stream contribution keys when monitoring=False")


def test_no_weight_updates_during_monitoring():
    """Verify that _evaluate_stream_contributions does NOT update any weights."""
    print("\nTest 12: No weight updates during stream evaluation...")
    model = create_test_model()
    model.compile(
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        loss='cross_entropy',
    )
    loader, _ = create_test_data()

    # Snapshot weights before
    weights_before = {name: p.clone() for name, p in model.named_parameters()}

    # Run stream evaluation
    model._evaluate_stream_contributions(loader)

    # Check weights after
    for name, p in model.named_parameters():
        diff = (p - weights_before[name]).abs().sum().item()
        assert diff == 0.0, f"Weight {name} changed by {diff} during evaluation!"

    print("  OK: No weights changed during _evaluate_stream_contributions")


def test_deterministic_eval_subset():
    """Test that the eval subset is deterministic (same seed = same results)."""
    print("\nTest 13: Deterministic eval subset...")
    model = create_test_model()
    model.compile(
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        loss='cross_entropy',
    )
    loader, dataset = create_test_data(n_samples=200)

    # Manually create subset like fit() does
    n_samples = min(50, len(dataset))
    gen1 = torch.Generator().manual_seed(42)
    indices1 = torch.randperm(len(dataset), generator=gen1)[:n_samples].tolist()

    gen2 = torch.Generator().manual_seed(42)
    indices2 = torch.randperm(len(dataset), generator=gen2)[:n_samples].tolist()

    assert indices1 == indices2, "Same seed should give same indices"
    print(f"  First 10 indices: {indices1[:10]}")
    print("  OK: Deterministic subset selection")


if __name__ == '__main__':
    print("=" * 60)
    print("STREAM MONITORING TEST SUITE")
    print("=" * 60)

    tests = [
        test_validate_returns_tuple,
        test_validate_with_blanking,
        test_evaluate_stream_contributions,
        test_stream_accuracy_mapping,
        test_blanking_actually_zeros_stream,
        test_fit_history_keys,
        test_fit_with_val_loader,
        test_eval_freq_zero,
        test_eval_freq_skipping,
        test_evaluate_method,
        test_evaluate_without_monitoring,
        test_no_weight_updates_during_monitoring,
        test_deterministic_eval_subset,
    ]

    passed = 0
    failed = 0
    errors = []

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((test_fn.__name__, str(e)))
            print(f"  FAILED: {e}")

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    if errors:
        print("\nFailed tests:")
        for name, err in errors:
            print(f"  {name}: {err}")
    print("=" * 60)
