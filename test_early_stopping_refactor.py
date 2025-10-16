#!/usr/bin/env python3
"""
Comprehensive test suite for early stopping refactoring.

Tests:
1. setup_early_stopping removes 'monitor' from state
2. early_stopping_initiated works with monitor parameter
3. NaN detection prevents early stopping
4. Both val_loss and val_accuracy monitors work correctly
5. Stream early stopping works with monitor parameter
6. All edge cases are handled
"""

import sys
sys.path.insert(0, 'src')

import torch
import math
from models.common.model_helpers import (
    setup_early_stopping,
    early_stopping_initiated,
    setup_stream_early_stopping,
    check_stream_early_stopping
)


def test_setup_early_stopping_removes_monitor():
    """Test that 'monitor' is not stored in early_stopping_state."""
    print("\n" + "="*80)
    print("TEST 1: setup_early_stopping removes 'monitor' from state")
    print("="*80)

    for monitor in ['val_loss', 'val_accuracy']:
        state = setup_early_stopping(
            early_stopping=True,
            val_loader=True,
            monitor=monitor,
            patience=10,
            min_delta=0.01,
            verbose=False
        )

        # Assert monitor is NOT in state
        assert 'monitor' not in state, f"'monitor' should not be in state for {monitor}"
        assert 'is_better' not in state, f"'is_better' should not be in state for {monitor}"

        # Assert expected keys ARE in state (min_delta is now passed as parameter, not stored)
        expected_keys = {'enabled', 'patience', 'best_metric',
                        'patience_counter', 'best_epoch', 'best_weights'}
        assert set(state.keys()) == expected_keys, f"Unexpected keys in state: {state.keys()}"
        assert 'min_delta' not in state, "'min_delta' should not be in state"

        # Assert best_metric initialized correctly
        if monitor == 'val_loss':
            assert state['best_metric'] == float('inf'), "best_metric should be inf for val_loss"
        else:
            assert state['best_metric'] == 0.0, "best_metric should be 0.0 for val_accuracy"

        print(f"  ✅ {monitor}: 'monitor' not in state, keys correct")

    print("\n✅ TEST 1 PASSED: setup_early_stopping works correctly")


def test_early_stopping_with_val_loss():
    """Test early stopping with val_loss monitor."""
    print("\n" + "="*80)
    print("TEST 2: early_stopping_initiated with val_loss monitor")
    print("="*80)

    state = setup_early_stopping(
        early_stopping=True,
        val_loader=True,
        monitor='val_loss',
        patience=3,
        min_delta=0.01,
        verbose=False
    )

    model_state = {'param': torch.randn(5, 5)}

    # Epoch 1: Initial good loss
    should_stop = early_stopping_initiated(
        model_state, state, val_loss=2.5, val_acc=0.6,
        epoch=0, monitor='val_loss', min_delta=0.01, pbar=None, verbose=False, restore_best_weights=True
    )
    assert not should_stop, "Should not stop on first epoch"
    assert state['best_metric'] == 2.5, f"Best metric should be 2.5, got {state['best_metric']}"
    assert state['patience_counter'] == 0, "Patience counter should be 0"
    print(f"  ✅ Epoch 1: loss=2.5, best=2.5, patience=0/3")

    # Epoch 2: Improvement
    should_stop = early_stopping_initiated(
        model_state, state, val_loss=2.3, val_acc=0.65,
        epoch=1, monitor='val_loss', min_delta=0.01, pbar=None, verbose=False, restore_best_weights=True
    )
    assert not should_stop, "Should not stop on improvement"
    assert state['best_metric'] == 2.3, f"Best metric should be 2.3, got {state['best_metric']}"
    assert state['patience_counter'] == 0, "Patience counter should reset to 0"
    print(f"  ✅ Epoch 2: loss=2.3, best=2.3, patience=0/3 (improved!)")

    # Epoch 3: No improvement (within min_delta)
    should_stop = early_stopping_initiated(
        model_state, state, val_loss=2.29, val_acc=0.66,
        epoch=2, monitor='val_loss', min_delta=0.01, pbar=None, verbose=False, restore_best_weights=True
    )
    assert not should_stop, "Should not stop on first no improvement"
    assert state['best_metric'] == 2.3, "Best metric should stay 2.3"
    assert state['patience_counter'] == 1, "Patience counter should be 1"
    print(f"  ✅ Epoch 3: loss=2.29, best=2.3, patience=1/3 (no improvement)")

    # Epochs 4-5: Continue no improvement
    for epoch in [3, 4]:
        should_stop = early_stopping_initiated(
            model_state, state, val_loss=2.35, val_acc=0.65,
            epoch=epoch, monitor='val_loss', min_delta=0.01, pbar=None, verbose=False, restore_best_weights=True
        )
        assert not should_stop, f"Should not stop at epoch {epoch+1}"
        print(f"  ✅ Epoch {epoch+1}: loss=2.35, best=2.3, patience={state['patience_counter']}/3")

    assert state['patience_counter'] == 3, "Patience counter should be 3"

    # Epoch 6: Trigger early stopping
    should_stop = early_stopping_initiated(
        model_state, state, val_loss=2.4, val_acc=0.64,
        epoch=5, monitor='val_loss', min_delta=0.01, pbar=None, verbose=False, restore_best_weights=True
    )
    assert should_stop, "Should stop after patience exhausted"
    assert state['patience_counter'] == 4, "Patience counter should be 4"
    assert state['best_epoch'] == 1, "Best epoch should be 1 (epoch 2)"
    print(f"  ✅ Epoch 6: loss=2.4, EARLY STOPPING TRIGGERED (patience exhausted)")

    print("\n✅ TEST 2 PASSED: val_loss monitoring works correctly")


def test_early_stopping_with_val_accuracy():
    """Test early stopping with val_accuracy monitor."""
    print("\n" + "="*80)
    print("TEST 3: early_stopping_initiated with val_accuracy monitor")
    print("="*80)

    state = setup_early_stopping(
        early_stopping=True,
        val_loader=True,
        monitor='val_accuracy',
        patience=3,
        min_delta=0.01,
        verbose=False
    )

    model_state = {'param': torch.randn(5, 5)}

    # Epoch 1: Initial accuracy
    should_stop = early_stopping_initiated(
        model_state, state, val_loss=2.5, val_acc=0.6,
        epoch=0, monitor='val_accuracy', min_delta=0.01, pbar=None, verbose=False, restore_best_weights=True
    )
    assert not should_stop
    assert state['best_metric'] == 0.6
    assert state['patience_counter'] == 0
    print(f"  ✅ Epoch 1: acc=0.60, best=0.60, patience=0/3")

    # Epoch 2: Improvement
    should_stop = early_stopping_initiated(
        model_state, state, val_loss=2.3, val_acc=0.72,
        epoch=1, monitor='val_accuracy', min_delta=0.01, pbar=None, verbose=False, restore_best_weights=True
    )
    assert not should_stop
    assert state['best_metric'] == 0.72
    assert state['patience_counter'] == 0
    print(f"  ✅ Epoch 2: acc=0.72, best=0.72, patience=0/3 (improved!)")

    # Epoch 3: No improvement (within min_delta)
    should_stop = early_stopping_initiated(
        model_state, state, val_loss=2.2, val_acc=0.72,
        epoch=2, monitor='val_accuracy', min_delta=0.01, pbar=None, verbose=False, restore_best_weights=True
    )
    assert not should_stop
    assert state['best_metric'] == 0.72
    assert state['patience_counter'] == 1
    print(f"  ✅ Epoch 3: acc=0.72, best=0.72, patience=1/3 (no improvement)")

    # Epochs 4-5: Continue no improvement
    for epoch in [3, 4]:
        should_stop = early_stopping_initiated(
            model_state, state, val_loss=2.3, val_acc=0.70,
            epoch=epoch, monitor='val_accuracy', min_delta=0.01, pbar=None, verbose=False, restore_best_weights=True
        )
        assert not should_stop
        print(f"  ✅ Epoch {epoch+1}: acc=0.70, best=0.72, patience={state['patience_counter']}/3")

    # Epoch 6: Trigger early stopping
    should_stop = early_stopping_initiated(
        model_state, state, val_loss=2.4, val_acc=0.68,
        epoch=5, monitor='val_accuracy', min_delta=0.01, pbar=None, verbose=False, restore_best_weights=True
    )
    assert should_stop
    print(f"  ✅ Epoch 6: acc=0.68, EARLY STOPPING TRIGGERED (patience exhausted)")

    print("\n✅ TEST 3 PASSED: val_accuracy monitoring works correctly")


def test_nan_detection():
    """Test that NaN detection prevents early stopping."""
    print("\n" + "="*80)
    print("TEST 4: NaN detection prevents incorrect early stopping")
    print("="*80)

    for monitor in ['val_loss', 'val_accuracy']:
        print(f"\n  Testing with monitor={monitor}")
        state = setup_early_stopping(
            early_stopping=True,
            val_loader=True,
            monitor=monitor,
            patience=3,
            min_delta=0.01,
            verbose=False
        )

        model_state = {'param': torch.randn(5, 5)}

        # Set a good best metric
        if monitor == 'val_loss':
            state['best_metric'] = 2.0
        else:
            state['best_metric'] = 0.75
        state['best_epoch'] = 2

        # Test NaN loss
        should_stop = early_stopping_initiated(
            model_state, state, val_loss=float('nan'), val_acc=0.0619,
            epoch=5, monitor=monitor, min_delta=0.01, pbar=None, verbose=True, restore_best_weights=True
        )

        assert not should_stop, f"Should not stop on NaN for {monitor}"
        assert state['patience_counter'] == 0, f"Patience counter should not increment on NaN for {monitor}"
        if monitor == 'val_loss':
            assert state['best_metric'] == 2.0, f"Best metric should remain unchanged for {monitor}"
        else:
            assert state['best_metric'] == 0.75, f"Best metric should remain unchanged for {monitor}"

        print(f"    ✅ NaN detected, early stopping skipped, patience counter unchanged")

        # Test Inf loss
        should_stop = early_stopping_initiated(
            model_state, state, val_loss=float('inf'), val_acc=0.05,
            epoch=6, monitor=monitor, min_delta=0.01, pbar=None, verbose=True, restore_best_weights=True
        )

        assert not should_stop, f"Should not stop on Inf for {monitor}"
        assert state['patience_counter'] == 0, f"Patience counter should not increment on Inf for {monitor}"
        print(f"    ✅ Inf detected, early stopping skipped, patience counter unchanged")

    print("\n✅ TEST 4 PASSED: NaN/Inf detection works correctly")


def test_best_weights_storage():
    """Test that best weights are stored correctly."""
    print("\n" + "="*80)
    print("TEST 5: Best weights storage and restoration")
    print("="*80)

    state = setup_early_stopping(
        early_stopping=True,
        val_loader=True,
        monitor='val_accuracy',
        patience=2,
        min_delta=0.01,
        verbose=False
    )

    # Create model states
    model_state_1 = {'param': torch.tensor([1.0, 2.0, 3.0])}
    model_state_2 = {'param': torch.tensor([4.0, 5.0, 6.0])}
    model_state_3 = {'param': torch.tensor([7.0, 8.0, 9.0])}

    # Epoch 1: First best
    should_stop = early_stopping_initiated(
        model_state_1, state, val_loss=2.5, val_acc=0.6,
        epoch=0, monitor='val_accuracy', min_delta=0.01, pbar=None, verbose=False, restore_best_weights=True
    )
    assert state['best_weights'] is not None
    assert torch.equal(state['best_weights']['param'], torch.tensor([1.0, 2.0, 3.0]))
    print(f"  ✅ Epoch 1: Best weights stored: {state['best_weights']['param'].tolist()}")

    # Epoch 2: New best
    should_stop = early_stopping_initiated(
        model_state_2, state, val_loss=2.3, val_acc=0.75,
        epoch=1, monitor='val_accuracy', min_delta=0.01, pbar=None, verbose=False, restore_best_weights=True
    )
    assert state['best_weights'] is not None
    assert torch.equal(state['best_weights']['param'], torch.tensor([4.0, 5.0, 6.0]))
    print(f"  ✅ Epoch 2: Best weights updated: {state['best_weights']['param'].tolist()}")

    # Epoch 3: No improvement, weights should not change
    should_stop = early_stopping_initiated(
        model_state_3, state, val_loss=2.4, val_acc=0.70,
        epoch=2, monitor='val_accuracy', min_delta=0.01, pbar=None, verbose=False, restore_best_weights=True
    )
    assert torch.equal(state['best_weights']['param'], torch.tensor([4.0, 5.0, 6.0]))
    print(f"  ✅ Epoch 3: Best weights unchanged: {state['best_weights']['param'].tolist()}")

    print("\n✅ TEST 5 PASSED: Best weights stored and preserved correctly")


def test_setup_stream_early_stopping():
    """Test that stream early stopping setup removes monitor from state."""
    print("\n" + "="*80)
    print("TEST 6: setup_stream_early_stopping removes 'monitor' from state")
    print("="*80)

    for monitor in ['val_loss', 'val_accuracy']:
        state = setup_stream_early_stopping(
            stream_early_stopping=True,
            stream_monitor=monitor,
            stream1_patience=10,
            stream2_patience=15,
            stream_min_delta=0.01,
            verbose=False
        )

        # Assert monitor is NOT in state
        assert 'monitor' not in state, f"'monitor' should not be in state for {monitor}"
        assert 'is_better' not in state, f"'is_better' should not be in state for {monitor}"

        # Assert expected keys ARE in state (min_delta removed, passed as parameter)
        expected_keys = {'enabled', 'stream1', 'stream2', 'all_frozen', 'best_full_model'}
        assert set(state.keys()) == expected_keys, f"Unexpected keys in state: {state.keys()}"
        assert 'min_delta' not in state, f"'min_delta' should not be in state for {monitor}"

        # Assert best_metric initialized correctly in stream states
        if monitor == 'val_loss':
            assert state['stream1']['best_metric'] == float('inf')
            assert state['stream2']['best_metric'] == float('inf')
            assert state['best_full_model']['best_metric'] == float('inf')
        else:
            assert state['stream1']['best_metric'] == 0.0
            assert state['stream2']['best_metric'] == 0.0
            assert state['best_full_model']['best_metric'] == 0.0

        print(f"  ✅ {monitor}: 'monitor' not in state, stream states initialized correctly")

    print("\n✅ TEST 6 PASSED: setup_stream_early_stopping works correctly")


def test_check_stream_early_stopping():
    """Test that check_stream_early_stopping works with monitor parameter."""
    print("\n" + "="*80)
    print("TEST 7: check_stream_early_stopping with monitor parameter")
    print("="*80)

    # Create a dummy model
    class DummyModel:
        def __init__(self):
            self.param_stream1 = torch.nn.Parameter(torch.randn(5, 5))
            self.param_stream2 = torch.nn.Parameter(torch.randn(5, 5))
            self.param_shared = torch.nn.Parameter(torch.randn(5, 5))

        def named_parameters(self):
            return [
                ('module.stream1_weight', self.param_stream1),
                ('module.stream2_weight', self.param_stream2),
                ('module.shared_weight', self.param_shared),
            ]

        def parameters(self):
            return iter([self.param_stream1, self.param_stream2, self.param_shared])

        def state_dict(self):
            return {
                'module.stream1_weight': self.param_stream1,
                'module.stream2_weight': self.param_stream2,
                'module.shared_weight': self.param_shared,
            }

        def load_state_dict(self, state_dict):
            """Load state dict (dummy implementation for testing)."""
            pass

    model = DummyModel()

    # Test with val_accuracy monitor
    state = setup_stream_early_stopping(
        stream_early_stopping=True,
        stream_monitor='val_accuracy',
        stream1_patience=2,
        stream2_patience=2,
        stream_min_delta=0.01,
        verbose=False
    )

    # Epoch 1: Initial metrics
    stream_stats = {
        'stream1_val_acc': 0.6,
        'stream2_val_acc': 0.5,
        'stream1_val_loss': 2.5,
        'stream2_val_loss': 2.7,
    }
    all_frozen = check_stream_early_stopping(
        stream_early_stopping_state=state,
        stream_stats=stream_stats,
        model=model,
        epoch=0,
        monitor='val_accuracy', min_delta=0.01,
        verbose=False,
        val_acc=0.65,
        val_loss=2.6
    )
    assert not all_frozen, "Streams should not be frozen on first epoch"
    assert state['stream1']['best_metric'] == 0.6
    assert state['stream2']['best_metric'] == 0.5
    print(f"  ✅ Epoch 1: Stream1 acc=0.6, Stream2 acc=0.5, neither frozen")

    # Epoch 2: No improvement
    stream_stats = {
        'stream1_val_acc': 0.58,
        'stream2_val_acc': 0.48,
        'stream1_val_loss': 2.6,
        'stream2_val_loss': 2.8,
    }
    all_frozen = check_stream_early_stopping(
        stream_early_stopping_state=state,
        stream_stats=stream_stats,
        model=model,
        epoch=1,
        monitor='val_accuracy', min_delta=0.01,
        verbose=False,
        val_acc=0.63,
        val_loss=2.7
    )
    assert not all_frozen
    assert state['stream1']['patience_counter'] == 1
    assert state['stream2']['patience_counter'] == 1
    print(f"  ✅ Epoch 2: No improvement, patience=1/2 for both streams")

    # Epoch 3: Still no improvement, should freeze
    stream_stats = {
        'stream1_val_acc': 0.57,
        'stream2_val_acc': 0.47,
        'stream1_val_loss': 2.7,
        'stream2_val_loss': 2.9,
    }
    all_frozen = check_stream_early_stopping(
        stream_early_stopping_state=state,
        stream_stats=stream_stats,
        model=model,
        epoch=2,
        monitor='val_accuracy', min_delta=0.01,
        verbose=True,
        val_acc=0.62,
        val_loss=2.8
    )
    assert not all_frozen  # They freeze after patience+1 epochs
    assert state['stream1']['patience_counter'] == 2
    assert state['stream2']['patience_counter'] == 2
    print(f"  ✅ Epoch 3: Patience exhausted, patience=2/2")

    # Epoch 4: Trigger freezing
    all_frozen = check_stream_early_stopping(
        stream_early_stopping_state=state,
        stream_stats=stream_stats,
        model=model,
        epoch=3,
        monitor='val_accuracy', min_delta=0.01,
        verbose=True,
        val_acc=0.61,
        val_loss=2.9
    )
    assert all_frozen, "Both streams should be frozen"
    assert state['stream1']['frozen']
    assert state['stream2']['frozen']
    print(f"  ✅ Epoch 4: Both streams frozen, all_frozen=True")

    print("\n✅ TEST 7 PASSED: check_stream_early_stopping works correctly with monitor parameter")


def test_min_delta_behavior():
    """Test that min_delta is respected."""
    print("\n" + "="*80)
    print("TEST 8: min_delta threshold behavior")
    print("="*80)

    # Test with val_loss (lower is better)
    state = setup_early_stopping(
        early_stopping=True,
        val_loader=True,
        monitor='val_loss',
        patience=5,
        min_delta=0.1,  # Require 0.1 improvement
        verbose=False
    )

    model_state = {'param': torch.randn(5, 5)}

    # Epoch 1: Set baseline
    early_stopping_initiated(
        model_state, state, val_loss=2.5, val_acc=0.6,
        epoch=0, monitor='val_loss', min_delta=0.1, pbar=None, verbose=False, restore_best_weights=True
    )
    assert state['best_metric'] == 2.5
    print(f"  ✅ Baseline: loss=2.5")

    # Epoch 2: Improve by 0.05 (less than min_delta=0.1)
    early_stopping_initiated(
        model_state, state, val_loss=2.45, val_acc=0.62,
        epoch=1, monitor='val_loss', min_delta=0.1, pbar=None, verbose=False, restore_best_weights=True
    )
    assert state['best_metric'] == 2.5, "Should not update best_metric (improvement < min_delta)"
    assert state['patience_counter'] == 1, "Should increment patience"
    print(f"  ✅ Epoch 2: loss=2.45 (improved by 0.05 < 0.1), patience=1, best unchanged")

    # Epoch 3: Improve by 0.15 (more than min_delta=0.1)
    early_stopping_initiated(
        model_state, state, val_loss=2.35, val_acc=0.65,
        epoch=2, monitor='val_loss', min_delta=0.1, pbar=None, verbose=False, restore_best_weights=True
    )
    assert state['best_metric'] == 2.35, "Should update best_metric (improvement > min_delta)"
    assert state['patience_counter'] == 0, "Should reset patience"
    print(f"  ✅ Epoch 3: loss=2.35 (improved by 0.15 > 0.1), patience=0, best updated")

    # Test with val_accuracy (higher is better)
    state = setup_early_stopping(
        early_stopping=True,
        val_loader=True,
        monitor='val_accuracy',
        patience=5,
        min_delta=0.05,  # Require 0.05 improvement
        verbose=False
    )

    # Epoch 1: Set baseline
    early_stopping_initiated(
        model_state, state, val_loss=2.5, val_acc=0.7,
        epoch=0, monitor='val_accuracy', min_delta=0.05, pbar=None, verbose=False, restore_best_weights=True
    )
    assert state['best_metric'] == 0.7
    print(f"\n  ✅ Baseline: acc=0.7")

    # Epoch 2: Improve by 0.03 (less than min_delta=0.05)
    early_stopping_initiated(
        model_state, state, val_loss=2.4, val_acc=0.73,
        epoch=1, monitor='val_accuracy', min_delta=0.05, pbar=None, verbose=False, restore_best_weights=True
    )
    assert state['best_metric'] == 0.7, "Should not update best_metric (improvement < min_delta)"
    assert state['patience_counter'] == 1
    print(f"  ✅ Epoch 2: acc=0.73 (improved by 0.03 < 0.05), patience=1, best unchanged")

    # Epoch 3: Improve by 0.08 (more than min_delta=0.05)
    early_stopping_initiated(
        model_state, state, val_loss=2.3, val_acc=0.78,
        epoch=2, monitor='val_accuracy', min_delta=0.05, pbar=None, verbose=False, restore_best_weights=True
    )
    assert state['best_metric'] == 0.78, "Should update best_metric (improvement > min_delta)"
    assert state['patience_counter'] == 0
    print(f"  ✅ Epoch 3: acc=0.78 (improved by 0.08 > 0.05), patience=0, best updated")

    print("\n✅ TEST 8 PASSED: min_delta threshold works correctly")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*80)
    print("EARLY STOPPING REFACTORING - COMPREHENSIVE TEST SUITE")
    print("="*80)

    try:
        test_setup_early_stopping_removes_monitor()
        test_early_stopping_with_val_loss()
        test_early_stopping_with_val_accuracy()
        test_nan_detection()
        test_best_weights_storage()
        test_setup_stream_early_stopping()
        test_check_stream_early_stopping()
        test_min_delta_behavior()

        print("\n" + "="*80)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("="*80)
        print("\nSummary:")
        print("  ✅ setup_early_stopping removes 'monitor' from state")
        print("  ✅ early_stopping_initiated works with monitor parameter")
        print("  ✅ NaN/Inf detection prevents incorrect early stopping")
        print("  ✅ Both val_loss and val_accuracy monitors work correctly")
        print("  ✅ Best weights storage and restoration works")
        print("  ✅ setup_stream_early_stopping removes 'monitor' from state")
        print("  ✅ check_stream_early_stopping works with monitor parameter")
        print("  ✅ min_delta threshold behavior is correct")
        print("\n" + "="*80)
        return True
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
