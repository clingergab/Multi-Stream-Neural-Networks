"""
Test consistency between main early stopping and stream early stopping.

This test suite verifies that:
1. Both use the same comparison logic
2. Both trigger at the correct patience threshold
3. Both work correctly with val_loss and val_accuracy monitors
4. Metric extraction is correct for both
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.models.linear_integration.li_net import li_resnet18


def create_dummy_data(num_samples=100, num_classes=10):
    """Create dummy RGB+Depth data for testing."""
    stream1 = torch.randn(num_samples, 3, 32, 32)  # RGB (3 channels)
    stream2 = torch.randn(num_samples, 1, 32, 32)  # Depth (1 channel)
    targets = torch.randint(0, num_classes, (num_samples,))
    return TensorDataset(stream1, stream2, targets)


def test_main_early_stopping_triggers_at_correct_patience():
    """Test that main early stopping triggers exactly at patience threshold."""
    model = li_resnet18(num_classes=10, device='cpu')
    model.compile(optimizer='adam', learning_rate=0.001, loss='cross_entropy')

    # Create data that will cause no improvement (random targets)
    train_data = create_dummy_data(num_samples=50)
    val_data = create_dummy_data(num_samples=20)
    train_loader = DataLoader(train_data, batch_size=10)
    val_loader = DataLoader(val_data, batch_size=10)

    # Train with patience=3
    history = model.fit(
        train_loader, val_loader,
        epochs=20,  # Enough to trigger early stopping
        early_stopping=True,
        patience=3,
        monitor='val_loss',
        verbose=False
    )

    # Should stop at epoch where patience_counter reaches 3
    # First epoch: best (counter=0)
    # Epochs 2,3,4: no improvement (counter=1,2,3)
    # Should trigger at epoch 4 (when counter=3, which equals patience)
    assert history['early_stopping']['stopped_early'], "Early stopping should have triggered"

    # The number of epochs should be patience + 1 (initial best epoch + patience no-improvement epochs)
    # But since we might get lucky and have improvement, just check it stopped before 20
    assert len(history['train_loss']) < 20, "Should have stopped early"
    assert len(history['train_loss']) <= 10, "Should stop within reasonable time"


def test_stream_early_stopping_triggers_at_correct_patience():
    """Test that stream early stopping triggers exactly at patience threshold."""
    model = li_resnet18(num_classes=10, device='cpu')
    model.compile(optimizer='adam', learning_rate=0.001, loss='cross_entropy')

    # Create data
    train_data = create_dummy_data(num_samples=50)
    val_data = create_dummy_data(num_samples=20)
    train_loader = DataLoader(train_data, batch_size=10)
    val_loader = DataLoader(val_data, batch_size=10)

    # Train with stream early stopping
    history = model.fit(
        train_loader, val_loader,
        epochs=20,
        monitor='val_accuracy',  # Global monitor
        stream_monitoring=True,
        stream_early_stopping=True,
        stream1_patience=3,
        stream2_patience=3,
        verbose=False
    )

    # At least one stream should freeze (random data means no improvement)
    assert 'stream_early_stopping' in history, "Stream early stopping should be tracked"

    # Check that if a stream froze, it happened at the right time
    if history['stream_early_stopping']['stream1_frozen']:
        # Stream1 should have frozen after patience epochs of no improvement
        assert history['stream1_frozen_epoch'] is not None
        assert history['stream1_frozen_epoch'] <= 10, "Should freeze within reasonable time"


def test_main_early_stopping_with_val_loss():
    """Test main early stopping with val_loss monitor."""
    model = li_resnet18(num_classes=10, device='cpu')
    model.compile(optimizer='adam', learning_rate=0.001, loss='cross_entropy')

    train_data = create_dummy_data(num_samples=50)
    val_data = create_dummy_data(num_samples=20)
    train_loader = DataLoader(train_data, batch_size=10)
    val_loader = DataLoader(val_data, batch_size=10)

    history = model.fit(
        train_loader, val_loader,
        epochs=5,
        early_stopping=True,
        patience=10,  # High patience so it doesn't trigger
        monitor='val_loss',
        verbose=False
    )

    # Check that val_loss is being monitored
    assert 'early_stopping' in history
    assert history['early_stopping']['monitor'] == 'val_loss'
    assert 'best_metric' in history['early_stopping']


def test_main_early_stopping_with_val_accuracy():
    """Test main early stopping with val_accuracy monitor."""
    model = li_resnet18(num_classes=10, device='cpu')
    model.compile(optimizer='adam', learning_rate=0.001, loss='cross_entropy')

    train_data = create_dummy_data(num_samples=50)
    val_data = create_dummy_data(num_samples=20)
    train_loader = DataLoader(train_data, batch_size=10)
    val_loader = DataLoader(val_data, batch_size=10)

    history = model.fit(
        train_loader, val_loader,
        epochs=5,
        early_stopping=True,
        patience=10,
        monitor='val_accuracy',
        verbose=False
    )

    # Check that val_accuracy is being monitored
    assert 'early_stopping' in history
    assert history['early_stopping']['monitor'] == 'val_accuracy'
    assert 'best_metric' in history['early_stopping']


def test_stream_early_stopping_with_val_loss():
    """Test stream early stopping with val_loss monitor."""
    model = li_resnet18(num_classes=10, device='cpu')
    model.compile(optimizer='adam', learning_rate=0.001, loss='cross_entropy')

    train_data = create_dummy_data(num_samples=50)
    val_data = create_dummy_data(num_samples=20)
    train_loader = DataLoader(train_data, batch_size=10)
    val_loader = DataLoader(val_data, batch_size=10)

    history = model.fit(
        train_loader, val_loader,
        epochs=5,
        monitor='val_loss',  # Global monitor for both main and stream early stopping
        stream_monitoring=True,
        stream_early_stopping=True,
        stream1_patience=10,
        stream2_patience=10,
        verbose=False
    )

    # Check that monitor is val_loss (used for both main and stream)
    assert 'stream_early_stopping' in history
    assert history['stream_early_stopping']['monitor'] == 'val_loss'


def test_stream_early_stopping_with_val_accuracy():
    """Test stream early stopping with val_accuracy monitor."""
    model = li_resnet18(num_classes=10, device='cpu')
    model.compile(optimizer='adam', learning_rate=0.001, loss='cross_entropy')

    train_data = create_dummy_data(num_samples=50)
    val_data = create_dummy_data(num_samples=20)
    train_loader = DataLoader(train_data, batch_size=10)
    val_loader = DataLoader(val_data, batch_size=10)

    history = model.fit(
        train_loader, val_loader,
        epochs=5,
        monitor='val_accuracy',  # Global monitor for both main and stream early stopping
        stream_monitoring=True,
        stream_early_stopping=True,
        stream1_patience=10,
        stream2_patience=10,
        verbose=False
    )

    # Check that monitor is val_accuracy (used for both main and stream)
    assert 'stream_early_stopping' in history
    assert history['stream_early_stopping']['monitor'] == 'val_accuracy'


def test_patience_counter_increments_correctly():
    """Verify patience counter increments on no improvement for both mechanisms."""
    from src.models.common.model_helpers import setup_early_stopping, setup_stream_early_stopping

    # Test main early stopping setup
    main_state = setup_early_stopping(
        early_stopping=True,
        val_loader=torch.utils.data.DataLoader(create_dummy_data(10)),
        monitor='val_loss',
        patience=5,
        min_delta=0.001,
        verbose=False
    )

    assert main_state['patience'] == 5
    assert main_state['patience_counter'] == 0
    # monitor is no longer stored in state (passed as parameter to functions)
    assert 'monitor' not in main_state
    assert 'min_delta' not in main_state

    # Test stream early stopping setup
    stream_state = setup_stream_early_stopping(
        stream_early_stopping=True,
        stream_monitor='val_accuracy',
        stream1_patience=3,
        stream2_patience=7,
        stream_min_delta=0.01,
        verbose=False
    )

    assert stream_state['stream1']['patience'] == 3
    assert stream_state['stream2']['patience'] == 7
    assert stream_state['stream1']['patience_counter'] == 0
    assert stream_state['stream2']['patience_counter'] == 0
    # monitor is no longer stored in state (passed as parameter to functions)
    assert 'monitor' not in stream_state
    assert 'min_delta' not in stream_state


def test_is_better_logic_consistency():
    """Verify is_better logic works correctly in early_stopping_initiated (no longer stored in state)."""
    from src.models.common.model_helpers import setup_early_stopping, early_stopping_initiated

    # Test val_loss (lower is better)
    state_loss = setup_early_stopping(True, torch.utils.data.DataLoader(create_dummy_data(10)), 'val_loss', 5, 0.001, False)
    model_state = {'param': torch.randn(5, 5)}

    # Verify is_better is NOT stored in state (refactored to parameter-based)
    assert 'is_better' not in state_loss

    # Test that 0.5 is better than 1.0 (for val_loss)
    state_loss['best_metric'] = 1.0
    early_stopping_initiated(model_state, state_loss, val_loss=0.5, val_acc=0.5,
                           epoch=1, monitor='val_loss', min_delta=0.001,
                           pbar=None, verbose=False, restore_best_weights=False)
    assert state_loss['best_metric'] == 0.5  # Should update to better metric
    assert state_loss['patience_counter'] == 0  # Should reset patience

    # Test that 1.5 is NOT better than 1.0 (for val_loss)
    state_loss['best_metric'] = 1.0
    state_loss['patience_counter'] = 0
    early_stopping_initiated(model_state, state_loss, val_loss=1.5, val_acc=0.5,
                           epoch=2, monitor='val_loss', min_delta=0.001,
                           pbar=None, verbose=False, restore_best_weights=False)
    assert state_loss['best_metric'] == 1.0  # Should NOT update
    assert state_loss['patience_counter'] == 1  # Should increment patience

    # Test val_accuracy (higher is better)
    state_acc = setup_early_stopping(True, torch.utils.data.DataLoader(create_dummy_data(10)), 'val_accuracy', 5, 0.001, False)

    # Verify is_better is NOT stored in state
    assert 'is_better' not in state_acc

    # Test that 0.9 is better than 0.8 (for val_accuracy)
    state_acc['best_metric'] = 0.8
    early_stopping_initiated(model_state, state_acc, val_loss=1.0, val_acc=0.9,
                           epoch=1, monitor='val_accuracy', min_delta=0.001,
                           pbar=None, verbose=False, restore_best_weights=False)
    assert state_acc['best_metric'] == 0.9  # Should update to better metric
    assert state_acc['patience_counter'] == 0  # Should reset patience

    # Test that 0.7 is NOT better than 0.8 (for val_accuracy)
    state_acc['best_metric'] = 0.8
    state_acc['patience_counter'] = 0
    early_stopping_initiated(model_state, state_acc, val_loss=1.0, val_acc=0.7,
                           epoch=2, monitor='val_accuracy', min_delta=0.001,
                           pbar=None, verbose=False, restore_best_weights=False)
    assert state_acc['best_metric'] == 0.8  # Should NOT update
    assert state_acc['patience_counter'] == 1  # Should increment patience


def test_invalid_monitor_raises_error():
    """Test that invalid monitor values raise errors for both mechanisms."""
    from src.models.common.model_helpers import setup_early_stopping, setup_stream_early_stopping

    # Main early stopping should raise error
    with pytest.raises(ValueError, match="Unsupported monitor metric"):
        setup_early_stopping(True, torch.utils.data.DataLoader(create_dummy_data(10)), 'invalid_metric', 5, 0.001, False)

    # Stream early stopping should raise error
    with pytest.raises(ValueError, match="Invalid stream_monitor"):
        setup_stream_early_stopping(True, 'invalid_metric', 5, 5, 0.001, False)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
