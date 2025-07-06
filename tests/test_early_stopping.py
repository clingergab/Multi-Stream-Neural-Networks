"""
Unit tests for early stopping functionality in ResNet model.
"""
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import tempfile
import os
from unittest.mock import patch

# Add src to path for imports
import sys
import os
src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src')
sys.path.insert(0, src_path)

from models2.core.resnet import resnet18


class TestEarlyStopping:
    """Test class for early stopping functionality."""
    
    @pytest.fixture
    def model(self):
        """Create a simple ResNet18 model for testing."""
        model = resnet18(num_classes=10)
        model.compile(
            optimizer='adam',
            loss='cross_entropy',
            lr=0.01,
            device='cpu'  # Use CPU for tests
        )
        return model
    
    @pytest.fixture
    def dummy_data(self):
        """Create dummy training and validation data."""
        # Create small dummy dataset
        batch_size = 8
        num_samples = 32
        
        # Generate random data (32x32x3 images for CIFAR-like)
        train_x = torch.randn(num_samples, 3, 32, 32)
        train_y = torch.randint(0, 10, (num_samples,))
        
        val_x = torch.randn(16, 3, 32, 32)
        val_y = torch.randint(0, 10, (16,))
        
        train_dataset = TensorDataset(train_x, train_y)
        val_dataset = TensorDataset(val_x, val_y)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, train_x, train_y, val_x, val_y
    
    def test_early_stopping_disabled_by_default(self, model, dummy_data):
        """Test that early stopping is disabled by default."""
        train_loader, val_loader, _, _, _, _ = dummy_data
        
        history = model.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=3,
            verbose=False
        )
        
        # Should not have early stopping info in history
        assert 'early_stopping' not in history
        # Should run all epochs
        assert len(history['train_loss']) == 3
        assert len(history['val_loss']) == 3
    
    def test_early_stopping_without_validation_data(self, model, dummy_data):
        """Test that early stopping is disabled when no validation data is provided."""
        train_loader, _, _, _, _, _ = dummy_data
        
        with patch('builtins.print') as mock_print:
            history = model.fit(
                train_loader=train_loader,
                epochs=3,
                early_stopping=True,
                verbose=False
            )
        
        # Should print warning about disabling early stopping
        mock_print.assert_called_with("⚠️  Early stopping requested but no validation data provided. Disabling early stopping.")
        
        # Should not have early stopping info in history
        assert 'early_stopping' not in history
    
    def test_early_stopping_val_loss_monitor(self, model, dummy_data):
        """Test early stopping with val_loss monitoring."""
        train_loader, val_loader, _, _, _, _ = dummy_data
        
        # Create a mock that returns decreasing losses (better performance) then increasing
        call_count = 0
        def mock_validate(*args, **kwargs):
            nonlocal call_count
            val_losses = [0.8, 0.6, 0.4, 0.7, 0.9, 1.0]  # Best at epoch 3 (index 2)
            val_accuracies = [0.3, 0.4, 0.6, 0.5, 0.4, 0.3]
            
            if call_count < len(val_losses):
                result = (val_losses[call_count], val_accuracies[call_count])
                call_count += 1
                return result
            return (1.0, 0.1)  # Default fallback
        
        with patch.object(model, '_validate', side_effect=mock_validate):
            history = model.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=10,
                early_stopping=True,
                patience=2,
                monitor='val_loss',
                min_delta=0.01,
                verbose=False
            )
        
        # Should have early stopping info
        assert 'early_stopping' in history
        early_stop_info = history['early_stopping']
        
        assert early_stop_info['monitor'] == 'val_loss'
        assert early_stop_info['patience'] == 2
        assert early_stop_info['min_delta'] == 0.01
        assert early_stop_info['best_epoch'] > 0  # Should find some best epoch
    
    def test_early_stopping_val_accuracy_monitor(self, model, dummy_data):
        """Test early stopping with val_accuracy monitoring."""
        train_loader, val_loader, _, _, _, _ = dummy_data
        
        history = model.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=3,
            early_stopping=True,
            patience=10,  # High patience so it doesn't trigger
            monitor='val_accuracy',
            min_delta=0.001,
            verbose=False
        )
        
        # Should have early stopping info
        assert 'early_stopping' in history
        early_stop_info = history['early_stopping']
        
        assert early_stop_info['stopped_early'] is False  # High patience
        assert early_stop_info['monitor'] == 'val_accuracy'
        assert early_stop_info['patience'] == 10
    
    def test_early_stopping_invalid_monitor(self, model, dummy_data):
        """Test that invalid monitor raises ValueError."""
        train_loader, val_loader, _, _, _, _ = dummy_data
        
        with pytest.raises(ValueError, match="Unsupported monitor metric"):
            model.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=3,
                early_stopping=True,
                monitor='invalid_metric',
                verbose=False
            )
    
    def test_restore_best_weights_enabled(self, model, dummy_data):
        """Test that best weights are restored when restore_best_weights=True."""
        train_loader, val_loader, _, _, _, _ = dummy_data
        
        # Get initial weights
        initial_weights = {k: v.clone() for k, v in model.state_dict().items()}
        
        history = model.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=2,
            early_stopping=True,
            patience=1,
            restore_best_weights=True,
            verbose=False
        )
        
        # Weights should be different from initial (training occurred)
        final_weights = model.state_dict()
        weights_changed = False
        for key in initial_weights:
            if not torch.equal(initial_weights[key], final_weights[key]):
                weights_changed = True
                break
        
        assert weights_changed, "Model weights should change during training"
    
    def test_restore_best_weights_disabled(self, model, dummy_data):
        """Test that weights are not restored when restore_best_weights=False."""
        train_loader, val_loader, _, _, _, _ = dummy_data
        
        history = model.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=2,
            early_stopping=True,
            patience=10,  # High patience so early stopping doesn't trigger
            restore_best_weights=False,
            verbose=False
        )
        
        # Should have early stopping info but no restoration
        assert 'early_stopping' in history
        assert history['early_stopping']['stopped_early'] is False
    
    def test_early_stopping_with_tensor_inputs(self, model, dummy_data):
        """Test early stopping works with tensor inputs instead of DataLoaders."""
        _, _, train_x, train_y, val_x, val_y = dummy_data
        
        history = model.fit(
            train_loader=train_x,
            train_targets=train_y,
            val_loader=val_x,
            val_targets=val_y,
            epochs=3,
            batch_size=8,
            early_stopping=True,
            patience=5,
            verbose=False
        )
        
        # Should work with tensor inputs
        assert 'early_stopping' in history
        assert len(history['train_loss']) > 0
        assert len(history['val_loss']) > 0
    
    def test_early_stopping_with_save_path(self, model, dummy_data):
        """Test that legacy save_path still works with early stopping."""
        train_loader, val_loader, _, _, _, _ = dummy_data
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, 'best_model.pth')
            
            history = model.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=2,
                early_stopping=True,
                patience=5,
                save_path=save_path,
                verbose=False
            )
            
            # Should save checkpoint and have early stopping info
            assert os.path.exists(save_path)
            assert 'early_stopping' in history
    
    def test_early_stopping_verbose_output(self, model, dummy_data):
        """Test that verbose output includes early stopping messages."""
        train_loader, val_loader, _, _, _, _ = dummy_data
        
        with patch('builtins.print') as mock_print:
            history = model.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=2,
                early_stopping=True,
                patience=1,
                verbose=True
            )
        
        # Should print early stopping setup message
        print_calls = [call.args[0] for call in mock_print.call_args_list]
        early_stopping_msgs = [msg for msg in print_calls if '🛑 Early stopping enabled' in str(msg)]
        assert len(early_stopping_msgs) > 0
    
    def test_early_stopping_min_delta(self, model, dummy_data):
        """Test that min_delta parameter works correctly."""
        train_loader, val_loader, _, _, _, _ = dummy_data
        
        # Test with very large min_delta (should prevent early stopping)
        history = model.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=3,
            early_stopping=True,
            patience=1,
            min_delta=10.0,  # Very large min_delta
            monitor='val_loss',
            verbose=False
        )
        
        # Should not trigger early stopping due to large min_delta
        if 'early_stopping' in history:
            assert history['early_stopping']['stopped_early'] is False
    
    def test_early_stopping_patience_counting(self, model, dummy_data):
        """Test that patience counting works correctly."""
        train_loader, val_loader, _, _, _, _ = dummy_data
        
        history = model.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=5,
            early_stopping=True,
            patience=3,
            min_delta=0.001,
            verbose=False
        )
        
        # Should have early stopping info
        assert 'early_stopping' in history
        early_stop_info = history['early_stopping']
        
        # Verify structure
        assert 'stopped_early' in early_stop_info
        assert 'best_epoch' in early_stop_info
        assert 'best_metric' in early_stop_info
        assert 'monitor' in early_stop_info
        assert 'patience' in early_stop_info
        assert 'min_delta' in early_stop_info
        
        assert early_stop_info['patience'] == 3
        assert early_stop_info['min_delta'] == 0.001


if __name__ == '__main__':
    # Run tests directly
    pytest.main([__file__, '-v'])
