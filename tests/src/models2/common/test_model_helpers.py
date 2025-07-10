"""
Comprehensive unit tests for model_helpers.py module.
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, 
    OneCycleLR, 
    StepLR, 
    ReduceLROnPlateau
)
from unittest.mock import Mock, MagicMock, patch, call
import os
import tempfile
import shutil
from tqdm import tqdm

from models2.common.model_helpers import (
    setup_scheduler,
    update_scheduler,
    print_epoch_progress,
    save_checkpoint,
    create_dataloader_from_tensors,
    setup_early_stopping,
    early_stopping_initiated,
    create_progress_bar,
    finalize_progress_bar,
    update_history
)


class MockModel:
    """Mock model class for testing."""
    def __init__(self, device='cpu', scheduler_type=None):
        self.device = torch.device(device)
        self.scheduler_type = scheduler_type
        self.scheduler = None
        self.optimizer = optim.SGD([torch.nn.Parameter(torch.randn(1))], lr=0.01)
        self._state_dict = {'param': torch.randn(5, 5)}
    
    def state_dict(self):
        return self._state_dict
    
    def load_state_dict(self, state_dict):
        self._state_dict = state_dict


class TestSetupScheduler:
    """Test setup_scheduler function."""
    
    def test_setup_scheduler_none(self):
        """Test that no scheduler is set when scheduler_type is None."""
        optimizer = optim.SGD([torch.nn.Parameter(torch.randn(1))], lr=0.01)
        scheduler = setup_scheduler(optimizer, None, epochs=10, train_loader_len=100)
        assert scheduler is None
    
    def test_setup_scheduler_cosine(self):
        """Test cosine annealing scheduler setup."""
        optimizer = optim.SGD([torch.nn.Parameter(torch.randn(1))], lr=0.01)
        scheduler = setup_scheduler(optimizer, 'cosine', epochs=10, train_loader_len=100)
        assert isinstance(scheduler, CosineAnnealingLR)
        assert scheduler.T_max == 10
    
    def test_setup_scheduler_cosine_custom_t_max(self):
        """Test cosine annealing scheduler with custom t_max."""
        optimizer = optim.SGD([torch.nn.Parameter(torch.randn(1))], lr=0.01)
        scheduler = setup_scheduler(optimizer, 'cosine', epochs=10, train_loader_len=100, t_max=15)
        assert isinstance(scheduler, CosineAnnealingLR)
        assert scheduler.T_max == 15
    
    def test_setup_scheduler_onecycle(self):
        """Test OneCycle scheduler setup."""
        optimizer = optim.SGD([torch.nn.Parameter(torch.randn(1))], lr=0.01)
        scheduler = setup_scheduler(optimizer, 'onecycle', epochs=10, train_loader_len=100)
        assert isinstance(scheduler, OneCycleLR)
        assert scheduler.total_steps == 1000  # 10 * 100
    
    def test_setup_scheduler_onecycle_custom_params(self):
        """Test OneCycle scheduler with custom parameters."""
        optimizer = optim.SGD([torch.nn.Parameter(torch.randn(1))], lr=0.01)
        scheduler = setup_scheduler(
            optimizer, 'onecycle', epochs=5, train_loader_len=50,
            max_lr=0.1, pct_start=0.2, anneal_strategy='linear',
            div_factor=10.0, final_div_factor=100.0
        )
        assert isinstance(scheduler, OneCycleLR)
        assert scheduler.total_steps == 250  # 5 * 50
    
    def test_setup_scheduler_step(self):
        """Test StepLR scheduler setup."""
        optimizer = optim.SGD([torch.nn.Parameter(torch.randn(1))], lr=0.01)
        scheduler = setup_scheduler(optimizer, 'step', epochs=10, train_loader_len=100)
        assert isinstance(scheduler, StepLR)
        assert scheduler.step_size == 30
        assert scheduler.gamma == 0.1
    
    def test_setup_scheduler_step_custom_params(self):
        """Test StepLR scheduler with custom parameters."""
        optimizer = optim.SGD([torch.nn.Parameter(torch.randn(1))], lr=0.01)
        scheduler = setup_scheduler(optimizer, 'step', epochs=10, train_loader_len=100, step_size=20, gamma=0.5)
        assert isinstance(scheduler, StepLR)
        assert scheduler.step_size == 20
        assert scheduler.gamma == 0.5
    
    def test_setup_scheduler_plateau(self):
        """Test ReduceLROnPlateau scheduler setup."""
        optimizer = optim.SGD([torch.nn.Parameter(torch.randn(1))], lr=0.01)
        scheduler = setup_scheduler(optimizer, 'plateau', epochs=10, train_loader_len=100)
        assert isinstance(scheduler, ReduceLROnPlateau)
        assert scheduler.patience == 10
        assert scheduler.factor == 0.5
    
    def test_setup_scheduler_plateau_custom_params(self):
        """Test ReduceLROnPlateau scheduler with custom parameters."""
        optimizer = optim.SGD([torch.nn.Parameter(torch.randn(1))], lr=0.01)
        scheduler = setup_scheduler(
            optimizer, 'plateau', epochs=10, train_loader_len=100,
            scheduler_patience=5, factor=0.2
        )
        assert isinstance(scheduler, ReduceLROnPlateau)
        assert scheduler.patience == 5
        assert scheduler.factor == 0.2
    
    def test_setup_scheduler_plateau_fallback_patience(self):
        """Test ReduceLROnPlateau scheduler with fallback patience parameter."""
        optimizer = optim.SGD([torch.nn.Parameter(torch.randn(1))], lr=0.01)
        scheduler = setup_scheduler(optimizer, 'plateau', epochs=10, train_loader_len=100, patience=15)
        assert isinstance(scheduler, ReduceLROnPlateau)
        assert scheduler.patience == 15
    
    def test_setup_scheduler_invalid_type(self):
        """Test that invalid scheduler type raises ValueError."""
        optimizer = optim.SGD([torch.nn.Parameter(torch.randn(1))], lr=0.01)
        with pytest.raises(ValueError, match="Unsupported scheduler type: invalid"):
            setup_scheduler(optimizer, 'invalid', epochs=10, train_loader_len=100)


class TestUpdateScheduler:
    """Test update_scheduler function."""
    
    def test_update_scheduler_none(self):
        """Test that no error occurs when scheduler is None."""
        update_scheduler(None, val_loss=0.5)  # Should not raise
    
    def test_update_scheduler_plateau(self):
        """Test ReduceLROnPlateau scheduler update."""
        optimizer = optim.SGD([torch.nn.Parameter(torch.randn(1))], lr=0.01)
        scheduler = setup_scheduler(optimizer, 'plateau', epochs=10, train_loader_len=100)
        
        with patch.object(scheduler, 'step') as mock_step:
            update_scheduler(scheduler, val_loss=0.5)
            mock_step.assert_called_once_with(0.5)
    
    def test_update_scheduler_cosine(self):
        """Test CosineAnnealingLR scheduler update."""
        optimizer = optim.SGD([torch.nn.Parameter(torch.randn(1))], lr=0.01)
        scheduler = setup_scheduler(optimizer, 'cosine', epochs=10, train_loader_len=100)
        
        with patch.object(scheduler, 'step') as mock_step:
            update_scheduler(scheduler, val_loss=0.5)
            mock_step.assert_called_once_with()
    
    def test_update_scheduler_onecycle_skipped(self):
        """Test OneCycleLR scheduler is skipped in update."""
        optimizer = optim.SGD([torch.nn.Parameter(torch.randn(1))], lr=0.01)
        scheduler = setup_scheduler(optimizer, 'onecycle', epochs=10, train_loader_len=100)
        
        with patch.object(scheduler, 'step') as mock_step:
            update_scheduler(scheduler, val_loss=0.5)
            mock_step.assert_not_called()


class TestPrintEpochProgress:
    """Test print_epoch_progress function."""
    
    def test_print_epoch_progress_with_validation(self, capsys):
        """Test printing epoch progress with validation data."""
        print_epoch_progress(
            epoch=0, epochs=10, epoch_time=12.34,
            avg_train_loss=0.567, train_accuracy=0.8,
            val_loss=0.432, val_acc=0.85, val_loader=True
        )
        
        captured = capsys.readouterr()
        assert "Epoch 1/10" in captured.out
        assert "Time: 12.34s" in captured.out
        assert "Train Loss: 0.5670" in captured.out
        assert "Train Acc: 80.00%" in captured.out
        assert "Val Loss: 0.4320" in captured.out
        assert "Val Acc: 0.85%" in captured.out
    
    def test_print_epoch_progress_without_validation(self, capsys):
        """Test printing epoch progress without validation data."""
        print_epoch_progress(
            epoch=4, epochs=10, epoch_time=15.67,
            avg_train_loss=0.234, train_accuracy=0.92,
            val_loss=0.0, val_acc=0.0, val_loader=False
        )
        
        captured = capsys.readouterr()
        assert "Epoch 5/10" in captured.out
        assert "Time: 15.67s" in captured.out
        assert "Train Loss: 0.2340" in captured.out
        assert "Train Acc: 92.00%" in captured.out
        assert "Val Loss" not in captured.out
        assert "Val Acc" not in captured.out


class TestSaveCheckpoint:
    """Test save_checkpoint function."""
    
    def test_save_checkpoint_basic(self):
        """Test basic checkpoint saving."""
        model_state_dict = {'param': torch.randn(5, 5)}
        optimizer_state_dict = {'param_groups': [{'lr': 0.01}]}
        scheduler_state_dict = {'step_size': 30}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, 'checkpoint.pth')
            save_checkpoint(model_state_dict, optimizer_state_dict, scheduler_state_dict, checkpoint_path)
            
            assert os.path.exists(checkpoint_path)
            checkpoint = torch.load(checkpoint_path)
            assert 'model_state_dict' in checkpoint
            assert 'optimizer_state_dict' in checkpoint
            assert 'scheduler_state_dict' in checkpoint
    
    def test_save_checkpoint_with_history(self):
        """Test checkpoint saving with history."""
        model_state_dict = {'param': torch.randn(5, 5)}
        optimizer_state_dict = {'param_groups': [{'lr': 0.01}]}
        scheduler_state_dict = {'step_size': 30}
        history = {'train_loss': [0.5, 0.3], 'val_loss': [0.4, 0.2]}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, 'checkpoint.pth')
            save_checkpoint(model_state_dict, optimizer_state_dict, scheduler_state_dict, checkpoint_path, history=history)
            
            checkpoint = torch.load(checkpoint_path)
            assert 'history' in checkpoint
            assert checkpoint['history'] == history
    
    def test_save_checkpoint_creates_directory(self):
        """Test that checkpoint saving creates necessary directories."""
        model_state_dict = {'param': torch.randn(5, 5)}
        optimizer_state_dict = {'param_groups': [{'lr': 0.01}]}
        scheduler_state_dict = {'step_size': 30}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_path = os.path.join(temp_dir, 'nested', 'dir', 'checkpoint.pth')
            save_checkpoint(model_state_dict, optimizer_state_dict, scheduler_state_dict, nested_path)
            
            assert os.path.exists(nested_path)
    
    def test_save_checkpoint_no_optimizer(self):
        """Test checkpoint saving when no optimizer state dict is provided."""
        model_state_dict = {'param': torch.randn(5, 5)}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, 'checkpoint.pth')
            save_checkpoint(model_state_dict, None, None, checkpoint_path)
            
            checkpoint = torch.load(checkpoint_path)
            assert checkpoint['optimizer_state_dict'] is None
    
    def test_save_checkpoint_no_scheduler(self):
        """Test checkpoint saving when no scheduler state dict is provided."""
        model_state_dict = {'param': torch.randn(5, 5)}
        optimizer_state_dict = {'param_groups': [{'lr': 0.01}]}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, 'checkpoint.pth')
            save_checkpoint(model_state_dict, optimizer_state_dict, None, checkpoint_path)
            
            checkpoint = torch.load(checkpoint_path)
            assert checkpoint['scheduler_state_dict'] is None


class TestCreateDataloaderFromTensors:
    """Test create_dataloader_from_tensors function."""
    
    def test_create_dataloader_cuda_default_params(self):
        """Test DataLoader creation for CUDA device with default parameters."""
        X = torch.randn(100, 10)
        y = torch.randint(0, 2, (100,))
        
        dataloader = create_dataloader_from_tensors(X, y, batch_size=16, shuffle=True, device=torch.device('cuda'))
        
        assert isinstance(dataloader, DataLoader)
        assert dataloader.batch_size == 16
        assert dataloader.num_workers == 8  # Updated default
        assert dataloader.pin_memory == True
        assert dataloader.persistent_workers == True
        # prefetch_factor is only accessible in PyTorch 1.7+, so we'll skip checking it directly

    def test_create_dataloader_cuda_custom_params(self):
        """Test DataLoader creation for CUDA with custom GPU optimization parameters."""
        X = torch.randn(100, 10)
        y = torch.randint(0, 2, (100,))
        
        dataloader = create_dataloader_from_tensors(
            X, y, batch_size=16, shuffle=True, device=torch.device('cuda'),
            num_workers=4, pin_memory=False, persistent_workers=False, prefetch_factor=1
        )
        
        assert isinstance(dataloader, DataLoader)
        assert dataloader.batch_size == 16
        assert dataloader.num_workers == 4
        assert dataloader.pin_memory == False
        assert dataloader.persistent_workers == False
    
    def test_create_dataloader_mps_optimizations(self):
        """Test DataLoader creation for MPS device with MPS-specific optimizations."""
        X = torch.randn(100, 10)
        y = torch.randint(0, 2, (100,))
        
        dataloader = create_dataloader_from_tensors(
            X, y, batch_size=32, device=torch.device('mps'),
            num_workers=8  # Should be limited to 2 for MPS
        )
        
        assert isinstance(dataloader, DataLoader)
        assert dataloader.batch_size == 32
        assert dataloader.num_workers == 2  # Limited for MPS
        assert dataloader.pin_memory == False  # MPS doesn't benefit from pin_memory
        assert dataloader.persistent_workers == False  # MPS doesn't benefit from persistent workers
    
    def test_create_dataloader_cpu_params(self):
        """Test DataLoader creation for CPU device with custom parameters."""
        X = torch.randn(100, 10)
        y = torch.randint(0, 2, (100,))
        
        dataloader = create_dataloader_from_tensors(
            X, y, batch_size=8, device=torch.device('cpu'),
            num_workers=6, persistent_workers=True
        )
        
        assert isinstance(dataloader, DataLoader)
        assert dataloader.batch_size == 8
        assert dataloader.num_workers == 6
        assert dataloader.pin_memory == False  # No GPU transfer for CPU
        assert dataloader.persistent_workers == True
    
    def test_create_dataloader_no_workers_params(self):
        """Test DataLoader creation with num_workers=0 (disables persistent_workers and prefetch_factor)."""
        X = torch.randn(100, 10)
        y = torch.randint(0, 2, (100,))
        
        dataloader = create_dataloader_from_tensors(
            X, y, batch_size=16, device=torch.device('cuda'),
            num_workers=0, persistent_workers=True  # Should be disabled when num_workers=0
        )
        
        assert isinstance(dataloader, DataLoader)
        assert dataloader.num_workers == 0
        assert dataloader.persistent_workers == False  # Should be disabled
    
    def test_create_dataloader_with_targets(self):
        """Test DataLoader creation with target tensors."""
        X = torch.randn(100, 10)
        y = torch.randint(0, 2, (100,))
        
        dataloader = create_dataloader_from_tensors(X, y, batch_size=16)
        
        # Check that dataset has both X and y
        sample = next(iter(dataloader))
        assert len(sample) == 2  # Both X and y

    def test_create_dataloader_auto_device_detection(self):
        """Test automatic device detection when device=None."""
        X = torch.randn(50, 5)
        y = torch.randint(0, 2, (50,))
        
        dataloader = create_dataloader_from_tensors(X, y, batch_size=10, device=None)
        
        assert isinstance(dataloader, DataLoader)
        assert dataloader.batch_size == 10
        # Device-specific settings will be applied based on what's available


class TestSetupEarlyStopping:
    """Test setup_early_stopping function."""
    
    def test_setup_early_stopping_disabled(self):
        """Test early stopping setup when disabled."""
        val_loader = Mock()
        
        config = setup_early_stopping(
            early_stopping=False, val_loader=val_loader,
            monitor='val_loss', patience=5, min_delta=0.01, verbose=False
        )
        
        assert config == {'enabled': False}
    
    def test_setup_early_stopping_no_val_loader(self, capsys):
        """Test early stopping setup without validation loader."""
        config = setup_early_stopping(
            early_stopping=True, val_loader=None,
            monitor='val_loss', patience=5, min_delta=0.01, verbose=True
        )
        
        captured = capsys.readouterr()
        assert "Early stopping requested but no validation data provided" in captured.out
        assert config == {'enabled': False}
    
    def test_setup_early_stopping_val_loss_monitor(self, capsys):
        """Test early stopping setup with val_loss monitor."""
        val_loader = Mock()
        
        config = setup_early_stopping(
            early_stopping=True, val_loader=val_loader,
            monitor='val_loss', patience=10, min_delta=0.001, verbose=True
        )
        
        captured = capsys.readouterr()
        assert "Early stopping enabled: monitoring val_loss" in captured.out
        assert config['enabled'] == True
        assert config['monitor'] == 'val_loss'
        assert config['patience'] == 10
        assert config['min_delta'] == 0.001
        assert config['best_metric'] == float('inf')
        assert config['patience_counter'] == 0
        assert config['best_epoch'] == 0
    
    def test_setup_early_stopping_val_accuracy_monitor(self):
        """Test early stopping setup with val_accuracy monitor."""
        val_loader = Mock()
        
        config = setup_early_stopping(
            early_stopping=True, val_loader=val_loader,
            monitor='val_accuracy', patience=7, min_delta=0.005, verbose=False
        )
        
        assert config['enabled'] == True
        assert config['monitor'] == 'val_accuracy'
        assert config['best_metric'] == 0.0
        assert callable(config['is_better'])
    
    def test_setup_early_stopping_invalid_monitor(self):
        """Test early stopping setup with invalid monitor."""
        val_loader = Mock()
        
        with pytest.raises(ValueError, match="Unsupported monitor metric: invalid"):
            setup_early_stopping(
                early_stopping=True, val_loader=val_loader,
                monitor='invalid', patience=5, min_delta=0.01, verbose=False
            )
    
    def test_setup_early_stopping_is_better_val_loss(self):
        """Test is_better function for val_loss monitor."""
        val_loader = Mock()
        
        config = setup_early_stopping(
            early_stopping=True, val_loader=val_loader,
            monitor='val_loss', patience=5, min_delta=0.01, verbose=False
        )
        
        is_better = config['is_better']
        assert is_better(0.5, 0.6) == True  # Lower is better
        assert is_better(0.6, 0.5) == False  # Higher is worse
        assert is_better(0.59, 0.6) == False  # Not enough improvement
    
    def test_setup_early_stopping_is_better_val_accuracy(self):
        """Test is_better function for val_accuracy monitor."""
        val_loader = Mock()
        
        config = setup_early_stopping(
            early_stopping=True, val_loader=val_loader,
            monitor='val_accuracy', patience=5, min_delta=0.01, verbose=False
        )
        
        is_better = config['is_better']
        assert is_better(0.9, 0.8) == True  # Higher is better
        assert is_better(0.8, 0.9) == False  # Lower is worse
        assert is_better(0.809, 0.8) == False  # Not enough improvement


class TestEarlyStoppingInitiated:
    """Test early_stopping_initiated function."""
    
    def test_early_stopping_initiated_disabled(self):
        """Test early stopping when disabled."""
        model_state_dict = {'param': torch.randn(5, 5)}
        early_stopping_state = {'enabled': False}
        
        should_stop = early_stopping_initiated(
            model_state_dict, early_stopping_state, val_loss=0.5, val_acc=0.8,
            epoch=5, pbar=None, verbose=False, restore_best_weights=False
        )
        
        assert should_stop == False
    
    def test_early_stopping_initiated_improvement_val_loss(self, capsys):
        """Test early stopping with improvement in val_loss."""
        model_state_dict = {'param': torch.randn(5, 5)}
        early_stopping_state = {
            'enabled': True,
            'monitor': 'val_loss',
            'patience': 3,
            'min_delta': 0.01,
            'best_metric': 0.6,
            'is_better': lambda current, best: current < (best - 0.01),
            'patience_counter': 1,
            'best_epoch': 2,
            'best_weights': None
        }
        
        should_stop = early_stopping_initiated(
            model_state_dict, early_stopping_state, val_loss=0.5, val_acc=0.8,
            epoch=5, pbar=None, verbose=True, restore_best_weights=True
        )
        
        captured = capsys.readouterr()
        assert "New best val_loss: 0.5000" in captured.out
        assert should_stop == False
        assert early_stopping_state['best_metric'] == 0.5
        assert early_stopping_state['best_epoch'] == 5
        assert early_stopping_state['patience_counter'] == 0
        assert early_stopping_state['best_weights'] is not None
    
    def test_early_stopping_initiated_no_improvement(self, capsys):
        """Test early stopping with no improvement."""
        model_state_dict = {'param': torch.randn(5, 5)}
        early_stopping_state = {
            'enabled': True,
            'monitor': 'val_loss',
            'patience': 3,
            'min_delta': 0.01,
            'best_metric': 0.4,
            'is_better': lambda current, best: current < (best - 0.01),
            'patience_counter': 1,
            'best_epoch': 2,
            'best_weights': None
        }
        
        should_stop = early_stopping_initiated(
            model_state_dict, early_stopping_state, val_loss=0.5, val_acc=0.8,
            epoch=5, pbar=None, verbose=True, restore_best_weights=False
        )
        
        captured = capsys.readouterr()
        assert "No improvement for 2/3 epochs" in captured.out
        assert "best val_loss: 0.4000 at epoch 3" in captured.out
        assert should_stop == False
        assert early_stopping_state['patience_counter'] == 2
    
    def test_early_stopping_initiated_triggered(self):
        """Test early stopping when triggered."""
        model_state_dict = {'param': torch.randn(5, 5)}
        early_stopping_state = {
            'enabled': True,
            'monitor': 'val_loss',
            'patience': 3,
            'min_delta': 0.01,
            'best_metric': 0.4,
            'is_better': lambda current, best: current < (best - 0.01),
            'patience_counter': 2,
            'best_epoch': 2,
            'best_weights': {'param': torch.randn(5, 5)}
        }
        
        should_stop = early_stopping_initiated(
            model_state_dict, early_stopping_state, val_loss=0.5, val_acc=0.8,
            epoch=5, pbar=None, verbose=True, restore_best_weights=True
        )
        
        assert should_stop == True
        assert early_stopping_state['patience_counter'] == 3
    
    def test_early_stopping_initiated_val_accuracy_improvement(self):
        """Test early stopping with val_accuracy improvement."""
        model_state_dict = {'param': torch.randn(5, 5)}
        early_stopping_state = {
            'enabled': True,
            'monitor': 'val_accuracy',
            'patience': 3,
            'min_delta': 0.01,
            'best_metric': 0.8,
            'is_better': lambda current, best: current > (best + 0.01),
            'patience_counter': 1,
            'best_epoch': 2,
            'best_weights': None
        }
        
        should_stop = early_stopping_initiated(
            model_state_dict, early_stopping_state, val_loss=0.5, val_acc=0.85,
            epoch=5, pbar=None, verbose=False, restore_best_weights=False
        )
        
        assert should_stop == False
        assert early_stopping_state['best_metric'] == 0.85
        assert early_stopping_state['patience_counter'] == 0



class TestCreateProgressBar:
    """Test create_progress_bar function."""
    
    def test_create_progress_bar_verbose(self):
        """Test progress bar creation when verbose=True."""
        with patch('models2.common.model_helpers.tqdm') as mock_tqdm:
            mock_pbar = Mock()
            mock_tqdm.return_value = mock_pbar
            
            result = create_progress_bar(
                verbose=True, epoch=5, epochs=10, total_steps=100
            )
            
            assert result == mock_pbar
            mock_tqdm.assert_called_once_with(
                total=100,
                desc="Epoch 6/10",
                leave=True,
                dynamic_ncols=True,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
            )
    
    def test_create_progress_bar_not_verbose(self):
        """Test progress bar creation when verbose=False."""
        result = create_progress_bar(
            verbose=False, epoch=5, epochs=10, total_steps=100
        )
        
        assert result is None


class TestFinalizeProgressBar:
    """Test finalize_progress_bar function."""
    
    def test_finalize_progress_bar_none(self):
        """Test finalize progress bar when pbar is None."""
        # Should not raise any errors
        finalize_progress_bar(
            pbar=None, avg_train_loss=0.5, train_accuracy=0.8,
            val_loader=None, val_loss=0.0, val_acc=0.0,
            early_stopping_state={'enabled': False}, current_lr=0.01
        )
    
    def test_finalize_progress_bar_no_validation(self):
        """Test finalize progress bar without validation."""
        mock_pbar = Mock()
        early_stopping_state = {'enabled': False}
        
        finalize_progress_bar(
            pbar=mock_pbar, avg_train_loss=0.5, train_accuracy=0.8,
            val_loader=None, val_loss=0.0, val_acc=0.0,
            early_stopping_state=early_stopping_state, current_lr=0.01
        )
        
        expected_postfix = {
            'train_loss': '0.5000',
            'train_acc': '0.8000',
            'lr': '0.010000'
        }
        mock_pbar.set_postfix.assert_called_once_with(expected_postfix)
        mock_pbar.refresh.assert_called_once()
        mock_pbar.close.assert_called_once()
    
    def test_finalize_progress_bar_with_validation(self):
        """Test finalize progress bar with validation."""
        mock_pbar = Mock()
        early_stopping_state = {'enabled': False}
        
        finalize_progress_bar(
            pbar=mock_pbar, avg_train_loss=0.3, train_accuracy=0.9,
            val_loader=Mock(), val_loss=0.4, val_acc=0.85,
            early_stopping_state=early_stopping_state, current_lr=0.001
        )
        
        expected_postfix = {
            'train_loss': '0.3000',
            'train_acc': '0.9000',
            'val_loss': '0.4000',
            'val_acc': '0.8500',
            'lr': '0.001000'
        }
        mock_pbar.set_postfix.assert_called_once_with(expected_postfix)
    
    def test_finalize_progress_bar_early_stopping_triggered(self):
        """Test finalize progress bar with early stopping triggered."""
        mock_pbar = Mock()
        early_stopping_state = {
            'enabled': True,
            'patience_counter': 5,
            'patience': 5
        }
        
        finalize_progress_bar(
            pbar=mock_pbar, avg_train_loss=0.3, train_accuracy=0.9,
            val_loader=Mock(), val_loss=0.4, val_acc=0.85,
            early_stopping_state=early_stopping_state, current_lr=0.001
        )
        
        expected_postfix = {
            'train_loss': '0.3000',
            'train_acc': '0.9000',
            'val_loss': '0.4000',
            'val_acc': '0.8500',
            'early_stop': 'TRIGGERED',
            'lr': '0.001000'
        }
        mock_pbar.set_postfix.assert_called_once_with(expected_postfix)
    
    def test_finalize_progress_bar_early_stopping_patience(self):
        """Test finalize progress bar with early stopping patience counter."""
        mock_pbar = Mock()
        early_stopping_state = {
            'enabled': True,
            'patience_counter': 2,
            'patience': 5
        }
        
        finalize_progress_bar(
            pbar=mock_pbar, avg_train_loss=0.3, train_accuracy=0.9,
            val_loader=Mock(), val_loss=0.4, val_acc=0.85,
            early_stopping_state=early_stopping_state, current_lr=0.001
        )
        
        expected_postfix = {
            'train_loss': '0.3000',
            'train_acc': '0.9000',
            'val_loss': '0.4000',
            'val_acc': '0.8500',
            'patience': '2/5',
            'lr': '0.001000'
        }
        mock_pbar.set_postfix.assert_called_once_with(expected_postfix)
    
    def test_finalize_progress_bar_early_stopping_best(self):
        """Test finalize progress bar with early stopping best metric."""
        mock_pbar = Mock()
        early_stopping_state = {
            'enabled': True,
            'patience_counter': 0,
            'patience': 5,
            'best_metric': 0.123
        }
        
        finalize_progress_bar(
            pbar=mock_pbar, avg_train_loss=0.3, train_accuracy=0.9,
            val_loader=Mock(), val_loss=0.4, val_acc=0.85,
            early_stopping_state=early_stopping_state, current_lr=0.001
        )
        
        expected_postfix = {
            'train_loss': '0.3000',
            'train_acc': '0.9000',
            'val_loss': '0.4000',
            'val_acc': '0.8500',
            'best': '0.1230',
            'lr': '0.001000'
        }
        mock_pbar.set_postfix.assert_called_once_with(expected_postfix)


class TestUpdateHistory:
    """Test update_history function."""
    
    def test_update_history_basic(self):
        """Test basic history update."""
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rates': []
        }
        
        update_history(
            history, avg_train_loss=0.5, train_accuracy=0.8,
            val_loss=0.4, val_acc=0.85, current_lr=0.01,
            has_validation=False
        )
        
        assert history['train_loss'] == [0.5]
        assert history['train_accuracy'] == [0.8]
        assert history['val_loss'] == []  # Not added when has_validation=False
        assert history['val_accuracy'] == []  # Not added when has_validation=False
        assert history['learning_rates'] == [0.01]
    
    def test_update_history_with_validation(self):
        """Test history update with validation."""
        history = {
            'train_loss': [0.6],
            'train_accuracy': [0.7],
            'val_loss': [0.5],
            'val_accuracy': [0.75],
            'learning_rates': [0.02]
        }
        
        update_history(
            history, avg_train_loss=0.3, train_accuracy=0.9,
            val_loss=0.25, val_acc=0.95, current_lr=0.005,
            has_validation=True
        )
        
        assert history['train_loss'] == [0.6, 0.3]
        assert history['train_accuracy'] == [0.7, 0.9]
        assert history['val_loss'] == [0.5, 0.25]
        assert history['val_accuracy'] == [0.75, 0.95]
        assert history['learning_rates'] == [0.02, 0.005]
    
    def test_update_history_no_lr(self):
        """Test history update without learning rate."""
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rates': []
        }
        
        update_history(
            history, avg_train_loss=0.5, train_accuracy=0.8,
            val_loss=0.4, val_acc=0.85, current_lr=0.0,
            has_validation=True
        )
        
        assert history['train_loss'] == [0.5]
        assert history['train_accuracy'] == [0.8]
        assert history['val_loss'] == [0.4]
        assert history['val_accuracy'] == [0.85]
        assert history['learning_rates'] == []  # Not added when current_lr <= 0


class TestIntegration:
    """Integration tests for model helpers."""
    
    def test_full_early_stopping_workflow(self):
        """Test complete early stopping workflow."""
        val_loader = Mock()
        
        # Setup early stopping
        early_stopping_state = setup_early_stopping(
            early_stopping=True, val_loader=val_loader,
            monitor='val_loss', patience=2, min_delta=0.01, verbose=False
        )
        
        # First epoch - improvement
        model_state_dict = {'param': torch.randn(5, 5)}
        should_stop = early_stopping_initiated(
            model_state_dict, early_stopping_state, val_loss=0.5, val_acc=0.8,
            epoch=0, pbar=None, verbose=False, restore_best_weights=True
        )
        assert should_stop == False
        assert early_stopping_state['best_metric'] == 0.5
        
        # Second epoch - no improvement
        should_stop = early_stopping_initiated(
            model_state_dict, early_stopping_state, val_loss=0.6, val_acc=0.7,
            epoch=1, pbar=None, verbose=False, restore_best_weights=True
        )
        assert should_stop == False
        assert early_stopping_state['patience_counter'] == 1
        
        # Third epoch - no improvement, should trigger
        should_stop = early_stopping_initiated(
            model_state_dict, early_stopping_state, val_loss=0.7, val_acc=0.6,
            epoch=2, pbar=None, verbose=False, restore_best_weights=True
        )
        assert should_stop == True
        assert early_stopping_state['patience_counter'] == 2
    
    def test_scheduler_and_dataloader_integration(self):
        """Test scheduler setup and dataloader creation together."""
        # Setup scheduler
        optimizer = optim.SGD([torch.nn.Parameter(torch.randn(1))], lr=0.01)
        scheduler = setup_scheduler(optimizer, 'cosine', epochs=5, train_loader_len=50)
        assert isinstance(scheduler, CosineAnnealingLR)
        
        # Create dataloader
        X = torch.randn(100, 10)
        y = torch.randint(0, 2, (100,))
        dataloader = create_dataloader_from_tensors(X, y, batch_size=20, device=torch.device('cpu'))
        
        assert len(dataloader) == 5  # 100 samples / 20 batch_size
        
        # Test scheduler update
        update_scheduler(scheduler, val_loss=0.5)
        # Should not raise any errors
