"""
Model helper utilities for training and management.

This module provides common functionality that can be shared across different model architectures.
These are utility functions that take a model instance as the first parameter.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, 
    OneCycleLR, 
    StepLR, 
    ReduceLROnPlateau
)
from tqdm import tqdm
from typing import Optional, Union, Dict, Any


def setup_scheduler(model, epochs: int, train_loader_len: int, **scheduler_kwargs) -> None:
    """
    Set up the learning rate scheduler based on the scheduler type.
    
    Args:
        model: The model instance
        epochs: Number of training epochs
        train_loader_len: Length of the training data loader
        **scheduler_kwargs: Additional arguments for the scheduler
    """
    if not model.scheduler_type:
        model.scheduler = None
        return
        
    if model.scheduler_type == 'cosine':
        # Default t_max is set to number of epochs
        t_max = scheduler_kwargs.get('t_max', epochs)
        model.scheduler = CosineAnnealingLR(model.optimizer, T_max=t_max)
    elif model.scheduler_type == 'onecycle':
        # For OneCycleLR, we need total number of steps (epochs * steps_per_epoch)
        steps_per_epoch = scheduler_kwargs.get('steps_per_epoch', train_loader_len)
        max_lr = scheduler_kwargs.get('max_lr', model.optimizer.param_groups[0]['lr'] * 10)
        pct_start = scheduler_kwargs.get('pct_start', 0.3)
        anneal_strategy = scheduler_kwargs.get('anneal_strategy', 'cos')
        div_factor = scheduler_kwargs.get('div_factor', 25.0)
        final_div_factor = scheduler_kwargs.get('final_div_factor', 1e4)
        
        # Create the OneCycleLR scheduler with calculated total_steps
        model.scheduler = OneCycleLR(
            model.optimizer, 
            max_lr=max_lr,
            total_steps=epochs * steps_per_epoch,
            pct_start=pct_start,
            anneal_strategy=anneal_strategy,
            div_factor=div_factor,
            final_div_factor=final_div_factor
        )
    elif model.scheduler_type == 'step':
        step_size = scheduler_kwargs.get('step_size', 30)
        gamma = scheduler_kwargs.get('gamma', 0.1)
        model.scheduler = StepLR(model.optimizer, step_size=step_size, gamma=gamma)
    elif model.scheduler_type == 'plateau':
        # Use scheduler_patience if provided, otherwise fall back to patience from scheduler_kwargs
        scheduler_patience = scheduler_kwargs.get('scheduler_patience', scheduler_kwargs.get('patience', 10))
        factor = scheduler_kwargs.get('factor', 0.5)
        model.scheduler = ReduceLROnPlateau(
            model.optimizer, mode='min', patience=scheduler_patience, factor=factor
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {model.scheduler_type}")

def update_scheduler(model, val_loss: float) -> None:
    """
    Update the learning rate scheduler.
    
    Args:
        model: The model instance
        val_loss: Validation loss for plateau scheduler
    """
    if model.scheduler is not None:
        # Skip OneCycleLR as it's updated after each batch
        if isinstance(model.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            model.scheduler.step(val_loss)
        elif not isinstance(model.scheduler, OneCycleLR):
            model.scheduler.step()

def print_epoch_progress(model, epoch: int, epochs: int, epoch_time: float, 
                       avg_train_loss: float, train_accuracy: float,
                       val_loss: float, val_acc: float, val_loader: bool) -> None:
    """
    Print training progress for the current epoch.
    
    Args:
        model: The model instance (not used but kept for consistency)
        epoch: Current epoch number (0-based)
        epochs: Total number of epochs
        epoch_time: Time taken for this epoch
        avg_train_loss: Average training loss
        train_accuracy: Training accuracy
        val_loss: Validation loss
        val_acc: Validation accuracy
        val_loader: Whether validation loader was provided
    """
    print(f"Epoch {epoch+1}/{epochs} - "
          f"Time: {epoch_time:.2f}s - "
          f"Train Loss: {avg_train_loss:.4f} - "
          f"Train Acc: {train_accuracy*100:.2f}%", end="")
    
    if val_loader:
        print(f" - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%")
    else:
        print("")

def save_checkpoint(model, path: str, history: Optional[Dict[str, Any]] = None) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: The model instance
        path: Path to save checkpoint
        history: Training history to save (optional)
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': model.optimizer.state_dict() if model.optimizer else None,
        'scheduler_state_dict': model.scheduler.state_dict() if model.scheduler else None,
    }
    
    # Add history if provided
    if history is not None:
        checkpoint['history'] = history
    
    torch.save(checkpoint, path)

def create_dataloader_from_tensors(model, X: torch.Tensor, y: Optional[torch.Tensor] = None, 
                                 batch_size: int = 32, shuffle: bool = False) -> DataLoader:
    """
    Create a GPU-optimized DataLoader from tensor data.
    
    Args:
        model: The model instance
        X: Input tensor data
        y: Target tensor data (optional for prediction)
        batch_size: Batch size for the DataLoader
        shuffle: Whether to shuffle the data
        
    Returns:
        DataLoader containing the tensor data with GPU optimizations
    """
    if y is not None:
        dataset = TensorDataset(X, y)
    else:
        dataset = TensorDataset(X)
    
    # GPU-optimized DataLoader settings
    if model.device.type == 'cuda':
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=4,  # Use multiple workers for CUDA
            pin_memory=True,  # Pin memory for faster GPU transfer
            persistent_workers=True  # Keep workers alive between epochs
        )
    elif model.device.type == 'mps':
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=0,  # MPS works better with num_workers=0
            pin_memory=False  # Pin memory not beneficial for MPS
        )
    else:
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=0,  # CPU training
            pin_memory=False
        )

def setup_early_stopping(model, early_stopping: bool, val_loader, monitor: str, 
                       patience: int, min_delta: float, verbose: bool) -> Dict[str, Any]:
    """
    Set up early stopping configuration and state.
    
    Args:
        model: The model instance (not used but kept for consistency)
        early_stopping: Whether to enable early stopping
        val_loader: Validation data loader
        monitor: Metric to monitor ('val_loss' or 'val_accuracy')
        patience: Number of epochs to wait for improvement
        min_delta: Minimum change to qualify as improvement
        verbose: Whether to print early stopping info
    
    Returns:
        Dictionary containing early stopping state and configuration
    """
    if early_stopping and val_loader is None:
        if verbose:
            print("⚠️  Early stopping requested but no validation data provided. Disabling early stopping.")
        early_stopping = False
    
    if not early_stopping:
        return {'enabled': False}
    
    if monitor == 'val_loss':
        best_metric = float('inf')
        is_better = lambda current, best: current < (best - min_delta)
    elif monitor == 'val_accuracy':
        best_metric = 0.0
        is_better = lambda current, best: current > (best + min_delta)
    else:
        raise ValueError(f"Unsupported monitor metric: {monitor}. Use 'val_loss' or 'val_accuracy'.")
    
    if verbose:
        print(f"🛑 Early stopping enabled: monitoring {monitor} with patience={patience}, min_delta={min_delta}")
    
    return {
        'enabled': True,
        'monitor': monitor,
        'patience': patience,
        'min_delta': min_delta,
        'best_metric': best_metric,
        'is_better': is_better,
        'patience_counter': 0,
        'best_epoch': 0,
        'best_weights': None
    }

def early_stopping_initiated(model, early_stopping_state: Dict[str, Any], val_loss: float, val_acc: float,
                           epoch: int, pbar, verbose: bool, restore_best_weights: bool) -> bool:
    """
    Handle early stopping logic and return whether training should stop.
    
    Args:
        model: The model instance
        early_stopping_state: Early stopping state dictionary
        val_loss: Current validation loss
        val_acc: Current validation accuracy
        epoch: Current epoch number
        pbar: Progress bar object
        verbose: Whether to print progress
        restore_best_weights: Whether to restore best weights
    
    Returns:
        True if training should stop, False otherwise
    """
    if not early_stopping_state['enabled']:
        return False
        
    monitor = early_stopping_state['monitor']
    current_metric = val_loss if monitor == 'val_loss' else val_acc
    
    if early_stopping_state['is_better'](current_metric, early_stopping_state['best_metric']):
        early_stopping_state['best_metric'] = current_metric
        early_stopping_state['best_epoch'] = epoch
        early_stopping_state['patience_counter'] = 0
        
        # Save best weights for restoration
        if restore_best_weights:
            early_stopping_state['best_weights'] = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
        
        if verbose and pbar is None:
            print(f"✅ New best {monitor}: {current_metric:.4f}")
    else:
        early_stopping_state['patience_counter'] += 1
        if verbose and pbar is None:
            print(f"⏳ No improvement for {early_stopping_state['patience_counter']}/{early_stopping_state['patience']} epochs (best {monitor}: {early_stopping_state['best_metric']:.4f} at epoch {early_stopping_state['best_epoch'] + 1})")
    
    # Check if we should stop early
    if early_stopping_state['patience_counter'] >= early_stopping_state['patience']:
        # Handle early stopping termination
        finalize_early_stopping(model, early_stopping_state, epoch, pbar, verbose, restore_best_weights)
        return True
        
    return False

def finalize_early_stopping(model, early_stopping_state: Dict[str, Any], epoch: int, pbar, 
                          verbose: bool, restore_best_weights: bool) -> None:
    """
    Handle final early stopping procedures.
    
    Args:
        model: The model instance
        early_stopping_state: Early stopping state dictionary
        epoch: Current epoch number
        pbar: Progress bar object
        verbose: Whether to print progress
        restore_best_weights: Whether to restore best weights
    """
    monitor = early_stopping_state['monitor']
    
    if verbose and pbar is None:
        print(f"🛑 Early stopping triggered after {epoch + 1} epochs")
        print(f"   Best {monitor}: {early_stopping_state['best_metric']:.4f} at epoch {early_stopping_state['best_epoch'] + 1}")
    
    # Restore best weights if requested
    if restore_best_weights and early_stopping_state['best_weights'] is not None:
        model.load_state_dict({
            k: v.to(model.device) for k, v in early_stopping_state['best_weights'].items()
        })
        if verbose and pbar is None:
            print("🔄 Restored best model weights")

def create_progress_bar(model, verbose: bool, epoch: int, epochs: int, total_steps: int):
    """
    Create and configure progress bar for epoch.
    
    Args:
        model: The model instance (not used but kept for consistency)
        verbose: Whether to show progress bar
        epoch: Current epoch number (0-based)
        epochs: Total number of epochs
        total_steps: Total steps in the epoch
        
    Returns:
        Progress bar object or None if not verbose
    """
    if not verbose:
        return None
        
    return tqdm(
        total=total_steps,
        desc=f"Epoch {epoch+1}/{epochs}",
        leave=True,
        dynamic_ncols=True,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
    )

def finalize_progress_bar(model, pbar, avg_train_loss: float, train_accuracy: float,
                        val_loader, val_loss: float, val_acc: float, 
                        early_stopping_state: Dict[str, Any], current_lr: float) -> None:
    """
    Update and close progress bar with final epoch metrics.
    
    Args:
        model: The model instance (not used but kept for consistency)
        pbar: Progress bar object
        avg_train_loss: Average training loss
        train_accuracy: Training accuracy
        val_loader: Validation data loader
        val_loss: Validation loss
        val_acc: Validation accuracy
        early_stopping_state: Early stopping state dictionary
        current_lr: Current learning rate
    """
    if pbar is None:
        return
        
    final_postfix = {
        'train_loss': f'{avg_train_loss:.4f}',
        'train_acc': f'{train_accuracy:.4f}'
    }
    
    if val_loader:
        final_postfix.update({
            'val_loss': f'{val_loss:.4f}',
            'val_acc': f'{val_acc:.4f}'
        })
    
    # Add early stopping info to progress bar
    if early_stopping_state['enabled'] and val_loader is not None:
        if early_stopping_state['patience_counter'] >= early_stopping_state['patience']:
            final_postfix['early_stop'] = 'TRIGGERED'
        elif early_stopping_state['patience_counter'] > 0:
            final_postfix['patience'] = f"{early_stopping_state['patience_counter']}/{early_stopping_state['patience']}"
        else:
            final_postfix['best'] = f"{early_stopping_state['best_metric']:.4f}"
    
    # Add lr at the end
    final_postfix['lr'] = f'{current_lr:.6f}'
    
    pbar.set_postfix(final_postfix)
    pbar.refresh()
    pbar.close()

def update_history(model, history: Dict[str, Any], avg_train_loss: float, train_accuracy: float,
                  val_loss: float = 0.0, val_acc: float = 0.0, current_lr: float = 0.0,
                  has_validation: bool = False) -> None:
    """
    Update training history with current epoch metrics.
    
    Args:
        model: The model instance (not used but kept for consistency)
        history: Training history dictionary
        avg_train_loss: Average training loss
        train_accuracy: Training accuracy
        val_loss: Validation loss
        val_acc: Validation accuracy
        current_lr: Current learning rate
        has_validation: Whether validation was performed
    """
    history['train_loss'].append(avg_train_loss)
    history['train_accuracy'].append(train_accuracy)
    if has_validation:  # Only add validation metrics if validation was performed
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
    if current_lr > 0:
        history['learning_rates'].append(current_lr)
