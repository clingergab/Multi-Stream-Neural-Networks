"""
Model helper utilities for training and management.

This module provides common functionality that can be shared across different model architectures.
All functions are standalone utilities that don't require model instances, making them reusable
across different model implementations.
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
from typing import Optional, Union, Dict, Any, Callable


def setup_scheduler(optimizer, scheduler_type: str, epochs: int, train_loader_len: int, **scheduler_kwargs):
    """
    Set up and return the learning rate scheduler based on the scheduler type.
    
    Args:
        optimizer: The optimizer instance
        scheduler_type: Type of scheduler ('cosine', 'onecycle', 'step', 'plateau', or None)
        epochs: Number of training epochs
        train_loader_len: Length of the training data loader
        **scheduler_kwargs: Additional arguments for the scheduler
        
    Returns:
        Scheduler instance or None if no scheduler requested
    """
    if not scheduler_type:
        return None
        
    if scheduler_type == 'cosine':
        # T_max should be total number of steps (epochs * batches_per_epoch)
        # since scheduler.step() is called per batch
        t_max = scheduler_kwargs.get('t_max', epochs * train_loader_len)
        return CosineAnnealingLR(optimizer, T_max=t_max)
    elif scheduler_type == 'onecycle':
        # For OneCycleLR, we need total number of steps (epochs * steps_per_epoch)
        steps_per_epoch = scheduler_kwargs.get('steps_per_epoch', train_loader_len)
        max_lr = scheduler_kwargs.get('max_lr', optimizer.param_groups[0]['lr'] * 10)
        pct_start = scheduler_kwargs.get('pct_start', 0.3)
        anneal_strategy = scheduler_kwargs.get('anneal_strategy', 'cos')
        div_factor = scheduler_kwargs.get('div_factor', 25.0)
        final_div_factor = scheduler_kwargs.get('final_div_factor', 1e4)
        
        # Create the OneCycleLR scheduler with calculated total_steps
        return OneCycleLR(
            optimizer, 
            max_lr=max_lr,
            total_steps=epochs * steps_per_epoch,
            pct_start=pct_start,
            anneal_strategy=anneal_strategy,
            div_factor=div_factor,
            final_div_factor=final_div_factor
        )
    elif scheduler_type == 'step':
        step_size = scheduler_kwargs.get('step_size', 30)
        gamma = scheduler_kwargs.get('gamma', 0.1)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == 'plateau':
        # Use scheduler_patience if provided, otherwise fall back to patience from scheduler_kwargs
        scheduler_patience = scheduler_kwargs.get('scheduler_patience', scheduler_kwargs.get('patience', 5))
        factor = scheduler_kwargs.get('factor', 0.5)
        mode = scheduler_kwargs.get('mode', 'min')  # 'min' for loss (default), 'max' for accuracy
        min_lr = scheduler_kwargs.get('min_lr', 1e-7)  # Minimum learning rate
        threshold = scheduler_kwargs.get('threshold', 1e-4)  # Minimum change to qualify as improvement
        cooldown = scheduler_kwargs.get('cooldown', 0)  # Cooldown period after LR reduction
        eps = scheduler_kwargs.get('eps', 1e-8)  # Minimal decay applied to lr
        return ReduceLROnPlateau(
            optimizer, mode=mode, patience=scheduler_patience, factor=factor, min_lr=min_lr, threshold=threshold, cooldown=cooldown, eps=eps
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

def update_scheduler(scheduler, val_loss: float) -> None:
    """
    Update the learning rate scheduler.
    
    Args:
        scheduler: The scheduler instance
        val_loss: Validation loss for plateau scheduler
    """
    if scheduler is not None:
        # Skip OneCycleLR as it's updated after each batch
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        elif not isinstance(scheduler, OneCycleLR):
            scheduler.step()

def print_epoch_progress(epoch: int, epochs: int, epoch_time: float, 
                       avg_train_loss: float, train_accuracy: float,
                       val_loss: float, val_acc: float, val_loader: bool) -> None:
    """
    Print training progress for the current epoch.
    
    Args:
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

def save_checkpoint(model_state_dict: Dict[str, torch.Tensor], 
                   optimizer_state_dict: Optional[Dict[str, Any]] = None,
                   scheduler_state_dict: Optional[Dict[str, Any]] = None,
                   path: str = None, 
                   history: Optional[Dict[str, Any]] = None) -> None:
    """
    Save model checkpoint.
    
    Args:
        model_state_dict: The model's state dictionary
        optimizer_state_dict: The optimizer's state dictionary (optional)
        scheduler_state_dict: The scheduler's state dictionary (optional)
        path: Path to save checkpoint
        history: Training history to save (optional)
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer_state_dict,
        'scheduler_state_dict': scheduler_state_dict,
    }
    
    # Add history if provided
    if history is not None:
        checkpoint['history'] = history
    
    torch.save(checkpoint, path)

def create_dataloader_from_tensors(X: torch.Tensor, 
                                y: torch.Tensor, 
                                batch_size: int = 32,
                                shuffle: bool = True,
                                device: Optional[torch.device] = None,
                                num_workers: Optional[int] = None,
                                pin_memory: Optional[bool] = None,
                                persistent_workers: Optional[bool] = None,
                                prefetch_factor: int = 2) -> DataLoader:
    """
    Create a GPU-optimized DataLoader from tensor data.
    
    Args:
        X: Input tensor data
        y: Target tensor data (optional for prediction)
        batch_size: Batch size for the DataLoader
        shuffle: Whether to shuffle the data
        device: Device to optimize DataLoader for (if None, will auto-detect)
        num_workers: Number of parallel data loading workers (if None, will auto-select based on device:
                    CUDA: 8, MPS: 2, CPU: 2)
        pin_memory: Whether to pin memory for faster CPU→GPU transfer (if None, True for CUDA, False otherwise)
        persistent_workers: Whether to keep workers alive between epochs (if None, True for CUDA, False otherwise)
        prefetch_factor: Number of batches to prefetch per worker
        
    Returns:
        DataLoader containing the tensor data with GPU optimizations
    """
    if y is not None:
        dataset = TensorDataset(X, y)
    else:
        dataset = TensorDataset(X)
    
    # Auto-detect device if not provided
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    
    # Set device-specific defaults if not provided
    if num_workers is None:
        if device.type == 'cuda':
            num_workers = 8  # CUDA can handle more workers efficiently
        elif device.type == 'mps':
            num_workers = 2  # MPS works better with fewer workers
        else:
            num_workers = 2  # Conservative default for CPU
    
    if pin_memory is None:
        pin_memory = device.type == 'cuda'  # Only beneficial for CUDA
    
    if persistent_workers is None:
        persistent_workers = device.type == 'cuda'  # Most beneficial for CUDA
    
    # Device-specific optimizations
    if device.type == 'cuda':
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers and num_workers > 0,
            prefetch_factor=prefetch_factor if num_workers > 0 else None
        )
    elif device.type == 'mps':
        # MPS works better with limited workers
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=min(num_workers, 2),  # MPS works better with fewer workers
            pin_memory=False,  # Pin memory not beneficial for MPS
            persistent_workers=False,  # MPS doesn't benefit from persistent workers
            prefetch_factor=prefetch_factor if num_workers > 0 else None
        )
    else:
        # CPU training
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=False,  # No GPU transfer needed
            persistent_workers=persistent_workers and num_workers > 0,
            prefetch_factor=prefetch_factor if num_workers > 0 else None
        )

def setup_early_stopping(early_stopping: bool, val_loader, monitor: str, 
                       patience: int, min_delta: float, verbose: bool) -> Dict[str, Any]:
    """
    Set up early stopping configuration and state.
    
    Args:
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

def early_stopping_initiated(model_state_dict: Dict[str, torch.Tensor], early_stopping_state: Dict[str, Any], 
                           val_loss: float, val_acc: float, epoch: int, pbar, verbose: bool, 
                           restore_best_weights: bool) -> bool:
    """
    Handle early stopping logic and return whether training should stop.
    
    Args:
        model_state_dict: The model's current state dictionary
        early_stopping_state: Early stopping state dictionary
        val_loss: Current validation loss
        val_acc: Current validation accuracy
        epoch: Current epoch number
        pbar: Progress bar object
        verbose: Whether to print progress
        restore_best_weights: Whether to save best weights for restoration
    
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
                k: v.cpu().clone() for k, v in model_state_dict.items()
            }
        
        if verbose and pbar is None:
            print(f"✅ New best {monitor}: {current_metric:.4f}")
    else:
        early_stopping_state['patience_counter'] += 1
        if verbose and pbar is None:
            print(f"⏳ No improvement for {early_stopping_state['patience_counter']}/{early_stopping_state['patience']} epochs (best {monitor}: {early_stopping_state['best_metric']:.4f} at epoch {early_stopping_state['best_epoch'] + 1})")
    
    # Check if we should stop early
    if early_stopping_state['patience_counter'] >= early_stopping_state['patience']:
        # Print early stopping info but don't restore weights here
        if verbose and pbar is None:
            print(f"🛑 Early stopping triggered after {epoch + 1} epochs")
            print(f"   Best {monitor}: {early_stopping_state['best_metric']:.4f} at epoch {early_stopping_state['best_epoch'] + 1}")
        return True
        
    return False

def create_progress_bar(verbose: bool, epoch: int, epochs: int, total_steps: int):
    """
    Create and configure progress bar for epoch.
    
    Args:
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

def finalize_progress_bar(pbar, avg_train_loss: float, train_accuracy: float,
                        val_loader, val_loss: float, val_acc: float, 
                        early_stopping_state: Dict[str, Any], current_lr: float) -> None:
    """
    Update and close progress bar with final epoch metrics.
    
    Args:
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

def update_history(history: Dict[str, Any], avg_train_loss: float, train_accuracy: float,
                  val_loss: float = 0.0, val_acc: float = 0.0, current_lr: float = 0.0,
                  has_validation: bool = False) -> None:
    """
    Update training history with current epoch metrics.
    
    Args:
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
