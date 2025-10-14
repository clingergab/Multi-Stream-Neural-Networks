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
from tqdm import tqdm
from typing import Optional, Union, Dict, Any, Callable


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

def setup_stream_early_stopping(stream_early_stopping: bool, stream_monitor: str,
                               stream1_patience: int, stream2_patience: int,
                               stream_min_delta: float, verbose: bool = True) -> Dict[str, Any]:
    """
    Set up stream-specific early stopping state.

    When enabled, streams will freeze when they plateau and automatically restore
    their best weights before freezing. When all streams freeze, the full model's
    best weights are also restored.

    Args:
        stream_early_stopping: Whether to enable stream-specific early stopping
        stream_monitor: Metric to monitor ('val_loss' or 'val_accuracy')
        stream1_patience: Patience for Stream1 (RGB)
        stream2_patience: Patience for Stream2 (Depth)
        stream_min_delta: Minimum improvement to qualify as progress
        verbose: Whether to print setup messages

    Returns:
        Dictionary containing stream early stopping state
    """
    if not stream_early_stopping:
        return {'enabled': False}

    # Set up comparison logic based on monitor type
    if stream_monitor == 'val_loss':
        best_metric = float('inf')
        is_better = lambda current, best: current < (best - stream_min_delta)
    elif stream_monitor == 'val_accuracy':
        best_metric = 0.0
        is_better = lambda current, best: current > (best + stream_min_delta)
    else:
        raise ValueError(f"Invalid stream_monitor: {stream_monitor}. Must be 'val_loss' or 'val_accuracy'")

    if verbose:
        print(f"❄️  Stream-specific early stopping enabled:")
        print(f"   Monitor: {stream_monitor}")
        print(f"   Stream1 patience: {stream1_patience}, Stream2 patience: {stream2_patience}")
        print(f"   Min delta: {stream_min_delta}")
        print(f"   Restore best weights: Enabled (streams + full model when all frozen)")

    return {
        'enabled': True,
        'monitor': stream_monitor,
        'is_better': is_better,
        'stream1': {
            'best_metric': best_metric,
            'patience': stream1_patience,
            'patience_counter': 0,
            'best_epoch': 0,
            'frozen': False,
            'best_weights': None  # Will store stream-specific weights
        },
        'stream2': {
            'best_metric': best_metric,
            'patience': stream2_patience,
            'patience_counter': 0,
            'best_epoch': 0,
            'frozen': False,
            'best_weights': None  # Will store stream-specific weights
        },
        'min_delta': stream_min_delta,
        'all_frozen': False,
        # Full model state tracking (for when all streams freeze)
        'best_full_model': {
            'val_acc': 0.0,
            'epoch': 0,
            'weights': None  # Will store full model state_dict
        }
    }

def check_stream_early_stopping(stream_early_stopping_state: Dict[str, Any],
                                stream_stats: Dict[str, float],
                                model,
                                epoch: int,
                                monitor: str,
                                verbose: bool,
                                val_acc: float = 0.0,
                                val_loss: float = 0.0) -> bool:
    """
    Check stream-specific early stopping and freeze streams when they plateau.

    When a stream plateaus (no improvement for patience epochs), its best weights
    are restored and then its parameters are frozen, while integration weights remain
    trainable, allowing the model to continue learning from the other stream.

    When all streams freeze, training stops and the full model's best weights are restored.

    Args:
        stream_early_stopping_state: Stream early stopping state dictionary
        stream_stats: Dictionary with stream metrics (stream1_val_acc, stream2_val_acc,
                     stream1_val_loss, stream2_val_loss)
        model: The model (to access stream1 and stream2 parameters)
        epoch: Current epoch number
        verbose: Whether to print freeze messages
        val_acc: Full model validation accuracy (for tracking best overall performance)
        val_loss: Full model validation loss (for tracking best overall performance)

    Returns:
        True if all streams are frozen, False otherwise
    """
    if not stream_early_stopping_state['enabled']:
        return False

    monitor = stream_early_stopping_state['monitor']
    is_better = stream_early_stopping_state['is_better']

    # Track best full model performance (for restoration when all streams freeze)
    # Use the same monitor metric as stream early stopping
    best_full = stream_early_stopping_state['best_full_model']

    # Get current metric based on monitor type
    if monitor == 'val_loss':
        current_full_metric = val_loss
    else:  # val_accuracy
        current_full_metric = val_acc

    # Initialize best_metric if not exists (first call)
    if 'best_metric' not in best_full:
        best_full['best_metric'] = float('inf') if monitor == 'val_loss' else 0.0

    # Check if current is better than best
    if is_better(current_full_metric, best_full['best_metric']):
        best_full['best_metric'] = current_full_metric
        best_full['epoch'] = epoch
        # Save full model state
        best_full['weights'] = {
            k: v.cpu().clone() for k, v in model.state_dict().items()
        }

    # Check Stream1
    if not stream_early_stopping_state['stream1']['frozen']:
        # Get current metric based on monitor type
        if monitor == 'val_loss':
            stream1_metric = stream_stats.get('stream1_val_loss', float('inf'))
        else:  # val_accuracy
            stream1_metric = stream_stats.get('stream1_val_acc', 0.0)

        stream1_state = stream_early_stopping_state['stream1']

        if is_better(stream1_metric, stream1_state['best_metric']):
            # Improvement detected
            stream1_state['best_metric'] = stream1_metric
            stream1_state['best_epoch'] = epoch
            stream1_state['patience_counter'] = 0

            # Always save best stream1 weights
            stream1_state['best_weights'] = {
                name: param.data.cpu().clone()
                for name, param in model.named_parameters()
                if '.stream1_' in name
            }
        else:
            # No improvement
            stream1_state['patience_counter'] += 1

            if stream1_state['patience_counter'] > stream1_state['patience']:
                # Always restore best weights before freezing
                if stream1_state['best_weights'] is not None:
                    for name, param in model.named_parameters():
                        if '.stream1_' in name and name in stream1_state['best_weights']:
                            param.data.copy_(stream1_state['best_weights'][name].to(param.device))

                # Freeze Stream1
                stream1_state['frozen'] = True
                stream1_state['freeze_epoch'] = epoch  # Track actual freeze epoch

                # Freeze stream1 parameters (stream-specific weights only)
                # Integration weights remain trainable to allow rebalancing
                for name, param in model.named_parameters():
                    if '.stream1_' in name:
                        param.requires_grad = False

                if verbose:
                    metric_str = f"{monitor}: {stream1_state['best_metric']:.4f}"
                    print(f"❄️  Stream1 frozen (no improvement for {stream1_state['patience']} epochs, "
                          f"best {metric_str} at epoch {stream1_state['best_epoch'] + 1})")

    # Check Stream2
    if not stream_early_stopping_state['stream2']['frozen']:
        # Get current metric based on monitor type
        if monitor == 'val_loss':
            stream2_metric = stream_stats.get('stream2_val_loss', float('inf'))
        else:  # val_accuracy
            stream2_metric = stream_stats.get('stream2_val_acc', 0.0)

        stream2_state = stream_early_stopping_state['stream2']

        if is_better(stream2_metric, stream2_state['best_metric']):
            # Improvement detected
            stream2_state['best_metric'] = stream2_metric
            stream2_state['best_epoch'] = epoch
            stream2_state['patience_counter'] = 0

            # Always save best stream2 weights
            stream2_state['best_weights'] = {
                name: param.data.cpu().clone()
                for name, param in model.named_parameters()
                if '.stream2_' in name
            }
        else:
            # No improvement
            stream2_state['patience_counter'] += 1

            if stream2_state['patience_counter'] > stream2_state['patience']:
                # Always restore best weights before freezing
                if stream2_state['best_weights'] is not None:
                    for name, param in model.named_parameters():
                        if '.stream2_' in name and name in stream2_state['best_weights']:
                            param.data.copy_(stream2_state['best_weights'][name].to(param.device))

                # Freeze Stream2
                stream2_state['frozen'] = True
                stream2_state['freeze_epoch'] = epoch  # Track actual freeze epoch

                # Freeze stream2 parameters (stream-specific weights only)
                # Integration weights remain trainable to allow rebalancing
                for name, param in model.named_parameters():
                    if '.stream2_' in name:
                        param.requires_grad = False

                if verbose:
                    metric_str = f"{monitor}: {stream2_state['best_metric']:.4f}"
                    print(f"❄️  Stream2 frozen (no improvement for {stream2_state['patience']} epochs, "
                          f"best {metric_str} at epoch {stream2_state['best_epoch'] + 1})")

    # Check if all streams are frozen
    all_frozen = (stream_early_stopping_state['stream1']['frozen'] and
                  stream_early_stopping_state['stream2']['frozen'])

    # If all streams just became frozen, restore full model's best weights
    # BUT preserve the first frozen stream's weights (it stays at its best)
    if all_frozen and not stream_early_stopping_state['all_frozen']:
        best_full = stream_early_stopping_state['best_full_model']
        if best_full['weights'] is not None:
            # Determine which stream was frozen first (to preserve its weights)
            stream1_frozen_first = (stream_early_stopping_state['stream1'].get('freeze_epoch', float('inf')) <
                                   stream_early_stopping_state['stream2'].get('freeze_epoch', float('inf')))

            # Get the first-frozen stream's best weights to preserve them
            first_frozen_weights = {}
            preserved_stream = None
            if stream1_frozen_first and stream_early_stopping_state['stream1']['best_weights'] is not None:
                first_frozen_weights = stream_early_stopping_state['stream1']['best_weights'].copy()
                preserved_stream = "Stream1"
            elif not stream1_frozen_first and stream_early_stopping_state['stream2']['best_weights'] is not None:
                first_frozen_weights = stream_early_stopping_state['stream2']['best_weights'].copy()
                preserved_stream = "Stream2"

            # Restore full model best weights (includes the second-frozen stream + integration)
            model.load_state_dict({
                k: v.to(next(model.parameters()).device) for k, v in best_full['weights'].items()
            })

            # Restore first-frozen stream's weights back (preserve its best state)
            if first_frozen_weights:
                for name, param in model.named_parameters():
                    if name in first_frozen_weights:
                        param.data.copy_(first_frozen_weights[name].to(next(model.parameters()).device))

            if verbose:
                metric_str = f"{monitor}: {best_full['best_metric']:.4f}"
                if preserved_stream:
                    print(f"🔄 Restored full model best weights from epoch {best_full['epoch'] + 1} "
                          f"({metric_str}, preserved {preserved_stream})")
                else:
                    print(f"🔄 Restored full model best weights from epoch {best_full['epoch'] + 1} "
                          f"({metric_str})")

    stream_early_stopping_state['all_frozen'] = all_frozen

    return all_frozen

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
    if early_stopping_state['patience_counter'] > early_stopping_state['patience']:
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
        if early_stopping_state['patience_counter'] > early_stopping_state['patience']:
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
