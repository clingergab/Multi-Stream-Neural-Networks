"""
GPU-optimized training utilities for multi-channel models.

This module provides optimized training functions that automatically handle
device management, mixed precision, and performance optimizations.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from typing import Optional, Dict, Any, Tuple
import time

from ..utils.device_utils import get_device_manager, print_memory_info, clear_gpu_cache


class MultiChannelTrainer:
    """
    GPU-optimized trainer for multi-channel models.
    
    Handles device management, mixed precision training, and performance monitoring.
    """
    
    def __init__(self, model: nn.Module, device: Optional[str] = None, 
                 enable_mixed_precision: bool = True):
        """
        Initialize the trainer.
        
        Args:
            model: MultiChannel model to train
            device: Target device (auto-detected if None)
            enable_mixed_precision: Whether to use automatic mixed precision
        """
        self.device_manager = get_device_manager(device)
        self.device = self.device_manager.device
        
        # Move and optimize model
        self.model = model
        if hasattr(model, 'to_device_optimized'):
            self.model.to_device_optimized()
        else:
            self.model.to(self.device)
            self.device_manager.optimize_for_device(self.model)
        
        # Setup mixed precision
        self.use_amp = (enable_mixed_precision and 
                       self.device_manager.enable_mixed_precision())
        self.scaler = GradScaler() if self.use_amp else None
        
        if self.use_amp:
            print("🚀 Automatic Mixed Precision (AMP) enabled")
        
        print(f"🎯 Trainer initialized on: {self.device}")
    
    def train_step(self, color_input: torch.Tensor, brightness_input: torch.Tensor, 
                   targets: torch.Tensor, optimizer: optim.Optimizer, 
                   criterion: nn.Module) -> Dict[str, float]:
        """
        Perform one optimized training step.
        
        Args:
            color_input: Color input batch
            brightness_input: Brightness input batch
            targets: Target labels
            optimizer: Optimizer
            criterion: Loss function
            
        Returns:
            Dictionary with loss and timing information
        """
        start_time = time.time()
        
        # Move data to device efficiently
        color_input = color_input.to(self.device, non_blocking=True)
        brightness_input = brightness_input.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Forward pass with optional mixed precision
        if self.use_amp:
            with autocast():
                outputs = self.model(color_input, brightness_input)
                loss = criterion(outputs, targets)
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            outputs = self.model(color_input, brightness_input)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        step_time = time.time() - start_time
        
        return {
            'loss': loss.item(),
            'step_time': step_time,
            'throughput': color_input.size(0) / step_time  # samples per second
        }
    
    def validate_step(self, color_input: torch.Tensor, brightness_input: torch.Tensor, 
                     targets: torch.Tensor, criterion: nn.Module) -> Dict[str, float]:
        """
        Perform one optimized validation step.
        
        Args:
            color_input: Color input batch
            brightness_input: Brightness input batch
            targets: Target labels
            criterion: Loss function
            
        Returns:
            Dictionary with loss and accuracy information
        """
        # Move data to device efficiently
        color_input = color_input.to(self.device, non_blocking=True)
        brightness_input = brightness_input.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        
        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    outputs = self.model(color_input, brightness_input)
                    loss = criterion(outputs, targets)
            else:
                outputs = self.model(color_input, brightness_input)
                loss = criterion(outputs, targets)
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == targets).sum().item()
            accuracy = correct / targets.size(0)
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy,
            'correct': correct,
            'total': targets.size(0)
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model and device information."""
        info = {}
        
        # Model information
        if hasattr(self.model, 'get_memory_usage'):
            info.update(self.model.get_memory_usage())
        else:
            total_params = sum(p.numel() for p in self.model.parameters())
            info['total_parameters'] = total_params
        
        # Device information
        info['device'] = str(self.device)
        info['device_type'] = self.device.type
        info['mixed_precision'] = self.use_amp
        
        # GPU memory information
        if self.device.type == 'cuda':
            gpu_info = self.device_manager.get_memory_info()
            info.update(gpu_info)
        
        return info
    
    def clear_cache_and_print_memory(self):
        """Clear GPU cache and print memory information."""
        if self.device.type in ['cuda', 'mps']:
            clear_gpu_cache()
            print_memory_info()
    
    def optimize_dataloader_settings(self, batch_size: int) -> Dict[str, Any]:
        """
        Get optimized DataLoader settings for the current device.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Dictionary with optimized DataLoader parameters
        """
        settings = {
            'batch_size': batch_size,
            'shuffle': True,
            'drop_last': True,  # Ensures consistent batch sizes for optimization
        }
        
        if self.device.type == 'cuda':
            # Optimize for CUDA
            settings.update({
                'num_workers': min(4, torch.cuda.device_count() * 2),  # 2 workers per GPU
                'pin_memory': True,  # Faster GPU transfer
                'persistent_workers': True,  # Keep workers alive between epochs
            })
        elif self.device.type == 'mps':
            # Optimize for MPS (Mac)
            settings.update({
                'num_workers': 2,  # Conservative for MPS
                'pin_memory': False,  # Not beneficial for MPS
            })
        else:
            # CPU settings
            settings.update({
                'num_workers': 2,
                'pin_memory': False,
            })
        
        return settings


def create_optimized_trainer(model: nn.Module, device: Optional[str] = None,
                           enable_mixed_precision: bool = True) -> MultiChannelTrainer:
    """
    Create a GPU-optimized trainer for multi-channel models.
    
    Args:
        model: MultiChannel model
        device: Target device (auto-detected if None)
        enable_mixed_precision: Whether to use automatic mixed precision
        
    Returns:
        Configured MultiChannelTrainer
    """
    return MultiChannelTrainer(model, device, enable_mixed_precision)


def get_optimized_optimizer(model: nn.Module, learning_rate: float = 0.001,
                          weight_decay: float = 1e-4) -> optim.Optimizer:
    """
    Get an optimized optimizer for multi-channel models.
    
    Args:
        model: Model to optimize
        learning_rate: Learning rate
        weight_decay: Weight decay
        
    Returns:
        Configured optimizer (AdamW for stability)
    """
    return optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )


def benchmark_model(model: nn.Module, input_shape: Tuple[int, ...] = (2, 3, 224, 224),
                   num_iterations: int = 100) -> Dict[str, float]:
    """
    Benchmark model performance on current device.
    
    Args:
        model: Model to benchmark
        input_shape: Input tensor shape (batch_size, channels, height, width)
        num_iterations: Number of iterations to run
        
    Returns:
        Benchmark results
    """
    device_manager = get_device_manager()
    device = device_manager.device
    
    # Prepare model and data
    model.eval()
    color_input = torch.randn(input_shape, device=device)
    brightness_input = torch.randn(input_shape, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(color_input, brightness_input)
    
    # Benchmark
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(color_input, brightness_input)
            torch.cuda.synchronize() if device.type == 'cuda' else None
    
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = total_time / num_iterations
    throughput = input_shape[0] / avg_time  # samples per second
    
    return {
        'avg_inference_time_ms': avg_time * 1000,
        'throughput_samples_per_sec': throughput,
        'total_time_sec': total_time,
        'device': str(device)
    }
