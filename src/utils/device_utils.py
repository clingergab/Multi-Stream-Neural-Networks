"""
Device utilities for GPU optimization.

This module handles device detection and provides utilities for optimal GPU usage
across different hardware (CUDA, MPS, CPU).
"""

import torch
from typing import Optional, Union


class DeviceManager:
    """Manages device selection and optimization for multi-channel models."""
    
    def __init__(self, preferred_device: Optional[str] = None):
        """
        Initialize device manager.
        
        Args:
            preferred_device: Optional preferred device ('cuda', 'mps', 'cpu')
        """
        self.device = self._get_optimal_device(preferred_device)
        self.device_type = self.device.type
        self._log_device_info()
    
    def _get_optimal_device(self, preferred: Optional[str] = None) -> torch.device:
        """
        Automatically detect and return the best available device.
        
        Priority order:
        1. User preference (if available)
        2. CUDA (if available)
        3. MPS (Mac M-series chips, if available)
        4. CPU (fallback)
        
        Args:
            preferred: Optional preferred device type
            
        Returns:
            torch.device: Best available device
        """
        # Check user preference first
        if preferred:
            if preferred == 'cuda' and torch.cuda.is_available():
                return torch.device('cuda')
            elif preferred == 'mps' and torch.backends.mps.is_available():
                return torch.device('mps')
            elif preferred == 'cpu':
                return torch.device('cpu')
            else:
                print(f"Warning: Preferred device '{preferred}' not available, auto-detecting...")
        
        # Auto-detect best device
        if torch.cuda.is_available():
            # Use the device with most memory if multiple GPUs available
            if torch.cuda.device_count() > 1:
                max_memory = 0
                best_device = 0
                for i in range(torch.cuda.device_count()):
                    memory = torch.cuda.get_device_properties(i).total_memory
                    if memory > max_memory:
                        max_memory = memory
                        best_device = i
                return torch.device(f'cuda:{best_device}')
            else:
                return torch.device('cuda')
        
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device('mps')
        
        else:
            return torch.device('cpu')
    
    def _log_device_info(self):
        """Log information about the selected device."""
        print(f"🚀 Device Manager initialized with: {self.device}")
        
        if self.device_type == 'cuda':
            gpu_name = torch.cuda.get_device_name(self.device)
            memory_gb = torch.cuda.get_device_properties(self.device).total_memory / 1e9
            print(f"   GPU: {gpu_name}")
            print(f"   Memory: {memory_gb:.1f} GB")
            print(f"   CUDA Version: {torch.version.cuda}")
            
        elif self.device_type == 'mps':
            print("   Apple Metal Performance Shaders (MPS) enabled")
            print("   Optimized for Mac M-series chips")
            
        else:
            print("   Using CPU (consider upgrading to GPU for better performance)")
    
    def to_device(self, tensor_or_model: Union[torch.Tensor, torch.nn.Module]) -> Union[torch.Tensor, torch.nn.Module]:
        """
        Move tensor or model to the optimal device.
        
        Args:
            tensor_or_model: Tensor or model to move
            
        Returns:
            Tensor or model on the optimal device
        """
        return tensor_or_model.to(self.device)
    
    def optimize_for_device(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Apply device-specific optimizations to the model.
        
        Args:
            model: Model to optimize
            
        Returns:
            Optimized model
        """
        # Move model to device
        model = model.to(self.device)
        
        # Apply device-specific optimizations
        if self.device_type == 'cuda':
            # Enable CUDA optimizations
            torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
            torch.backends.cudnn.deterministic = False  # Allow non-deterministic algorithms for speed
            
            # Enable mixed precision if supported
            if hasattr(torch.cuda, 'amp') and torch.cuda.get_device_capability(self.device)[0] >= 7:
                print("   ⚡ Mixed precision training available (Tensor Cores)")
            
        elif self.device_type == 'mps':
            # MPS-specific optimizations
            print("   ⚡ MPS optimizations enabled")
            
        return model
    
    def get_memory_info(self) -> dict:
        """
        Get current memory usage information.
        
        Returns:
            Dictionary with memory information
        """
        if self.device_type == 'cuda':
            allocated = torch.cuda.memory_allocated(self.device) / 1e9
            cached = torch.cuda.memory_reserved(self.device) / 1e9
            total = torch.cuda.get_device_properties(self.device).total_memory / 1e9
            
            return {
                'allocated_gb': allocated,
                'cached_gb': cached,
                'total_gb': total,
                'free_gb': total - cached,
                'utilization': (allocated / total) * 100
            }
        else:
            return {'message': f'Memory info not available for {self.device_type}'}
    
    def clear_cache(self):
        """Clear GPU cache if available."""
        if self.device_type == 'cuda':
            torch.cuda.empty_cache()
            print("🧹 CUDA cache cleared")
        elif self.device_type == 'mps':
            torch.mps.empty_cache()
            print("🧹 MPS cache cleared")
    
    def enable_mixed_precision(self) -> bool:
        """
        Check if mixed precision training is available.
        
        Returns:
            True if mixed precision is available
        """
        if self.device_type == 'cuda':
            return hasattr(torch.cuda, 'amp') and torch.cuda.get_device_capability(self.device)[0] >= 7
        return False


# Global device manager instance
_device_manager = None

def get_device_manager(preferred_device: Optional[str] = None) -> DeviceManager:
    """
    Get the global device manager instance.
    
    Args:
        preferred_device: Optional preferred device type
        
    Returns:
        DeviceManager instance
    """
    global _device_manager
    if _device_manager is None:
        _device_manager = DeviceManager(preferred_device)
    return _device_manager

def get_device() -> torch.device:
    """Get the current optimal device."""
    return get_device_manager().device

def to_device(tensor_or_model: Union[torch.Tensor, torch.nn.Module]) -> Union[torch.Tensor, torch.nn.Module]:
    """Move tensor or model to the optimal device."""
    return get_device_manager().to_device(tensor_or_model)

def optimize_model(model: torch.nn.Module) -> torch.nn.Module:
    """Apply device-specific optimizations to the model."""
    return get_device_manager().optimize_for_device(model)

def clear_gpu_cache():
    """Clear GPU cache."""
    get_device_manager().clear_cache()

def print_memory_info():
    """Print current GPU memory information."""
    info = get_device_manager().get_memory_info()
    if 'allocated_gb' in info:
        print(f"GPU Memory - Allocated: {info['allocated_gb']:.2f}GB, "
              f"Free: {info['free_gb']:.2f}GB, "
              f"Utilization: {info['utilization']:.1f}%")
    else:
        print(info['message'])
