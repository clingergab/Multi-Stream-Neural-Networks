"""Gradient monitoring callback."""

import torch
import numpy as np


class GradientMonitor:
    """Monitor gradient flow during training."""
    
    def __init__(self, log_freq=10):
        self.log_freq = log_freq
        self.gradient_norms = []
    
    def __call__(self, model, metrics):
        """Called after each training step."""
        epoch = metrics.get('epoch', 0)
        
        if epoch % self.log_freq == 0:
            total_norm = 0.0
            param_count = 0
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
            
            total_norm = total_norm ** (1. / 2)
            self.gradient_norms.append(total_norm)
            
            print(f"Epoch {epoch}: Gradient norm: {total_norm:.6f}")