"""Mixing weight logging callback."""

import torch
import numpy as np


class MixingWeightLogger:
    """Log mixing weights during training."""
    
    def __init__(self, log_freq=25):
        self.log_freq = log_freq
        self.weight_history = {'alpha': [], 'beta': [], 'gamma': []}
    
    def __call__(self, model, metrics):
        """Log mixing weights."""
        epoch = metrics.get('epoch', 0)
        
        if epoch % self.log_freq == 0:
            weights = self._extract_mixing_weights(model)
            
            for param_name, values in weights.items():
                if values is not None:
                    self.weight_history[param_name].append({
                        'epoch': epoch,
                        'mean': values.mean().item(),
                        'std': values.std().item(),
                        'min': values.min().item(),
                        'max': values.max().item()
                    })
            
            self._print_weights(epoch, weights)
    
    def _extract_mixing_weights(self, model):
        """Extract mixing weights from model."""
        weights = {'alpha': None, 'beta': None, 'gamma': None}
        
        for name, param in model.named_parameters():
            if 'alpha' in name.lower():
                weights['alpha'] = param.data.clone()
            elif 'beta' in name.lower():
                weights['beta'] = param.data.clone()
            elif 'gamma' in name.lower():
                weights['gamma'] = param.data.clone()
        
        return weights
    
    def _print_weights(self, epoch, weights):
        """Print weight summary."""
        print(f"Epoch {epoch} - Mixing Weights:")
        for name, values in weights.items():
            if values is not None:
                print(f"  {name}: mean={values.mean():.4f}, std={values.std():.4f}")