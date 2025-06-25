"""Regularization techniques for training."""

import torch
import torch.nn as nn


class DropPath(nn.Module):
    """Drop paths (connections) for regularization."""
    
    def __init__(self, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        
        keep_prob = 1 - self.drop_prob
        random_tensor = keep_prob + torch.rand_like(x)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class PathwayDropout(nn.Module):
    """Dropout specific to pathway features."""
    
    def __init__(self, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, color_features, brightness_features):
        if not self.training:
            return color_features, brightness_features
        
        # Randomly drop entire pathways during training
        if torch.rand(1) < self.drop_prob:
            color_features = torch.zeros_like(color_features)
        
        if torch.rand(1) < self.drop_prob:
            brightness_features = torch.zeros_like(brightness_features)
        
        return color_features, brightness_features


class MixingWeightRegularizer:
    """Regularization for mixing weights in direct mixing models."""
    
    def __init__(self, l1_weight=1e-4, l2_weight=1e-4):
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
    
    def __call__(self, model):
        """Compute regularization loss for mixing weights."""
        reg_loss = 0.0
        
        for name, param in model.named_parameters():
            if 'alpha' in name or 'beta' in name or 'gamma' in name:
                # L1 regularization
                reg_loss += self.l1_weight * torch.abs(param).sum()
                # L2 regularization
                reg_loss += self.l2_weight * torch.pow(param, 2).sum()
        
        return reg_loss