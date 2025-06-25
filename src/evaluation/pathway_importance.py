"""Pathway importance analysis."""

import torch
import torch.nn.functional as F
import numpy as np


class PathwayAnalyzer:
    """Analyze the importance of different pathways."""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
    
    def analyze_pathway_importance(self, dataloader=None):
        """Analyze relative importance of color vs brightness pathways."""
        if dataloader is None:
            # Use synthetic data for analysis
            color_input = torch.randn(10, 3, 224, 224).to(self.device)
            brightness_input = torch.randn(10, 1, 224, 224).to(self.device)
            return self._analyze_synthetic(color_input, brightness_input)
        
        self.model.eval()
        color_importance = 0.0
        brightness_importance = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                color_input = batch['color'].to(self.device)
                brightness_input = batch['brightness'].to(self.device)
                
                # Analyze batch
                batch_importance = self._analyze_batch(color_input, brightness_input)
                color_importance += batch_importance['color'] * color_input.size(0)
                brightness_importance += batch_importance['brightness'] * color_input.size(0)
                total_samples += color_input.size(0)
        
        return {
            'color': color_importance / total_samples,
            'brightness': brightness_importance / total_samples
        }
    
    def _analyze_batch(self, color_input, brightness_input):
        """Analyze pathway importance for a single batch."""
        # Get baseline output
        baseline_output = self.model(color_input, brightness_input)
        
        # Zero out color pathway
        zero_color = torch.zeros_like(color_input)
        brightness_only_output = self.model(zero_color, brightness_input)
        
        # Zero out brightness pathway
        zero_brightness = torch.zeros_like(brightness_input)
        color_only_output = self.model(color_input, zero_brightness)
        
        # Calculate importance based on output change
        color_importance = F.mse_loss(baseline_output, brightness_only_output).item()
        brightness_importance = F.mse_loss(baseline_output, color_only_output).item()
        
        # Normalize
        total_importance = color_importance + brightness_importance
        if total_importance > 0:
            color_importance /= total_importance
            brightness_importance /= total_importance
        
        return {
            'color': color_importance,
            'brightness': brightness_importance
        }
    
    def _analyze_synthetic(self, color_input, brightness_input):
        """Quick analysis using synthetic data."""
        return self._analyze_batch(color_input, brightness_input)


def gradient_based_importance(model, color_input, brightness_input, target_class=None):
    """Compute pathway importance using gradients."""
    model.eval()
    
    color_input.requires_grad_(True)
    brightness_input.requires_grad_(True)
    
    output = model(color_input, brightness_input)
    
    if target_class is None:
        target_class = torch.argmax(output, dim=1)
    
    # Compute gradients
    loss = F.cross_entropy(output, target_class)
    loss.backward()
    
    # Calculate importance based on gradient magnitude
    color_grad_mag = torch.mean(torch.abs(color_input.grad))
    brightness_grad_mag = torch.mean(torch.abs(brightness_input.grad))
    
    total_grad = color_grad_mag + brightness_grad_mag
    
    return {
        'color': (color_grad_mag / total_grad).item(),
        'brightness': (brightness_grad_mag / total_grad).item()
    }