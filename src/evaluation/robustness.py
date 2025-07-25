"""Robustness evaluation for multi-stream models."""

import torch
import torch.nn.functional as F
import numpy as np


class RobustnessEvaluator:
    """Evaluate model robustness to various perturbations."""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
    
    def evaluate_noise_robustness(self, dataloader, noise_levels=[0.01, 0.05, 0.1, 0.2]):
        """Evaluate robustness to Gaussian noise."""
        results = {}
        
        for noise_level in noise_levels:
            accuracies = []
            
            for batch in dataloader:
                rgb_input = batch['rgb'].to(self.device)
                brightness_input = batch['brightness'].to(self.device)
                targets = batch['target'].to(self.device)
                
                # Add noise
                rgb_noise = torch.randn_like(rgb_input) * noise_level
                brightness_noise = torch.randn_like(brightness_input) * noise_level
                
                noisy_rgb = torch.clamp(rgb_input + rgb_noise, 0, 1)
                noisy_brightness = torch.clamp(brightness_input + brightness_noise, 0, 1)
                
                with torch.no_grad():
                    outputs = self.model(noisy_rgb, noisy_brightness)
                    predictions = torch.argmax(outputs, dim=1)
                    accuracy = (predictions == targets).float().mean().item()
                    accuracies.append(accuracy)
            
            results[f'noise_{noise_level}'] = np.mean(accuracies) * 100
        
        return results
    
    def evaluate_adversarial_robustness(self, dataloader, epsilon=0.1, steps=10):
        """Evaluate robustness to adversarial attacks (PGD)."""
        self.model.eval()
        total_correct = 0
        total_samples = 0
        
        for batch in dataloader:
            rgb_input = batch['rgb'].to(self.device)
            brightness_input = batch['brightness'].to(self.device)
            targets = batch['target'].to(self.device)
            
            # Generate adversarial examples
            adv_rgb, adv_brightness = self._pgd_attack(
                rgb_input, brightness_input, targets, epsilon, steps
            )
            
            with torch.no_grad():
                outputs = self.model(adv_rgb, adv_brightness)
                predictions = torch.argmax(outputs, dim=1)
                total_correct += (predictions == targets).sum().item()
                total_samples += targets.size(0)
        
        return (total_correct / total_samples) * 100
    
    def _pgd_attack(self, rgb_input, brightness_input, targets, epsilon, steps):
        """Projected Gradient Descent attack."""
        alpha = epsilon / steps
        
        # Initialize adversarial examples
        adv_rgb = rgb_input.clone().detach().requires_grad_(True)
        adv_brightness = brightness_input.clone().detach().requires_grad_(True)
        
        for _ in range(steps):
            outputs = self.model(adv_rgb, adv_brightness)
            loss = F.cross_entropy(outputs, targets)
            
            loss.backward()
            
            # Update adversarial examples
            adv_rgb = adv_rgb.detach() + alpha * adv_rgb.grad.sign()
            adv_brightness = adv_brightness.detach() + alpha * adv_brightness.grad.sign()
            
            # Project back to valid range
            adv_rgb = torch.clamp(adv_rgb, 0, 1)
            adv_brightness = torch.clamp(adv_brightness, 0, 1)
            
            # Reset gradients
            adv_rgb.requires_grad_(True)
            adv_brightness.requires_grad_(True)
        
        return adv_rgb.detach(), adv_brightness.detach()