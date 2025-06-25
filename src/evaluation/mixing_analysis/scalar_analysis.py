"""Scalar mixing analysis."""

import torch
import numpy as np
import matplotlib.pyplot as plt


class ScalarMixingAnalyzer:
    """Analyze scalar mixing weights and their effects."""
    
    def __init__(self, model):
        self.model = model
    
    def analyze_weight_distribution(self):
        """Analyze the distribution of learned scalar weights."""
        weights = {}
        
        for name, param in self.model.named_parameters():
            if name in ['alpha', 'beta', 'gamma']:
                weights[name] = param.data.item()
        
        return weights
    
    def analyze_sensitivity(self, dataloader, weight_ranges=None):
        """Analyze sensitivity to weight changes."""
        if weight_ranges is None:
            weight_ranges = {
                'alpha': np.linspace(0.0, 1.0, 11),
                'beta': np.linspace(0.0, 1.0, 11),
                'gamma': np.linspace(0.0, 0.5, 6)
            }
        
        results = {}
        original_weights = {}
        
        # Store original weights
        for name, param in self.model.named_parameters():
            if name in ['alpha', 'beta', 'gamma']:
                original_weights[name] = param.data.clone()
        
        # Test each weight parameter
        for weight_name, values in weight_ranges.items():
            accuracies = []
            
            for value in values:
                # Set weight value
                for name, param in self.model.named_parameters():
                    if name == weight_name:
                        param.data.fill_(value)
                
                # Evaluate
                accuracy = self._evaluate_accuracy(dataloader)
                accuracies.append(accuracy)
            
            results[weight_name] = {
                'values': values.tolist(),
                'accuracies': accuracies
            }
            
            # Restore original weight
            for name, param in self.model.named_parameters():
                if name == weight_name:
                    param.data.copy_(original_weights[name])
        
        return results
    
    def _evaluate_accuracy(self, dataloader):
        """Quick accuracy evaluation."""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= 10:  # Quick evaluation
                    break
                    
                color_input = batch['color']
                brightness_input = batch['brightness']
                targets = batch['target']
                
                if torch.cuda.is_available():
                    color_input = color_input.cuda()
                    brightness_input = brightness_input.cuda()
                    targets = targets.cuda()
                
                outputs = self.model(color_input, brightness_input)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        return 100 * correct / total if total > 0 else 0.0
    
    def plot_weight_effects(self, sensitivity_results, save_path=None):
        """Plot the effects of weight changes."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, (weight_name, results) in enumerate(sensitivity_results.items()):
            axes[i].plot(results['values'], results['accuracies'], 'o-')
            axes[i].set_xlabel(f'{weight_name.capitalize()} Value')
            axes[i].set_ylabel('Accuracy (%)')
            axes[i].set_title(f'Sensitivity to {weight_name.capitalize()}')
            axes[i].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
        else:
            plt.show()
        
        plt.close()