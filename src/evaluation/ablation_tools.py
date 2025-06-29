"""Ablation study tools."""

import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy


class AblationStudy:
    """Tools for conducting ablation studies."""
    
    def __init__(self, model, dataloader):
        self.model = model
        self.dataloader = dataloader
        self.baseline_performance = None
    
    def run_pathway_ablation(self):
        """Ablate individual pathways to measure their importance."""
        print("Running pathway ablation study...")
        
        # Get baseline performance
        baseline_acc = self._evaluate_model(self.model)
        print(f"Baseline accuracy: {baseline_acc:.2f}%")
        
        results = {'baseline': baseline_acc}
        
        # Test without brightness pathway
        rgb_only_acc = self._evaluate_with_pathway_ablation('brightness')
        results['rgb_only'] = rgb_only_acc
        print(f"RGB pathway only: {rgb_only_acc:.2f}%")
        
        # Test without RGB pathway
        brightness_only_acc = self._evaluate_with_pathway_ablation('rgb')
        results['brightness_only'] = brightness_only_acc
        print(f"Brightness pathway only: {brightness_only_acc:.2f}%")
        
        # Compute importance scores
        results['rgb_importance'] = baseline_acc - brightness_only_acc
        results['brightness_importance'] = baseline_acc - rgb_only_acc
        
        return results
    
    def run_mixing_weight_ablation(self):
        """Ablate mixing weights to study their effects."""
        if not self._has_mixing_weights():
            print("Model does not have mixing weights for ablation.")
            return {}
        
        print("Running mixing weight ablation study...")
        
        baseline_acc = self._evaluate_model(self.model)
        results = {'baseline': baseline_acc}
        
        # Test with different weight configurations
        weight_configs = {
            'no_alpha': {'alpha': 0.0},
            'no_beta': {'beta': 0.0},
            'no_gamma': {'gamma': 0.0},
            'equal_weights': {'alpha': 0.5, 'beta': 0.5, 'gamma': 0.0},
            'no_interaction': {'gamma': 0.0}
        }
        
        for config_name, weight_values in weight_configs.items():
            acc = self._evaluate_with_weight_config(weight_values)
            results[config_name] = acc
            print(f"{config_name}: {acc:.2f}%")
        
        return results
    
    def run_architecture_ablation(self):
        """Ablate architectural components."""
        print("Running architecture ablation study...")
        
        baseline_acc = self._evaluate_model(self.model)
        results = {'baseline': baseline_acc}
        
        # Test simpler integration methods
        if hasattr(self.model, 'integration'):
            # Replace with simple concatenation
            original_integration = self.model.integration
            simple_integration = nn.Linear(
                original_integration.in_features if hasattr(original_integration, 'in_features') else 128,
                original_integration.out_features if hasattr(original_integration, 'out_features') else 128
            )
            
            self.model.integration = simple_integration
            simple_acc = self._evaluate_model(self.model)
            results['simple_integration'] = simple_acc
            
            # Restore original
            self.model.integration = original_integration
            print(f"Simple integration: {simple_acc:.2f}%")
        
        return results
    
    def _evaluate_model(self, model):
        """Evaluate model accuracy."""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.dataloader:
                rgb_input = batch['rgb']
                brightness_input = batch['brightness']
                targets = batch['target']
                
                if torch.cuda.is_available():
                    rgb_input = rgb_input.cuda()
                    brightness_input = brightness_input.cuda()
                    targets = targets.cuda()
                
                outputs = model(rgb_input, brightness_input)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        return 100 * correct / total if total > 0 else 0.0
    
    def _evaluate_with_pathway_ablation(self, ablate_pathway):
        """Evaluate with one pathway ablated."""
        model_copy = deepcopy(self.model)
        
        # Zero out the specified pathway
        if ablate_pathway == 'rgb':
            # Set RGB pathway to return zeros
            original_forward = model_copy.rgb_pathway.forward
            model_copy.rgb_pathway.forward = lambda x: torch.zeros_like(original_forward(x))
        elif ablate_pathway == 'brightness':
            # Set brightness pathway to return zeros
            original_forward = model_copy.brightness_pathway.forward
            model_copy.brightness_pathway.forward = lambda x: torch.zeros_like(original_forward(x))
        
        return self._evaluate_model(model_copy)
    
    def _evaluate_with_weight_config(self, weight_config):
        """Evaluate with specific weight configuration."""
        # Store original weights
        original_weights = {}
        for name, param in self.model.named_parameters():
            if any(w in name.lower() for w in weight_config.keys()):
                original_weights[name] = param.data.clone()
        
        # Set new weights
        for name, param in self.model.named_parameters():
            for weight_name, value in weight_config.items():
                if weight_name in name.lower():
                    param.data.fill_(value)
        
        # Evaluate
        accuracy = self._evaluate_model(self.model)
        
        # Restore original weights
        for name, param in self.model.named_parameters():
            if name in original_weights:
                param.data.copy_(original_weights[name])
        
        return accuracy
    
    def _has_mixing_weights(self):
        """Check if model has mixing weights."""
        for name, param in self.model.named_parameters():
            if any(w in name.lower() for w in ['alpha', 'beta', 'gamma']):
                return True
        return False