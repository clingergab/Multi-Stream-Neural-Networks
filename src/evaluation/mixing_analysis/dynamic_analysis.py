"""Dynamic mixing analysis."""

import torch
import numpy as np


class DynamicMixingAnalyzer:
    """Analyze dynamic mixing behavior."""
    
    def __init__(self, model):
        self.model = model
    
    def analyze_weight_adaptation(self, dataloader, num_samples=100):
        """Analyze how weights adapt to different inputs."""
        if not hasattr(self.model, 'weight_network'):
            return None
        
        weight_samples = []
        self.model.eval()
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_samples // dataloader.batch_size:
                    break
                
                color_input = batch['color']
                brightness_input = batch['brightness']
                
                if torch.cuda.is_available():
                    color_input = color_input.cuda()
                    brightness_input = brightness_input.cuda()
                
                # Extract features and compute dynamic weights
                color_features = self.model.color_pathway(color_input)
                brightness_features = self.model.brightness_pathway(brightness_input)
                
                concat_features = torch.cat([color_features, brightness_features], dim=1)
                weights = self.model.weight_network(concat_features)
                
                weight_samples.append(weights.cpu().numpy())
        
        if weight_samples:
            all_weights = np.concatenate(weight_samples, axis=0)
            return self._analyze_weight_statistics(all_weights)
        
        return None
    
    def _analyze_weight_statistics(self, weights):
        """Compute statistics on dynamic weights."""
        n_params = weights.shape[1] // 3  # alpha, beta, gamma
        
        analysis = {}
        param_names = ['alpha', 'beta', 'gamma']
        
        for i, name in enumerate(param_names):
            param_weights = weights[:, i*n_params:(i+1)*n_params]
            
            analysis[name] = {
                'mean_across_samples': np.mean(param_weights, axis=0),
                'std_across_samples': np.std(param_weights, axis=0),
                'mean_adaptation': np.mean(np.std(param_weights, axis=1)),
                'correlation_with_input': self._compute_input_correlation(param_weights)
            }
        
        return analysis
    
    def _compute_input_correlation(self, weights):
        """Compute correlation between weights and input characteristics."""
        # Simplified correlation analysis
        return {
            'variability': np.std(weights, axis=1).mean(),
            'range': np.ptp(weights, axis=1).mean()
        }