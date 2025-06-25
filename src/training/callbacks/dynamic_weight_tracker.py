"""Dynamic weight tracking callback."""

import torch
import numpy as np
import json


class DynamicWeightTracker:
    """Track dynamic mixing weights during training."""
    
    def __init__(self, track_freq=50, save_path='results/dynamic_weights.json'):
        self.track_freq = track_freq
        self.save_path = save_path
        self.weight_samples = []
    
    def __call__(self, model, metrics):
        """Track dynamic weights."""
        epoch = metrics.get('epoch', 0)
        
        if epoch % self.track_freq == 0:
            if hasattr(model, 'weight_network'):
                samples = self._sample_dynamic_weights(model)
                self.weight_samples.append({
                    'epoch': epoch,
                    'samples': samples
                })
                
                self._save_samples()
                print(f"Epoch {epoch}: Tracked dynamic weight samples")
    
    def _sample_dynamic_weights(self, model):
        """Sample dynamic weights for different inputs."""
        samples = []
        
        # Generate various input samples
        for _ in range(5):
            color_input = torch.randn(1, 3, 32, 32)
            brightness_input = torch.randn(1, 1, 32, 32)
            
            if torch.cuda.is_available():
                color_input = color_input.cuda()
                brightness_input = brightness_input.cuda()
            
            model.eval()
            with torch.no_grad():
                # Get dynamic weights (this is model-specific)
                if hasattr(model, 'integrate_features'):
                    # For dynamic mixing models
                    color_features = model.color_pathway(color_input)
                    brightness_features = model.brightness_pathway(brightness_input)
                    
                    concat_features = torch.cat([color_features, brightness_features], dim=1)
                    weights = model.weight_network(concat_features)
                    
                    samples.append({
                        'alpha': weights[0, :weights.size(1)//3].cpu().numpy().tolist(),
                        'beta': weights[0, weights.size(1)//3:2*weights.size(1)//3].cpu().numpy().tolist(),
                        'gamma': weights[0, 2*weights.size(1)//3:].cpu().numpy().tolist()
                    })
        
        return samples
    
    def _save_samples(self):
        """Save weight samples to file."""
        try:
            with open(self.save_path, 'w') as f:
                json.dump(self.weight_samples, f, indent=2)
        except Exception as e:
            print(f"Failed to save dynamic weight samples: {e}")