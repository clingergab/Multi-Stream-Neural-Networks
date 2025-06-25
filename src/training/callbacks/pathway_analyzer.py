"""Pathway analysis callback."""

import torch


class PathwayAnalyzer:
    """Analyze pathway contributions during training."""
    
    def __init__(self, analysis_freq=50):
        self.analysis_freq = analysis_freq
        self.pathway_importance = []
    
    def __call__(self, model, metrics):
        """Analyze pathway importance periodically."""
        epoch = metrics.get('epoch', 0)
        
        if epoch % self.analysis_freq == 0:
            importance = self._analyze_pathways(model)
            self.pathway_importance.append({
                'epoch': epoch,
                'color_importance': importance['color'],
                'brightness_importance': importance['brightness']
            })
            
            print(f"Epoch {epoch}: Pathway importance - "
                  f"Color: {importance['color']:.3f}, "
                  f"Brightness: {importance['brightness']:.3f}")
    
    def _analyze_pathways(self, model):
        """Quick pathway importance analysis."""
        # Create dummy inputs
        color_input = torch.randn(4, 3, 32, 32)
        brightness_input = torch.randn(4, 1, 32, 32)
        
        if torch.cuda.is_available():
            color_input = color_input.cuda()
            brightness_input = brightness_input.cuda()
        
        model.eval()
        with torch.no_grad():
            # Normal output
            normal_output = model(color_input, brightness_input)
            
            # Zero color pathway
            zero_color = torch.zeros_like(color_input)
            brightness_only = model(zero_color, brightness_input)
            
            # Zero brightness pathway  
            zero_brightness = torch.zeros_like(brightness_input)
            color_only = model(color_input, zero_brightness)
            
            # Calculate importance based on output change
            color_impact = torch.nn.functional.mse_loss(normal_output, brightness_only).item()
            brightness_impact = torch.nn.functional.mse_loss(normal_output, color_only).item()
            
            total_impact = color_impact + brightness_impact
            if total_impact > 0:
                return {
                    'color': color_impact / total_impact,
                    'brightness': brightness_impact / total_impact
                }
            else:
                return {'color': 0.5, 'brightness': 0.5}