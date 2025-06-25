"""Loss functions for multi-stream neural networks."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiStreamLoss(nn.Module):
    """Combined loss for multi-stream training."""
    
    def __init__(self, classification_weight=1.0, pathway_consistency_weight=0.1):
        super().__init__()
        self.classification_weight = classification_weight
        self.pathway_consistency_weight = pathway_consistency_weight
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, outputs, targets, color_features=None, brightness_features=None):
        """Compute combined loss."""
        # Classification loss
        class_loss = self.ce_loss(outputs, targets)
        
        total_loss = self.classification_weight * class_loss
        
        # Pathway consistency loss (if features provided)
        if color_features is not None and brightness_features is not None:
            consistency_loss = self._pathway_consistency_loss(color_features, brightness_features)
            total_loss += self.pathway_consistency_weight * consistency_loss
        
        return total_loss
    
    def _pathway_consistency_loss(self, color_features, brightness_features):
        """Encourage consistent representations between pathways."""
        # L2 distance between pathway features
        diff = color_features - brightness_features
        return torch.mean(diff ** 2)


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()