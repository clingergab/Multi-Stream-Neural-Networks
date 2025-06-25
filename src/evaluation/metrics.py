"""Evaluation metrics for multi-stream neural networks."""

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np


class ModelEvaluator:
    """Comprehensive model evaluation."""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
    
    def evaluate(self, dataloader):
        """Evaluate model on given dataloader."""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_confidences = []
        
        with torch.no_grad():
            for batch in dataloader:
                color_input = batch['color'].to(self.device)
                brightness_input = batch['brightness'].to(self.device)
                targets = batch['target'].to(self.device)
                
                outputs = self.model(color_input, brightness_input)
                probabilities = F.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                confidences = torch.max(probabilities, dim=1)[0]
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())
        
        return self._compute_metrics(all_targets, all_predictions, all_confidences)
    
    def _compute_metrics(self, targets, predictions, confidences):
        """Compute comprehensive metrics."""
        accuracy = accuracy_score(targets, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(targets, predictions, average='weighted')
        conf_matrix = confusion_matrix(targets, predictions)
        
        return {
            'accuracy': accuracy * 100,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix.tolist(),
            'avg_confidence': np.mean(confidences),
            'std_confidence': np.std(confidences)
        }


def top_k_accuracy(outputs, targets, k=5):
    """Compute top-k accuracy."""
    _, pred = outputs.topk(k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
    return correct_k.mul_(100.0 / targets.size(0))


def calibration_error(confidences, predictions, targets, n_bins=10):
    """Compute Expected Calibration Error (ECE)."""
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower.item()) & (confidences <= bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        
        if prop_in_bin.item() > 0:
            accuracy_in_bin = (predictions[in_bin] == targets[in_bin]).float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece.item()