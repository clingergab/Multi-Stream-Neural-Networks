"""Checkpointing callback."""

import torch
from pathlib import Path


class Checkpointer:
    """Save model checkpoints during training."""
    
    def __init__(self, save_dir='results/checkpoints', save_freq=50, save_best=True):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_freq = save_freq
        self.save_best = save_best
        self.best_accuracy = 0.0
    
    def __call__(self, model, metrics):
        """Save checkpoint based on frequency and performance."""
        epoch = metrics.get('epoch', 0)
        val_accuracy = metrics.get('val_accuracy', 0.0)
        
        # Save periodic checkpoint
        if epoch % self.save_freq == 0:
            self._save_checkpoint(model, metrics, f'checkpoint_epoch_{epoch}.pt')
        
        # Save best model
        if self.save_best and val_accuracy > self.best_accuracy:
            self.best_accuracy = val_accuracy
            self._save_checkpoint(model, metrics, 'best_model.pt')
            print(f"New best model saved! Accuracy: {val_accuracy:.2f}%")
    
    def _save_checkpoint(self, model, metrics, filename):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': metrics.get('epoch', 0),
            'model_state_dict': model.state_dict(),
            'metrics': metrics
        }
        
        save_path = self.save_dir / filename
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved: {save_path}")