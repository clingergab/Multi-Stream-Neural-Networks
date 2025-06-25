"""Training visualization callback."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


class TrainingVisualizer:
    """Visualize training progress."""
    
    def __init__(self, save_dir='results/training_plots', plot_freq=25):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.plot_freq = plot_freq
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.epochs = []
    
    def __call__(self, model, metrics):
        """Update and save training plots."""
        epoch = metrics.get('epoch', 0)
        train_loss = metrics.get('train_loss', 0.0)
        val_loss = metrics.get('val_loss', 0.0)
        val_accuracy = metrics.get('val_accuracy', 0.0)
        
        # Record metrics
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_accuracy)
        
        # Plot periodically
        if epoch % self.plot_freq == 0 and epoch > 0:
            self._plot_training_curves()
    
    def _plot_training_curves(self):
        """Plot training and validation curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Loss curves
        ax1.plot(self.epochs, self.train_losses, label='Train Loss', color='blue')
        ax1.plot(self.epochs, self.val_losses, label='Val Loss', color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curve
        ax2.plot(self.epochs, self.val_accuracies, label='Val Accuracy', color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_curves.png', dpi=150)
        plt.close()
        
        print(f"Training plots updated and saved to {self.save_dir}")