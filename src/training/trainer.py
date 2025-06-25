"""Training utilities and trainer class."""

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
import time
import logging


class MultiStreamTrainer:
    """Trainer class for multi-stream neural networks."""
    
    def __init__(self, model, device='cuda', logger=None):
        self.model = model
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        
        # Move model to device
        self.model.to(device)
        
        # Training state
        self.current_epoch = 0
        self.best_accuracy = 0.0
        self.training_history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    
    def train_epoch(self, train_loader, optimizer, criterion):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            color_input = batch['color'].to(self.device)
            brightness_input = batch['brightness'].to(self.device)
            targets = batch['target'].to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(color_input, brightness_input)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate(self, val_loader, criterion):
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                color_input = batch['color'].to(self.device)
                brightness_input = batch['brightness'].to(self.device)
                targets = batch['target'].to(self.device)
                
                outputs = self.model(color_input, brightness_input)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(val_loader)
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, optimizer, criterion, num_epochs, 
              save_path=None, callbacks=None):
        """Full training loop."""
        callbacks = callbacks or []
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            start_time = time.time()
            
            # Training
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            
            # Validation
            val_loss, val_accuracy = self.validate(val_loader, criterion)
            
            # Update history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_accuracy'].append(val_accuracy)
            
            # Logging
            epoch_time = time.time() - start_time
            self.logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                           f"Train Loss: {train_loss:.4f}, "
                           f"Val Loss: {val_loss:.4f}, "
                           f"Val Acc: {val_accuracy:.2f}%, "
                           f"Time: {epoch_time:.2f}s")
            
            # Save best model
            if val_accuracy > self.best_accuracy:
                self.best_accuracy = val_accuracy
                if save_path:
                    self.save_checkpoint(save_path)
            
            # Run callbacks
            for callback in callbacks:
                callback.on_epoch_end(self, epoch, {
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy
                })
    
    def save_checkpoint(self, path):
        """Save model checkpoint."""
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'best_accuracy': self.best_accuracy,
            'training_history': self.training_history
        }, path)
    
    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_accuracy = checkpoint['best_accuracy']
        self.training_history = checkpoint['training_history']
