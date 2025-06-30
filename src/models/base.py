"""
Base classes for Multi-Stream Neural Networks
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler
from torch import optim
from tqdm import tqdm
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, List, Union, Optional


class BaseMultiStreamModel(nn.Module, ABC):
    """
    Base class for all Multi-Stream Neural Network models.
    
    Provides common functionality and interface for MSNN architectures.
    """
    
    def __init__(
        self,
        input_size: Tuple[int, ...],
        hidden_size: int,
        num_classes: int,
        **kwargs
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        # Track pathway gradients for analysis
        self.pathway_gradients = {}
        
        # Training state - will be set by subclasses
        self.is_compiled = False
        self.optimizer = None
        self.criterion = None
        self.metrics = []
        
    
    @abstractmethod
    def forward(self, color_input: torch.Tensor, brightness_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the multi-stream network.
        
        Args:
            color_input: Color/RGB input tensor
            brightness_input: Brightness/luminance input tensor
            
        Returns:
            Single tensor for training/classification (not tuple)
        """
        pass
        
    @abstractmethod
    def analyze_pathways(self, color_input: torch.Tensor, brightness_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Analyze individual pathway contributions for research purposes.
        
        Args:
            color_input: Color/RGB input tensor
            brightness_input: Brightness/luminance input tensor
            
        Returns:
            Tuple of (color_logits, brightness_logits) for analysis
        """
        pass
        
    @abstractmethod
    def extract_features(self, color_input: torch.Tensor, brightness_input: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Extract features before final classification.
        
        Args:
            color_input: Color/RGB input tensor
            brightness_input: Brightness/luminance input tensor
            
        Returns:
            Features tensor(s) before classification
        """
        pass
        
    @abstractmethod
    def get_pathway_importance(self) -> Dict[str, float]:
        """
        Calculate relative importance of different pathways.
        
        Returns:
            Dictionary mapping pathway names to importance scores
        """
        pass
    
    def extract_color_brightness(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract color (RGB) and brightness (L) channels from input.
        
        Args:
            x: Input tensor of shape (B, 4, H, W) containing RGB+L channels
            
        Returns:
            Tuple of (color_channels, brightness_channel)
        """
        if x.size(1) != 4:
            raise ValueError(f"Expected 4 channels (RGB+L), got {x.size(1)}")
            
        color_channels = x[:, :3, :, :]  # RGB channels
        brightness_channel = x[:, 3:4, :, :]  # Luminance channel
        
        return color_channels, brightness_channel

    # Common training methods that all models will share
    def compile(self, optimizer: str = 'adam', learning_rate: float = 0.001, 
                weight_decay: float = 0.0, loss: str = 'cross_entropy', metrics: List[str] = None):
        """
        Compile the model with optimizer and loss function (Keras-like API).
        
        Args:
            optimizer: Optimizer name ('adam', 'adamw', 'sgd', 'rmsprop')
            learning_rate: Learning rate
            weight_decay: Weight decay (L2 regularization)
            loss: Loss function name ('cross_entropy')
            metrics: List of metrics to track
        """
        if optimizer.lower() == 'adam':
            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer.lower() == 'adamw':
            self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer.lower() == 'sgd':
            self.optimizer = optim.SGD(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer.lower() == 'rmsprop':
            self.optimizer = optim.RMSprop(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
        
        if loss.lower() == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss function: {loss}")
        
        self.metrics = metrics or ['accuracy']
        self.is_compiled = True
        
        model_name = self.__class__.__name__
        print(f"{model_name} compiled with {optimizer} optimizer, {loss} loss, "
              f"learning rate: {learning_rate}, weight decay: {weight_decay}")

    @abstractmethod
    def fit(self, *args, **kwargs):
        """
        Fit the model to data. Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement fit() method.")
    
    # def fit_dataloader(
    #     self,
    #     train_loader: "DataLoader",
    #     val_loader: Optional["DataLoader"] = None,
    #     epochs: int = 10,
    #     early_stopping_patience: int = 5,
    #     scheduler_type: str = 'cosine',
    #     min_lr: float = 1e-6,
    #     verbose: int = 1
    # ) -> Dict[str, List[float]]:
    #     """
    #     Fit the model using pre-configured DataLoaders (for augmented training).
        
    #     This method provides a unified training interface for all model types
    #     and handles augmented DataLoaders seamlessly.
        
    #     Args:
    #         train_loader: Training DataLoader (with optional augmentation)
    #         val_loader: Validation DataLoader (optional)
    #         epochs: Number of epochs to train
    #         early_stopping_patience: Patience for early stopping
    #         scheduler_type: Learning rate scheduler type ('cosine', 'step', 'none')
    #         min_lr: Minimum learning rate for cosine annealing
    #         verbose: Verbosity level (0: silent, 1: progress bar, 2: detailed)
            
    #     Returns:
    #         Training history dictionary with losses and accuracies
    #     """
    #     if not self.is_compiled:
    #         raise RuntimeError("Model must be compiled before training. Call model.compile() first.")
                
    #     if verbose > 0:
    #         model_name = self.__class__.__name__
    #         print(f"🚀 Training {model_name} with enhanced DataLoader pipeline:")
    #         print(f"   Device: {getattr(self, 'device', 'cpu')}")
    #         print(f"   Mixed precision: {getattr(self, 'use_mixed_precision', False)}")
    #         print(f"   Train batches: {len(train_loader)}")
    #         if val_loader:
    #             print(f"   Val batches: {len(val_loader)}")
    #         print(f"   Scheduler: {scheduler_type}")
        
    #     # Setup learning rate scheduler
    #     if scheduler_type.lower() == 'cosine':
    #         scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=min_lr)
    #     elif scheduler_type.lower() == 'step':
    #         scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=epochs//3, gamma=0.1)
    #     else:
    #         scheduler = None
        
    #     # Mixed precision support
    #     use_mixed_precision = getattr(self, 'use_mixed_precision', False)
    #     scaler = GradScaler() if use_mixed_precision else None
        
    #     # Training history
    #     history = {
    #         'train_loss': [],
    #         'train_accuracy': [],
    #         'val_loss': [],
    #         'val_accuracy': []
    #     }
        
    #     # Training loop with enhanced tracking
    #     best_val_loss = float('inf')
    #     patience_counter = 0
    #     device = getattr(self, 'device', torch.device('cpu'))
        
    #     for epoch in range(epochs):
    #         # Single progress bar for training batches only
    #         if verbose == 1:
    #             epoch_pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            
    #         # Training phase
    #         self.train()
    #         total_loss = 0.0
    #         train_correct = 0
    #         train_total = 0
            
    #         for batch_idx, (batch_color, batch_brightness, batch_labels) in enumerate(train_loader):
    #             # Move data to device
    #             batch_color = batch_color.to(device, non_blocking=True)
    #             batch_brightness = batch_brightness.to(device, non_blocking=True)
    #             batch_labels = batch_labels.to(device, non_blocking=True)
                
    #             self.optimizer.zero_grad()
                
    #             if use_mixed_precision and scaler:
    #                 with torch.cuda.amp.autocast():
    #                     outputs = self(batch_color, batch_brightness)
    #                     loss = self.criterion(outputs, batch_labels)
                    
    #                 scaler.scale(loss).backward()
    #                 scaler.step(self.optimizer)
    #                 scaler.update()
    #             else:
    #                 outputs = self(batch_color, batch_brightness)
    #                 loss = self.criterion(outputs, batch_labels)
    #                 loss.backward()
    #                 self.optimizer.step()
                
    #             total_loss += loss.item()
                
    #             # Calculate training accuracy
    #             _, predicted = torch.max(outputs.data, 1)
    #             train_total += batch_labels.size(0)
    #             train_correct += (predicted == batch_labels).sum().item()
                
    #             # Update progress bar with current training metrics
    #             if verbose == 1:
    #                 train_acc = train_correct / train_total
    #                 epoch_pbar.set_postfix({
    #                     'Loss': f'{total_loss/(batch_idx+1):.4f}',
    #                     'Acc': f'{train_acc:.4f}'
    #                 })
    #                 epoch_pbar.update(1)
            
    #         if scheduler:
    #             scheduler.step()
            
    #         avg_train_loss = total_loss / len(train_loader)
    #         train_accuracy = train_correct / train_total
            
    #         # Store training metrics
    #         history['train_loss'].append(avg_train_loss)
    #         history['train_accuracy'].append(train_accuracy)
            
    #         # Validation phase
    #         if val_loader is not None:
    #             self.eval()
    #             total_val_loss = 0.0
    #             val_correct = 0
    #             val_total = 0
                
    #             with torch.no_grad():
    #                 for batch_idx, (batch_color, batch_brightness, batch_labels) in enumerate(val_loader):
    #                     # Move data to device
    #                     batch_color = batch_color.to(device, non_blocking=True)
    #                     batch_brightness = batch_brightness.to(device, non_blocking=True)
    #                     batch_labels = batch_labels.to(device, non_blocking=True)
                        
    #                     if use_mixed_precision:
    #                         with torch.cuda.amp.autocast():
    #                             outputs = self(batch_color, batch_brightness)
    #                             loss = self.criterion(outputs, batch_labels)
    #                     else:
    #                         outputs = self(batch_color, batch_brightness)
    #                         loss = self.criterion(outputs, batch_labels)
    #                     total_val_loss += loss.item()
                        
    #                     # Calculate validation accuracy
    #                     _, predicted = torch.max(outputs.data, 1)
    #                     val_total += batch_labels.size(0)
    #                     val_correct += (predicted == batch_labels).sum().item()
                
    #             avg_val_loss = total_val_loss / len(val_loader)
    #             val_accuracy = val_correct / val_total
                
    #             # Store validation metrics
    #             history['val_loss'].append(avg_val_loss)
    #             history['val_accuracy'].append(val_accuracy)
                
    #             # Update progress bar with final validation metrics
    #             if verbose == 1:
    #                 epoch_pbar.set_postfix({
    #                     'Loss': f'{avg_train_loss:.4f}',
    #                     'Acc': f'{train_accuracy:.4f}',
    #                     'Val_Loss': f'{avg_val_loss:.4f}',
    #                     'Val_Acc': f'{val_accuracy:.4f}'
    #                 })
    #         else:
    #             # Training only - final update
    #             if verbose == 1:
    #                 epoch_pbar.set_postfix({
    #                     'Loss': f'{avg_train_loss:.4f}',
    #                     'Acc': f'{train_accuracy:.4f}'
    #                 })
    #             avg_val_loss = float('inf')  # For early stopping logic
    #             history['val_loss'].append(float('nan'))
    #             history['val_accuracy'].append(float('nan'))
            
    #         # Close progress bar
    #         if verbose == 1:
    #             epoch_pbar.close()
            
    #         # Early stopping check (only if validation data provided)
    #         if val_loader is not None:
    #             if avg_val_loss < best_val_loss:
    #                 best_val_loss = avg_val_loss
    #                 patience_counter = 0
    #                 # Save the best model
    #                 self.save_model()
    #             else:
    #                 patience_counter += 1
                
    #             if patience_counter >= early_stopping_patience:
    #                 if verbose > 0:
    #                     print(f"Early stopping triggered. Stopping training at epoch {epoch + 1}.")
    #                 break
            
    #         # Print epoch summary
    #         if verbose > 0:
    #             lr = self.optimizer.param_groups[0]['lr']
    #             if val_loader is not None:
    #                 print(f"Epoch {epoch + 1}/{epochs} - "
    #                       f"Train Loss: {avg_train_loss:.4f} - Train Acc: {train_accuracy:.4f} - "
    #                       f"Val Loss: {avg_val_loss:.4f} - Val Acc: {val_accuracy:.4f} - "
    #                       f"LR: {lr:.6f}")
    #             else:
    #                 print(f"Epoch {epoch + 1}/{epochs} - "
    #                       f"Train Loss: {avg_train_loss:.4f} - Train Acc: {train_accuracy:.4f} - "
    #                       f"LR: {lr:.6f}")
        
    #     # Clear cache after training (if device manager exists)
    #     if hasattr(self, 'device_manager'):
    #         self.device_manager.clear_cache()
        
    #     return history

    def save_model(self, file_path: str = None):
        """Save the model parameters to a file."""
        if file_path is None:
            model_name = self.__class__.__name__.lower()
            file_path = f"best_{model_name}_model.pth"
        torch.save(self.state_dict(), file_path)
        if hasattr(self, 'verbose') and self.verbose:
            print(f"Model parameters saved to {file_path}.")
    
    def load_model(self, file_path: str = None):
        """Load model parameters from a file."""
        if file_path is None:
            model_name = self.__class__.__name__.lower()
            file_path = f"best_{model_name}_model.pth"
        
        device = getattr(self, 'device', torch.device('cpu'))
        self.load_state_dict(torch.load(file_path, map_location=device))
        self.to(device)
        if hasattr(self, 'verbose') and self.verbose:
            print(f"Model parameters loaded from {file_path}.")

    def predict(self, color_data: Union[np.ndarray, torch.Tensor], brightness_data: Union[np.ndarray, torch.Tensor], 
                batch_size: int = None) -> np.ndarray:
        """
        Make predictions on new data with GPU optimizations.
        
        Args:
            color_data: Color input data
            brightness_data: Brightness input data
            batch_size: Batch size for prediction (auto-detected if None)
            
        Returns:
            Predicted class labels
        """
        # Auto-detect optimal batch size for inference
        if batch_size is None:
            if self.device.type == 'cuda':
                batch_size = 512  # Larger batches for inference
            else:
                batch_size = 128
        
        self.eval()
        
        # Convert to tensors if needed
        if isinstance(color_data, np.ndarray):
            color_tensor = torch.tensor(color_data, dtype=torch.float32)
        else:
            color_tensor = color_data
            
        if isinstance(brightness_data, np.ndarray):
            brightness_tensor = torch.tensor(brightness_data, dtype=torch.float32)
        else:
            brightness_tensor = brightness_data
        
        # Move to device efficiently
        color_tensor = color_tensor.to(self.device, non_blocking=True)
        brightness_tensor = brightness_tensor.to(self.device, non_blocking=True)
        
        # Create dataset and loader with optimizations
        dataset = TensorDataset(color_tensor, brightness_tensor)
        loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False,
            pin_memory=self.device.type == 'cuda'
        )
        
        predictions = []
        with torch.no_grad():
            for batch_color, batch_brightness in loader:
                if self.use_mixed_precision:
                    with autocast(device_type='cuda'):
                        outputs = self(batch_color, batch_brightness)
                else:
                    outputs = self(batch_color, batch_brightness)
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.cpu().numpy())
        
        return np.array(predictions)

    def predict_proba(self, color_data: Union[np.ndarray, torch.Tensor], brightness_data: Union[np.ndarray, torch.Tensor], 
                      batch_size: int = 32) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            color_data: Color input data
            brightness_data: Brightness input data
            batch_size: Batch size for prediction
            
        Returns:
            Prediction probabilities
        """
        self.eval()
        
        # Convert to tensors if needed
        if isinstance(color_data, np.ndarray):
            color_tensor = torch.tensor(color_data, dtype=torch.float32)
        else:
            color_tensor = color_data
            
        if isinstance(brightness_data, np.ndarray):
            brightness_tensor = torch.tensor(brightness_data, dtype=torch.float32)
        else:
            brightness_tensor = brightness_data
        
        # Move to device
        color_tensor = color_tensor.to(self.device)
        brightness_tensor = brightness_tensor.to(self.device)
        
        # Create dataset and loader
        dataset = TensorDataset(color_tensor, brightness_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        probabilities = []
        with torch.no_grad():
            for batch_color, batch_brightness in loader:
                outputs = self(batch_color, batch_brightness)
                probs = torch.softmax(outputs, dim=1)
                probabilities.extend(probs.cpu().numpy())
        
        return np.array(probabilities)

    def evaluate(self, test_color_data: Union[np.ndarray, torch.Tensor], 
                 test_brightness_data: Union[np.ndarray, torch.Tensor], 
                 test_labels: Union[np.ndarray, torch.Tensor], batch_size: int = 32) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            test_color_data: Test color data
            test_brightness_data: Test brightness data
            test_labels: Test labels
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.eval()
        
        # Convert to tensors if needed and move to device
        if isinstance(test_color_data, np.ndarray):
            color_tensor = torch.tensor(test_color_data, dtype=torch.float32).to(self.device)
        else:
            color_tensor = test_color_data.to(self.device)
            
        if isinstance(test_brightness_data, np.ndarray):
            brightness_tensor = torch.tensor(test_brightness_data, dtype=torch.float32).to(self.device)
        else:
            brightness_tensor = test_brightness_data.to(self.device)
            
        if isinstance(test_labels, np.ndarray):
            labels_tensor = torch.tensor(test_labels, dtype=torch.long).to(self.device)
        else:
            labels_tensor = test_labels.to(self.device)
        
        # Create dataset and loader
        dataset = TensorDataset(color_tensor, brightness_tensor, labels_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_color, batch_brightness, batch_labels in loader:
                outputs = self(batch_color, batch_brightness)
                loss = criterion(outputs, batch_labels)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        
        accuracy = correct / total
        avg_loss = total_loss / len(loader)
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'correct': correct,
            'total': total
        }
    
    def register_pathway_hooks(self):
        """Register hooks to monitor gradient flow through pathways."""
        def make_hook(name):
            def hook(grad):
                self.pathway_gradients[name] = grad.clone()
                return grad
            return hook
            
        # Register hooks for pathway analysis
        # Subclasses should override to register specific pathway hooks
        pass
    
    def get_model_stats(self) -> Dict[str, Any]:
        """
        Get model statistics including parameter counts and pathway information.
        
        Returns:
            Dictionary containing model statistics
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_classes': self.num_classes,
            'model_type': self.__class__.__name__
        }


class IntegrationMixin:
    """
    Mixin providing common integration functionality for multi-stream models.
    
    This mixin provides utility methods for pathway integration, gradient monitoring,
    and parameter management that can be shared across different integration strategies.
    """
    
    def register_integration_hooks(self):
        """Register hooks to monitor gradients and activations during training."""
        self.integration_hooks = {}
        
        # Register hooks for integration parameters
        for name, param in self.named_parameters():
            if any(key in name.lower() for key in ['alpha', 'beta', 'gamma', 'mixing']):
                hook = param.register_hook(
                    lambda grad, param_name=name: self._track_integration_gradient(param_name, grad)
                )
                self.integration_hooks[name] = hook
    
    def _track_integration_gradient(self, param_name: str, grad: torch.Tensor):
        """Track gradients for integration parameters."""
        if not hasattr(self, 'integration_gradients'):
            self.integration_gradients = {}
        
        self.integration_gradients[param_name] = {
            'norm': grad.norm().item(),
            'mean': grad.mean().item(),
            'std': grad.std().item(),
            'max': grad.max().item(),
            'min': grad.min().item()
        }
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get statistics about integration parameters and their gradients."""
        stats = {}
        
        # Parameter statistics
        for name, param in self.named_parameters():
            if any(key in name.lower() for key in ['alpha', 'beta', 'gamma', 'mixing']):
                stats[f'{name}_stats'] = {
                    'mean': param.data.mean().item(),
                    'std': param.data.std().item(),
                    'min': param.data.min().item(),
                    'max': param.data.max().item(),
                    'norm': param.data.norm().item()
                }
        
        # Gradient statistics (if available)
        if hasattr(self, 'integration_gradients'):
            stats['gradients'] = self.integration_gradients.copy()
            
        return stats
    
    def apply_integration_constraints(self):
        """Apply constraints to integration parameters (e.g., non-negativity, normalization)."""
        for name, param in self.named_parameters():
            with torch.no_grad():
                if 'alpha' in name.lower() or 'beta' in name.lower():
                    # Ensure non-negativity for mixing weights
                    param.data = torch.clamp(param.data, min=0.0)
                elif 'gamma' in name.lower():
                    # Clamp interaction term to reasonable range
                    param.data = torch.clamp(param.data, min=-1.0, max=1.0)
    
    def normalize_pathway_weights(self, alpha: torch.Tensor, beta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Normalize pathway weights so they sum to 1.
        
        Args:
            alpha: Color pathway weights
            beta: Brightness pathway weights
            
        Returns:
            Normalized (alpha, beta) weights
        """
        total = alpha + beta + 1e-8  # Small epsilon for numerical stability
        return alpha / total, beta / total
    
    def compute_pathway_importance(self, alpha: torch.Tensor, beta: torch.Tensor) -> Dict[str, float]:
        """
        Compute relative importance of each pathway.
        
        Args:
            alpha: Color pathway weights
            beta: Brightness pathway weights
            
        Returns:
            Dictionary with pathway importance metrics
        """
        alpha_mean = alpha.abs().mean().item()
        beta_mean = beta.abs().mean().item()
        total = alpha_mean + beta_mean + 1e-8
        
        return {
            'color_importance': alpha_mean / total,
            'brightness_importance': beta_mean / total,
            'importance_ratio': alpha_mean / (beta_mean + 1e-8)
        }
