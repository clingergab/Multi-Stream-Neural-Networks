import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Union, Optional, Any, Tuple
from torch.utils.data import DataLoader
from torch.amp import GradScaler
import time
import matplotlib.pyplot as plt
from pathlib import Path

class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all models in the Multi-Stream Neural Networks framework.
    Defines the interface that all model implementations must adhere to.
    """
    
    # Default compilation configuration - can be overridden by subclasses
    DEFAULT_COMPILE_CONFIG = {
        'learning_rate': 0.001,
        'weight_decay': 0.0,
        'gradient_clip': 1.0,
        'scheduler': 'cosine',
        'early_stopping_patience': 5
    }
    
    def __init__(self, 
                 num_classes: int,  # Made required (no default)
                 activation: str = 'relu',
                 dropout: float = 0.0,
                 device: str = 'auto',
                 **kwargs):
        """
        Initialize the base model.
        
        Args:
            num_classes: Number of output classes (required)
            activation: Activation function to use ('relu', 'selu', etc.)
            dropout: Dropout rate
            device: Device for model training/inference - 'auto', 'cpu', 'cuda', or 'mps'
            **kwargs: Additional keyword arguments for specific model implementations
        """
        super().__init__()
        self.num_classes = num_classes
        self.activation = activation
        self.dropout = dropout
        
        # Training state
        self.is_compiled = False
        self.optimizer = None
        self.criterion = None
        self.scheduler = None
        
        # Device setup
        self.device_manager = self._get_device_manager(device)
        self.device = self.device_manager.device if hasattr(self.device_manager, 'device') else self._get_device(device)
        
        # Mixed precision support (for CUDA)
        self.use_mixed_precision = self._enable_mixed_precision()
        self.scaler = GradScaler() if self.use_mixed_precision else None
        
        # Build and initialize the network
        self._build_network()
        self._initialize_weights()
        
        # Move model to device
        self.to(self.device)
    
    def _get_device_manager(self, preferred_device):
        """Get device manager or create a simple one if DeviceManager is not available"""
        try:
            from ...utils.device_utils import DeviceManager
            return DeviceManager(preferred_device=preferred_device if preferred_device != 'auto' else None)
        except ImportError:
            return None
    
    def _get_device(self, device_str):
        """Fallback method to get device if DeviceManager is not available"""
        if device_str == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        return torch.device(device_str)
    
    def _enable_mixed_precision(self):
        """Enable mixed precision if supported"""
        if hasattr(self, 'device_manager') and self.device_manager and hasattr(self.device_manager, 'enable_mixed_precision'):
            return self.device_manager.enable_mixed_precision()
        else:
            # Simple fallback - enable for CUDA only
            return torch.cuda.is_available()
    
    @abstractmethod
    def _build_network(self):
        """
        Build the network architecture.
        This method should be implemented by all subclasses.
        """
        pass
    
    @abstractmethod
    def _initialize_weights(self):
        """
        Initialize the weights of the network.
        This method should be implemented by all subclasses.
        """
        pass
    
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
    def _forward_color_pathway(self, color_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the color pathway.
        
        Args:
            color_input: The color input tensor.
            
        Returns:
            The color pathway output tensor.
        """
        pass
    
    @abstractmethod
    def _forward_brightness_pathway(self, brightness_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the brightness pathway.
        
        Args:
            brightness_input: The brightness input tensor.
            
        Returns:
            The brightness pathway output tensor.
        """
        pass
    
    @property
    @abstractmethod
    def fusion_type(self) -> str:
        """
        The type of fusion used in the model.
        
        Returns:
            A string representing the fusion type.
        """
        pass
    
    @abstractmethod
    def compile(self, optimizer: str = 'adam', learning_rate: float = None, 
                weight_decay: float = None, loss: str = 'cross_entropy', 
                metrics: List[str] = None, gradient_clip: float = None, 
                scheduler: str = None, early_stopping_patience: int = None, 
                min_lr: float = 1e-6):
        """
        Compile the model with the specified optimization parameters.
        
        Args:
            optimizer: Optimizer name ('adam', 'adamw', 'sgd', 'rmsprop')
            learning_rate: Learning rate (uses class default if None)
            weight_decay: Weight decay for regularization (uses class default if None)
            loss: Loss function name ('cross_entropy')
            metrics: List of metrics to track
            gradient_clip: Maximum norm for gradient clipping (uses class default if None)
            scheduler: Learning rate scheduler (uses class default if None)
            early_stopping_patience: Patience for early stopping (uses class default if None)
            min_lr: Minimum learning rate for schedulers
        """
        # Use architecture-specific defaults if not specified
        config = self.DEFAULT_COMPILE_CONFIG.copy()
        
        if learning_rate is None:
            learning_rate = config['learning_rate']
        if weight_decay is None:
            weight_decay = config['weight_decay']
        if gradient_clip is None:
            gradient_clip = config['gradient_clip']
        if scheduler is None:
            scheduler = config['scheduler']
        if early_stopping_patience is None:
            early_stopping_patience = config['early_stopping_patience']
        
        # Set default metrics if not provided
        if metrics is None:
            metrics = ['accuracy']
        
        # Configure optimizers with appropriate parameters
        if optimizer.lower() == 'adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay, eps=1e-8)
        elif optimizer.lower() == 'adamw':
            self.optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay, eps=1e-8)
        elif optimizer.lower() == 'sgd':
            # Use SGD with Nesterov momentum (standard practice)
            self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9, nesterov=True, weight_decay=weight_decay)
        elif optimizer.lower() == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(self.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
        
        # Configure loss function
        if loss.lower() == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss function: {loss}")
        
        # Store configuration in a centralized dictionary
        self.training_config = {
            'optimizer': optimizer.lower(),
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'loss': loss.lower(),
            'metrics': metrics or ['accuracy'],
            'gradient_clip': gradient_clip,
            'scheduler': scheduler.lower() if scheduler else 'none',
            'early_stopping_patience': early_stopping_patience,
            'min_lr': min_lr
        }
        
        # Set compilation flag
        self.is_compiled = True
        
        # Configure learning rate scheduler
        self.scheduler = None
        scheduler_type = self.training_config['scheduler']
        
        if scheduler_type == 'cosine':
            # Will be properly configured in fit() with actual epochs
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=100, eta_min=self.training_config['min_lr']
            )
        elif scheduler_type == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        elif scheduler_type == 'onecycle':
            # Will be properly configured in fit() with actual steps_per_epoch and epochs
            # Default max_lr set higher than initial learning rate for proper cycling
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=learning_rate * 10,  # Higher max_lr for proper cycling
                steps_per_epoch=500,  # Placeholder - will be updated in fit()
                epochs=100,  # Placeholder - will be updated in fit()
                pct_start=0.3
            )
        elif scheduler_type != 'none':
            raise ValueError(f"Unsupported scheduler: {scheduler}")
        
        # Log compilation details
        model_name = self.__class__.__name__
        print(f"{model_name} compiled with {optimizer} optimizer, {loss} loss")
        print(f"  Learning rate: {learning_rate}, Weight decay: {weight_decay}")
        print(f"  Gradient clip: {gradient_clip}, Scheduler: {scheduler}")
        print(f"  Early stopping patience: {early_stopping_patience}")
        print("  Using architecture-specific defaults where applicable")
        
        return self
    
    @abstractmethod
    def fit(self, color_data: Union[np.ndarray, torch.Tensor] = None,
            brightness_data: Union[np.ndarray, torch.Tensor] = None,
            labels: Union[np.ndarray, torch.Tensor] = None,
            val_color_data: Union[np.ndarray, torch.Tensor] = None,
            val_brightness_data: Union[np.ndarray, torch.Tensor] = None,
            val_labels: Union[np.ndarray, torch.Tensor] = None,
            train_loader: DataLoader = None,
            val_loader: DataLoader = None,
            epochs: int = 10,
            batch_size: int = 32,
            enable_diagnostics: bool = False,
            diagnostic_output_dir: str = "diagnostics",
            **kwargs):
        """
        Train the model.
        
        This method supports two input modes:
        1. Direct data arrays: color_data, brightness_data, labels
        2. DataLoader: train_loader (containing color, brightness, labels)
        
        Args:
            color_data: Training color/RGB data array (if not using train_loader)
            brightness_data: Training brightness data array (if not using train_loader)
            labels: Training labels (if not using train_loader)
            val_color_data: Validation color/RGB data array (if not using val_loader)
            val_brightness_data: Validation brightness data array (if not using val_loader)
            val_labels: Validation labels (if not using val_loader)
            train_loader: DataLoader for training data (alternative to direct arrays)
            val_loader: DataLoader for validation data (alternative to direct arrays)
            epochs: Number of epochs to train
            batch_size: Batch size (used only when passing direct data arrays)
            enable_diagnostics: Whether to enable diagnostic tracking
            diagnostic_output_dir: Directory to save diagnostic outputs
            **kwargs: Additional keyword arguments
            
        Returns:
            Training history or results
        """
        pass
    
    @abstractmethod
    def predict(self, color_data: Union[np.ndarray, torch.Tensor] = None, 
                brightness_data: Union[np.ndarray, torch.Tensor] = None, 
                data_loader: DataLoader = None,
                batch_size: int = None) -> np.ndarray:
        """
        Generate predictions for the input data.
        
        This method supports two input modes:
        1. Direct data arrays: color_data and brightness_data
        2. DataLoader: data_loader (containing color and brightness data)
        
        Args:
            color_data: Color/RGB input data (if not using data_loader)
            brightness_data: Brightness input data (if not using data_loader)
            data_loader: DataLoader containing both input modalities (alternative to direct arrays)
            batch_size: Batch size for prediction (used only when passing direct data arrays)
            
        Returns:
            Predicted class labels as numpy array
        """
        pass
    
    @abstractmethod
    def predict_proba(self, color_data: Union[np.ndarray, torch.Tensor] = None, 
                     brightness_data: Union[np.ndarray, torch.Tensor] = None,
                     data_loader: DataLoader = None,
                     batch_size: int = None) -> np.ndarray:
        """
        Generate probability predictions for the input data.
        
        This method supports two input modes:
        1. Direct data arrays: color_data and brightness_data
        2. DataLoader: data_loader (containing color and brightness data)
        
        Args:
            color_data: Color/RGB input data (if not using data_loader)
            brightness_data: Brightness input data (if not using data_loader)
            data_loader: DataLoader containing both input modalities (alternative to direct arrays)
            batch_size: Batch size for prediction (used only when passing direct data arrays)
            
        Returns:
            Predicted probabilities as numpy array
        """
        pass
    
    @abstractmethod
    def evaluate(self, color_data: Union[np.ndarray, torch.Tensor] = None, 
                brightness_data: Union[np.ndarray, torch.Tensor] = None, 
                labels: Union[np.ndarray, torch.Tensor] = None,
                data_loader: DataLoader = None,
                batch_size: int = 32) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        This method supports two input modes:
        1. Direct data arrays: color_data, brightness_data, and labels
        2. DataLoader: data_loader (containing color, brightness, and labels)
        
        Args:
            color_data: Color/RGB test data (if not using data_loader)
            brightness_data: Brightness test data (if not using data_loader)
            labels: Test labels (if not using data_loader)
            data_loader: DataLoader containing test data (alternative to direct arrays)
            batch_size: Batch size for evaluation (used only when passing direct data arrays)
            
        Returns:
            Dictionary with evaluation metrics (e.g., accuracy, loss)
        """
        pass
    
    def _validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate the model on a validation dataset loader.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (average_validation_loss, validation_accuracy)
        """
        self.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    color_data, brightness_data, targets = batch
                elif isinstance(batch, dict) and all(k in batch for k in ['color', 'brightness', 'labels']):
                    # Handle dictionary format
                    color_data = batch['color']
                    brightness_data = batch['brightness']
                    targets = batch['labels']
                else:
                    # Handle default format
                    color_data, brightness_data = batch[0], batch[1]
                    targets = batch[2] if len(batch) > 2 else None
                
                # Move to device
                color_data = color_data.to(self.device)
                brightness_data = brightness_data.to(self.device)
                if targets is not None:
                    targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.forward(color_data, brightness_data)
                
                if targets is not None:
                    # Calculate loss
                    if not hasattr(self, 'criterion') or self.criterion is None:
                        criterion = nn.CrossEntropyLoss()
                    else:
                        criterion = self.criterion
                    
                    loss = criterion(outputs, targets)
                    total_loss += loss.item()
                    
                    # Calculate accuracy
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == targets).sum().item()
                    total += targets.size(0)
        
        val_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_accuracy = correct / total if total > 0 else 0
        
        return val_loss, val_accuracy
    
    def _create_dataloader(self, color_data: Union[np.ndarray, torch.Tensor], 
                        brightness_data: Union[np.ndarray, torch.Tensor], 
                        labels: Union[np.ndarray, torch.Tensor] = None,
                        batch_size: int = 32,
                        shuffle: bool = False) -> DataLoader:
        """
        Create a DataLoader from raw numpy arrays or tensors.
        
        Args:
            color_data: Color/RGB data
            brightness_data: Brightness data
            labels: Labels (optional)
            batch_size: Batch size
            shuffle: Whether to shuffle the data
            
        Returns:
            DataLoader object
        """
        # Convert numpy arrays to tensors if needed
        if isinstance(color_data, np.ndarray):
            color_data = torch.from_numpy(color_data).float()
        if isinstance(brightness_data, np.ndarray):
            brightness_data = torch.from_numpy(brightness_data).float()
        
        if labels is not None:
            if isinstance(labels, np.ndarray):
                labels = torch.from_numpy(labels).long()
            dataset = torch.utils.data.TensorDataset(color_data, brightness_data, labels)
        else:
            dataset = torch.utils.data.TensorDataset(color_data, brightness_data)
        
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

