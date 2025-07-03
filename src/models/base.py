"""
Base classes for Multi-Stream Neural Networks
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from ..utils.device_utils import DeviceManager
from torch.amp import GradScaler
from torch import optim
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, List, Union


class BaseMultiStreamModel(nn.Module, ABC):
    """
    Base class for all Multi-Stream Neural Network models.
    
    Provides common functionality and interface for MSNN architectures.
    """
    
    # Default compilation configuration - can be overridden by subclasses
    DEFAULT_COMPILE_CONFIG = {
        'learning_rate': 0.001,
        'weight_decay': 0.0,
        'gradient_clip': 1.0,
        'scheduler': 'cosine',
        'early_stopping_patience': 5
    }
    
    def __init__(
        self,
        num_classes: int,
        activation: str = 'relu',
        dropout: float = 0.0,
        device: str = 'auto',
        **kwargs
    ):
        super().__init__()
        self.num_classes = num_classes
        self.activation = activation
        self.dropout = dropout
        
        # Setup device management with proper detection
        self.device_manager = DeviceManager(preferred_device=device if device != 'auto' else None)
        self.device = self.device_manager.device
        
        # Track pathway gradients for analysis
        self.pathway_gradients = {}
        
        # Training state
        self.is_compiled = False
        self.optimizer = None
        self.criterion = None
        self.metrics = []
        
        # Mixed precision support
        self.use_mixed_precision = self.device_manager.enable_mixed_precision()
        self.scaler = GradScaler() if self.use_mixed_precision else None
        
        # Note: Subclasses should call self._finalize_initialization() after their setup
        
    
    @abstractmethod
    def _build_network(self):
        """
        Build the network architecture. Must be implemented by subclasses.
        
        This method should construct all the layers and components specific to the model architecture.
        Called during initialization to set up the network structure.
        """
        pass
    
    @abstractmethod
    def _initialize_weights(self):
        """
        Initialize network weights. Must be implemented by subclasses.
        
        This method should initialize the weights of all network parameters according to
        the best practices for the specific architecture (e.g., Xavier, Kaiming, etc.).
        Called after network construction to set up initial parameter values.
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
    def extract_features(self, color_input: torch.Tensor, brightness_input: torch.Tensor) -> torch.Tensor:
        """
        Extract concatenated features before final classification.
        
        This method should return fused features ready for external classifiers.
        For separate pathway features, use get_separate_features() instead.
        
        Args:
            color_input: Color/RGB input tensor
            brightness_input: Brightness/luminance input tensor
            
        Returns:
            Concatenated fused features ready for external classifiers
        """
        pass
       
    @abstractmethod
    def get_separate_features(self, color_input: torch.Tensor, brightness_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract separate features from both pathways without classification.
        
        This method should return individual pathway features for research and analysis.
        For fused features ready for external classifiers, use extract_features() instead.
        
        Args:
            color_input: Color/RGB input tensor
            brightness_input: Brightness/luminance input tensor
            
        Returns:
            Tuple of (color_features, brightness_features) for pathway analysis
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
    def analyze_pathway_weights(self) -> Dict[str, float]:
        """
        Analyze the relative importance of pathways based on learned weights.
        
        Returns:
            Dictionary with detailed pathway weight statistics
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
        
    @property
    @abstractmethod
    def fusion_type(self) -> str:
        """
        Return the type of fusion used by this model.
        
        Returns:
            String describing the fusion strategy (e.g., 'shared_classifier', 'separate_classifiers', etc.)
        """
        pass
        
    @abstractmethod
    def get_classifier_info(self) -> Dict[str, Any]:
        """
        Get information about the classifier architecture.
        
        Returns:
            Dictionary containing classifier architecture details
        """
        pass
        
    

    # Common training methods that all models will share
    def compile(self, optimizer: str = 'adam', learning_rate: float = None, 
                weight_decay: float = None, loss: str = 'cross_entropy', metrics: List[str] = None,
                gradient_clip: float = None, scheduler: str = None, 
                early_stopping_patience: int = None, min_lr: float = 1e-6):
        """
        Compile the model with optimizer, loss function, and training configuration (Keras-like API).
        
        Uses architecture-specific defaults from DEFAULT_COMPILE_CONFIG if parameters are not specified.
        Each model type (Dense/CNN/etc.) can define optimal defaults while allowing full customization.
        
        Args:
            optimizer: Optimizer name ('adam', 'adamw', 'sgd', 'rmsprop')
            learning_rate: Learning rate (uses class default if None)
            weight_decay: Weight decay (uses class default if None)
            loss: Loss function name ('cross_entropy')
            metrics: List of metrics to track
            gradient_clip: Maximum norm for gradient clipping (uses class default if None)
            scheduler: Learning rate scheduler (uses class default if None)
            early_stopping_patience: Patience for early stopping (uses class default if None)
            min_lr: Minimum learning rate for cosine annealing
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
            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay, eps=1e-8)
        elif optimizer.lower() == 'adamw':
            self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay, eps=1e-8)
        elif optimizer.lower() == 'sgd':
            # Use SGD with Nesterov momentum (standard practice)
            self.optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9, nesterov=True, weight_decay=weight_decay)
        elif optimizer.lower() == 'rmsprop':
            self.optimizer = optim.RMSprop(self.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
        
        # Configure loss function
        if loss.lower() == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss function: {loss}")
        
        # Store training configuration
        self.training_config = {
            'optimizer': optimizer.lower(),
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'loss': loss.lower(),
            'metrics': metrics,
            'gradient_clip': gradient_clip,
            'scheduler': scheduler.lower(),
            'early_stopping_patience': early_stopping_patience,
            'min_lr': min_lr
        }
        
        # Store individual attributes for backward compatibility
        self.metrics = metrics or ['accuracy']
        self.gradient_clip = gradient_clip
        self.early_stopping_patience = early_stopping_patience
        self.scheduler_type = scheduler.lower()
        self.min_lr = min_lr
        self.is_compiled = True
        
        # Configure learning rate scheduler (basic setup - will be finalized in fit())
        self.scheduler = None
        if self.scheduler_type == 'cosine':
            # Will be properly configured in fit() with actual epochs
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=100, eta_min=min_lr
            )
        elif self.scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        elif self.scheduler_type == 'onecycle':
            # Will be properly configured in fit() with actual steps_per_epoch and epochs
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=learning_rate,
                steps_per_epoch=500,  # Placeholder - will be updated in fit()
                epochs=100,  # Placeholder - will be updated in fit()
                pct_start=0.3
            )
        elif self.scheduler_type != 'none':
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
    def fit(self, *args, **kwargs):
        """
        Fit the model to data. Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement fit() method.")

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

    @abstractmethod
    def predict(self, color_data: Union[np.ndarray, torch.Tensor, DataLoader], brightness_data: Union[np.ndarray, torch.Tensor] = None, 
                batch_size: int = None) -> np.ndarray:
        """
        Make predictions on new data.
        
        Each model implementation should handle appropriate input reshaping for its architecture.
        This method should support both direct data arrays and DataLoader inputs.
        
        Args:
            color_data: Color input data or a DataLoader containing both color and brightness data
            brightness_data: Brightness input data (not needed if color_data is a DataLoader)
            batch_size: Batch size for prediction (auto-detected if None)
            
        Returns:
            Predicted class labels
        """
        pass

    @abstractmethod
    def predict_proba(self, color_data: Union[np.ndarray, torch.Tensor, DataLoader], brightness_data: Union[np.ndarray, torch.Tensor] = None,
                      batch_size: int = None) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Each model implementation should handle appropriate input reshaping for its architecture.
        This method should support both direct data arrays and DataLoader inputs.
        
        Args:
            color_data: Color input data or a DataLoader containing both color and brightness data
            brightness_data: Brightness input data (not needed if color_data is a DataLoader)
            batch_size: Batch size for prediction (auto-detected if None)
            
        Returns:
            Prediction probabilities
        """
        pass

    @abstractmethod
    def evaluate(self, test_color_data=None, test_brightness_data=None, test_labels=None, 
                 test_loader=None, batch_size: int = 32) -> Dict[str, float]:
        """
        Evaluate the model on test data or test data loader.
        
        Each model implementation should handle appropriate input reshaping for its architecture.
        This method should support both direct data arrays and DataLoader inputs.
        
        Args:
            test_color_data: Test color data (optional if test_loader is provided)
            test_brightness_data: Test brightness data (optional if test_loader is provided)
            test_labels: Test labels (optional if test_loader is provided)
            test_loader: Test data loader (optional if direct data is provided)
            batch_size: Batch size for evaluation (used when creating a loader from direct data)
            
        Returns:
            Dictionary with evaluation metrics
        """
        pass
    
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
            'num_classes': self.num_classes,
            'model_type': self.__class__.__name__
        }
    
    @abstractmethod
    def _validate(self, val_loader):
        """
        Validate the model on a validation dataset loader.
        
        Each model implementation should handle appropriate input reshaping for its architecture.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (average_validation_loss, validation_accuracy)
        """
        pass

    def _finalize_initialization(self):
        """
        Finalize model initialization by moving to device and optimizing.
        Should be called by subclasses after their setup is complete.
        """
        # Move model to device and optimize
        self.to(self.device)
        self.device_manager.optimize_for_device(self)
