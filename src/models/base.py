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
from typing import Tuple, Dict, Any, List, Union, Optional
import time
import matplotlib.pyplot as plt
from pathlib import Path


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
        # Initialize diagnostic tracking
        self._init_diagnostics()
    
    def _init_diagnostics(self):
        """Initialize diagnostic tracking variables."""
        self.diagnostic_history = {
            'gradient_norms': [],
            'weight_norms': [],
            'dead_neuron_counts': [],
            'pathway_balance': [],
            'epoch_times': [],
            'learning_rates': [],
            'nan_inf_detections': []
        }
        self.diagnostic_hooks = []
    
    def _calculate_gradient_norm(self) -> float:
        """Calculate total gradient norm for diagnostic purposes."""
        total_norm = 0.0
        param_count = 0
        
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        return total_norm ** 0.5 if param_count > 0 else 0.0
    
    def _calculate_weight_norm(self) -> float:
        """Calculate total weight norm for diagnostic purposes."""
        total_norm = 0.0
        param_count = 0
        
        for p in self.parameters():
            if p.requires_grad:
                param_norm = p.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        return total_norm ** 0.5 if param_count > 0 else 0.0
    
    def _count_dead_neurons(self, sample_batch: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> int:
        """
        Count dead neurons using activation tracking.
        
        Args:
            sample_batch: Optional sample batch for activation analysis
            
        Returns:
            Number of dead neurons detected
        """
        if sample_batch is None:
            # Simple check without activation tracking
            return 0
        
        # Use the debug_utils function for comprehensive dead neuron detection
        try:
            from ..utils.debug_utils import check_for_dead_neurons
            dead_stats = check_for_dead_neurons(self, sample_batch, num_batches=1, threshold=0.05)
            
            # Sum up dead neurons across all layers
            total_dead = 0
            for layer_name, dead_percent in dead_stats.items():
                if isinstance(dead_percent, dict):
                    # Multi-channel layers
                    total_dead += dead_percent.get('color_dead_percent', 0)
                    total_dead += dead_percent.get('brightness_dead_percent', 0)
                else:
                    # Standard layers
                    total_dead += dead_percent
            
            return int(total_dead)
        except Exception:
            return 0
    
    def _analyze_pathway_balance(self) -> float:
        """
        Analyze balance between pathways.
        
        Returns:
            Balance ratio (1.0 = perfectly balanced)
        """
        try:
            # Use the model's pathway analysis if available
            if hasattr(self, 'analyze_pathway_weights'):
                weights = self.analyze_pathway_weights()
                return weights.get('balance_ratio', 1.0)
            
            # Fallback: analyze gradient balance
            from ..utils.debug_utils import check_pathway_gradients
            pathway_stats = check_pathway_gradients(self)
            
            if 'pathway_balance' in pathway_stats:
                balance_info = pathway_stats['pathway_balance']
                color_percent = balance_info.get('color_pathway_grad_percent', 50)
                brightness_percent = balance_info.get('brightness_pathway_grad_percent', 50)
                
                # Calculate balance ratio (closer to 1.0 = more balanced)
                if color_percent > 0 and brightness_percent > 0:
                    return min(color_percent, brightness_percent) / max(color_percent, brightness_percent)
            
            return 1.0
        except Exception:
            return 1.0
    
    def _setup_diagnostic_hooks(self):
        """Set up hooks for NaN/Inf detection during training."""
        try:
            from ..utils.debug_utils import add_diagnostic_hooks
            self.diagnostic_hooks = add_diagnostic_hooks(self)
        except Exception:
            self.diagnostic_hooks = []
    
    def _cleanup_diagnostic_hooks(self):
        """Clean up diagnostic hooks."""
        for hook in self.diagnostic_hooks:
            try:
                hook.remove()
            except Exception:
                pass
        self.diagnostic_hooks = []
    
    def _collect_epoch_diagnostics(self, epoch: int, sample_batch: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, 
                                  epoch_time: float = 0.0, current_lr: float = 0.0) -> Dict[str, float]:
        """
        Collect diagnostic information for the current epoch.
        
        Args:
            epoch: Current epoch number
            sample_batch: Optional sample batch for activation analysis
            epoch_time: Time taken for this epoch
            current_lr: Current learning rate
            
        Returns:
            Dictionary with diagnostic metrics
        """
        diagnostics = {}
        
        # Collect gradient norm
        gradient_norm = self._calculate_gradient_norm()
        diagnostics['gradient_norm'] = gradient_norm
        self.diagnostic_history['gradient_norms'].append(gradient_norm)
        
        # Collect weight norm
        weight_norm = self._calculate_weight_norm()
        diagnostics['weight_norm'] = weight_norm
        self.diagnostic_history['weight_norms'].append(weight_norm)
        
        # Collect dead neuron count
        dead_count = self._count_dead_neurons(sample_batch)
        diagnostics['dead_neuron_count'] = dead_count
        self.diagnostic_history['dead_neuron_counts'].append(dead_count)
        
        # Collect pathway balance
        pathway_balance = self._analyze_pathway_balance()
        diagnostics['pathway_balance'] = pathway_balance
        self.diagnostic_history['pathway_balance'].append(pathway_balance)
        
        # Store timing and learning rate
        diagnostics['epoch_time'] = epoch_time
        diagnostics['learning_rate'] = current_lr
        self.diagnostic_history['epoch_times'].append(epoch_time)
        self.diagnostic_history['learning_rates'].append(current_lr)
        
        return diagnostics
    
    def _save_diagnostic_plots(self, output_dir: str, model_name: str = None):
        """
        Save diagnostic plots to the specified directory.
        
        Args:
            output_dir: Directory to save plots
            model_name: Name of the model (for plot titles)
        """
        if model_name is None:
            model_name = self.__class__.__name__
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create diagnostic plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Gradient norms
        if self.diagnostic_history['gradient_norms']:
            axes[0, 0].plot(self.diagnostic_history['gradient_norms'])
            axes[0, 0].set_title('Gradient Norms')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Gradient Norm')
            axes[0, 0].grid(True)
        
        # Weight norms
        if self.diagnostic_history['weight_norms']:
            axes[0, 1].plot(self.diagnostic_history['weight_norms'])
            axes[0, 1].set_title('Weight Norms')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Weight Norm')
            axes[0, 1].grid(True)
        
        # Dead neuron counts
        if self.diagnostic_history['dead_neuron_counts']:
            axes[0, 2].plot(self.diagnostic_history['dead_neuron_counts'])
            axes[0, 2].set_title('Dead Neuron Counts')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Dead Neurons')
            axes[0, 2].grid(True)
        
        # Pathway balance
        if self.diagnostic_history['pathway_balance']:
            axes[1, 0].plot(self.diagnostic_history['pathway_balance'])
            axes[1, 0].set_title('Pathway Balance')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Balance Ratio')
            axes[1, 0].grid(True)
        
        # Epoch times
        if self.diagnostic_history['epoch_times']:
            axes[1, 1].plot(self.diagnostic_history['epoch_times'])
            axes[1, 1].set_title('Epoch Times')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Time (seconds)')
            axes[1, 1].grid(True)
        
        # Learning rates
        if self.diagnostic_history['learning_rates']:
            axes[1, 2].plot(self.diagnostic_history['learning_rates'])
            axes[1, 2].set_title('Learning Rates')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Learning Rate')
            axes[1, 2].grid(True)
        
        plt.suptitle(f'{model_name} - Diagnostic History')
        plt.tight_layout()
        
        # Save plot
        plot_path = output_path / f"{model_name}_diagnostics.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Diagnostic plots saved to {plot_path}")
    
    def get_diagnostic_summary(self) -> Dict[str, Any]:
        """
        Get a summary of diagnostic information.
        
        Returns:
            Dictionary with diagnostic summary
        """
        summary = {
            'model_name': self.__class__.__name__,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'diagnostic_history': self.diagnostic_history.copy(),
            'final_stats': {}
        }
        
        # Add final statistics
        if self.diagnostic_history['gradient_norms']:
            summary['final_stats']['final_gradient_norm'] = self.diagnostic_history['gradient_norms'][-1]
            summary['final_stats']['avg_gradient_norm'] = np.mean(self.diagnostic_history['gradient_norms'])
        
        if self.diagnostic_history['weight_norms']:
            summary['final_stats']['final_weight_norm'] = self.diagnostic_history['weight_norms'][-1]
            summary['final_stats']['avg_weight_norm'] = np.mean(self.diagnostic_history['weight_norms'])
        
        if self.diagnostic_history['pathway_balance']:
            summary['final_stats']['final_pathway_balance'] = self.diagnostic_history['pathway_balance'][-1]
            summary['final_stats']['avg_pathway_balance'] = np.mean(self.diagnostic_history['pathway_balance'])
        
        if self.diagnostic_history['epoch_times']:
            summary['final_stats']['total_training_time'] = sum(self.diagnostic_history['epoch_times'])
            summary['final_stats']['avg_epoch_time'] = np.mean(self.diagnostic_history['epoch_times'])
        
        return summary
    
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
            # Default max_lr set higher than initial learning rate for proper cycling
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=learning_rate * 10,  # Higher max_lr for proper cycling
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
    def fit(self, *args, enable_diagnostics: bool = False, diagnostic_output_dir: str = "diagnostics", **kwargs):
        """
        Fit the model to data. Must be implemented by subclasses.
        
        Args:
            *args: Variable positional arguments specific to each model
            enable_diagnostics: Enable comprehensive diagnostic tracking and reporting
            diagnostic_output_dir: Directory to save diagnostic outputs when diagnostics are enabled
            **kwargs: Variable keyword arguments specific to each model
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
    
    # === DIAGNOSTIC METHODS ===
    
    def enable_diagnostics(self):
        """Enable comprehensive diagnostic tracking during training."""
        self.diagnostic_mode = True
        self.diagnostic_history = {
            'gradient_norms': [],
            'weight_norms': [],
            'pathway_balance': [],
            'dead_neurons': [],
            'epoch_times': []
        }
        self._add_diagnostic_hooks()
    
    def disable_diagnostics(self):
        """Disable diagnostic tracking and remove hooks."""
        self.diagnostic_mode = False
        self._remove_diagnostic_hooks()
    
    def _add_diagnostic_hooks(self):
        """Add hooks for monitoring gradients and activations."""
        from ..utils.debug_utils import add_diagnostic_hooks
        self.diagnostic_hooks = add_diagnostic_hooks(self)
    
    def _remove_diagnostic_hooks(self):
        """Remove diagnostic hooks."""
        for hook in self.diagnostic_hooks:
            hook.remove()
        self.diagnostic_hooks = []
    
    def calculate_gradient_norm(self) -> float:
        """Calculate total gradient norm across all parameters."""
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
    
    def calculate_weight_norm(self) -> float:
        """Calculate total weight norm across all parameters."""
        total_norm = 0.0
        for p in self.parameters():
            param_norm = p.data.norm(2)
            total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
    
    def count_dead_neurons(self) -> int:
        """Count neurons that are always inactive (simplified implementation)."""
        # This is a simplified check - proper implementation would track activations
        dead_count = 0
        for name, module in self.named_modules():
            if isinstance(module, nn.ReLU):
                # Placeholder - proper implementation would track activations over time
                pass
        return dead_count
    
    def analyze_pathway_balance(self) -> float:
        """Analyze balance between pathways (model-specific implementation)."""
        try:
            if hasattr(self, 'analyze_pathway_weights'):
                weights = self.analyze_pathway_weights()
                return weights.get('balance_ratio', 1.0)
        except Exception:
            pass
        return 1.0
    
    def record_diagnostic_metrics(self, epoch_time: float = 0.0):
        """Record diagnostic metrics for current training step."""
        if not self.diagnostic_mode:
            return
        
        self.diagnostic_history['gradient_norms'].append(self.calculate_gradient_norm())
        self.diagnostic_history['weight_norms'].append(self.calculate_weight_norm())
        self.diagnostic_history['pathway_balance'].append(self.analyze_pathway_balance())
        self.diagnostic_history['dead_neurons'].append(self.count_dead_neurons())
        self.diagnostic_history['epoch_times'].append(epoch_time)
    
    def get_diagnostic_history(self) -> Dict[str, List[float]]:
        """Get complete diagnostic history."""
        return self.diagnostic_history.copy()
    
    def save_diagnostic_plots(self, output_path: str, model_name: str = None):
        """Save diagnostic plots to file."""
        if not self.diagnostic_mode:
            print("⚠️ Diagnostics not enabled. Call enable_diagnostics() first.")
            return
        
        model_name = model_name or self.__class__.__name__
        
        try:
            from ..utils.debug_utils import plot_diagnostic_history
            plot_diagnostic_history(self.diagnostic_history, output_path, model_name)
            print(f"✅ Diagnostic plots saved to {output_path}")
        except ImportError:
            print("⚠️ Diagnostic plotting not available. Install matplotlib.")
        except Exception as e:
            print(f"❌ Error saving diagnostic plots: {e}")
    
    def generate_diagnostic_report(self) -> Dict[str, Any]:
        """Generate comprehensive diagnostic report."""
        if not self.diagnostic_mode:
            return {"error": "Diagnostics not enabled"}
        
        history = self.diagnostic_history
        if not history['gradient_norms']:
            return {"error": "No diagnostic data recorded"}
        
        report = {
            "model_name": self.__class__.__name__,
            "total_epochs": len(history['gradient_norms']),
            "final_gradient_norm": history['gradient_norms'][-1] if history['gradient_norms'] else 0,
            "final_weight_norm": history['weight_norms'][-1] if history['weight_norms'] else 0,
            "final_pathway_balance": history['pathway_balance'][-1] if history['pathway_balance'] else 1.0,
            "average_epoch_time": np.mean(history['epoch_times']) if history['epoch_times'] else 0,
            "gradient_norm_trend": self._analyze_trend(history['gradient_norms']),
            "weight_norm_trend": self._analyze_trend(history['weight_norms']),
            "diagnostic_summary": self._generate_diagnostic_summary()
        }
        
        return report
    
    def _analyze_trend(self, values: List[float]) -> str:
        """Analyze trend in a list of values."""
        if len(values) < 2:
            return "insufficient_data"
        
        # Simple trend analysis
        first_half = np.mean(values[:len(values)//2])
        second_half = np.mean(values[len(values)//2:])
        
        if second_half > first_half * 1.1:
            return "increasing"
        elif second_half < first_half * 0.9:
            return "decreasing"
        else:
            return "stable"
    
    def _generate_diagnostic_summary(self) -> str:
        """Generate human-readable diagnostic summary."""
        history = self.diagnostic_history
        
        if not history['gradient_norms']:
            return "No training data available"
        
        grad_trend = self._analyze_trend(history['gradient_norms'])
        weight_trend = self._analyze_trend(history['weight_norms'])
        
        summary = f"Training completed over {len(history['gradient_norms'])} epochs. "
        summary += f"Gradient norms are {grad_trend}, weight norms are {weight_trend}. "
        
        final_balance = history['pathway_balance'][-1] if history['pathway_balance'] else 1.0
        if abs(final_balance - 1.0) > 0.2:
            summary += "Pathway imbalance detected. "
        else:
            summary += "Pathways appear balanced. "
        
        return summary

    def _collect_validation_diagnostics(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Collect comprehensive validation diagnostics including loss analysis, 
        gradient flow, and pathway balance during validation.
        
        Args:
            val_loader: Validation data loader
            epoch: Current epoch number
            
        Returns:
            Dictionary with validation diagnostic metrics
        """
        self.eval()
        val_diagnostics = {
            'epoch': epoch,
            'val_loss': 0.0,
            'val_accuracy': 0.0,
            'val_samples': 0,
            'prediction_confidence': [],
            'class_predictions': [],
            'pathway_activations': {'color': [], 'brightness': []},
            'layer_activations': {},
            'gradient_flow': {},
            'dead_neurons_detected': 0,
            'pathway_balance': 1.0
        }
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Setup hooks for activation tracking during validation
        activation_hooks = []
        
        def make_activation_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    # Store activation statistics
                    val_diagnostics['layer_activations'][name] = {
                        'mean': output.mean().item(),
                        'std': output.std().item(),
                        'max': output.max().item(),
                        'min': output.min().item(),
                        'zeros_percent': (output == 0).float().mean().item() * 100
                    }
                elif isinstance(output, tuple) and len(output) == 2:
                    # Multi-channel output
                    color_out, brightness_out = output
                    val_diagnostics['layer_activations'][name] = {
                        'color': {
                            'mean': color_out.mean().item(),
                            'std': color_out.std().item(),
                            'zeros_percent': (color_out == 0).float().mean().item() * 100
                        },
                        'brightness': {
                            'mean': brightness_out.mean().item(),
                            'std': brightness_out.std().item(),
                            'zeros_percent': (brightness_out == 0).float().mean().item() * 100
                        }
                    }
            return hook
        
        # Register hooks for key layers
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.ReLU, nn.BatchNorm2d)):
                hook = module.register_forward_hook(make_activation_hook(name))
                activation_hooks.append(hook)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if len(batch) == 3:
                    color_data, brightness_data, targets = batch
                else:
                    # Handle different batch formats
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
                    
                    # Store predictions and confidence
                    probs = torch.softmax(outputs, dim=1)
                    max_probs, _ = torch.max(probs, dim=1)
                    val_diagnostics['prediction_confidence'].extend(max_probs.cpu().numpy().tolist())
                    val_diagnostics['class_predictions'].extend(predicted.cpu().numpy().tolist())
                
                # Analyze pathway outputs for balance
                try:
                    if hasattr(self, 'analyze_pathways'):
                        color_out, brightness_out = self.analyze_pathways(color_data, brightness_data)
                        
                        # Store pathway activation statistics
                        val_diagnostics['pathway_activations']['color'].append({
                            'mean': color_out.mean().item(),
                            'std': color_out.std().item(),
                            'max': color_out.max().item()
                        })
                        val_diagnostics['pathway_activations']['brightness'].append({
                            'mean': brightness_out.mean().item(),
                            'std': brightness_out.std().item(),
                            'max': brightness_out.max().item()
                        })
                except Exception as e:
                    # Skip if pathway analysis fails
                    pass
                
                # Limit validation analysis to prevent excessive memory usage
                if batch_idx >= 10:  # Analyze first 10 batches
                    break
        
        # Calculate final metrics
        if total > 0:
            val_diagnostics['val_loss'] = total_loss / len(val_loader)
            val_diagnostics['val_accuracy'] = correct / total
            val_diagnostics['val_samples'] = total
        
        # Analyze prediction confidence
        if val_diagnostics['prediction_confidence']:
            confidences = val_diagnostics['prediction_confidence']
            val_diagnostics['avg_confidence'] = np.mean(confidences)
            val_diagnostics['low_confidence_percent'] = (np.array(confidences) < 0.5).mean() * 100
        
        # Analyze pathway balance
        try:
            val_diagnostics['pathway_balance'] = self._analyze_pathway_balance()
        except Exception:
            val_diagnostics['pathway_balance'] = 1.0
        
        # Check for dead neurons using a sample batch
        try:
            if len(val_loader) > 0:
                sample_batch = next(iter(val_loader))
                if len(sample_batch) >= 2:
                    sample_color = sample_batch[0][:4].to(self.device)  # Small sample
                    sample_brightness = sample_batch[1][:4].to(self.device)
                    val_diagnostics['dead_neurons_detected'] = self._count_dead_neurons(
                        (sample_color, sample_brightness)
                    )
        except Exception:
            val_diagnostics['dead_neurons_detected'] = 0
        
        # Clean up hooks
        for hook in activation_hooks:
            try:
                hook.remove()
            except Exception:
                pass
        
        return val_diagnostics
