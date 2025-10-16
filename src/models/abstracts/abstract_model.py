import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Union, Optional, Any, Tuple, Callable
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import time
import matplotlib.pyplot as plt
from pathlib import Path

# Loss function imports
from src.training.losses import FocalLoss

class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all models in the Multi-Stream Neural Networks framework.
    Defines the interface that all model implementations must adhere to.
    """
    
    # Default compilation configuration - can be overridden by subclasses
    DEFAULT_COMPILE_CONFIG = {
        'learning_rate': 0.001,
        'weight_decay': 0.0,
        'scheduler': 'cosine'
    }
    
    def __init__(self, 
                 block: type,
                 layers: list[int],
                 num_classes: int,
                 zero_init_residual: bool = False,
                 groups: int = 1,
                 width_per_group: int = 64,
                 replace_stride_with_dilation: Optional[list[bool]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 device: Optional[str] = None,
                 use_amp: bool = False):
        """
        Initialize the base model.
        
        Args:
            block: Block type for ResNet architectures (e.g., BasicBlock, Bottleneck) - required
            layers: List of layer depths for ResNet architectures - required
            num_classes: Number of output classes - required
            zero_init_residual: Whether to zero-initialize the last BN in each residual branch
            groups: Number of blocked connections from input channels to output channels
            width_per_group: Width of each group (used in ResNet variants)
            replace_stride_with_dilation: Whether to replace stride with dilation for layers
            norm_layer: Normalization layer to use (defaults to appropriate layer in subclass)
            device: Device for model training/inference - 'cpu', 'cuda', 'mps', or None for auto-detection
            use_amp: Whether to use automatic mixed precision (AMP) for training
        """
        super().__init__()
        
        # Store runtime-relevant configuration parameters (keep num_classes for practical use)
        self.num_classes = num_classes
        self.groups = groups
        self.base_width = width_per_group  
        
        # Norm layer setup
        self._norm_layer = norm_layer  # store as _norm_layer for ResNet compatibility
        
        # Initialize internal parameters - always dual-channel tracking
        self.stream1_inplanes = 64
        self.stream2_inplanes = 64
        
        self.dilation = 1   # Standard ResNet starting dilation
        
        # Process replace_stride_with_dilation parameter like PyTorch does (but don't store)
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        
        # Device and mixed precision setup
        self._setup_device_and_precision(device, use_amp)
        
        # Build the network architecture - pass construction parameters
        self._build_network(block, layers, replace_stride_with_dilation)
        
        # Move to device before weight initialization to ensure initialization happens on target device
        self.to(self.device)
        
        # Initialize weights after moving to device - pass construction parameters
        self._initialize_weights(zero_init_residual)
        
        # Initialize training state
        self._init_training_components()
    
    def _setup_device_and_precision(self, device: Optional[str], use_amp: bool):
        """Setup device and mixed precision - consistent with ResNet approach."""
        # Set device with improved detection
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        # Set up automatic mixed precision (AMP) for CUDA
        if self.device.type == 'cuda' and use_amp:
            self.use_amp = True
            self.scaler = GradScaler()  # Use the imported GradScaler
            print(f"✅ Enabled Automatic Mixed Precision (AMP) training on {self.device}")
        else:
            self.use_amp = False
            self.scaler = None
            if use_amp and self.device.type != 'cuda':
                print(f"⚠️  AMP requested but not available on {self.device.type}, using standard precision")
    
    def _init_training_components(self):
        """Initialize training-related components for Keras-style API."""
        # Training components (set by compile())
        self.is_compiled = False
        self.optimizer = None
        self.criterion = None
        self.scheduler = None
        self.scheduler_type = None
    
    @abstractmethod
    def _build_network(self, block: type, layers: list[int], replace_stride_with_dilation: list[bool]):
        """
        Build the network architecture.
        This method should be implemented by all subclasses.
        
        Args:
            block: Block type for ResNet architectures (e.g., BasicBlock, Bottleneck)
            layers: List of layer depths for ResNet architectures
            replace_stride_with_dilation: Whether to replace stride with dilation for layers
        """
        pass
    
    @abstractmethod
    def _initialize_weights(self, zero_init_residual: bool):
        """
        Initialize the weights of the network.
        This method should be implemented by all subclasses.
        
        Args:
            zero_init_residual: Whether to zero-initialize the last BN in each residual branch
        """
        pass
    
    @abstractmethod
    def forward(self, stream1_input: torch.Tensor, stream2_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the multi-stream network.
        
        Args:
            stream1_input: Color/RGB input tensor
            stream2_input: Brightness/luminance input tensor
            
        Returns:
            Single tensor for training/classification (not tuple)
        """
        pass
    
    @abstractmethod
    def _forward_stream1_pathway(self, stream1_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the stream1 pathway.
        
        Args:
            stream1_input: The stream1 input tensor.
            
        Returns:
            The stream1 pathway output tensor.
        """
        pass
    
    @abstractmethod
    def _forward_stream2_pathway(self, stream2_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the stream2 pathway.
        
        Args:
            stream2_input: The stream2 input tensor.
            
        Returns:
            The stream2 pathway output tensor.
        """
        pass

    def compile(self, optimizer: str = 'adam', learning_rate: float = None,
                weight_decay: float = None, loss: str = 'cross_entropy',
                metrics: List[str] = None, scheduler: str = None,
                min_lr: float = 1e-6,
                # Stream-specific optimization parameters
                stream1_lr: float = None,
                stream2_lr: float = None,
                stream1_weight_decay: float = None,
                stream2_weight_decay: float = None,
                **kwargs):
        """
        Compile the model with the specified optimization parameters.

        Args:
            optimizer: Optimizer name ('adam', 'adamw', 'sgd', 'rmsprop')
            learning_rate: Learning rate (uses class default if None)
            weight_decay: Weight decay for regularization (uses class default if None)
            loss: Loss function name ('cross_entropy', 'focal')
            metrics: List of metrics to track
            scheduler: Learning rate scheduler (uses class default if None)
            min_lr: Minimum learning rate for schedulers
            stream1_lr: Learning rate for stream1 pathway (if None, uses learning_rate)
            stream2_lr: Learning rate for stream2 pathway (if None, uses learning_rate)
            stream1_weight_decay: Weight decay for stream1 pathway (if None, uses weight_decay)
            stream2_weight_decay: Weight decay for stream2 pathway (if None, uses weight_decay)
            device: Device to use for computation ('cpu', 'cuda', 'mps', etc.)
            use_amp: Whether to use automatic mixed precision training (only for CUDA)
            **kwargs: Additional arguments for optimizers and loss functions
        """
        # Use architecture-specific defaults if not specified
        config = self.DEFAULT_COMPILE_CONFIG.copy()
        
        # Set defaults for unspecified parameters
        if learning_rate is None:
            learning_rate = config['learning_rate']
        if weight_decay is None:
            weight_decay = config['weight_decay']
        if scheduler is None:
            scheduler = config['scheduler']
        
        # Set default metrics if not provided
        if metrics is None:
            metrics = ['accuracy']
        
        # Note: Device and mixed precision are already set in constructor
        
        # Store scheduler type for use in fit method
        self.scheduler_type = scheduler
        
        # Filter optimizer-specific kwargs using whitelist approach
        # Define allowed parameters for each optimizer
        common_optimizer_params = {'eps', 'amsgrad', 'maximize'}
        adam_params = common_optimizer_params | {'betas'}
        # Note: For SGD, 'momentum', 'dampening', and 'nesterov' are handled explicitly
        # to provide sensible defaults and enforce PyTorch constraints
        sgd_params = common_optimizer_params
        adamw_params = adam_params  # AdamW uses same params as Adam
        
        # Get allowed parameters based on optimizer
        if optimizer.lower() == 'adam':
            allowed_params = adam_params
        elif optimizer.lower() == 'sgd':
            allowed_params = sgd_params
        elif optimizer.lower() == 'adamw':
            allowed_params = adamw_params
        else:
            allowed_params = common_optimizer_params
        
        # Filter kwargs to only include allowed optimizer parameters
        optimizer_kwargs = {k: v for k, v in kwargs.items() if k in allowed_params}

        # Extract eps from optimizer_kwargs to avoid duplicate keyword argument error
        # (we set default eps=1e-8 explicitly, but user might pass custom eps)
        eps = optimizer_kwargs.pop('eps', 1e-8)

        # Check if stream-specific optimization is requested
        use_stream_specific = any([stream1_lr is not None, stream2_lr is not None,
                                   stream1_weight_decay is not None, stream2_weight_decay is not None])

        if use_stream_specific:
            # Set defaults for stream-specific parameters
            stream1_lr = stream1_lr if stream1_lr is not None else learning_rate
            stream2_lr = stream2_lr if stream2_lr is not None else learning_rate
            stream1_weight_decay = stream1_weight_decay if stream1_weight_decay is not None else weight_decay
            stream2_weight_decay = stream2_weight_decay if stream2_weight_decay is not None else weight_decay

            # Separate parameters by stream
            stream1_params = []
            stream2_params = []
            shared_params = []

            for name, param in self.named_parameters():
                if 'stream1' in name:
                    stream1_params.append(param)
                elif 'stream2' in name:
                    stream2_params.append(param)
                else:
                    shared_params.append(param)

            # Create parameter groups
            param_groups = []
            if stream1_params:
                param_groups.append({'params': stream1_params, 'lr': stream1_lr, 'weight_decay': stream1_weight_decay})
            if stream2_params:
                param_groups.append({'params': stream2_params, 'lr': stream2_lr, 'weight_decay': stream2_weight_decay})
            if shared_params:
                param_groups.append({'params': shared_params, 'lr': learning_rate, 'weight_decay': weight_decay})

            # Configure optimizer with parameter groups
            if optimizer.lower() == 'adam':
                self.optimizer = torch.optim.Adam(param_groups, eps=eps, **optimizer_kwargs)
            elif optimizer.lower() == 'adamw':
                self.optimizer = torch.optim.AdamW(param_groups, eps=eps, **optimizer_kwargs)
            elif optimizer.lower() == 'sgd':
                momentum = kwargs.get('momentum', 0.9)
                nesterov = kwargs.get('nesterov', True)
                dampening = kwargs.get('dampening', 0)
                if nesterov and dampening != 0:
                    raise ValueError("Nesterov momentum requires zero dampening. Either disable nesterov or set dampening=0")
                self.optimizer = torch.optim.SGD(
                    param_groups,
                    momentum=momentum,
                    dampening=dampening,
                    nesterov=nesterov,
                    **optimizer_kwargs
                )
            elif optimizer.lower() == 'rmsprop':
                self.optimizer = torch.optim.RMSprop(param_groups, momentum=0.9, **optimizer_kwargs)
            else:
                raise ValueError(f"Unsupported optimizer: {optimizer}")
        else:
            # Standard single-LR optimization
            if optimizer.lower() == 'adam':
                self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay, eps=eps, **optimizer_kwargs)
            elif optimizer.lower() == 'adamw':
                self.optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay, eps=eps, **optimizer_kwargs)
            elif optimizer.lower() == 'sgd':
                # Use SGD with Nesterov momentum (standard practice)
                # momentum and nesterov are handled explicitly (not in optimizer_kwargs) to provide defaults
                momentum = kwargs.get('momentum', 0.9)
                nesterov = kwargs.get('nesterov', True)
                # PyTorch constraint: Nesterov requires zero dampening
                # If user wants custom dampening, they must disable nesterov
                dampening = kwargs.get('dampening', 0)
                if nesterov and dampening != 0:
                    raise ValueError("Nesterov momentum requires zero dampening. Either disable nesterov or set dampening=0")
                self.optimizer = torch.optim.SGD(
                    self.parameters(),
                    lr=learning_rate,
                    momentum=momentum,
                    dampening=dampening,
                    weight_decay=weight_decay,
                    nesterov=nesterov,
                    **optimizer_kwargs
                )
            elif optimizer.lower() == 'rmsprop':
                self.optimizer = torch.optim.RMSprop(self.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9, **optimizer_kwargs)
            else:
                raise ValueError(f"Unsupported optimizer: {optimizer}")
        
        # Configure loss function - only standard losses supported
        if loss.lower() == 'cross_entropy':
            # Add label smoothing for better regularization (prevents overconfident predictions)
            label_smoothing = kwargs.get('label_smoothing', 0.0)
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        elif loss.lower() == 'focal':
            alpha = kwargs.get('alpha', 1.0)
            gamma = kwargs.get('gamma', 2.0)
            self.criterion = FocalLoss(alpha=alpha, gamma=gamma)
        else:
            raise ValueError(f"Unsupported loss function: {loss}. Supported losses: 'cross_entropy', 'focal'")
        
        # Store configuration in a centralized dictionary
        self.training_config = {
            'optimizer': optimizer.lower(),
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'loss': loss.lower(),
            'metrics': metrics or ['accuracy'],
            'scheduler': scheduler.lower() if scheduler else 'none',
            'min_lr': min_lr,
            'device': str(self.device),
            'use_amp': self.use_amp  # Use the attribute set in constructor
        }
        
        # Set compilation flag
        self.is_compiled = True
        
        # Scheduler will be configured in fit() with actual training parameters
        self.scheduler = None
        
        # Log compilation details
        model_name = self.__class__.__name__
        print(f"{model_name} compiled with {optimizer} optimizer, {loss} loss")
        print(f"  Learning rate: {learning_rate}, Weight decay: {weight_decay}, Scheduler: {scheduler}")
        print(f"  Device: {self.device}, AMP: {self.use_amp}, Groups: {self.groups}, Width per group: {self.base_width}")
        print("  Using architecture-specific defaults where applicable")
        
        return self
    
    @abstractmethod
    def fit(self, 
            train_loader: Optional[torch.utils.data.DataLoader] = None,
            val_loader: Optional[torch.utils.data.DataLoader] = None,
            train_stream1_data: Optional[torch.Tensor] = None,
            train_stream2_data: Optional[torch.Tensor] = None,
            train_targets: Optional[torch.Tensor] = None,
            val_stream1_data: Optional[torch.Tensor] = None,
            val_stream2_data: Optional[torch.Tensor] = None,
            val_targets: Optional[torch.Tensor] = None,
            epochs: int = 10,
            batch_size: int = 32,
            callbacks: Optional[list] = None,
            verbose: bool = True,
            save_path: Optional[str] = None,
            early_stopping: bool = False,
            patience: int = 10,
            min_delta: float = 0.001,
            monitor: str = 'val_loss',
            restore_best_weights: bool = True,
            **scheduler_kwargs):
        """
        Train the model.
        
        This method supports two input modes:
        1. Direct data arrays: stream1_data, stream2_data, labels
        2. DataLoader: train_loader (containing color, brightness, labels)
        
        Args:
            train_loader: DataLoader for training data (if not using direct arrays)
            val_loader: DataLoader for validation data (if not using direct arrays)
            train_stream1_data: Training color/RGB data array (if not using train_loader)
            train_stream2_data: Training brightness data array (if not using train_loader)
            train_targets: Training labels (if not using train_loader)
            val_stream1_data: Validation color/RGB data array (if not using val_loader)
            val_stream2_data: Validation brightness data array (if not using val_loader)
            val_targets: Validation labels (if not using val_loader)
            epochs: Number of epochs to train
            batch_size: Batch size (used only when passing direct data arrays)
            callbacks: List of callbacks to apply during training
            verbose: Whether to print progress during training
            save_path: Path to save best model checkpoint
            early_stopping: Whether to enable early stopping
            patience: Number of epochs to wait for improvement before stopping
            min_delta: Minimum change to qualify as an improvement
            monitor: Metric to monitor ('val_loss' or 'val_accuracy')
            restore_best_weights: Whether to restore best weights when early stopping
            **scheduler_kwargs: Additional arguments for the scheduler
            
        Returns:
            Training history or results
        """
        pass
    
    @abstractmethod
    def predict(self, stream1_data: Optional[Union[np.ndarray, torch.Tensor]] = None, 
                stream2_data: Optional[Union[np.ndarray, torch.Tensor]] = None, 
                data_loader: Optional[DataLoader] = None,
                batch_size: Optional[int] = None) -> np.ndarray:
        """
        Generate predictions for the input data.
        
        This method supports two input modes:
        1. Direct data arrays: stream1_data and stream2_data
        2. DataLoader: data_loader (containing stream1 and stream2 data)
        
        Args:
            stream1_data: Color/RGB input data (if not using data_loader)
            stream2_data: Brightness input data (if not using data_loader)
            data_loader: DataLoader containing both input modalities (alternative to direct arrays)
            batch_size: Batch size for prediction (used only when passing direct data arrays)
            
        Returns:
            Predicted class labels as numpy array
        """
        pass
    
    @abstractmethod
    def predict_proba(self, stream1_data: Optional[Union[np.ndarray, torch.Tensor]] = None, 
                     stream2_data: Optional[Union[np.ndarray, torch.Tensor]] = None,
                     data_loader: Optional[DataLoader] = None,
                     batch_size: Optional[int] = None) -> np.ndarray:
        """
        Generate probability predictions for the input data.
        
        This method supports two input modes:
        1. Direct data arrays: stream1_data and stream2_data
        2. DataLoader: data_loader (containing stream1 and stream2 data)
        
        Args:
            stream1_data: Color/RGB input data (if not using data_loader)
            stream2_data: Brightness input data (if not using data_loader)
            data_loader: DataLoader containing both input modalities (alternative to direct arrays)
            batch_size: Batch size for prediction (used only when passing direct data arrays)
            
        Returns:
            Predicted probabilities as numpy array
        """
        pass
    
    @abstractmethod
    def evaluate(self, data_loader: DataLoader, stream_monitoring: bool = True) -> Dict[str, float]:
        """
        Evaluate the model on the given data.

        Args:
            data_loader: DataLoader containing dual-channel input data and targets
            stream_monitoring: Whether to calculate stream-specific metrics (default: True)

        Returns:
            Dictionary containing evaluation metrics (e.g., accuracy, loss, stream accuracies)
        """
        pass
    
    # def _validate(self, val_loader: DataLoader) -> Tuple[float, float]:
    #     """
    #     Validate the model on a validation dataset loader.
        
    #     Args:
    #         val_loader: Validation data loader
            
    #     Returns:
    #         Tuple of (average_validation_loss, validation_accuracy)
    #     """
    #     self.eval()
    #     total_loss = 0.0
    #     correct = 0
    #     total = 0
        
    #     with torch.no_grad():
    #         for batch in val_loader:
    #             if len(batch) == 3:
    #                 stream1_data, stream2_data, targets = batch
    #             elif isinstance(batch, dict) and all(k in batch for k in ['color', 'brightness', 'labels']):
    #                 # Handle dictionary format
    #                 stream1_data = batch['color']
    #                 stream2_data = batch['brightness']
    #                 targets = batch['labels']
    #             else:
    #                 # Handle default format
    #                 stream1_data, stream2_data = batch[0], batch[1]
    #                 targets = batch[2] if len(batch) > 2 else None
                
    #             # Move to device
    #             stream1_data = stream1_data.to(self.device)
    #             stream2_data = stream2_data.to(self.device)
    #             if targets is not None:
    #                 targets = targets.to(self.device)
                
    #             # Forward pass
    #             outputs = self.forward(stream1_data, stream2_data)
                
    #             if targets is not None:
    #                 # Calculate loss
    #                 if not hasattr(self, 'criterion') or self.criterion is None:
    #                     criterion = nn.CrossEntropyLoss()
    #                 else:
    #                     criterion = self.criterion
                    
    #                 loss = criterion(outputs, targets)
    #                 total_loss += loss.item()
                    
    #                 # Calculate accuracy
    #                 _, predicted = torch.max(outputs, 1)
    #                 correct += (predicted == targets).sum().item()
    #                 total += targets.size(0)
        
    #     val_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0
    #     val_accuracy = correct / total if total > 0 else 0
        
    #     return val_loss, val_accuracy


