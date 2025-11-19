import re
import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Optional, Union
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
        
        # Initialize internal parameters - dynamic N-stream tracking
        # num_streams will be set by subclasses based on their stream configuration
        # self.num_streams and self.stream_inplanes will be initialized in subclass
        
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

    def get_stream_parameter_groups(self,
                                     stream_lrs: Union[float, list[float]],
                                     stream_weight_decays: Union[float, list[float]] = 0.0,
                                     shared_lr: Optional[float] = None,
                                     shared_weight_decay: float = 0.0):
        """
        Create parameter groups for stream-specific learning rates in N-stream models.

        This method separates model parameters into N+1 groups:
        - N stream-specific groups (one per stream): Contains ONLY stream_weights.i and stream_biases.i
          (the stream's own feature extraction parameters)
        - 1 shared group: Contains integration_from_streams.*, integrated_weight, integrated_bias,
          fc layers, and other shared parameters (everything that builds/uses the integrated stream)

        Args:
            stream_lrs: Learning rate(s) for stream-specific parameters.
                       - float: Same LR for all streams
                       - list[float]: Per-stream LRs (length must match number of streams)
            stream_weight_decays: Weight decay for stream-specific parameters.
                                 - float: Same weight decay for all streams
                                 - list[float]: Per-stream weight decays (length must match number of streams)
            shared_lr: Learning rate for shared/integrated parameters.
                      If None, defaults to mean of stream_lrs.
            shared_weight_decay: Weight decay for shared parameters. Default: 0.0.

        Returns:
            List of parameter group dicts that can be passed to PyTorch optimizers.
            Format: [{'params': [...], 'lr': ..., 'weight_decay': ...}, ...]

        Raises:
            ValueError: If no streams detected or if list lengths don't match number of streams.

        Example:
            >>> # For a 3-stream model (RGB, Depth, HHA)
            >>> param_groups = model.get_stream_parameter_groups(
            ...     stream_lrs=[2e-4, 7e-4, 5e-4],           # Different LR per stream
            ...     stream_weight_decays=[1e-4, 2e-4, 1.5e-4],  # Different WD per stream
            ...     shared_lr=5e-4,                          # Shared params LR
            ...     shared_weight_decay=1e-4                 # Shared params WD
            ... )
            >>> optimizer = torch.optim.AdamW(param_groups)
        """
        # Detect number of streams
        num_streams = 0
        for name, _ in self.named_parameters():
            if '.stream_weights.' in name:
                match = re.search(r'\.stream_weights\.(\d+)(?:\.|$)', name)
                if match:
                    num_streams = max(num_streams, int(match.group(1)) + 1)

        if num_streams == 0:
            raise ValueError("No streams detected.")

        # Convert to lists
        if isinstance(stream_lrs, (int, float)):
            stream_lrs = [stream_lrs] * num_streams
        if isinstance(stream_weight_decays, (int, float)):
            stream_weight_decays = [stream_weight_decays] * num_streams

        if len(stream_lrs) != num_streams:
            raise ValueError(f"stream_lrs length must match num_streams ({num_streams})")
        if len(stream_weight_decays) != num_streams:
            raise ValueError(f"stream_weight_decays length must match num_streams ({num_streams})")

        if shared_lr is None:
            shared_lr = sum(stream_lrs) / len(stream_lrs)

        # Separate parameters
        stream_params = [[] for _ in range(num_streams)]
        shared_params = []

        for name, param in self.named_parameters():
            # Match stream-specific parameters: ONLY stream_weights.i and stream_biases.i
            # integration_from_streams.i goes to shared group (it builds the integrated stream)
            if '.stream_weights.' in name or '.stream_biases.' in name:
                match = re.search(r'\.stream_(?:weights|biases)\.(\d+)(?:\.|$)', name)
                if match:
                    stream_params[int(match.group(1))].append(param)
                    continue
            # Everything else is shared:
            # - integration_from_streams.* (builds integrated stream from all streams)
            # - integrated_weight, integrated_bias (integrated stream's own processing)
            # - fc (final classifier)
            shared_params.append(param)

        # Build groups
        param_groups = []
        for i in range(num_streams):
            if stream_params[i]:
                param_groups.append({
                    'params': stream_params[i],
                    'lr': stream_lrs[i],
                    'weight_decay': stream_weight_decays[i]
                })

        if shared_params:
            param_groups.append({
                'params': shared_params,
                'lr': shared_lr,
                'weight_decay': shared_weight_decay
            })

        return param_groups

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
    def forward(self, stream_inputs: list[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the N-stream network.

        Args:
            stream_inputs: List of input tensors, one per stream
                          Each tensor has shape [batch_size, channels, height, width]
                          Number of tensors in list should match the model's num_streams

        Returns:
            Single tensor for training/classification (not tuple)
            Shape: [batch_size, num_classes]

        Example:
            >>> # For a 3-stream model (e.g., RGB, Depth, HHA)
            >>> stream_inputs = [
            ...     torch.randn(32, 3, 224, 224),  # stream 0: RGB
            ...     torch.randn(32, 1, 224, 224),  # stream 1: Depth
            ...     torch.randn(32, 3, 224, 224)   # stream 2: HHA
            ... ]
            >>> output = model(stream_inputs)
            >>> # output shape: [32, num_classes]
        """
        pass
    
    @abstractmethod
    def _forward_stream_pathway(self, stream_idx: int, stream_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through a single stream pathway.

        Args:
            stream_idx: Index of the stream (0, 1, 2, ...)
            stream_input: The input tensor for this stream

        Returns:
            The stream pathway output tensor
        """
        pass

    def compile(self,
                optimizer: torch.optim.Optimizer,
                loss: str = 'cross_entropy',
                scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                metrics: Optional[list[str]] = None,
                **kwargs):
        """
        Compile the model with optimizer, loss, and scheduler (Keras-style API).

        Following the Keras philosophy:
        - compile() = Configuration (what to optimize)
        - fit() = Execution (how to train)

        Args:
            optimizer: PyTorch optimizer instance (e.g., torch.optim.AdamW(...))
                      Users must create the optimizer with desired parameters before calling compile()
            loss: Loss function name ('cross_entropy', 'focal')
            scheduler: Optional PyTorch LR scheduler instance (e.g., from setup_scheduler())
                      If None, no learning rate scheduling is applied
            metrics: List of metrics to track during training (default: ['accuracy'])
            **kwargs: Additional arguments for loss functions (e.g., label_smoothing, alpha, gamma)

        Example:
            >>> # Option 1: Simple optimizer (single learning rate for all parameters)
            >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            >>>
            >>> # Option 2: Stream-specific learning rates using helper method (3-stream example)
            >>> param_groups = model.get_stream_parameter_groups(
            ...     stream_lrs=[2e-4, 7e-4, 5e-4],
            ...     stream_weight_decays=[1e-4, 2e-4, 1.5e-4],
            ...     shared_lr=5e-4
            ... )
            >>> optimizer = torch.optim.AdamW(param_groups)
            >>>
            >>> # Create scheduler using public setup_scheduler() function
            >>> from src.training.schedulers import setup_scheduler
            >>> scheduler = setup_scheduler(
            ...     optimizer, 'decaying_cosine', epochs=80, train_loader_len=40,
            ...     t_max=10, eta_min=6e-5, max_factor=0.6, min_factor=0.6
            ... )
            >>>
            >>> # Compile model with optimizer and scheduler objects
            >>> model.compile(optimizer=optimizer, scheduler=scheduler, loss='cross_entropy')
            >>>
            >>> # Train model
            >>> model.fit(train_loader, val_loader, epochs=80)

        Returns:
            self (for method chaining)
        """
        # Validate optimizer
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(
                f"optimizer must be a torch.optim.Optimizer instance, got {type(optimizer)}. "
                "Create your optimizer before calling compile(). Example:\n"
                "  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)"
            )

        # Store optimizer
        self.optimizer = optimizer

        # Store scheduler (can be None)
        if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler._LRScheduler):
            # Also check for custom schedulers that don't inherit from _LRScheduler
            if not hasattr(scheduler, 'step'):
                raise TypeError(
                    f"scheduler must be a PyTorch LRScheduler instance or have a step() method, "
                    f"got {type(scheduler)}. Create your scheduler before calling compile(). Example:\n"
                    "  from src.training.schedulers import setup_scheduler\n"
                    "  scheduler = setup_scheduler(optimizer, 'cosine', epochs=80, train_loader_len=40)"
                )
        self.scheduler = scheduler

        # Set default metrics if not provided
        if metrics is None:
            metrics = ['accuracy']

        # Configure loss function
        if loss.lower() == 'cross_entropy':
            label_smoothing = kwargs.get('label_smoothing', 0.0)
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        elif loss.lower() == 'focal':
            alpha = kwargs.get('alpha', 1.0)
            gamma = kwargs.get('gamma', 2.0)
            self.criterion = FocalLoss(alpha=alpha, gamma=gamma)
        else:
            raise ValueError(f"Unsupported loss function: {loss}. Supported: 'cross_entropy', 'focal'")

        # Extract optimizer info for logging
        optimizer_name = optimizer.__class__.__name__
        base_lr = optimizer.param_groups[0]['lr'] if optimizer.param_groups else 'N/A'
        scheduler_name = scheduler.__class__.__name__ if scheduler else 'None'

        # Store configuration
        self.training_config = {
            'optimizer': optimizer_name,
            'base_lr': base_lr,
            'loss': loss.lower(),
            'metrics': metrics,
            'scheduler': scheduler_name,
            'device': str(self.device),
            'use_amp': self.use_amp,
            'num_param_groups': len(optimizer.param_groups)
        }

        # Set compilation flag
        self.is_compiled = True

        # Log compilation details
        model_name = self.__class__.__name__
        print(f"{model_name} compiled with {optimizer_name} optimizer, {loss} loss")

        # Log parameter groups info
        if len(optimizer.param_groups) > 1:
            print(f"  Using {len(optimizer.param_groups)} parameter groups:")
            for i, pg in enumerate(optimizer.param_groups):
                lr = pg.get('lr', 'N/A')
                wd = pg.get('weight_decay', 'N/A')
                print(f"    Group {i+1}: lr={lr:.2e}, weight_decay={wd:.2e}")
        else:
            print(f"  Learning rate: {base_lr:.2e}")

        print(f"  Scheduler: {scheduler_name}")
        print(f"  Device: {self.device}, AMP: {self.use_amp}")

        return self
    
    @abstractmethod
    def fit(self,
            train_loader: torch.utils.data.DataLoader,
            val_loader: Optional[torch.utils.data.DataLoader] = None,
            epochs: int = 10,
            callbacks: Optional[list] = None,
            verbose: bool = True,
            save_path: Optional[str] = None,
            early_stopping: bool = False,
            patience: int = 10,
            min_delta: float = 0.001,
            monitor: str = 'val_loss',
            restore_best_weights: bool = True,
            gradient_accumulation_steps: int = 1,
            grad_clip_norm: Optional[float] = None,
            clear_cache_per_epoch: bool = False,
            stream_monitoring: bool = False,
            stream_early_stopping: bool = False,
            stream_patience: Union[int, list[int]] = 10,
            stream_min_delta: float = 0.001) -> dict:
        """
        Train the model.

        DataLoader format: Each batch from the DataLoader should be a dictionary with:
            - 'streams': List of tensors (one per stream)
            - 'labels': Target labels tensor

        Example DataLoader setup:
            >>> # For a 3-stream model (RGB, Depth, HHA)
            >>> class MultiStreamDataset(Dataset):
            ...     def __getitem__(self, idx):
            ...         return {
            ...             'streams': [
            ...                 rgb_data[idx],    # stream 0
            ...                 depth_data[idx],  # stream 1
            ...                 hha_data[idx]     # stream 2
            ...             ],
            ...             'labels': labels[idx]
            ...         }
            >>>
            >>> train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
            >>> model.fit(train_loader, val_loader, epochs=80)

        Args:
            train_loader: DataLoader for training data (required)
            val_loader: DataLoader for validation data (optional)
            epochs: Number of epochs to train
            callbacks: List of callbacks to apply during training
            verbose: Whether to print progress during training
            save_path: Path to save best model checkpoint
            early_stopping: Whether to enable early stopping
            patience: Number of epochs to wait for improvement before stopping
            min_delta: Minimum change to qualify as an improvement
            monitor: Metric to monitor ('val_loss' or 'val_accuracy')
            restore_best_weights: Whether to restore best weights when early stopping
            gradient_accumulation_steps: Number of steps to accumulate gradients (default: 1)
            grad_clip_norm: Maximum gradient norm for clipping (None = disabled)
            clear_cache_per_epoch: Clear CUDA cache after each epoch (for OOM issues)
            stream_monitoring: Enable stream-specific monitoring (LR, WD, accuracy per stream)
            stream_early_stopping: Enable stream-specific early stopping (freezes streams when they plateau)
            stream_patience: Patience per stream (int for all, or list[int] per stream)
            stream_min_delta: Minimum improvement for stream early stopping

        Returns:
            Training history or results
        """
        pass
    
    @abstractmethod
    def predict(self, data_loader: DataLoader) -> np.ndarray:
        """
        Generate predictions for the input data.

        DataLoader format: Each batch should be a dictionary with:
            - 'streams': List of tensors (one per stream)
            - 'labels': Target labels tensor (optional for prediction)

        Args:
            data_loader: DataLoader containing N-stream input data

        Returns:
            Predicted class labels as numpy array

        Example:
            >>> # For a 3-stream model
            >>> predictions = model.predict(test_loader)
            >>> # predictions shape: [num_samples]
        """
        pass
    
    @abstractmethod
    def predict_proba(self, data_loader: DataLoader) -> np.ndarray:
        """
        Generate probability predictions for the input data.

        DataLoader format: Each batch should be a dictionary with:
            - 'streams': List of tensors (one per stream)
            - 'labels': Target labels tensor (optional for prediction)

        Args:
            data_loader: DataLoader containing N-stream input data

        Returns:
            Predicted probabilities as numpy array [num_samples, num_classes]

        Example:
            >>> # For a 3-stream model
            >>> probabilities = model.predict_proba(test_loader)
            >>> # probabilities shape: [num_samples, num_classes]
        """
        pass
    
    @abstractmethod
    def evaluate(self, data_loader: DataLoader, stream_monitoring: bool = True) -> dict[str, float]:
        """
        Evaluate the model on the given data.

        DataLoader format: Each batch should be a dictionary with:
            - 'streams': List of tensors (one per stream)
            - 'labels': Target labels tensor

        Args:
            data_loader: DataLoader containing N-stream input data and targets
            stream_monitoring: Whether to calculate stream-specific metrics (default: True)

        Returns:
            Dictionary containing evaluation metrics (e.g., accuracy, loss, stream accuracies)

        Example:
            >>> # For a 3-stream model
            >>> metrics = model.evaluate(test_loader, stream_monitoring=True)
            >>> # metrics = {'loss': 0.25, 'accuracy': 0.92,
            >>> #            'stream0_accuracy': 0.85, 'stream1_accuracy': 0.88, 'stream2_accuracy': 0.87}
        """
        pass


