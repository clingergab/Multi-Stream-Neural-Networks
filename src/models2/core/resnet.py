from typing import Any, Callable, Optional, Union
import time

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import OneCycleLR
from training.losses import FocalLoss
from models2.common import (
    setup_scheduler,
    update_scheduler,
    save_checkpoint,
    create_dataloader_from_tensors,
    setup_early_stopping,
    early_stopping_initiated,
    create_progress_bar,
    finalize_progress_bar,
    update_history
)
try:
    # Check if we're in a proper Jupyter environment with widgets support
    from IPython import get_ipython
    if get_ipython() is not None and get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
        # We're in Jupyter, try to use notebook tqdm with ipywidgets
        try:
            from tqdm.notebook import tqdm
            # Test if widgets work by creating a dummy progress bar
            test_bar = tqdm(total=1, disable=True)
            test_bar.close()
        except ImportError:
            # ipywidgets not available, fall back to regular tqdm
            from tqdm import tqdm
    else:
        # Not in Jupyter, use regular tqdm
        from tqdm import tqdm
except:
    # Any other issue, fall back to regular tqdm
    from tqdm import tqdm

# For type hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from tqdm import tqdm as TqdmType

from .conv import conv1x1, conv3x3
from .blocks import BasicBlock, Bottleneck


__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
]


class ResNet(nn.Module):
    def __init__(
                self,
                block: type[Union[BasicBlock, Bottleneck]],
                layers: list[int],
                num_classes: int,
                zero_init_residual: bool = False,
                groups: int = 1,
                width_per_group: int = 64,
                replace_stride_with_dilation: Optional[list[bool]] = None,
                norm_layer: Optional[Callable[..., nn.Module]] = None,
                device: Optional[str] = None,
                use_amp: bool = False,
            ) -> None:
        super().__init__()
        
        # Store runtime-relevant configuration parameters (keep num_classes for practical use)
        self.num_classes = num_classes
        self.groups = groups
        self.base_width = width_per_group
        
        # Norm layer setup
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        # Initialize internal parameters
        self.inplanes = 64
        self.dilation = 1
        
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
        """Setup device and mixed precision."""
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
            self.scaler = torch.cuda.amp.GradScaler()
            print(f"✅ Enabled Automatic Mixed Precision (AMP) training on {self.device}")
        else:
            self.use_amp = False
            self.scaler = None
            if use_amp and self.device.type != 'cuda':
                print(f"⚠️  AMP requested but not available on {self.device.type}, using standard precision")
    
    def _init_training_components(self):
        """Initialize training-related components for Keras-style API."""
        # Training components (set by compile())
        self.optimizer = None
        self.criterion = None
        self.scheduler = None
        self.scheduler_type = None
    
    def _build_network(self, block: type[Union[BasicBlock, Bottleneck]], layers: list[int], replace_stride_with_dilation: list[bool]):
        """Build the ResNet network architecture - exactly like PyTorch."""
        # Network architecture - exactly like PyTorch
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, self.num_classes)
    
    def _initialize_weights(self, zero_init_residual: bool):
        """Initialize network weights - exactly like PyTorch."""
        # Weight initialization - exactly like PyTorch
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]
        
    def _make_layer(
        self,
        block: type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        """
        Create a layer composed of multiple residual blocks.
        Fully compliant with PyTorch ResNet implementation.
        """
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        # First block with potential downsampling - pass parameters exactly like PyTorch
        layers.append(
            block(
                self.inplanes, 
                planes, 
                stride, 
                downsample, 
                self.groups,
                self.base_width, 
                previous_dilation, 
                norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    # def _forward_impl(self, x: Tensor) -> Tensor:
    #     """
    #     Implementation of the forward pass.
        
    #     Args:
    #         x: Input tensor
            
    #     Returns:
    #         Output tensor
    #     """
    #     # Initial convolution and pooling
    #     x = self.conv1(x)
    #     x = self.bn1(x)
    #     x = self.relu(x)
    #     x = self.maxpool(x)

    #     # Residual blocks
    #     x = self.layer1(x)
    #     x = self.layer2(x)
    #     x = self.layer3(x)
    #     x = self.layer4(x)

    #     # Final pooling and classification
    #     x = self.avgpool(x)
    #     x = torch.flatten(x, 1)
    #     x = self.fc(x)

    #     return x

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Initial convolution and pooling
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Final pooling and classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def compile(
                self,
                optimizer: str = 'adam',
                loss: str = 'cross_entropy',
                learning_rate: float = 0.01,
                weight_decay: float = 1e-5,
                scheduler: Optional[str] = None,
                **kwargs
            ) -> None:
        """
        Compile the model with the given optimizer and loss function.
        
        Args:
            optimizer: Name of the optimizer to use ('adam', 'sgd', or 'adamw')
            loss: Name of the loss function ('cross_entropy' or 'focal') - ResNet only supports single-stream losses
            learning_rate: Learning rate
            weight_decay: Weight decay (L2 regularization)
            scheduler: Type of learning rate scheduler ('cosine', 'step', 'plateau', 'onecycle', or None)
            **kwargs: Additional arguments to pass to the optimizer or loss function
        """
        # Store scheduler type for use in fit method
        self.scheduler_type = scheduler
        self.scheduler_type = scheduler
        
         # Filter optimizer-specific kwargs using whitelist approach
        # Define allowed parameters for each optimizer
        common_optimizer_params = {'eps', 'amsgrad', 'maximize'}
        adam_params = common_optimizer_params | {'betas'}
        sgd_params = common_optimizer_params | {'momentum', 'dampening', 'nesterov'}
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
        
        # Set optimizer
        if optimizer == 'adam':
            self.optimizer = Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay, **optimizer_kwargs)
        elif optimizer == 'sgd':
            momentum = kwargs.get('momentum', 0.9)
            self.optimizer = SGD(self.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        elif optimizer == 'adamw':
            self.optimizer = AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay, **optimizer_kwargs)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
        
        # Set loss function - ResNet only supports single-stream losses
        if loss == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
        elif loss == 'focal':
            alpha = kwargs.get('alpha', 1.0)
            gamma = kwargs.get('gamma', 2.0)
            self.criterion = FocalLoss(alpha=alpha, gamma=gamma)
        else:
            raise ValueError(f"Unsupported loss function for ResNet: {loss}. ResNet only supports 'cross_entropy' and 'focal' losses. Use MCResNet for multi-stream losses.")
    
    def fit(
        self,
        train_loader: Optional[Union[torch.utils.data.DataLoader, torch.Tensor]] = None,
        val_loader: Optional[Union[torch.utils.data.DataLoader, torch.Tensor]] = None,
        train_targets: Optional[torch.Tensor] = None,
        val_targets: Optional[torch.Tensor] = None,
        epochs: int = 10,
        batch_size: int = 32,
        callbacks: Optional[list] = None,
        verbose: bool = True,
        save_path: Optional[str] = None,
        early_stopping: bool = False,
        patience: int = 10,
        min_delta: float = 0.001,
        monitor: str = 'val_loss',  # 'val_loss' or 'val_accuracy'
        restore_best_weights: bool = True,
        **scheduler_kwargs
    ) -> dict:
        """
        Train the model with optional early stopping.
        
        Args:
            train_loader: DataLoader for training data OR tensor of training inputs
            val_loader: DataLoader for validation data OR tensor of validation inputs (optional)
            train_targets: Training targets (required if train_loader is a tensor)
            val_targets: Validation targets (required if val_loader is a tensor)
            epochs: Number of epochs to train
            batch_size: Batch size (used when creating DataLoaders from tensors)
            callbacks: List of callbacks to apply during training
            verbose: Whether to print progress
            save_path: Path to save best model checkpoint
            early_stopping: Whether to enable early stopping
            patience: Number of epochs to wait for improvement before stopping
            min_delta: Minimum change to qualify as an improvement
            monitor: Metric to monitor ('val_loss' or 'val_accuracy')
            restore_best_weights: Whether to restore best weights when early stopping
            **scheduler_kwargs: Additional arguments for the scheduler:
                - For 'step' scheduler: step_size, gamma
                - For 'cosine' scheduler: t_max
                - For 'plateau' scheduler: scheduler_patience (or patience), factor
                  Note: Use 'scheduler_patience' to avoid conflict with early stopping patience
                - For 'onecycle' scheduler: steps_per_epoch (required), max_lr, pct_start, 
                  anneal_strategy, div_factor, final_div_factor
            
        Returns:
            Training history dictionary
        """
        if self.optimizer is None or self.criterion is None:
            raise ValueError("Model not compiled. Call compile() before fit().")
        
        # Handle tensor inputs by converting to DataLoaders
        if isinstance(train_loader, torch.Tensor):
            if train_targets is None:
                raise ValueError("train_targets must be provided when train_loader is a tensor")
            train_loader = create_dataloader_from_tensors(
                train_loader, train_targets, batch_size=batch_size, shuffle=True, device=self.device
            )
        
        if isinstance(val_loader, torch.Tensor):
            if val_targets is None:
                raise ValueError("val_targets must be provided when val_loader is a tensor")
            val_loader = create_dataloader_from_tensors(
                val_loader, val_targets, batch_size=batch_size*2, shuffle=False, device=self.device
            )
        
        callbacks = callbacks or []
        
        # Early stopping setup
        early_stopping_state = setup_early_stopping(early_stopping, val_loader, monitor, patience, min_delta, verbose)
        
        # Legacy best model saving (preserve existing functionality)
        best_val_acc = 0.0
        
        # Initialize training history
        history = {
            'train_loss': [], 
            'val_loss': [], 
            'train_accuracy': [],
            'val_accuracy': [],
            'learning_rates': []
        }
        
        # Set up scheduler
        self.scheduler = setup_scheduler(self.optimizer, self.scheduler_type, epochs, len(train_loader), **scheduler_kwargs)
        
        for epoch in range(epochs):
            # Calculate total steps for this epoch (training + validation)
            total_steps = len(train_loader)
            if val_loader:
                total_steps += len(val_loader)
            
            # Create progress bar for the entire epoch
            pbar = create_progress_bar(verbose, epoch, epochs, total_steps)
            
            # Training phase - use helper method
            avg_train_loss, train_accuracy = self._train_epoch(train_loader, history, pbar)
            
            # Validation phase
            val_loss = 0.0
            val_acc = 0.0
            
            if val_loader:
                val_loss, val_acc = self._validate(val_loader, pbar=pbar)
                
                # Legacy save best model (preserve existing functionality)
                if save_path and val_acc > best_val_acc:
                    best_val_acc = val_acc
                    save_checkpoint(
                        model_state_dict=self.state_dict(),
                        optimizer_state_dict=self.optimizer.state_dict() if self.optimizer else None,
                        scheduler_state_dict=self.scheduler.state_dict() if self.scheduler else None,
                        path=save_path,
                        history=history
                    )
                
                # Check for early stopping
                if early_stopping_initiated(
                    self.state_dict(), early_stopping_state, val_loss, val_acc, epoch, pbar, verbose, restore_best_weights
                ):
                    # Restore best weights if requested
                    if restore_best_weights and early_stopping_state['best_weights'] is not None:
                        self.load_state_dict({
                            k: v.to(self.device) for k, v in early_stopping_state['best_weights'].items()
                        })
                        if verbose:
                            print("🔄 Restored best model weights")
                    break
            
            # Update learning rate scheduler and get current LR
            update_scheduler(self.scheduler, val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history and finalize progress bar
            update_history(history, avg_train_loss, train_accuracy, val_loss, val_acc, current_lr, bool(val_loader))
            finalize_progress_bar(
                pbar, avg_train_loss, train_accuracy, val_loader, 
                val_loss, val_acc, early_stopping_state, current_lr
            )
            
            # Call callbacks
            for callback in callbacks:
                callback.on_epoch_end(epoch, {
                    'train_loss': avg_train_loss,
                    'train_accuracy': train_accuracy,
                    'val_loss': val_loss,
                    'val_accuracy': val_acc
                })
        
        # Final early stopping summary
        if early_stopping_state['enabled'] and val_loader is not None:
            history['early_stopping'] = {
                'stopped_early': early_stopping_state['patience_counter'] >= early_stopping_state['patience'],
                'best_epoch': early_stopping_state['best_epoch'] + 1,
                'best_metric': early_stopping_state['best_metric'],
                'monitor': early_stopping_state['monitor'],
                'patience': early_stopping_state['patience'],
                'min_delta': early_stopping_state['min_delta']
            }
            
            if verbose and early_stopping_state['patience_counter'] < early_stopping_state['patience']:
                print(f"🏁 Training completed without early stopping. Best {early_stopping_state['monitor']}: {early_stopping_state['best_metric']:.4f} at epoch {early_stopping_state['best_epoch'] + 1}")
        
        return history
    
    def predict(self, data_loader: Optional[Union[torch.utils.data.DataLoader, torch.Tensor]] = None, 
                targets: Optional[torch.Tensor] = None, batch_size: int = 32) -> torch.Tensor:
        """
        Generate predictions for the input data.
        
        Args:
            data_loader: DataLoader containing input data OR tensor of input data
            targets: Target tensor (required when data_loader is a tensor)
            batch_size: Batch size (used when creating DataLoader from tensor)
            
        Returns:
            Tensor of predicted classes
        """
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.to(self.device)
        
        # Handle tensor input by converting to DataLoader
        if isinstance(data_loader, torch.Tensor):
            if targets is None:
                raise ValueError("targets must be provided when data_loader is a tensor")
            data_loader = create_dataloader_from_tensors(
                data_loader, targets, batch_size=batch_size, shuffle=False, device=self.device
            )
        
        self.eval()
        all_predictions = []
        
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, dict):
                    inputs = batch['input'].to(self.device, non_blocking=True)
                else:
                    inputs = batch[0].to(self.device, non_blocking=True)
                
                outputs = self(inputs)
                _, predictions = torch.max(outputs, 1)
                all_predictions.append(predictions.cpu())
        
        return torch.cat(all_predictions, dim=0)
    
    def evaluate(self, data_loader: Union[torch.utils.data.DataLoader, torch.Tensor], 
                 targets: Optional[torch.Tensor] = None, batch_size: int = 32) -> dict:
        """
        Evaluate the model on the given data.
        
        Args:
            data_loader: DataLoader containing input data and targets OR tensor of input data
            targets: Target tensor (required if data_loader is a tensor)
            batch_size: Batch size (used when creating DataLoader from tensor)
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if self.criterion is None:
            raise ValueError("Model not compiled. Call compile() before evaluate().")
        
        # Handle tensor input by converting to DataLoader
        if isinstance(data_loader, torch.Tensor):
            if targets is None:
                raise ValueError("targets must be provided when data_loader is a tensor")
            data_loader = create_dataloader_from_tensors(
                data_loader, targets, batch_size=batch_size, shuffle=False, device=self.device
            )
        
        loss, accuracy = self._validate(data_loader)
        
        return {
            'loss': loss,
            'accuracy': accuracy
        }
    
    def _train_epoch(self, train_loader: DataLoader, history: dict, pbar: Optional['TqdmType'] = None) -> tuple:
        """
        Train the model for one epoch with GPU optimizations.
        
        Args:
            train_loader: DataLoader for training data
            history: Training history dictionary
            pbar: Optional progress bar to update
            
        Returns:
            Tuple of (average_train_loss, train_accuracy)
        """
        self.train()
        train_loss = 0.0
        train_batches = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            if isinstance(batch, dict):
                # GPU optimization: non-blocking transfer
                inputs = batch['input'].to(self.device, non_blocking=True)
                targets = batch['target'].to(self.device, non_blocking=True)
            else:
                inputs, targets = batch
                # GPU optimization: non-blocking transfer
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
            
            # Forward pass with optional AMP
            self.optimizer.zero_grad()
            
            if self.use_amp:
                # Use automatic mixed precision
                with torch.amp.autocast(device_type='cuda'):
                    outputs = self(inputs)
                    loss = self.criterion(outputs, targets)
                
                # Scale loss and backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard precision training
                outputs = self(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
            
            # Step OneCycleLR scheduler after each batch
            if self.scheduler is not None and isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()
                history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # Calculate training metrics
            train_loss += loss.item()
            train_batches += 1
            
            # Calculate training accuracy
            with torch.no_grad():
                _, predicted = torch.max(outputs, 1)
                train_total += targets.size(0)
                train_correct += (predicted == targets).sum().item()
            
            # GPU memory management: clear cache periodically
            if batch_idx % 50 == 0 and self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Update progress bar if provided
            if pbar is not None:
                current_lr = self.optimizer.param_groups[0]['lr']
                current_train_loss = train_loss / train_batches
                current_train_acc = train_correct / train_total
                
                # Update progress bar with training metrics (validation metrics will be updated later)
                postfix = {
                    'train_loss': f'{current_train_loss:.4f}',
                    'train_acc': f'{current_train_acc:.4f}'
                }
                
                # Keep existing validation metrics if they exist
                current_postfix = getattr(pbar, 'postfix', {})
                if isinstance(current_postfix, dict):
                    for key in ['val_loss', 'val_acc']:
                        if key in current_postfix:
                            postfix[key] = current_postfix[key]
                
                # Add lr at the end
                postfix['lr'] = f'{current_lr:.6f}'
                
                pbar.set_postfix(postfix)
                pbar.update(1)
                
                # Force refresh for better notebook display
                try:
                    pbar.refresh()
                except:
                    print("⚠️  Failed to refresh progress bar, continuing without it.")
        
        avg_train_loss = train_loss / train_batches
        train_accuracy = train_correct / train_total
        
        return avg_train_loss, train_accuracy
    
    def _validate(self, data_loader: DataLoader, 
                  pbar: Optional['TqdmType'] = None) -> tuple:
        """
        Validate the model on the given data with GPU optimizations.
        
        Args:
            data_loader: DataLoader containing input data and targets OR tensor of input data
            pbar: Optional progress bar to update during validation
            
        Returns:
            Tuple of (loss, accuracy)
        """
        
        self.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                inputs, targets = batch
                # GPU optimization: non-blocking transfer
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                # Use AMP for validation if enabled
                if self.use_amp:
                    with torch.amp.autocast(device_type='cuda'):
                        outputs = self(inputs)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self(inputs)
                    loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
                # GPU memory management: clear cache periodically
                if batch_idx % 20 == 0 and self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                # Update progress bar if provided
                if pbar is not None:
                    current_val_loss = total_loss / (batch_idx + 1)
                    current_val_acc = correct / total
                    current_lr = self.optimizer.param_groups[0]['lr']
                    
                    # Get existing postfix and update with validation metrics
                    current_postfix = getattr(pbar, 'postfix', {})
                    if isinstance(current_postfix, dict):
                        postfix = current_postfix.copy()
                    else:
                        postfix = {}
                    
                    # Update validation metrics while preserving training metrics
                    postfix.update({
                        'val_loss': f'{current_val_loss:.4f}',
                        'val_acc': f'{current_val_acc:.4f}'
                    })
                    
                    # Add lr at the end
                    postfix['lr'] = f'{current_lr:.6f}'
                    
                    pbar.set_postfix(postfix)
                    pbar.update(1)
        
        avg_loss = total_loss / len(data_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
def resnet18(
    num_classes: int,
    device: Optional[str] = None,
    use_amp: bool = False,
    zero_init_residual: bool = False,
    groups: int = 1,
    width_per_group: int = 64,
    replace_stride_with_dilation: Optional[list[bool]] = None,
    norm_layer: Optional[Callable[..., nn.Module]] = None,
    **kwargs: Any
) -> ResNet:
    """ResNet-18 model for image classification.

    Args:
        num_classes: Number of output classes (required)
        device: Device to place the model on ('cpu', 'cuda', 'mps', etc.). Default: auto-detect
        use_amp: Whether to enable automatic mixed precision training. Default: False
        zero_init_residual: Zero-initialize the last BatchNorm in each residual branch. Default: False
        groups: Number of blocked connections from input channels to output channels. Default: 1
        width_per_group: Width of each group. Default: 64
        replace_stride_with_dilation: Replace 2x2 stride with dilated convolution. Default: None
        norm_layer: Normalization layer. Default: BatchNorm2d
        **kwargs: Additional arguments passed to the ResNet constructor
            
    Returns:
        ResNet: A ResNet-18 model instance
    """
    return ResNet(
        BasicBlock, [2, 2, 2, 2],
        num_classes=num_classes,
        zero_init_residual=zero_init_residual,
        groups=groups,
        width_per_group=width_per_group,
        replace_stride_with_dilation=replace_stride_with_dilation,
        norm_layer=norm_layer,
        device=device,
        use_amp=use_amp,
        **kwargs
    )


def resnet34(
    num_classes: int,
    device: Optional[str] = None,
    use_amp: bool = False,
    zero_init_residual: bool = False,
    groups: int = 1,
    width_per_group: int = 64,
    replace_stride_with_dilation: Optional[list[bool]] = None,
    norm_layer: Optional[Callable[..., nn.Module]] = None,
    **kwargs: Any
) -> ResNet:
    """ResNet-34 model for image classification.

    Args:
        num_classes: Number of output classes (required)
        device: Device to place the model on ('cpu', 'cuda', 'mps', etc.). Default: auto-detect
        use_amp: Whether to enable automatic mixed precision training. Default: False
        zero_init_residual: Zero-initialize the last BatchNorm in each residual branch. Default: False
        groups: Number of blocked connections from input channels to output channels. Default: 1
        width_per_group: Width of each group. Default: 64
        replace_stride_with_dilation: Replace 2x2 stride with dilated convolution. Default: None
        norm_layer: Normalization layer. Default: BatchNorm2d
        **kwargs: Additional arguments passed to the ResNet constructor
            
    Returns:
        ResNet: A ResNet-34 model instance
    """
    return ResNet(
        BasicBlock, [3, 4, 6, 3],
        num_classes=num_classes,
        zero_init_residual=zero_init_residual,
        groups=groups,
        width_per_group=width_per_group,
        replace_stride_with_dilation=replace_stride_with_dilation,
        norm_layer=norm_layer,
        device=device,
        use_amp=use_amp,
        **kwargs
    )


def resnet50(
    num_classes: int,
    device: Optional[str] = None,
    use_amp: bool = False,
    zero_init_residual: bool = False,
    groups: int = 1,
    width_per_group: int = 64,
    replace_stride_with_dilation: Optional[list[bool]] = None,
    norm_layer: Optional[Callable[..., nn.Module]] = None,
    **kwargs: Any
) -> ResNet:
    """ResNet-50 model for image classification.

    This implementation follows the ResNet V1.5 variant which places the stride for 
    downsampling to the second 3x3 convolution, improving accuracy over the original.

    Args:
        num_classes: Number of output classes (required)
        device: Device to place the model on ('cpu', 'cuda', 'mps', etc.). Default: auto-detect
        use_amp: Whether to enable automatic mixed precision training. Default: False
        zero_init_residual: Zero-initialize the last BatchNorm in each residual branch. Default: False
        groups: Number of blocked connections from input channels to output channels. Default: 1
        width_per_group: Width of each group. Default: 64
        replace_stride_with_dilation: Replace 2x2 stride with dilated convolution. Default: None
        norm_layer: Normalization layer. Default: BatchNorm2d
        **kwargs: Additional arguments passed to the ResNet constructor
            
    Returns:
        ResNet: A ResNet-50 model instance
    """
    return ResNet(
        Bottleneck, [3, 4, 6, 3],
        num_classes=num_classes,
        zero_init_residual=zero_init_residual,
        groups=groups,
        width_per_group=width_per_group,
        replace_stride_with_dilation=replace_stride_with_dilation,
        norm_layer=norm_layer,
        device=device,
        use_amp=use_amp,
        **kwargs
    )


def resnet101(
    num_classes: int,
    device: Optional[str] = None,
    use_amp: bool = False,
    zero_init_residual: bool = False,
    groups: int = 1,
    width_per_group: int = 64,
    replace_stride_with_dilation: Optional[list[bool]] = None,
    norm_layer: Optional[Callable[..., nn.Module]] = None,
    **kwargs: Any
) -> ResNet:
    """ResNet-101 model for image classification.

    This implementation follows the ResNet V1.5 variant which places the stride for 
    downsampling to the second 3x3 convolution, improving accuracy over the original.

    Args:
        num_classes: Number of output classes (required)
        device: Device to place the model on ('cpu', 'cuda', 'mps', etc.). Default: auto-detect
        use_amp: Whether to enable automatic mixed precision training. Default: False
        zero_init_residual: Zero-initialize the last BatchNorm in each residual branch. Default: False
        groups: Number of blocked connections from input channels to output channels. Default: 1
        width_per_group: Width of each group. Default: 64
        replace_stride_with_dilation: Replace 2x2 stride with dilated convolution. Default: None
        norm_layer: Normalization layer. Default: BatchNorm2d
        **kwargs: Additional arguments passed to the ResNet constructor
            
    Returns:
        ResNet: A ResNet-101 model instance
    """
    return ResNet(
        Bottleneck, [3, 4, 23, 3],
        num_classes=num_classes,
        zero_init_residual=zero_init_residual,
        groups=groups,
        width_per_group=width_per_group,
        replace_stride_with_dilation=replace_stride_with_dilation,
        norm_layer=norm_layer,
        device=device,
        use_amp=use_amp,
        **kwargs
    )


def resnet152(
    num_classes: int,
    device: Optional[str] = None,
    use_amp: bool = False,
    zero_init_residual: bool = False,
    groups: int = 1,
    width_per_group: int = 64,
    replace_stride_with_dilation: Optional[list[bool]] = None,
    norm_layer: Optional[Callable[..., nn.Module]] = None,
    **kwargs: Any
) -> ResNet:
    """ResNet-152 model for image classification.

    This implementation follows the ResNet V1.5 variant which places the stride for 
    downsampling to the second 3x3 convolution, improving accuracy over the original.

    Args:
        num_classes: Number of output classes (required)
        device: Device to place the model on ('cpu', 'cuda', 'mps', etc.). Default: auto-detect
        use_amp: Whether to enable automatic mixed precision training. Default: False
        zero_init_residual: Zero-initialize the last BatchNorm in each residual branch. Default: False
        groups: Number of blocked connections from input channels to output channels. Default: 1
        width_per_group: Width of each group. Default: 64
        replace_stride_with_dilation: Replace 2x2 stride with dilated convolution. Default: None
        norm_layer: Normalization layer. Default: BatchNorm2d
        **kwargs: Additional arguments passed to the ResNet constructor
            
    Returns:
        ResNet: A ResNet-152 model instance
    """
    return ResNet(
        Bottleneck, [3, 8, 36, 3],
        num_classes=num_classes,
        zero_init_residual=zero_init_residual,
        groups=groups,
        width_per_group=width_per_group,
        replace_stride_with_dilation=replace_stride_with_dilation,
        norm_layer=norm_layer,
        device=device,
        use_amp=use_amp,
        **kwargs
    )