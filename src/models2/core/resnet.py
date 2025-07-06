from functools import partial
from typing import Any, Callable, Optional, Union
import os
import time
import numpy as np

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import (
    StepLR, 
    CosineAnnealingLR, 
    ReduceLROnPlateau, 
    OneCycleLR
)
from tqdm import tqdm

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
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[list[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        
        # Store configuration parameters
        self.block = block
        self.layers = layers
        self.num_classes = num_classes
        self.zero_init_residual = zero_init_residual
        self.groups = groups
        self.width_per_group = width_per_group
        
        # Norm layer setup
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        
        # Dilation setup
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.replace_stride_with_dilation = replace_stride_with_dilation
        
        # Build the network and initialize weights
        self._build_network()
        self._initialize_weights()
        
        # Training components
        self.optimizer = None
        self.criterion = None
        self.scheduler = None
        self.scheduler_type = None
        self.device = None
        # GPU optimization components
        self.use_amp = False
        self.scaler = None
    
    def _build_network(self):
        """Build the ResNet network architecture."""
        self.inplanes = 64
        
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Build residual blocks
        self.layer1 = self._make_layer(self.block, 64, self.layers[0])
        self.layer2 = self._make_layer(self.block, 128, self.layers[1], stride=2, dilate=self.replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(self.block, 256, self.layers[2], stride=2, dilate=self.replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(self.block, 512, self.layers[3], stride=2, dilate=self.replace_stride_with_dilation[2])
        
        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * self.block.expansion, self.num_classes)
    
    def _initialize_weights(self):
        """Initialize model weights using standard initialization schemes."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if self.zero_init_residual:
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
        
        Args:
            block: Block type (BasicBlock or Bottleneck)
            planes: Number of output channels
            blocks: Number of blocks in this layer
            stride: Stride for the first block
            dilate: Whether to use dilated convolutions
            
        Returns:
            Sequential container of blocks
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
        # First block with potential downsampling
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.width_per_group, previous_dilation, norm_layer
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
                    base_width=self.width_per_group,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        """
        Implementation of the forward pass.
        
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

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        return self._forward_impl(x)

    def compile(
        self,
        optimizer: str = 'adam',
        loss: str = 'cross_entropy',
        lr: float = 0.01,
        weight_decay: float = 1e-5,
        device: Optional[str] = None,
        scheduler: Optional[str] = None,
        use_amp: bool = False,  # Enable automatic mixed precision by default
        **kwargs
    ) -> None:
        """
        Compile the model with the given optimizer and loss function.
        
        Args:
            optimizer: Name of the optimizer to use ('adam', 'sgd', or 'adamw')
            loss: Name of the loss function ('cross_entropy', 'focal', or 'multi_stream')
            lr: Learning rate
            weight_decay: Weight decay (L2 regularization)
            device: Device to use for computation ('cpu', 'cuda', 'mps', etc.)
            scheduler: Type of learning rate scheduler ('cosine', 'step', 'plateau', 'onecycle', or None)
            use_amp: Whether to use automatic mixed precision training (only for CUDA)
            **kwargs: Additional arguments to pass to the optimizer or loss function
        """
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
        
        self.to(self.device)
        
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
        
        # Store scheduler type for use in fit method
        self.scheduler_type = scheduler
        
        # Filter optimizer-specific kwargs
        optimizer_kwargs = {k: v for k, v in kwargs.items() 
                           if k not in ['criterion', 'scheduler', 'alpha', 'gamma', 
                                      'classification_weight', 'pathway_consistency_weight']}
        
        # Set optimizer
        if optimizer == 'adam':
            self.optimizer = Adam(self.parameters(), lr=lr, weight_decay=weight_decay, **optimizer_kwargs)
        elif optimizer == 'sgd':
            momentum = kwargs.get('momentum', 0.9)
            self.optimizer = SGD(self.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif optimizer == 'adamw':
            self.optimizer = AdamW(self.parameters(), lr=lr, weight_decay=weight_decay, **optimizer_kwargs)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
        
        # Set loss function
        if loss == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
        elif loss == 'focal':
            from src.training.losses import FocalLoss
            alpha = kwargs.get('alpha', 1.0)
            gamma = kwargs.get('gamma', 2.0)
            self.criterion = FocalLoss(alpha=alpha, gamma=gamma)
        elif loss == 'multi_stream':
            from src.training.losses import MultiStreamLoss
            class_weight = kwargs.get('classification_weight', 1.0)
            path_weight = kwargs.get('pathway_consistency_weight', 0.1)
            self.criterion = MultiStreamLoss(
                classification_weight=class_weight,
                pathway_consistency_weight=path_weight
            )
        else:
            raise ValueError(f"Unsupported loss function: {loss}")
    
    def _setup_scheduler(self, epochs: int, train_loader_len: int, **scheduler_kwargs) -> None:
        """
        Set up the learning rate scheduler based on the scheduler type.
        
        Args:
            epochs: Number of training epochs
            train_loader_len: Length of the training data loader
            **scheduler_kwargs: Additional arguments for the scheduler
        """
        if not self.scheduler_type:
            self.scheduler = None
            return
            
        if self.scheduler_type == 'cosine':
            # Default t_max is set to number of epochs
            t_max = scheduler_kwargs.get('t_max', epochs)
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=t_max)
        elif self.scheduler_type == 'onecycle':
            # For OneCycleLR, we need total number of steps (epochs * steps_per_epoch)
            steps_per_epoch = scheduler_kwargs.get('steps_per_epoch', train_loader_len)
            max_lr = scheduler_kwargs.get('max_lr', self.optimizer.param_groups[0]['lr'] * 10)
            pct_start = scheduler_kwargs.get('pct_start', 0.3)
            anneal_strategy = scheduler_kwargs.get('anneal_strategy', 'cos')
            div_factor = scheduler_kwargs.get('div_factor', 25.0)
            final_div_factor = scheduler_kwargs.get('final_div_factor', 1e4)
            
            # Create the OneCycleLR scheduler with calculated total_steps
            self.scheduler = OneCycleLR(
                self.optimizer, 
                max_lr=max_lr,
                total_steps=epochs * steps_per_epoch,
                pct_start=pct_start,
                anneal_strategy=anneal_strategy,
                div_factor=div_factor,
                final_div_factor=final_div_factor
            )
        elif self.scheduler_type == 'step':
            step_size = scheduler_kwargs.get('step_size', 30)
            gamma = scheduler_kwargs.get('gamma', 0.1)
            self.scheduler = StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif self.scheduler_type == 'plateau':
            # Use scheduler_patience if provided, otherwise fall back to patience from scheduler_kwargs
            scheduler_patience = scheduler_kwargs.get('scheduler_patience', scheduler_kwargs.get('patience', 10))
            factor = scheduler_kwargs.get('factor', 0.5)
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode='min', patience=scheduler_patience, factor=factor
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {self.scheduler_type}")
    
    def _train_epoch(self, train_loader: DataLoader, history: dict, pbar: Optional[tqdm] = None) -> tuple:
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
        
        avg_train_loss = train_loss / train_batches
        train_accuracy = train_correct / train_total
        
        return avg_train_loss, train_accuracy
    
    def _update_scheduler(self, val_loss: float) -> None:
        """
        Update the learning rate scheduler.
        
        Args:
            val_loss: Validation loss for plateau scheduler
        """
        if self.scheduler is not None:
            # Skip OneCycleLR as it's updated after each batch
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            elif not isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()
    
    def _print_epoch_progress(self, epoch: int, epochs: int, epoch_time: float, 
                             avg_train_loss: float, train_accuracy: float,
                             val_loss: float, val_acc: float, val_loader: bool) -> None:
        """
        Print training progress for the current epoch.
        
        Args:
            epoch: Current epoch number (0-based)
            epochs: Total number of epochs
            epoch_time: Time taken for this epoch
            avg_train_loss: Average training loss
            train_accuracy: Training accuracy
            val_loss: Validation loss
            val_acc: Validation accuracy
            val_loader: Whether validation loader was provided
        """
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Time: {epoch_time:.2f}s - "
              f"Train Loss: {avg_train_loss:.4f} - "
              f"Train Acc: {train_accuracy*100:.2f}%", end="")
        
        if val_loader:
            print(f" - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%")
        else:
            print("")
    
    def fit(
        self,
        train_loader: Union[torch.utils.data.DataLoader, torch.Tensor],
        val_loader: Optional[Union[torch.utils.data.DataLoader, torch.Tensor]] = None,
        train_targets: Optional[torch.Tensor] = None,
        val_targets: Optional[torch.Tensor] = None,
        epochs: int = 10,
        batch_size: int = 32,
        callbacks: Optional[list] = None,
        verbose: bool = True,
        save_path: Optional[str] = None,
        # Early stopping parameters
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
            train_loader = self._create_dataloader_from_tensors(
                train_loader, train_targets, batch_size=batch_size, shuffle=True
            )
        
        if isinstance(val_loader, torch.Tensor):
            if val_targets is None:
                raise ValueError("val_targets must be provided when val_loader is a tensor")
            val_loader = self._create_dataloader_from_tensors(
                val_loader, val_targets, batch_size=batch_size, shuffle=False
            )
        
        callbacks = callbacks or []
        
        # Early stopping setup
        if early_stopping and val_loader is None:
            print("⚠️  Early stopping requested but no validation data provided. Disabling early stopping.")
            early_stopping = False
        
        # Initialize early stopping variables
        if early_stopping:
            if monitor == 'val_loss':
                best_metric = float('inf')
                is_better = lambda current, best: current < (best - min_delta)
            elif monitor == 'val_accuracy':
                best_metric = 0.0
                is_better = lambda current, best: current > (best + min_delta)
            else:
                raise ValueError(f"Unsupported monitor metric: {monitor}. Use 'val_loss' or 'val_accuracy'.")
            
            patience_counter = 0
            best_epoch = 0
            best_weights = None
            
            if verbose:
                print(f"🛑 Early stopping enabled: monitoring {monitor} with patience={patience}, min_delta={min_delta}")
        
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

        history['learning_rates'] = []
        
        # Set up scheduler
        self._setup_scheduler(epochs, len(train_loader), **scheduler_kwargs)
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Calculate total steps for this epoch (training + validation)
            total_steps = len(train_loader)
            if val_loader:
                total_steps += len(val_loader)
            
            # Create progress bar for the entire epoch
            if verbose:
                pbar = tqdm(
                    total=total_steps,
                    desc=f"Epoch {epoch+1}/{epochs}",
                    leave=True,
                    dynamic_ncols=True,
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
                )
            else:
                pbar = None
            
            # Training phase - use helper method
            avg_train_loss, train_accuracy = self._train_epoch(train_loader, history, pbar)
            history['train_loss'].append(avg_train_loss)
            history['train_accuracy'].append(train_accuracy)
            
            # Validation phase
            val_loss = 0.0
            val_acc = 0.0
            
            if val_loader:
                val_loss, val_acc = self._validate(val_loader, pbar=pbar)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_acc)
                
                # Legacy save best model (preserve existing functionality)
                if save_path and val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self._save_checkpoint(save_path, history)
                
                # Early stopping logic
                if early_stopping:
                    current_metric = val_loss if monitor == 'val_loss' else val_acc
                    
                    if is_better(current_metric, best_metric):
                        best_metric = current_metric
                        best_epoch = epoch
                        patience_counter = 0
                        
                        # Save best weights for restoration
                        if restore_best_weights:
                            best_weights = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                        
                        if verbose and pbar is None:  # Only print if no progress bar
                            print(f"✅ New best {monitor}: {current_metric:.4f}")
                    else:
                        patience_counter += 1
                        if verbose and pbar is None:  # Only print if no progress bar
                            print(f"⏳ No improvement for {patience_counter}/{patience} epochs (best {monitor}: {best_metric:.4f} at epoch {best_epoch + 1})")
                    
                    # Check if we should stop early
                    if patience_counter >= patience:
                        # Update progress bar with final early stopping status before closing
                        if pbar is not None:
                            final_postfix = {
                                'train_loss': f'{avg_train_loss:.4f}',
                                'train_acc': f'{train_accuracy:.4f}',
                                'val_loss': f'{val_loss:.4f}',
                                'val_acc': f'{val_acc:.4f}',
                                'early_stop': 'TRIGGERED',
                                'lr': f'{current_lr:.6f}'
                            }
                            pbar.set_postfix(final_postfix)
                            pbar.refresh()
                            pbar.close()
                            pbar = None  # Prevent double closing
                        
                        if verbose and pbar is None:  # Only print if no progress bar was used
                            print(f"🛑 Early stopping triggered after {epoch + 1} epochs")
                            print(f"   Best {monitor}: {best_metric:.4f} at epoch {best_epoch + 1}")
                        
                        # Restore best weights if requested
                        if restore_best_weights and best_weights is not None:
                            self.load_state_dict({k: v.to(self.device) for k, v in best_weights.items()})
                            if verbose and pbar is None:  # Only print if no progress bar
                                print("🔄 Restored best model weights")
                        
                        break
            
            # Update learning rate scheduler
            self._update_scheduler(val_loss)
                    
            current_lr = self.optimizer.param_groups[0]['lr']
            history['learning_rates'].append(current_lr)
            
            # Update progress bar with final epoch metrics and close it (if not already closed by early stopping)
            if pbar is not None:
                final_postfix = {
                    'train_loss': f'{avg_train_loss:.4f}',
                    'train_acc': f'{train_accuracy:.4f}'
                }
                if val_loader:
                    final_postfix.update({
                        'val_loss': f'{val_loss:.4f}',
                        'val_acc': f'{val_acc:.4f}'
                    })
                
                # Add early stopping info to progress bar
                if early_stopping and val_loader is not None:
                    if patience_counter >= patience:
                        final_postfix['early_stop'] = 'TRIGGERED'
                    elif patience_counter > 0:
                        final_postfix['patience'] = f'{patience_counter}/{patience}'
                    else:
                        final_postfix['best'] = f'{best_metric:.4f}'
                
                # Add lr at the end
                final_postfix['lr'] = f'{current_lr:.6f}'
                
                pbar.set_postfix(final_postfix)
                pbar.refresh()  # Force update before closing
                pbar.close()
                pbar = None  # Ensure we don't try to close it again
            
            # Call callbacks
            for callback in callbacks:
                callback.on_epoch_end(epoch, {
                    'train_loss': avg_train_loss,
                    'train_accuracy': train_accuracy,
                    'val_loss': val_loss,
                    'val_accuracy': val_acc
                })
        
        # Final early stopping summary
        if early_stopping and val_loader is not None:
            history['early_stopping'] = {
                'stopped_early': patience_counter >= patience,
                'best_epoch': best_epoch + 1,
                'best_metric': best_metric,
                'monitor': monitor,
                'patience': patience,
                'min_delta': min_delta
            }
            
            if verbose and patience_counter < patience:
                print(f"🏁 Training completed without early stopping. Best {monitor}: {best_metric:.4f} at epoch {best_epoch + 1}")
        
        return history
    
    def predict(self, data_loader: Union[torch.utils.data.DataLoader, torch.Tensor], 
                batch_size: int = 32) -> torch.Tensor:
        """
        Generate predictions for the input data.
        
        Args:
            data_loader: DataLoader containing input data OR tensor of input data
            batch_size: Batch size (used when creating DataLoader from tensor)
            
        Returns:
            Tensor of predicted classes
        """
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.to(self.device)
        
        # Handle tensor input by converting to DataLoader
        if isinstance(data_loader, torch.Tensor):
            data_loader = self._create_dataloader_from_tensors(
                data_loader, batch_size=batch_size, shuffle=False
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
            data_loader = self._create_dataloader_from_tensors(
                data_loader, targets, batch_size=batch_size, shuffle=False
            )
        
        loss, accuracy = self._validate(data_loader)
        
        return {
            'loss': loss,
            'accuracy': accuracy
        }
    
    def _validate(self, data_loader: Union[DataLoader, torch.Tensor], 
                  targets: Optional[torch.Tensor] = None, batch_size: int = 32, 
                  pbar: Optional[tqdm] = None) -> tuple:
        """
        Validate the model on the given data with GPU optimizations.
        
        Args:
            data_loader: DataLoader containing input data and targets OR tensor of input data
            targets: Target tensor (required if data_loader is a tensor)
            batch_size: Batch size (used when creating DataLoader from tensor)
            pbar: Optional progress bar to update during validation
            
        Returns:
            Tuple of (loss, accuracy)
        """
        # Handle tensor input by converting to DataLoader
        if isinstance(data_loader, torch.Tensor):
            if targets is None:
                raise ValueError("targets must be provided when data_loader is a tensor")
            data_loader = self._create_dataloader_from_tensors(
                data_loader, targets, batch_size=batch_size, shuffle=False
            )
        
        self.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                if isinstance(batch, dict):
                    # GPU optimization: non-blocking transfer
                    inputs = batch['input'].to(self.device, non_blocking=True)
                    targets = batch['target'].to(self.device, non_blocking=True)
                else:
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
    
    def _save_checkpoint(self, path: str, history: Optional[dict] = None) -> None:
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            history: Training history to save (optional)
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
        }
        
        # Add history if provided
        if history is not None:
            checkpoint['history'] = history
        
        torch.save(checkpoint, path)

    def _create_dataloader_from_tensors(self, X: torch.Tensor, y: Optional[torch.Tensor] = None, 
                                       batch_size: int = 32, shuffle: bool = False) -> DataLoader:
        """
        Create a GPU-optimized DataLoader from tensor data.
        
        Args:
            X: Input tensor data
            y: Target tensor data (optional for prediction)
            batch_size: Batch size for the DataLoader
            shuffle: Whether to shuffle the data
            
        Returns:
            DataLoader containing the tensor data with GPU optimizations
        """
        from torch.utils.data import TensorDataset
        
        if y is not None:
            dataset = TensorDataset(X, y)
        else:
            dataset = TensorDataset(X)
        
        # GPU-optimized DataLoader settings
        if self.device.type == 'cuda':
            return DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=shuffle,
                num_workers=4,  # Use multiple workers for CUDA
                pin_memory=True,  # Pin memory for faster GPU transfer
                persistent_workers=True  # Keep workers alive between epochs
            )
        elif self.device.type == 'mps':
            return DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=shuffle,
                num_workers=0,  # MPS works better with num_workers=0
                pin_memory=False  # Pin memory not beneficial for MPS
            )
        else:
            return DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=shuffle,
                num_workers=0,  # CPU training
                pin_memory=False
            )


def _resnet(
    block: type[Union[BasicBlock, Bottleneck]],
    layers: list[int],
    progress: bool,
    **kwargs: Any,
) -> ResNet:

    model = ResNet(block, layers, **kwargs)

    return model



def resnet18(*, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-18 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.

    Args:
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    """

    return _resnet(BasicBlock, [2, 2, 2, 2], progress, **kwargs)


def resnet34(*, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-34 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.

    Args:
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    """

    return _resnet(BasicBlock, [3, 4, 6, 3], progress, **kwargs)


def resnet50(*, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-50 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.

    .. note::
       The bottleneck of TorchVision places the stride for downsampling to the second 3x3
       convolution while the original paper places it to the first 1x1 convolution.
       This variant improves the accuracy and is known as `ResNet V1.5
       <https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch>`_.

    Args:
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    """

    return _resnet(Bottleneck, [3, 4, 6, 3], progress, **kwargs)


def resnet101(*, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-101 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.

    .. note::
       The bottleneck of TorchVision places the stride for downsampling to the second 3x3
       convolution while the original paper places it to the first 1x1 convolution.
       This variant improves the accuracy and is known as `ResNet V1.5
       <https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch>`_.

    Args:
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    """

    return _resnet(Bottleneck, [3, 4, 23, 3], progress, **kwargs)


def resnet152(*, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-152 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.

    .. note::
       The bottleneck of TorchVision places the stride for downsampling to the second 3x3
       convolution while the original paper places it to the first 1x1 convolution.
       This variant improves the accuracy and is known as `ResNet V1.5
       <https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch>`_.

    Args:
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    """

    return _resnet(Bottleneck, [3, 8, 36, 3], progress, **kwargs)