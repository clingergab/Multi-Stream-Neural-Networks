"""
Multi-Channel ResNet implementations.
"""

from typing import Any, Callable, Optional, Union, TYPE_CHECKING
import time
import numpy as np

if TYPE_CHECKING:
    from tqdm import tqdm as TqdmType

from models2.abstracts.abstract_model import BaseModel
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from data_utils.dual_channel_dataset import DualChannelDataset, create_dual_channel_dataloaders, create_dual_channel_dataloader 
from models2.common import (
    setup_scheduler,
    update_scheduler,
    save_checkpoint,
    setup_early_stopping,
    early_stopping_initiated,
    create_progress_bar,
    finalize_progress_bar,
    update_history
)

# Smart tqdm import - detect environment
try:
    from IPython import get_ipython
    if get_ipython() is not None and get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
        try:
            from tqdm.notebook import tqdm
            # Test if widgets work
            test_bar = tqdm(total=1, disable=True)
            test_bar.close()
        except ImportError:
            from tqdm import tqdm
    else:
        from tqdm import tqdm
except:
    from tqdm import tqdm

# Import the multi-channel components
from models2.multi_channel.conv import MCConv2d, MCBatchNorm2d
from models2.multi_channel.blocks import MCBasicBlock, MCBottleneck
from models2.multi_channel.container import MCSequential, MCReLU
from models2.multi_channel.pooling import MCMaxPool2d, MCAdaptiveAvgPool2d


class MCResNet(BaseModel):
    """
    Multi-Channel ResNet implementation.
    
    This model processes color and brightness information through separate pathways
    while maintaining the ResNet architecture's residual connections.
    """
    
    def __init__(
        self,
        block: type[Union[MCBasicBlock, MCBottleneck]],
        layers: list[int],
        num_classes: int,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[list[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        device: Optional[str] = None,
        use_amp: bool = False,
        # MCResNet-specific parameters come AFTER all ResNet parameters
        color_input_channels: int = 3,
        brightness_input_channels: int = 1,
        **kwargs
    ) -> None:
        # Store MCResNet-specific parameters BEFORE calling super().__init__
        # because _build_network() (called by super()) needs these attributes
        self.color_input_channels = color_input_channels
        self.brightness_input_channels = brightness_input_channels
        
        # Set MCResNet default norm layer if not specified
        if norm_layer is None:
            norm_layer = MCBatchNorm2d
        
        # Initialize BaseModel with all ResNet-compatible parameters in exact order
        # BaseModel will handle all the standard ResNet setup (device, inplanes, dilation, etc.)
        super().__init__(
            block=block,
            layers=layers,
            num_classes=num_classes,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            replace_stride_with_dilation=replace_stride_with_dilation,
            norm_layer=norm_layer,
            device=device,
            use_amp=use_amp
        )
    
    def _build_network(
        self,
        block: type[Union[MCBasicBlock, MCBottleneck]],
        layers: list[int],
        replace_stride_with_dilation: list[bool]
    ):
        """Build the Multi-Channel ResNet network architecture."""
        # BaseModel already initialized self.color_inplanes and self.brightness_inplanes to 64
        # for equal scaling, so we can use them directly
        
        # Network architecture - exactly like ResNet but with multi-channel components
        self.conv1 = MCConv2d(
            self.color_input_channels, self.brightness_input_channels, 
            self.color_inplanes, self.brightness_inplanes,
            kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = self._norm_layer(self.color_inplanes, self.brightness_inplanes)
        self.relu = MCReLU(inplace=True)
        self.maxpool = MCMaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, 64, layers[0])       # Equal scaling: 64, 64
        self.layer2 = self._make_layer(block, 128, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])  # Equal scaling: 128, 128
        self.layer3 = self._make_layer(block, 256, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])  # Equal scaling: 256, 256
        self.layer4 = self._make_layer(block, 512, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])  # Equal scaling: 512, 512
        self.avgpool = MCAdaptiveAvgPool2d((1, 1))
        
        # Single classifier for integrated features
        self.fc = nn.Linear(512 * block.expansion * 2, self.num_classes)  # *2 for concatenated features
    
    def _initialize_weights(self, zero_init_residual: bool):
        """Initialize network weights - exactly like ResNet."""
        # Weight initialization - exactly like ResNet
        for m in self.modules():
            if isinstance(m, MCConv2d):
                # Handle MCConv2d - initialize both pathways
                nn.init.kaiming_normal_(m.color_weight, mode="fan_out", nonlinearity="relu")
                nn.init.kaiming_normal_(m.brightness_weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, MCBatchNorm2d):
                # Handle MCBatchNorm2d - initialize both pathways
                nn.init.constant_(m.color_weight, 1)
                nn.init.constant_(m.color_bias, 0)
                nn.init.constant_(m.brightness_weight, 1)
                nn.init.constant_(m.brightness_bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, MCBottleneck) and m.bn3.affine:
                    nn.init.constant_(m.bn3.color_weight, 0)
                    nn.init.constant_(m.bn3.brightness_weight, 0)
                elif isinstance(m, MCBasicBlock) and m.bn2.affine:
                    nn.init.constant_(m.bn2.color_weight, 0)
                    nn.init.constant_(m.bn2.brightness_weight, 0)

    def _make_layer(
        self,
        block: type[Union[MCBasicBlock, MCBottleneck]],
        color_planes: int,
        brightness_planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> MCSequential:
        """
        Create a layer composed of multiple residual blocks.
        Fully compliant with ResNet implementation but using multi-channel blocks with separate channel tracking.
        """
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        
        # Check if downsampling is needed for either pathway
        need_downsample = (stride != 1 or 
                          self.color_inplanes != color_planes * block.expansion or
                          self.brightness_inplanes != brightness_planes * block.expansion)
        
        if need_downsample:
            downsample = MCSequential(
                MCConv2d(
                    self.color_inplanes, self.brightness_inplanes, 
                    color_planes * block.expansion, brightness_planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                norm_layer(color_planes * block.expansion, brightness_planes * block.expansion)
            )

        layers = []
        # First block with potential downsampling - pass parameters exactly like ResNet
        layers.append(
            block(
                self.color_inplanes, 
                self.brightness_inplanes, 
                color_planes,
                brightness_planes,
                stride, 
                downsample, 
                self.groups,
                self.base_width, 
                previous_dilation, 
                norm_layer
            )
        )
        
        # Update channel tracking
        self.color_inplanes = color_planes * block.expansion
        self.brightness_inplanes = brightness_planes * block.expansion
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.color_inplanes,
                    self.brightness_inplanes,
                    color_planes,
                    brightness_planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return MCSequential(*layers)
    
    def forward(self, color_input: Tensor, brightness_input: Tensor) -> Tensor:
        """
        Forward pass through the multi-channel ResNet.
        
        Args:
            color_input: Color input tensor [batch_size, color_channels, height, width]
            brightness_input: Brightness input tensor [batch_size, brightness_channels, height, width]
            
        Returns:
            Classification logits [batch_size, num_classes]
        """
        # Initial convolution
        color_x, brightness_x = self.conv1(color_input, brightness_input)
        color_x, brightness_x = self.bn1(color_x, brightness_x)
        color_x, brightness_x = self.relu(color_x, brightness_x)
        
        # Max pooling
        color_x, brightness_x = self.maxpool(color_x, brightness_x)
        
        # ResNet layers
        color_x, brightness_x = self.layer1(color_x, brightness_x)
        color_x, brightness_x = self.layer2(color_x, brightness_x)
        color_x, brightness_x = self.layer3(color_x, brightness_x)
        color_x, brightness_x = self.layer4(color_x, brightness_x)
        
        # Global average pooling
        color_x, brightness_x = self.avgpool(color_x, brightness_x)
        
        # Flatten
        color_x = torch.flatten(color_x, 1)
        brightness_x = torch.flatten(brightness_x, 1)
        
        # Concatenate features and classify
        fused_features = torch.cat([color_x, brightness_x], dim=1)
        logits = self.fc(fused_features)
        return logits

    def fit(
        self,
        train_loader: Optional[torch.utils.data.DataLoader] = None,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        train_color_data: Optional[torch.Tensor] = None,
        train_brightness_data: Optional[torch.Tensor] = None,
        train_targets: Optional[torch.Tensor] = None,
        val_color_data: Optional[torch.Tensor] = None,
        val_brightness_data: Optional[torch.Tensor] = None,
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
        gradient_accumulation_steps: int = 1,  # NEW: Gradient accumulation for larger effective batch size
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
            gradient_accumulation_steps: Number of steps to accumulate gradients before updating.
                                       Useful for simulating larger batch sizes (e.g., 4 steps = 4x effective batch size)
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
        if train_loader is None:
            # Create DataLoaders from tensor data
            if train_color_data is None or train_brightness_data is None or train_targets is None:
                raise ValueError("Either provide train_loader or all of train_color_data, train_brightness_data, and train_targets")
            
            # Create both train and validation DataLoaders
            print("🔄 Creating DataLoaders from tensors in MCResNet fit() for training using create_dual_channel_dataloaders")
            train_loader, val_loader = create_dual_channel_dataloaders(
                train_color_data, train_brightness_data, train_targets,
                val_color_data, val_brightness_data, val_targets,
                batch_size=batch_size, num_workers=0
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
            avg_train_loss, train_accuracy = self._train_epoch(train_loader, history, pbar, gradient_accumulation_steps)
            
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

    def evaluate(self, 
                data_loader: Optional[torch.utils.data.DataLoader] = None, 
                color_data: Optional[torch.Tensor] = None,
                brightness_data: Optional[torch.Tensor] = None, 
                targets: Optional[torch.Tensor] = None,
                batch_size: int = 32) -> dict:
        """
        Evaluate the model on the given data.
        
        Args:
            data_loader: DataLoader containing dual-channel input data and targets (alternative to direct arrays)
            color_data: Color/RGB input data (if not using data_loader)
            brightness_data: Brightness input data (if not using data_loader)
            targets: Target tensor (required if not using data_loader)
            batch_size: Batch size (used when creating DataLoader from tensors)
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if self.criterion is None:
            raise ValueError("Model not compiled. Call compile() before evaluate().")
        
        # Handle tensor input by converting to DataLoader
        if data_loader is None:
            if color_data is None or brightness_data is None or targets is None:
                raise ValueError("Either provide data_loader or all of color_data, brightness_data, and targets")
            
            # Create DataLoader for evaluation
            data_loader = create_dual_channel_dataloader(
                color_data, brightness_data, targets,
                batch_size=batch_size, num_workers=0
            )
        
        loss, accuracy = self._validate(data_loader)
        
        return {
            'loss': loss,
            'accuracy': accuracy
        }
    def predict(self, 
                data_loader: Optional[torch.utils.data.DataLoader] = None,
                color_data: Optional[torch.Tensor] = None,
                brightness_data: Optional[torch.Tensor] = None, 
                targets: Optional[torch.Tensor] = None,
                batch_size: int = 32) -> torch.Tensor:
        """
        Generate predictions for the input data.
        
        Args:
            data_loader: DataLoader containing dual-channel input data (alternative to direct arrays)
            color_data: Color/RGB input data (if not using data_loader)
            brightness_data: Brightness input data (if not using data_loader) 
            targets: Target tensor for creating dataset (dummy targets if None for prediction)
            batch_size: Batch size (used when creating DataLoader from tensors)
            
        Returns:
            Tensor of predicted classes
        """
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.to(self.device)
        
        # Handle tensor input by converting to DataLoader
        if data_loader is None:
            if color_data is None or brightness_data is None:
                raise ValueError("Either provide data_loader or both color_data and brightness_data")
            
            # Create DataLoader for prediction
            data_loader = create_dual_channel_dataloader(
                color_data, brightness_data, 
                targets if targets is not None else torch.zeros(len(color_data)),
                batch_size=batch_size, num_workers=0
            )
        
        self.eval()
        all_predictions = []
        
        with torch.no_grad():
            for rgb_batch, brightness_batch, targets in data_loader:
                # GPU optimization: non-blocking transfer for dual-channel data
                rgb_batch = rgb_batch.to(self.device, non_blocking=True)
                brightness_batch = brightness_batch.to(self.device, non_blocking=True)
                
                outputs = self(rgb_batch, brightness_batch)
                _, predictions = torch.max(outputs, 1)
                all_predictions.append(predictions.cpu())
        
        return torch.cat(all_predictions, dim=0)
    
    def predict_proba(self, 
                     data_loader: Optional[torch.utils.data.DataLoader] = None,
                     color_data: Optional[torch.Tensor] = None,
                     brightness_data: Optional[torch.Tensor] = None, 
                     targets: Optional[torch.Tensor] = None,
                     batch_size: int = 32) -> np.ndarray:
        """
        Generate probability predictions for the input data.
        
        Args:
            data_loader: DataLoader containing dual-channel input data (alternative to direct arrays)
            color_data: Color/RGB input data (if not using data_loader)
            brightness_data: Brightness input data (if not using data_loader)
            targets: Target tensor for creating dataset (dummy targets if None for prediction)
            batch_size: Batch size (used when creating DataLoader from tensors)
            
        Returns:
            Numpy array of predicted probabilities with shape (n_samples, n_classes)
        """
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.to(self.device)
        
        # Handle tensor input by converting to DataLoader
        if data_loader is None:
            if color_data is None or brightness_data is None:
                raise ValueError("Either provide data_loader or both color_data and brightness_data")
            
            # Create DataLoader for prediction
            data_loader = create_dual_channel_dataloader(
                color_data, brightness_data, 
                targets if targets is not None else torch.zeros(len(color_data)),
                batch_size=batch_size, num_workers=0
            )
        
        self.eval()
        all_probabilities = []
        
        with torch.no_grad():
            for rgb_batch, brightness_batch, targets in data_loader:
                # GPU optimization: non-blocking transfer for dual-channel data
                rgb_batch = rgb_batch.to(self.device, non_blocking=True)
                brightness_batch = brightness_batch.to(self.device, non_blocking=True)
                
                outputs = self(rgb_batch, brightness_batch)
                # Apply softmax to get probabilities
                probabilities = torch.softmax(outputs, dim=1)
                all_probabilities.append(probabilities.cpu().numpy())
        
        return np.concatenate(all_probabilities, axis=0)
    
    
    def _train_epoch(self, train_loader: DataLoader, history: dict, pbar: Optional['TqdmType'] = None, gradient_accumulation_steps: int = 1) -> tuple:
        """
        Train the model for one epoch with GPU optimizations and gradient accumulation.
        
        Args:
            train_loader: DataLoader for training data
            history: Training history dictionary
            pbar: Optional progress bar to update
            gradient_accumulation_steps: Number of steps to accumulate gradients before updating
            
        Returns:
            Tuple of (average_train_loss, train_accuracy)
        """
        self.train()
        train_loss = 0.0
        train_batches = 0
        train_correct = 0
        train_total = 0
        
        # Ensure gradient_accumulation_steps is at least 1 to avoid division by zero
        gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        
        # OPTIMIZATION 1: Progress bar update frequency - major performance improvement
        update_frequency = max(1, len(train_loader) // 50)  # Update only 50 times per epoch
        
        for batch_idx, (rgb_batch, brightness_batch, targets) in enumerate(train_loader):
            # GPU optimization: non-blocking transfer for dual-channel data
            rgb_batch = rgb_batch.to(self.device, non_blocking=True)
            brightness_batch = brightness_batch.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            # OPTIMIZATION 5: Gradient accumulation - zero gradients only when starting accumulation
            if batch_idx % gradient_accumulation_steps == 0:
                self.optimizer.zero_grad()
            
            if self.use_amp:
                # Use automatic mixed precision
                with torch.amp.autocast(device_type='cuda'):
                    outputs = self(rgb_batch, brightness_batch)
                    loss = self.criterion(outputs, targets)
                    # Scale loss for gradient accumulation
                    if gradient_accumulation_steps > 1:
                        loss = loss / gradient_accumulation_steps
                
                # Scale loss and backward pass
                self.scaler.scale(loss).backward()
                
                # OPTIMIZATION 5: Only step optimizer every accumulation_steps
                if (batch_idx + 1) % gradient_accumulation_steps == 0 or batch_idx == len(train_loader) - 1:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                # Standard precision training
                outputs = self(rgb_batch, brightness_batch)
                loss = self.criterion(outputs, targets)
                # Scale loss for gradient accumulation
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps
                loss.backward()
                
                # OPTIMIZATION 5: Only step optimizer every accumulation_steps
                if (batch_idx + 1) % gradient_accumulation_steps == 0 or batch_idx == len(train_loader) - 1:
                    self.optimizer.step()
            
            # Step OneCycleLR scheduler after each optimizer step (not every batch if accumulating)
            if self.scheduler is not None and isinstance(self.scheduler, OneCycleLR):
                if (batch_idx + 1) % gradient_accumulation_steps == 0 or batch_idx == len(train_loader) - 1:
                    self.scheduler.step()
                    history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # Calculate training metrics (use original loss for metrics, not scaled)
            original_loss = loss.item() * gradient_accumulation_steps if gradient_accumulation_steps > 1 else loss.item()
            train_loss += original_loss
            train_batches += 1
            
            # Calculate training accuracy
            with torch.no_grad():
                _, predicted = torch.max(outputs, 1)
                train_total += targets.size(0)
                train_correct += (predicted == targets).sum().item()
            
            # OPTIMIZATION 3: More aggressive memory management for ImageNet
            if batch_idx % 10 == 0 and self.device.type == 'cuda':  # Changed from 50 to 10
                torch.cuda.empty_cache()
            
            # OPTIMIZATION 1: Update progress bar much less frequently - MAJOR SPEEDUP
            if pbar is not None and (batch_idx % update_frequency == 0 or batch_idx == len(train_loader) - 1):
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
                
                # Calculate how many steps to update (avoiding over-updating)
                steps_to_update = min(update_frequency, len(train_loader) - batch_idx)
                if batch_idx == len(train_loader) - 1:
                    # Final batch - only update remaining steps
                    remaining_steps = len(train_loader) - (batch_idx // update_frequency) * update_frequency
                    pbar.update(remaining_steps)
                else:
                    pbar.update(steps_to_update)
                
                # OPTIMIZATION 1: Remove expensive refresh() call - major performance gain
                # try:
                #     pbar.refresh()  # REMOVED - this was causing major slowdowns
                # except:
                #     print("⚠️  Failed to refresh progress bar, continuing without it.")
        
        avg_train_loss = train_loss / train_batches
        train_accuracy = train_correct / train_total
        
        return avg_train_loss, train_accuracy
    
    def _validate(self, data_loader: DataLoader, 
                  pbar: Optional['TqdmType'] = None) -> tuple:
        """
        Validate the model on the given data with GPU optimizations and reduced progress updates.
        
        Args:
            data_loader: DataLoader containing dual-channel input data and targets
            pbar: Optional progress bar to update during validation
            
        Returns:
            Tuple of (loss, accuracy)
        """
        
        self.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # OPTIMIZATION 1: Progress bar update frequency for validation - major performance improvement
        update_frequency = max(1, len(data_loader) // 25)  # Update only 25 times during validation
        
        with torch.no_grad():
            for batch_idx, (rgb_batch, brightness_batch, targets) in enumerate(data_loader):
                # GPU optimization: non-blocking transfer for dual-channel data
                rgb_batch = rgb_batch.to(self.device, non_blocking=True)
                brightness_batch = brightness_batch.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                # Use AMP for validation if enabled
                if self.use_amp:
                    with torch.amp.autocast(device_type='cuda'):
                        outputs = self(rgb_batch, brightness_batch)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self(rgb_batch, brightness_batch)
                    loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
                # OPTIMIZATION 3: More aggressive memory management for validation
                if batch_idx % 5 == 0 and self.device.type == 'cuda':  # Changed from 20 to 5
                    torch.cuda.empty_cache()
                
                # OPTIMIZATION 1: Update progress bar much less frequently during validation
                if pbar is not None and (batch_idx % update_frequency == 0 or batch_idx == len(data_loader) - 1):
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
                    
                    # Calculate how many steps to update (avoiding over-updating)
                    steps_to_update = min(update_frequency, len(data_loader) - batch_idx)
                    if batch_idx == len(data_loader) - 1:
                        # Final batch - only update remaining steps
                        remaining_steps = len(data_loader) - (batch_idx // update_frequency) * update_frequency
                        pbar.update(remaining_steps)
                    else:
                        pbar.update(steps_to_update)
        
        avg_loss = total_loss / len(data_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy

    @property
    def fusion_type(self) -> str:
        """
        The type of fusion used in the model.
        
        Returns:
            A string representing the fusion type.
        """
        return "concatenation"
    
    def _forward_color_pathway(self, color_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the color pathway only.
        
        Args:
            color_input: The color input tensor.
            
        Returns:
            The color pathway output tensor (flattened features).
        """
        # Process through color pathway of multi-channel layers
        color_x = color_input
        
        # Initial convolution (color pathway only)
        color_x = self.conv1.forward_color(color_x)
        color_x = self.bn1.forward_color(color_x)
        color_x = self.relu.forward_color(color_x)
        
        # Max pooling (color pathway only)
        color_x = self.maxpool.forward_color(color_x)
        
        # ResNet layers (color pathway only)
        color_x = self.layer1.forward_color(color_x)
        color_x = self.layer2.forward_color(color_x)
        color_x = self.layer3.forward_color(color_x)
        color_x = self.layer4.forward_color(color_x)
        
        # Global average pooling (color pathway only)
        color_x = self.avgpool.forward_color(color_x)
        
        # Flatten
        color_x = torch.flatten(color_x, 1)
        
        return color_x
    
    def _forward_brightness_pathway(self, brightness_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the brightness pathway only.
        
        Args:
            brightness_input: The brightness input tensor.
            
        Returns:
            The brightness pathway output tensor (flattened features).
        """
        # Process through brightness pathway of multi-channel layers
        brightness_x = brightness_input
        
        # Initial convolution (brightness pathway only)
        brightness_x = self.conv1.forward_brightness(brightness_x)
        brightness_x = self.bn1.forward_brightness(brightness_x)
        brightness_x = self.relu.forward_brightness(brightness_x)
        
        # Max pooling (brightness pathway only)
        brightness_x = self.maxpool.forward_brightness(brightness_x)
        
        # ResNet layers (brightness pathway only)
        brightness_x = self.layer1.forward_brightness(brightness_x)
        brightness_x = self.layer2.forward_brightness(brightness_x)
        brightness_x = self.layer3.forward_brightness(brightness_x)
        brightness_x = self.layer4.forward_brightness(brightness_x)
        
        # Global average pooling (brightness pathway only)
        brightness_x = self.avgpool.forward_brightness(brightness_x)
        
        # Flatten
        brightness_x = torch.flatten(brightness_x, 1)
        
        return brightness_x

    def analyze_pathways(self, 
                        data_loader: Optional[torch.utils.data.DataLoader] = None,
                        color_data: Optional[torch.Tensor] = None,
                        brightness_data: Optional[torch.Tensor] = None,
                        targets: Optional[torch.Tensor] = None,
                        batch_size: int = 32,
                        num_samples: int = 100) -> dict:
        """
        Analyze the contribution and performance of individual pathways.
        
        Args:
            data_loader: DataLoader containing dual-channel input data
            color_data: Color/RGB input data (if not using data_loader)
            brightness_data: Brightness input data (if not using data_loader)
            targets: Target tensor (required if not using data_loader)
            batch_size: Batch size for analysis
            num_samples: Number of samples to analyze (for efficiency)
            
        Returns:
            Dictionary containing pathway analysis results
        """
        if self.criterion is None:
            raise ValueError("Model not compiled. Call compile() before analyze_pathways().")
            
        # Handle tensor input by converting to DataLoader
        if data_loader is None:
            if color_data is None or brightness_data is None or targets is None:
                raise ValueError("Either provide data_loader or all of color_data, brightness_data, and targets")
            
            # Limit samples for efficiency
            if len(color_data) > num_samples:
                indices = torch.randperm(len(color_data))[:num_samples]
                color_data = color_data[indices]
                brightness_data = brightness_data[indices]
                targets = targets[indices]
            
            data_loader = create_dual_channel_dataloader(
                color_data, brightness_data, targets,
                batch_size=batch_size, num_workers=0
            )
        
        self.eval()
        
        # Initialize metrics
        full_model_correct = 0
        color_only_correct = 0
        brightness_only_correct = 0
        total_samples = 0
        
        full_model_loss = 0.0
        color_only_loss = 0.0
        brightness_only_loss = 0.0
        
        color_feature_norms = []
        brightness_feature_norms = []
        
        with torch.no_grad():
            for rgb_batch, brightness_batch, targets_batch in data_loader:
                # Move to device
                rgb_batch = rgb_batch.to(self.device, non_blocking=True)
                brightness_batch = brightness_batch.to(self.device, non_blocking=True)
                targets_batch = targets_batch.to(self.device, non_blocking=True)
                
                batch_size_actual = rgb_batch.size(0)
                total_samples += batch_size_actual
                
                # Full model prediction
                full_outputs = self(rgb_batch, brightness_batch)
                full_loss = self.criterion(full_outputs, targets_batch)
                full_model_loss += full_loss.item() * batch_size_actual
                
                _, full_predicted = torch.max(full_outputs, 1)
                full_model_correct += (full_predicted == targets_batch).sum().item()
                
                # Color pathway only
                color_features = self._forward_color_pathway(rgb_batch)
                color_feature_norms.append(torch.norm(color_features, dim=1).cpu())
                
                # Create dummy brightness features with same shape for FC layer
                brightness_dummy = torch.zeros_like(color_features)
                color_fused = torch.cat([color_features, brightness_dummy], dim=1)
                color_outputs = self.fc(color_fused)
                color_loss = self.criterion(color_outputs, targets_batch)
                color_only_loss += color_loss.item() * batch_size_actual
                
                _, color_predicted = torch.max(color_outputs, 1)
                color_only_correct += (color_predicted == targets_batch).sum().item()
                
                # Brightness pathway only
                brightness_features = self._forward_brightness_pathway(brightness_batch)
                brightness_feature_norms.append(torch.norm(brightness_features, dim=1).cpu())
                
                # Create dummy color features with same shape for FC layer
                color_dummy = torch.zeros_like(brightness_features)
                brightness_fused = torch.cat([color_dummy, brightness_features], dim=1)
                brightness_outputs = self.fc(brightness_fused)
                brightness_loss = self.criterion(brightness_outputs, targets_batch)
                brightness_only_loss += brightness_loss.item() * batch_size_actual
                
                _, brightness_predicted = torch.max(brightness_outputs, 1)
                brightness_only_correct += (brightness_predicted == targets_batch).sum().item()
        
        # Calculate metrics
        full_accuracy = full_model_correct / total_samples
        color_accuracy = color_only_correct / total_samples
        brightness_accuracy = brightness_only_correct / total_samples
        
        avg_full_loss = full_model_loss / total_samples
        avg_color_loss = color_only_loss / total_samples
        avg_brightness_loss = brightness_only_loss / total_samples
        
        # Feature analysis
        color_norms = torch.cat(color_feature_norms, dim=0)
        brightness_norms = torch.cat(brightness_feature_norms, dim=0)
        
        return {
            'accuracy': {
                'full_model': full_accuracy,
                'color_only': color_accuracy,
                'brightness_only': brightness_accuracy,
                'color_contribution': color_accuracy / full_accuracy if full_accuracy > 0 else 0,
                'brightness_contribution': brightness_accuracy / full_accuracy if full_accuracy > 0 else 0
            },
            'loss': {
                'full_model': avg_full_loss,
                'color_only': avg_color_loss,
                'brightness_only': avg_brightness_loss
            },
            'feature_norms': {
                'color_mean': color_norms.mean().item(),
                'color_std': color_norms.std().item(),
                'brightness_mean': brightness_norms.mean().item(),
                'brightness_std': brightness_norms.std().item(),
                'color_to_brightness_ratio': (color_norms.mean() / brightness_norms.mean()).item()
            },
            'samples_analyzed': total_samples
        }
    
    def analyze_pathway_weights(self) -> dict:
        """
        Analyze the weight distributions and magnitudes across pathways.
        
        Returns:
            Dictionary containing weight analysis for color and brightness pathways
        """
        color_weights = {}
        brightness_weights = {}
        
        # Analyze multi-channel layers - look for modules with color_weight and brightness_weight
        for name, module in self.named_modules():
            if hasattr(module, 'color_weight') and hasattr(module, 'brightness_weight'):
                # MCConv2d and MCBatchNorm2d modules
                color_weight = module.color_weight
                brightness_weight = module.brightness_weight
                
                color_weights[name] = {
                    'mean': color_weight.mean().item(),
                    'std': color_weight.std().item(),
                    'norm': torch.norm(color_weight).item(),
                    'shape': list(color_weight.shape)
                }
                
                brightness_weights[name] = {
                    'mean': brightness_weight.mean().item(),
                    'std': brightness_weight.std().item(),
                    'norm': torch.norm(brightness_weight).item(),
                    'shape': list(brightness_weight.shape)
                }
        
        # Calculate overall statistics
        color_norms = [w['norm'] for w in color_weights.values()]
        brightness_norms = [w['norm'] for w in brightness_weights.values()]
        
        return {
            'color_pathway': {
                'layer_weights': color_weights,
                'total_norm': sum(color_norms),
                'mean_norm': sum(color_norms) / len(color_norms) if color_norms else 0,
                'num_layers': len(color_weights)
            },
            'brightness_pathway': {
                'layer_weights': brightness_weights,
                'total_norm': sum(brightness_norms),
                'mean_norm': sum(brightness_norms) / len(brightness_norms) if brightness_norms else 0,
                'num_layers': len(brightness_weights)
            },
            'ratio_analysis': {
                'color_to_brightness_norm_ratio': (sum(color_norms) / sum(brightness_norms)) if brightness_norms else float('inf'),
                'layer_ratios': {
                    name: color_weights[name]['norm'] / brightness_weights[name]['norm'] 
                    if name in brightness_weights and brightness_weights[name]['norm'] > 0 else float('inf')
                    for name in color_weights.keys()
                    if name in brightness_weights
                }
            }
        }
    
    def get_pathway_importance(self, 
                              data_loader: Optional[torch.utils.data.DataLoader] = None,
                              color_data: Optional[torch.Tensor] = None,
                              brightness_data: Optional[torch.Tensor] = None,
                              targets: Optional[torch.Tensor] = None,
                              batch_size: int = 32,
                              method: str = 'gradient') -> dict:
        """
        Calculate pathway importance using different methods.
        
        Args:
            data_loader: DataLoader containing dual-channel input data
            color_data: Color/RGB input data (if not using data_loader)
            brightness_data: Brightness input data (if not using data_loader)
            targets: Target tensor (required if not using data_loader)
            batch_size: Batch size for analysis
            method: Method for importance calculation ('gradient', 'ablation', 'feature_norm')
            
        Returns:
            Dictionary containing pathway importance scores
        """
        if method == 'gradient':
            return self._calculate_gradient_importance(data_loader, color_data, brightness_data, targets, batch_size)
        elif method == 'ablation':
            return self._calculate_ablation_importance(data_loader, color_data, brightness_data, targets, batch_size)
        elif method == 'feature_norm':
            return self._calculate_feature_norm_importance(data_loader, color_data, brightness_data, targets, batch_size)
        else:
            raise ValueError(f"Unknown importance method: {method}. Choose from 'gradient', 'ablation', 'feature_norm'")
    
    def _calculate_gradient_importance(self, data_loader, color_data, brightness_data, targets, batch_size):
        """Calculate importance based on gradient magnitudes."""
        if self.criterion is None:
            raise ValueError("Model not compiled. Call compile() before calculating importance.")
            
        # Handle tensor input
        if data_loader is None:
            if color_data is None or brightness_data is None or targets is None:
                raise ValueError("Either provide data_loader or all input tensors")
                
            # Use a subset for efficiency
            num_samples = min(50, len(color_data))
            indices = torch.randperm(len(color_data))[:num_samples]
            data_loader = create_dual_channel_dataloader(
                color_data[indices], brightness_data[indices], targets[indices],
                batch_size=batch_size, num_workers=0
            )
        
        self.train()  # Enable gradients
        
        color_gradients = []
        brightness_gradients = []
        
        for rgb_batch, brightness_batch, targets_batch in data_loader:
            rgb_batch = rgb_batch.to(self.device, non_blocking=True)
            brightness_batch = brightness_batch.to(self.device, non_blocking=True)
            targets_batch = targets_batch.to(self.device, non_blocking=True)
            
            # Require gradients for inputs
            rgb_batch.requires_grad_(True)
            brightness_batch.requires_grad_(True)
            
            # Forward pass
            outputs = self(rgb_batch, brightness_batch)
            loss = self.criterion(outputs, targets_batch)
            
            # Backward pass
            loss.backward()
            
            # Calculate gradient norms - flatten and then compute norm
            color_grad_norm = torch.norm(rgb_batch.grad.flatten(1), dim=1).mean().item()
            brightness_grad_norm = torch.norm(brightness_batch.grad.flatten(1), dim=1).mean().item()
            
            color_gradients.append(color_grad_norm)
            brightness_gradients.append(brightness_grad_norm)
            
            # Clear gradients
            self.zero_grad()
            rgb_batch.grad = None
            brightness_batch.grad = None
        
        avg_color_grad = sum(color_gradients) / len(color_gradients)
        avg_brightness_grad = sum(brightness_gradients) / len(brightness_gradients)
        total_grad = avg_color_grad + avg_brightness_grad
        
        return {
            'method': 'gradient',
            'color_importance': avg_color_grad / total_grad if total_grad > 0 else 0.5,
            'brightness_importance': avg_brightness_grad / total_grad if total_grad > 0 else 0.5,
            'raw_gradients': {
                'color_avg': avg_color_grad,
                'brightness_avg': avg_brightness_grad
            }
        }
    
    def _calculate_ablation_importance(self, data_loader, color_data, brightness_data, targets, batch_size):
        """Calculate importance based on performance drop when ablating each pathway."""
        # Use the analyze_pathways method for ablation analysis
        pathway_analysis = self.analyze_pathways(data_loader, color_data, brightness_data, targets, batch_size)
        
        full_accuracy = pathway_analysis['accuracy']['full_model']
        color_accuracy = pathway_analysis['accuracy']['color_only']
        brightness_accuracy = pathway_analysis['accuracy']['brightness_only']
        
        # Calculate importance as relative contribution to full model performance
        color_drop = max(0, full_accuracy - brightness_accuracy)  # Performance drop without color
        brightness_drop = max(0, full_accuracy - color_accuracy)  # Performance drop without brightness
        
        total_contribution = color_drop + brightness_drop
        
        return {
            'method': 'ablation',
            'color_importance': color_drop / total_contribution if total_contribution > 0 else 0.5,
            'brightness_importance': brightness_drop / total_contribution if total_contribution > 0 else 0.5,
            'performance_drops': {
                'without_color': color_drop,
                'without_brightness': brightness_drop
            },
            'individual_accuracies': {
                'full_model': full_accuracy,
                'color_only': color_accuracy,
                'brightness_only': brightness_accuracy
            }
        }
    
    def _calculate_feature_norm_importance(self, data_loader, color_data, brightness_data, targets, batch_size):
        """Calculate importance based on feature norm magnitudes."""
        # Use the analyze_pathways method for feature analysis
        pathway_analysis = self.analyze_pathways(data_loader, color_data, brightness_data, targets, batch_size)
        
        color_norm = pathway_analysis['feature_norms']['color_mean']
        brightness_norm = pathway_analysis['feature_norms']['brightness_mean']
        total_norm = color_norm + brightness_norm
        
        return {
            'method': 'feature_norm',
            'color_importance': color_norm / total_norm if total_norm > 0 else 0.5,
            'brightness_importance': brightness_norm / total_norm if total_norm > 0 else 0.5,
            'feature_norms': {
                'color_mean': color_norm,
                'brightness_mean': brightness_norm,
                'ratio': color_norm / brightness_norm if brightness_norm > 0 else float('inf')
            }
        } 

# Factory functions for common architectures
def mc_resnet18(num_classes: int = 1000, **kwargs) -> MCResNet:
    """Create a Multi-Channel ResNet-18 model."""
    return MCResNet(
        MCBasicBlock,
        [2, 2, 2, 2],
        num_classes=num_classes,
        **kwargs
    )

def mc_resnet34(num_classes: int = 1000, **kwargs) -> MCResNet:
    """Create a Multi-Channel ResNet-34 model."""
    return MCResNet(
        MCBasicBlock,
        [3, 4, 6, 3],
        num_classes=num_classes,
        **kwargs
    )


def mc_resnet50(num_classes: int = 1000, **kwargs) -> MCResNet:
    """Create a Multi-Channel ResNet-50 model."""
    return MCResNet(
        MCBottleneck,
        [3, 4, 6, 3],
        num_classes=num_classes,
        **kwargs
    )


def mc_resnet101(num_classes: int = 1000, **kwargs) -> MCResNet:
    """Create a Multi-Channel ResNet-101 model."""
    return MCResNet(
        MCBottleneck,
        [3, 4, 23, 3],
        num_classes=num_classes,
        **kwargs
    )


def mc_resnet152(num_classes: int = 1000, **kwargs) -> MCResNet:
    """Create a Multi-Channel ResNet-152 model."""
    return MCResNet(
        MCBottleneck,
        [3, 8, 36, 3],
        num_classes=num_classes,
        **kwargs
    )