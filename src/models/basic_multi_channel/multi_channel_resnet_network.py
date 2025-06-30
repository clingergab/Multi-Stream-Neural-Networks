"""
Multi-Channel ResNet implementation following standard ResNet architecture.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import time
import os
import sys
from abc import ABC

# Import Multi-Channel model components
from src.models.base import BaseMultiStreamModel
from src.utils.device_utils import DeviceManager
from src.models.layers.conv_layers import (
    MultiChannelConv2d,
    MultiChannelBatchNorm2d,
    MultiChannelActivation,
    MultiChannelAdaptiveAvgPool2d
)
from src.models.layers.resnet_blocks import (
    MultiChannelResNetBasicBlock,
    MultiChannelResNetBottleneck,
    MultiChannelDownsample,
    MultiChannelSequential
)

def safe_clip_grad_norm(parameters, max_value, norm_type=2.0):
    """
    Safely clips gradient norm while handling edge cases like NaN/Inf.
    This prevents catastrophic failure when gradients are unstable.
    
    Args:
        parameters: Model parameters to clip
        max_value: Maximum norm value
        norm_type: Type of norm (usually 2.0 for L2 norm)
    
    Returns:
        Total norm of parameters (before clipping)
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    
    # Calculate the norm
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), 
        norm_type
    )
    
    # Handle NaN/Inf
    if torch.isnan(total_norm) or torch.isinf(total_norm):
        for p in parameters:
            # Set gradients to zero if they contain NaN/Inf
            if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                p.grad.detach().zero_()
        return torch.tensor(0.0)
    
    # Apply clipping if needed
    if max_value > 0 and total_norm > max_value:
        clip_coef = max_value / (total_norm + 1e-6)
        for p in parameters:
            if p.grad is not None:
                p.grad.detach().mul_(clip_coef)
    
    return total_norm

class MultiChannelResNetNetwork(BaseMultiStreamModel):
    """
    Multi-Channel ResNet Network for image-based multi-stream data.
    
    Follows standard ResNet architecture but processes two streams 
    (color and brightness) separately with proper residual connections.
    
    Suitable for:
    - Image classification with RGB + brightness streams
    - Computer vision tasks requiring multi-modal processing
    - Spatial feature extraction with residual learning
    
    Fusion Strategies:
    -----------------
    - **Shared Classifier (default)**: Feature-level fusion where features from both 
      streams are concatenated and passed through a unified classifier for integrated 
      decision-making. Better for learning complex inter-stream relationships.
    - **Separate Classifiers**: Output-level fusion where each stream has its own 
      classifier and outputs are added. Simpler but less expressive fusion.
    
    API Design:
    -----------
    This model follows a simplified, clean API design:
    
    1. **forward()** - The primary method for training, inference, and evaluation
       - Called automatically by model(x, y)
       - Returns single tensor suitable for loss computation
       - Use this for all training and classification tasks
    
    2. **analyze_pathways()** - For research and analysis purposes only
       - Returns separate outputs for each stream/pathway
       - Use this to analyze individual pathway contributions
       - Never use this for training (returns tuple, not single tensor)
    
    Example Usage:
    -------------
    # Training/inference (standard PyTorch pattern):
    model = MultiChannelResNetNetwork(...)
    output = model(color_data, brightness_data)  # Single tensor
    loss = criterion(output, labels)  # Works seamlessly
    
    # Research/analysis:
    color_logits, brightness_logits = model.analyze_pathways(color_data, brightness_data)
    color_accuracy = accuracy_metric(color_logits, labels)
    brightness_accuracy = accuracy_metric(brightness_logits, labels)
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        color_input_channels: int = 3,
        brightness_input_channels: int = 1,
        num_blocks: List[int] = [2, 2, 2, 2],
        block_type: str = 'basic',
        activation: str = 'relu',
        dropout: float = 0.0,  
        use_shared_classifier: bool = True,
        device: str = 'auto',
        reduce_architecture: bool = False,  # Set to True for small datasets like CIFAR
        **kwargs
    ):
        
        # Initialize base class
        super().__init__(
            input_size=(max(color_input_channels, brightness_input_channels), 224, 224),
            hidden_size=64,  # Initial channel count
            num_classes=num_classes,
            **kwargs
        )
        
        self.color_input_channels = color_input_channels
        self.brightness_input_channels = brightness_input_channels
        self.num_blocks = num_blocks
        self.block_type = block_type
        self.activation = activation
        self.dropout = dropout
        self.use_shared_classifier = use_shared_classifier
        self.reduce_architecture = reduce_architecture
        
        # Setup device management with proper detection
        self.device_manager = DeviceManager(preferred_device=device if device != 'auto' else None)
        self.device = self.device_manager.device
        
        # Mixed precision support
        self.use_mixed_precision = self.device_manager.enable_mixed_precision()
        self.scaler = GradScaler() if self.use_mixed_precision else None
        
        # Track current channel count for proper ResNet progression
        self.inplanes = 64
        
        # Choose block type
        if block_type == 'basic':
            self.block = MultiChannelResNetBasicBlock
        elif block_type == 'bottleneck':
            self.block = MultiChannelResNetBottleneck
        else:
            raise ValueError(f"Unsupported block type: {block_type}")
        
        # Build network
        self._build_network()
        
        # Initialize weights
        self._initialize_weights()
        
        # Move model to device and optimize
        self.to(self.device)
        self.device_manager.optimize_for_device(self)
        
        # Training state
        self.is_compiled = False
        self.optimizer = None
        self.criterion = None
        self.metrics = []
    
    def _build_network(self):
        """Build the ResNet network with optimized architecture for different input channels."""
        # Adjust initial channel count for reduced architecture
        initial_channels = 32 if self.reduce_architecture else 64
        self.inplanes = initial_channels  # Update inplanes for reduced architecture
        
        # Initial layers (stem) - OPTIMIZED for CIFAR-100
        # For CIFAR-100, a smaller kernel and stride works better than ImageNet-style (7x7, stride 2)
        self.conv1 = MultiChannelConv2d(
            color_in_channels=self.color_input_channels,        # 3 for RGB
            brightness_in_channels=self.brightness_input_channels,  # 1 for brightness - efficient!
            out_channels=initial_channels, 
            kernel_size=3, stride=1, padding=1, bias=False  # Smaller kernel and no stride for small images
        )
        self.bn1 = MultiChannelBatchNorm2d(initial_channels)
        self.activation_initial = MultiChannelActivation(self.activation, inplace=True)
        # Optional maxpool - for CIFAR, skipping initial maxpool helps preserve spatial information
        # Since CIFAR images are only 32x32, aggressive downsampling can hurt performance
        self.maxpool = nn.Identity() if self.reduce_architecture else nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers with proper channel progression
        # For CIFAR-100, we need fewer channels and less aggressive downsampling
        if self.reduce_architecture:
            # Use smaller strides in earlier layers to preserve spatial information
            self.layer1 = self._make_layer(64, self.num_blocks[0], stride=1)
            self.layer2 = self._make_layer(128, self.num_blocks[1], stride=2)
            self.layer3 = self._make_layer(256, self.num_blocks[2], stride=2)
            self.layer4 = self._make_layer(512, self.num_blocks[3], stride=2)
        else:
            # Standard ResNet architecture for larger images
            self.layer1 = self._make_layer(64, self.num_blocks[0], stride=1)
            self.layer2 = self._make_layer(128, self.num_blocks[1], stride=2)
            self.layer3 = self._make_layer(256, self.num_blocks[2], stride=2)
            self.layer4 = self._make_layer(512, self.num_blocks[3], stride=2)
        
        # Global average pooling and classifier
        self.avgpool = MultiChannelAdaptiveAvgPool2d((1, 1))
        
        # Final classifier - calculate final feature size based on architecture
        # If using reduced architecture, final features will be smaller
        final_features = (512 if not self.reduce_architecture else 256) * self.block.expansion
        
        # Add dropout before classifier if specified - helps prevent overfitting
        # Higher dropout for reduced architecture to prevent overfitting on smaller dataset
        dropout_rate = self.dropout
        if self.reduce_architecture and dropout_rate < 0.2:
            dropout_rate = max(0.2, dropout_rate)  # Minimum 0.2 dropout for CIFAR
        
        if dropout_rate > 0:
            self.dropout_layer = nn.Dropout(dropout_rate)
        else:
            self.dropout_layer = None
        
        if self.use_shared_classifier:
            # Shared classifier with proper fusion - concatenates features from both streams
            self.shared_classifier = nn.Linear(final_features * 2, self.num_classes, bias=True)
            # Also create separate projection heads for research/analysis purposes
            self.color_head = nn.Linear(final_features, self.num_classes, bias=True)
            self.brightness_head = nn.Linear(final_features, self.num_classes, bias=True)
        else:
            # Legacy: Separate classifiers for each stream
            self.shared_classifier = None  # Not used in separate mode
            self.color_classifier = nn.Linear(final_features, self.num_classes)
            self.brightness_classifier = nn.Linear(final_features, self.num_classes)
            
        # Adjust batch normalization momentum for all BatchNorm layers
        # Use a lower momentum value for more stable updates on small batches
        momentum = 0.05 if self.reduce_architecture else 0.1
        for module in self.modules():
            if isinstance(module, MultiChannelBatchNorm2d):
                module.color_bn.momentum = momentum
                module.brightness_bn.momentum = momentum
    
    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> MultiChannelSequential:
        """
        Create a ResNet layer with multiple blocks.
        
        Args:
            planes: Number of output channels for this layer
            blocks: Number of blocks in this layer
            stride: Stride for the first block (for downsampling)
            
        Returns:
            MultiChannelSequential containing all blocks
        """
        downsample = None
        
        # Reduce the number of channels if using reduced architecture
        if self.reduce_architecture:
            planes = planes // 2  # Halve the number of channels for CIFAR-100
        
        # Create downsample if dimensions change
        if stride != 1 or self.inplanes != planes * self.block.expansion:
            downsample = MultiChannelDownsample(
                self.inplanes,
                planes * self.block.expansion,
                stride
            )
        
        layers = []
        
        # First block (might downsample)
        layers.append(self.block(
            self.inplanes, 
            planes, 
            stride, 
            downsample,
            self.activation
        ))
        
        # Update inplanes for subsequent blocks
        self.inplanes = planes * self.block.expansion
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(self.block(
                self.inplanes,
                planes,
                activation=self.activation
            ))
        
        return MultiChannelSequential(*layers)
    
    def _initialize_weights(self):
        """Initialize network weights with conservative values to prevent exploding gradients."""
        for module in self.modules():
            if isinstance(module, MultiChannelConv2d):
                # Switch to fan_in mode for convolutions to prevent exploding activations
                # This is more appropriate for deep networks on smaller datasets like CIFAR
                nn.init.kaiming_normal_(module.color_weight, mode='fan_in', nonlinearity='relu')
                nn.init.kaiming_normal_(module.brightness_weight, mode='fan_in', nonlinearity='relu')
                # Zero out bias if present (for stability)
                if module.color_bias is not None:
                    nn.init.zeros_(module.color_bias)
                if module.brightness_bias is not None:
                    nn.init.zeros_(module.brightness_bias)
            elif isinstance(module, MultiChannelBatchNorm2d):
                # Initialize BatchNorm with more conservative weights (0.5 instead of 1.0)
                # to prevent activation explosion in deeper layers
                nn.init.constant_(module.color_bn.weight, 0.5)
                nn.init.constant_(module.color_bn.bias, 0.0)
                nn.init.constant_(module.brightness_bn.weight, 0.5)
                nn.init.constant_(module.brightness_bn.bias, 0.0)
            elif isinstance(module, nn.Linear):
                # Use Xavier initialization for fully connected layers with smaller gain
                nn.init.xavier_normal_(module.weight, gain=0.8)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, color_input: torch.Tensor, brightness_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ResNet network for training and classification.
        
        This is the primary method called by model(x, y) and used for training, inference, and evaluation.
        Returns a single tensor suitable for loss computation and classification.
        
        Args:
            color_input: Color image tensor [batch_size, channels, height, width]
            brightness_input: Brightness image tensor [batch_size, channels, height, width]
            
        Returns:
            Combined classification logits [batch_size, num_classes]
        """
        # Normalize inputs properly with CIFAR-100 mean and std
        if color_input.max() > 1.0 and color_input.min() >= 0.0:
            # Scale from [0-255] to [0-1] range
            color_input = color_input / 255.0
            
        # Apply CIFAR-100 mean/std normalization for RGB channels (ImageNet-style normalization)
        # CIFAR-100 approximate mean: [0.5071, 0.4867, 0.4408], std: [0.2675, 0.2565, 0.2761]
        if color_input.shape[1] == 3:  # Only apply to RGB inputs
            mean = torch.tensor([0.5071, 0.4867, 0.4408], device=color_input.device).view(1, 3, 1, 1)
            std = torch.tensor([0.2675, 0.2565, 0.2761], device=color_input.device).view(1, 3, 1, 1)
            color_input = (color_input - mean) / std
            
        # Normalize brightness channel (standardize it for better gradient flow)
        if brightness_input.max() > 1.0 and brightness_input.min() >= 0.0:
            brightness_input = brightness_input / 255.0
            
        # Standardize brightness for better training
        if brightness_input.shape[1] == 1:  # Brightness channel
            # Compute batch statistics for standardization
            batch_mean = brightness_input.mean(dim=(2, 3), keepdim=True)
            batch_std = brightness_input.std(dim=(2, 3), keepdim=True) + 1e-6  # Avoid division by zero
            brightness_input = (brightness_input - batch_mean) / batch_std
            
        # Ensure input shapes are compatible
        if color_input.dim() != 4 or brightness_input.dim() != 4:
            raise ValueError(f"Expected 4D tensors (batch_size, channels, height, width), got {color_input.shape} and {brightness_input.shape}")
            
        # Check for NaN inputs
        if torch.isnan(color_input).any() or torch.isnan(brightness_input).any():
            raise ValueError("Input contains NaN values")
            
        # Initial layers
        color_x, brightness_x = self.conv1(color_input, brightness_input)
        color_x, brightness_x = self.bn1(color_x, brightness_x)
        color_x, brightness_x = self.activation_initial(color_x, brightness_x)
        
        # Apply maxpool to both streams
        color_x = self.maxpool(color_x)
        brightness_x = self.maxpool(brightness_x)
        
        # ResNet layers with stabilization
        color_x, brightness_x = self.layer1(color_x, brightness_x)
        color_x = self._stabilize_activations(color_x)
        brightness_x = self._stabilize_activations(brightness_x)
        
        color_x, brightness_x = self.layer2(color_x, brightness_x)
        color_x = self._stabilize_activations(color_x)
        brightness_x = self._stabilize_activations(brightness_x)
        
        color_x, brightness_x = self.layer3(color_x, brightness_x)
        color_x = self._stabilize_activations(color_x)
        brightness_x = self._stabilize_activations(brightness_x)
        
        color_x, brightness_x = self.layer4(color_x, brightness_x)
        color_x = self._stabilize_activations(color_x)
        brightness_x = self._stabilize_activations(brightness_x)
        
        # Global average pooling
        color_x, brightness_x = self.avgpool(color_x, brightness_x)
        
        # Flatten features
        color_x = torch.flatten(color_x, 1)
        brightness_x = torch.flatten(brightness_x, 1)
        
        # Safety check for NaN features
        if torch.isnan(color_x).any():
            color_x = torch.where(torch.isnan(color_x), torch.zeros_like(color_x), color_x)
        if torch.isnan(brightness_x).any():
            brightness_x = torch.where(torch.isnan(brightness_x), torch.zeros_like(brightness_x), brightness_x)
        
        # Apply dropout if enabled
        if self.dropout_layer is not None:
            color_x = self.dropout_layer(color_x)
            brightness_x = self.dropout_layer(brightness_x)
        
        # Stabilize final features before fusion to prevent gradient explosion
        color_x = self._stabilize_features(color_x)
        brightness_x = self._stabilize_features(brightness_x)
        
        # Combine outputs for classification
        if self.use_shared_classifier:
            # Use shared classifier for optimal fusion
            fused_features = torch.cat([color_x, brightness_x], dim=1)
            logits = self.shared_classifier(fused_features)
            
            # Safety check for NaN outputs
            if torch.isnan(logits).any():
                logits = torch.zeros_like(logits)
                
            return logits
        else:
            # Legacy: Add separate outputs
            color_logits = self.color_classifier(color_x)
            brightness_logits = self.brightness_classifier(brightness_x)
            
            # Combine outputs
            combined_logits = color_logits + brightness_logits
            
            # Safety check for NaN outputs
            if torch.isnan(combined_logits).any():
                combined_logits = torch.zeros_like(combined_logits)
                
            return combined_logits
            
    def _stabilize_activations(self, x):
        """
        Apply activation stabilization to prevent gradient explosion.
        
        Args:
            x: Input tensor
            
        Returns:
            Stabilized tensor
        """
        # Check for extreme values and clip if necessary
        if x.abs().max() > 50.0:
            x = torch.clamp(x, min=-50.0, max=50.0)
        
        # Check for NaN values and replace with zeros
        if torch.isnan(x).any():
            x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
                
        return x
    
    def _stabilize_features(self, x):
        """
        Stabilize feature vectors before classification to prevent gradient explosion.
        
        Args:
            x: Feature tensor [batch_size, features]
            
        Returns:
            Stabilized features
        """
        # Apply L2 normalization if magnitudes are too large
        norm = x.norm(dim=1, keepdim=True)
        if (norm > 10.0).any():
            # Scale down large feature vectors
            scale_factor = torch.clamp(10.0 / norm, max=1.0)
            x = x * scale_factor
            
        return x
    
    def compile(self, optimizer: str = 'adam', learning_rate: float = 0.001, 
                weight_decay: float = 0.0, loss: str = 'cross_entropy', metrics: List[str] = None,
                gradient_clip: float = 1.0, scheduler: str = 'cosine'):
        """
        Compile the model with optimizer, loss function, and gradient clipping (Keras-like API).
        
        Args:
            optimizer: Optimizer name ('adam', 'adamw', 'sgd', 'rmsprop')
            learning_rate: Learning rate
            weight_decay: Weight decay (L2 regularization)
            loss: Loss function name ('cross_entropy')
            metrics: List of metrics to track
            gradient_clip: Maximum norm for gradient clipping
            scheduler: Learning rate scheduler ('cosine', 'onecycle', 'none')
        """
        # Configure optimizers with appropriate parameters for CIFAR-100
        if optimizer.lower() == 'adam':
            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay, eps=1e-8)
        elif optimizer.lower() == 'adamw':
            self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay, eps=1e-8)
        elif optimizer.lower() == 'sgd':
            # Use SGD with Nesterov momentum for CIFAR-100 (standard practice)
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
        
        # Configure learning rate scheduler
        self.scheduler = None
        self.scheduler_type = scheduler.lower()
        
        if self.scheduler_type == 'cosine':
            # Cosine annealing is effective for CIFAR-100
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=100, eta_min=learning_rate * 0.01
            )
        elif self.scheduler_type == 'onecycle':
            # OneCycle is another good option for CIFAR-100
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=learning_rate,
                steps_per_epoch=500,  # This will be updated in fit() with actual steps
                epochs=100,  # This will also be updated in fit()
                pct_start=0.3  # Spend 30% of training warming up
            )
        elif self.scheduler_type != 'none':
            raise ValueError(f"Unsupported scheduler: {scheduler}")
        
        # Store configuration
        self.metrics = metrics or ['accuracy']
        self.gradient_clip = gradient_clip
        self.is_compiled = True
        
        # Log compilation details
        model_name = self.__class__.__name__
        print(f"{model_name} compiled with {optimizer} optimizer, {loss} loss, "
              f"learning rate: {learning_rate}, weight decay: {weight_decay}, "
              f"gradient clip: {gradient_clip}, scheduler: {scheduler}")
              
        # Record configuration for debugging
        self._config = {
            'optimizer': optimizer,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'loss': loss,
            'gradient_clip': gradient_clip,
            'scheduler': scheduler,
            'reduce_architecture': self.reduce_architecture,
            'batch_norm_momentum': 0.05 if self.reduce_architecture else 0.1,
            'dropout': self.dropout,
        }
        
        return self
    
    def fit(self, train_color_data=None, train_brightness_data=None, train_labels=None,
             val_color_data=None, val_brightness_data=None, val_labels=None,
             train_loader=None, val_loader=None, batch_size=None, epochs=10,
             early_stopping_patience=5, verbose=1, num_workers=None, pin_memory=None):
        """
        Unified fit method that handles both direct data arrays and DataLoaders with on-the-fly augmentation.
        
        This enhanced method supports two modes of operation:
        1. Direct data array input: Provide train_color_data, train_brightness_data, train_labels
        2. DataLoader input (for on-the-fly augmentation): Provide train_loader directly
        
        Memory-Efficient Training:
        -------------------------
        For large datasets like ImageNet, it's recommended to:
        - Use custom DataLoaders with transforms (Option 2) to enable on-the-fly augmentation
        - Use a smaller batch_size (16-32 even on large GPUs) to prevent OOM errors
        - Consider using the mixed precision training (enabled by default on compatible GPUs)
        
        Args:
            train_color_data: Training data for color stream [N, C, H, W] (used if train_loader not provided)
            train_brightness_data: Training data for brightness stream [N, C, H, W] (used if train_loader not provided)
            train_labels: Training labels (used if train_loader not provided)
            val_color_data: Validation data for color stream (used if val_loader not provided)
            val_brightness_data: Validation data for brightness stream (used if val_loader not provided)
            val_labels: Validation labels (used if val_loader not provided)
            train_loader: Optional DataLoader for training (will be used instead of direct data if provided)
            val_loader: Optional DataLoader for validation (will be used instead of direct data if provided)
            batch_size: Batch size for training when using direct data (auto-detected if None)
            epochs: Number of epochs to train
            early_stopping_patience: Stop training when validation loss doesn't improve
            verbose: Verbosity level (0: silent, 1: progress bar, 2: one line per epoch)
            num_workers: Number of workers for data loading (auto-detected if None)
            pin_memory: Whether to pin memory for faster GPU transfer (auto-detected if None)
            
        Returns:
            Dictionary with training history
        """
        if not self.is_compiled:
            raise RuntimeError("Model must be compiled before training. Call model.compile() first.")
        
        # Determine if we're using DataLoaders or direct data
        using_dataloaders = train_loader is not None
        
        # Auto-detect optimal batch size for CNN (used only if creating DataLoaders from direct data)
        if batch_size is None and not using_dataloaders:
            if self.device.type == 'cuda':
                # For CNNs, use appropriate batch sizes based on GPU memory
                memory_gb = torch.cuda.get_device_properties(self.device).total_memory / 1e9
                if memory_gb >= 80:  # A100 80GB
                    batch_size = 256  # Much larger batch size for A100 80GB
                elif memory_gb >= 40:  # A100 40GB
                    batch_size = 128  # Larger batch size for A100 40GB
                elif memory_gb >= 16:  # V100
                    batch_size = 64   # Medium batch size for V100
                else:
                    batch_size = 32   # Conservative for smaller GPUs
            else:
                batch_size = 16  # More conservative default for CPU/MPS
        
        # Auto-detect optimal number of workers
        if num_workers is None:
            import os
            num_workers = min(4, os.cpu_count() or 1)  # More conservative default
        
        # Auto-detect pin_memory
        if pin_memory is None:
            pin_memory = self.device.type == 'cuda'
            
        # Training history
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rates': []
        }
        
        # Set up train and validation loaders based on input mode
        if not using_dataloaders:
            # Check that required data is provided when not using DataLoaders
            if train_color_data is None or train_brightness_data is None or train_labels is None:
                raise ValueError("When train_loader is not provided, you must provide train_color_data, train_brightness_data, and train_labels")
            
            # Create DataLoaders from direct data - keep tensors on CPU for memory efficiency
            train_color_tensor = torch.tensor(train_color_data, dtype=torch.float32)
            train_brightness_tensor = torch.tensor(train_brightness_data, dtype=torch.float32)
            train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
            
            train_dataset = TensorDataset(train_color_tensor, train_brightness_tensor, train_labels_tensor)
            train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True,
                num_workers=num_workers if self.device.type != 'mps' else 0,  # MPS doesn't support multiprocessing
                pin_memory=pin_memory,
                persistent_workers=num_workers > 0 and self.device.type != 'mps'
            )
            
            # Set up validation loader if validation data is provided
            if val_color_data is not None and val_brightness_data is not None and val_labels is not None:
                val_color_tensor = torch.tensor(val_color_data, dtype=torch.float32)
                val_brightness_tensor = torch.tensor(val_brightness_data, dtype=torch.float32)
                val_labels_tensor = torch.tensor(val_labels, dtype=torch.long)
                
                val_dataset = TensorDataset(val_color_tensor, val_brightness_tensor, val_labels_tensor)
                val_loader = DataLoader(
                    val_dataset, 
                    batch_size=batch_size, 
                    shuffle=False,
                    num_workers=num_workers if self.device.type != 'mps' else 0,
                    pin_memory=pin_memory,
                    persistent_workers=num_workers > 0 and self.device.type != 'mps'
                )
            
        # Print training configuration
        if verbose > 0:
            model_name = self.__class__.__name__
            print(f"🚀 Training {model_name} with {'DataLoader pipeline' if using_dataloaders else 'direct data'}:")
            print(f"   Device: {self.device}")
            print(f"   Architecture: {'Reduced (CIFAR optimized)' if self.reduce_architecture else 'Full'}")
            print(f"   Mixed precision: {self.use_mixed_precision}")
            print(f"   Gradient clipping: {self.gradient_clip}")
            print(f"   Scheduler: {self.scheduler_type}")
            print(f"   BatchNorm momentum: {0.05 if self.reduce_architecture else 0.1}")
            if using_dataloaders:
                print(f"   Train batches: {len(train_loader)}")
                if val_loader:
                    print(f"   Val batches: {len(val_loader)}")
            else:
                print(f"   Batch size: {batch_size}")
            print(f"   Workers: {num_workers}")
            print(f"   Pin memory: {pin_memory}")
                
        # Update OneCycleLR scheduler if used
        if self.scheduler_type == 'onecycle' and self.scheduler is not None:
            # Reconfigure scheduler with actual steps
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer, 
                max_lr=self._config['learning_rate'],
                steps_per_epoch=len(train_loader),
                epochs=epochs,
                pct_start=0.3
            )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Clear any existing tqdm instances
            try:
                for inst in list(getattr(tqdm, '_instances', [])):
                    try:
                        inst.close()
                    except Exception:
                        pass  # Ignore if we can't close an instance
            except Exception:
                pass  # Ignore any errors in cleaning up tqdm instances
                
            # Single progress bar for training
            epoch_pbar = None
            if verbose == 1:
                epoch_pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}")
                
            # Training phase
            self.train()
            total_loss = 0.0
            correct = 0
            total = 0
            batch_lr = []
            
            # Training loop
            for batch_idx, data in enumerate(train_loader):
                # Unpack data - handle different formats
                if isinstance(data, (list, tuple)) and len(data) == 3:
                    color_input, brightness_input, labels = data
                else:
                    raise ValueError("DataLoader must provide (color, brightness, labels) tuples")
                
                # Move data to device with non-blocking transfer for async
                if color_input.device != self.device:
                    color_input = color_input.to(self.device, non_blocking=True)
                if brightness_input.device != self.device:
                    brightness_input = brightness_input.to(self.device, non_blocking=True)
                if labels.device != self.device:
                    labels = labels.to(self.device, non_blocking=True)
                
                # Check for NaN inputs
                if torch.isnan(color_input).any() or torch.isnan(brightness_input).any():
                    print(f"Warning: NaN detected in batch {batch_idx} inputs. Skipping batch.")
                    continue
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass with mixed precision if available
                if self.use_mixed_precision and self.scaler:
                    with autocast(device_type='cuda'):
                        outputs = self(color_input, brightness_input)
                        loss = self.criterion(outputs, labels)
                        
                    # Backward pass with scaled gradients
                    self.scaler.scale(loss).backward()
                    
                    # Apply gradient clipping if needed
                    if self.gradient_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        safe_clip_grad_norm(self.parameters(), self.gradient_clip)
                    
                    # Update weights with scaled optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Standard precision training
                    outputs = self(color_input, brightness_input)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    
                    # Apply gradient clipping
                    if self.gradient_clip > 0:
                        safe_clip_grad_norm(self.parameters(), self.gradient_clip)
                        
                    # Update weights
                    self.optimizer.step()
                
                # Step the scheduler (for batch-level schedulers like OneCycle)
                if self.scheduler_type == 'onecycle' and self.scheduler:
                    self.scheduler.step()
                    batch_lr.append(self.scheduler.get_last_lr()[0])
                
                # Calculate training metrics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Explicitly clear unnecessary tensors to help with memory management
                del outputs, loss, predicted
                if self.use_mixed_precision and self.scaler is not None and batch_idx % 20 == 0:
                    torch.cuda.empty_cache()  # Periodic cache clearing for large datasets
                
                # Update progress bar
                if verbose == 1 and epoch_pbar is not None:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    epoch_pbar.set_postfix({
                        'loss': f'{total_loss / (batch_idx + 1):.4f}',
                        'acc': f'{correct/total:.4f}',
                        'lr': f'{current_lr:.6f}'
                    })
                    epoch_pbar.update(1)
            
            # Close progress bar
            if verbose == 1 and epoch_pbar is not None:
                epoch_pbar.close()
                
            # Calculate epoch-level metrics
            avg_train_loss = total_loss / len(train_loader)
            train_accuracy = correct / total
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Step the scheduler (for epoch-level schedulers)
            if self.scheduler_type == 'cosine' and self.scheduler:
                self.scheduler.step()
                
            # Record training metrics
            history['train_loss'].append(avg_train_loss)
            history['train_accuracy'].append(train_accuracy)
            history['learning_rates'].append(batch_lr if batch_lr else [current_lr])
            
            # Validation phase
            if val_loader is not None:
                val_loss, val_accuracy = self._validate(val_loader)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_accuracy)
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save the best model
                    self.best_model_state = self.state_dict().copy()
                    if verbose > 0:
                        print(f"✅ New best validation loss: {best_val_loss:.4f}")
                else:
                    patience_counter += 1
                    if verbose > 0:
                        print(f"⏳ No improvement for {patience_counter}/{early_stopping_patience} epochs")
                    
                # Print epoch summary
                if verbose > 0:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"loss: {avg_train_loss:.4f}, acc: {train_accuracy:.4f}, "
                          f"val_loss: {val_loss:.4f}, val_acc: {val_accuracy:.4f}, "
                          f"lr: {current_lr:.6f}")
                    
                # Check for early stopping
                if patience_counter >= early_stopping_patience:
                    print(f"🛑 Early stopping triggered after {epoch+1} epochs.")
                    break
            else:
                # Print epoch summary without validation metrics
                if verbose > 0:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"loss: {avg_train_loss:.4f}, acc: {train_accuracy:.4f}, "
                          f"lr: {current_lr:.6f}")
                          
        # Load best model if validation was performed and we have saved a best state
        if val_loader is not None and hasattr(self, 'best_model_state'):
            self.load_state_dict(self.best_model_state)
            if verbose > 0:
                print("📊 Loaded best model state from early stopping")
        
        # Clear cache after training
        self.device_manager.clear_cache()
        
        return history
        
    def _validate(self, val_loader):
        """
        Validate the model on validation data.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (validation_loss, validation_accuracy)
        """
        self.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for color_input, brightness_input, labels in val_loader:
                # Move data to device
                color_input = color_input.to(self.device, non_blocking=True)
                brightness_input = brightness_input.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                # Forward pass (no need for mixed precision during validation)
                outputs = self(color_input, brightness_input)
                loss = self.criterion(outputs, labels)
                
                # Calculate validation metrics
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        # Calculate average validation metrics
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total
        
        return avg_val_loss, val_accuracy
        
    def _forward_color_pathway(self, color_input):
        """
        Forward pass through the color pathway only.
        
        Args:
            color_input: Color image tensor [batch_size, channels, height, width]
            
        Returns:
            Color pathway features [batch_size, features]
        """
        # Normalize inputs properly with CIFAR-100 mean and std
        if color_input.max() > 1.0 and color_input.min() >= 0.0:
            # Scale from [0-255] to [0-1] range
            color_input = color_input / 255.0
            
        # Apply CIFAR-100 mean/std normalization for RGB channels (ImageNet-style normalization)
        # CIFAR-100 approximate mean: [0.5071, 0.4867, 0.4408], std: [0.2675, 0.2565, 0.2761]
        if color_input.shape[1] == 3:  # Only apply to RGB inputs
            mean = torch.tensor([0.5071, 0.4867, 0.4408], device=color_input.device).view(1, 3, 1, 1)
            std = torch.tensor([0.2675, 0.2565, 0.2761], device=color_input.device).view(1, 3, 1, 1)
            color_input = (color_input - mean) / std
            
        # Initial layers - only process color pathway
        color_x = self.conv1.forward_color(color_input)
        color_x = self.bn1.forward_color(color_x)
        color_x = self.activation_initial.forward_single(color_x)
        
        # Apply maxpool
        color_x = self.maxpool(color_x)
        
        # ResNet layers with stabilization
        color_x = self.layer1.forward_color(color_x)
        color_x = self._stabilize_activations(color_x)
        
        color_x = self.layer2.forward_color(color_x)
        color_x = self._stabilize_activations(color_x)
        
        color_x = self.layer3.forward_color(color_x)
        color_x = self._stabilize_activations(color_x)
        
        color_x = self.layer4.forward_color(color_x)
        color_x = self._stabilize_activations(color_x)
        
        # Global average pooling
        color_x = self.avgpool.forward_single(color_x)
        
        # Flatten features
        color_x = torch.flatten(color_x, 1)
        
        # Safety check for NaN features
        if torch.isnan(color_x).any():
            color_x = torch.where(torch.isnan(color_x), torch.zeros_like(color_x), color_x)
        
        # Apply dropout if enabled
        if self.dropout_layer is not None:
            color_x = self.dropout_layer(color_x)
        
        # Stabilize final features
        color_x = self._stabilize_features(color_x)
        
        return color_x
        
    def _forward_brightness_pathway(self, brightness_input):
        """
        Forward pass through the brightness pathway only.
        
        Args:
            brightness_input: Brightness image tensor [batch_size, channels, height, width]
            
        Returns:
            Brightness pathway features [batch_size, features]
        """
        # Normalize brightness channel
        if brightness_input.max() > 1.0 and brightness_input.min() >= 0.0:
            brightness_input = brightness_input / 255.0
            
        # Standardize brightness for better training
        if brightness_input.shape[1] == 1:  # Brightness channel
            # Compute batch statistics for standardization
            batch_mean = brightness_input.mean(dim=(2, 3), keepdim=True)
            batch_std = brightness_input.std(dim=(2, 3), keepdim=True) + 1e-6  # Avoid division by zero
            brightness_input = (brightness_input - batch_mean) / batch_std
        
        # Initial layers - only process brightness pathway
        brightness_x = self.conv1.forward_brightness(brightness_input)
        brightness_x = self.bn1.forward_brightness(brightness_x)
        brightness_x = self.activation_initial.forward_single(brightness_x)
        
        # Apply maxpool
        brightness_x = self.maxpool(brightness_x)
        
        # ResNet layers with stabilization
        brightness_x = self.layer1.forward_brightness(brightness_x)
        brightness_x = self._stabilize_activations(brightness_x)
        
        brightness_x = self.layer2.forward_brightness(brightness_x)
        brightness_x = self._stabilize_activations(brightness_x)
        
        brightness_x = self.layer3.forward_brightness(brightness_x)
        brightness_x = self._stabilize_activations(brightness_x)
        
        brightness_x = self.layer4.forward_brightness(brightness_x)
        brightness_x = self._stabilize_activations(brightness_x)
        
        # Global average pooling
        brightness_x = self.avgpool.forward_single(brightness_x)
        
        # Flatten features
        brightness_x = torch.flatten(brightness_x, 1)
        
        # Safety check for NaN features
        if torch.isnan(brightness_x).any():
            brightness_x = torch.where(torch.isnan(brightness_x), torch.zeros_like(brightness_x), brightness_x)
        
        # Apply dropout if enabled
        if self.dropout_layer is not None:
            brightness_x = self.dropout_layer(brightness_x)
        
        # Stabilize final features
        brightness_x = self._stabilize_features(brightness_x)
        
        return brightness_x
    
    def analyze_pathways(self, color_input, brightness_input):
        """
        Analyze the contribution of each pathway separately.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output logits from color and brightness pathways
        """
        # Move inputs to the correct device
        color_input = color_input.to(self.device)
        brightness_input = brightness_input.to(self.device)
        
        # Feature extraction
        color_features = self._forward_color_pathway(color_input)
        brightness_features = self._forward_brightness_pathway(brightness_input)
        
        # Get the prediction from each pathway individually
        if self.use_shared_classifier:
            # Apply the separate projection heads to each feature separately
            color_out = self.color_head(color_features)
            brightness_out = self.brightness_head(brightness_features)
        else:
            # Use the separate classifiers
            color_out = self.color_classifier(color_features)
            brightness_out = self.brightness_classifier(brightness_features)
        
        return color_out, brightness_out

    def extract_features(self, color_input, brightness_input):
        """
        Extract features from both pathways without classification.
        
        Args:
            color_input (torch.Tensor): Color input data
            brightness_input (torch.Tensor): Brightness input data
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Features from color and brightness pathways
        """
        # Move inputs to the correct device
        color_input = color_input.to(self.device)
        brightness_input = brightness_input.to(self.device)
        
        # Feature extraction
        color_features = self._forward_color_pathway(color_input)
        brightness_features = self._forward_brightness_pathway(brightness_input)
        
        return color_features, brightness_features

    def get_pathway_importance(self):
        """
        Calculate the relative importance of each pathway based on weights.
        
        Returns:
            Dict[str, float]: Dictionary with importance scores for each pathway
        """
        # This is a simple implementation based on parameter magnitudes
        # A more sophisticated implementation would use attribution methods
        
        color_params = []
        brightness_params = []
        
        # Collect parameters for each pathway
        for name, param in self.named_parameters():
            if 'color' in name:
                color_params.append(param.abs().mean().item())
            elif 'brightness' in name:
                brightness_params.append(param.abs().mean().item())
        
        # Calculate average magnitude for each pathway
        color_importance = sum(color_params) / len(color_params) if color_params else 0
        brightness_importance = sum(brightness_params) / len(brightness_params) if brightness_params else 0
        
        # Normalize to sum to 1.0
        total = color_importance + brightness_importance
        if total > 0:
            color_importance /= total
            brightness_importance /= total
        
        return {
            'color_pathway': color_importance,
            'brightness_pathway': brightness_importance
        }
    
    @classmethod
    def for_cifar100(cls, use_shared_classifier=True, dropout=0.3, block_type='basic'):
        """
        Factory method to create a model pre-configured for CIFAR-100.
        
        This creates a model with reduced architecture and appropriate settings
        specifically optimized for the CIFAR-100 dataset.
        
        Args:
            use_shared_classifier: Whether to use shared classifier for fusion
            dropout: Dropout rate (recommended: 0.2-0.5 for CIFAR)
            block_type: Block type ('basic' or 'bottleneck')
            
        Returns:
            Configured MultiChannelResNetNetwork for CIFAR-100
        """
        model = cls(
            num_classes=100,  # CIFAR-100 has 100 classes
            color_input_channels=3,  # RGB channels
            brightness_input_channels=1,  # L channel
            num_blocks=[2, 2, 2, 2],  # Smaller ResNet-18 style configuration
            block_type=block_type,
            activation='relu',
            dropout=dropout,
            use_shared_classifier=use_shared_classifier,
            reduce_architecture=True,  # Enable CIFAR optimization
            device='auto'  # Auto-detect best device
        )
        
        # Print model configuration summary
        params = sum(p.numel() for p in model.parameters())
        print("🚀 Created CIFAR-100 optimized model:")
        print(f"   Architecture: Reduced ResNet with {block_type} blocks")
        print(f"   Parameters: {params:,}")
        print(f"   Classifier: {'Shared (fusion)' if use_shared_classifier else 'Separate'}")
        print(f"   Device: {model.device}")
        
        return model

# Factory functions to create ResNet variants
def multi_channel_resnet18(num_classes=10, color_input_channels=3, brightness_input_channels=1, **kwargs):
    """
    Creates a ResNet-18 variant of the MultiChannelResNetNetwork
    """
    return MultiChannelResNetNetwork(
        num_classes=num_classes,
        color_input_channels=color_input_channels,
        brightness_input_channels=brightness_input_channels,
        num_blocks=[2, 2, 2, 2],
        block_type='basic',
        **kwargs
    )

def multi_channel_resnet34(num_classes=10, color_input_channels=3, brightness_input_channels=1, **kwargs):
    """
    Creates a ResNet-34 variant of the MultiChannelResNetNetwork
    """
    return MultiChannelResNetNetwork(
        num_classes=num_classes,
        color_input_channels=color_input_channels,
        brightness_input_channels=brightness_input_channels,
        num_blocks=[3, 4, 6, 3],
        block_type='basic',
        **kwargs
    )

def multi_channel_resnet50(num_classes=10, color_input_channels=3, brightness_input_channels=1, **kwargs):
    """
    Creates a ResNet-50 variant of the MultiChannelResNetNetwork
    """
    return MultiChannelResNetNetwork(
        num_classes=num_classes,
        color_input_channels=color_input_channels,
        brightness_input_channels=brightness_input_channels,
        num_blocks=[3, 4, 6, 3],
        block_type='bottleneck',
        **kwargs
    )

def multi_channel_resnet101(num_classes=10, color_input_channels=3, brightness_input_channels=1, **kwargs):
    """
    Creates a ResNet-101 variant of the MultiChannelResNetNetwork
    """
    return MultiChannelResNetNetwork(
        num_classes=num_classes,
        color_input_channels=color_input_channels,
        brightness_input_channels=brightness_input_channels,
        num_blocks=[3, 4, 23, 3],
        block_type='bottleneck',
        **kwargs
    )

def multi_channel_resnet152(num_classes=10, color_input_channels=3, brightness_input_channels=1, **kwargs):
    """
    Creates a ResNet-152 variant of the MultiChannelResNetNetwork
    """
    return MultiChannelResNetNetwork(
        num_classes=num_classes,
        color_input_channels=color_input_channels,
        brightness_input_channels=brightness_input_channels,
        num_blocks=[3, 8, 36, 3],
        block_type='bottleneck',
        **kwargs
    )


