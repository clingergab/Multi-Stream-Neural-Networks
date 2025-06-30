"""
Multi-Channel ResNet implementation following standard ResNet architecture.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import GradScaler, autocast
import numpy as np
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
from ..base import BaseMultiStreamModel
from ..layers.resnet_blocks import (
    MultiChannelResNetBasicBlock, 
    MultiChannelResNetBottleneck,
    MultiChannelDownsample,
    MultiChannelSequential
)
from ..layers.conv_layers import (
    MultiChannelConv2d, 
    MultiChannelBatchNorm2d, 
    MultiChannelActivation,
    MultiChannelAdaptiveAvgPool2d
)
from ...utils.device_utils import DeviceManager


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
        
        # Debug helpers - these will be filled during training if issues occur
        self.debug_mode = False
        self.gradient_norms = []
        self.feature_norms = []
    
    def _build_network(self):
        """Build the ResNet network with optimized architecture for different input channels."""
        # Initial layers (stem) - OPTIMIZED: use different input channels!
        self.conv1 = MultiChannelConv2d(
            color_in_channels=self.color_input_channels,        # 3 for RGB
            brightness_in_channels=self.brightness_input_channels,  # 1 for brightness - efficient!
            out_channels=64, 
            kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = MultiChannelBatchNorm2d(64)
        self.activation_initial = MultiChannelActivation(self.activation, inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers with proper channel progression
        self.layer1 = self._make_layer(64, self.num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, self.num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, self.num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, self.num_blocks[3], stride=2)
        
        # Global average pooling and classifier
        self.avgpool = MultiChannelAdaptiveAvgPool2d((1, 1))
        
        # Final classifier - choose between shared fusion or separate classifiers
        final_features = 512 * self.block.expansion
        
        # Add dropout before classifier if specified
        if self.dropout > 0:
            self.dropout_layer = nn.Dropout(self.dropout)
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
        # Use a smaller momentum value for more stable updates
        for module in self.modules():
            if isinstance(module, MultiChannelBatchNorm2d):
                module.color_bn.momentum = 0.05  # Lower momentum for more stable updates
                module.brightness_bn.momentum = 0.05  # Lower momentum for more stable updates
    
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
        """Initialize network weights following ResNet conventions with improved stability."""
        for module in self.modules():
            if isinstance(module, MultiChannelConv2d):
                # Initialize both color and brightness conv weights with more stable initialization
                # Use fan_out mode for better gradient flow in CNNs (standard for ResNet)
                nn.init.kaiming_normal_(module.color_weight, mode='fan_out', nonlinearity='relu')
                nn.init.kaiming_normal_(module.brightness_weight, mode='fan_out', nonlinearity='relu')
                # Zero out bias if present (for stability)
                if module.color_bias is not None:
                    nn.init.zeros_(module.color_bias)
                if module.brightness_bias is not None:
                    nn.init.zeros_(module.brightness_bias)
            elif isinstance(module, MultiChannelBatchNorm2d):
                # Initialize both color and brightness batch norm with slightly higher initial values
                # for better gradient flow in early training
                nn.init.constant_(module.color_bn.weight, 1.1)
                nn.init.constant_(module.color_bn.bias, 0.0)
                nn.init.constant_(module.brightness_bn.weight, 1.1)
                nn.init.constant_(module.brightness_bn.bias, 0.0)
            elif isinstance(module, nn.Linear):
                # Use Xavier initialization for fully connected layers
                # This works better for the final classification layers
                nn.init.xavier_normal_(module.weight)
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
        
        # ResNet layers
        color_x, brightness_x = self.layer1(color_x, brightness_x)
        color_x, brightness_x = self.layer2(color_x, brightness_x)
        color_x, brightness_x = self.layer3(color_x, brightness_x)
        color_x, brightness_x = self.layer4(color_x, brightness_x)
        
        # Global average pooling
        color_x, brightness_x = self.avgpool(color_x, brightness_x)
        
        # Flatten features
        color_x = torch.flatten(color_x, 1)
        brightness_x = torch.flatten(brightness_x, 1)
        
        # Check for NaN features after processing
        if torch.isnan(color_x).any() or torch.isnan(brightness_x).any():
            # Enable feature map recovery if NaNs are detected
            # This helps prevent model from getting stuck
            color_x = torch.where(torch.isnan(color_x), torch.zeros_like(color_x), color_x)
            brightness_x = torch.where(torch.isnan(brightness_x), torch.zeros_like(brightness_x), brightness_x)
        
        # Apply dropout if enabled
        if self.dropout_layer is not None:
            color_x = self.dropout_layer(color_x)
            brightness_x = self.dropout_layer(brightness_x)
        
        # Combine outputs for classification
        if self.use_shared_classifier:
            # Use shared classifier for optimal fusion
            fused_features = torch.cat([color_x, brightness_x], dim=1)
            logits = self.shared_classifier(fused_features)
            
            # Final check for NaN outputs
            if torch.isnan(logits).any():
                logits = torch.zeros_like(logits)
                print("Warning: NaN detected in model output. Using zero outputs.")
                
            return logits
        else:
            # Legacy: Add separate outputs
            color_logits = self.color_classifier(color_x)
            brightness_logits = self.brightness_classifier(brightness_x)
            
            # Final check for NaN outputs
            combined_logits = color_logits + brightness_logits
            if torch.isnan(combined_logits).any():
                combined_logits = torch.zeros_like(combined_logits)
                print("Warning: NaN detected in model output. Using zero outputs.")
                
            return combined_logits
    
    def analyze_pathways(self, color_input: torch.Tensor, brightness_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Analyze individual pathway contributions for research purposes.
        
        Returns separate outputs for each stream to analyze individual pathway contributions.
        Use this method only for research, visualization, and pathway analysis.
        
        Args:
            color_input: Color image tensor [batch_size, channels, height, width]
            brightness_input: Brightness image tensor [batch_size, channels, height, width]
            
        Returns:
            Tuple of (color_logits, brightness_logits) [batch_size, num_classes] each
            Separate outputs for analyzing individual pathway performance
        """
        # Apply the same normalization as in forward() for consistency
        if color_input.max() > 1.0 and color_input.min() >= 0.0:
            color_input = color_input / 255.0
            
        # Apply CIFAR-100 mean/std normalization for RGB channels
        if color_input.shape[1] == 3:
            mean = torch.tensor([0.5071, 0.4867, 0.4408], device=color_input.device).view(1, 3, 1, 1)
            std = torch.tensor([0.2675, 0.2565, 0.2761], device=color_input.device).view(1, 3, 1, 1)
            color_input = (color_input - mean) / std
            
        # Normalize brightness channel
        if brightness_input.max() > 1.0 and brightness_input.min() >= 0.0:
            brightness_input = brightness_input / 255.0
            
        # Standardize brightness
        if brightness_input.shape[1] == 1:
            batch_mean = brightness_input.mean(dim=(2, 3), keepdim=True)
            batch_std = brightness_input.std(dim=(2, 3), keepdim=True) + 1e-6
            brightness_input = (brightness_input - batch_mean) / batch_std
            
        # Initial layers
        color_x, brightness_x = self.conv1(color_input, brightness_input)
        color_x, brightness_x = self.bn1(color_x, brightness_x)
        color_x, brightness_x = self.activation_initial(color_x, brightness_x)
        
        # Apply maxpool to both streams
        color_x = self.maxpool(color_x)
        brightness_x = self.maxpool(brightness_x)
        
        # ResNet layers
        color_x, brightness_x = self.layer1(color_x, brightness_x)
        color_x, brightness_x = self.layer2(color_x, brightness_x)
        color_x, brightness_x = self.layer3(color_x, brightness_x)
        color_x, brightness_x = self.layer4(color_x, brightness_x)
        
        # Global average pooling
        color_x, brightness_x = self.avgpool(color_x, brightness_x)
        
        # Flatten features
        color_x = torch.flatten(color_x, 1)
        brightness_x = torch.flatten(brightness_x, 1)
        
        # Apply dropout if enabled (for consistency)
        if self.dropout_layer is not None:
            color_x = self.dropout_layer(color_x)
            brightness_x = self.dropout_layer(brightness_x)
        
        # Apply projection/classifier heads to get separate outputs
        if self.use_shared_classifier:
            # Use separate heads for research analysis
            color_logits = self.color_head(color_x)
            brightness_logits = self.brightness_head(brightness_x)
        else:
            # Legacy: Use separate classifiers
            color_logits = self.color_classifier(color_x)
            brightness_logits = self.brightness_classifier(brightness_x)
        
        return color_logits, brightness_logits
    
    def extract_features(self, color_input: torch.Tensor, brightness_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features before final classification."""
        # Apply the same normalization as in forward() for consistency
        if color_input.max() > 1.0 and color_input.min() >= 0.0:
            color_input = color_input / 255.0
            
        # Apply CIFAR-100 mean/std normalization for RGB channels
        if color_input.shape[1] == 3:
            mean = torch.tensor([0.5071, 0.4867, 0.4408], device=color_input.device).view(1, 3, 1, 1)
            std = torch.tensor([0.2675, 0.2565, 0.2761], device=color_input.device).view(1, 3, 1, 1)
            color_input = (color_input - mean) / std
            
        # Normalize brightness channel
        if brightness_input.max() > 1.0 and brightness_input.min() >= 0.0:
            brightness_input = brightness_input / 255.0
            
        # Standardize brightness
        if brightness_input.shape[1] == 1:
            batch_mean = brightness_input.mean(dim=(2, 3), keepdim=True)
            batch_std = brightness_input.std(dim=(2, 3), keepdim=True) + 1e-6
            brightness_input = (brightness_input - batch_mean) / batch_std
            
        # Process through all layers except classifier
        color_x, brightness_x = self.conv1(color_input, brightness_input)
        color_x, brightness_x = self.bn1(color_x, brightness_x)
        color_x, brightness_x = self.activation_initial(color_x, brightness_x)
        
        color_x = self.maxpool(color_x)
        brightness_x = self.maxpool(brightness_x)
        
        color_x, brightness_x = self.layer1(color_x, brightness_x)
        color_x, brightness_x = self.layer2(color_x, brightness_x)
        color_x, brightness_x = self.layer3(color_x, brightness_x)
        color_x, brightness_x = self.layer4(color_x, brightness_x)
        
        color_x, brightness_x = self.avgpool(color_x, brightness_x)
        
        color_x = torch.flatten(color_x, 1)
        brightness_x = torch.flatten(brightness_x, 1)
        
        return color_x, brightness_x
    
    def get_pathway_importance(self) -> Dict[str, float]:
        """Calculate pathway importance based on classifier weights."""
        with torch.no_grad():
            if self.use_shared_classifier:
                # For shared classifier, analyze both the shared weights and separate heads
                shared_weights = self.shared_classifier.weight.data
                feature_size = shared_weights.shape[1] // 2
                color_shared_weights = shared_weights[:, :feature_size]
                brightness_shared_weights = shared_weights[:, feature_size:]
                
                # Also analyze separate heads
                color_head_weights = self.color_head.weight.data
                brightness_head_weights = self.brightness_head.weight.data
                
                color_norm = torch.norm(color_shared_weights).item() + torch.norm(color_head_weights).item()
                brightness_norm = torch.norm(brightness_shared_weights).item() + torch.norm(brightness_head_weights).item()
                total_norm = color_norm + brightness_norm + 1e-8
                
                return {
                    'color_pathway': color_norm / total_norm,
                    'brightness_pathway': brightness_norm / total_norm,
                    'pathway_ratio': color_norm / (brightness_norm + 1e-8),
                    'fusion_type': 'shared_with_separate_heads'
                }
            else:
                # For separate classifiers, analyze each pathway
                color_weight_norm = torch.norm(self.color_classifier.weight.data).item()
                brightness_weight_norm = torch.norm(self.brightness_classifier.weight.data).item()
                total_norm = color_weight_norm + brightness_weight_norm + 1e-8
                
                return {
                    'color_pathway': color_weight_norm / total_norm,
                    'brightness_pathway': brightness_weight_norm / total_norm,
                    'pathway_ratio': color_weight_norm / (brightness_weight_norm + 1e-8),
                    'fusion_type': 'separate_classifiers'
                }

    def fit(self, train_color_data=None, train_brightness_data=None, train_labels=None,
            val_color_data=None, val_brightness_data=None, val_labels=None,
            train_loader=None, val_loader=None, batch_size=None, epochs=10,
            learning_rate=None, weight_decay=0.0001, early_stopping_patience=5,
            scheduler_type='cosine', min_lr=1e-6, verbose=1, 
            num_workers=None, pin_memory=None, max_grad_norm=1.0) -> Dict[str, List[float]]:
        """
        Unified fit method that handles both direct data arrays and DataLoaders with on-the-fly augmentation.
        
        This enhanced method supports two modes of operation:
        1. Direct data array input (original behavior): Provide train_color_data, train_brightness_data, train_labels
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
            learning_rate: Learning rate for the optimizer (defaults to 0.001 if None)
            weight_decay: Weight decay (L2 regularization)
            early_stopping_patience: Patience for early stopping
            scheduler_type: Learning rate scheduler type ('cosine', 'step', 'none')
            min_lr: Minimum learning rate for cosine annealing
            verbose: Verbosity level (0: silent, 1: progress bar, 2: one line per epoch)
            num_workers: Number of workers for data loading (auto-detected if None)
            pin_memory: Whether to pin memory for faster GPU transfer (auto-detected if None)
            max_grad_norm: Maximum gradient norm for gradient clipping (helps prevent exploding gradients)
            
        Returns:
            history: Dictionary containing training and validation metrics
        """
        # Enable debug mode for issue investigation
        self.debug_mode = True
        self.gradient_norms = []
        self.feature_norms = []
        
        # Use a default learning rate if not provided
        if learning_rate is None:
            learning_rate = 0.001  # Standard default learning rate
            if verbose > 0:
                print(f"Using default learning rate: {learning_rate}")
        
        # Initialize history dictionary to track metrics
        history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'gradient_norm': []  # Track gradient norms for debugging
        }
        
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
        
        if verbose > 0:
            print(f"🚀 Training {self.__class__.__name__} with {'DataLoader pipeline' if using_dataloaders else 'direct data'}:")
            print(f"   Device: {self.device}")
            if using_dataloaders:
                print(f"   Train batches: {len(train_loader)}")
                if val_loader:
                    print(f"   Val batches: {len(val_loader)}")
            else:
                print(f"   Batch size: {batch_size}")
            print(f"   Mixed precision: {self.use_mixed_precision}")
            print(f"   Workers: {num_workers}")
            print(f"   Pin memory: {pin_memory}")
            print(f"   Gradient clipping: {max_grad_norm}")
            print(f"   Weight decay: {weight_decay}")
            print(f"   Learning rate: {learning_rate}")
        
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
        
        # Optimizer and loss function - use AdamW with more stable defaults
        # Similar to what BaseMultiChannelNetwork uses successfully
        optimizer = optim.AdamW(
            self.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay,
            betas=(0.9, 0.999),  # Standard AdamW betas
            eps=1e-8  # More stable epsilon value
        )
        criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler setup based on scheduler type
        if scheduler_type.lower() == 'onecycle':
            # OneCycleLR is often more effective than standard schedulers
            # It implements a cyclical learning rate policy with warmup
            steps_per_epoch = len(train_loader)
            total_steps = steps_per_epoch * epochs
            
            # Use OneCycleLR with sensible defaults
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=learning_rate,
                total_steps=total_steps,
                pct_start=0.3,  # 30% of training for warmup
                div_factor=25,  # LR starts at max_lr/25
                final_div_factor=1000,  # Final LR is max_lr/25000
                anneal_strategy='cos'  # Cosine annealing
            )
            # This scheduler needs to step after each batch
            scheduler_step_batch = True
        elif scheduler_type.lower() == 'cosine':
            # Standard cosine annealing
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=epochs, 
                eta_min=min_lr
            )
            scheduler_step_batch = False
        elif scheduler_type.lower() == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
            scheduler_step_batch = False
        elif scheduler_type.lower() == 'reduce_on_plateau':
            # Add ReduceLROnPlateau option which can be more stable
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=2, 
                min_lr=min_lr, verbose=verbose > 0
            )
            scheduler_step_batch = False
        else:
            scheduler = None
            scheduler_step_batch = False
        
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
            
            # Training phase
            self.train()
            total_loss = 0.0
            train_correct = 0
            train_total = 0
            epoch_grad_norm = 0.0
            num_batches = 0
            
            # Create progress bar for training
            epoch_pbar = None
            if verbose == 1:
                epoch_pbar = tqdm(
                    total=len(train_loader), 
                    desc=f"Epoch {epoch+1}/{epochs}", 
                    leave=True
                )
            
            # Training loop
            for batch_idx, data in enumerate(train_loader):
                # Unpack data - handle different formats
                if isinstance(data, (list, tuple)) and len(data) == 3:
                    batch_color, batch_brightness, batch_labels = data
                else:
                    raise ValueError("DataLoader must provide (color, brightness, labels) tuples")
                
                # Move data to device if not already there
                if batch_color.device != self.device:
                    batch_color = batch_color.to(self.device, non_blocking=True)
                if batch_brightness.device != self.device:
                    batch_brightness = batch_brightness.to(self.device, non_blocking=True)
                if batch_labels.device != self.device:
                    batch_labels = batch_labels.to(self.device, non_blocking=True)
                
                # Check for NaN in inputs
                if torch.isnan(batch_color).any() or torch.isnan(batch_brightness).any():
                    print(f"Warning: NaN detected in batch {batch_idx} inputs. Skipping batch.")
                    continue
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass with mixed precision if available
                if self.use_mixed_precision and self.scaler is not None:
                    with autocast("cuda" if torch.cuda.is_available() else "cpu"):
                        outputs = self(batch_color, batch_brightness)
                        loss = criterion(outputs, batch_labels)
                    
                    # Backward pass with scaler
                    self.scaler.scale(loss).backward()
                    
                    # Unscale gradients once before gradient clipping and debugging
                    self.scaler.unscale_(optimizer)
                    
                    # Track gradient norm for debugging (now safely unscaled)
                    if self.debug_mode:
                        with torch.no_grad():
                            # Calculate gradient norm manually to avoid potential recursion issues
                            total_norm = 0.0
                            parameters = [p for p in self.parameters() if p.grad is not None]
                            for p in parameters:
                                param_norm = p.grad.detach().data.norm(2)
                                total_norm += param_norm.item() ** 2
                            grad_norm = total_norm ** 0.5
                            epoch_grad_norm += grad_norm
                            num_batches += 1
                    
                    # Apply gradient clipping (on unscaled gradients)
                    if max_grad_norm > 0:
                        # Direct implementation to avoid recursion issues
                        total_norm = 0
                        parameters = [p for p in self.parameters() if p.grad is not None]
                        for p in parameters:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                        total_norm = total_norm ** (1. / 2)
                        clip_coef = max_grad_norm / (total_norm + 1e-6)
                        if clip_coef < 1:
                            for p in parameters:
                                p.grad.data.mul_(clip_coef)
                    
                    # Step optimizer with scaler
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    
                    # Step OneCycleLR scheduler after each batch if needed
                    if scheduler is not None and scheduler_step_batch:
                        scheduler.step()
                else:
                    # Standard forward and backward pass
                    outputs = self(batch_color, batch_brightness)
                    loss = criterion(outputs, batch_labels)
                    loss.backward()
                    
                    # Track gradient norm for debugging in a more stable way
                    if self.debug_mode:
                        with torch.no_grad():
                            # Calculate gradient norm manually to avoid potential recursion issues
                            total_norm = 0.0
                            parameters = [p for p in self.parameters() if p.grad is not None]
                            for p in parameters:
                                param_norm = p.grad.detach().data.norm(2)
                                total_norm += param_norm.item() ** 2
                            grad_norm = total_norm ** 0.5
                            epoch_grad_norm += grad_norm
                            num_batches += 1
                    
                    # Apply gradient clipping
                    if max_grad_norm > 0:
                        # Direct implementation to avoid recursion issues
                        total_norm = 0
                        parameters = [p for p in self.parameters() if p.grad is not None]
                        for p in parameters:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                        total_norm = total_norm ** (1. / 2)
                        clip_coef = max_grad_norm / (total_norm + 1e-6)
                        if clip_coef < 1:
                            for p in parameters:
                                p.grad.data.mul_(clip_coef)
                    
                    # Step optimizer
                    optimizer.step()
                
                # Step OneCycleLR scheduler after each batch if needed
                if scheduler is not None and scheduler_step_batch:
                    scheduler.step()
                
                # Track metrics
                total_loss += loss.item()
                
                # Calculate training accuracy
                with torch.no_grad():  # Ensure no unnecessary memory is retained for gradient calculation
                    _, predicted = torch.max(outputs.data, 1)
                    train_total += batch_labels.size(0)
                    train_correct += (predicted == batch_labels).sum().item()
                
                # Explicitly clear unnecessary tensors to help with memory management
                del outputs, loss, predicted
                if self.use_mixed_precision and self.scaler is not None:
                    torch.cuda.empty_cache()  # Periodic cache clearing for large datasets
                
                # Update progress bar
                if verbose == 1 and epoch_pbar is not None:
                    epoch_pbar.set_postfix({
                        'loss': total_loss / (batch_idx + 1), 
                        'acc': 100. * train_correct / train_total
                    })
                    epoch_pbar.update(1)
            
            # Calculate average gradient norm for this epoch
            avg_grad_norm = epoch_grad_norm / max(num_batches, 1)
            history['gradient_norm'].append(avg_grad_norm)
            
            # Update learning rate scheduler at epoch level for non-batch schedulers
            if scheduler is not None and not scheduler_step_batch and scheduler_type.lower() != 'reduce_on_plateau':
                scheduler.step()
            
            # Calculate epoch metrics
            avg_train_loss = total_loss / len(train_loader)
            train_accuracy = train_correct / train_total
            
            # Store training metrics in history
            history['loss'].append(avg_train_loss)
            history['accuracy'].append(train_accuracy)
            
            # Validation phase
            if val_loader is not None:
                self.eval()
                total_val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch_idx, data in enumerate(val_loader):
                        # Unpack data - handle different formats
                        if isinstance(data, (list, tuple)) and len(data) == 3:
                            batch_color, batch_brightness, batch_labels = data
                        else:
                            raise ValueError("Val DataLoader must provide (color, brightness, labels) tuples")
                        
                        # Move data to device if not already there
                        if batch_color.device != self.device:
                            batch_color = batch_color.to(self.device, non_blocking=True)
                        if batch_brightness.device != self.device:
                            batch_brightness = batch_brightness.to(self.device, non_blocking=True)
                        if batch_labels.device != self.device:
                            batch_labels = batch_labels.to(self.device, non_blocking=True)
                        
                        # Forward pass
                        if self.use_mixed_precision and self.scaler is not None:
                            with autocast("cuda" if torch.cuda.is_available() else "cpu"):
                                outputs = self(batch_color, batch_brightness)
                                loss = criterion(outputs, batch_labels)
                        else:
                            outputs = self(batch_color, batch_brightness)
                            loss = criterion(outputs, batch_labels)
                        
                        # Track metrics
                        total_val_loss += loss.item()
                        
                        # Calculate validation accuracy
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += batch_labels.size(0)
                        val_correct += (predicted == batch_labels).sum().item()
                        
                        # Explicitly clear unnecessary tensors to help with memory management
                        del outputs, loss, predicted
                        
                        # Periodically clear cache during validation
                        if batch_idx % 20 == 0 and self.device.type == 'cuda':
                            torch.cuda.empty_cache()
                
                # Calculate epoch validation metrics
                avg_val_loss = total_val_loss / len(val_loader)
                val_accuracy = val_correct / val_total
                
                # Store validation metrics in history
                history['val_loss'].append(avg_val_loss)
                history['val_accuracy'].append(val_accuracy)
                
                # Update scheduler if using ReduceLROnPlateau
                if scheduler is not None and scheduler_type.lower() == 'reduce_on_plateau':
                    scheduler.step(avg_val_loss)
                
                # Prepare summary for the final epoch progress bar update
                summary = {
                    'Loss': f'{avg_train_loss:.4f}',
                    'Acc': f'{train_accuracy:.4f}',
                    'Val_Loss': f'{avg_val_loss:.4f}',
                    'Val_Acc': f'{val_accuracy:.4f}',
                    'LR': f'{optimizer.param_groups[0]["lr"]:.6f}',
                    'GradNorm': f'{avg_grad_norm:.2f}'
                }
                
                # Update progress bar with all metrics in one line
                if verbose == 1 and epoch_pbar is not None:
                    epoch_pbar.set_postfix(summary)
                    epoch_pbar.refresh()
                    epoch_pbar.close()
                    print("\r\033[K", end="")  # Clear the current line
                elif verbose > 0:
                    # If no progress bar but verbose, print a summary line
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Loss: {avg_train_loss:.4f}, Acc: {train_accuracy:.4f}, "
                          f"Val_Loss: {avg_val_loss:.4f}, Val_Acc: f{val_accuracy:.4f}, "
                          f"LR: {optimizer.param_groups[0]['lr']:.6f}, "
                          f"GradNorm: {avg_grad_norm:.2f}")
            else:
                # Prepare summary for training-only progress bar update
                summary = {
                    'Loss': f'{avg_train_loss:.4f}',
                    'Acc': f'{train_accuracy:.4f}',
                    'LR': f'{optimizer.param_groups[0]["lr"]:.6f}',
                    'GradNorm': f'{avg_grad_norm:.2f}'
                }
                
                # Update progress bar with training metrics in one line
                if verbose == 1 and epoch_pbar is not None:
                    epoch_pbar.set_postfix(summary)
                    epoch_pbar.refresh()
                    epoch_pbar.close()
                    print("\r\033[K", end="")  # Clear the current line
                elif verbose > 0:
                    # If no progress bar but verbose, print a summary line
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Loss: {avg_train_loss:.4f}, Acc: {train_accuracy:.4f}, "
                          f"LR: {optimizer.param_groups[0]['lr']:.6f}, "
                          f"GradNorm: {avg_grad_norm:.2f}")
                
                # Store empty validation metrics for consistency
                history['val_loss'].append(None)
                history['val_accuracy'].append(None)
                avg_val_loss = float('inf')  # For early stopping logic
            
            # Early stopping check (only if validation data provided)
            if val_loader is not None:
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    # Save the best model
                    self.best_model_state = self.state_dict().copy()
                    if verbose > 0:
                        print(f"✅ New best validation loss: {best_val_loss:.4f}")
                else:
                    patience_counter += 1
                    if verbose > 0:
                        print(f"⏳ No improvement for {patience_counter}/{early_stopping_patience} epochs")
                
                if patience_counter >= early_stopping_patience:
                    if verbose > 0:
                        print(f"🛑 Early stopping triggered. Stopping training at epoch {epoch + 1}.")
                    break
        
        # Load best model if validation was performed and we have saved a best state
        if val_loader is not None and hasattr(self, 'best_model_state'):
            self.load_state_dict(self.best_model_state)
            if verbose > 0:
                print("📊 Loaded best model state from early stopping")
        
        # Clear cache after training
        self.device_manager.clear_cache()
        
        # Return the history dictionary with training metrics
        return history


# Factory functions following ResNet naming conventions
def multi_channel_resnet18(num_classes: int = 10, **kwargs) -> MultiChannelResNetNetwork:
    """Create a Multi-Channel ResNet-18."""
    return MultiChannelResNetNetwork(
        num_classes=num_classes,
        num_blocks=[2, 2, 2, 2],
        block_type='basic',
        **kwargs
    )


def multi_channel_resnet34(num_classes: int = 10, **kwargs) -> MultiChannelResNetNetwork:
    """Create a Multi-Channel ResNet-34."""
    return MultiChannelResNetNetwork(
        num_classes=num_classes,
        num_blocks=[3, 4, 6, 3],
        block_type='basic',
        **kwargs
    )


def multi_channel_resnet50(num_classes: int = 10, **kwargs) -> MultiChannelResNetNetwork:
    """Create a Multi-Channel ResNet-50."""
    return MultiChannelResNetNetwork(
        num_classes=num_classes,
        num_blocks=[3, 4, 6, 3],
        block_type='bottleneck',
        **kwargs
    )


def multi_channel_resnet101(num_classes: int = 10, **kwargs) -> MultiChannelResNetNetwork:
    """Create a Multi-Channel ResNet-101."""
    return MultiChannelResNetNetwork(
        num_classes=num_classes,
        num_blocks=[3, 4, 23, 3],
        block_type='bottleneck',
        **kwargs
    )


def multi_channel_resnet152(num_classes: int = 10, **kwargs) -> MultiChannelResNetNetwork:
    """Create a Multi-Channel ResNet-152."""
    return MultiChannelResNetNetwork(
        num_classes=num_classes,
        num_blocks=[3, 8, 36, 3],
        block_type='bottleneck',
        **kwargs
    )


