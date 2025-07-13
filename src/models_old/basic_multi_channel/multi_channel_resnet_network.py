"""
Multi-Channel ResNet implementation following standard ResNet architecture.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import time
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast

# Smart tqdm import - detect notebook environment
def _get_tqdm():
    """Get the appropriate tqdm based on environment."""
    try:
        # Check if we're in a Jupyter notebook
        from IPython import get_ipython
        if get_ipython() is not None:
            # We're in IPython/Jupyter, check if it's a notebook
            ipython = get_ipython()
            if hasattr(ipython, 'kernel'):
                # We're in a Jupyter notebook, use notebook tqdm
                from tqdm.notebook import tqdm
                return tqdm
    except ImportError:
        pass
    
    # Fallback to regular tqdm
    from tqdm import tqdm
    return tqdm

# Get the appropriate tqdm for our environment
tqdm = _get_tqdm()

from typing import Dict, List, Any, Union
from models_old.base import BaseMultiStreamModel
from utils.grad_utils import safe_clip_grad_norm
from models_old.layers.conv_layers import (
    MultiChannelConv2d,
    MultiChannelBatchNorm2d,
    MultiChannelActivation,
    MultiChannelAdaptiveAvgPool2d
)
from models_old.layers.resnet_blocks import (
    MultiChannelResNetBasicBlock,
    MultiChannelResNetBottleneck,
    MultiChannelDownsample,
    MultiChannelSequential
)

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
    
    # Compilation defaults optimized for CNN/ResNet architectures
    DEFAULT_COMPILE_CONFIG = {
        'learning_rate': 0.0003,   # Lower learning rate for stable CNN training
        'weight_decay': 1e-4,      # Regularization important for CNNs
        'gradient_clip': 1.0,      # Enabled for deep network stability
        'scheduler': 'cosine',     # Cosine annealing works well for ResNets
        'early_stopping_patience': 5  # CNNs typically converge faster
    }
    
    def __init__(
        self,
        num_classes: int,
        color_input_channels: int = 3,
        brightness_input_channels: int = 1,
        num_blocks: List[int] = [2, 2, 2, 2],
        block_type: str = 'basic',
        activation: str = 'relu',
        dropout: float = 0.0,  
        device: str = 'auto',
        reduce_architecture: bool = False,  # Set to True for small datasets like CIFAR
        **kwargs
    ):
        
        
        # Initialize base class with common parameters
        super().__init__(
            num_classes=num_classes,
            activation=activation,
            dropout=dropout,
            device=device,
            **kwargs
        )
        
        self.color_input_channels = color_input_channels
        self.brightness_input_channels = brightness_input_channels
        self.num_blocks = num_blocks
        self.block_type = block_type
        self.reduce_architecture = reduce_architecture
        
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
        
        # Finalize initialization (device setup, etc.)
        self._finalize_initialization()
    
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
        
        # Shared classifier with proper fusion - concatenates features from both streams
        self.shared_classifier = nn.Linear(final_features * 2, self.num_classes, bias=True)
        # Also create separate projection heads for research/analysis purposes
        self.color_head = nn.Linear(final_features, self.num_classes, bias=True)
        self.brightness_head = nn.Linear(final_features, self.num_classes, bias=True)
            
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
        first_block = self.block(
            self.inplanes, 
            planes, 
            stride, 
            downsample,
            self.activation
        )
        layers.append(first_block)
        
        # Update inplanes for subsequent blocks
        self.inplanes = planes * self.block.expansion
        
        # Remaining blocks
        for _ in range(1, blocks):
            block = self.block(
                self.inplanes,
                planes,
                activation=self.activation
            )
            layers.append(block)
        
        # Create MultiChannelSequential
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
                # Initialize BatchNorm with standard weights (1.0) and zero bias
                # This follows standard BatchNorm initialization practices
                nn.init.constant_(module.color_bn.weight, 1.0)
                nn.init.constant_(module.color_bn.bias, 0.0)
                nn.init.constant_(module.brightness_bn.weight, 1.0)
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
        # Basic input normalization (transforms should handle detailed normalization)
        if color_input.max() > 1.0:
            color_input = color_input / 255.0
        if brightness_input.max() > 1.0:
            brightness_input = brightness_input / 255.0
            
        # Basic input validation
        if color_input.dim() != 4 or brightness_input.dim() != 4:
            raise ValueError(f"Expected 4D tensors (batch_size, channels, height, width), got {color_input.shape} and {brightness_input.shape}")
            
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
        
        # Apply dropout if enabled
        if self.dropout_layer is not None:
            color_x = self.dropout_layer(color_x)
            brightness_x = self.dropout_layer(brightness_x)
        
        # Use shared classifier for optimal fusion
        fused_features = torch.cat([color_x, brightness_x], dim=1)
        logits = self.shared_classifier(fused_features)
        return logits
    
    def _forward_color_pathway(self, color_input):
        """
        Forward pass through the color pathway only.
        
        Args:
            color_input: Color image tensor [batch_size, channels, height, width]
            
        Returns:
            Color pathway features [batch_size, features]
        """
        # Basic normalization
        if color_input.max() > 1.0:
            color_input = color_input / 255.0
            
        # Initial layers - only process color pathway
        color_x = self.conv1.forward_color(color_input)
        color_x = self.bn1.forward_color(color_x)
        color_x = self.activation_initial.forward_single(color_x)
        
        # Apply maxpool
        color_x = self.maxpool(color_x)
        
        # ResNet layers
        color_x = self.layer1.forward_color(color_x)
        color_x = self.layer2.forward_color(color_x)
        color_x = self.layer3.forward_color(color_x)
        color_x = self.layer4.forward_color(color_x)
        
        # Global average pooling
        color_x = self.avgpool.forward_single(color_x)
        
        # Flatten features
        color_x = torch.flatten(color_x, 1)
        
        # Apply dropout if enabled
        if self.dropout_layer is not None:
            color_x = self.dropout_layer(color_x)
        
        return color_x
        
    def _forward_brightness_pathway(self, brightness_input):
        """
        Forward pass through the brightness pathway only.
        
        Args:
            brightness_input: Brightness image tensor [batch_size, channels, height, width]
            
        Returns:
            Brightness pathway features [batch_size, features]
        """
        # Basic normalization
        if brightness_input.max() > 1.0:
            brightness_input = brightness_input / 255.0
        
        # Initial layers - only process brightness pathway
        brightness_x = self.conv1.forward_brightness(brightness_input)
        brightness_x = self.bn1.forward_brightness(brightness_x)
        brightness_x = self.activation_initial.forward_single(brightness_x)
        
        # Apply maxpool
        brightness_x = self.maxpool(brightness_x)
        
        # ResNet layers
        brightness_x = self.layer1.forward_brightness(brightness_x)
        brightness_x = self.layer2.forward_brightness(brightness_x)
        brightness_x = self.layer3.forward_brightness(brightness_x)
        brightness_x = self.layer4.forward_brightness(brightness_x)
        
        # Global average pooling
        brightness_x = self.avgpool.forward_single(brightness_x)
        
        # Flatten features
        brightness_x = torch.flatten(brightness_x, 1)
        
        # Apply dropout if enabled
        if self.dropout_layer is not None:
            brightness_x = self.dropout_layer(brightness_x)
        
        return brightness_x
    
    def extract_features(self, color_input, brightness_input):
        """
        Extract concatenated features before final classification.
        
        This method returns fused features ready for external classifiers.
        For separate pathway features, use get_separate_features() instead.
        
        Args:
            color_input (torch.Tensor): Color input data
            brightness_input (torch.Tensor): Brightness input data
            
        Returns:
            torch.Tensor: Concatenated fused features [batch_size, feature_size * 2]
            Ready for external linear classifiers or further processing
        """
        # Get separate features and concatenate them
        color_features, brightness_features = self.get_separate_features(color_input, brightness_input)
        
        # Return concatenated features (fusion approach for external use)
        return torch.cat([color_features, brightness_features], dim=1)

    def get_separate_features(self, color_input, brightness_input):
        """
        Extract separate features from both pathways without classification.
        
        This method returns individual pathway features for research and analysis.
        For fused features ready for external classifiers, use extract_features() instead.
        
        Args:
            color_input (torch.Tensor): Color input data
            brightness_input (torch.Tensor): Brightness input data
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Features from color and brightness pathways
            Separate features for pathway analysis and research
        """
        # Move inputs to the correct device
        color_input = color_input.to(self.device)
        brightness_input = brightness_input.to(self.device)
        
        # Feature extraction
        color_features = self._forward_color_pathway(color_input)
        brightness_features = self._forward_brightness_pathway(brightness_input)
        
        return color_features, brightness_features
    
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
        # Apply the separate projection heads to each feature separately
        color_out = self.color_head(color_features)
        brightness_out = self.brightness_head(brightness_features)
        
        return color_out, brightness_out

    def analyze_pathway_weights(self) -> Dict[str, float]:
        """
        Analyze the relative importance of color vs brightness pathways.
        
        Returns:
            Dictionary with pathway weight statistics
        """
        # Analyze both the shared weights and separate heads
        shared_weights = self.shared_classifier.weight.data
        feature_size = shared_weights.shape[1] // 2
        color_shared_weights = shared_weights[:, :feature_size]
        brightness_shared_weights = shared_weights[:, feature_size:]
        
        # Also analyze separate heads
        color_head_weights = self.color_head.weight.data
        brightness_head_weights = self.brightness_head.weight.data
        
        # Calculate pathway norms
        color_norm = torch.norm(color_shared_weights).item() + torch.norm(color_head_weights).item()
        brightness_norm = torch.norm(brightness_shared_weights).item() + torch.norm(brightness_head_weights).item()
        total_norm = color_norm + brightness_norm + 1e-8
        
        # Calculate balance ratio (closer to 1.0 = more balanced)
        balance_ratio = min(color_norm, brightness_norm) / max(color_norm, brightness_norm) if max(color_norm, brightness_norm) > 0 else 1.0
        
        return {
            'color_pathway': color_norm / total_norm,
            'brightness_pathway': brightness_norm / total_norm,
            'pathway_ratio': color_norm / (brightness_norm + 1e-8),
            'balance_ratio': balance_ratio,  # This is what diagnostics look for
            'fusion_type': 'shared_with_separate_heads',
            'shared_color_norm': torch.norm(color_shared_weights).item(),
            'shared_brightness_norm': torch.norm(brightness_shared_weights).item(),
            'head_color_norm': torch.norm(color_head_weights).item(),
            'head_brightness_norm': torch.norm(brightness_head_weights).item(),
            'architecture_type': 'resnet'
        }

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
    
    @property
    def fusion_type(self) -> str:
        """Return the type of fusion used by this model."""
        return "shared_classifier"
    
    def get_classifier_info(self) -> Dict[str, Any]:
        """Get information about the classifier architecture."""
        shared_params = sum(p.numel() for p in self.shared_classifier.parameters())
        color_params = sum(p.numel() for p in self.color_head.parameters())
        brightness_params = sum(p.numel() for p in self.brightness_head.parameters())
        
        final_features = (512 if not self.reduce_architecture else 256) * self.block.expansion
        
        return {
            'type': 'shared_with_separate_heads',
            'shared_classifier_params': shared_params,
            'color_head_params': color_params, 
            'brightness_head_params': brightness_params,
            'total_params': shared_params + color_params + brightness_params,
            'shared_input_size': self.shared_classifier.in_features,
            'output_size': self.shared_classifier.out_features,
            'feature_size_per_stream': final_features,
            'architecture_type': 'reduced' if self.reduce_architecture else 'full'
        }
    
    def fit(
        self,
        train_color_data=None,
        train_brightness_data=None,
        train_labels=None,
        val_color_data=None,
        val_brightness_data=None,
        val_labels=None,
        train_loader=None,
        val_loader=None,
        batch_size=None,
        epochs: int = 10,
        verbose: int = 1,
        enable_diagnostics: bool = False,
        diagnostic_output_dir: str = "diagnostics"
    ) -> Dict[str, List[float]]:
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
            verbose: Verbosity level (0: silent, 1: progress bar, 2: one line per epoch)
            
        Returns:
            Dictionary with training history
        """
        if not self.is_compiled:
            raise RuntimeError("Model must be compiled before training. Call model.compile() first.")
        
        # Training history
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rates': []
        }
        
        # Initialize diagnostics if enabled
        if enable_diagnostics:
            if verbose > 0:
                print("🔍 Diagnostic mode enabled - comprehensive monitoring active")
            self._setup_diagnostic_hooks()
            
            # Add diagnostic metrics to history
            history.update({
                'gradient_norms': [],
                'weight_norms': [],
                'dead_neuron_counts': [],
                'pathway_balance': [],
                'epoch_times': []
            })
        
        # Determine if we're using DataLoaders or direct data
        using_dataloaders = train_loader is not None
        
        # Use reasonable default batch size if not specified
        if batch_size is None and not using_dataloaders:
            # For CNNs, smaller batch sizes work better for CIFAR-100
            # Based on empirical testing: batch_size=32 significantly outperforms larger sizes
            batch_size = 32
        
        # Get optimal DataLoader configuration from device manager (conservative for CNNs)
        dataloader_config = self.device_manager.get_dataloader_config(conservative=True)
        num_workers = dataloader_config['num_workers']
        pin_memory = dataloader_config['pin_memory']
            
        
        # Set up train and validation loaders based on input mode
        if not using_dataloaders:
            # Check that required data is provided when not using DataLoaders
            if train_color_data is None or train_brightness_data is None or train_labels is None:
                raise ValueError("When train_loader is not provided, you must provide train_color_data, train_brightness_data, and train_labels")
            
            # Create DataLoaders from direct data
            if isinstance(train_color_data, torch.Tensor):
                train_color_tensor = train_color_data.detach().clone()
            else:
                train_color_tensor = torch.tensor(train_color_data, dtype=torch.float32)
            if isinstance(train_brightness_data, torch.Tensor):
                train_brightness_tensor = train_brightness_data.detach().clone()
            else:
                train_brightness_tensor = torch.tensor(train_brightness_data, dtype=torch.float32)
            train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
            
            train_dataset = TensorDataset(train_color_tensor, train_brightness_tensor, train_labels_tensor)
            train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True,
                **dataloader_config
            )
            
            # Set up validation loader if validation data is provided
            if val_color_data is not None and val_brightness_data is not None and val_labels is not None:
                if isinstance(val_color_data, np.ndarray):
                    val_color_tensor = torch.from_numpy(val_color_data).float()
                elif isinstance(val_color_data, torch.Tensor):
                    val_color_tensor = val_color_data.detach().clone()
                else:
                    val_color_tensor = val_color_data
                    
                if isinstance(val_brightness_data, np.ndarray):
                    val_brightness_tensor = torch.from_numpy(val_brightness_data).float()
                elif isinstance(val_brightness_data, torch.Tensor):
                    val_brightness_tensor = val_brightness_data.detach().clone()
                else:
                    val_brightness_tensor = val_brightness_data
                    
                if isinstance(val_labels, np.ndarray):
                    val_labels_tensor = torch.from_numpy(val_labels).long()
                elif isinstance(val_labels, torch.Tensor):
                    val_labels_tensor = val_labels.detach().clone()
                else:
                    val_labels_tensor = val_labels
                
                val_dataset = TensorDataset(val_color_tensor, val_brightness_tensor, val_labels_tensor)
                val_loader = DataLoader(
                    val_dataset, 
                    batch_size=batch_size, 
                    shuffle=False,
                    **dataloader_config
                )
            
        # Update scheduler configuration if used
        if self.scheduler is not None:
            if self.scheduler_type == 'cosine':
                # Reconfigure cosine scheduler with actual epochs
                current_lr = self.optimizer.param_groups[0]['lr']
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=epochs, eta_min=self.min_lr
                )
            elif self.scheduler_type == 'onecycle':
                # Reconfigure OneCycle scheduler with actual steps
                # For OneCycle, we need a higher max_lr than the initial lr
                max_lr = 0.001  # A more appropriate max_lr for OneCycle with ResNet
                self.scheduler = optim.lr_scheduler.OneCycleLR(
                    self.optimizer, 
                    max_lr=max_lr,
                    steps_per_epoch=len(train_loader),
                    epochs=epochs,
                    pct_start=0.3
                )
        
        # Print training configuration (after all setup is complete)
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
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Create progress bar for this epoch with notebook-aware tqdm
            epoch_pbar = None
            if verbose == 1:
                # Detect if we're in a notebook and configure accordingly
                try:
                    from IPython import get_ipython
                    if get_ipython() is not None and hasattr(get_ipython(), 'kernel'):
                        # We're in a Jupyter notebook
                        epoch_pbar = tqdm(
                            total=len(train_loader), 
                            desc=f"Epoch {epoch+1}/{epochs}",
                            leave=False,
                            position=0,
                            dynamic_ncols=True
                        )
                    else:
                        # We're in IPython terminal or regular Python
                        epoch_pbar = tqdm(
                            total=len(train_loader), 
                            desc=f"Epoch {epoch+1}/{epochs}",
                            leave=False,
                            dynamic_ncols=True,
                            file=sys.stdout
                        )
                except ImportError:
                    # No IPython, use regular tqdm
                    epoch_pbar = tqdm(
                        total=len(train_loader), 
                        desc=f"Epoch {epoch+1}/{epochs}",
                        leave=False,
                        dynamic_ncols=True,
                        file=sys.stdout
                    )
                
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
                
                # Calculate training accuracy with memory efficiency
                with torch.no_grad():  # Ensure no unnecessary memory is retained for gradient calculation
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                
                # Explicitly clear unnecessary tensors to help with memory management
                del outputs, loss, predicted
                if self.use_mixed_precision and self.scaler is not None and batch_idx % 20 == 0:
                    torch.cuda.empty_cache()  # Periodic cache clearing for large datasets
                
                # Update progress bar with notebook-friendly approach
                if verbose == 1 and epoch_pbar is not None:
                    try:
                        current_lr = self.optimizer.param_groups[0]['lr']
                        postfix = {
                            'loss': f'{total_loss / (batch_idx + 1):.4f}',
                            'acc': f'{correct/total:.4f}',
                            'lr': f'{current_lr:.6f}'
                        }
                        epoch_pbar.set_postfix(postfix)
                        epoch_pbar.update(1)
                        
                        # Force display update for notebooks
                        try:
                            epoch_pbar.refresh()
                        except:
                            pass
                    except Exception:
                        # If progress bar fails, continue without it
                        pass
            
            # Close progress bar properly
            if verbose == 1 and epoch_pbar is not None:
                try:
                    epoch_pbar.close()
                except Exception:
                    pass
            
            # Step the scheduler (for epoch-level schedulers)
            if self.scheduler_type == 'cosine' and self.scheduler:
                self.scheduler.step()
                    
            # Calculate epoch-level metrics
            avg_train_loss = total_loss / len(train_loader)
            train_accuracy = correct / total
            current_lr = self.optimizer.param_groups[0]['lr']
                
            # Record training metrics
            history['train_loss'].append(avg_train_loss)
            history['train_accuracy'].append(train_accuracy)
            history['learning_rates'].append(batch_lr if batch_lr else [current_lr])
            
            # Collect diagnostic metrics if enabled
            if enable_diagnostics:
                # Get sample batch for dead neuron analysis
                sample_batch = None
                for batch_data in train_loader:
                    if isinstance(batch_data, (list, tuple)) and len(batch_data) == 3:
                        color_batch, brightness_batch, _ = batch_data
                        # Move to device
                        if color_batch.device != self.device:
                            color_batch = color_batch.to(self.device)
                        if brightness_batch.device != self.device:
                            brightness_batch = brightness_batch.to(self.device)
                        sample_batch = (color_batch, brightness_batch)
                        break
                
                # Collect comprehensive diagnostics
                diagnostics = self._collect_epoch_diagnostics(
                    epoch=epoch,
                    sample_batch=sample_batch,
                    epoch_time=0.0,  # Will be calculated later
                    current_lr=current_lr
                )
                
                # Store diagnostic metrics in history
                history['gradient_norms'].append(diagnostics['gradient_norm'])
                history['weight_norms'].append(diagnostics['weight_norm'])
                history['dead_neuron_counts'].append(diagnostics['dead_neuron_count'])
                history['pathway_balance'].append(diagnostics['pathway_balance'])
                history['epoch_times'].append(diagnostics['epoch_time'])
                
                if verbose > 0:
                    print(f"📊 Diagnostics - Grad: {diagnostics['gradient_norm']:.4f}, "
                          f"Weight: {diagnostics['weight_norm']:.4f}, "
                          f"Balance: {diagnostics['pathway_balance']:.4f}")
            
            # Validation phase
            if val_loader is not None:
                val_loss, val_accuracy = self._validate(val_loader)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_accuracy)
                
                # Collect validation diagnostics if enabled
                if enable_diagnostics:
                    val_diagnostics = self._collect_validation_diagnostics(val_loader, epoch)
                    
                    # Store validation diagnostic metrics
                    for key, value in val_diagnostics.items():
                        if key not in history:
                            history[key] = []
                        history[key].append(value)
                    
                    if verbose > 0:
                        print(f"📋 Val Diagnostics - Dead neurons: {val_diagnostics.get('dead_neurons_detected', 0)}, "
                              f"Avg confidence: {val_diagnostics.get('avg_confidence', 0):.4f}, "
                              f"Low confidence %: {val_diagnostics.get('low_confidence_percent', 0):.1f}%")
                
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
                        print(f"⏳ No improvement for {patience_counter}/{self.early_stopping_patience} epochs")
                    
                # Calculate epoch time
                epoch_time = time.time() - epoch_start_time
                
                # Print epoch summary
                if verbose > 0:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"loss: {avg_train_loss:.4f}, acc: {train_accuracy:.4f}, "
                          f"val_loss: {val_loss:.4f}, val_acc: {val_accuracy:.4f}, "
                          f"lr: {current_lr:.6f}, time: {epoch_time:.1f}s")
                    
                # Check for early stopping
                if patience_counter >= self.early_stopping_patience:
                    print(f"🛑 Early stopping triggered after {epoch+1} epochs.")
                    break
            else:
                # Calculate epoch time
                epoch_time = time.time() - epoch_start_time
                
                # Print epoch summary without validation metrics
                if verbose > 0:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"loss: {avg_train_loss:.4f}, acc: {train_accuracy:.4f}, "
                          f"lr: {current_lr:.6f}, time: {epoch_time:.1f}s")
                
                # Store empty validation metrics for consistency
                history['val_loss'].append(None)
                history['val_accuracy'].append(None)
                          
        # Load best model if validation was performed and we have saved a best state
        if val_loader is not None and hasattr(self, 'best_model_state'):
            self.load_state_dict(self.best_model_state)
            if verbose > 0:
                print("📊 Loaded best model state from early stopping")
        
        # Clear cache after training
        self.device_manager.clear_cache()
        
        # Generate diagnostic outputs if enabled
        if enable_diagnostics:
            if verbose > 0:
                print("📊 Generating diagnostic outputs...")
            
            # Clean up diagnostic hooks
            self._cleanup_diagnostic_hooks()
            
            # Save diagnostic plots
            self._save_diagnostic_plots(diagnostic_output_dir)
            
            # Generate and save diagnostic summary
            summary = self.get_diagnostic_summary()
            import json
            from pathlib import Path
            
            output_path = Path(diagnostic_output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            summary_path = output_path / f"{self.__class__.__name__}_diagnostic_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            if verbose > 0:
                print(f"📋 Diagnostic summary saved to {summary_path}")
                print("🏁 Training completed with comprehensive diagnostics")
        
        return history
    
    def evaluate(self, test_color_data=None, test_brightness_data=None, test_labels=None, 
                 test_loader=None, batch_size: int = 32) -> Dict[str, float]:
        """
        Evaluate the model on test data or test data loader.
        
        Supports both direct data arrays and DataLoader inputs.
        Expects 4D tensor inputs (no reshaping required).
        
        Args:
            test_color_data: Test color data (optional if test_loader is provided)
            test_brightness_data: Test brightness data (optional if test_loader is provided)
            test_labels: Test labels (optional if test_loader is provided)
            test_loader: Test data loader (optional if direct data is provided)
            batch_size: Batch size for evaluation (used when creating a loader from direct data)
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.eval()
        
        if test_loader is None and (test_color_data is not None and test_brightness_data is not None and test_labels is not None):
            # Convert to tensors if needed - avoiding torch.tensor(tensor) pattern
            if isinstance(test_color_data, np.ndarray):
                color_tensor = torch.from_numpy(test_color_data).float()
            elif isinstance(test_color_data, torch.Tensor):
                color_tensor = test_color_data.detach().clone()
            else:
                color_tensor = test_color_data
                
            if isinstance(test_brightness_data, np.ndarray):
                brightness_tensor = torch.from_numpy(test_brightness_data).float()
            elif isinstance(test_brightness_data, torch.Tensor):
                brightness_tensor = test_brightness_data.detach().clone()
            else:
                brightness_tensor = test_brightness_data
                
            if isinstance(test_labels, np.ndarray):
                labels_tensor = torch.from_numpy(test_labels).long()
            elif isinstance(test_labels, torch.Tensor):
                labels_tensor = test_labels.detach().clone()
            else:
                labels_tensor = test_labels
            
            # Make sure inputs are 4D tensors
            if len(color_tensor.shape) != 4:
                raise ValueError("MultiChannelResNetNetwork expects 4D input tensors in shape [batch, channels, height, width]")
            
            # Move to device
            color_tensor = color_tensor.to(self.device)
            brightness_tensor = brightness_tensor.to(self.device)
            labels_tensor = labels_tensor.to(self.device)
            
            # Create dataset and loader
            dataset = TensorDataset(color_tensor, brightness_tensor, labels_tensor)
            test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        elif test_loader is None:
            raise ValueError("Either test_loader or all of test_color_data, test_brightness_data, and test_labels must be provided")
        
        # Use the validation method to get metrics
        val_loss, val_accuracy = self._validate(test_loader)
        
        # Count total samples
        total = sum(len(batch[2]) for batch in test_loader)
        correct = int(val_accuracy * total)
        
        return {
            'accuracy': val_accuracy,
            'loss': val_loss,
            'correct': correct,
            'total': total
        }
        
    @classmethod
    def for_cifar100(cls, dropout=0.3, block_type='basic'):
        """
        Factory method to create a model pre-configured for CIFAR-100.
        
        This creates a model with reduced architecture and appropriate settings
        specifically optimized for the CIFAR-100 dataset.
        
        Args:
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
            reduce_architecture=True,  # Enable CIFAR optimization
            device='auto'  # Auto-detect best device
        )
        
        # Print model configuration summary
        params = sum(p.numel() for p in model.parameters())
        print("🚀 Created CIFAR-100 optimized model:")
        print(f"   Architecture: Reduced ResNet with {block_type} blocks")
        print(f"   Parameters: {params:,}")
        print("   Classifier: Shared (fusion)")
        print(f"   Device: {model.device}")
        
        return model

    def _validate(self, val_loader):
        """
        Validation method for MultiChannelResNetNetwork, which expects 4D tensors.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (average_validation_loss, validation_accuracy)
        """
        self.eval()
        total_val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):
                # Unpack data - handle different formats flexibly
                if isinstance(data, (list, tuple)) and len(data) == 3:
                    color_batch, brightness_batch, labels_batch = data
                elif hasattr(data, '__iter__') and len(data) == 3:
                    # Handle other iterable formats (e.g., from different DataLoader implementations)
                    color_batch, brightness_batch, labels_batch = data
                else:
                    raise ValueError("DataLoader must provide (color, brightness, labels) tuples")
                
                # Move data to device if not already there (with non-blocking for performance)
                if color_batch.device != self.device:
                    color_batch = color_batch.to(self.device, non_blocking=True)
                if brightness_batch.device != self.device:
                    brightness_batch = brightness_batch.to(self.device, non_blocking=True)
                if labels_batch.device != self.device:
                    labels_batch = labels_batch.to(self.device, non_blocking=True)
                
                # For MultiChannelResNetNetwork, we expect 4D inputs (no reshaping needed)
                # Make sure we have 4D tensors
                if len(color_batch.shape) != 4:
                    raise ValueError("MultiChannelResNetNetwork expects 4D input tensors in shape [batch, channels, height, width]")
                
                # Forward pass with mixed precision if available and configured
                if self.use_mixed_precision and hasattr(self, 'scaler') and self.scaler is not None:
                    with autocast(device_type=self.device.type):
                        outputs = self(color_batch, brightness_batch)
                        loss = self.criterion(outputs, labels_batch)
                else:
                    outputs = self(color_batch, brightness_batch)
                    loss = self.criterion(outputs, labels_batch)
                
                # Track metrics
                total_val_loss += loss.item()
                
                # Calculate validation accuracy
                _, predicted = outputs.max(1)
                val_total += labels_batch.size(0)
                val_correct += predicted.eq(labels_batch).sum().item()
                
                # Memory management - explicitly clear unnecessary tensors
                del outputs, loss, predicted
                
                # Periodic cache clearing for large datasets (especially important for GPU)
                if batch_idx % 20 == 0 and self.device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        # Calculate average validation metrics
        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = val_correct / val_total
        
        return avg_val_loss, val_accuracy

    def predict(self, color_data: Union[np.ndarray, torch.Tensor, DataLoader], brightness_data: Union[np.ndarray, torch.Tensor] = None, 
                batch_size: int = None) -> np.ndarray:
        """
        Make predictions on new data with proper input handling for MultiChannelResNetNetwork.
        
        This implementation overrides the parent method to:
        1. Support both direct data arrays and DataLoader inputs
        2. Enforce 4D input tensors required by MultiChannelResNetNetwork
        
        Args:
            color_data: Color input data or a DataLoader containing both color and brightness data
            brightness_data: Brightness input data (not needed if color_data is a DataLoader)
            batch_size: Batch size for prediction (auto-detected if None)
            
        Returns:
            Predicted class labels
        """
        # Auto-detect optimal batch size for inference
        if batch_size is None:
            if self.device.type == 'cuda':
                batch_size = 512  # Larger batches for inference
            else:
                batch_size = 128
        
        self.eval()
        
        # Handle DataLoader input
        if isinstance(color_data, DataLoader):
            loader = color_data
            
            predictions = []
            with torch.no_grad():
                for batch_data in loader:
                    # Handle different DataLoader formats
                    if len(batch_data) >= 2:  # Expecting (color, brightness, [labels]) format
                        batch_color = batch_data[0]
                        batch_brightness = batch_data[1]
                    else:
                        raise ValueError("DataLoader must provide at least (color, brightness) tuples")
                    
                    # Move to device if needed
                    if batch_color.device != self.device:
                        batch_color = batch_color.to(self.device, non_blocking=True)
                    if batch_brightness.device != self.device:
                        batch_brightness = batch_brightness.to(self.device, non_blocking=True)
                    
                    # Enforce 4D input tensors for MultiChannelResNetNetwork
                    if len(batch_color.shape) != 4:
                        raise ValueError("MultiChannelResNetNetwork expects 4D input tensors in shape [batch, channels, height, width]")
                    
                    # Forward pass with mixed precision if available
                    if self.use_mixed_precision:
                        with autocast(device_type='cuda'):
                            outputs = self(batch_color, batch_brightness)
                    else:
                        outputs = self(batch_color, batch_brightness)
                    
                    _, predicted = torch.max(outputs.data, 1)
                    predictions.extend(predicted.cpu().numpy())
            
            return np.array(predictions)
        
        # Handle direct data input
        else:
            if brightness_data is None:
                raise ValueError("brightness_data must be provided when not using a DataLoader")
            
            # Convert to tensors if needed - avoiding torch.tensor(tensor) pattern
            if isinstance(color_data, np.ndarray):
                color_tensor = torch.from_numpy(color_data).float()
            elif isinstance(color_data, torch.Tensor):
                color_tensor = color_data.detach().clone()
            else:
                color_tensor = color_data
                
            if isinstance(brightness_data, np.ndarray):
                brightness_tensor = torch.from_numpy(brightness_data).float()
            elif isinstance(brightness_data, torch.Tensor):
                brightness_tensor = brightness_data.detach().clone()
            else:
                brightness_tensor = brightness_data
            
            # Enforce 4D input tensors for MultiChannelResNetNetwork
            if len(color_tensor.shape) != 4:
                raise ValueError("MultiChannelResNetNetwork expects 4D input tensors in shape [batch, channels, height, width]")
            
            # Move to device efficiently
            color_tensor = color_tensor.to(self.device, non_blocking=True)
            brightness_tensor = brightness_tensor.to(self.device, non_blocking=True)
            
            # Create dataset and loader with optimizations
            dataset = TensorDataset(color_tensor, brightness_tensor)
            loader = DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=False,
                pin_memory=self.device.type == 'cuda'
            )
            
            predictions = []
            with torch.no_grad():
                for batch_color, batch_brightness in loader:
                    if self.use_mixed_precision:
                        with autocast(device_type='cuda'):
                            outputs = self(batch_color, batch_brightness)
                    else:
                        outputs = self(batch_color, batch_brightness)
                    _, predicted = torch.max(outputs.data, 1)
                    predictions.extend(predicted.cpu().numpy())
            
            return np.array(predictions)
    
    def predict_proba(self, color_data: Union[np.ndarray, torch.Tensor, DataLoader], brightness_data: Union[np.ndarray, torch.Tensor] = None,
                      batch_size: int = None) -> np.ndarray:
        """
        Get prediction probabilities with proper input handling for MultiChannelResNetNetwork.
        
        This implementation overrides the parent method to:
        1. Support both direct data arrays and DataLoader inputs
        2. Enforce 4D input tensors required by MultiChannelResNetNetwork
        
        Args:
            color_data: Color input data or a DataLoader containing both color and brightness data
            brightness_data: Brightness input data (not needed if color_data is a DataLoader)
            batch_size: Batch size for prediction (auto-detected if None)
            
        Returns:
            Prediction probabilities
        """
        # Auto-detect optimal batch size for inference
        if batch_size is None:
            if self.device.type == 'cuda':
                batch_size = 512  # Larger batches for inference
            else:
                batch_size = 128
                
        self.eval()
        
        # Handle DataLoader input
        if isinstance(color_data, DataLoader):
            loader = color_data
            
            probabilities = []
            with torch.no_grad():
                for batch_data in loader:
                    # Handle different DataLoader formats
                    if len(batch_data) >= 2:  # Expecting (color, brightness, [labels]) format
                        batch_color = batch_data[0]
                        batch_brightness = batch_data[1]
                    else:
                        raise ValueError("DataLoader must provide at least (color, brightness) tuples")
                    
                    # Move to device if needed
                    if batch_color.device != self.device:
                        batch_color = batch_color.to(self.device, non_blocking=True)
                    if batch_brightness.device != self.device:
                        batch_brightness = batch_brightness.to(self.device, non_blocking=True)
                    
                    # Enforce 4D input tensors for MultiChannelResNetNetwork
                    if len(batch_color.shape) != 4:
                        raise ValueError("MultiChannelResNetNetwork expects 4D input tensors in shape [batch, channels, height, width]")
                    
                    # Forward pass
                    outputs = self(batch_color, batch_brightness)
                    probs = torch.softmax(outputs, dim=1)
                    probabilities.extend(probs.cpu().numpy())
            
            return np.array(probabilities)
        
        # Handle direct data input
        else:
            if brightness_data is None:
                raise ValueError("brightness_data must be provided when not using a DataLoader")
            
            # Convert to tensors if needed - avoiding torch.tensor(tensor) pattern
            if isinstance(color_data, np.ndarray):
                color_tensor = torch.from_numpy(color_data).float()
            elif isinstance(color_data, torch.Tensor):
                color_tensor = color_data.detach().clone()
            else:
                color_tensor = color_data
                
            if isinstance(brightness_data, np.ndarray):
                brightness_tensor = torch.from_numpy(brightness_data).float()
            elif isinstance(brightness_data, torch.Tensor):
                brightness_tensor = brightness_data.detach().clone()
            else:
                brightness_tensor = brightness_data
            
            # Enforce 4D input tensors for MultiChannelResNetNetwork
            if len(color_tensor.shape) != 4:
                raise ValueError("MultiChannelResNetNetwork expects 4D input tensors in shape [batch, channels, height, width]")
            
            # Move to device efficiently
            color_tensor = color_tensor.to(self.device, non_blocking=True)
            brightness_tensor = brightness_tensor.to(self.device, non_blocking=True)
            
            # Create dataset and loader with optimizations
            dataset = TensorDataset(color_tensor, brightness_tensor)
            loader = DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=False,
                pin_memory=self.device.type == 'cuda'
            )
            
            probabilities = []
            with torch.no_grad():
                for batch_color, batch_brightness in loader:
                    outputs = self(batch_color, batch_brightness)
                    probs = torch.softmax(outputs, dim=1)
                    probabilities.extend(probs.cpu().numpy())
            
            return np.array(probabilities)
        

# Factory functions to create ResNet variants
def multi_channel_resnet18(num_classes=1000, **kwargs):
    """Create a MultiChannelResNetNetwork with ResNet-18 architecture."""
    return MultiChannelResNetNetwork(
        num_classes=num_classes,
        num_blocks=[2, 2, 2, 2],
        block_type='basic',
        **kwargs
    )

def multi_channel_resnet34(num_classes=1000, **kwargs):
    """Create a MultiChannelResNetNetwork with ResNet-34 architecture."""
    return MultiChannelResNetNetwork(
        num_classes=num_classes,
        num_blocks=[3, 4, 6, 3],
        block_type='basic',
        **kwargs
    )

def multi_channel_resnet50(num_classes=1000, **kwargs):
    """Create a MultiChannelResNetNetwork with ResNet-50 architecture."""
    return MultiChannelResNetNetwork(
        num_classes=num_classes,
        num_blocks=[3, 4, 6, 3],
        block_type='bottleneck',
        **kwargs
    )

def multi_channel_resnet101(num_classes=1000, **kwargs):
    """Create a MultiChannelResNetNetwork with ResNet-101 architecture."""
    return MultiChannelResNetNetwork(
        num_classes=num_classes,
        num_blocks=[3, 4, 23, 3],
        block_type='bottleneck',
        **kwargs
    )

def multi_channel_resnet152(num_classes=1000, **kwargs):
    """Create a MultiChannelResNetNetwork with ResNet-152 architecture."""
    return MultiChannelResNetNetwork(
        num_classes=num_classes,
        num_blocks=[3, 8, 36, 3],
        block_type='bottleneck',
        **kwargs
    )


