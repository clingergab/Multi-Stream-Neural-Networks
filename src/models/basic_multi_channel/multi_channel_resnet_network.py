"""
Multi-Channel ResNet implementation following standard ResNet architecture.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch import GradScaler, autocast
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
        dropout: float = 0.0,  # NEW: Dropout support
        use_shared_classifier: bool = True,  # NEW: Enable proper fusion
        input_channels: int = None,  # For backward compatibility
        device: str = 'auto',  # NEW: Automatic device detection
        **kwargs
    ):
        # Handle backward compatibility
        if input_channels is not None:
            color_input_channels = input_channels
            brightness_input_channels = input_channels
        
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
        self.dropout = dropout  # NEW: Store dropout rate
        self.use_shared_classifier = use_shared_classifier
        
        # Setup device management with proper detection
        self.device_manager = DeviceManager(preferred_device=device if device != 'auto' else None)
        self.device = self.device_manager.device
        
        # Mixed precision support
        self.use_mixed_precision = self.device_manager.enable_mixed_precision()
        self.scaler = GradScaler('cuda') if self.use_mixed_precision else None
        
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
        """Initialize network weights following ResNet conventions."""
        for module in self.modules():
            if isinstance(module, MultiChannelConv2d):
                # Initialize both color and brightness conv weights
                nn.init.kaiming_normal_(module.color_weight, mode='fan_out', nonlinearity='relu')
                nn.init.kaiming_normal_(module.brightness_weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, MultiChannelBatchNorm2d):
                # Initialize both color and brightness batch norm
                nn.init.constant_(module.color_bn.weight, 1)
                nn.init.constant_(module.color_bn.bias, 0)
                nn.init.constant_(module.brightness_bn.weight, 1)
                nn.init.constant_(module.brightness_bn.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
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
        
        # Combine outputs for classification
        if self.use_shared_classifier:
            # Use shared classifier for optimal fusion
            fused_features = torch.cat([color_x, brightness_x], dim=1)
            return self.shared_classifier(fused_features)
        else:
            # Legacy: Add separate outputs
            color_logits = self.color_classifier(color_x)
            brightness_logits = self.brightness_classifier(brightness_x)
            return color_logits + brightness_logits
    
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
        
        # Separate classification for each stream (for analysis)
        if self.use_shared_classifier:
            # Use individual heads for analysis when shared classifier is used
            color_logits = self.color_head(color_x)
            brightness_logits = self.brightness_head(brightness_x)
        else:
            # Use separate classifiers when in legacy mode
            color_logits = self.color_classifier(color_x)
            brightness_logits = self.brightness_classifier(brightness_x)
        
        return color_logits, brightness_logits
    
    def extract_features(self, color_input: torch.Tensor, brightness_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features before final classification."""
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

    def fit(self, train_color_data: np.ndarray, train_brightness_data: np.ndarray, train_labels: np.ndarray,
            val_color_data: Optional[np.ndarray] = None, val_brightness_data: Optional[np.ndarray] = None,
            val_labels: Optional[np.ndarray] = None, batch_size: int = None, epochs: int = 10,
            learning_rate: float = 0.001, weight_decay: float = 0.0, early_stopping_patience: int = 5,
            verbose: int = 1, num_workers: int = None, pin_memory: bool = None):
        """
        Fit the ResNet model to the data using Keras-like training API with GPU optimizations.
        
        Args:
            train_color_data: Training data for color stream [N, C, H, W]
            train_brightness_data: Training data for brightness stream [N, C, H, W] 
            train_labels: Training labels
            val_color_data: Validation data for color stream
            val_brightness_data: Validation data for brightness stream
            val_labels: Validation labels
            batch_size: Batch size for training (auto-detected for CNN if None)
            epochs: Number of epochs to train
            learning_rate: Learning rate for the optimizer
            weight_decay: Weight decay (L2 regularization)
            early_stopping_patience: Patience for early stopping
            verbose: Verbosity level (0: silent, 1: progress bar, 2: one line per epoch)
            num_workers: Number of workers for data loading (auto-detected if None)
            pin_memory: Whether to pin memory for faster GPU transfer (auto-detected if None)
        """
        # Auto-detect optimal batch size for CNN
        if batch_size is None:
            if self.device.type == 'cuda':
                # For CNNs, use smaller batch sizes due to memory requirements
                memory_gb = torch.cuda.get_device_properties(self.device).total_memory / 1e9
                if memory_gb >= 40:  # A100
                    batch_size = 64
                elif memory_gb >= 16:  # V100
                    batch_size = 32
                else:
                    batch_size = 16
            else:
                batch_size = 8  # Conservative for CPU/MPS
        
        # Auto-detect optimal number of workers
        if num_workers is None:
            import os
            num_workers = min(8, os.cpu_count() or 1)
        
        # Auto-detect pin_memory
        if pin_memory is None:
            pin_memory = self.device.type == 'cuda'
        
        if verbose > 0:
            print("🚀 Training ResNet with optimized settings:")
            print(f"   Device: {self.device}")
            print(f"   Batch size: {batch_size}")
            print(f"   Mixed precision: {self.use_mixed_precision}")
            print(f"   Workers: {num_workers}")
            print(f"   Pin memory: {pin_memory}")
        
        # Move data to device efficiently
        train_color_tensor = torch.tensor(train_color_data, dtype=torch.float32).to(self.device, non_blocking=True)
        train_brightness_tensor = torch.tensor(train_brightness_data, dtype=torch.float32).to(self.device, non_blocking=True)
        train_labels_tensor = torch.tensor(train_labels, dtype=torch.long).to(self.device, non_blocking=True)
        
        train_dataset = TensorDataset(train_color_tensor, train_brightness_tensor, train_labels_tensor)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers if self.device.type != 'mps' else 0,  # MPS doesn't support multiprocessing
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0 and self.device.type != 'mps'
        )
        
        if val_color_data is not None and val_brightness_data is not None and val_labels is not None:
            val_color_tensor = torch.tensor(val_color_data, dtype=torch.float32).to(self.device, non_blocking=True)
            val_brightness_tensor = torch.tensor(val_brightness_data, dtype=torch.float32).to(self.device, non_blocking=True)
            val_labels_tensor = torch.tensor(val_labels, dtype=torch.long).to(self.device, non_blocking=True)
            
            val_dataset = TensorDataset(val_color_tensor, val_brightness_tensor, val_labels_tensor)
            val_loader = DataLoader(
                val_dataset, 
                batch_size=batch_size, 
                shuffle=False,
                num_workers=num_workers if self.device.type != 'mps' else 0,  # MPS doesn't support multiprocessing
                pin_memory=pin_memory,
                persistent_workers=num_workers > 0 and self.device.type != 'mps'
            )
        else:
            val_loader = None
        
        # Optimizer and loss function - use AdamW for better generalization
        optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler optimized for CNNs
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Training loop with mixed precision
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Single progress bar for training batches only
            if verbose == 1:
                epoch_pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            
            # Training phase
            self.train()
            total_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (batch_color, batch_brightness, batch_labels) in enumerate(train_loader):
                optimizer.zero_grad()
                
                if self.use_mixed_precision:
                    with autocast('cuda'):
                        outputs = self(batch_color, batch_brightness)
                        loss = criterion(outputs, batch_labels)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    outputs = self(batch_color, batch_brightness)
                    loss = criterion(outputs, batch_labels)
                    loss.backward()
                    optimizer.step()
                
                total_loss += loss.item()
                
                # Calculate training accuracy
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_labels.size(0)
                train_correct += (predicted == batch_labels).sum().item()
                
                # Update progress bar with current training metrics
                if verbose == 1:
                    train_acc = train_correct / train_total
                    epoch_pbar.set_postfix({
                        'Loss': f'{total_loss/(batch_idx+1):.4f}',
                        'Acc': f'{train_acc:.4f}'
                    })
                    epoch_pbar.update(1)
            
            scheduler.step()
            avg_train_loss = total_loss / len(train_loader)
            train_accuracy = train_correct / train_total
            
            # Validation phase
            if val_loader is not None:
                self.eval()
                total_val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch_idx, (batch_color, batch_brightness, batch_labels) in enumerate(val_loader):
                        if self.use_mixed_precision:
                            with autocast('cuda'):
                                outputs = self(batch_color, batch_brightness)
                                loss = criterion(outputs, batch_labels)
                        else:
                            outputs = self(batch_color, batch_brightness)
                            loss = criterion(outputs, batch_labels)
                        total_val_loss += loss.item()
                        
                        # Calculate validation accuracy
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += batch_labels.size(0)
                        val_correct += (predicted == batch_labels).sum().item()
                
                avg_val_loss = total_val_loss / len(val_loader)
                val_accuracy = val_correct / val_total
                
                # Update progress bar with final validation metrics
                if verbose == 1:
                    epoch_pbar.set_postfix({
                        'Loss': f'{avg_train_loss:.4f}',
                        'Acc': f'{train_accuracy:.4f}',
                        'Val_Loss': f'{avg_val_loss:.4f}',
                        'Val_Acc': f'{val_accuracy:.4f}'
                    })
            else:
                # Training only - final update
                if verbose == 1:
                    epoch_pbar.set_postfix({
                        'Loss': f'{avg_train_loss:.4f}',
                        'Acc': f'{train_accuracy:.4f}'
                    })
                avg_val_loss = float('inf')  # For early stopping logic
            
            # Close progress bar
            if verbose == 1:
                epoch_pbar.close()
            
            # Early stopping check (only if validation data provided)
            if val_loader is not None:
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    # Save the best model
                    self.save_model()
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    if verbose > 0:
                        print(f"Early stopping triggered. Stopping training at epoch {epoch + 1}.")
                    break
            
            # Print epoch summary
            if verbose > 0:
                if val_loader is not None:
                    print(f"Epoch {epoch + 1}/{epochs} - "
                          f"Train Loss: {avg_train_loss:.4f} - Train Acc: {train_accuracy:.4f} - "
                          f"Val Loss: {avg_val_loss:.4f} - Val Acc: {val_accuracy:.4f} - "
                          f"LR: {optimizer.param_groups[0]['lr']:.6f}")
                else:
                    print(f"Epoch {epoch + 1}/{epochs} - "
                          f"Train Loss: {avg_train_loss:.4f} - Train Acc: {train_accuracy:.4f} - "
                          f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Clear cache after training
        self.device_manager.clear_cache()


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


