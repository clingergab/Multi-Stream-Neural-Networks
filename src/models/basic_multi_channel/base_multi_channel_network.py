"""
Base Multi-Channel Network using BasicMultiChannelLayer for dense/tabular data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch import GradScaler, autocast
import numpy as np
from typing import Tuple, Dict, List, Union, Optional
from tqdm import tqdm
from ..base import BaseMultiStreamModel
from ..layers.basic_layers import BasicMultiChannelLayer
from ...utils.device_utils import DeviceManager


class BaseMultiChannelNetwork(BaseMultiStreamModel):
    """
    Base Multi-Channel Network for dense/tabular multi-stream data.
    
    Uses BasicMultiChannelLayer components for fully-connected processing.
    Suitable for:
    - Tabular multi-modal data
    - Dense feature vectors
    - Embeddings
    - Flattened image features
    
    Features:
    - Automatic GPU detection and optimization
    - Keras-like training API
    - Built-in training loop with progress tracking
    
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
    model = BaseMultiChannelNetwork(...)
    output = model(color_data, brightness_data)  # Single tensor
    loss = criterion(output, labels)  # Works seamlessly
    
    # Research/analysis:
    color_logits, brightness_logits = model.analyze_pathways(color_data, brightness_data)
    color_accuracy = accuracy_metric(color_logits, labels)
    brightness_accuracy = accuracy_metric(brightness_logits, labels)
    """
    
    def __init__(
        self,
        input_size: int = None,
        color_input_size: int = None, 
        brightness_input_size: int = None,
        hidden_sizes: List[int] = [512, 256, 128],
        num_classes: int = 10,
        activation: str = 'relu',
        dropout: float = 0.0,
        use_shared_classifier: bool = True,  # NEW: Enable proper fusion
        device: str = 'auto',  # NEW: Automatic device detection
        **kwargs
    ):
        """
        Initialize BaseMultiChannelNetwork.
        
        Args:
            input_size: For backward compatibility - both streams use same input size
            color_input_size: Input size for color stream
            brightness_input_size: Input size for brightness stream
            hidden_sizes: List of hidden layer sizes
            num_classes: Number of output classes
            activation: Activation function
            dropout: Dropout rate
            use_shared_classifier: If True, use a shared classifier for proper fusion.
                                 If False, use separate classifiers (legacy behavior)
            device: Device for model training/inference - 'auto', 'cpu', or 'cuda'
        """
        # Handle backward compatibility
        if input_size is not None:
            color_input_size = input_size
            brightness_input_size = input_size
        
        if color_input_size is None or brightness_input_size is None:
            raise ValueError("Must provide either input_size or both color_input_size and brightness_input_size")
            
        # Initialize base class with appropriate input_size tuple
        super().__init__(
            input_size=(max(color_input_size, brightness_input_size),),  # Use larger for base class
            hidden_size=hidden_sizes[0] if hidden_sizes else 256,
            num_classes=num_classes,
            **kwargs
        )
        
        self.color_input_size = color_input_size
        self.brightness_input_size = brightness_input_size
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.dropout = dropout
        self.use_shared_classifier = use_shared_classifier
        
        # Setup device management with proper detection
        self.device_manager = DeviceManager(preferred_device=device if device != 'auto' else None)
        self.device = self.device_manager.device
        
        # Mixed precision support
        self.use_mixed_precision = self.device_manager.enable_mixed_precision()
        self.scaler = GradScaler('cuda') if self.use_mixed_precision else None
        
        # Build network layers
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
        """Build the multi-channel network using BasicMultiChannelLayer."""
        layers = []
        
        # Input layer
        color_current_size = self.color_input_size
        brightness_current_size = self.brightness_input_size
        
        if self.hidden_sizes:
            first_hidden = self.hidden_sizes[0]
            layers.append(BasicMultiChannelLayer(
                color_input_size=color_current_size,
                brightness_input_size=brightness_current_size,
                output_size=first_hidden,
                activation=self.activation,
                bias=True
            ))
            # After first layer, both streams have same size
            color_current_size = first_hidden
            brightness_current_size = first_hidden
            
            # Add dropout if specified
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
        
        # Hidden layers - now both streams have same size
        for hidden_size in self.hidden_sizes[1:]:
            layers.append(BasicMultiChannelLayer(
                color_input_size=color_current_size,
                brightness_input_size=brightness_current_size,
                output_size=hidden_size,
                activation=self.activation,
                bias=True
            ))
            color_current_size = hidden_size
            brightness_current_size = hidden_size
            
            # Add dropout if specified
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
        
        # Store layers
        self.layers = nn.ModuleList(layers)
        
        # Output classifier - choose between shared fusion or separate classifiers
        final_size = self.hidden_sizes[-1] if self.hidden_sizes else max(self.color_input_size, self.brightness_input_size)
        
        if self.use_shared_classifier:
            # Shared classifier with proper fusion - concatenates features from both streams
            self.shared_classifier = nn.Linear(final_size * 2, self.num_classes, bias=True)
            # Also create separate projection heads for research/analysis purposes
            self.color_head = nn.Linear(final_size, self.num_classes, bias=True)
            self.brightness_head = nn.Linear(final_size, self.num_classes, bias=True)
            self.multi_channel_classifier = None  # Not used in shared mode
        else:
            # Legacy: Separate classifiers for each stream
            self.shared_classifier = None  # Not used in separate mode
            self.color_head = None
            self.brightness_head = None
            self.multi_channel_classifier = BasicMultiChannelLayer(
                color_input_size=final_size,
                brightness_input_size=final_size,
                output_size=self.num_classes,
                activation='linear',  # No activation for final layer
                bias=True
            )
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, BasicMultiChannelLayer):
                # BasicMultiChannelLayer handles its own initialization
                pass
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _extract_features(self, color_input: torch.Tensor, brightness_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract features through all hidden layers.
        
        Private method that handles all the layer processing.
        
        Args:
            color_input: Color features tensor
            brightness_input: Brightness features tensor
            
        Returns:
            Tuple of (color_features, brightness_features) after processing through all layers
        """
        color_x, brightness_x = color_input, brightness_input
        
        # Process through all layers
        for layer in self.layers:
            if isinstance(layer, BasicMultiChannelLayer):
                color_x, brightness_x = layer(color_x, brightness_x)
            elif isinstance(layer, nn.Dropout):
                # Apply dropout to both streams
                color_x = layer(color_x)
                brightness_x = layer(brightness_x)
        
        return color_x, brightness_x

    def forward(self, color_input: torch.Tensor, brightness_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the multi-channel network for training and classification.
        
        This is the primary method called by model(x, y) and used for training, inference, and evaluation.
        Returns a single tensor suitable for loss computation and classification.
        
        Args:
            color_input: Color features tensor
            brightness_input: Brightness features tensor
            
        Returns:
            Combined classification logits [batch_size, num_classes]
        """
        # Extract features through all layers
        color_x, brightness_x = self._extract_features(color_input, brightness_input)
        
        if self.use_shared_classifier:
            # Use shared classifier for optimal fusion
            fused_features = torch.cat([color_x, brightness_x], dim=1)
            return self.shared_classifier(fused_features)
        else:
            # Legacy: Add separate outputs
            color_logits, brightness_logits = self.multi_channel_classifier(color_x, brightness_x)
            return color_logits + brightness_logits
    
    def analyze_pathways(self, color_input: torch.Tensor, brightness_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Analyze individual pathway contributions for research purposes.
        
        Returns separate outputs for each stream to analyze individual pathway contributions.
        Use this method only for research, visualization, and pathway analysis.
        
        Args:
            color_input: Color features tensor [batch_size, input_size]
            brightness_input: Brightness features tensor [batch_size, input_size]
            
        Returns:
            Tuple of (color_logits, brightness_logits) [batch_size, num_classes] each
            Separate outputs for analyzing individual pathway performance
        """
        # Extract features through all layers
        color_x, brightness_x = self._extract_features(color_input, brightness_input)
        
        # Apply separate classifiers/heads for analysis
        if self.use_shared_classifier:
            # Use separate heads for meaningful individual stream analysis
            color_logits = self.color_head(color_x)
            brightness_logits = self.brightness_head(brightness_x)
            return color_logits, brightness_logits
        else:
            # Legacy: Separate classifiers
            color_logits, brightness_logits = self.multi_channel_classifier(color_x, brightness_x)
            return color_logits, brightness_logits
    
    def extract_features(self, color_input: torch.Tensor, brightness_input: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Extract features before final classification.
        
        Args:
            color_input: Color features tensor
            brightness_input: Brightness features tensor
            
        Returns:
            If use_shared_classifier=True: Concatenated features [batch_size, feature_size * 2]
            If use_shared_classifier=False: Tuple of (color_features, brightness_features)
        """
        # Use the private helper method to avoid code duplication
        color_x, brightness_x = self._extract_features(color_input, brightness_input)
        
        if self.use_shared_classifier:
            # Return concatenated features
            return torch.cat([color_x, brightness_x], dim=1)
        else:
            # Return separate features
            return color_x, brightness_x
    
    @property
    def fusion_type(self) -> str:
        """Return the type of fusion used by this model."""
        return "shared_classifier" if self.use_shared_classifier else "separate_classifiers"
    
    def get_classifier_info(self) -> Dict:
        """Get information about the classifier architecture."""
        if self.use_shared_classifier:
            shared_params = sum(p.numel() for p in self.shared_classifier.parameters())
            color_params = sum(p.numel() for p in self.color_head.parameters())
            brightness_params = sum(p.numel() for p in self.brightness_head.parameters())
            return {
                'type': 'shared_with_separate_heads',
                'shared_classifier_params': shared_params,
                'color_head_params': color_params,
                'brightness_head_params': brightness_params,
                'total_params': shared_params + color_params + brightness_params,
                'shared_input_size': self.shared_classifier.in_features,
                'output_size': self.shared_classifier.out_features
            }
        else:
            return {
                'type': 'separate', 
                'color_params': sum(p.numel() for p in [self.multi_channel_classifier.color_weights, self.multi_channel_classifier.color_bias] if p is not None),
                'brightness_params': sum(p.numel() for p in [self.multi_channel_classifier.brightness_weights, self.multi_channel_classifier.brightness_bias] if p is not None),
                'total_params': sum(p.numel() for p in self.multi_channel_classifier.parameters())
            }
    
    def get_pathway_importance(self) -> Dict[str, float]:
        """Calculate pathway importance based on final classifier weights."""
        if hasattr(self.classifier, 'color_weights') and hasattr(self.classifier, 'brightness_weights'):
            color_norm = torch.norm(self.classifier.color_weights.data).item()
            brightness_norm = torch.norm(self.classifier.brightness_weights.data).item()
            total_norm = color_norm + brightness_norm + 1e-8
            
            return {
                'color_pathway': color_norm / total_norm,
                'brightness_pathway': brightness_norm / total_norm,
                'pathway_ratio': color_norm / (brightness_norm + 1e-8)
            }
        
        return {'color_pathway': 0.5, 'brightness_pathway': 0.5, 'pathway_ratio': 1.0}

    def analyze_pathway_weights(self) -> Dict[str, float]:
        """
        Analyze the relative importance of color vs brightness pathways.
        
        Returns:
            Dictionary with pathway weight statistics
        """
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
                'fusion_type': 'shared_with_separate_heads',
                'shared_color_norm': torch.norm(color_shared_weights).item(),
                'shared_brightness_norm': torch.norm(brightness_shared_weights).item(),
                'head_color_norm': torch.norm(color_head_weights).item(),
                'head_brightness_norm': torch.norm(brightness_head_weights).item()
            }
        else:
            # For separate classifiers, analyze each pathway
            color_norm = torch.norm(self.multi_channel_classifier.color_weights.data).item()
            brightness_norm = torch.norm(self.multi_channel_classifier.brightness_weights.data).item()
            total_norm = color_norm + brightness_norm + 1e-8
            
            return {
                'color_pathway': color_norm / total_norm,
                'brightness_pathway': brightness_norm / total_norm,
                'pathway_ratio': color_norm / (brightness_norm + 1e-8),
                'fusion_type': 'separate_classifiers'
            }
        
    def fit(
        self,
        train_color_data: np.ndarray,
        train_brightness_data: np.ndarray,
        train_labels: np.ndarray,
        val_color_data: Optional[np.ndarray] = None,
        val_brightness_data: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None,
        batch_size: int = None,  # Auto-detect optimal batch size
        epochs: int = 10,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        early_stopping_patience: int = 5,
        verbose: int = 1,
        num_workers: int = None,  # Auto-detect optimal workers
        pin_memory: bool = None   # Auto-detect based on device
    ):
        """
        Fit the model to the data using Keras-like training API with GPU optimizations.
        
        Args:
            train_color_data: Training data for color stream
            train_brightness_data: Training data for brightness stream
            train_labels: Training labels
            val_color_data: Validation data for color stream
            val_brightness_data: Validation data for brightness stream
            val_labels: Validation labels
            batch_size: Batch size for training (auto-detected if None)
            epochs: Number of epochs to train
            learning_rate: Learning rate for the optimizer
            weight_decay: Weight decay (L2 regularization)
            early_stopping_patience: Patience for early stopping
            verbose: Verbosity level (0: silent, 1: progress bar, 2: one line per epoch)
            num_workers: Number of workers for data loading (auto-detected if None)
            pin_memory: Whether to pin memory for faster GPU transfer (auto-detected if None)
        """
        # Auto-detect optimal batch size for GPU
        if batch_size is None:
            if self.device.type == 'cuda':
                # For A100/V100 GPUs, use larger batch sizes
                memory_gb = torch.cuda.get_device_properties(self.device).total_memory / 1e9
                if memory_gb >= 40:  # A100
                    batch_size = 256
                elif memory_gb >= 16:  # V100
                    batch_size = 128
                else:
                    batch_size = 64
            else:
                batch_size = 32
        
        # Auto-detect optimal number of workers
        if num_workers is None:
            import os
            num_workers = min(8, os.cpu_count() or 1)
        
        # Auto-detect pin_memory
        if pin_memory is None:
            pin_memory = self.device.type == 'cuda'
        
        if verbose > 0:
            print("🚀 Training with optimized settings:")
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
        
        # Optimizer and loss function
        optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler for better convergence
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Training loop with mixed precision
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Single progress bar for training batches only
            epoch_pbar = None
            if verbose == 1:
                # Clear any existing tqdm instances to avoid duplicate progress bars
                try:
                    # Force clear any existing progress bars to prevent duplication
                    # Check if any existing progress bars are in the tqdm instances list
                    for inst in list(getattr(tqdm, '_instances', [])):
                        try:
                            inst.close()
                        except Exception:
                            pass  # Ignore if we can't close an instance
                except Exception:
                    pass  # Ignore any errors in cleaning up tqdm instances
                
                # Create a new progress bar with appropriate settings to prevent duplication
                epoch_pbar = tqdm(
                    total=len(train_loader), 
                    desc=f"Epoch {epoch+1}/{epochs}", 
                    leave=True,  # Leave this progress bar after completion
                    dynamic_ncols=True,  # Adapt to terminal width
                    position=0,  # Keep it at position 0 (top)
                    smoothing=0.3  # Smooth updates for better display
                )
            
            # Training phase
            self.train()
            total_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (batch_color, batch_brightness, batch_labels) in enumerate(train_loader):
                optimizer.zero_grad()
                
                if self.use_mixed_precision and self.scaler is not None:
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
                if verbose == 1 and epoch_pbar is not None:
                    train_acc = train_correct / train_total
                    epoch_pbar.set_postfix({
                        'Loss': f'{total_loss/(batch_idx+1):.4f}',
                        'Acc': f'{train_acc:.4f}'
                    })
                    epoch_pbar.update(1)
                    # Force refresh the display
                    epoch_pbar.refresh()
            
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
                        if self.use_mixed_precision and self.scaler is not None:
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
                if verbose == 1 and epoch_pbar is not None:
                    epoch_pbar.set_postfix({
                        'Loss': f'{avg_train_loss:.4f}',
                        'Acc': f'{train_accuracy:.4f}',
                        'Val_Loss': f'{avg_val_loss:.4f}',
                        'Val_Acc': f'{val_accuracy:.4f}'
                    })
                    # Make sure to display the final metrics before closing
                    epoch_pbar.refresh()
            else:
                # Training only - final update
                if verbose == 1 and epoch_pbar is not None:
                    epoch_pbar.set_postfix({
                        'Loss': f'{avg_train_loss:.4f}',
                        'Acc': f'{train_accuracy:.4f}'
                    })
                    # Make sure to display the final metrics
                    epoch_pbar.refresh()
                avg_val_loss = float('inf')  # For early stopping logic
            
            # Close progress bar and clear line to prevent duplication
            if verbose == 1 and epoch_pbar is not None:
                epoch_pbar.close()
                print("\r\033[K", end="")  # Clear the current line
            
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
    

# Factory functions
def base_multi_channel_small(
    *,  # Force keyword-only arguments for clarity
    input_size: int = None, 
    color_input_size: int = None,
    brightness_input_size: int = None,
    num_classes: int = 10, 
    **kwargs
) -> BaseMultiChannelNetwork:
    """
    Create a small BaseMultiChannelNetwork.
    
    Args:
        input_size: For backward compatibility - both streams use same input size
        color_input_size: Input size for color stream
        brightness_input_size: Input size for brightness stream
        num_classes: Number of output classes
        **kwargs: Additional arguments passed to BaseMultiChannelNetwork
    
    Example:
        # Different input sizes (preferred for multi-stream)
        model = base_multi_channel_small(
            color_input_size=3072, 
            brightness_input_size=1024, 
            num_classes=100
        )
        
        # Same input size (backward compatibility)
        model = base_multi_channel_small(input_size=2048, num_classes=10)
    """
    return BaseMultiChannelNetwork(
        input_size=input_size,
        color_input_size=color_input_size,
        brightness_input_size=brightness_input_size,
        hidden_sizes=[256, 128],
        num_classes=num_classes,
        **kwargs
    )


def base_multi_channel_medium(
    *,  # Force keyword-only arguments for clarity
    input_size: int = None,
    color_input_size: int = None,
    brightness_input_size: int = None,
    num_classes: int = 10,
    **kwargs
) -> BaseMultiChannelNetwork:
    """
    Create a medium BaseMultiChannelNetwork.
    
    Args:
        input_size: For backward compatibility - both streams use same input size
        color_input_size: Input size for color stream
        brightness_input_size: Input size for brightness stream
        num_classes: Number of output classes
        **kwargs: Additional arguments passed to BaseMultiChannelNetwork
    
    Example:
        # Different input sizes (preferred for multi-stream)
        model = base_multi_channel_medium(
            color_input_size=3072, 
            brightness_input_size=1024, 
            num_classes=100
        )
        
        # Same input size (backward compatibility)
        model = base_multi_channel_medium(input_size=2048, num_classes=10)
    """
    return BaseMultiChannelNetwork(
        input_size=input_size,
        color_input_size=color_input_size,
        brightness_input_size=brightness_input_size,
        hidden_sizes=[512, 256, 128],
        num_classes=num_classes,
        **kwargs
    )


def base_multi_channel_large(
    *,  # Force keyword-only arguments for clarity
    input_size: int = None,
    color_input_size: int = None,
    brightness_input_size: int = None,
    num_classes: int = 10,
    **kwargs
) -> BaseMultiChannelNetwork:
    """
    Create a large BaseMultiChannelNetwork.
    
    Args:
        input_size: For backward compatibility - both streams use same input size
        color_input_size: Input size for color stream
        brightness_input_size: Input size for brightness stream
        num_classes: Number of output classes
        **kwargs: Additional arguments passed to BaseMultiChannelNetwork
    
    Example:
        # Different input sizes (preferred for multi-stream)
        model = base_multi_channel_large(
            color_input_size=3072, 
            brightness_input_size=1024, 
            num_classes=100
        )
        
        # Same input size (backward compatibility)
        model = base_multi_channel_large(input_size=2048, num_classes=10)
    """
    return BaseMultiChannelNetwork(
        input_size=input_size,
        color_input_size=color_input_size,
        brightness_input_size=brightness_input_size,
        hidden_sizes=[1024, 512, 256, 128],
        num_classes=num_classes,
        **kwargs
    )
