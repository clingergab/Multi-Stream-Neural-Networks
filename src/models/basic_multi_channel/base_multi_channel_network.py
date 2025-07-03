"""
Base Multi-Channel Network using BasicMultiChannelLayer for dense/tabular data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast
from typing import Tuple, Dict, List, Union
import sys
from tqdm import tqdm
from ..base import BaseMultiStreamModel
from ..layers.basic_layers import BasicMultiChannelLayer
from ...utils.grad_utils import safe_clip_grad_norm


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
    
    # Compilation defaults optimized for dense/tabular data
    DEFAULT_COMPILE_CONFIG = {
        'learning_rate': 0.001,    # Higher learning rate for dense networks
        'weight_decay': 0.0,       # Less regularization needed for tabular data
        'gradient_clip': 0.0,      # Disabled - dense networks have stable gradients
        'scheduler': 'cosine',     # Cosine works well for tabular data
        'early_stopping_patience': 10  # Longer patience - tabular convergence can be slower
    }
    
    def __init__(
        self,
        color_input_size: int, 
        brightness_input_size: int,
        num_classes: int,
        hidden_sizes: List[int] = [512, 256, 128],
        activation: str = 'relu',
        dropout: float = 0.0,
        device: str = 'auto', 
        **kwargs
    ):
        """
        Initialize BaseMultiChannelNetwork.
        
        Args:
            color_input_size: Input size for color stream
            brightness_input_size: Input size for brightness stream
            num_classes: Number of output classes
            hidden_sizes: List of hidden layer sizes
            activation: Activation function
            dropout: Dropout rate
            device: Device for model training/inference - 'auto', 'cpu', or 'cuda'
        """
        if color_input_size is None or brightness_input_size is None:
            raise ValueError("Must provide both color_input_size and brightness_input_size")
        
        # Initialize base class with common parameters
        super().__init__(
            num_classes=num_classes,
            activation=activation,
            dropout=dropout,
            device=device,
            **kwargs
        )
        
        self.color_input_size = color_input_size
        self.brightness_input_size = brightness_input_size
        self.hidden_sizes = hidden_sizes
        
        # Build network layers
        self._build_network()
        
        # Initialize weights
        self._initialize_weights()
        
        # Finalize initialization (device setup, etc.)
        self._finalize_initialization()
    
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
        
        # Output classifier - always use shared classifier with proper fusion
        final_size = self.hidden_sizes[-1] if self.hidden_sizes else max(self.color_input_size, self.brightness_input_size)
        
        # Shared classifier with proper fusion - concatenates features from both streams
        self.shared_classifier = nn.Linear(final_size * 2, self.num_classes, bias=True)
        # Also create separate projection heads for research/analysis purposes
        self.color_head = nn.Linear(final_size, self.num_classes, bias=True)
        self.brightness_head = nn.Linear(final_size, self.num_classes, bias=True)
    
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
        color_x, brightness_x = self._forward_through_layers(color_input, brightness_input)
        
        # Use shared classifier for optimal fusion
        fused_features = torch.cat([color_x, brightness_x], dim=1)
        return self.shared_classifier(fused_features)
    
    def _forward_through_layers(self, color_input: torch.Tensor, brightness_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through all hidden layers for both streams.
        
        Private method that handles all the layer processing for internal use.
        
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
    
    def extract_features(self, color_input: torch.Tensor, brightness_input: torch.Tensor) -> torch.Tensor:
        """
        Extract concatenated features before final classification.
        
        This method returns fused features ready for external classifiers.
        For separate pathway features, use get_separate_features() instead.
        
        Args:
            color_input: Color features tensor
            brightness_input: Brightness features tensor
            
        Returns:
            Concatenated fused features [batch_size, feature_size * 2]
            Ready for external linear classifiers or further processing
        """
        # Get separate features and concatenate them
        color_x, brightness_x = self.get_separate_features(color_input, brightness_input)
        
        # Return concatenated features (fusion approach for external use)
        return torch.cat([color_x, brightness_x], dim=1)
    
    def get_separate_features(self, color_input: torch.Tensor, brightness_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract separate features from both pathways without classification.
        
        This method returns individual pathway features for research and analysis.
        For fused features ready for external classifiers, use extract_features() instead.
        
        Args:
            color_input: Color features tensor
            brightness_input: Brightness features tensor
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Features from color and brightness pathways
            Separate features for pathway analysis and research
        """
        # Use the private helper method to get separate features
        color_x, brightness_x = self._forward_through_layers(color_input, brightness_input)
        
        # Return separate features (for research and pathway analysis)
        return color_x, brightness_x
    
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
        color_x, brightness_x = self._forward_through_layers(color_input, brightness_input)
        
        # Use separate heads for meaningful individual stream analysis
        color_logits = self.color_head(color_x)
        brightness_logits = self.brightness_head(brightness_x)
        return color_logits, brightness_logits
    
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

    @property
    def fusion_type(self) -> str:
        """Return the type of fusion used by this model."""
        return "shared_classifier"
    
    def get_classifier_info(self) -> Dict:
        """Get information about the classifier architecture."""
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
        verbose: int = 1
    ) -> Dict[str, List[float]]:
        """
        Unified fit method that handles both direct data arrays and DataLoaders with on-the-fly augmentation.
        
        This enhanced method supports two modes of operation:
        1. Direct data array input (original behavior): Provide train_color_data, train_brightness_data, train_labels
        2. DataLoader input (for on-the-fly augmentation): Provide train_loader directly
        
        Memory-Efficient Training:
        -------------------------
        For large datasets:
        - Tabular data can use much larger batch sizes (512-4096) compared to CNNs
        - For extremely large datasets, use DataLoaders with num_workers > 0
        - Mixed precision training is enabled by default on compatible GPUs
        
        Args:
            train_color_data: Training data for color stream [N, features] (used if train_loader not provided)
            train_brightness_data: Training data for brightness stream [N, features] (used if train_loader not provided)
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
            history: Dictionary containing training and validation metrics
        """
        if not self.is_compiled:
            raise RuntimeError("Model must be compiled before training. Call model.compile() first.")
            
        # Initialize history dictionary to track metrics
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rates': []
        }
        
        # Determine if we're using DataLoaders or direct data
        using_dataloaders = train_loader is not None
        
        # Use reasonable default batch size if not specified
        if batch_size is None and not using_dataloaders:
            # For tabular/dense data, moderate batch sizes work well across devices
            batch_size = 256 if self.device.type == 'cuda' else 64
        
        # Get optimal DataLoader configuration from device manager
        dataloader_config = self.device_manager.get_dataloader_config(conservative=False)
        num_workers = dataloader_config['num_workers']
        pin_memory = dataloader_config['pin_memory']
        
        
        # Set up train and validation loaders based on input mode
        if not using_dataloaders:
            # Check that required data is provided when not using DataLoaders
            if train_color_data is None or train_brightness_data is None or train_labels is None:
                raise ValueError("When train_loader is not provided, you must provide train_color_data, train_brightness_data, and train_labels")
            
            # Create DataLoaders from direct data
            if isinstance(train_color_data, np.ndarray):
                train_color_tensor = torch.from_numpy(train_color_data).float()
            elif isinstance(train_color_data, torch.Tensor):
                train_color_tensor = train_color_data.detach().clone()
            else:
                train_color_tensor = train_color_data
                
            if isinstance(train_brightness_data, np.ndarray):
                train_brightness_tensor = torch.from_numpy(train_brightness_data).float()
            elif isinstance(train_brightness_data, torch.Tensor):
                train_brightness_tensor = train_brightness_data.detach().clone()
            else:
                train_brightness_tensor = train_brightness_data
                
            if isinstance(train_labels, np.ndarray):
                train_labels_tensor = torch.from_numpy(train_labels).long()
            elif isinstance(train_labels, torch.Tensor):
                train_labels_tensor = train_labels.detach().clone()
            else:
                train_labels_tensor = train_labels
            
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
        
        # Update scheduler if needed (for schedulers that need actual epoch/batch info)
        if self.scheduler is not None:
            if self.scheduler_type == 'cosine':
                # Reconfigure cosine scheduler with actual epochs
                current_lr = self.optimizer.param_groups[0]['lr']
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=epochs, eta_min=self.min_lr
                )
            elif self.scheduler_type == 'onecycle':
                # Reconfigure OneCycle scheduler with actual steps
                current_lr = self.optimizer.param_groups[0]['lr']
                self.scheduler = optim.lr_scheduler.OneCycleLR(
                    self.optimizer, 
                    max_lr=current_lr,
                    steps_per_epoch=len(train_loader),
                    epochs=epochs,
                    pct_start=0.3
                )
            elif self.scheduler_type == 'step':
                # Reconfigure step scheduler
                self.scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=epochs//3, gamma=0.1
                )
        
        # Print training configuration (after all setup is complete)
        if verbose > 0:
            model_name = self.__class__.__name__
            print(f"🚀 Training {model_name} with {'DataLoader pipeline' if using_dataloaders else 'direct data'}:")
            print(f"   Device: {self.device}")
            print("   Architecture: Dense/Tabular (BasicMultiChannelLayer)")
            print(f"   Mixed precision: {self.use_mixed_precision}")
            if hasattr(self, 'gradient_clip') and self.gradient_clip > 0:
                print(f"   Gradient clipping: {self.gradient_clip}")
            print(f"   Scheduler: {getattr(self, 'scheduler_type', 'None')}")
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
            # Clear any existing tqdm instances and ensure clean state
            try:
                # Force close any existing instances
                tqdm._instances.clear() if hasattr(tqdm, '_instances') else None
                # Flush stdout to clear any remaining output
                sys.stdout.flush()
                sys.stderr.flush()
            except Exception:
                pass  # Ignore any errors in cleaning up
            
            # Create progress bar for training
            epoch_pbar = None
            if verbose == 1:
                epoch_pbar = tqdm(
                    total=len(train_loader), 
                    desc=f"Epoch {epoch+1}/{epochs}",
                    leave=False,  # Don't leave the progress bar after completion
                    dynamic_ncols=True,  # Dynamically adjust width
                    file=sys.stdout,  # Ensure output goes to stdout
                    position=0,  # Position at top
                    ascii=False,  # Use unicode characters
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
                )
                
            # Training phase
            self.train()
            total_loss = 0.0
            train_correct = 0
            train_total = 0
            batch_lr = []
            
            # Training loop
            for batch_idx, data in enumerate(train_loader):
                # Unpack data - handle different formats
                if isinstance(data, (list, tuple)) and len(data) == 3:
                    color_batch, brightness_batch, labels_batch = data
                else:
                    raise ValueError("DataLoader must provide (color, brightness, labels) tuples")
                
                # Move data to device if not already there
                if color_batch.device != self.device:
                    color_batch = color_batch.to(self.device, non_blocking=True)
                if brightness_batch.device != self.device:
                    brightness_batch = brightness_batch.to(self.device, non_blocking=True)
                if labels_batch.device != self.device:
                    labels_batch = labels_batch.to(self.device, non_blocking=True)
                
                # For BaseMultiChannelNetwork, we need to reshape the data from (N, C, H, W) to (N, C*H*W)
                if len(color_batch.shape) > 2:  # If data is in image format (N, C, H, W)
                    color_batch = color_batch.reshape(color_batch.size(0), -1)
                    brightness_batch = brightness_batch.reshape(brightness_batch.size(0), -1)
                
                # Check for NaN inputs
                if torch.isnan(color_batch).any() or torch.isnan(brightness_batch).any():
                    print(f"Warning: NaN detected in batch {batch_idx} inputs. Skipping batch.")
                    continue
                
                # Forward pass
                self.optimizer.zero_grad()
                
                if self.use_mixed_precision and self.scaler is not None:
                    with autocast(device_type='cuda'):
                        outputs = self(color_batch, brightness_batch)
                        loss = self.criterion(outputs, labels_batch)
                    
                    # Backward pass with scaler
                    self.scaler.scale(loss).backward()
                    
                    # Apply gradient clipping if enabled
                    if hasattr(self, 'gradient_clip') and self.gradient_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        safe_clip_grad_norm(self.parameters(), self.gradient_clip)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Standard forward and backward pass
                    outputs = self(color_batch, brightness_batch)
                    loss = self.criterion(outputs, labels_batch)
                    loss.backward()
                    
                    # Apply gradient clipping if enabled
                    if hasattr(self, 'gradient_clip') and self.gradient_clip > 0:
                        safe_clip_grad_norm(self.parameters(), self.gradient_clip)
                    
                    self.optimizer.step()
                
                # Step the scheduler (for batch-level schedulers like OneCycle)
                if self.scheduler_type == 'onecycle' and self.scheduler:
                    self.scheduler.step()
                    batch_lr.append(self.scheduler.get_last_lr()[0])
                
                # Track metrics
                total_loss += loss.item()
                
                # Calculate training accuracy
                with torch.no_grad():  # Ensure no unnecessary memory is retained for gradient calculation
                    _, predicted = outputs.max(1)
                    train_total += labels_batch.size(0)
                    train_correct += predicted.eq(labels_batch).sum().item()
                
                # Explicitly clear unnecessary tensors to help with memory management
                del outputs, loss, predicted
                if self.use_mixed_precision and self.scaler is not None and batch_idx % 50 == 0:
                    torch.cuda.empty_cache()  # Less frequent cache clearing for dense models
                
                # Update progress bar
                if verbose == 1 and epoch_pbar is not None:
                    try:
                        current_lr = self.optimizer.param_groups[0]['lr']
                        epoch_pbar.set_postfix({
                            'loss': f'{total_loss / (batch_idx + 1):.4f}',
                            'acc': f'{train_correct/train_total:.4f}',
                            'lr': f'{current_lr:.6f}'
                        })
                        epoch_pbar.update(1)
                        epoch_pbar.refresh()  # Force refresh to ensure proper display
                    except Exception:
                        # If progress bar fails, continue without it
                        pass
            
            # Close progress bar properly
            if verbose == 1 and epoch_pbar is not None:
                try:
                    epoch_pbar.close()
                    # Add a small delay to ensure proper cleanup
                    sys.stdout.flush()
                except Exception:
                    pass
            
            # Update learning rate scheduler (for epoch-level schedulers)
            if self.scheduler is not None and self.scheduler_type != 'onecycle':
                self.scheduler.step()
            
            # Calculate epoch metrics
            avg_train_loss = total_loss / len(train_loader)
            train_accuracy = train_correct / train_total
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Store training metrics in history
            history['train_loss'].append(avg_train_loss)
            history['train_accuracy'].append(train_accuracy)
            history['learning_rates'].append(batch_lr if batch_lr else [current_lr])
            
            # Validation phase
            if val_loader is not None:
                avg_val_loss, val_accuracy = self._validate(val_loader)
                history['val_loss'].append(avg_val_loss)
                history['val_accuracy'].append(val_accuracy)
                
                # Early stopping check
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    # Save best model state
                    self.best_model_state = self.state_dict().copy()
                    patience_counter = 0
                    if verbose > 0:
                        print(f"✅ New best validation loss: {best_val_loss:.4f}")
                else:
                    patience_counter += 1
                    if verbose > 0:
                        print(f"⏳ No improvement for {patience_counter}/{self.early_stopping_patience} epochs")
                
                # Print epoch summary
                if verbose > 0:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"train_loss: {avg_train_loss:.4f}, train_acc: {train_accuracy:.4f}, "
                          f"val_loss: {avg_val_loss:.4f}, val_acc: {val_accuracy:.4f}, "
                          f"lr: {self.optimizer.param_groups[0]['lr']:.6f}")
                
                # Check for early stopping
                if patience_counter >= self.early_stopping_patience:
                    if verbose > 0:
                        print(f"🛑 Early stopping triggered after {epoch+1} epochs")
                    break
            else:
                # Print epoch summary without validation metrics
                if verbose > 0:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"train_loss: {avg_train_loss:.4f}, train_acc: {train_accuracy:.4f}, "
                          f"lr: {self.optimizer.param_groups[0]['lr']:.6f}")
                
                # Store empty validation metrics for consistency
                history['val_loss'].append(None)
                history['val_accuracy'].append(None)
        
        # Load best model if validation was performed
        if val_loader is not None and hasattr(self, 'best_model_state'):
            self.load_state_dict(self.best_model_state)
            if verbose > 0:
                print("📊 Loaded best model state from early stopping")
        
        # Clear cache after training
        self.device_manager.clear_cache()
        
        return history
    
    def _validate(self, val_loader):
        """
        Validation method for BaseMultiChannelNetwork, with proper input reshaping.
        
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
                
                # Handle different input formats - reshape inputs for BaseMultiChannelNetwork
                if len(color_batch.shape) > 2:  # If data is in image format (N, C, H, W)
                    color_batch = color_batch.reshape(color_batch.size(0), -1)
                    brightness_batch = brightness_batch.reshape(brightness_batch.size(0), -1)
                
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
        
    def evaluate(self, test_color_data=None, test_brightness_data=None, test_labels=None, 
                 test_loader=None, batch_size: int = 32) -> Dict[str, float]:
        """
        Evaluate the model on test data with proper input reshaping.
        
        Supports both direct data arrays and DataLoader inputs.
        
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
    
    def predict(self, color_data: Union[np.ndarray, torch.Tensor, DataLoader], brightness_data: Union[np.ndarray, torch.Tensor] = None, 
                batch_size: int = None) -> np.ndarray:
        """
        Make predictions on new data with proper input reshaping for BaseMultiChannelNetwork.
        
        This implementation overrides the parent method to:
        1. Support both direct data arrays and DataLoader inputs
        2. Handle input reshaping specifically for BaseMultiChannelNetwork (flattening 4D inputs to 2D)
        
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
                    
                    # Reshape inputs for BaseMultiChannelNetwork if needed
                    if len(batch_color.shape) > 2:  # If data is in image format (N, C, H, W)
                        batch_color = batch_color.reshape(batch_color.size(0), -1)
                        batch_brightness = batch_brightness.reshape(batch_brightness.size(0), -1)
                    
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
            
            # Reshape inputs for BaseMultiChannelNetwork if needed
            if len(color_tensor.shape) > 2:  # If data is in image format (N, C, H, W)
                color_tensor = color_tensor.reshape(color_tensor.size(0), -1)
                brightness_tensor = brightness_tensor.reshape(brightness_tensor.size(0), -1)
            
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
        Get prediction probabilities with proper input reshaping for BaseMultiChannelNetwork.
        
        This implementation overrides the parent method to:
        1. Support both direct data arrays and DataLoader inputs
        2. Handle input reshaping specifically for BaseMultiChannelNetwork (flattening 4D inputs to 2D)
        
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
                    
                    # Reshape inputs for BaseMultiChannelNetwork if needed
                    if len(batch_color.shape) > 2:  # If data is in image format (N, C, H, W)
                        batch_color = batch_color.reshape(batch_color.size(0), -1)
                        batch_brightness = batch_brightness.reshape(batch_brightness.size(0), -1)
                    
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
            
            # Reshape inputs for BaseMultiChannelNetwork if needed
            if len(color_tensor.shape) > 2:  # If data is in image format (N, C, H, W)
                color_tensor = color_tensor.reshape(color_tensor.size(0), -1)
                brightness_tensor = brightness_tensor.reshape(brightness_tensor.size(0), -1)
            
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
    

# Factory functions
def base_multi_channel_small(
    *,  # Force keyword-only arguments for clarity
    color_input_size: int,
    brightness_input_size: int,
    num_classes: int = 10, 
    **kwargs
) -> BaseMultiChannelNetwork:
    """
    Create a small BaseMultiChannelNetwork.
    
    Args:
        color_input_size: Input size for color stream
        brightness_input_size: Input size for brightness stream
        num_classes: Number of output classes
        **kwargs: Additional arguments passed to BaseMultiChannelNetwork
    
    Example:
        model = base_multi_channel_small(
            color_input_size=3072, 
            brightness_input_size=1024, 
            num_classes=100
        )
    """
    return BaseMultiChannelNetwork(
        color_input_size=color_input_size,
        brightness_input_size=brightness_input_size,
        hidden_sizes=[256, 128],
        num_classes=num_classes,
        **kwargs
    )


def base_multi_channel_medium(
    *,  # Force keyword-only arguments for clarity
    color_input_size: int,
    brightness_input_size: int,
    num_classes: int = 10,
    **kwargs
) -> BaseMultiChannelNetwork:
    """
    Create a medium BaseMultiChannelNetwork.
    
    Args:
        color_input_size: Input size for color stream
        brightness_input_size: Input size for brightness stream
        num_classes: Number of output classes
        **kwargs: Additional arguments passed to BaseMultiChannelNetwork
    
    Example:
        model = base_multi_channel_medium(
            color_input_size=3072, 
            brightness_input_size=1024, 
            num_classes=100
        )
    """
    return BaseMultiChannelNetwork(
        color_input_size=color_input_size,
        brightness_input_size=brightness_input_size,
        hidden_sizes=[512, 256, 128],
        num_classes=num_classes,
        **kwargs
    )


def base_multi_channel_large(
    *,  # Force keyword-only arguments for clarity
    color_input_size: int,
    brightness_input_size: int,
    num_classes: int = 10,
    **kwargs
) -> BaseMultiChannelNetwork:
    """
    Create a large BaseMultiChannelNetwork.
    
    Args:
        color_input_size: Input size for color stream
        brightness_input_size: Input size for brightness stream
        num_classes: Number of output classes
        **kwargs: Additional arguments passed to BaseMultiChannelNetwork
    
    Example:
        model = base_multi_channel_large(
            color_input_size=3072, 
            brightness_input_size=1024, 
            num_classes=100
        )
    """
    return BaseMultiChannelNetwork(
        color_input_size=color_input_size,
        brightness_input_size=brightness_input_size,
        hidden_sizes=[1024, 512, 256, 128],
        num_classes=num_classes,
        **kwargs
    )
