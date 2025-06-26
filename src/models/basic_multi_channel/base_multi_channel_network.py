"""
Base Multi-Channel Network using BasicMultiChannelLayer for dense/tabular data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast
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
        self.scaler = GradScaler() if self.use_mixed_precision else None
        
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
            self.classifier = nn.Linear(final_size * 2, self.num_classes, bias=True)
            self.multi_channel_classifier = None  # Not used in shared mode
        else:
            # Legacy: Separate classifiers for each stream
            self.classifier = None  # Not used in separate mode
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
    
    def forward(self, color_input: torch.Tensor, brightness_input: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the multi-channel network.
        
        Args:
            color_input: Color features tensor [batch_size, input_size]
            brightness_input: Brightness features tensor [batch_size, input_size]
            
        Returns:
            If use_shared_classifier=True: Single output tensor [batch_size, num_classes]
            If use_shared_classifier=False: Tuple of (color_logits, brightness_logits) [batch_size, num_classes] each
        """
        color_x, brightness_x = color_input, brightness_input
        
        # Process through layers
        for layer in self.layers:
            if isinstance(layer, BasicMultiChannelLayer):
                color_x, brightness_x = layer(color_x, brightness_x)
            elif isinstance(layer, nn.Dropout):
                # Apply dropout to both streams
                color_x = layer(color_x)
                brightness_x = layer(brightness_x)
        
        # Final classification
        if self.use_shared_classifier:
            # Concatenate features and pass through shared classifier
            fused_features = torch.cat([color_x, brightness_x], dim=1)
            logits = self.classifier(fused_features)
            return logits
        else:
            # Legacy: Separate classifiers
            color_logits, brightness_logits = self.multi_channel_classifier(color_x, brightness_x)
            return color_logits, brightness_logits
    
    def forward_combined(self, color_input: torch.Tensor, brightness_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with combined output for standard classification.
        
        Args:
            color_input: Color features tensor
            brightness_input: Brightness features tensor
            
        Returns:
            Combined classification logits [batch_size, num_classes]
        """
        if self.use_shared_classifier:
            # Already combined in the forward method
            return self.forward(color_input, brightness_input)
        else:
            # Legacy: Add separate outputs
            color_logits, brightness_logits = self.forward(color_input, brightness_input)
            return color_logits + brightness_logits
    
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
        color_x, brightness_x = color_input, brightness_input
        
        # Process through all layers except classifier
        for layer in self.layers:
            if isinstance(layer, BasicMultiChannelLayer):
                color_x, brightness_x = layer(color_x, brightness_x)
            elif isinstance(layer, nn.Dropout):
                color_x = layer(color_x)
                brightness_x = layer(brightness_x)
        
        if self.use_shared_classifier:
            # Return concatenated features
            return torch.cat([color_x, brightness_x], dim=1)
        else:
            # Return separate features
            return color_x, brightness_x
        
        return color_x, brightness_x
    
    @property
    def fusion_type(self) -> str:
        """Return the type of fusion used by this model."""
        return "shared_classifier" if self.use_shared_classifier else "separate_classifiers"
    
    def get_classifier_info(self) -> Dict:
        """Get information about the classifier architecture."""
        if self.use_shared_classifier:
            return {
                'type': 'shared',
                'input_size': self.classifier.in_features,
                'output_size': self.classifier.out_features,
                'parameters': sum(p.numel() for p in self.classifier.parameters())
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
            # For shared classifier, analyze the concatenated input weights
            classifier_weights = self.classifier.weight.data
            feature_size = classifier_weights.shape[1] // 2
            color_weights = classifier_weights[:, :feature_size]
            brightness_weights = classifier_weights[:, feature_size:]
            
            color_norm = torch.norm(color_weights).item()
            brightness_norm = torch.norm(brightness_weights).item()
            total_norm = color_norm + brightness_norm + 1e-8
            
            return {
                'color_pathway': color_norm / total_norm,
                'brightness_pathway': brightness_norm / total_norm,
                'pathway_ratio': color_norm / (brightness_norm + 1e-8),
                'fusion_type': 'shared_classifier'
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
            # Single progress bar for the entire epoch
            total_batches = len(train_loader) + (len(val_loader) if val_loader is not None else 0)
            if verbose == 1:
                epoch_pbar = tqdm(total=total_batches, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            
            # Training phase
            self.train()
            total_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (batch_color, batch_brightness, batch_labels) in enumerate(train_loader):
                optimizer.zero_grad()
                
                if self.use_mixed_precision:
                    with autocast():
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
                        'T_loss': f'{total_loss/(batch_idx+1):.4f}',
                        'T_acc': f'{train_acc:.4f}',
                        'V_loss': 'N/A',
                        'V_acc': 'N/A'
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
                            with autocast():
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
                        
                        # Update progress bar with both training and validation metrics
                        if verbose == 1:
                            val_acc = val_correct / val_total
                            epoch_pbar.set_postfix({
                                'T_loss': f'{avg_train_loss:.4f}',
                                'T_acc': f'{train_accuracy:.4f}',
                                'V_loss': f'{total_val_loss/(batch_idx+1):.4f}',
                                'V_acc': f'{val_acc:.4f}'
                            })
                            epoch_pbar.update(1)
                
                avg_val_loss = total_val_loss / len(val_loader)
                val_accuracy = val_correct / val_total
                
                # Final update with complete metrics
                if verbose == 1:
                    epoch_pbar.set_postfix({
                        'T_loss': f'{avg_train_loss:.4f}',
                        'T_acc': f'{train_accuracy:.4f}',
                        'V_loss': f'{avg_val_loss:.4f}',
                        'V_acc': f'{val_accuracy:.4f}'
                    })
            else:
                # Training only - final update
                if verbose == 1:
                    epoch_pbar.set_postfix({
                        'T_loss': f'{avg_train_loss:.4f}',
                        'T_acc': f'{train_accuracy:.4f}',
                        'V_loss': 'N/A',
                        'V_acc': 'N/A'
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
    
    def save_model(self, file_path: str = None):
        """Save the model parameters to a file."""
        if file_path is None:
            file_path = "best_model.pth"
        torch.save(self.state_dict(), file_path)
        print(f"Model parameters saved to {file_path}.")
    
    def load_model(self, file_path: str = None):
        """Load model parameters from a file."""
        if file_path is None:
            file_path = "best_model.pth"
        self.load_state_dict(torch.load(file_path, map_location=self.device))
        self.to(self.device)
        print(f"Model parameters loaded from {file_path}.")
    
    def predict(self, color_data: Union[np.ndarray, torch.Tensor], brightness_data: Union[np.ndarray, torch.Tensor], 
                batch_size: int = None) -> np.ndarray:
        """
        Make predictions on new data with GPU optimizations.
        
        Args:
            color_data: Color input data
            brightness_data: Brightness input data
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
        
        # Convert to tensors if needed
        if isinstance(color_data, np.ndarray):
            color_tensor = torch.tensor(color_data, dtype=torch.float32)
        else:
            color_tensor = color_data
            
        if isinstance(brightness_data, np.ndarray):
            brightness_tensor = torch.tensor(brightness_data, dtype=torch.float32)
        else:
            brightness_tensor = brightness_data
        
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
                    with autocast():
                        outputs = self(batch_color, batch_brightness)
                else:
                    outputs = self(batch_color, batch_brightness)
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.cpu().numpy())
        
        return np.array(predictions)
    
    def predict_proba(self, color_data: Union[np.ndarray, torch.Tensor], brightness_data: Union[np.ndarray, torch.Tensor], 
                      batch_size: int = 32) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            color_data: Color input data
            brightness_data: Brightness input data
            batch_size: Batch size for prediction
            
        Returns:
            Prediction probabilities
        """
        self.eval()
        
        # Convert to tensors if needed
        if isinstance(color_data, np.ndarray):
            color_tensor = torch.tensor(color_data, dtype=torch.float32)
        else:
            color_tensor = color_data
            
        if isinstance(brightness_data, np.ndarray):
            brightness_tensor = torch.tensor(brightness_data, dtype=torch.float32)
        else:
            brightness_tensor = brightness_data
        
        # Move to device
        color_tensor = color_tensor.to(self.device)
        brightness_tensor = brightness_tensor.to(self.device)
        
        # Create dataset and loader
        dataset = TensorDataset(color_tensor, brightness_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        probabilities = []
        with torch.no_grad():
            for batch_color, batch_brightness in loader:
                outputs = self(batch_color, batch_brightness)
                probs = torch.softmax(outputs, dim=1)
                probabilities.extend(probs.cpu().numpy())
        
        return np.array(probabilities)
    
    def evaluate(self, test_color_data: Union[np.ndarray, torch.Tensor], 
                 test_brightness_data: Union[np.ndarray, torch.Tensor], 
                 test_labels: Union[np.ndarray, torch.Tensor], batch_size: int = 32) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            test_color_data: Test color data
            test_brightness_data: Test brightness data
            test_labels: Test labels
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.eval()
        
        # Convert to tensors if needed and move to device
        if isinstance(test_color_data, np.ndarray):
            color_tensor = torch.tensor(test_color_data, dtype=torch.float32).to(self.device)
        else:
            color_tensor = test_color_data.to(self.device)
            
        if isinstance(test_brightness_data, np.ndarray):
            brightness_tensor = torch.tensor(test_brightness_data, dtype=torch.float32).to(self.device)
        else:
            brightness_tensor = test_brightness_data.to(self.device)
            
        if isinstance(test_labels, np.ndarray):
            labels_tensor = torch.tensor(test_labels, dtype=torch.long).to(self.device)
        else:
            labels_tensor = test_labels.to(self.device)
        
        # Create dataset and loader
        dataset = TensorDataset(color_tensor, brightness_tensor, labels_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_color, batch_brightness, batch_labels in loader:
                outputs = self(batch_color, batch_brightness)
                loss = criterion(outputs, batch_labels)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        
        accuracy = correct / total
        avg_loss = total_loss / len(loader)
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'correct': correct,
            'total': total
        }
    
    def compile(self, optimizer: str = 'adam', learning_rate: float = 0.001, 
                loss: str = 'cross_entropy', metrics: List[str] = None):
        """
        Compile the model with optimizer and loss function (Keras-like API).
        
        Args:
            optimizer: Optimizer name ('adam', 'sgd', 'rmsprop')
            learning_rate: Learning rate
            loss: Loss function name ('cross_entropy')
            metrics: List of metrics to track
        """
        if optimizer.lower() == 'adam':
            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        elif optimizer.lower() == 'sgd':
            self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        elif optimizer.lower() == 'rmsprop':
            self.optimizer = optim.RMSprop(self.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
        
        if loss.lower() == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss function: {loss}")
        
        self.metrics = metrics or ['accuracy']
        self.is_compiled = True
        
        print(f"Model compiled with {optimizer} optimizer, {loss} loss, learning rate: {learning_rate}")
        

# Factory functions
def base_multi_channel_small(input_size: int, num_classes: int = 10, **kwargs) -> BaseMultiChannelNetwork:
    """Create a small BaseMultiChannelNetwork."""
    return BaseMultiChannelNetwork(
        input_size=input_size,
        hidden_sizes=[256, 128],
        num_classes=num_classes,
        **kwargs
    )


def base_multi_channel_medium(input_size: int, num_classes: int = 10, **kwargs) -> BaseMultiChannelNetwork:
    """Create a medium BaseMultiChannelNetwork."""
    return BaseMultiChannelNetwork(
        input_size=input_size,
        hidden_sizes=[512, 256, 128],
        num_classes=num_classes,
        **kwargs
    )


def base_multi_channel_large(input_size: int, num_classes: int = 10, **kwargs) -> BaseMultiChannelNetwork:
    """Create a large BaseMultiChannelNetwork."""
    return BaseMultiChannelNetwork(
        input_size=input_size,
        hidden_sizes=[1024, 512, 256, 128],
        num_classes=num_classes,
        **kwargs
    )
