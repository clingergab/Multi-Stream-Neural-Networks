"""
Base Multi-Channel Network using BasicMultiChannelLayer for dense/tabular data.
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, List, Union
from ..base import BaseMultiStreamModel
from ..layers.basic_layers import BasicMultiChannelLayer


class BaseMultiChannelNetwork(BaseMultiStreamModel):
    """
    Base Multi-Channel Network for dense/tabular multi-stream data.
    
    Uses BasicMultiChannelLayer components for fully-connected processing.
    Suitable for:
    - Tabular multi-modal data
    - Dense feature vectors
    - Embeddings
    - Flattened image features
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
        
        # Build network layers
        self._build_network()
        
        # Initialize weights
        self._initialize_weights()
    
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
