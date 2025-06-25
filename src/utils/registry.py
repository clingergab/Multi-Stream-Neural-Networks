"""Model registry for easy model creation and management."""

import torch
import torch.nn as nn
from typing import Dict, Type, Any, Optional
from pathlib import Path


class ModelRegistry:
    """Registry for model classes and factory functions."""
    
    def __init__(self):
        self._models: Dict[str, Type[nn.Module]] = {}
        self._configs: Dict[str, Dict] = {}
    
    def register(self, name: str, model_class: Type[nn.Module], default_config: Optional[Dict] = None):
        """Register a model class with optional default configuration."""
        self._models[name] = model_class
        if default_config:
            self._configs[name] = default_config
    
    def create_model(self, name: str, config: Optional[Dict] = None, **kwargs) -> nn.Module:
        """Create model instance from registered class."""
        if name not in self._models:
            raise ValueError(f"Model '{name}' not registered. Available models: {list(self._models.keys())}")
        
        model_class = self._models[name]
        
        # Merge default config with provided config
        final_config = {}
        if name in self._configs:
            final_config.update(self._configs[name])
        if config:
            final_config.update(config)
        final_config.update(kwargs)
        
        return model_class(**final_config)
    
    def list_models(self) -> list:
        """List all registered model names."""
        return list(self._models.keys())
    
    def get_model_class(self, name: str) -> Type[nn.Module]:
        """Get the model class for a given name."""
        if name not in self._models:
            raise ValueError(f"Model '{name}' not registered")
        return self._models[name]


# Global registry instance
_registry = ModelRegistry()


def register_model(name: str, model_class: Type[nn.Module], default_config: Optional[Dict] = None):
    """Register a model class globally."""
    _registry.register(name, model_class, default_config)


def get_model(name: str, config: Optional[Dict] = None, **kwargs) -> nn.Module:
    """Create a model instance from the global registry."""
    return _registry.create_model(name, config, **kwargs)


def list_available_models() -> list:
    """List all available models in the global registry."""
    return _registry.list_models()


# Register common models when module is imported
def _register_builtin_models():
    """Register built-in model types."""
    try:
        # Import model classes
        from ..models.continuous_integration.direct_mixing import (
            ScalarMixingModel,
            ChannelAdaptiveMixingModel, 
            DynamicMixingModel,
            SpatialAdaptiveMixingModel
        )
        from ..models.continuous_integration.concat_linear import ConcatLinearModel
        from ..models.continuous_integration.attention_based import AttentionBasedModel
        from ..models.continuous_integration.neural_processing import NeuralProcessingModel
        from ..models.basic_multi_channel import (
            BaseMultiChannelNetwork,
            MultiChannelResNetNetwork
        )
        
        # Register models
        register_model('scalar_mixing', ScalarMixingModel)
        register_model('channel_mixing', ChannelAdaptiveMixingModel)
        register_model('dynamic_mixing', DynamicMixingModel)
        register_model('spatial_adaptive_mixing', SpatialAdaptiveMixingModel)
        register_model('concat_linear', ConcatLinearModel)
        register_model('attention_based', AttentionBasedModel)
        register_model('neural_processing', NeuralProcessingModel)
        register_model('base_multi_channel', BaseMultiChannelNetwork)
        register_model('multi_channel_resnet', MultiChannelResNetNetwork)
        
    except ImportError:
        # Models not available yet - will be registered when imported
        pass


# Auto-register on import
_register_builtin_models()
