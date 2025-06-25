"""Model factory for creating basic multi-channel MSNN models."""

from ..basic_multi_channel import (
    BaseMultiChannelNetwork,
    MultiChannelResNetNetwork,
    base_multi_channel_small,
    base_multi_channel_medium,
    base_multi_channel_large,
    multi_channel_resnet18,
    multi_channel_resnet34,
    multi_channel_resnet50
)


MODEL_REGISTRY = {
    # Dense/tabular multi-channel models
    'base_multi_channel': BaseMultiChannelNetwork,
    'base_multi_channel_small': base_multi_channel_small,
    'base_multi_channel_medium': base_multi_channel_medium,
    'base_multi_channel_large': base_multi_channel_large,
    
    # ResNet-based multi-channel models
    'multi_channel_resnet': MultiChannelResNetNetwork,
    'multi_channel_resnet18': multi_channel_resnet18,
    'multi_channel_resnet34': multi_channel_resnet34,
    'multi_channel_resnet50': multi_channel_resnet50,
}


def create_model(model_type, **kwargs):
    """
    Create a model instance.
    
    Args:
        model_type (str): Type of model to create
        **kwargs: Model-specific arguments
        
    Returns:
        nn.Module: Model instance
    """
    if model_type not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model type '{model_type}'. Available: {available}")
    
    model_class = MODEL_REGISTRY[model_type]
    return model_class(**kwargs)


def list_available_models():
    """List all available model types."""
    return list(MODEL_REGISTRY.keys())
