"""
Configuration management for Multi-Stream Neural Networks
"""

import yaml
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from omegaconf import OmegaConf, DictConfig


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    model_type: str = "direct_mixing"
    variant: str = "scalar"
    input_size: tuple = (4, 32, 32)  # RGB + Luminance
    hidden_size: int = 512
    num_classes: int = 10
    num_layers: int = 3
    
    # Integration parameters
    alpha_init: float = 1.0
    beta_init: float = 1.0
    gamma_init: float = 0.2
    
    # Regularization
    min_weight_val: float = 0.01
    dropout_rate: float = 0.1


@dataclass
class TrainingConfig:
    """Configuration for training process."""
    batch_size: int = 128
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    epochs: int = 100
    optimizer: str = "adam"
    scheduler: str = "cosine"
    
    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4
    
    # Checkpointing
    save_every: int = 10
    checkpoint_dir: str = "checkpoints"


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    dataset: str = "cifar10"
    data_dir: str = "data"
    num_workers: int = 4
    pin_memory: bool = True
    
    # Transforms
    image_size: int = 32
    normalize: bool = True
    augmentation: bool = True
    
    # Split ratios
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    name: str = "msnn_experiment"
    output_dir: str = "outputs"
    seed: int = 42
    device: str = "auto"  # auto, cpu, cuda
    
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # Logging
    log_level: str = "INFO"
    wandb_project: Optional[str] = None
    tensorboard: bool = True


class ConfigManager:
    """Manages configuration loading and merging."""
    
    @staticmethod
    def load_config(config_path: str) -> DictConfig:
        """Load configuration from YAML file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return OmegaConf.create(config_dict)
    
    @staticmethod
    def save_config(config: DictConfig, save_path: str):
        """Save configuration to YAML file."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            OmegaConf.save(config, f)
    
    @staticmethod
    def merge_configs(*configs: DictConfig) -> DictConfig:
        """Merge multiple configurations with later ones taking precedence."""
        merged = OmegaConf.create({})
        
        for config in configs:
            merged = OmegaConf.merge(merged, config)
        
        return merged
    
    @staticmethod
    def create_default_config() -> ExperimentConfig:
        """Create default experiment configuration."""
        return ExperimentConfig()
    
    @staticmethod
    def validate_config(config: DictConfig) -> bool:
        """Validate configuration structure and values."""
        required_sections = ['model', 'training', 'data']
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required config section: {section}")
        
        # Validate model config
        if 'model_type' not in config.model:
            raise ValueError("Missing model_type in model config")
        
        # Validate training config
        if config.training.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        if config.training.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        
        # Validate data config
        if not (0 < config.data.train_ratio < 1):
            raise ValueError("Train ratio must be between 0 and 1")
        
        return True


def load_model_config(config_name: str) -> DictConfig:
    """Load a specific model configuration."""
    config_path = f"configs/model_configs/{config_name}.yaml"
    return ConfigManager.load_config(config_path)


def load_experiment_config(config_name: str) -> DictConfig:
    """Load a specific experiment configuration."""
    config_path = f"configs/experiment_configs/{config_name}.yaml"
    return ConfigManager.load_config(config_path)
