"""
Simple data loading utilities for Colab training.
Provides easy-to-use functions for loading and preprocessing common datasets.
"""

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from typing import Tuple, Dict, Any
from ..transforms.rgb_to_rgbl import RGBtoRGBL


def load_mnist(flatten: bool = True, normalize: bool = True) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Load MNIST dataset for multi-channel training.
    
    Args:
        flatten: Whether to flatten images for dense models
        normalize: Whether to normalize pixel values
        
    Returns:
        Tuple of ((train_data, train_labels), (test_data, test_labels))
    """
    # Define transforms
    transform_list = []
    if normalize:
        transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ] + transform_list)
    
    # Download datasets
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Convert to numpy arrays
    train_data = train_dataset.data.numpy().astype(np.float32) / 255.0
    train_labels = train_dataset.targets.numpy()
    test_data = test_dataset.data.numpy().astype(np.float32) / 255.0
    test_labels = test_dataset.targets.numpy()
    
    if flatten:
        train_data = train_data.reshape(train_data.shape[0], -1)
        test_data = test_data.reshape(test_data.shape[0], -1)
    else:
        # Add channel dimension for CNN models
        train_data = np.expand_dims(train_data, axis=1)
        test_data = np.expand_dims(test_data, axis=1)
    
    return (train_data, train_labels), (test_data, test_labels)


def load_cifar10(normalize: bool = True) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Load CIFAR-10 dataset for multi-channel training.
    
    Args:
        normalize: Whether to normalize pixel values
        
    Returns:
        Tuple of ((train_data, train_labels), (test_data, test_labels))
    """
    # Define transforms
    transform_list = [transforms.ToTensor()]
    if normalize:
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    
    transform = transforms.Compose(transform_list)
    
    # Download datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Convert to numpy arrays
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    
    train_data, train_labels = next(iter(train_loader))
    test_data, test_labels = next(iter(test_loader))
    
    return (train_data.numpy(), train_labels.numpy()), (test_data.numpy(), test_labels.numpy())


def prepare_multi_channel_data(rgb_data: np.ndarray, flatten_for_dense: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert RGB data to multi-channel format (RGB + Brightness).
    
    Args:
        rgb_data: RGB data with shape [N, C, H, W] or [N, H, W, C] or [N, features] (if flattened)
        flatten_for_dense: Whether to flatten the output for dense models
        
    Returns:
        Tuple of (color_data, brightness_data)
    """
    # Convert to tensor for processing
    if isinstance(rgb_data, np.ndarray):
        data_tensor = torch.from_numpy(rgb_data).float()
    else:
        data_tensor = rgb_data.float()
    
    # Handle different input formats
    if len(data_tensor.shape) == 2:  # Already flattened [N, features]
        # Assume it's flattened RGB data, reshape to [N, C, H, W]
        n_samples = data_tensor.shape[0]
        total_features = data_tensor.shape[1]
        
        # Try to infer dimensions (assume square images and RGB)
        if total_features % 3 == 0:
            pixels_per_channel = total_features // 3
            side_length = int(np.sqrt(pixels_per_channel))
            if side_length * side_length == pixels_per_channel:
                data_tensor = data_tensor.reshape(n_samples, 3, side_length, side_length)
            else:
                raise ValueError(f"Cannot infer image dimensions from flattened data with {total_features} features")
        else:
            raise ValueError("Flattened data must have features divisible by 3 (RGB channels)")
    
    elif len(data_tensor.shape) == 4 and data_tensor.shape[1] == 1:  # Grayscale [N, 1, H, W]
        # Convert grayscale to RGB by repeating channel
        data_tensor = data_tensor.repeat(1, 3, 1, 1)
    
    elif len(data_tensor.shape) == 4 and data_tensor.shape[-1] == 3:  # [N, H, W, C]
        # Convert from NHWC to NCHW
        data_tensor = data_tensor.permute(0, 3, 1, 2)
    
    # Apply RGB to RGBL transform
    transform = RGBtoRGBL()
    
    if len(data_tensor.shape) == 4:  # Image data [N, C, H, W]
        color_data_list = []
        brightness_data_list = []
        
        for i in range(data_tensor.shape[0]):
            color, brightness = transform(data_tensor[i])
            color_data_list.append(color.unsqueeze(0))
            brightness_data_list.append(brightness.unsqueeze(0))
        
        color_data = torch.cat(color_data_list, dim=0)
        brightness_data = torch.cat(brightness_data_list, dim=0)
        
        if flatten_for_dense:
            # Flatten for dense models
            color_data = color_data.reshape(color_data.shape[0], -1)
            brightness_data = brightness_data.reshape(brightness_data.shape[0], -1)
    
    else:
        raise ValueError(f"Unsupported data shape: {data_tensor.shape}")
    
    return color_data.numpy(), brightness_data.numpy()


def create_sample_data(n_samples: int = 1000, input_size: int = 784, num_classes: int = 10, 
                      for_cnn: bool = False) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], 
                                                      Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Create sample multi-channel data for testing.
    
    Args:
        n_samples: Number of samples to generate
        input_size: Input size for dense models (ignored if for_cnn=True)
        num_classes: Number of classes
        for_cnn: Whether to generate CNN-compatible data
        
    Returns:
        Tuple of ((train_color, train_brightness, train_labels), (test_color, test_brightness, test_labels))
    """
    if for_cnn:
        # Generate image-like data [N, C, H, W]
        train_color = np.random.randn(n_samples, 3, 32, 32).astype(np.float32)
        train_brightness = np.random.randn(n_samples, 1, 32, 32).astype(np.float32)
        test_color = np.random.randn(n_samples // 5, 3, 32, 32).astype(np.float32)
        test_brightness = np.random.randn(n_samples // 5, 1, 32, 32).astype(np.float32)
    else:
        # Generate dense data
        color_size = (input_size * 3) // 4  # 3/4 of features for color
        brightness_size = input_size - color_size  # Remaining for brightness
        
        train_color = np.random.randn(n_samples, color_size).astype(np.float32)
        train_brightness = np.random.randn(n_samples, brightness_size).astype(np.float32)
        test_color = np.random.randn(n_samples // 5, color_size).astype(np.float32)
        test_brightness = np.random.randn(n_samples // 5, brightness_size).astype(np.float32)
    
    # Generate labels
    train_labels = np.random.randint(0, num_classes, size=n_samples)
    test_labels = np.random.randint(0, num_classes, size=n_samples // 5)
    
    return (train_color, train_brightness, train_labels), (test_color, test_brightness, test_labels)


def load_and_prepare_mnist_for_dense() -> Dict[str, Any]:
    """
    Load and prepare MNIST for dense multi-channel models.
    
    Returns:
        Dictionary with prepared data
    """
    (train_data, train_labels), (test_data, test_labels) = load_mnist(flatten=False)
    
    # Convert grayscale to pseudo-RGB for transform
    train_rgb = np.repeat(train_data, 3, axis=1)  # [N, 3, 28, 28]
    test_rgb = np.repeat(test_data, 3, axis=1)
    
    # Prepare multi-channel data
    train_color, train_brightness = prepare_multi_channel_data(train_rgb, flatten_for_dense=True)
    test_color, test_brightness = prepare_multi_channel_data(test_rgb, flatten_for_dense=True)
    
    return {
        'train_color': train_color,
        'train_brightness': train_brightness,
        'train_labels': train_labels,
        'test_color': test_color,
        'test_brightness': test_brightness,
        'test_labels': test_labels,
        'color_input_size': train_color.shape[1],
        'brightness_input_size': train_brightness.shape[1],
        'num_classes': len(np.unique(train_labels))
    }


def load_and_prepare_cifar10_for_cnn() -> Dict[str, Any]:
    """
    Load and prepare CIFAR-10 for CNN multi-channel models.
    
    Returns:
        Dictionary with prepared data
    """
    (train_data, train_labels), (test_data, test_labels) = load_cifar10()
    
    # Prepare multi-channel data
    train_color, train_brightness = prepare_multi_channel_data(train_data, flatten_for_dense=False)
    test_color, test_brightness = prepare_multi_channel_data(test_data, flatten_for_dense=False)
    
    return {
        'train_color': train_color,
        'train_brightness': train_brightness,
        'train_labels': train_labels,
        'test_color': test_color,
        'test_brightness': test_brightness,
        'test_labels': test_labels,
        'color_input_channels': train_color.shape[1],
        'brightness_input_channels': train_brightness.shape[1],
        'num_classes': len(np.unique(train_labels))
    }
