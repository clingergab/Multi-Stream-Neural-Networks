"""
Dataset utilities for multi-stream neural network processing.

This module provides efficient CIFAR-100 data loading optimized for 
multi-stream neural networks with minimal memory overhead.

Main function:
- load_cifar100_data: Core data loading with flexible return types
"""

import pickle
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, List, Union, Optional, Literal


def _load_cifar100_pickle(file_path: Path) -> dict:
    """
    Internal helper to load a CIFAR-100 pickle file.
    
    Args:
        file_path: Path to the pickle file
        
    Returns:
        Dictionary containing the batch data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        IOError: If the file can't be read
    """
    if not file_path.exists():
        raise FileNotFoundError(f"CIFAR-100 file not found: {file_path}")
    
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f, encoding='bytes')
    except Exception as e:
        raise IOError(f"Failed to load CIFAR-100 pickle file {file_path}: {e}")


def load_cifar100_data(
    data_dir: Union[str, Path] = "./data/cifar-100", 
    return_type: Literal['torch', 'numpy'] = 'torch',
    normalize: bool = True,
    verbose: bool = True
) -> Tuple[Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray], 
           Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]]:
    """
    Load CIFAR-100 data directly from pickle files.
    
    This function efficiently loads and preprocesses CIFAR-100 data with minimal memory overhead.
    Data is reshaped to proper image format (3, 32, 32) and optionally normalized.
    
    Args:
        data_dir: Path to directory containing train, test, meta files
        return_type: Return format - 'torch' for PyTorch tensors, 'numpy' for NumPy arrays
        normalize: Whether to normalize pixel values from [0, 255] to [0, 1] range
        verbose: Whether to print loading progress and statistics
        
    Returns:
        Tuple of (train_data, train_labels, test_data, test_labels)
        - train_data: Training images [N, 3, 32, 32] 
        - train_labels: Training labels [N]
        - test_data: Test images [N, 3, 32, 32]
        - test_labels: Test labels [N]
        
    Note:
        Class names are available as CIFAR100_FINE_LABELS constant in this module.
        
    Raises:
        FileNotFoundError: If data directory or required files don't exist
        ValueError: If return_type is not 'torch' or 'numpy'
        IOError: If files can't be loaded
    """
    if return_type not in ('torch', 'numpy'):
        raise ValueError(f"return_type must be 'torch' or 'numpy', got '{return_type}'")
    
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"CIFAR-100 data directory not found: {data_path}")
    
    if verbose:
        print(f"📁 Loading CIFAR-100 from: {data_path}")
    
    # Load all required files
    train_batch = _load_cifar100_pickle(data_path / "train")
    test_batch = _load_cifar100_pickle(data_path / "test") 
    
    # Extract data efficiently
    train_data = train_batch[b'data']
    train_labels = train_batch[b'fine_labels']
    test_data = test_batch[b'data']
    test_labels = test_batch[b'fine_labels']
    
    # Reshape and optionally normalize for efficiency
    train_data = train_data.reshape(-1, 3, 32, 32).astype(np.float32)
    test_data = test_data.reshape(-1, 3, 32, 32).astype(np.float32)
    
    if normalize:
        train_data = train_data / 255.0
        test_data = test_data / 255.0
    
    # Convert to requested format
    if return_type == 'torch':
        train_data = torch.from_numpy(train_data)
        test_data = torch.from_numpy(test_data)
        train_labels = torch.tensor(train_labels, dtype=torch.long)
        test_labels = torch.tensor(test_labels, dtype=torch.long)
    else:  # numpy
        train_labels = np.array(train_labels, dtype=np.int64)
        test_labels = np.array(test_labels, dtype=np.int64)
    
    if verbose:
        print(f"✅ Loaded CIFAR-100 ({return_type} format):")
        print(f"   Training: {train_data.shape}, labels: {len(train_labels)}")
        print(f"   Test: {test_data.shape}, labels: {len(test_labels)}")
    
    return train_data, train_labels, test_data, test_labels


# CIFAR-100 class names for reference
CIFAR100_FINE_LABELS = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]


def load_cifar100_numpy(data_dir: Union[str, Path] = "./data/cifar-100", normalize: bool = True, verbose: bool = True):
    """
    Backward compatibility wrapper for load_cifar100_data with numpy return type.
    
    Args:
        data_dir: Path to CIFAR-100 data directory
        normalize: Whether to normalize pixel values to [0, 1]
        verbose: Whether to print loading progress
        
    Returns:
        Tuple of (train_data, train_labels, test_data, test_labels, label_names)
        All arrays are numpy arrays.
    """
    train_data, train_labels, test_data, test_labels = load_cifar100_data(
        data_dir=data_dir,
        return_type='numpy',
        normalize=normalize,
        verbose=verbose
    )
    
    return train_data, train_labels, test_data, test_labels, CIFAR100_FINE_LABELS

