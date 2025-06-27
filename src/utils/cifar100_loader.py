"""
CIFAR-100 data loading utilities.
Provides direct pickle-based loading without torchvision naming conventions.
"""

import pickle
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, List, Union


def load_cifar100_batch(file_path: Union[str, Path]) -> dict:
    """
    Load a CIFAR-100 pickle file.
    
    Args:
        file_path: Path to the pickle file
        
    Returns:
        Dictionary containing the batch data
    """
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    return batch


def load_cifar100_raw(data_dir: Union[str, Path] = "./data/cifar-100") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """
    Load CIFAR-100 data directly from pickle files.
    
    Args:
        data_dir: Path to directory containing train, test, meta files
        
    Returns:
        Tuple of (train_data, train_labels, test_data, test_labels, label_names)
        - train_data: Training images as tensor [N, 3, 32, 32]
        - train_labels: Training labels as tensor [N]
        - test_data: Test images as tensor [N, 3, 32, 32]
        - test_labels: Test labels as tensor [N]
        - label_names: List of class names
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"CIFAR-100 data not found at: {data_path}")
    
    print(f"📁 Loading CIFAR-100 from: {data_path}")
    
    # Load training data
    train_file = data_path / "train"
    if not train_file.exists():
        raise FileNotFoundError(f"Training data file not found: {train_file}")
    
    train_batch = load_cifar100_batch(train_file)
    train_data = train_batch[b'data']
    train_labels = train_batch[b'fine_labels']
    
    # Load test data  
    test_file = data_path / "test"
    if not test_file.exists():
        raise FileNotFoundError(f"Test data file not found: {test_file}")
    
    test_batch = load_cifar100_batch(test_file)
    test_data = test_batch[b'data']
    test_labels = test_batch[b'fine_labels']
    
    # Load metadata
    meta_file = data_path / "meta"
    if not meta_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_file}")
    
    meta_batch = load_cifar100_batch(meta_file)
    label_names = [name.decode('utf-8') for name in meta_batch[b'fine_label_names']]
    
    # Reshape from flat to image format: (N, 3072) -> (N, 3, 32, 32)
    train_data = train_data.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    test_data = test_data.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    
    # Convert to tensors
    train_data = torch.from_numpy(train_data)
    test_data = torch.from_numpy(test_data)
    train_labels = torch.tensor(train_labels)
    test_labels = torch.tensor(test_labels)
    
    print(f"✅ Loaded CIFAR-100:")
    print(f"   Training: {train_data.shape}, labels: {len(train_labels)}")
    print(f"   Test: {test_data.shape}, labels: {len(test_labels)}")
    print(f"   Classes: {len(label_names)}")
    
    return train_data, train_labels, test_data, test_labels, label_names


def load_cifar100_numpy(data_dir: Union[str, Path] = "./data/cifar-100") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Load CIFAR-100 data directly from pickle files, returning numpy arrays.
    
    Args:
        data_dir: Path to directory containing train, test, meta files
        
    Returns:
        Tuple of (train_data, train_labels, test_data, test_labels, label_names)
        - train_data: Training images as numpy array [N, 3, 32, 32]
        - train_labels: Training labels as numpy array [N]
        - test_data: Test images as numpy array [N, 3, 32, 32]
        - test_labels: Test labels as numpy array [N]
        - label_names: List of class names
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"CIFAR-100 data not found at: {data_path}")
    
    print(f"📁 Loading CIFAR-100 from: {data_path}")
    
    # Load training data
    train_batch = load_cifar100_batch(data_path / "train")
    train_data = train_batch[b'data']
    train_labels = train_batch[b'fine_labels']
    
    # Load test data  
    test_batch = load_cifar100_batch(data_path / "test")
    test_data = test_batch[b'data']
    test_labels = test_batch[b'fine_labels']
    
    # Load metadata
    meta_batch = load_cifar100_batch(data_path / "meta")
    label_names = [name.decode('utf-8') for name in meta_batch[b'fine_label_names']]
    
    # Reshape from flat to image format: (N, 3072) -> (N, 3, 32, 32)
    train_data = train_data.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    test_data = test_data.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    
    # Convert labels to numpy arrays
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    
    print(f"✅ Loaded CIFAR-100:")
    print(f"   Training: {train_data.shape}, labels: {len(train_labels)}")
    print(f"   Test: {test_data.shape}, labels: {len(test_labels)}")
    print(f"   Classes: {len(label_names)}")
    
    return train_data, train_labels, test_data, test_labels, label_names


class SimpleDataset:
    """
    Simple dataset wrapper for compatibility with PyTorch training loops.
    """
    def __init__(self, data: torch.Tensor, labels: torch.Tensor):
        """
        Initialize dataset.
        
        Args:
            data: Input data tensor
            labels: Label tensor
        """
        self.data = data
        self.labels = labels
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset."""
        return self.data[idx], self.labels[idx]


def get_cifar100_datasets(data_dir: Union[str, Path] = "./data/cifar-100") -> Tuple['SimpleDataset', 'SimpleDataset', List[str]]:
    """
    Load CIFAR-100 data and return as dataset objects.
    
    Args:
        data_dir: Path to directory containing train, test, meta files
        
    Returns:
        Tuple of (train_dataset, test_dataset, label_names)
    """
    train_data, train_labels, test_data, test_labels, label_names = load_cifar100_raw(data_dir)
    
    train_dataset = SimpleDataset(train_data, train_labels)
    test_dataset = SimpleDataset(test_data, test_labels)
    
    print("✅ CIFAR-100 datasets ready:")
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Test samples: {len(test_dataset)}")
    print(f"   Classes: {len(label_names)}")
    print("💡 No torchvision naming conventions needed - loaded directly from pickle files!")
    
    return train_dataset, test_dataset, label_names


def create_validation_split(train_dataset: 'SimpleDataset', val_split: float = 0.1) -> Tuple['SimpleDataset', 'SimpleDataset']:
    """
    Create a validation split from the training dataset.
    
    Args:
        train_dataset: Training dataset to split
        val_split: Fraction of training data to use for validation
        
    Returns:
        Tuple of (new_train_dataset, val_dataset)
    """
    num_train = len(train_dataset)
    num_val = int(num_train * val_split)
    
    # Create random indices for train/val split
    indices = torch.randperm(num_train)
    train_indices = indices[num_val:]
    val_indices = indices[:num_val]
    
    # Split the data
    train_data = train_dataset.data[train_indices]
    train_labels = train_dataset.labels[train_indices]
    val_data = train_dataset.data[val_indices]
    val_labels = train_dataset.labels[val_indices]
    
    # Create new dataset objects
    new_train_dataset = SimpleDataset(train_data, train_labels)
    val_dataset = SimpleDataset(val_data, val_labels)
    
    print(f"📊 Created validation split: {val_split*100:.1f}% ({num_val} samples)")
    print(f"   New training: {len(new_train_dataset)} samples")
    print(f"   Validation: {len(val_dataset)} samples")
    
    return new_train_dataset, val_dataset


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


if __name__ == "__main__":
    """Test the CIFAR-100 loader."""
    print("🔍 Testing CIFAR-100 loader...")
    
    try:
        train_dataset, test_dataset, class_names = get_cifar100_datasets()
        
        # Test a sample
        sample = train_dataset[0]
        image, label = sample
        print("\n🔍 Sample verification:")
        print(f"   Image shape: {image.shape}")
        print(f"   Label: {label} ({class_names[label]})")
        print(f"   Image range: [{image.min():.3f}, {image.max():.3f}]")
        
        print("\n✅ CIFAR-100 loader test passed!")
        
    except Exception as e:
        print(f"❌ CIFAR-100 loader test failed: {e}")
