#!/usr/bin/env python3
"""
Download datasets for Multi-Channel Model training.

This script downloads MNIST and CIFAR-100 datasets to ./data directory.
"""

import os
import sys
import ssl
import torchvision
import torchvision.transforms as transforms

# Handle SSL certificate issues on macOS
ssl._create_default_https_context = ssl._create_unverified_context

def download_datasets(data_path="./data"):
    """Download MNIST and CIFAR-100 datasets."""
    
    # Create data directory if it doesn't exist
    os.makedirs(data_path, exist_ok=True)
    
    print("Downloading datasets...")
    print(f"Data will be saved to: {os.path.abspath(data_path)}")
    
    # Simple transform for downloading
    transform = transforms.ToTensor()
    
    try:
        # Download MNIST
        print("\nDownloading MNIST dataset...")
        torchvision.datasets.MNIST(
            root=data_path, train=True, download=True, transform=transform
        )
        torchvision.datasets.MNIST(
            root=data_path, train=False, download=True, transform=transform
        )
        print("✓ MNIST dataset downloaded successfully")
        
        # Download CIFAR-100
        print("\nDownloading CIFAR-100 dataset...")
        torchvision.datasets.CIFAR100(
            root=data_path, train=True, download=True, transform=transform
        )
        torchvision.datasets.CIFAR100(
            root=data_path, train=False, download=True, transform=transform
        )
        print("✓ CIFAR-100 dataset downloaded successfully")
        
        print(f"\nAll datasets downloaded successfully to {os.path.abspath(data_path)}")
        
        # Print dataset info
        print("\nDataset Information:")
        
        # MNIST info
        mnist_train = torchvision.datasets.MNIST(root=data_path, train=True, transform=transform)
        mnist_test = torchvision.datasets.MNIST(root=data_path, train=False, transform=transform)
        print(f"MNIST - Train: {len(mnist_train)} samples, Test: {len(mnist_test)} samples")
        
        # CIFAR-100 info
        cifar_train = torchvision.datasets.CIFAR100(root=data_path, train=True, transform=transform)
        cifar_test = torchvision.datasets.CIFAR100(root=data_path, train=False, transform=transform)
        print(f"CIFAR-100 - Train: {len(cifar_train)} samples, Test: {len(cifar_test)} samples")
        
    except Exception as e:
        print(f"Error downloading datasets: {e}")
        return False
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download datasets for Multi-Channel Model")
    parser.add_argument("--data_path", default="./data", help="Path to save datasets")
    
    args = parser.parse_args()
    
    success = download_datasets(args.data_path)
    
    if success:
        print("\n" + "="*50)
        print("Ready to train! Use the following commands:")
        print(f"python train_multi_channel.py --dataset mnist --epochs 5 --data_path {args.data_path}")
        print(f"python train_multi_channel.py --dataset cifar100 --epochs 10 --data_path {args.data_path}")
        print("="*50)
    else:
        sys.exit(1)
