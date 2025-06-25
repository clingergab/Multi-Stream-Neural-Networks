#!/usr/bin/env python3
"""Data preparation script."""

import argparse
import os
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))

from datasets.derived_brightness import DerivedBrightnessDataset
from data_utils.data_helpers import DataHelper, split_dataset


def main():
    parser = argparse.ArgumentParser(description='Prepare datasets for training')
    parser.add_argument('--data-root', type=str, required=True, help='Data root directory')
    parser.add_argument('--dataset-type', type=str, default='cifar10', 
                       choices=['cifar10', 'cifar100', 'imagenet'])
    parser.add_argument('--download', action='store_true', help='Download dataset if not exists')
    parser.add_argument('--compute-stats', action='store_true', help='Compute dataset statistics')
    
    args = parser.parse_args()
    
    print(f"Preparing {args.dataset_type} dataset...")
    
    # Create dataset
    dataset = DerivedBrightnessDataset(
        root=args.data_root,
        train=True,
        download=args.download,
        dataset_type=args.dataset_type
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Split dataset
    train_set, val_set, test_set = split_dataset(dataset)
    print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")
    
    if args.compute_stats:
        print("Computing dataset statistics...")
        from torch.utils.data import DataLoader
        
        train_loader = DataLoader(train_set, batch_size=32, shuffle=False)
        stats = DataHelper.calculate_mean_std(train_loader)
        
        print("Dataset Statistics:")
        print(f"Color mean: {stats['color_mean']}")
        print(f"Color std: {stats['color_std']}")
        print(f"Brightness mean: {stats['brightness_mean']}")
        print(f"Brightness std: {stats['brightness_std']}")


if __name__ == '__main__':
    main()