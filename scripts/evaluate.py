"""Evaluation script for Multi-Stream Neural Networks."""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.builders.model_factory import create_model
from data_utils import DualChannelDataset, create_dual_channel_dataloaders
from evaluation import evaluate_model, calculate_metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Multi-Stream Neural Network')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data-root', type=str, default='./data', help='Data root directory')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--output-dir', type=str, default='./results', help='Output directory')
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_model(config, checkpoint_path, device):
    """Load model from checkpoint."""
    model = create_model(**config['model'])
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def create_dataset(config, data_root):
    """Create test dataset."""
    dataset_config = config['dataset']
    dataset = DualChannelDataset(
        root=data_root,
        train=False,
        **dataset_config
    )
    return dataset


def main():
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = load_model(config, args.checkpoint, device)
    
    # Create dataset and dataloader
    print("Creating dataset...")
    test_dataset = create_dataset(config, args.data_root)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Evaluate model
    print("Evaluating model...")
    results = evaluate_model(model, test_loader, device)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    print(f"Test Loss: {results['loss']:.4f}")
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, 'evaluation_results.yaml')
    with open(results_path, 'w') as f:
        yaml.dump(results, f)
    print(f"Results saved to: {results_path}")


if __name__ == '__main__':
    main()
