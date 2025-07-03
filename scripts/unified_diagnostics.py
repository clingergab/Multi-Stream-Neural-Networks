"""
Unified comprehensive diagnostic script for Multi-Stream Neural Networks.

This script combines the best features from both diagnose_model.py and 
diagnose_full_resnet.py to provide a complete diagnostic suite.
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
import torch.nn as nn

# Add project root to path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import after sys.path is updated
from src.models.basic_multi_channel.multi_channel_resnet_network import MultiChannelResNetNetwork
from src.models.basic_multi_channel.base_multi_channel_network import BaseMultiChannelNetwork
from src.utils.cifar100_loader import get_cifar100_datasets
from src.transforms.rgb_to_rgbl import RGBtoRGBL
from src.utils.debug_utils import (
    analyze_gradient_flow, 
    check_for_dead_neurons,
    check_pathway_gradients
)


class CIFAR100WithRGBL:
    """Dataset wrapper for RGB to RGBL transformation"""
    def __init__(self, base_dataset, transform=None):
        self.base_dataset = base_dataset
        self.transform = transform or RGBtoRGBL()
        
    def __len__(self):
        return len(self.base_dataset)
        
    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        color, brightness = self.transform(image)
        return color, brightness, label


class UnifiedDiagnostics:
    """Unified diagnostic class for comprehensive model analysis"""
    
    def __init__(self, device='auto', output_dir='diagnostics'):
        self.device = device if device != 'auto' else (
            'cuda' if torch.cuda.is_available() else 
            'mps' if torch.backends.mps.is_available() else 'cpu'
        )
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def load_models(self, resnet_path=None, base_path=None):
        """Load both ResNet and Base models with diagnostic hooks"""
        models = {}
        
        # Load MultiChannelResNetNetwork (reduced)
        models['resnet_reduced'] = MultiChannelResNetNetwork(
            num_classes=100,
            color_input_channels=3,
            brightness_input_channels=1,
            reduce_architecture=True,
            debug_mode=True,
            device=self.device
        ).to(self.device)
        
        # Load MultiChannelResNetNetwork (full)  
        models['resnet_full'] = MultiChannelResNetNetwork(
            num_classes=100,
            color_input_channels=3,
            brightness_input_channels=1,
            reduce_architecture=False,
            debug_mode=True,
            device=self.device
        ).to(self.device)
        
        # Load BaseMultiChannelNetwork
        models['base'] = BaseMultiChannelNetwork(
            num_classes=100,
            color_input_size=3072,  # 32*32*3
            brightness_input_size=1024,  # 32*32*1
        ).to(self.device)
        
        # Load saved weights if available
        if resnet_path and os.path.exists(resnet_path):
            try:
                state_dict = torch.load(resnet_path, map_location=self.device)
                models['resnet_reduced'].load_state_dict(state_dict)
                print(f"Loaded ResNet weights from {resnet_path}")
            except Exception as e:
                print(f"Could not load ResNet weights: {e}")
                
        if base_path and os.path.exists(base_path):
            try:
                state_dict = torch.load(base_path, map_location=self.device)
                models['base'].load_state_dict(state_dict)
                print(f"Loaded Base model weights from {base_path}")
            except Exception as e:
                print(f"Could not load Base model weights: {e}")
        
        return models
    
    def load_data(self, data_dir='data/cifar-100', batch_size=32):
        """Load CIFAR-100 data with RGBL transformation"""
        train_dataset, test_dataset, class_names = get_cifar100_datasets(data_dir=data_dir)
        
        # Create RGBL datasets
        rgbl_train = CIFAR100WithRGBL(train_dataset)
        rgbl_test = CIFAR100WithRGBL(test_dataset)
        
        # Create data loaders
        train_loader = DataLoader(rgbl_train, batch_size=batch_size, shuffle=True, num_workers=2)
        test_loader = DataLoader(rgbl_test, batch_size=batch_size, shuffle=False, num_workers=2)
        
        print(f"Loaded CIFAR-100 with {len(rgbl_train)} train and {len(rgbl_test)} test samples")
        return train_loader, test_loader, class_names
    
    def analyze_architectures(self, models):
        """Compare model architectures and parameter counts"""
        print("\n=== ARCHITECTURE COMPARISON ===")
        
        for name, model in models.items():
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"\n{name.upper()}:")
            print(f"  Total parameters: {total_params:,}")
            print(f"  Trainable parameters: {trainable_params:,}")
            print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    def analyze_weight_distributions(self, models):
        """Analyze weight initialization patterns"""
        print("\n=== WEIGHT INITIALIZATION ANALYSIS ===")
        
        for name, model in models.items():
            weights = []
            biases = []
            
            for param_name, param in model.named_parameters():
                if 'weight' in param_name:
                    weights.extend(param.flatten().detach().cpu().numpy())
                elif 'bias' in param_name:
                    biases.extend(param.flatten().detach().cpu().numpy())
            
            if weights:
                weights = np.array(weights)
                print(f"\n{name.upper()} - Weights:")
                print(f"  Mean: {weights.mean():.6f}, Std: {weights.std():.6f}")
                print(f"  Min: {weights.min():.6f}, Max: {weights.max():.6f}")
                
            if biases:
                biases = np.array(biases)
                print(f"{name.upper()} - Biases:")
                print(f"  Mean: {biases.mean():.6f}, Std: {biases.std():.6f}")
                print(f"  Min: {biases.min():.6f}, Max: {biases.max():.6f}")
    
    def analyze_gradient_flow_all(self, models, data_loader):
        """Analyze gradient flow for all models"""
        print("\n=== GRADIENT FLOW ANALYSIS ===")
        
        for name, model in models.items():
            print(f"\nAnalyzing {name.upper()}...")
            
            # Generate gradient flow visualization
            save_path = os.path.join(self.output_dir, f'{name}_gradient_flow.png')
            analyze_gradient_flow(model, save_path)
            
            # Detailed gradient analysis
            self._detailed_gradient_analysis(model, data_loader, name)
    
    def _detailed_gradient_analysis(self, model, data_loader, model_name):
        """Perform detailed gradient analysis on a single model"""
        model.train()
        criterion = nn.CrossEntropyLoss()
        
        # Get one batch
        color, brightness, labels = next(iter(data_loader))
        color = color.to(self.device)
        brightness = brightness.to(self.device)
        labels = labels.to(self.device)
        
        # Handle different input formats
        if 'base' in model_name:
            color = color.view(color.size(0), -1)
            brightness = brightness.view(brightness.size(0), -1)
        
        # Forward and backward pass
        model.zero_grad()
        outputs = model(color, brightness)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Analyze gradients
        pathway_stats = check_pathway_gradients(model)
        if pathway_stats and 'pathway_balance' in pathway_stats:
            balance = pathway_stats['pathway_balance']
            print(f"  Color pathway: {balance['color_pathway_grad_percent']:.2f}%")
            print(f"  Brightness pathway: {balance['brightness_pathway_grad_percent']:.2f}%")
            print(f"  Pathway ratio: {balance['ratio_color_to_brightness']:.4f}")
    
    def analyze_activations_all(self, models, data_loader):
        """Analyze activation patterns for all models"""
        print("\n=== ACTIVATION ANALYSIS ===")
        
        for name, model in models.items():
            print(f"\nAnalyzing {name.upper()}...")
            self._analyze_single_model_activations(model, data_loader, name)
    
    def _analyze_single_model_activations(self, model, data_loader, model_name):
        """Analyze activations for a single model"""
        model.eval()
        
        # Get one batch
        color, brightness, labels = next(iter(data_loader))
        color = color.to(self.device)
        brightness = brightness.to(self.device)
        
        # Handle different input formats
        if 'base' in model_name:
            color = color.view(color.size(0), -1)
            brightness = brightness.view(brightness.size(0), -1)
        
        # Check for dead neurons
        dead_stats = check_for_dead_neurons(model, (color, brightness))
        
        print("  Dead neuron analysis:")
        for layer_name, stats in dead_stats.items():
            if isinstance(stats, dict):
                print(f"    {layer_name}:")
                print(f"      Color pathway: {stats['color_dead_percent']:.2f}% dead")
                print(f"      Brightness pathway: {stats['brightness_dead_percent']:.2f}% dead")
            else:
                print(f"    {layer_name}: {stats:.2f}% dead")
    
    def run_mini_training(self, model, train_loader, test_loader, model_name, epochs=3):
        """Run mini training loop to test learning capability"""
        print(f"\n=== MINI TRAINING TEST: {model_name.upper()} ===")
        
        # Only run training for full ResNet model (has compile/fit methods)
        if hasattr(model, 'compile') and hasattr(model, 'fit'):
            try:
                model.compile(
                    optimizer='adamw',
                    learning_rate=0.001,
                    weight_decay=0.0001,
                    loss='cross_entropy',
                    gradient_clip=1.0,
                    scheduler='onecycle'
                )
                
                history = model.fit(
                    train_loader=train_loader,
                    val_loader=test_loader,
                    epochs=epochs,
                    early_stopping_patience=epochs + 1,
                    verbose=1
                )
                
                # Plot learning curves
                self._plot_learning_curves(history, model_name)
                
                print(f"Final train accuracy: {history['train_accuracy'][-1]*100:.2f}%")
                print(f"Final val accuracy: {history['val_accuracy'][-1]*100:.2f}%")
                
            except Exception as e:
                print(f"Training failed for {model_name}: {e}")
        else:
            print(f"Model {model_name} doesn't support compile/fit methods")
    
    def _plot_learning_curves(self, history, model_name):
        """Plot and save learning curves"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], 'b-', label='Training')
        plt.plot(history['val_loss'], 'r-', label='Validation')
        plt.title(f'{model_name} - Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(history['train_accuracy'], 'b-', label='Training')
        plt.plot(history['val_accuracy'], 'r-', label='Validation')
        plt.title(f'{model_name} - Accuracy Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f'{model_name}_learning_curves.png')
        plt.savefig(save_path)
        print(f"Learning curves saved to {save_path}")
        plt.close()
    
    def run_complete_diagnosis(self, resnet_path=None, base_path=None, 
                             data_dir='data/cifar-100', batch_size=32):
        """Run complete diagnostic suite"""
        print(f"Running comprehensive diagnostics on device: {self.device}")
        
        # Load models and data
        models = self.load_models(resnet_path, base_path)
        train_loader, test_loader, _ = self.load_data(data_dir, batch_size)
        
        # Run all diagnostic tests
        self.analyze_architectures(models)
        self.analyze_weight_distributions(models)
        self.analyze_gradient_flow_all(models, test_loader)
        self.analyze_activations_all(models, test_loader)
        
        # Run mini training (only for models that support it)
        for name, model in models.items():
            if 'resnet_full' in name:  # Only test full ResNet training
                self.run_mini_training(model, train_loader, test_loader, name)
        
        print("\n=== DIAGNOSTICS COMPLETE ===")
        print(f"Results saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Unified Multi-Stream Neural Network Diagnostics')
    parser.add_argument('--resnet-model', type=str, 
                        default='best_multichannelresnetnetwork_model.pth',
                        help='Path to ResNet model checkpoint')
    parser.add_argument('--base-model', type=str, 
                        default='best_basemultichannelnetwork_model.pth',
                        help='Path to Base model checkpoint')
    parser.add_argument('--data-dir', type=str, default='data/cifar-100',
                        help='Path to CIFAR-100 data directory')
    parser.add_argument('--output-dir', type=str, default='diagnostics',
                        help='Directory to save diagnostic results')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to run on (auto/cuda/mps/cpu)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for analysis')
    
    args = parser.parse_args()
    
    # Run unified diagnostics
    diagnostics = UnifiedDiagnostics(device=args.device, output_dir=args.output_dir)
    diagnostics.run_complete_diagnosis(
        resnet_path=args.resnet_model,
        base_path=args.base_model,
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
