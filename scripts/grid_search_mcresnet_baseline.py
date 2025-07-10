#!/usr/bin/env python3
"""
MCResNet Baseline Grid Search Script

This script performs a baseline hyperparameter grid search for MCResNet using:
- 64 total combinations (2^6)
- 15 epochs per combination
- Parameters: learning_rate, batch_size, optimizer, weight_decay, scheduler, transform

Estimated runtime: 5-8 hours depending on hardware.
"""

import itertools
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

from sklearn.model_selection import train_test_split
import torch
from torchvision import transforms

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from data_utils.dataset_utils import load_cifar100_data
    from data_utils.rgb_to_rgbl import RGBtoRGBL
    from models2.multi_channel.mc_resnet import mc_resnet50
    from data_utils.dual_channel_dataset import create_dual_channel_dataloaders
except ImportError as e:
    print(f"Error importing modules: {e}")
    import traceback
    traceback.print_exc()
    print("Make sure you're running from the project root directory")
    sys.exit(1)


class MCResNetBaselineGridSearch:
    """Baseline grid search for MCResNet hyperparameters."""
    
    def __init__(self, 
                 results_dir: str = "mcresnet_baseline_grid_search",
                 device: Optional[str] = None):
        """
        Initialize grid search.
        
        Args:
            results_dir: Directory to save results
            device: Device to use for training
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Auto-detect device if not provided
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
                print(f"CUDA available: {torch.cuda.get_device_name()}")
            elif torch.backends.mps.is_available():
                self.device = "mps"
                print("MPS (Apple Silicon) available")
            else:
                self.device = "cpu"
                print("Using CPU")
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
    
    def define_parameter_grid(self) -> Dict[str, List[Any]]:
        """Define the baseline hyperparameter grid."""
        return {
            'learning_rate': [0.001, 0.01],        # 2 options
            'batch_size': [64, 128],               # 2 options  
            'optimizer': ['sgd', 'adamw'],         # 2 options
            'weight_decay': [1e-5, 1e-4],         # 2 options
            'scheduler': ['cosine', 'step'],       # 2 options
        }
    
    
    def train_single_configuration(self, 
                                   params: Dict[str, Any],
                                   train_rgb: torch.Tensor,
                                   train_brightness: torch.Tensor,
                                   train_labels: torch.Tensor,
                                   val_rgb: torch.Tensor,
                                   val_brightness: torch.Tensor,
                                   val_labels: torch.Tensor) -> Dict[str, float]:
        """Train model with a single parameter configuration."""
        print(f"Training with parameters: {params}")
        
        try:
            # Create model
            model = mc_resnet50(
                num_classes=100,  # ImageNet classes
                device=self.device,
                use_amp=True  # Enable mixed precision
            )
            
            # Create data loaders
            print("Creating data loaders...")
            train_loader, val_loader = create_dual_channel_dataloaders(
                train_rgb=train_rgb,
                train_brightness=train_brightness,
                train_labels=train_labels,
                val_rgb=val_rgb,
                val_brightness=val_brightness,
                val_labels=val_labels,
                batch_size=params['batch_size'],
            )
            
            # Compile model with parameters
            print("Compiling model...")
            model.compile(optimizer=params['optimizer'],
                          loss='cross_entropy',
                          learning_rate=params['learning_rate'],
                          weight_decay=params['weight_decay'],
                          scheduler=params['scheduler']
                          )
            
            # Train model for 15 epochs with early stopping
            print("Starting training...")
            history = model.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=20,
                early_stopping=True,
                patience=10,
                min_delta=0.001,
                monitor='val_loss',
                restore_best_weights=True,
                save_best=False,  # Don't save models during grid search
                verbose=True
            )
            
            # Get best validation metrics
            best_val_loss = min(history['val_loss'])
            best_val_accuracy = max(history['val_accuracy'])
            
            # Final metrics
            final_val_loss = history['val_loss'][-1]
            final_val_accuracy = history['val_accuracy'][-1]
            
            # Check if early stopping occurred
            early_stopped = len(history['val_loss']) < 15
            early_stopping_info = history.get('early_stopping', {})
            
            return {
                'best_val_loss': best_val_loss,
                'best_val_accuracy': best_val_accuracy,
                'final_val_loss': final_val_loss,
                'final_val_accuracy': final_val_accuracy,
                'epochs_completed': len(history['val_loss']),
                'converged': len(history['val_loss']) >= 15,
                'early_stopped': early_stopped,
                'early_stopping_info': early_stopping_info,
                'training_history': {
                    'train_loss': history['train_loss'],
                    'val_loss': history['val_loss'],
                    'train_accuracy': history['train_accuracy'],
                    'val_accuracy': history['val_accuracy']
                }
            }
            
        except Exception as e:
            print(f"Error training with params {params}: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'best_val_loss': float('inf'),
                'best_val_accuracy': 0.0,
                'final_val_loss': float('inf'),
                'final_val_accuracy': 0.0,
                'epochs_completed': 0,
                'converged': False,
                'early_stopped': False,
                'early_stopping_info': {},
                'error': str(e)
            }
    
    def run_grid_search(self,
                        train_rgb: torch.Tensor,
                        train_brightness: torch.Tensor,
                        train_labels: torch.Tensor,
                        val_rgb: torch.Tensor,
                        val_brightness: torch.Tensor,
                        val_labels: torch.Tensor) -> List[Dict[str, Any]]:
        """Run baseline grid search over all parameter combinations."""
        param_grid = self.define_parameter_grid()
        
        # Generate all combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))
        
        print(f"Running baseline grid search over {len(combinations)} combinations...")
        print(f"Estimated time: ~{len(combinations) * 15 * 8 / 60:.1f} hours")
        print(f"Results will be saved to: {self.results_dir}")
        
        results = []
        
        for i, combination in enumerate(combinations):
            # Create parameter dictionary
            params = dict(zip(param_names, combination))
            
            print(f"\n{'='*80}")
            print(f"Combination {i+1}/{len(combinations)}")
            print(f"{'='*80}")
            
            # Train with this configuration
            start_time = datetime.now()
            metrics = self.train_single_configuration(
                params, train_rgb, train_brightness, train_labels,
                val_rgb, val_brightness, val_labels
            )
            end_time = datetime.now()
            
            # Store results
            result = {
                'combination_id': i,
                'parameters': params,
                'metrics': metrics,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_minutes': (end_time - start_time).total_seconds() / 60
            }
            results.append(result)
            
            # Save intermediate results after each combination
            self.save_results(results)
            
            print(f"Completed in {result['duration_minutes']:.1f} minutes")
            print(f"Results: val_loss={metrics['final_val_loss']:.4f}, "
                  f"val_acc={metrics['final_val_accuracy']:.4f}")
            print(f"Epochs: {metrics['epochs_completed']}/15, "
                  f"Early stopped: {metrics.get('early_stopped', False)}")
            
            # Estimate remaining time
            if i > 0:
                avg_time = sum(r['duration_minutes'] for r in results) / len(results)
                remaining_time = avg_time * (len(combinations) - i - 1)
                print(f"Estimated remaining time: {remaining_time:.1f} minutes ({remaining_time/60:.1f} hours)")
            
            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]]):
        """Save results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.results_dir, f"baseline_grid_search_results_{timestamp}.json")
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {filename}")
    
    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze grid search results."""
        # Filter out failed runs
        valid_results = [r for r in results if 'error' not in r['metrics']]
        
        if not valid_results:
            return {"error": "No valid results found"}
        
        # Sort by validation accuracy (descending)
        sorted_by_accuracy = sorted(
            valid_results, 
            key=lambda x: x['metrics']['final_val_accuracy'], 
            reverse=True
        )
        
        # Sort by validation loss (ascending)
        sorted_by_loss = sorted(
            valid_results,
            key=lambda x: x['metrics']['final_val_loss']
        )
        
        best_accuracy = sorted_by_accuracy[0]
        best_loss = sorted_by_loss[0]
        
        # Calculate early stopping statistics
        early_stopped_count = sum(1 for r in valid_results if r['metrics'].get('early_stopped', False))
        avg_epochs = sum(r['metrics']['epochs_completed'] for r in valid_results) / len(valid_results)
        
        return {
            'total_combinations': len(results),
            'successful_combinations': len(valid_results),
            'failed_combinations': len(results) - len(valid_results),
            'early_stopped_count': early_stopped_count,
            'average_epochs': avg_epochs,
            'best_by_accuracy': {
                'parameters': best_accuracy['parameters'],
                'accuracy': best_accuracy['metrics']['final_val_accuracy'],
                'loss': best_accuracy['metrics']['final_val_loss'],
                'epochs': best_accuracy['metrics']['epochs_completed'],
                'early_stopped': best_accuracy['metrics'].get('early_stopped', False)
            },
            'best_by_loss': {
                'parameters': best_loss['parameters'],
                'accuracy': best_loss['metrics']['final_val_accuracy'],
                'loss': best_loss['metrics']['final_val_loss'],
                'epochs': best_loss['metrics']['epochs_completed'],
                'early_stopped': best_loss['metrics'].get('early_stopped', False)
            },
            'top_5_configurations': [
                {
                    'combination_id': r['combination_id'],
                    'parameters': r['parameters'],
                    'accuracy': r['metrics']['final_val_accuracy'],
                    'loss': r['metrics']['final_val_loss'],
                    'epochs': r['metrics']['epochs_completed'],
                    'early_stopped': r['metrics'].get('early_stopped', False)
                }
                for r in sorted_by_accuracy[:5]
            ]
        }


def main():
    """Main function to run baseline grid search."""
    print("MCResNet Baseline Grid Search")
    print("="*50)
    
    # TODO: Load your actual data here
    # For now, creating dummy data as an example
    print("Loading data...")
    converter = RGBtoRGBL()
    # Example: Load your actual tensors here
    # train_rgb, train_brightness, train_labels = load_train_data()
    # val_rgb, val_brightness, val_labels = load_val_data()
    
    # Dummy data for demonstration (replace with actual data loading)
    train_color, train_labels, test_color, test_labels = load_cifar100_data(
        data_dir="data/cifar-100",
        normalize=True  # Apply normalization to [0, 1] range
    )

    # Split the data
    train_color, val_color, train_labels, val_labels = train_test_split(
        train_color, train_labels, test_size=0.1, random_state=42
    )

    train_brightness = converter.get_brightness(train_color)
    val_brightness = converter.get_brightness(val_color)
    test_brightness = converter.get_brightness(test_color)

    print(f"Train data: {train_color.shape}, {train_brightness.shape}, {train_labels.shape}")
    print(f"Val data: {val_color.shape}, {val_brightness.shape}, {val_labels.shape}")

    # Initialize grid search
    grid_search = MCResNetBaselineGridSearch(
        results_dir="mcresnet_baseline_grid_search",
        device=None  # Auto-detect
    )
    
    # Run grid search with data
    results = grid_search.run_grid_search(
        train_rgb=train_color,
        train_brightness=train_brightness,
        train_labels=train_labels,
        val_rgb=val_color,
        val_brightness=val_brightness,
        val_labels=val_labels
    )
    
    # Analyze results
    analysis = grid_search.analyze_results(results)
    
    # Print analysis
    print("\n" + "="*80)
    print("BASELINE GRID SEARCH ANALYSIS")
    print("="*80)
    print(f"Total combinations: {analysis['total_combinations']}")
    print(f"Successful combinations: {analysis['successful_combinations']}")
    print(f"Failed combinations: {analysis['failed_combinations']}")
    print(f"Early stopped combinations: {analysis.get('early_stopped_count', 0)}")
    print(f"Average epochs completed: {analysis.get('average_epochs', 0):.1f}")
    
    if analysis['successful_combinations'] > 0:
        print(f"\nBest configuration by accuracy:")
        best_acc = analysis['best_by_accuracy']
        print(f"Parameters: {best_acc['parameters']}")
        print(f"Accuracy: {best_acc['accuracy']:.4f}")
        print(f"Loss: {best_acc['loss']:.4f}")
        print(f"Epochs: {best_acc['epochs']}, Early stopped: {best_acc['early_stopped']}")
        
        print(f"\nTop 5 configurations:")
        for i, config in enumerate(analysis['top_5_configurations'], 1):
            print(f"{i}. ID={config['combination_id']}: "
                  f"acc={config['accuracy']:.4f}, loss={config['loss']:.4f}, "
                  f"epochs={config['epochs']}, early_stop={config['early_stopped']}")
            print(f"   {config['parameters']}")
    
    print(f"\nAll results saved in: {grid_search.results_dir}")


if __name__ == "__main__":
    main()
