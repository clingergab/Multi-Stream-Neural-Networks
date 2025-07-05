#!/usr/bin/env python3
"""
Batch Size Effect Experiment

This script tests how different batch sizes affect training accuracy
for multi-stream neural networks on CIFAR-100.
"""

import sys
import torch
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import json
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

try:
    from scripts.modern_comprehensive_diagnostics import ModernComprehensiveModelDiagnostics
except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure you're running this script from the project root directory.")
    sys.exit(1)


class BatchSizeExperiment:
    """
    Experiment to test the effect of batch size on training accuracy.
    """
    
    def __init__(self, output_dir: str = "experiments/batch_size_study", device: str = "auto"):
        """
        Initialize batch size experiment.
        
        Args:
            output_dir: Directory to save experimental results
            device: Device to run experiments on
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"🧪 Batch Size Experiment initialized")
        print(f"📁 Output directory: {self.output_dir}")
    
    def run_batch_size_comparison(
        self, 
        batch_sizes: List[int] = [16, 32, 64, 128],
        epochs: int = 10,
        data_dir: str = "data/cifar-100"
    ):
        """
        Run experiments with different batch sizes.
        
        Args:
            batch_sizes: List of batch sizes to test
            epochs: Number of epochs for each experiment
            data_dir: Path to CIFAR-100 data
        """
        print(f"🚀 Starting Batch Size Comparison Experiment")
        print(f"📊 Testing batch sizes: {batch_sizes}")
        print(f"⏱️  Epochs per experiment: {epochs}")
        print("=" * 60)
        
        all_results = {}
        
        for batch_size in batch_sizes:
            print(f"\n🔬 Testing Batch Size: {batch_size}")
            print("=" * 40)
            
            # Create experiment-specific output directory
            experiment_dir = self.output_dir / f"batch_size_{batch_size}"
            
            try:
                # Run diagnostics with current batch size
                diagnostics = ModernComprehensiveModelDiagnostics(
                    output_dir=str(experiment_dir),
                    device=self.device
                )
                
                # Setup data loaders with current batch size
                train_loader, val_loader = diagnostics.setup_data_loaders(
                    data_dir=data_dir, 
                    batch_size=batch_size
                )
                
                # Test both models
                models = diagnostics.create_models()
                batch_results = {}
                
                for model_name, model in models.items():
                    print(f"\n🔧 Training {model_name} with batch size {batch_size}")
                    
                    # Train with current batch size
                    training_results = diagnostics.train_with_comprehensive_diagnostics(
                        model, model_name, train_loader, val_loader, epochs
                    )
                    
                    # Extract key metrics
                    perf_metrics = training_results.get('performance_metrics', {})
                    batch_results[model_name] = {
                        'batch_size': batch_size,
                        'best_val_accuracy': perf_metrics.get('best_val_accuracy', 0),
                        'final_train_accuracy': perf_metrics.get('final_train_accuracy', 0),
                        'final_val_accuracy': perf_metrics.get('final_val_accuracy', 0),
                        'total_epochs': perf_metrics.get('total_epochs', epochs),
                        'avg_gradient_norm': perf_metrics.get('avg_gradient_norm', 0),
                        'gradient_stability': perf_metrics.get('gradient_stability', 0),
                        'avg_pathway_balance': perf_metrics.get('avg_pathway_balance', 1.0),
                        'training_history': training_results.get('training_history', {})
                    }
                    
                    print(f"✅ {model_name} - Best Val Accuracy: {perf_metrics.get('best_val_accuracy', 0):.4f}")
                
                all_results[batch_size] = batch_results
                print(f"✅ Batch size {batch_size} experiment complete")
                
            except Exception as e:
                print(f"❌ Error with batch size {batch_size}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Analyze and visualize results
        self.analyze_batch_size_effects(all_results)
        
        return all_results
    
    def analyze_batch_size_effects(self, results: Dict[int, Dict[str, Any]]):
        """
        Analyze and visualize the effect of batch size on training.
        
        Args:
            results: Dictionary of batch_size -> model_results
        """
        print(f"\n📊 Analyzing Batch Size Effects")
        print("=" * 40)
        
        # Prepare data for analysis
        analysis = {
            'timestamp': self.timestamp,
            'batch_size_effects': {},
            'model_comparisons': {},
            'recommendations': []
        }
        
        # Extract metrics by model and batch size
        model_names = set()
        for batch_results in results.values():
            model_names.update(batch_results.keys())
        
        for model_name in model_names:
            batch_sizes = []
            accuracies = []
            train_accuracies = []
            gradient_stabilities = []
            pathway_balances = []
            
            for batch_size, batch_results in results.items():
                if model_name in batch_results:
                    result = batch_results[model_name]
                    batch_sizes.append(batch_size)
                    accuracies.append(result['best_val_accuracy'])
                    train_accuracies.append(result['final_train_accuracy'])
                    gradient_stabilities.append(result['gradient_stability'])
                    pathway_balances.append(result['avg_pathway_balance'])
            
            if batch_sizes:
                # Find optimal batch size for this model
                best_idx = np.argmax(accuracies)
                optimal_batch_size = batch_sizes[best_idx]
                best_accuracy = accuracies[best_idx]
                
                analysis['model_comparisons'][model_name] = {
                    'batch_sizes': batch_sizes,
                    'val_accuracies': accuracies,
                    'train_accuracies': train_accuracies,
                    'gradient_stabilities': gradient_stabilities,
                    'pathway_balances': pathway_balances,
                    'optimal_batch_size': optimal_batch_size,
                    'best_accuracy': best_accuracy,
                    'accuracy_range': max(accuracies) - min(accuracies),
                    'stability_trend': 'improving' if gradient_stabilities[-1] < gradient_stabilities[0] else 'degrading'
                }
                
                print(f"\n🎯 {model_name} Results:")
                print(f"   Optimal batch size: {optimal_batch_size}")
                print(f"   Best accuracy: {best_accuracy:.4f}")
                print(f"   Accuracy range: {max(accuracies) - min(accuracies):.4f}")
                
                # Generate recommendations
                if max(accuracies) - min(accuracies) > 0.02:  # 2% difference
                    analysis['recommendations'].append(
                        f"{model_name}: Batch size significantly affects performance (range: {max(accuracies) - min(accuracies):.4f})"
                    )
                
                if optimal_batch_size <= 32:
                    analysis['recommendations'].append(
                        f"{model_name}: Benefits from smaller batch sizes (optimal: {optimal_batch_size})"
                    )
                elif optimal_batch_size >= 64:
                    analysis['recommendations'].append(
                        f"{model_name}: Benefits from larger batch sizes (optimal: {optimal_batch_size})"
                    )
        
        # Overall analysis
        all_optimal_sizes = [comp['optimal_batch_size'] for comp in analysis['model_comparisons'].values()]
        if all_optimal_sizes:
            analysis['batch_size_effects'] = {
                'most_common_optimal': max(set(all_optimal_sizes), key=all_optimal_sizes.count),
                'optimal_range': f"{min(all_optimal_sizes)}-{max(all_optimal_sizes)}",
                'significant_differences': len(analysis['recommendations']) > 0
            }
        
        # Create visualizations
        self.create_batch_size_plots(analysis['model_comparisons'])
        
        # Save detailed analysis
        analysis_path = self.output_dir / f"batch_size_analysis_{self.timestamp}.json"
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print(f"\n💡 Key Findings:")
        for rec in analysis['recommendations']:
            print(f"   • {rec}")
        
        if 'batch_size_effects' in analysis:
            effects = analysis['batch_size_effects']
            print(f"\n🎯 Overall Summary:")
            print(f"   Most common optimal batch size: {effects['most_common_optimal']}")
            print(f"   Optimal range across models: {effects['optimal_range']}")
            print(f"   Significant batch size effects: {'Yes' if effects['significant_differences'] else 'No'}")
        
        print(f"\n✅ Analysis saved to {analysis_path}")
    
    def create_batch_size_plots(self, model_comparisons: Dict[str, Any]):
        """
        Create visualizations of batch size effects.
        
        Args:
            model_comparisons: Dictionary of model comparison data
        """
        print(f"📈 Creating batch size effect visualizations...")
        
        # Create subplots for each model
        n_models = len(model_comparisons)
        fig, axes = plt.subplots(2, n_models, figsize=(6*n_models, 10))
        if n_models == 1:
            axes = axes.reshape(2, 1)
        
        for idx, (model_name, data) in enumerate(model_comparisons.items()):
            batch_sizes = data['batch_sizes']
            val_accuracies = data['val_accuracies']
            train_accuracies = data['train_accuracies']
            gradient_stabilities = data['gradient_stabilities']
            
            # Plot accuracy vs batch size
            ax1 = axes[0, idx] if n_models > 1 else axes[0]
            ax1.plot(batch_sizes, val_accuracies, 'o-', label='Validation', linewidth=2, markersize=8)
            ax1.plot(batch_sizes, train_accuracies, 's-', label='Training', linewidth=2, markersize=8)
            ax1.set_xlabel('Batch Size')
            ax1.set_ylabel('Accuracy')
            ax1.set_title(f'{model_name}\nAccuracy vs Batch Size')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_xscale('log', base=2)
            
            # Mark optimal batch size
            optimal_idx = np.argmax(val_accuracies)
            ax1.axvline(batch_sizes[optimal_idx], color='red', linestyle='--', alpha=0.7, 
                       label=f'Optimal: {batch_sizes[optimal_idx]}')
            ax1.legend()
            
            # Plot gradient stability vs batch size
            ax2 = axes[1, idx] if n_models > 1 else axes[1]
            ax2.plot(batch_sizes, gradient_stabilities, 'D-', color='orange', linewidth=2, markersize=8)
            ax2.set_xlabel('Batch Size')
            ax2.set_ylabel('Gradient Stability (Std Dev)')
            ax2.set_title(f'{model_name}\nGradient Stability vs Batch Size')
            ax2.grid(True, alpha=0.3)
            ax2.set_xscale('log', base=2)
        
        plt.tight_layout()
        
        # Save plots
        plot_path = self.output_dir / f"batch_size_effects_{self.timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plots saved to {plot_path}")
        plt.close()


def main():
    """Main function to run batch size experiment."""
    
    parser = argparse.ArgumentParser(description='Batch Size Effect Experiment for Multi-Stream Neural Networks')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[16, 32, 64, 128],
                        help='List of batch sizes to test')
    parser.add_argument('--epochs', type=int, default=8,
                        help='Number of epochs for each experiment (reduced for comparison)')
    parser.add_argument('--data-dir', type=str, default='data/cifar-100',
                        help='Path to CIFAR-100 data directory')
    parser.add_argument('--output-dir', type=str, default='experiments/batch_size_study',
                        help='Directory to save experimental results')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to run experiments on (auto, cuda, mps, cpu)')
    
    args = parser.parse_args()
    
    # Create and run batch size experiment
    experiment = BatchSizeExperiment(
        output_dir=args.output_dir,
        device=args.device
    )
    
    results = experiment.run_batch_size_comparison(
        batch_sizes=args.batch_sizes,
        epochs=args.epochs,
        data_dir=args.data_dir
    )
    
    print(f"\n🎉 Batch Size Experiment Complete!")
    print(f"📁 Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
