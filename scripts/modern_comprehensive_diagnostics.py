#!/usr/bin/env python3
"""
Updated comprehensive diagnostic script that uses existing model APIs.

This script demonstrates how to get the same comprehensive diagnostics as the original
comprehensive_model_diagnostics.py, but using the integrated diagnostic capabilities
in the model fit() methods and other existing APIs.

The goal is to show that you can get the same analytical power with much cleaner code
by leveraging the integrated diagnostic system.
"""

import sys
import torch
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import json

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.models.basic_multi_channel.multi_channel_resnet_network import multi_channel_resnet50
    from src.models.basic_multi_channel.base_multi_channel_network import base_multi_channel_large
    from src.data_utils.dataset_utils import get_cifar100_datasets, create_validation_split
    from src.data_utils.rgb_to_rgbl import process_dataset_to_streams
    from src.data_utils.augmentation import create_augmented_dataloaders
    from src.data_utils.rgb_to_rgbl import RGBtoRGBL
except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure you're running this script from the project root directory.")
    sys.exit(1)




class ModernComprehensiveModelDiagnostics:
    """
    Modern comprehensive diagnostics using integrated model APIs.
    
    This class provides the same analytical capabilities as the original
    comprehensive_model_diagnostics.py but leverages the integrated diagnostic
    system for cleaner, more maintainable code.
    """
    
    def __init__(self, output_dir: str = "diagnostics", device: str = "auto"):
        """
        Initialize modern diagnostics system.
        
        Args:
            output_dir: Directory to save all diagnostic outputs
            device: Device to run diagnostics on ('auto', 'cuda', 'mps', 'cpu')
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set device with proper Apple Silicon support
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        print(f"🔧 Modern Diagnostics initialized - Device: {self.device}")
        print(f"📁 Output directory: {self.output_dir}")
        
        # Initialize results storage
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def setup_data_loaders(self, data_dir: str = "data/cifar-100", batch_size: int = 32) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """
        Set up CIFAR-100 data loaders using existing project utilities.
        
        Args:
            data_dir: Path to CIFAR-100 data directory
            batch_size: Batch size for training
            
        Returns:
            Tuple of (train_loader, val_loader) that works for all model types
        """
        print(f"📊 Setting up CIFAR-100 data loaders with full dataset...")
        
        try:
            # Use existing utilities to get CIFAR-100 datasets
            train_dataset, test_dataset, class_names = get_cifar100_datasets(data_dir=data_dir)
            
            # Create validation split from training data (10% for validation)
            train_dataset, val_dataset = create_validation_split(train_dataset, val_split=0.1)
            
            # Process datasets to RGB+L streams
            print("Converting datasets to RGB+L streams...")
            train_rgb, train_brightness, train_labels_tensor = process_dataset_to_streams(
                train_dataset, desc="Training data"
            )
            val_rgb, val_brightness, val_labels_tensor = process_dataset_to_streams(
                val_dataset, desc="Validation data"
            )
            
            # Create augmented data loaders that work for all model types
            print("Creating augmented data loaders...")
            train_loader, val_loader = create_augmented_dataloaders(
                train_rgb, train_brightness, train_labels_tensor,
                val_rgb, val_brightness, val_labels_tensor,
                batch_size=batch_size,
                dataset="cifar100",
                num_workers=0,  # Avoid multiprocessing issues
                pin_memory=False
            )
            
            print(f"✅ CIFAR-100 data loaders created:")
            print(f"   Training batches: {len(train_loader)} ({len(train_labels_tensor)} samples)")
            print(f"   Validation batches: {len(val_loader)} ({len(val_labels_tensor)} samples)")
            print(f"   Classes: {len(class_names)}")
            print(f"   Augmentation: Enabled")
            
            return train_loader, val_loader
            
        except Exception as e:
            print(f"❌ Error setting up data loaders: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def create_models(self) -> Dict[str, torch.nn.Module]:
        """
        Create full-size models for comprehensive testing.
        
        Returns:
            Dictionary of model name -> model instance
        """
        print(f"🏗️  Creating full-size models...")
        
        models = {}
        
        # Create base_multi_channel_large for flattened CIFAR-100
        try:
            model = base_multi_channel_large(
                color_input_size=3072,      # 32*32*3 flattened
                brightness_input_size=1024, # 32*32*1 flattened  
                num_classes=100,
                device=str(self.device)
            )
            
            # Compile with optimized settings for dense networks
            model.compile(
                optimizer='adamw',
                learning_rate=0.001,
                weight_decay=1e-4,
                early_stopping_patience=5,
                loss='cross_entropy',
                metrics=['accuracy']
            )
            
            models['base_multi_channel_large'] = model
            print(f"✅ Created base_multi_channel_large: {sum(p.numel() for p in model.parameters()):,} parameters")
            
        except Exception as e:
            print(f"❌ Error creating base_multi_channel_large: {e}")
            import traceback
            traceback.print_exc()
        
        # Create multi_channel_resnet50 for image data
        try:
            model = multi_channel_resnet50(
                num_classes=100,
                device=str(self.device)
            )
            
            # Compile with CNN-optimized settings
            model.compile(
                optimizer='adamw',
                learning_rate=0.0003,  # Lower learning rate for CNN stability
                weight_decay=1e-4,
                early_stopping_patience=5,
                loss='cross_entropy',
                metrics=['accuracy']
            )
            
            models['multi_channel_resnet50'] = model
            print(f"✅ Created multi_channel_resnet50: {sum(p.numel() for p in model.parameters()):,} parameters")
            
        except Exception as e:
            print(f"❌ Error creating multi_channel_resnet50: {e}")
            import traceback
            traceback.print_exc()
        
        return models
    
    def analyze_model_architecture(self, model: torch.nn.Module, model_name: str) -> Dict[str, Any]:
        """
        Analyze model architecture using existing model methods.
        
        Args:
            model: Model to analyze
            model_name: Name of the model
            
        Returns:
            Dictionary with architecture analysis results
        """
        print(f"🔍 Analyzing {model_name} architecture...")
        
        # Use existing model methods for analysis
        model_stats = model.get_model_stats()
        classifier_info = model.get_classifier_info()
        pathway_weights = model.analyze_pathway_weights()
        pathway_importance = model.get_pathway_importance()
        
        analysis = {
            'model_name': model_name,
            'timestamp': self.timestamp,
            'model_stats': model_stats,
            'classifier_info': classifier_info,
            'pathway_analysis': {
                'pathway_weights': pathway_weights,
                'pathway_importance': pathway_importance,
                'fusion_type': model.fusion_type
            },
            'architecture_summary': {
                'total_parameters': model_stats['total_parameters'],
                'trainable_parameters': model_stats['trainable_parameters'],
                'architecture_type': pathway_weights.get('architecture_type', 'dense'),
                'balance_ratio': pathway_weights.get('balance_ratio', 1.0)
            }
        }
        
        # Save analysis
        analysis_path = self.output_dir / f"{model_name}_architecture_analysis_{self.timestamp}.json"
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print(f"✅ Architecture analysis saved to {analysis_path}")
        return analysis
    
    def train_with_comprehensive_diagnostics(
        self, 
        model: torch.nn.Module, 
        model_name: str, 
        train_loader: torch.utils.data.DataLoader, 
        val_loader: torch.utils.data.DataLoader,
        epochs: int = 20
    ) -> Dict[str, Any]:
        """
        Train model using integrated diagnostic capabilities.
        
        Args:
            model: Model to train
            model_name: Name of the model
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            
        Returns:
            Dictionary with training results and comprehensive diagnostics
        """
        print(f"🚀 Starting comprehensive diagnostic training for {model_name}...")
        
        # Use the model's integrated fit method with diagnostics enabled
        diagnostic_dir = self.output_dir / model_name
        
        history = model.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            verbose=1,
            enable_diagnostics=True,  # 🔍 This enables all comprehensive diagnostics
            diagnostic_output_dir=str(diagnostic_dir)
        )
        
        # Get comprehensive diagnostic summary using model methods
        diagnostic_summary = model.get_diagnostic_summary()
        
        # Analyze final model state
        final_pathway_analysis = model.analyze_pathway_weights()
        final_pathway_importance = model.get_pathway_importance()
        
        # Compile results
        results = {
            'model_name': model_name,
            'training_history': history,
            'diagnostic_summary': diagnostic_summary,
            'final_analysis': {
                'pathway_weights': final_pathway_analysis,
                'pathway_importance': final_pathway_importance,
                'fusion_type': model.fusion_type,
                'classifier_info': model.get_classifier_info()
            },
            'training_config': getattr(model, 'training_config', {}),
            'performance_metrics': {
                'best_val_accuracy': max(history.get('val_accuracy', [0])),
                'final_train_accuracy': history['train_accuracy'][-1] if history['train_accuracy'] else 0,
                'final_val_accuracy': history['val_accuracy'][-1] if history['val_accuracy'] else 0,
                'total_epochs': len(history.get('train_loss', [])),
            }
        }
        
        # Add diagnostic-specific metrics if available
        if 'gradient_norms' in history:
            results['performance_metrics'].update({
                'final_gradient_norm': history['gradient_norms'][-1],
                'avg_gradient_norm': np.mean(history['gradient_norms']),
                'gradient_stability': np.std(history['gradient_norms']),
                'final_pathway_balance': history['pathway_balance'][-1],
                'avg_pathway_balance': np.mean(history['pathway_balance']),
            })
        
        # Save detailed results
        results_path = diagnostic_dir / f"{model_name}_comprehensive_results_{self.timestamp}.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"✅ Comprehensive results saved to {results_path}")
        return results
    
    def compare_models(self, results: Dict[str, Any]):
        """
        Compare results between models using enhanced analytics.
        
        Args:
            results: Dictionary of model_name -> training results
        """
        print("📊 Comparing model results...")
        
        comparison = {
            'timestamp': self.timestamp,
            'comparison_summary': {},
            'detailed_comparison': {},
            'recommendations': []
        }
        
        # Extract key metrics for comparison
        for model_name, result in results.items():
            perf_metrics = result.get('performance_metrics', {})
            pathway_analysis = result.get('final_analysis', {}).get('pathway_weights', {})
            
            comparison['detailed_comparison'][model_name] = {
                'architecture': pathway_analysis.get('architecture_type', 'unknown'),
                'parameters': result.get('training_history', {}).get('total_parameters', 0),
                'best_accuracy': perf_metrics.get('best_val_accuracy', 0),
                'pathway_balance': pathway_analysis.get('balance_ratio', 1.0),
                'gradient_stability': perf_metrics.get('gradient_stability', 0),
                'training_efficiency': perf_metrics.get('total_epochs', 0)
            }
        
        # Generate recommendations
        if len(results) >= 2:
            accuracies = [(name, res['performance_metrics'].get('best_val_accuracy', 0)) 
                         for name, res in results.items()]
            best_model = max(accuracies, key=lambda x: x[1])
            
            comparison['recommendations'].append(f"Best performing model: {best_model[0]} with {best_model[1]:.4f} accuracy")
            
            # Pathway balance analysis
            for model_name, result in results.items():
                balance = result.get('final_analysis', {}).get('pathway_weights', {}).get('balance_ratio', 1.0)
                if balance < 0.7:
                    comparison['recommendations'].append(f"{model_name}: Pathway imbalance detected (ratio: {balance:.3f})")
                
        # Save comparison
        comparison_path = self.output_dir / f"model_comparison_{self.timestamp}.json"
        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=2, default=str)
        
        print(f"✅ Model comparison saved to {comparison_path}")
        
        # Print summary
        print("\n📈 Comparison Summary:")
        for model_name, details in comparison['detailed_comparison'].items():
            print(f"   {model_name}:")
            print(f"     • Accuracy: {details['best_accuracy']:.4f}")
            print(f"     • Parameters: {details['parameters']:,}")
            print(f"     • Pathway Balance: {details['pathway_balance']:.3f}")
            print(f"     • Architecture: {details['architecture']}")
        
        if comparison['recommendations']:
            print("\n💡 Recommendations:")
            for rec in comparison['recommendations']:
                print(f"   • {rec}")
    
    def generate_final_report(self, results: Dict[str, Any]):
        """
        Generate comprehensive final report using model analytics.
        
        Args:
            results: Dictionary of model results
        """
        print("📋 Generating comprehensive final report...")
        
        report = {
            'experiment_info': {
                'timestamp': self.timestamp,
                'device': str(self.device),
                'models_tested': list(results.keys()),
                'total_models': len(results)
            },
            'executive_summary': {},
            'detailed_results': results,
            'analysis_insights': [],
            'methodology': 'Integrated diagnostic system using model APIs'
        }
        
        # Generate executive summary
        if results:
            all_accuracies = []
            all_parameters = []
            all_balance_ratios = []
            
            for model_name, result in results.items():
                perf = result.get('performance_metrics', {})
                pathway = result.get('final_analysis', {}).get('pathway_weights', {})
                
                accuracy = perf.get('best_val_accuracy', 0)
                params = result.get('diagnostic_summary', {}).get('total_parameters', 0)
                balance = pathway.get('balance_ratio', 1.0)
                
                all_accuracies.append(accuracy)
                all_parameters.append(params)
                all_balance_ratios.append(balance)
            
            report['executive_summary'] = {
                'best_accuracy': max(all_accuracies) if all_accuracies else 0,
                'avg_accuracy': np.mean(all_accuracies) if all_accuracies else 0,
                'total_parameters_tested': sum(all_parameters),
                'avg_pathway_balance': np.mean(all_balance_ratios) if all_balance_ratios else 1.0,
                'training_stability': 'Good' if np.mean(all_balance_ratios) > 0.8 else 'Needs attention'
            }
            
            # Generate insights
            if max(all_accuracies) > 0.8:
                report['analysis_insights'].append("High accuracy achieved - models are learning effectively")
            
            if np.mean(all_balance_ratios) < 0.7:
                report['analysis_insights'].append("Pathway imbalance detected - consider adjusting learning rates")
            
            if len(set(all_accuracies)) > 1:
                report['analysis_insights'].append("Significant performance differences between architectures")
        
        # Save final report
        report_path = self.output_dir / f"comprehensive_final_report_{self.timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"✅ Final report saved to {report_path}")
        
        # Print executive summary
        exec_summary = report.get('executive_summary', {})
        print(f"\n🎯 Executive Summary:")
        print(f"   Best Accuracy: {exec_summary.get('best_accuracy', 0):.4f}")
        print(f"   Average Accuracy: {exec_summary.get('avg_accuracy', 0):.4f}")
        print(f"   Training Stability: {exec_summary.get('training_stability', 'Unknown')}")
        print(f"   Average Pathway Balance: {exec_summary.get('avg_pathway_balance', 1.0):.3f}")
    
    def run_comprehensive_diagnostics(
        self, 
        data_dir: str = "data/cifar-100", 
        batch_size: int = 64,
        epochs: int = 10  # Reduced for demo
    ):
        """
        Run comprehensive diagnostics using integrated model capabilities.
        
        This method demonstrates how to achieve the same analytical depth as
        the original comprehensive_model_diagnostics.py but with much cleaner code.
        
        Args:
            data_dir: Path to CIFAR-100 data
            batch_size: Batch size for training
            epochs: Number of epochs (reduced for demo)
        """
        print("🚀 Starting Modern Comprehensive Model Diagnostics")
        print("=" * 60)
        print("This demonstrates the same analytical power as the original script")
        print("but using integrated diagnostic capabilities for cleaner code.")
        print("=" * 60)
        
        try:
            # Setup data loaders that work for all model types
            train_loader, val_loader = self.setup_data_loaders(data_dir, batch_size)
            
            # Create models
            models = self.create_models()
            
            if not models:
                print("❌ No models were created successfully")
                return
            
            # Run comprehensive analysis for each model
            all_results = {}
            
            for model_name, model in models.items():
                print(f"\n{'='*40}")
                print(f"🔬 Analyzing {model_name}")
                print(f"{'='*40}")
                
                print(f"🔧 Using data loaders for {model_name}")
                
                # Architecture analysis
                arch_analysis = self.analyze_model_architecture(model, model_name)
                
                # Comprehensive training with diagnostics
                training_results = self.train_with_comprehensive_diagnostics(
                    model, model_name, train_loader, val_loader, epochs
                )
                
                # Combine results
                all_results[model_name] = {
                    **training_results,
                    'architecture_analysis': arch_analysis
                }
                
                print(f"✅ {model_name} analysis complete")
            
            # Compare models
            self.compare_models(all_results)
            
            # Generate final report
            self.generate_final_report(all_results)
            
            print(f"\n🎉 Modern Comprehensive Diagnostics Complete!")
            print(f"📁 All results saved to: {self.output_dir}")
            print(f"💡 The integrated diagnostic system provides the same analytical")
            print(f"   depth as the original script with much cleaner, maintainable code.")
            
        except Exception as e:
            print(f"❌ Error during comprehensive diagnostics: {e}")
            import traceback
            traceback.print_exc()
    


def main():
    """Main function to run modern comprehensive diagnostics."""
    
    parser = argparse.ArgumentParser(description='Modern Comprehensive Multi-Stream Neural Network Diagnostics')
    parser.add_argument('--data-dir', type=str, default='data/cifar-100',
                        help='Path to the CIFAR-100 data directory')
    parser.add_argument('--output-dir', type=str, default='diagnostics/modern_comprehensive',
                        help='Directory to save diagnostic outputs')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to run diagnostics on (auto, cuda, mps, cpu)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs for full diagnostics')
    
    args = parser.parse_args()
    
    # Create and run modern diagnostics
    diagnostics = ModernComprehensiveModelDiagnostics(
        output_dir=args.output_dir,
        device=args.device
    )
    
    diagnostics.run_comprehensive_diagnostics(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        epochs=args.epochs
    )


if __name__ == "__main__":
    main()
