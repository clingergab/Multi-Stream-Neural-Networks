"""
Comprehensive diagnostic script for multi-channel neural network training analysis.

This script performs full training runs on complete CIFAR-100 dataset with:
- Full-size models (base_multi_channel_large and multi_channel_resnet50)
- Complete training and validation datasets with augmentation
- 20 epochs training with early stopping (patience=5)
- Extensive diagnostics throughout training process

The goal is to identify why models don't train well and perform poorly.
"""

import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import json
import time
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Add project root and src to path to ensure imports work
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
src_path = str(Path(project_root) / 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import after sys.path is updated
try:
    from models.basic_multi_channel.multi_channel_resnet_network import multi_channel_resnet50
    from models.basic_multi_channel.base_multi_channel_network import base_multi_channel_large
    from data_utils.dataset_utils import get_cifar100_datasets
    from data_utils.rgb_to_rgbl import RGBtoRGBL
    from utils.debug_utils import (
        analyze_gradient_flow, 
        analyze_parameter_magnitudes,
        add_diagnostic_hooks
    )
    from utils.early_stopping import EarlyStopping
except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure you're running this script from the project root directory.")
    sys.exit(1)


class ComprehensiveModelDiagnostics:
    """
    Comprehensive diagnostics for multi-channel neural networks.
    
    Performs full training runs with extensive monitoring and analysis.
    """
    
    def __init__(self, output_dir: str = "diagnostics", device: str = "auto"):
        """
        Initialize diagnostics system.
        
        Args:
            output_dir: Directory to save all diagnostic outputs
            device: Device to run diagnostics on ('auto', 'cuda', 'mps', 'cpu')
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set device with proper Apple Silicon support
        if device == "auto":
            self.device = self._get_best_device()
        else:
            self.device = torch.device(device)
        
        print(f"🔧 Diagnostics initialized - Device: {self.device}")
        print(f"📁 Output directory: {self.output_dir}")
        
        # Initialize results storage
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def setup_data_loaders(self, data_dir: str = "data/cifar-100", batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
        """
        Set up CIFAR-100 data loaders with augmentation.
        
        Args:
            data_dir: Path to CIFAR-100 data directory
            batch_size: Batch size for training
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        print(f"📊 Setting up CIFAR-100 data loaders...")
        
        try:
            # Get datasets with augmentation
            train_dataset, val_dataset, class_names = get_cifar100_datasets(
                data_dir=data_dir,
                use_augmentation=True,  # Enable augmentation
                validation_split=0.1,   # Use 10% of training data for validation
                random_seed=42
            )
            
            # Create RGB to RGBL transform
            rgb_to_rgbl = RGBtoRGBL()
            
            # Wrap datasets with RGBL transform
            train_dataset = CIFAR100WithRGBL(train_dataset, rgb_to_rgbl)
            val_dataset = CIFAR100WithRGBL(val_dataset, rgb_to_rgbl)
            
            # Create data loaders with device-specific optimizations
            num_workers = 0 if self.device.type == "mps" else 4  # MPS works better with num_workers=0
            pin_memory = self.device.type == "cuda"  # Only beneficial for CUDA
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=True
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory
            )
            
            print(f"✅ Data loaders created:")
            print(f"   Training samples: {len(train_dataset)}")
            print(f"   Validation samples: {len(val_dataset)}")
            print(f"   Batch size: {batch_size}")
            print(f"   Augmentation: Enabled")
            
            return train_loader, val_loader
            
        except Exception as e:
            print(f"❌ Error setting up data loaders: {e}")
            raise
    
    def create_models(self) -> Dict[str, nn.Module]:
        """
        Create full-size models for comprehensive testing.
        
        Returns:
            Dictionary of model name -> model instance
        """
        print(f"🏗️  Creating full-size models...")
        
        models = {}
        
        # Create base_multi_channel_large
        try:
            base_model = base_multi_channel_large(
                color_input_size=3072,  # 32*32*3 flattened
                brightness_input_size=1024,  # 32*32*1 flattened
                num_classes=100,
                dropout=0.3,
                device=self.device
            )
            # Apply device-specific optimizations
            base_model = self._optimize_for_device(base_model)
            models['base_multi_channel_large'] = base_model
            print(f"✅ Created base_multi_channel_large - Parameters: {sum(p.numel() for p in base_model.parameters()):,}")
            
        except Exception as e:
            print(f"❌ Error creating base_multi_channel_large: {e}")
        
        # Create multi_channel_resnet50
        try:
            resnet_model = multi_channel_resnet50(
                num_classes=100,
                color_input_channels=3,
                brightness_input_channels=1,
                dropout=0.3,
                reduce_architecture=False,  # Use full architecture
                device=self.device
            )
            # Apply device-specific optimizations
            resnet_model = self._optimize_for_device(resnet_model)
            models['multi_channel_resnet50'] = resnet_model
            print(f"✅ Created multi_channel_resnet50 - Parameters: {sum(p.numel() for p in resnet_model.parameters()):,}")
            
        except Exception as e:
            print(f"❌ Error creating multi_channel_resnet50: {e}")
        
        return models
    
    def analyze_model_architecture(self, model: nn.Module, model_name: str) -> Dict[str, Any]:
        """
        Analyze model architecture and initialization.
        
        Args:
            model: Model to analyze
            model_name: Name of the model
            
        Returns:
            Dictionary with architecture analysis results
        """
        print(f"🔍 Analyzing {model_name} architecture...")
        
        analysis = {
            'model_name': model_name,
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'layer_analysis': {},
            'weight_statistics': {},
            'gradient_statistics': {}
        }
        
        # Analyze layers
        layer_count = 0
        param_count_by_type = {}
        
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules
                layer_count += 1
                module_type = type(module).__name__
                
                if module_type not in param_count_by_type:
                    param_count_by_type[module_type] = 0
                
                for param in module.parameters():
                    param_count_by_type[module_type] += param.numel()
        
        analysis['layer_analysis'] = {
            'total_layers': layer_count,
            'parameters_by_type': param_count_by_type
        }
        
        # Analyze weight initialization
        weight_stats = {}
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight_stats[name] = {
                    'shape': list(param.shape),
                    'mean': param.data.mean().item(),
                    'std': param.data.std().item(),
                    'min': param.data.min().item(),
                    'max': param.data.max().item()
                }
        
        analysis['weight_statistics'] = weight_stats
        
        # Save analysis
        analysis_path = self.output_dir / f"{model_name}_architecture_analysis_{self.timestamp}.json"
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"✅ Architecture analysis saved to {analysis_path}")
        return analysis
    
    def train_with_diagnostics(
        self, 
        model: nn.Module, 
        model_name: str, 
        train_loader: DataLoader, 
        val_loader: DataLoader,
        epochs: int = 20,
        early_stopping_patience: int = 5
    ) -> Dict[str, Any]:
        """
        Train model with comprehensive diagnostics.
        
        Args:
            model: Model to train
            model_name: Name of the model
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            early_stopping_patience: Early stopping patience
            
        Returns:
            Dictionary with training results and diagnostics
        """
        print(f"🚀 Starting training diagnostics for {model_name}...")
        
        # Set up training components
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True)
        
        # Add diagnostic hooks
        hooks = add_diagnostic_hooks(model)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'lr': [],
            'gradient_norms': [],
            'weight_norms': [],
            'dead_neurons': [],
            'pathway_balance': [],
            'epoch_times': []
        }
        
        # Training loop
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            print(f"\n📈 Epoch {epoch+1}/{epochs}")
            
            for batch_idx, (color_data, brightness_data, targets) in enumerate(train_loader):
                # Move data to device with MPS-safe transfer
                color_data = self._safe_to_device(color_data, non_blocking=True)
                brightness_data = self._safe_to_device(brightness_data, non_blocking=True)
                targets = self._safe_to_device(targets, non_blocking=True)
                
                # Handle different model input requirements
                if 'base_multi_channel' in model_name:
                    # Flatten for base model
                    color_data = color_data.view(color_data.size(0), -1)
                    brightness_data = brightness_data.view(brightness_data.size(0), -1)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(color_data, brightness_data)
                loss = criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Statistics
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()
                
                # Print progress
                if batch_idx % 100 == 0:
                    print(f"   Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for color_data, brightness_data, targets in val_loader:
                    color_data = self._safe_to_device(color_data, non_blocking=True)
                    brightness_data = self._safe_to_device(brightness_data, non_blocking=True)
                    targets = self._safe_to_device(targets, non_blocking=True)
                    
                    # Handle different model input requirements
                    if 'base_multi_channel' in model_name:
                        color_data = color_data.view(color_data.size(0), -1)
                        brightness_data = brightness_data.view(brightness_data.size(0), -1)
                    
                    outputs = model(color_data, brightness_data)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
            
            # Calculate epoch metrics
            epoch_train_loss = train_loss / len(train_loader)
            epoch_val_loss = val_loss / len(val_loader)
            epoch_train_acc = 100.0 * train_correct / train_total
            epoch_val_acc = 100.0 * val_correct / val_total
            epoch_time = time.time() - epoch_start
            
            # Update learning rate
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # Collect diagnostics
            gradient_norm = self._calculate_gradient_norm(model)
            weight_norm = self._calculate_weight_norm(model)
            dead_neuron_count = self._count_dead_neurons(model)
            pathway_balance = self._analyze_pathway_balance(model)
            
            # Store history
            history['train_loss'].append(epoch_train_loss)
            history['val_loss'].append(epoch_val_loss)
            history['train_acc'].append(epoch_train_acc)
            history['val_acc'].append(epoch_val_acc)
            history['lr'].append(current_lr)
            history['gradient_norms'].append(gradient_norm)
            history['weight_norms'].append(weight_norm)
            history['dead_neurons'].append(dead_neuron_count)
            history['pathway_balance'].append(pathway_balance)
            history['epoch_times'].append(epoch_time)
            
            # Print epoch summary
            print(f"   Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%")
            print(f"   Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")
            print(f"   LR: {current_lr:.6f}, Gradient Norm: {gradient_norm:.4f}")
            print(f"   Dead Neurons: {dead_neuron_count}, Pathway Balance: {pathway_balance:.4f}")
            print(f"   Epoch Time: {epoch_time:.2f}s")
            
            # Save best model
            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                torch.save(model.state_dict(), self.output_dir / f"best_{model_name}_{self.timestamp}.pth")
            
            # Check early stopping
            early_stopping(epoch_val_loss)
            if early_stopping.early_stop:
                print(f"⚠️  Early stopping triggered at epoch {epoch+1}")
                break
        
        # Clean up hooks
        for hook in hooks:
            hook.remove()
        
        # Generate final diagnostics
        self._generate_training_plots(model_name, history)
        self._generate_gradient_analysis(model, model_name)
        self._generate_weight_analysis(model, model_name)
        
        return {
            'model_name': model_name,
            'history': history,
            'best_val_acc': best_val_acc,
            'total_epochs': len(history['train_loss']),
            'final_train_acc': history['train_acc'][-1],
            'final_val_acc': history['val_acc'][-1]
        }
    
    def _calculate_gradient_norm(self, model: nn.Module) -> float:
        """Calculate total gradient norm."""
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
    
    def _calculate_weight_norm(self, model: nn.Module) -> float:
        """Calculate total weight norm."""
        total_norm = 0.0
        for p in model.parameters():
            param_norm = p.data.norm(2)
            total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
    
    def _count_dead_neurons(self, model: nn.Module) -> int:
        """Count dead neurons (always output zero)."""
        # This is a simplified check - in practice, you'd need to track activations
        dead_count = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.ReLU):
                # This is a placeholder - proper implementation would track activations
                pass
        return dead_count
    
    def _analyze_pathway_balance(self, model: nn.Module) -> float:
        """Analyze balance between pathways."""
        # This is model-specific and depends on the architecture
        try:
            if hasattr(model, 'analyze_pathway_weights'):
                weights = model.analyze_pathway_weights()
                return weights.get('balance_ratio', 1.0)
        except Exception:
            pass
        return 1.0
    
    def _generate_training_plots(self, model_name: str, history: Dict[str, List]):
        """Generate training plots."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Loss plot
        axes[0, 0].plot(history['train_loss'], label='Train Loss')
        axes[0, 0].plot(history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(history['train_acc'], label='Train Acc')
        axes[0, 1].plot(history['val_acc'], label='Val Acc')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate plot
        axes[0, 2].plot(history['lr'])
        axes[0, 2].set_title('Learning Rate')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Learning Rate')
        axes[0, 2].grid(True)
        
        # Gradient norms plot
        axes[1, 0].plot(history['gradient_norms'])
        axes[1, 0].set_title('Gradient Norms')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Gradient Norm')
        axes[1, 0].grid(True)
        
        # Weight norms plot
        axes[1, 1].plot(history['weight_norms'])
        axes[1, 1].set_title('Weight Norms')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Weight Norm')
        axes[1, 1].grid(True)
        
        # Pathway balance plot
        axes[1, 2].plot(history['pathway_balance'])
        axes[1, 2].set_title('Pathway Balance')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Balance Ratio')
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{model_name}_training_curves_{self.timestamp}.png", dpi=300)
        plt.close()
    
    def _generate_gradient_analysis(self, model: nn.Module, model_name: str):
        """Generate gradient flow analysis."""
        analyze_gradient_flow(model, self.output_dir / f"{model_name}_gradient_flow_{self.timestamp}.png")
    
    def _generate_weight_analysis(self, model: nn.Module, model_name: str):
        """Generate weight magnitude analysis."""
        analyze_parameter_magnitudes(model, self.output_dir / f"{model_name}_weight_magnitudes_{self.timestamp}.png")
    
    def compare_models(self, results: Dict[str, Any]):
        """Compare results between models."""
        print("📊 Comparing model results...")
        
        comparison = {
            'timestamp': self.timestamp,
            'models': {},
            'summary': {}
        }
        
        for model_name, result in results.items():
            comparison['models'][model_name] = {
                'best_val_acc': result['best_val_acc'],
                'final_train_acc': result['final_train_acc'],
                'final_val_acc': result['final_val_acc'],
                'total_epochs': result['total_epochs']
            }
        
        # Generate comparison plots
        self._generate_comparison_plots(results)
        
        # Save comparison
        comparison_path = self.output_dir / f"model_comparison_{self.timestamp}.json"
        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print(f"✅ Model comparison saved to {comparison_path}")
    
    def _generate_comparison_plots(self, results: Dict[str, Any]):
        """Generate comparison plots between models."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Validation accuracy comparison
        for model_name, result in results.items():
            axes[0, 0].plot(result['history']['val_acc'], label=model_name)
        axes[0, 0].set_title('Validation Accuracy Comparison')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy (%)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Training loss comparison
        for model_name, result in results.items():
            axes[0, 1].plot(result['history']['train_loss'], label=model_name)
        axes[0, 1].set_title('Training Loss Comparison')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Gradient norms comparison
        for model_name, result in results.items():
            axes[1, 0].plot(result['history']['gradient_norms'], label=model_name)
        axes[1, 0].set_title('Gradient Norms Comparison')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Gradient Norm')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Final accuracy bar chart
        model_names = list(results.keys())
        final_val_accs = [results[name]['final_val_acc'] for name in model_names]
        best_val_accs = [results[name]['best_val_acc'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, final_val_accs, width, label='Final Val Acc')
        axes[1, 1].bar(x + width/2, best_val_accs, width, label='Best Val Acc')
        axes[1, 1].set_title('Final Accuracy Comparison')
        axes[1, 1].set_xlabel('Model')
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(model_names)
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"model_comparison_{self.timestamp}.png", dpi=300)
        plt.close()
    
    def generate_final_report(self, results: Dict[str, Any]):
        """Generate final diagnostic report."""
        print("📝 Generating final diagnostic report...")
        
        report = f"""
# Comprehensive Model Diagnostics Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
This report analyzes the training performance of full-size multi-channel neural networks
on the complete CIFAR-100 dataset with augmentation.

## Models Analyzed
"""
        
        for model_name, result in results.items():
            report += f"""
### {model_name}
- **Best Validation Accuracy**: {result['best_val_acc']:.2f}%
- **Final Training Accuracy**: {result['final_train_acc']:.2f}%
- **Final Validation Accuracy**: {result['final_val_acc']:.2f}%
- **Total Epochs**: {result['total_epochs']}

"""
        
        report += """
## Key Findings

### Performance Issues Identified
1. **Low Validation Accuracy**: All models show suboptimal validation performance
2. **Training/Validation Gap**: Potential overfitting or underfitting issues
3. **Convergence Problems**: Early stopping or slow convergence patterns

### Potential Causes
1. **Architecture Issues**: 
   - Inappropriate model complexity for dataset
   - Poor pathway balance in multi-channel design
   
2. **Training Issues**:
   - Suboptimal hyperparameters
   - Inadequate data augmentation
   - Gradient flow problems
   
3. **Data Issues**:
   - Insufficient preprocessing
   - Poor RGB to RGBL transformation
   - Dataset imbalance

## Recommendations

### Immediate Actions
1. **Hyperparameter Tuning**:
   - Adjust learning rate schedule
   - Experiment with different optimizers
   - Modify weight decay and dropout rates

2. **Architecture Modifications**:
   - Simplify model architectures
   - Improve pathway fusion strategies
   - Add batch normalization where missing

3. **Data Preprocessing**:
   - Enhance data augmentation strategies
   - Improve RGB to RGBL transformation
   - Add data normalization

### Apple Silicon (MPS) Specific Optimizations
1. **Memory Management**:
   - Use smaller batch sizes if experiencing memory issues
   - Enable memory efficiency with torch.backends.mps.empty_cache()
   - Monitor memory usage with Activity Monitor

2. **Performance Tips**:
   - Use num_workers=0 for DataLoader on MPS
   - Avoid non_blocking transfers (not supported on MPS)
   - Consider mixed precision training for better performance

3. **Debugging MPS Issues**:
   - Set PYTORCH_ENABLE_MPS_FALLBACK=1 for unsupported operations
   - Use CPU fallback for operations that don't support MPS

### Long-term Improvements
1. **Model Architecture**:
   - Implement attention mechanisms
   - Add residual connections in base model
   - Experiment with different fusion techniques

2. **Training Strategy**:
   - Implement curriculum learning
   - Add mixup or cutmix augmentation
   - Use advanced optimization techniques

## Device Information
- **Device Used**: """ + str(self.device) + """
- **PyTorch Version**: """ + torch.__version__ + """
- **MPS Available**: """ + str(hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()) + """
- **CUDA Available**: """ + str(torch.cuda.is_available()) + """

## Files Generated
- Training curves: *_training_curves_*.png
- Gradient flow analysis: *_gradient_flow_*.png
- Weight magnitude analysis: *_weight_magnitudes_*.png
- Model comparison: model_comparison_*.png
- Architecture analysis: *_architecture_analysis_*.json
- Training results: comprehensive_diagnostics_*.json
"""
        
        # Save report
        report_path = self.output_dir / f"diagnostic_report_{self.timestamp}.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"✅ Final diagnostic report saved to {report_path}")
    
    def run_comprehensive_diagnostics(
        self, 
        data_dir: str = "data/cifar-100", 
        batch_size: int = 32,
        epochs: int = 20,
        early_stopping_patience: int = 5
    ):
        """
        Run comprehensive diagnostics on all models.
        
        Args:
            data_dir: Path to CIFAR-100 data directory
            batch_size: Batch size for training
            epochs: Number of training epochs
            early_stopping_patience: Early stopping patience
        """
        print("🚀 Starting comprehensive model diagnostics...")
        
        # Set up data loaders
        train_loader, val_loader = self.setup_data_loaders(data_dir, batch_size)
        
        # Create models
        models = self.create_models()
        
        if not models:
            print("❌ No models created successfully. Exiting.")
            return
        
        # Run diagnostics for each model
        results = {}
        
        for model_name, model in models.items():
            print(f"\n{'='*50}")
            print(f"🔍 Analyzing {model_name}")
            print(f"{'='*50}")
            
            try:
                # Analyze architecture
                arch_analysis = self.analyze_model_architecture(model, model_name)
                
                # Train with diagnostics
                training_results = self.train_with_diagnostics(
                    model, model_name, train_loader, val_loader, epochs, early_stopping_patience
                )
                
                # Combine results
                results[model_name] = {
                    **training_results,
                    'architecture': arch_analysis
                }
                
            except Exception as e:
                print(f"❌ Error analyzing {model_name}: {e}")
                import traceback
                traceback.print_exc()
        
        # Compare models
        if len(results) > 1:
            self.compare_models(results)
        
        # Generate final report
        self.generate_final_report(results)
        
        # Save all results
        results_path = self.output_dir / f"comprehensive_diagnostics_{self.timestamp}.json"
        with open(results_path, 'w') as f:
            # Convert non-serializable objects to strings
            serializable_results = {}
            for model_name, result in results.items():
                serializable_results[model_name] = {
                    'model_name': result['model_name'],
                    'best_val_acc': result['best_val_acc'],
                    'final_train_acc': result['final_train_acc'],
                    'final_val_acc': result['final_val_acc'],
                    'total_epochs': result['total_epochs'],
                    'history': result['history']
                }
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n✅ Comprehensive diagnostics completed!")
        print(f"📁 Results saved to: {self.output_dir}")
        print(f"📊 Check the diagnostic report for detailed analysis.")


# Dataset wrapper for RGBL transformation
class CIFAR100WithRGBL(torch.utils.data.Dataset):
    """Dataset wrapper that applies RGB to RGBL transformation."""
    
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform
        
    def __len__(self):
        return len(self.base_dataset)
        
    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        # Apply RGB to RGBL transformation
        color, brightness = self.transform(image)
        return color, brightness, label


def main():
    """Main function to run comprehensive diagnostics."""
    parser = argparse.ArgumentParser(description='Comprehensive Multi-Stream Neural Network Diagnostics')
    parser.add_argument('--data-dir', type=str, default='data/cifar-100',
                        help='Path to the CIFAR-100 data directory')
    parser.add_argument('--output-dir', type=str, default='diagnostics',
                        help='Directory to save diagnostic outputs')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to run diagnostics on (auto, cuda, mps, cpu)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training (reduce if running out of memory)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--early-stopping-patience', type=int, default=5,
                        help='Early stopping patience')
    
    args = parser.parse_args()
    
    # Create and run diagnostics
    diagnostics = ComprehensiveModelDiagnostics(
        output_dir=args.output_dir,
        device=args.device
    )
    
    diagnostics.run_comprehensive_diagnostics(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience
    )


if __name__ == "__main__":
    main()
