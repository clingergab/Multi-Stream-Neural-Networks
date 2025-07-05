# Deep Debug Protocol - When Both Models Fail

from typing import Tuple
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from src.models.basic_multi_channel.multi_channel_resnet_network import multi_channel_resnet50
from src.transforms.augmentation import create_augmented_dataloaders
from src.transforms.dataset_utils import process_dataset_to_streams
from src.utils.cifar100_loader import create_validation_split, get_cifar100_datasets

def comprehensive_debug_pipeline_with_fit(train_loader, val_loader, multi_channel_resnet50_model):
    """
    Comprehensive debugging using your model's .fit() method instead of manual training loops.
    This will confirm if the issues persist with your actual training pipeline.
    """
    
    print("🔍 COMPREHENSIVE DEBUG ANALYSIS (Using .fit() Method)")
    print("=" * 70)
    
    # 1. DATA PIPELINE VERIFICATION (Same as before)
    print("\n📊 1. DATA PIPELINE VERIFICATION")
    print("-" * 40)
    
    batch_count = 0
    for batch_idx, batch_data in enumerate(train_loader):
        if batch_idx >= 3:  # Check first 3 batches
            break
            
        print(f"\nBatch {batch_idx}:")
        
        if len(batch_data) != 3:
            print(f"❌ CRITICAL: Expected 3 elements, got {len(batch_data)}")
            return "DATA_FORMAT_ERROR"
            
        color, brightness, labels = batch_data
        
        # Shape verification
        print(f"  Color shape: {color.shape}")
        print(f"  Brightness shape: {brightness.shape}")
        print(f"  Labels shape: {labels.shape}")
        
        # Expected shapes for CIFAR-100
        if color.shape[1:] != (3, 32, 32):
            print(f"❌ CRITICAL: Color shape wrong! Expected [batch, 3, 32, 32]")
            return "COLOR_SHAPE_ERROR"
            
        if brightness.shape[1:] != (1, 32, 32):
            print(f"❌ CRITICAL: Brightness shape wrong! Expected [batch, 1, 32, 32]")
            return "BRIGHTNESS_SHAPE_ERROR"
        
        # Data range verification
        print(f"  Color range: [{color.min():.4f}, {color.max():.4f}]")
        print(f"  Brightness range: [{brightness.min():.4f}, {brightness.max():.4f}]")
        print(f"  Label range: [{labels.min()}, {labels.max()}]")
        
        # Check for anomalies
        if torch.isnan(color).any():
            print("❌ CRITICAL: NaN values in color data!")
            return "COLOR_NAN_ERROR"
        if torch.isnan(brightness).any():
            print("❌ CRITICAL: NaN values in brightness data!")
            return "BRIGHTNESS_NAN_ERROR"
        if torch.isnan(labels.float()).any():
            print("❌ CRITICAL: NaN values in labels!")
            return "LABELS_NAN_ERROR"
            
        # Label verification
        if labels.min() < 0 or labels.max() >= 100:
            print(f"❌ CRITICAL: Labels out of range! Expected 0-99, got {labels.min()}-{labels.max()}")
            return "LABELS_RANGE_ERROR"
            
        # Check data diversity
        unique_labels = torch.unique(labels)
        print(f"  Unique labels in batch: {len(unique_labels)}/100 classes")
        
        batch_count += 1
    
    print("✅ Data pipeline verification passed!")
    
    # 2. MODEL ARCHITECTURE VERIFICATION (Same as before)
    print("\n🏗️ 2. MODEL ARCHITECTURE VERIFICATION")
    print("-" * 40)
    
    # Model parameter count
    total_params = sum(p.numel() for p in multi_channel_resnet50_model.parameters())
    trainable_params = sum(p.numel() for p in multi_channel_resnet50_model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    if total_params == 0:
        print("❌ CRITICAL: Model has no parameters!")
        return "NO_PARAMETERS_ERROR"
    
    if trainable_params == 0:
        print("❌ CRITICAL: Model has no trainable parameters!")
        return "NO_TRAINABLE_PARAMS_ERROR"
    
    # Expected parameter count for ResNet50
    expected_range = (20_000_000, 30_000_000)  # Rough range for multi-channel ResNet50
    if not (expected_range[0] <= total_params <= expected_range[1]):
        print(f"⚠️ WARNING: Parameter count unusual. Expected ~{expected_range[0]:,}-{expected_range[1]:,}")
    
    print("✅ Model architecture verification passed!")
    
    # 3. FORWARD PASS VERIFICATION (Same as before)
    print("\n🔄 3. FORWARD PASS VERIFICATION")
    print("-" * 40)
    
    multi_channel_resnet50_model.eval()
    
    # Test with single batch
    sample_color, sample_brightness, sample_labels = next(iter(train_loader))
    sample_color = sample_color.to(multi_channel_resnet50_model.device)
    sample_brightness = sample_brightness.to(multi_channel_resnet50_model.device)
    sample_labels = sample_labels.to(multi_channel_resnet50_model.device)
    
    print(f"Sample batch size: {sample_color.shape[0]}")
    
    try:
        with torch.no_grad():
            outputs = multi_channel_resnet50_model(sample_color, sample_brightness)
            
        print(f"✅ Forward pass successful!")
        print(f"   Output shape: {outputs.shape}")
        print(f"   Expected: [{sample_color.shape[0]}, 100]")
        
        if outputs.shape != (sample_color.shape[0], 100):
            print(f"❌ CRITICAL: Wrong output shape!")
            return "OUTPUT_SHAPE_ERROR"
            
        # Check output statistics
        print(f"   Output range: [{outputs.min():.4f}, {outputs.max():.4f}]")
        print(f"   Output mean: {outputs.mean():.4f}")
        print(f"   Output std: {outputs.std():.4f}")
        
        # Store initial outputs for comparison
        initial_outputs = outputs.clone()
        
        # Check for anomalies
        if torch.isnan(outputs).any():
            print("❌ CRITICAL: NaN in outputs!")
            return "OUTPUT_NAN_ERROR"
        if torch.isinf(outputs).any():
            print("❌ CRITICAL: Inf in outputs!")
            return "OUTPUT_INF_ERROR"
            
        # Check if outputs are reasonable (not all zeros, not all same)
        if outputs.std() < 1e-6:
            print("❌ CRITICAL: All outputs are nearly identical!")
            return "OUTPUT_IDENTICAL_ERROR"
            
    except Exception as e:
        print(f"❌ CRITICAL: Forward pass failed: {e}")
        return f"FORWARD_PASS_ERROR: {e}"
    
    print("✅ Forward pass verification passed!")
    
    # 4. LOSS COMPUTATION VERIFICATION (Same as before)
    print("\n📉 4. LOSS COMPUTATION VERIFICATION")
    print("-" * 40)
    
    try:
        loss = F.cross_entropy(outputs, sample_labels)
        print(f"✅ Loss computation successful!")
        print(f"   Loss value: {loss.item():.4f}")
        
        # Expected loss range for random predictions on 100 classes
        expected_random_loss = -np.log(1/100)  # ≈ 4.605
        print(f"   Expected random loss: {expected_random_loss:.4f}")
        
        if loss.item() > expected_random_loss * 2:
            print(f"⚠️ WARNING: Loss unusually high!")
        elif loss.item() < expected_random_loss * 0.5:
            print(f"⚠️ WARNING: Loss unusually low!")
            
        if torch.isnan(loss):
            print("❌ CRITICAL: NaN loss!")
            return "LOSS_NAN_ERROR"
            
    except Exception as e:
        print(f"❌ CRITICAL: Loss computation failed: {e}")
        return f"LOSS_COMPUTATION_ERROR: {e}"
    
    print("✅ Loss computation verification passed!")
    
    # 5. GRADIENT FLOW VERIFICATION (Same as before)
    print("\n📈 5. GRADIENT FLOW VERIFICATION")
    print("-" * 40)
    
    multi_channel_resnet50_model.train()
    multi_channel_resnet50_model.zero_grad()
    
    try:
        # Forward pass
        outputs = multi_channel_resnet50_model(sample_color, sample_brightness)
        loss = F.cross_entropy(outputs, sample_labels)
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        grad_norms = []
        no_grad_params = []
        
        for name, param in multi_channel_resnet50_model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    grad_norms.append(grad_norm)
                else:
                    no_grad_params.append(name)
        
        if no_grad_params:
            print(f"⚠️ WARNING: {len(no_grad_params)} parameters have no gradients:")
            for name in no_grad_params[:5]:  # Show first 5
                print(f"     {name}")
            if len(no_grad_params) > 5:
                print(f"     ... and {len(no_grad_params) - 5} more")
        
        if grad_norms:
            avg_grad = np.mean(grad_norms)
            max_grad = np.max(grad_norms)
            min_grad = np.min(grad_norms)
            
            print(f"✅ Gradient flow analysis:")
            print(f"   Parameters with gradients: {len(grad_norms)}")
            print(f"   Average gradient norm: {avg_grad:.6f}")
            print(f"   Max gradient norm: {max_grad:.6f}")
            print(f"   Min gradient norm: {min_grad:.6f}")
            
            # Check for gradient issues
            if max_grad > 10:
                print("⚠️ WARNING: Possible exploding gradients!")
            if avg_grad < 1e-8:
                print("⚠️ WARNING: Possible vanishing gradients!")
                return "VANISHING_GRADIENTS"
            
        else:
            print("❌ CRITICAL: No gradients found!")
            return "NO_GRADIENTS_ERROR"
            
    except Exception as e:
        print(f"❌ CRITICAL: Gradient computation failed: {e}")
        return f"GRADIENT_ERROR: {e}"
    
    print("✅ Gradient flow verification passed!")
    
    # 6. LEARNING CAPABILITY TEST WITH .fit() METHOD
    print("\n🧠 6. LEARNING CAPABILITY TEST (Using .fit() Method)")
    print("-" * 40)
    
    # Create tiny dataset for overfitting test
    tiny_color = sample_color[:16].clone()  # 16 samples
    tiny_brightness = sample_brightness[:16].clone()
    tiny_labels = sample_labels[:16].clone()
    
    # Create tiny dataset and loader
    tiny_dataset = TensorDataset(tiny_color, tiny_brightness, tiny_labels)
    tiny_loader = DataLoader(tiny_dataset, batch_size=16, shuffle=True)
    
    # Save model state
    original_state = multi_channel_resnet50_model.state_dict()
    
    print(f"Testing overfitting on {len(tiny_labels)} samples using .fit() method...")
    
    # Test with .fit() method
    try:
        # Create a temporary simple compilation for overfitting test
        multi_channel_resnet50_model.compile(
            optimizer='adam',
            learning_rate=0.001,
            weight_decay=0.0,
            gradient_clip=1.0,
            scheduler='none'
        )
        
        # Train using .fit() method
        overfit_history = multi_channel_resnet50_model.fit(
            train_loader=tiny_loader,
            val_loader=None,  # No validation for overfitting test
            epochs=10,
            verbose=1
        )
        
        # Check final overfitting results
        final_loss = overfit_history['train_loss'][-1]
        final_accuracy = overfit_history['train_accuracy'][-1]
        
        print(f"   Final overfitting loss: {final_loss:.4f}")
        print(f"   Final overfitting accuracy: {final_accuracy:.2%}")
        
        # Restore model state
        multi_channel_resnet50_model.load_state_dict(original_state)
        
        if final_accuracy < 0.8:
            print("❌ CRITICAL: Model cannot overfit tiny dataset using .fit() method!")
            return "CANNOT_OVERFIT_WITH_FIT"
        else:
            print("✅ Model can learn using .fit() method (overfitting test passed)!")
            
    except Exception as e:
        print(f"❌ CRITICAL: Overfitting test with .fit() failed: {e}")
        # Restore model state
        multi_channel_resnet50_model.load_state_dict(original_state)
        return f"OVERFIT_FIT_ERROR: {e}"
    
    # 7. ACTUAL TRAINING TEST WITH .fit() METHOD
    print("\n🚀 7. ACTUAL TRAINING TEST (Using .fit() Method)")
    print("-" * 40)
    
    # Check if model is compiled
    if not hasattr(multi_channel_resnet50_model, 'is_compiled') or not multi_channel_resnet50_model.is_compiled:
        print("❌ CRITICAL: Model not compiled for actual training!")
        return "MODEL_NOT_COMPILED"
    
    # Show compilation settings
    if hasattr(multi_channel_resnet50_model, 'training_config'):
        config = multi_channel_resnet50_model.training_config
        print(f"✅ Current compilation settings:")
        print(f"   Optimizer: {config.get('optimizer', 'unknown')}")
        print(f"   Learning rate: {config.get('learning_rate', 'unknown')}")
        print(f"   Weight decay: {config.get('weight_decay', 'unknown')}")
        print(f"   Scheduler: {config.get('scheduler', 'unknown')}")
        print(f"   Gradient clip: {config.get('gradient_clip', 'unknown')}")
        
        # Check if learning rate is reasonable
        lr = config.get('learning_rate', 0)
        if lr > 0.01:
            print(f"⚠️ WARNING: Learning rate might be too high: {lr}")
        elif lr < 1e-6:
            print(f"⚠️ WARNING: Learning rate might be too low: {lr}")
    
    # Test actual training for 1 epoch using .fit()
    print(f"\n🧪 Testing 1 epoch of actual training using .fit() method...")
    
    try:
        # Train for 1 epoch using your exact pipeline
        test_history = multi_channel_resnet50_model.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=1,
            verbose=1
        )
        
        # Analyze results
        train_loss = test_history['train_loss'][0]
        train_acc = test_history['train_accuracy'][0]
        val_loss = test_history['val_loss'][0] if test_history['val_loss'][0] is not None else 'N/A'
        val_acc = test_history['val_accuracy'][0] if test_history['val_accuracy'][0] is not None else 'N/A'
        
        print(f"\n📊 1-Epoch Training Results:")
        print(f"   Training loss: {train_loss:.4f}")
        print(f"   Training accuracy: {train_acc:.2%}")
        print(f"   Validation loss: {val_loss}")
        print(f"   Validation accuracy: {val_acc}")
        
        # Check for issues
        issues_found = []
        
        if train_loss > 10:
            issues_found.append(f"High training loss: {train_loss:.4f}")
        if train_acc < 0.05:  # Less than 5%
            issues_found.append(f"Low training accuracy: {train_acc:.2%}")
        if isinstance(val_loss, float) and val_loss > 10:
            issues_found.append(f"High validation loss: {val_loss:.4f}")
        if isinstance(val_acc, float) and val_acc < 0.05:
            issues_found.append(f"Low validation accuracy: {val_acc:.2%}")
        
        if issues_found:
            print(f"\n⚠️ ISSUES DETECTED:")
            for issue in issues_found:
                print(f"   - {issue}")
            return "TRAINING_ISSUES_WITH_FIT"
        else:
            print(f"\n✅ Training with .fit() method working normally!")
        
    except Exception as e:
        print(f"❌ CRITICAL: Training with .fit() method failed: {e}")
        return f"TRAINING_FIT_ERROR: {e}"
    
    print("\n🎯 DIAGNOSIS COMPLETE (Using .fit() Method)")
    print("=" * 70)
    return "ALL_CHECKS_PASSED_WITH_FIT"

# Main debug function to call
def debug_training_failure_with_fit(train_loader, val_loader, multi_channel_resnet50_model):
    """Main function to debug training failure using your .fit() method."""
    
    print("🚨 DEBUGGING CIFAR-100 TRAINING FAILURE (Using .fit() Method)")
    print("Confirming issues persist with actual training pipeline")
    print("=" * 80)
    
    # Run comprehensive debug using .fit() method
    result = comprehensive_debug_pipeline_with_fit(train_loader, val_loader, multi_channel_resnet50_model)
    
    print(f"\n🏁 FINAL DIAGNOSIS: {result}")
    
    if result == "ALL_CHECKS_PASSED_WITH_FIT":
        print("✅ All checks passed with .fit() method - training should work normally")
        return True
    else:
        print(f"❌ Issues confirmed with .fit() method: {result}")
        return False

def setup_data_loaders(data_dir: str = "data/cifar-100", batch_size: int = 32) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
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



if __name__ == "__main__":
    train_loader, val_loader = setup_data_loaders(data_dir="data/cifar-100", batch_size=64)
    multi_channel_resnet50_model = multi_channel_resnet50(
        num_classes=100,
    )
    
    # Compile with CNN-optimized settings
    multi_channel_resnet50_model.compile(
        optimizer='adamw',
        learning_rate=0.0003,  # Lower learning rate for CNN stability
        weight_decay=1e-4,
        early_stopping_patience=5,
        loss='cross_entropy',
        metrics=['accuracy']
    )
    debug_training_failure_with_fit(train_loader, val_loader, multi_channel_resnet50_model)
