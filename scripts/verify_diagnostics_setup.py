#!/usr/bin/env python3
"""
Verification script to check if all diagnostic components are properly set up
before running the comprehensive diagnostics.
"""

import sys
import torch
from pathlib import Path

def verify_imports():
    """Verify all necessary imports work."""
    print("🔍 Verifying imports...")
    
    try:
        from models.basic_multi_channel.multi_channel_resnet_network import multi_channel_resnet50
        from models.basic_multi_channel.base_multi_channel_network import base_multi_channel_large
        from data_utils.dataset_utils import get_cifar100_datasets, create_validation_split
        from data_utils.rgb_to_rgbl import RGBtoRGBL
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def verify_model_creation():
    """Verify models can be created with proper parameters."""
    print("🏗️  Verifying model creation...")
    
    try:
        from models.basic_multi_channel.multi_channel_resnet_network import multi_channel_resnet50
        from models.basic_multi_channel.base_multi_channel_network import base_multi_channel_large
        
        # Test base_multi_channel_large
        base_model = base_multi_channel_large(
            color_input_size=3072,
            brightness_input_size=1024,
            num_classes=100,
            dropout=0.2,
            device='cpu'
        )
        print(f"✅ base_multi_channel_large created: {sum(p.numel() for p in base_model.parameters()):,} parameters")
        
        # Test multi_channel_resnet50
        resnet_model = multi_channel_resnet50(
            num_classes=100,
            reduce_architecture=True,
            dropout=0.2,
            device='cpu'
        )
        print(f"✅ multi_channel_resnet50 created: {sum(p.numel() for p in resnet_model.parameters()):,} parameters")
        
        return True, base_model, resnet_model
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

def verify_diagnostic_methods(model, model_name):
    """Verify model has all required diagnostic methods."""
    print(f"🔬 Verifying diagnostic methods for {model_name}...")
    
    required_methods = [
        'get_diagnostic_summary',
        'analyze_pathway_weights',
        'get_pathway_importance',
        'get_classifier_info',
        'get_model_stats',
        'fit',
        'compile'
    ]
    
    missing_methods = []
    for method in required_methods:
        if not hasattr(model, method):
            missing_methods.append(method)
    
    if missing_methods:
        print(f"❌ Missing methods in {model_name}: {missing_methods}")
        return False
    else:
        print(f"✅ All diagnostic methods present in {model_name}")
        return True

def verify_model_compilation(model, model_name):
    """Verify model compilation works with early stopping."""
    print(f"⚙️  Verifying compilation for {model_name}...")
    
    try:
        model.compile(
            optimizer='adamw',
            learning_rate=0.001,
            weight_decay=1e-4,
            scheduler='cosine',
            early_stopping_patience=5
        )
        print(f"✅ {model_name} compiled successfully")
        return True
    except Exception as e:
        print(f"❌ Compilation error for {model_name}: {e}")
        return False

def verify_data_loading():
    """Verify data loading works with CIFAR-100."""
    print("📊 Verifying data loading...")
    
    try:
        # Check if data directory exists
        data_dir = Path("data/cifar-100")
        if not data_dir.exists():
            print(f"❌ CIFAR-100 data directory not found at {data_dir}")
            return False
        
        # Test data loading
        from data_utils.dataset_utils import get_cifar100_datasets, create_validation_split
        
        train_dataset, test_dataset, class_names = get_cifar100_datasets(data_dir="data/cifar-100")
        train_dataset, val_dataset = create_validation_split(train_dataset, val_split=0.1)
        
        print(f"✅ Data loading successful:")
        print(f"   Training samples: {len(train_dataset)}")
        print(f"   Validation samples: {len(val_dataset)}")
        print(f"   Test samples: {len(test_dataset)}")
        print(f"   Classes: {len(class_names)}")
        
        return True
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        return False

def verify_transforms():
    """Verify RGB to RGBL transform works."""
    print("🎨 Verifying transforms...")
    
    try:
        from data_utils.rgb_to_rgbl import RGBtoRGBL
        
        # Test transform
        transform = RGBtoRGBL()
        test_tensor = torch.randn(3, 32, 32)  # RGB image
        
        rgb_out, brightness_out = transform(test_tensor)
        
        print(f"✅ Transform successful:")
        print(f"   RGB output shape: {rgb_out.shape}")
        print(f"   Brightness output shape: {brightness_out.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Transform error: {e}")
        return False

def main():
    """Run all verifications."""
    print("🚀 Starting Diagnostic Setup Verification")
    print("=" * 50)
    
    all_passed = True
    
    # Verify imports
    if not verify_imports():
        all_passed = False
    
    # Verify model creation
    models_created, base_model, resnet_model = verify_model_creation()
    if not models_created:
        all_passed = False
    
    # Verify diagnostic methods for each model
    if models_created:
        if not verify_diagnostic_methods(base_model, "base_multi_channel_large"):
            all_passed = False
        if not verify_diagnostic_methods(resnet_model, "multi_channel_resnet50"):
            all_passed = False
        
        # Verify compilation
        if not verify_model_compilation(base_model, "base_multi_channel_large"):
            all_passed = False
        if not verify_model_compilation(resnet_model, "multi_channel_resnet50"):
            all_passed = False
    
    # Verify data loading
    if not verify_data_loading():
        all_passed = False
    
    # Verify transforms
    if not verify_transforms():
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL VERIFICATIONS PASSED!")
        print("✅ Ready to run comprehensive diagnostics")
        print("\nNext steps:")
        print("1. Run: python scripts/modern_comprehensive_diagnostics.py --epochs 20")
        print("2. Check results in: diagnostics/modern_comprehensive/")
    else:
        print("❌ SOME VERIFICATIONS FAILED!")
        print("Please fix the issues above before running diagnostics")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
