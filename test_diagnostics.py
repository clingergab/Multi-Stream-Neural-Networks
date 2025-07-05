#!/usr/bin/env python3
"""
Test script to verify the diagnostic script works properly
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all imports work correctly"""
    try:
        from scripts.modern_comprehensive_diagnostics import ModernComprehensiveModelDiagnostics
        print("✅ ModernComprehensiveModelDiagnostics imported successfully")
        
        from src.models.basic_multi_channel.multi_channel_resnet_network import multi_channel_resnet50
        print("✅ multi_channel_resnet50 imported successfully")
        
        from src.models.basic_multi_channel.base_multi_channel_network import base_multi_channel_large
        print("✅ base_multi_channel_large imported successfully")
        
        from src.utils.cifar100_loader import get_cifar100_datasets
        print("✅ get_cifar100_datasets imported successfully")
        
        from src.transforms.rgb_to_rgbl import RGBtoRGBL
        print("✅ RGBtoRGBL imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_creation():
    """Test model creation"""
    try:
        from src.models.basic_multi_channel.base_multi_channel_network import base_multi_channel_large
        
        model = base_multi_channel_large(
            color_input_size=3072,
            brightness_input_size=1024,
            num_classes=100,
            device='cpu'
        )
        
        print(f"✅ base_multi_channel_large created: {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test compilation
        model.compile(
            optimizer='adamw',
            learning_rate=0.001,
            weight_decay=1e-4,
            scheduler='cosine',
            early_stopping_patience=5
        )
        
        print("✅ Model compilation successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🧪 Testing diagnostic script components...")
    
    # Test imports
    if not test_imports():
        print("❌ Import test failed")
        return False
        
    # Test model creation
    if not test_model_creation():
        print("❌ Model creation test failed")
        return False
        
    print("✅ All tests passed!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
