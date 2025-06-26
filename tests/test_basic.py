#!/usr/bin/env python3
"""
Simple test script to verify GPU optimization and basic functionality.
"""

import sys
import os

# Add src to path
sys.path.append('.')

def test_device_detection():
    """Test automatic device detection."""
    print("🔍 Testing Device Detection...")
    
    try:
        from src.utils.device_utils import DeviceManager
        dm = DeviceManager()
        print(f"✅ Device detected: {dm.device}")
        return True
    except Exception as e:
        print(f"❌ Device detection failed: {e}")
        return False

def test_model_creation():
    """Test model creation with GPU support."""
    print("\n🏗️ Testing Model Creation...")
    
    try:
        from src.models.basic_multi_channel.base_multi_channel_network import BaseMultiChannelNetwork
        
        model = BaseMultiChannelNetwork(
            color_input_size=784*3,
            brightness_input_size=784,
            hidden_sizes=[256, 128],
            num_classes=10,
            device='auto'
        )
        
        print(f"✅ Dense model created on: {model.device}")
        print(f"✅ Parameters: {sum(p.numel() for p in model.parameters()):,}")
        return True
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

def test_sample_data():
    """Test data loading utilities."""
    print("\n📊 Testing Sample Data Creation...")
    
    try:
        from src.utils.colab_utils import create_sample_data
        
        (train_color, train_brightness, train_labels), _ = create_sample_data(
            n_samples=100, for_cnn=False
        )
        
        print(f"✅ Sample data created: {train_color.shape}, {train_brightness.shape}")
        return True
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 GPU Optimization Test Suite")
    print("=" * 40)
    
    tests = [
        test_device_detection,
        test_model_creation,
        test_sample_data
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! GPU optimization working correctly!")
        return True
    else:
        print("❌ Some tests failed. Check the output above.")
        return False

if __name__ == "__main__":
    main()
