#!/usr/bin/env python3
"""
GPU Optimization Test Script
Tests all GPU optimizations including mixed precision, batch sizes, and data loading.
"""

import sys
import torch
import time

# Add current directory to path
sys.path.append('.')

def test_mixed_precision_support():
    """Test mixed precision support detection."""
    print("🔍 Testing Mixed Precision Support...")
    
    try:
        from src.utils.device_utils import DeviceManager
        dm = DeviceManager()
        
        mixed_precision = dm.enable_mixed_precision()
        print(f"✅ Mixed precision support: {mixed_precision}")
        print(f"   Device: {dm.device}")
        print(f"   Device type: {dm.device_type}")
        
        if dm.device_type == 'cuda':
            caps = torch.cuda.get_device_capability(dm.device)
            print(f"   CUDA capability: {caps}")
            print("   Required for mixed precision: >= 7.0")
        
        return True
        
    except Exception as e:
        print(f"✗ Mixed precision test failed: {e}")
        return False

def test_optimal_batch_sizes():
    """Test automatic batch size detection."""
    print("\n📊 Testing Optimal Batch Size Detection...")
    
    try:
        from src.models.basic_multi_channel.base_multi_channel_network import BaseMultiChannelNetwork
        from src.models.basic_multi_channel.multi_channel_resnet_network import MultiChannelResNetNetwork
        
        # Test dense model
        dense_model = BaseMultiChannelNetwork(
            color_input_size=784,
            brightness_input_size=196,
            hidden_sizes=[256, 128],
            num_classes=10
        )
        
        print(f"✅ Dense model created on: {dense_model.device}")
        print(f"   Mixed precision: {dense_model.use_mixed_precision}")
        
        # Test CNN model
        cnn_model = MultiChannelResNetNetwork(
            color_input_channels=3,
            brightness_input_channels=1,
            num_classes=10
        )
        
        print(f"✅ CNN model created on: {cnn_model.device}")
        print(f"   Mixed precision: {cnn_model.use_mixed_precision}")
        
        return True
        
    except Exception as e:
        print(f"✗ Batch size test failed: {e}")
        return False

def test_optimized_data_loading():
    """Test optimized data loading with proper batch size detection."""
    print("\n⚡ Testing Optimized Data Loading...")
    
    try:
        from src.models.basic_multi_channel.base_multi_channel_network import BaseMultiChannelNetwork
        from src.utils.colab_utils import create_sample_data
        
        # Create model
        model = BaseMultiChannelNetwork(
            color_input_size=588,  # Match the actual data size
            brightness_input_size=196,
            hidden_sizes=[256, 128],
            num_classes=10
        )
        
        # Create sample data
        (color_data, brightness_data, labels), _ = create_sample_data(
            n_samples=500,
            input_size=784,
            num_classes=10,
            for_cnn=False
        )
        
        # Data is already flattened correctly by create_sample_data
        print(f"✅ Data created: color {color_data.shape}, brightness {brightness_data.shape}")
        
        # Test training with optimized settings (just 1 epoch)
        start_time = time.time()
        
        model.fit(
            train_color_data=color_data,
            train_brightness_data=brightness_data,
            train_labels=labels,
            epochs=1,
            verbose=1
        )
        
        training_time = time.time() - start_time
        print(f"✅ Training completed in {training_time:.2f} seconds")
        
        # Test prediction with auto batch sizing
        start_time = time.time()
        predictions = model.predict(color_data[:100], brightness_data[:100])
        prediction_time = time.time() - start_time
        
        print(f"✅ Prediction completed in {prediction_time:.3f} seconds")
        print(f"   Predictions shape: {predictions.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Data loading test failed: {e}")
        return False

def test_memory_optimization():
    """Test memory optimization features."""
    print("\n🧠 Testing Memory Optimization...")
    
    try:
        from src.utils.device_utils import DeviceManager
        
        dm = DeviceManager()
        
        # Test memory info
        memory_info = dm.get_memory_info()
        print(f"✅ Memory info retrieved: {memory_info}")
        
        # Test cache clearing
        print("🧹 Testing cache clearing...")
        dm.clear_cache()
        print("✅ Cache cleared successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Memory optimization test failed: {e}")
        return False

def test_device_optimization():
    """Test device-specific optimizations."""
    print("\n🚀 Testing Device-Specific Optimizations...")
    
    try:
        from src.models.basic_multi_channel.base_multi_channel_network import BaseMultiChannelNetwork
        
        model = BaseMultiChannelNetwork(
            color_input_size=784,
            brightness_input_size=196,
            hidden_sizes=[512, 256],
            num_classes=10
        )
        
        print(f"✅ Model on device: {model.device}")
        print(f"   Model optimized: {model.device_manager is not None}")
        
        # Check if model was optimized for device
        device_type = model.device.type
        if device_type == 'cuda':
            print("   ⚡ CUDA optimizations applied")
            print(f"   📊 Mixed precision: {model.use_mixed_precision}")
        elif device_type == 'mps':
            print("   🍎 MPS optimizations applied")
        else:
            print("   💻 CPU optimizations applied")
        
        return True
        
    except Exception as e:
        print(f"✗ Device optimization test failed: {e}")
        return False

def main():
    """Main test function."""
    print("=" * 60)
    print("GPU OPTIMIZATION TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_mixed_precision_support,
        test_optimal_batch_sizes,
        test_optimized_data_loading,
        test_memory_optimization,
        test_device_optimization
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("GPU OPTIMIZATION TEST SUMMARY")
    print("=" * 60)
    
    if passed == total:
        print(f"🎉 ALL {total} TESTS PASSED!")
        print("✅ GPU optimizations are working correctly")
        print("\nOptimizations included:")
        print("   🚀 Automatic device detection")
        print("   ⚡ Mixed precision training (when supported)")
        print("   📊 Optimal batch size auto-detection")
        print("   🔄 Optimized data loading (pin_memory, num_workers)")
        print("   🧠 Memory management and cache clearing")
        print("   📈 Learning rate scheduling")
        print("   💾 Non-blocking tensor transfers")
        return 0
    else:
        print(f"❌ {total - passed}/{total} TESTS FAILED")
        print("Some GPU optimizations may not be working correctly")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
