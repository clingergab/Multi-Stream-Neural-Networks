#!/usr/bin/env python3
"""
Quick training test with sample data to verify the Keras-like API.
"""

import sys
sys.path.append('.')

def quick_training_test():
    """Quick test of the training API."""
    print("🚀 Quick Training Test")
    print("=" * 30)
    
    try:
        # Import required modules
        from src.utils.colab_utils import create_sample_data
        from src.models.basic_multi_channel.base_multi_channel_network import BaseMultiChannelNetwork
        
        print("📊 Creating sample data...")
        # Create small dataset for quick test
        (train_color, train_brightness, train_labels), (test_color, test_brightness, test_labels) = create_sample_data(
            n_samples=200, input_size=784, num_classes=10, for_cnn=False
        )
        
        print(f"✅ Data shapes: Color {train_color.shape}, Brightness {train_brightness.shape}")
        
        print("\n🏗️ Creating model...")
        model = BaseMultiChannelNetwork(
            color_input_size=train_color.shape[1],
            brightness_input_size=train_brightness.shape[1],
            hidden_sizes=[128, 64],
            num_classes=10,
            device='auto'
        )
        
        print(f"✅ Model created on: {model.device}")
        
        print("\n🎯 Training model (2 epochs)...")
        model.fit(
            train_color_data=train_color,
            train_brightness_data=train_brightness,
            train_labels=train_labels,
            batch_size=32,
            epochs=2,
            learning_rate=0.01,
            verbose=1
        )
        
        print("\n📈 Testing prediction...")
        predictions = model.predict(test_color[:5], test_brightness[:5])
        print(f"✅ Predictions: {predictions}")
        
        print("\n📊 Testing evaluation...")
        results = model.evaluate(test_color, test_brightness, test_labels)
        print(f"✅ Test accuracy: {results['accuracy']:.3f}")
        
        print("\n🎉 Training test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    quick_training_test()
