"""
Simple example demonstrating the Keras-like API for Multi-Channel Neural Networks.
Perfect for Google Colab usage.
"""

import torch
from src.models.basic_multi_channel.base_multi_channel_network import BaseMultiChannelNetwork
from src.models.basic_multi_channel.multi_channel_resnet_network import MultiChannelResNetNetwork
from src.utils.colab_utils import load_and_prepare_mnist_for_dense, load_and_prepare_cifar10_for_cnn, create_sample_data


def train_dense_model_example():
    """Example of training a dense multi-channel model (Keras-like API)."""
    print("🚀 Training Dense Multi-Channel Model Example")
    print("=" * 50)
    
    # Load and prepare data
    print("📊 Loading MNIST data...")
    data = load_and_prepare_mnist_for_dense()
    
    print(f"Color input size: {data['color_input_size']}")
    print(f"Brightness input size: {data['brightness_input_size']}")
    print(f"Number of classes: {data['num_classes']}")
    print(f"Training samples: {len(data['train_labels'])}")
    
    # Create model (automatically detects best GPU)
    print("\n🏗️ Creating model...")
    model = BaseMultiChannelNetwork(
        color_input_size=data['color_input_size'],
        brightness_input_size=data['brightness_input_size'],
        hidden_sizes=[512, 256, 128],
        num_classes=data['num_classes'],
        dropout=0.2,
        device='auto'  # Auto-detects CUDA, MPS, or CPU
    )
    
    print(f"Model device: {model.device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model (Keras-like fit method)
    print("\n🎯 Training model...")
    model.fit(
        train_color_data=data['train_color'],
        train_brightness_data=data['train_brightness'],
        train_labels=data['train_labels'],
        val_color_data=data['test_color'][:1000],  # Use subset for validation
        val_brightness_data=data['test_brightness'][:1000],
        val_labels=data['test_labels'][:1000],
        batch_size=128,
        epochs=5,  # Quick training for demo
        learning_rate=0.001,
        verbose=1
    )
    
    # Evaluate model
    print("\n📈 Evaluating model...")
    results = model.evaluate(
        test_color_data=data['test_color'],
        test_brightness_data=data['test_brightness'],
        test_labels=data['test_labels']
    )
    
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    print(f"Test Loss: {results['loss']:.4f}")
    
    # Make predictions
    print("\n🔮 Making predictions...")
    predictions = model.predict(
        color_data=data['test_color'][:10],
        brightness_data=data['test_brightness'][:10]
    )
    
    print(f"Predictions: {predictions}")
    print(f"Actual: {data['test_labels'][:10]}")
    
    return model


def train_cnn_model_example():
    """Example of training a CNN multi-channel model (Keras-like API)."""
    print("🚀 Training CNN Multi-Channel Model Example")
    print("=" * 50)
    
    # Load and prepare data
    print("📊 Loading CIFAR-10 data...")
    data = load_and_prepare_cifar10_for_cnn()
    
    print(f"Color input channels: {data['color_input_channels']}")
    print(f"Brightness input channels: {data['brightness_input_channels']}")
    print(f"Number of classes: {data['num_classes']}")
    print(f"Training samples: {len(data['train_labels'])}")
    print(f"Data shape - Color: {data['train_color'].shape}, Brightness: {data['train_brightness'].shape}")
    
    # Create ResNet model (automatically detects best GPU)
    print("\n🏗️ Creating ResNet model...")
    model = MultiChannelResNetNetwork(
        num_classes=data['num_classes'],
        color_input_channels=data['color_input_channels'],
        brightness_input_channels=data['brightness_input_channels'],
        num_blocks=[2, 2, 2, 2],  # ResNet-18 architecture
        block_type='basic',
        device='auto'  # Auto-detects CUDA, MPS, or CPU
    )
    
    print(f"Model device: {model.device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model (Keras-like fit method)
    print("\n🎯 Training model...")
    model.fit(
        train_color_data=data['train_color'][:5000],  # Use subset for demo
        train_brightness_data=data['train_brightness'][:5000],
        train_labels=data['train_labels'][:5000],
        val_color_data=data['test_color'][:1000],
        val_brightness_data=data['test_brightness'][:1000],
        val_labels=data['test_labels'][:1000],
        batch_size=64,
        epochs=3,  # Quick training for demo
        learning_rate=0.001,
        verbose=1
    )
    
    # Evaluate model
    print("\n📈 Evaluating model...")
    results = model.evaluate(
        test_color_data=data['test_color'][:1000],
        test_brightness_data=data['test_brightness'][:1000],
        test_labels=data['test_labels'][:1000]
    )
    
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    print(f"Test Loss: {results['loss']:.4f}")
    
    return model


def quick_api_demo():
    """Quick demonstration of the API with sample data."""
    print("🚀 Quick API Demo with Sample Data")
    print("=" * 40)
    
    # Create sample data
    (train_color, train_brightness, train_labels), (test_color, test_brightness, test_labels) = create_sample_data(
        n_samples=1000, input_size=784, num_classes=10, for_cnn=False
    )
    
    print("Sample data shapes:")
    print(f"  Color: {train_color.shape}")
    print(f"  Brightness: {train_brightness.shape}")
    print(f"  Labels: {train_labels.shape}")
    
    # Create and train model
    print("\n🏗️ Creating model...")
    model = BaseMultiChannelNetwork(
        color_input_size=train_color.shape[1],
        brightness_input_size=train_brightness.shape[1],
        hidden_sizes=[256, 128],
        num_classes=10,
        device='auto'
    )
    
    print(f"Model device: {model.device}")
    
    # Compile model (optional, Keras-like)
    model.compile(optimizer='adam', learning_rate=0.001, loss='cross_entropy')
    
    # Train model
    print("\n🎯 Training model...")
    model.fit(
        train_color_data=train_color,
        train_brightness_data=train_brightness,
        train_labels=train_labels,
        batch_size=32,
        epochs=3,
        verbose=1
    )
    
    # Evaluate and predict
    results = model.evaluate(test_color, test_brightness, test_labels)
    predictions = model.predict(test_color[:5], test_brightness[:5])
    
    print(f"\nResults: Accuracy = {results['accuracy']:.3f}")
    print(f"Sample predictions: {predictions}")
    
    return model


if __name__ == "__main__":
    print("🎯 Multi-Channel Neural Networks - Colab Examples")
    print("=" * 60)
    
    # Check device availability
    print("🖥️ Device Information:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print()
    
    # Run examples
    try:
        # Quick demo with sample data
        quick_api_demo()
        print("\n" + "="*60 + "\n")
        
        # Dense model with MNIST
        train_dense_model_example()
        print("\n" + "="*60 + "\n")
        
        # CNN model with CIFAR-10 (comment out if too slow)
        # train_cnn_model_example()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("✅ Examples completed!")
