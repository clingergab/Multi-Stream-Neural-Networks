#!/usr/bin/env python3
"""
End-to-end test for both Dense and ResNet multi-stream models.
Tests building, training, and evaluation using our CIFAR-100 data loader and APIs.
"""

import sys
import os
import torch
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import our utilities and models
try:
    from src.utils.cifar100_loader import load_cifar100_numpy
    from src.models.basic_multi_channel.base_multi_channel_network import BaseMultiChannelNetwork
    from src.models.basic_multi_channel.multi_channel_resnet_network import MultiChannelResNetNetwork
    from src.transforms.rgb_to_rgbl import RGBtoRGBL
except ImportError:
    # Alternative import method
    sys.path.insert(0, os.path.dirname(__file__))
    from src.utils.cifar100_loader import load_cifar100_numpy
    from src.models.basic_multi_channel.base_multi_channel_network import BaseMultiChannelNetwork
    from src.models.basic_multi_channel.multi_channel_resnet_network import MultiChannelResNetNetwork
    from src.transforms.rgb_to_rgbl import RGBtoRGBL

def create_brightness_channel(rgb_data):
    """Convert RGB data to brightness channel using our transform."""
    rgb_to_rgbl = RGBtoRGBL()
    
    # Convert numpy to tensor, apply transform, convert back
    rgb_tensor = torch.from_numpy(rgb_data).float()
    rgb_out, brightness_out = rgb_to_rgbl(rgb_tensor)  # Transform returns tuple
    
    return brightness_out.numpy()

def split_data(data, labels, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, subset_size=None):
    """Split data into train/validation/test sets."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    n_samples = len(labels)
    if subset_size and subset_size < n_samples:
        # Use a subset for testing
        indices = np.random.choice(n_samples, subset_size, replace=False)
        data = data[indices]
        labels = labels[indices]
        n_samples = subset_size
    
    # Calculate split indices
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    # Create splits
    train_data = data[:train_end]
    train_labels = labels[:train_end]
    val_data = data[train_end:val_end]
    val_labels = labels[train_end:val_end]
    test_data = data[val_end:]
    test_labels = labels[val_end:]
    
    return {
        'train': {'color': train_data, 'labels': train_labels},
        'validation': {'color': val_data, 'labels': val_labels},
        'test': {'color': test_data, 'labels': test_labels}
    }

def test_data_loading():
    """Test our CIFAR-100 data loading pipeline."""
    print("🔄 Testing CIFAR-100 data loading...")
    
    # Load CIFAR-100 data
    train_data, train_labels, test_data, test_labels, label_names = load_cifar100_numpy()
    
    # Combine train and test for our own splitting
    all_data = np.concatenate([train_data, test_data], axis=0)
    all_labels = np.concatenate([train_labels, test_labels], axis=0)
    
    # Create our splits with a subset for testing
    data_splits = split_data(all_data, all_labels, subset_size=1000)
    
    # Create brightness channels
    for split_name in ['train', 'validation', 'test']:
        brightness = create_brightness_channel(data_splits[split_name]['color'])
        data_splits[split_name]['brightness'] = brightness
    
    print(f"✓ Data loaded and processed successfully:")
    print(f"  Train: {data_splits['train']['color'].shape[0]} samples")
    print(f"  Validation: {data_splits['validation']['color'].shape[0]} samples") 
    print(f"  Test: {data_splits['test']['color'].shape[0]} samples")
    print(f"  Color shape: {data_splits['train']['color'].shape}")
    print(f"  Brightness shape: {data_splits['train']['brightness'].shape}")
    print(f"  Labels range: {data_splits['train']['labels'].min()} - {data_splits['train']['labels'].max()}")
    print(f"  Number of classes: {len(np.unique(data_splits['train']['labels']))}")
    
    return data_splits

def test_dense_model(data):
    """Test the Dense multi-channel model."""
    print("\n" + "="*60)
    print("TESTING DENSE MODEL")
    print("="*60)
    
    # Model parameters
    num_classes = len(np.unique(data['train']['labels']))
    
    # Flatten image data for dense model (it expects flattened input)
    train_color_flat = data['train']['color'].reshape(data['train']['color'].shape[0], -1)
    train_brightness_flat = data['train']['brightness'].reshape(data['train']['brightness'].shape[0], -1)
    val_color_flat = data['validation']['color'].reshape(data['validation']['color'].shape[0], -1)
    val_brightness_flat = data['validation']['brightness'].reshape(data['validation']['brightness'].shape[0], -1)
    test_color_flat = data['test']['color'].reshape(data['test']['color'].shape[0], -1)
    test_brightness_flat = data['test']['brightness'].reshape(data['test']['brightness'].shape[0], -1)
    
    print(f"🏗️  Building Dense model...")
    print(f"   Color input size: {train_color_flat.shape[1]}")
    print(f"   Brightness input size: {train_brightness_flat.shape[1]}")
    print(f"   Number of classes: {num_classes}")
    
    # Test both fusion strategies
    for use_shared in [True, False]:
        fusion_name = "Shared Classifier" if use_shared else "Separate Classifiers"
        print(f"\n📊 Testing {fusion_name} fusion...")
        
        # Create model
        model = BaseMultiChannelNetwork(
            color_input_size=train_color_flat.shape[1],
            brightness_input_size=train_brightness_flat.shape[1],
            hidden_sizes=[256, 128],  # Smaller for testing
            num_classes=num_classes,
            use_shared_classifier=use_shared,
            dropout=0.1,
            device='cpu'  # Use CPU for testing
        )
        
        print(f"✓ Model created with {model.fusion_type} fusion")
        
        # Test forward pass
        with torch.no_grad():
            # Convert to tensors
            color_tensor = torch.FloatTensor(train_color_flat[:4])
            brightness_tensor = torch.FloatTensor(train_brightness_flat[:4])
            
            # Test forward method
            output = model(color_tensor, brightness_tensor)
            print(f"  Forward pass: {output.shape} -> Expected: (4, {num_classes})")
            assert output.shape == (4, num_classes), f"Wrong output shape: {output.shape}"
            
            # Test analyze_pathways method
            color_out, brightness_out = model.analyze_pathways(color_tensor, brightness_tensor)
            print(f"  Pathway analysis: color {color_out.shape}, brightness {brightness_out.shape}")
            assert color_out.shape == (4, num_classes), f"Wrong color analysis shape: {color_out.shape}"
            assert brightness_out.shape == (4, num_classes), f"Wrong brightness analysis shape: {brightness_out.shape}"
        
        # Test training for a few epochs
        print(f"🚀 Training {fusion_name} model...")
        model.fit(
            train_color_data=train_color_flat,
            train_brightness_data=train_brightness_flat,
            train_labels=data['train']['labels'],
            val_color_data=val_color_flat,
            val_brightness_data=val_brightness_flat,
            val_labels=data['validation']['labels'],
            batch_size=32,
            epochs=2,  # Just test training works
            learning_rate=0.001,
            verbose=1
        )
        
        # Test evaluation
        print(f"📈 Evaluating {fusion_name} model...")
        results = model.evaluate(
            test_color_flat,
            test_brightness_flat,
            data['test']['labels'],
            batch_size=32
        )
        print(f"  Test accuracy: {results['accuracy']:.4f}")
        print(f"  Test loss: {results['loss']:.4f}")
        
        # Test prediction
        predictions = model.predict(test_color_flat[:10], test_brightness_flat[:10])
        print(f"  Predictions shape: {predictions.shape} -> Expected: (10,)")
        assert predictions.shape == (10,), f"Wrong predictions shape: {predictions.shape}"
        
        # Test pathway analysis
        pathway_info = model.analyze_pathway_weights()
        print(f"  Pathway weights - Color: {pathway_info['color_pathway']:.3f}, Brightness: {pathway_info['brightness_pathway']:.3f}")
        
        print(f"✅ {fusion_name} Dense model test completed successfully!")

def test_resnet_model(data):
    """Test the ResNet multi-channel model."""
    print("\n" + "="*60)
    print("TESTING RESNET MODEL")
    print("="*60)
    
    # Model parameters
    num_classes = len(np.unique(data['train']['labels']))
    
    # ResNet expects image format (batch, channels, height, width)
    train_color = data['train']['color']
    train_brightness = data['train']['brightness']
    val_color = data['validation']['color']
    val_brightness = data['validation']['brightness']
    test_color = data['test']['color']
    test_brightness = data['test']['brightness']
    
    print(f"🏗️  Building ResNet model...")
    print(f"   Color input shape: {train_color.shape}")
    print(f"   Brightness input shape: {train_brightness.shape}")
    print(f"   Number of classes: {num_classes}")
    
    # Test both fusion strategies
    for use_shared in [True, False]:
        fusion_name = "Shared Classifier" if use_shared else "Separate Classifiers"
        print(f"\n📊 Testing {fusion_name} fusion...")
        
        # Create model
        model = MultiChannelResNetNetwork(
            num_classes=num_classes,
            color_input_channels=train_color.shape[1],  # 3 for RGB
            brightness_input_channels=train_brightness.shape[1],  # 1 for brightness
            num_blocks=[1, 1, 1, 1],  # Smaller ResNet for testing
            block_type='basic',
            use_shared_classifier=use_shared,
            device='cpu'  # Use CPU for testing
        )
        
        print(f"✓ Model created with {model.fusion_type if hasattr(model, 'fusion_type') else ('shared' if use_shared else 'separate')} fusion")
        
        # Test forward pass
        with torch.no_grad():
            # Convert to tensors
            color_tensor = torch.FloatTensor(train_color[:4])
            brightness_tensor = torch.FloatTensor(train_brightness[:4])
            
            # Test forward method
            output = model(color_tensor, brightness_tensor)
            print(f"  Forward pass: {output.shape} -> Expected: (4, {num_classes})")
            assert output.shape == (4, num_classes), f"Wrong output shape: {output.shape}"
            
            # Test analyze_pathways method
            color_out, brightness_out = model.analyze_pathways(color_tensor, brightness_tensor)
            print(f"  Pathway analysis: color {color_out.shape}, brightness {brightness_out.shape}")
            assert color_out.shape == (4, num_classes), f"Wrong color analysis shape: {color_out.shape}"
            assert brightness_out.shape == (4, num_classes), f"Wrong brightness analysis shape: {brightness_out.shape}"
        
        # Test training for a few epochs
        print(f"🚀 Training {fusion_name} ResNet model...")
        model.fit(
            train_color_data=train_color,
            train_brightness_data=train_brightness,
            train_labels=data['train']['labels'],
            val_color_data=val_color,
            val_brightness_data=val_brightness,
            val_labels=data['validation']['labels'],
            batch_size=16,  # Smaller batch for ResNet testing
            epochs=2,  # Just test training works
            learning_rate=0.001,
            verbose=1
        )
        
        # Test evaluation
        print(f"📈 Evaluating {fusion_name} ResNet model...")
        results = model.evaluate(
            test_color,
            test_brightness,
            data['test']['labels'],
            batch_size=16
        )
        print(f"  Test accuracy: {results['accuracy']:.4f}")
        print(f"  Test loss: {results['loss']:.4f}")
        
        # Test prediction
        predictions = model.predict(test_color[:10], test_brightness[:10], batch_size=4)
        print(f"  Predictions shape: {predictions.shape} -> Expected: (10,)")
        assert predictions.shape == (10,), f"Wrong predictions shape: {predictions.shape}"
        
        # Test pathway analysis
        pathway_info = model.get_pathway_importance()
        print(f"  Pathway importance - Color: {pathway_info['color_pathway']:.3f}, Brightness: {pathway_info['brightness_pathway']:.3f}")
        
        print(f"✅ {fusion_name} ResNet model test completed successfully!")

def test_model_consistency(data):
    """Test that both models produce consistent API behavior."""
    print("\n" + "="*60)
    print("TESTING MODEL CONSISTENCY")
    print("="*60)
    
    num_classes = len(np.unique(data['train']['labels']))
    
    # Create both models with shared classifiers
    print("🔗 Creating both models for consistency testing...")
    
    # Dense model with flattened data
    train_color_flat = data['train']['color'].reshape(data['train']['color'].shape[0], -1)
    train_brightness_flat = data['train']['brightness'].reshape(data['train']['brightness'].shape[0], -1)
    
    dense_model = BaseMultiChannelNetwork(
        color_input_size=train_color_flat.shape[1],
        brightness_input_size=train_brightness_flat.shape[1],
        hidden_sizes=[128, 64],
        num_classes=num_classes,
        use_shared_classifier=True,
        device='cpu'
    )
    
    # ResNet model with image data
    resnet_model = MultiChannelResNetNetwork(
        num_classes=num_classes,
        color_input_channels=data['train']['color'].shape[1],
        brightness_input_channels=data['train']['brightness'].shape[1],
        num_blocks=[1, 1, 1, 1],
        use_shared_classifier=True,
        device='cpu'
    )
    
    print("✓ Both models created successfully")
    
    # Test API consistency
    batch_size = 4
    
    # Dense model test
    with torch.no_grad():
        color_flat = torch.FloatTensor(train_color_flat[:batch_size])
        brightness_flat = torch.FloatTensor(train_brightness_flat[:batch_size])
        
        dense_output = dense_model(color_flat, brightness_flat)
        dense_color, dense_brightness = dense_model.analyze_pathways(color_flat, brightness_flat)
        
        print(f"Dense model API:")
        print(f"  forward(): {dense_output.shape}")
        print(f"  analyze_pathways(): {dense_color.shape}, {dense_brightness.shape}")
    
    # ResNet model test
    with torch.no_grad():
        color_img = torch.FloatTensor(data['train']['color'][:batch_size])
        brightness_img = torch.FloatTensor(data['train']['brightness'][:batch_size])
        
        resnet_output = resnet_model(color_img, brightness_img)
        resnet_color, resnet_brightness = resnet_model.analyze_pathways(color_img, brightness_img)
        
        print(f"ResNet model API:")
        print(f"  forward(): {resnet_output.shape}")
        print(f"  analyze_pathways(): {resnet_color.shape}, {resnet_brightness.shape}")
    
    # Verify consistency
    assert dense_output.shape[1] == resnet_output.shape[1] == num_classes, "Output dimensions should match"
    assert dense_color.shape[1] == resnet_color.shape[1] == num_classes, "Analysis output dimensions should match"
    
    print("✅ Both models have consistent APIs!")

def main():
    """Run the complete end-to-end test."""
    print("🚀 Starting End-to-End Multi-Stream Neural Network Test")
    print("="*60)
    
    try:
        # Test 1: Data loading
        data = test_data_loading()
        
        # Test 2: Dense model
        test_dense_model(data)
        
        # Test 3: ResNet model  
        test_resnet_model(data)
        
        # Test 4: Model consistency
        test_model_consistency(data)
        
        # Success summary
        print("\n" + "🎉" * 60)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("🎉" * 60)
        print("\n✅ Summary of successful tests:")
        print("   ✓ CIFAR-100 data loading and preprocessing")
        print("   ✓ Dense model with shared classifier fusion")
        print("   ✓ Dense model with separate classifier fusion")
        print("   ✓ ResNet model with shared classifier fusion")
        print("   ✓ ResNet model with separate classifier fusion")
        print("   ✓ Model training and evaluation")
        print("   ✓ Prediction and pathway analysis")
        print("   ✓ API consistency between models")
        print("\n🏆 Both models are ready for production use!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
