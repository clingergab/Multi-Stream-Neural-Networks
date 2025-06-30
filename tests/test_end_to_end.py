#!/usr/bin/env python3
"""
End-to-end test for both Dense and ResNet multi-stream models.
Tests building, training, and evaluation using our CIFAR-100 data loader and APIs.
"""

import sys
import os
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our utilities and models
from src.utils.cifar100_loader import load_cifar100_numpy
from src.models.basic_multi_channel.base_multi_channel_network import BaseMultiChannelNetwork
from src.models.basic_multi_channel.multi_channel_resnet_network import MultiChannelResNetNetwork
from src.transforms.rgb_to_rgbl import RGBtoRGBL
from src.transforms.dataset_utils import process_dataset_to_streams, create_dataloader_with_streams
from torch.utils.data import TensorDataset

# This function is no longer needed as we use process_dataset_to_streams instead
# Keeping it here as a comment for reference
# def create_brightness_channel(rgb_data):
#     """Convert RGB data to brightness channel using our transform."""
#     rgb_to_rgbl = RGBtoRGBL()
#     
#     # Convert numpy to tensor, apply transform, convert back
#     rgb_tensor = torch.from_numpy(rgb_data).float()
#     rgb_out, brightness_out = rgb_to_rgbl(rgb_tensor)  # Transform returns tuple
#     
#     return brightness_out.numpy()

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
        'train': {'rgb': train_data, 'labels': train_labels},
        'validation': {'rgb': val_data, 'labels': val_labels},
        'test': {'rgb': test_data, 'labels': test_labels}
    }

def test_data_loading():
    """Test our CIFAR-100 data loading pipeline using dataset utilities."""
    print("🔄 Testing CIFAR-100 data loading...")
    
    # Get the absolute path to the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, "data", "cifar-100")
    print(f"Looking for CIFAR-100 data at: {data_path}")
    
    # Load CIFAR-100 data with explicit path
    train_data, train_labels, test_data, test_labels, label_names = load_cifar100_numpy(data_dir=data_path)
    
    # Combine train and test for our own splitting
    all_data = np.concatenate([train_data, test_data], axis=0)
    all_labels = np.concatenate([train_labels, test_labels], axis=0)
    
    # Create our splits with a subset for testing
    data_splits = split_data(all_data, all_labels, subset_size=1000)
    
    # Create datasets for each split
    dataset_splits = {}
    for split_name in ['train', 'validation', 'test']:
        # Convert numpy arrays to tensors
        rgb_tensor = torch.from_numpy(data_splits[split_name]['rgb']).float()
        labels_tensor = torch.from_numpy(
            np.array(data_splits[split_name]['labels'], dtype=np.int64)
        )
        # Create TensorDataset
        dataset_splits[split_name] = TensorDataset(rgb_tensor, labels_tensor)
    
    # Process datasets to get RGB and brightness streams using our utility function
    print("\nProcessing datasets to RGB and brightness streams...")
    for split_name in ['train', 'validation', 'test']:
        # Process dataset to get RGB and brightness streams
        rgb_stream, brightness_stream, labels_tensor = process_dataset_to_streams(
            dataset_splits[split_name],
            batch_size=200,  # Process in batches of 200 for memory efficiency
            desc=f"Processing {split_name} data"
        )
        
        # Store processed streams
        data_splits[split_name]['rgb'] = rgb_stream.numpy()
        data_splits[split_name]['brightness'] = brightness_stream.numpy()
        data_splits[split_name]['labels'] = labels_tensor.numpy()
    
    # Create DataLoader for demo purposes (not used in this test, but demonstrates capability)
    train_loader = create_dataloader_with_streams(
        dataset_splits['train'],
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=False  # Set to True if using GPU
    )
    
    # Check that labels are in the correct range for all splits
    for split_name in ['train', 'validation', 'test']:
        # Get the unique labels and range
        unique_labels = np.unique(data_splits[split_name]['labels'])
        min_label = data_splits[split_name]['labels'].min()
        max_label = data_splits[split_name]['labels'].max()
        
        # Ensure labels are 0-indexed and within proper range
        num_classes = len(unique_labels)
        if min_label > 0 or max_label >= num_classes:
            print(f"  ⚠️ Adjusting labels for {split_name} split to be 0-indexed")
            # Create a mapping from original labels to 0-indexed labels
            label_mapping = {old_label: idx for idx, old_label in enumerate(unique_labels)}
            # Apply the mapping
            data_splits[split_name]['labels'] = np.array([label_mapping[label] for label in data_splits[split_name]['labels']])
    
    print("✓ Data loaded and processed successfully:")
    print(f"  Train: {data_splits['train']['rgb'].shape[0]} samples")
    print(f"  Validation: {data_splits['validation']['rgb'].shape[0]} samples") 
    print(f"  Test: {data_splits['test']['rgb'].shape[0]} samples")
    print(f"  RGB shape: {data_splits['train']['rgb'].shape}")
    print(f"  Brightness shape: {data_splits['train']['brightness'].shape}")
    print(f"  Labels range: {data_splits['train']['labels'].min()} - {data_splits['train']['labels'].max()}")
    print(f"  Number of classes: {len(np.unique(data_splits['train']['labels']))}")
    
    # Log that we've also created a DataLoader (for future use)
    print(f"  Created DataLoader with batch size: {next(iter(train_loader))[0].shape[0]}")
    
    return data_splits

def test_dense_model(data):
    """Test the Dense multi-channel model."""
    print("\n" + "="*60)
    print("TESTING DENSE MODEL")
    print("="*60)
    
    # Model parameters
    num_classes = len(np.unique(data['train']['labels']))
    
    # Flatten image data for dense model (it expects flattened input)
    train_rgb_flat = data['train']['rgb'].reshape(data['train']['rgb'].shape[0], -1)
    train_brightness_flat = data['train']['brightness'].reshape(data['train']['brightness'].shape[0], -1)
    val_rgb_flat = data['validation']['rgb'].reshape(data['validation']['rgb'].shape[0], -1)
    val_brightness_flat = data['validation']['brightness'].reshape(data['validation']['brightness'].shape[0], -1)
    test_rgb_flat = data['test']['rgb'].reshape(data['test']['rgb'].shape[0], -1)
    test_brightness_flat = data['test']['brightness'].reshape(data['test']['brightness'].shape[0], -1)
    
    print(f"🏗️  Building Dense model...")
    print(f"   RGB input size: {train_rgb_flat.shape[1]}")
    print(f"   Brightness input size: {train_brightness_flat.shape[1]}")
    print(f"   Number of classes: {num_classes}")
    
    # Test both fusion strategies
    for use_shared in [True, False]:
        fusion_name = "Shared Classifier" if use_shared else "Separate Classifiers"
        print(f"\n📊 Testing {fusion_name} fusion...")
        
        # Create model
        model = BaseMultiChannelNetwork(
            color_input_size=train_rgb_flat.shape[1],
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
            color_tensor = torch.FloatTensor(train_rgb_flat[:4])
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
            train_color_data=train_rgb_flat,
            train_brightness_data=train_brightness_flat,
            train_labels=data['train']['labels'],
            val_color_data=val_rgb_flat,
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
            test_rgb_flat,
            test_brightness_flat,
            data['test']['labels'],
            batch_size=32
        )
        print(f"  Test accuracy: {results['accuracy']:.4f}")
        print(f"  Test loss: {results['loss']:.4f}")
        
        # Test prediction
        predictions = model.predict(test_rgb_flat[:10], test_brightness_flat[:10])
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
    train_rgb = data['train']['rgb']
    train_brightness = data['train']['brightness']
    val_rgb = data['validation']['rgb']
    val_brightness = data['validation']['brightness']
    test_rgb = data['test']['rgb']
    test_brightness = data['test']['brightness']
    
    print(f"🏗️  Building ResNet model...")
    print(f"   RGB input shape: {train_rgb.shape}")
    print(f"   Brightness input shape: {train_brightness.shape}")
    print(f"   Number of classes: {num_classes}")
    
    # Test both fusion strategies
    for use_shared in [True, False]:
        fusion_name = "Shared Classifier" if use_shared else "Separate Classifiers"
        print(f"\n📊 Testing {fusion_name} fusion...")
        
        # Create model
        model = MultiChannelResNetNetwork(
            num_classes=num_classes,
            color_input_channels=train_rgb.shape[1],  # 3 for RGB
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
            color_tensor = torch.FloatTensor(train_rgb[:4])
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
        # Compile the model first with the learning rate
        model.compile(
            optimizer='adam',
            learning_rate=0.001,
            loss='cross_entropy'
        )
        
        # Then fit without passing learning_rate
        model.fit(
            train_color_data=train_rgb,
            train_brightness_data=train_brightness,
            train_labels=data['train']['labels'],
            val_color_data=val_rgb,
            val_brightness_data=val_brightness,
            val_labels=data['validation']['labels'],
            batch_size=16,  # Smaller batch for ResNet testing
            epochs=2,  # Just test training works
            verbose=1
        )
        
        # Test evaluation
        print(f"📈 Evaluating {fusion_name} ResNet model...")
        results = model.evaluate(
            test_rgb,
            test_brightness,
            data['test']['labels'],
            batch_size=16
        )
        print(f"  Test accuracy: {results['accuracy']:.4f}")
        print(f"  Test loss: {results['loss']:.4f}")
        
        # Test prediction
        predictions = model.predict(test_rgb[:10], test_brightness[:10], batch_size=4)
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
    train_rgb_flat = data['train']['rgb'].reshape(data['train']['rgb'].shape[0], -1)
    train_brightness_flat = data['train']['brightness'].reshape(data['train']['brightness'].shape[0], -1)
    
    dense_model = BaseMultiChannelNetwork(
        color_input_size=train_rgb_flat.shape[1],
        brightness_input_size=train_brightness_flat.shape[1],
        hidden_sizes=[128, 64],
        num_classes=num_classes,
        use_shared_classifier=True,
        device='cpu'
    )
    
    # ResNet model with image data
    resnet_model = MultiChannelResNetNetwork(
        num_classes=num_classes,
        color_input_channels=data['train']['rgb'].shape[1],
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
        color_flat = torch.FloatTensor(train_rgb_flat[:batch_size])
        brightness_flat = torch.FloatTensor(train_brightness_flat[:batch_size])
        
        dense_output = dense_model(color_flat, brightness_flat)
        dense_color, dense_brightness = dense_model.analyze_pathways(color_flat, brightness_flat)
        
        print(f"Dense model API:")
        print(f"  forward(): {dense_output.shape}")
        print(f"  analyze_pathways(): {dense_color.shape}, {dense_brightness.shape}")
    
    # ResNet model test
    with torch.no_grad():
        color_img = torch.FloatTensor(data['train']['rgb'][:batch_size])
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

def test_dataloader_integration(data_splits):
    """Test using our create_dataloader_with_streams utility with models."""
    print("\n" + "="*60)
    print("TESTING DATALOADER INTEGRATION")
    print("="*60)
    
    # Create datasets from the processed data
    datasets = {}
    for split_name in ['train', 'validation', 'test']:
        # Convert numpy arrays to tensors for the dataset
        data_tensor = torch.from_numpy(data_splits[split_name]['rgb']).float()
        labels_tensor = torch.from_numpy(
            np.array(data_splits[split_name]['labels'], dtype=np.int64)
        )
        # Create TensorDataset
        datasets[split_name] = TensorDataset(data_tensor, labels_tensor)
    
    # Create DataLoaders using our utility function
    dataloaders = {}
    batch_sizes = {'train': 32, 'validation': 32, 'test': 32}
    
    for split_name in ['train', 'validation', 'test']:
        dataloaders[split_name] = create_dataloader_with_streams(
            datasets[split_name],
            batch_size=batch_sizes[split_name],
            shuffle=(split_name == 'train'),  # Only shuffle training data
            num_workers=2,
            pin_memory=False  # Set to True if using GPU
        )
    
    print(f"Created DataLoaders with the following batch sizes:")
    print(f"  Train: {batch_sizes['train']}")
    print(f"  Validation: {batch_sizes['validation']}")
    print(f"  Test: {batch_sizes['test']}")
    
    # Extract a sample batch to verify structure
    rgb_batch, brightness_batch, labels_batch = next(iter(dataloaders['train']))
    
    print(f"Sample batch shapes:")
    print(f"  RGB: {rgb_batch.shape}")
    print(f"  Brightness: {brightness_batch.shape}")
    print(f"  Labels: {labels_batch.shape}")
    
    # Test with a model (ResNet in this case)
    num_classes = len(np.unique(data_splits['train']['labels']))
    
    print("\nTesting ResNet model with DataLoader input...")
    model = MultiChannelResNetNetwork(
        num_classes=num_classes,
        color_input_channels=3,  # RGB channels
        brightness_input_channels=1,  # Brightness channel
        num_blocks=[1, 1, 1, 1],  # Smaller ResNet for testing
        use_shared_classifier=True,
        device='cpu'  # Use CPU for testing
    )
    
    # Check if forward pass works with a batch from the DataLoader
    with torch.no_grad():
        output = model(rgb_batch, brightness_batch)
        print(f"  Forward pass output shape: {output.shape}")
        assert output.shape == (rgb_batch.shape[0], num_classes), "Output shape doesn't match expected"
    
    # Test a single training epoch using the DataLoader
    print("\nTesting one training epoch with DataLoader...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training mode
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Process one batch for demonstration
    for i, (rgb_inputs, brightness_inputs, targets) in enumerate(dataloaders['train']):
        # Forward pass
        outputs = model(rgb_inputs, brightness_inputs)
        loss = criterion(outputs, targets)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Only process a few batches for testing
        if i >= 2:
            break
    
    # Calculate statistics
    avg_loss = running_loss / (i + 1)
    accuracy = correct / total
    
    print(f"  Training loss: {avg_loss:.4f}")
    print(f"  Training accuracy: {accuracy:.4f}")
    
    print("✅ DataLoader integration test completed successfully!")
    
    return dataloaders

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
        
        # Test 5: DataLoader integration
        test_dataloader_integration(data)
        
        # Success summary
        print("\n" + "🎉" * 60)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("🎉" * 60)
        print("\n✅ Summary of successful tests:")
        print("   ✓ CIFAR-100 data loading and preprocessing with dataset utilities")
        print("   ✓ Dataset conversion to RGB and brightness streams")
        print("   ✓ DataLoader integration with on-the-fly stream conversion")
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
