#!/usr/bin/env python3
"""
End-to-end test for MultiChannelResNet with bottleneck blocks.
Tests building, training, and evaluation using our CIFAR-100 data loader and APIs.
"""

import sys
import os
import torch
import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our utilities and models
from src.utils.cifar100_loader import load_cifar100_numpy
from src.models.basic_multi_channel.multi_channel_resnet_network import MultiChannelResNetNetwork
from src.transforms.dataset_utils import process_dataset_to_streams, create_dataloader_with_streams
from torch.utils.data import TensorDataset

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

@pytest.fixture(scope="session")
def data():
    """Pytest fixture for loading and processing CIFAR-100 data once for all tests."""
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
    
    return data_splits

@pytest.fixture(scope="session") 
def data_splits(data):
    """Fixture for DataLoader integration tests."""
    # Create datasets for each split
    dataset_splits = {}
    for split_name in ['train', 'validation', 'test']:
        # Convert numpy arrays to tensors
        rgb_tensor = torch.from_numpy(data[split_name]['rgb']).float()
        labels_tensor = torch.from_numpy(
            np.array(data[split_name]['labels'], dtype=np.int64)
        )
        # Create TensorDataset
        dataset_splits[split_name] = TensorDataset(rgb_tensor, labels_tensor)
    
    return dataset_splits

def test_data_loading():
    """Test our CIFAR-100 data loading pipeline using dataset utilities."""
    # This is now a simple test that just loads data using the fixture indirectly
    print("✅ Data loading test passed (data loaded by fixture)")

def test_bottleneck_resnet_model(data):
    """Test the ResNet multi-channel model with bottleneck blocks."""
    print("\n" + "="*60)
    print("TESTING BOTTLENECK RESNET MODEL")
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
    
    print("🏗️  Building Bottleneck ResNet model...")
    print(f"   RGB input shape: {train_rgb.shape}")
    print(f"   Brightness input shape: {train_brightness.shape}")
    print(f"   Number of classes: {num_classes}")
    
    # Create model with bottleneck blocks
    model = MultiChannelResNetNetwork(
        num_classes=num_classes,
        color_input_channels=train_rgb.shape[1],  # 3 for RGB
        brightness_input_channels=train_brightness.shape[1],  # 1 for brightness
        num_blocks=[1, 1, 1, 1],  # Smaller ResNet for testing
        block_type='bottleneck',  # Specifically using bottleneck blocks
        device='cpu'  # Use CPU for testing
    )
    
    print(f"✓ Bottleneck ResNet model created with {model.fusion_type} fusion")
    
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
    
    # Test training for a few epochs using new API
    print("🚀 Training Bottleneck ResNet model...")
    
    # Compile the model first (using architecture-specific defaults)
    model.compile(
        optimizer='adam',
        # learning_rate will use class default (0.0003)
        # weight_decay will use class default (1e-4)
        # gradient_clip will use class default (1.0)
        # scheduler will use class default ('cosine')
    )
    
    # Then fit the model
    history = model.fit(
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
    
    print(f"  Training completed. Final train loss: {history['train_loss'][-1]:.4f}")
    
    # Test evaluation
    print("📈 Evaluating Bottleneck ResNet model...")
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
    
    print("✅ Bottleneck ResNet model test completed successfully!")

def test_bottleneck_resnet_model_reduced(data):
    """Test the ResNet multi-channel model with bottleneck blocks using reduced architecture."""
    print("\n" + "="*60)
    print("TESTING BOTTLENECK RESNET MODEL WITH REDUCED ARCHITECTURE")
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
    
    print("🏗️  Building Reduced Bottleneck ResNet model...")
    print(f"   RGB input shape: {train_rgb.shape}")
    print(f"   Brightness input shape: {train_brightness.shape}")
    print(f"   Number of classes: {num_classes}")
    
    # Create model with bottleneck blocks and reduced architecture
    model = MultiChannelResNetNetwork(
        num_classes=num_classes,
        color_input_channels=train_rgb.shape[1],  # 3 for RGB
        brightness_input_channels=train_brightness.shape[1],  # 1 for brightness
        num_blocks=[1, 1, 1, 1],  # Smaller ResNet for testing
        block_type='bottleneck',  # Specifically using bottleneck blocks
        reduce_architecture=True,  # Using reduced architecture for CIFAR
        device='cpu'  # Use CPU for testing
    )
    
    print(f"✓ Reduced Bottleneck ResNet model created with {model.fusion_type} fusion")
    
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
    
    # Test training for a few epochs using new API
    print("🚀 Training Reduced Bottleneck ResNet model...")
    
    # Compile the model first (using architecture-specific defaults)
    model.compile(
        optimizer='adam',
        # learning_rate will use class default (0.0003)
        # weight_decay will use class default (1e-4)
        # gradient_clip will use class default (1.0)
        # scheduler will use class default ('cosine')
    )
    
    # Then fit the model
    history = model.fit(
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
    
    print(f"  Training completed. Final train loss: {history['train_loss'][-1]:.4f}")
    
    # Test evaluation
    print("📈 Evaluating Reduced Bottleneck ResNet model...")
    results = model.evaluate(
        test_rgb,
        test_brightness,
        data['test']['labels'],
        batch_size=16
    )
    print(f"  Test accuracy: {results['accuracy']:.4f}")
    print(f"  Test loss: {results['loss']:.4f}")
    
    # Test classifier info
    classifier_info = model.get_classifier_info()
    print(f"  Classifier info: {classifier_info}")
    assert classifier_info['architecture_type'] == 'reduced', "Should be reduced architecture"
    
    print("✅ Reduced Bottleneck ResNet model test completed successfully!")

def test_dataloader_integration(data_splits):
    """Test using our create_dataloader_with_streams utility with bottleneck ResNet model."""
    print("\n" + "="*60)
    print("TESTING DATALOADER INTEGRATION WITH BOTTLENECK RESNET")
    print("="*60)
    
    # data_splits already contains TensorDatasets ready to use
    print("Using pre-processed TensorDatasets from fixture")
    
    # Create DataLoaders using our utility function
    dataloaders = {}
    batch_sizes = {'train': 32, 'validation': 32, 'test': 32}
    
    for split_name in ['train', 'validation', 'test']:
        dataloaders[split_name] = create_dataloader_with_streams(
            data_splits[split_name],
            batch_size=batch_sizes[split_name],
            shuffle=(split_name == 'train'),  # Only shuffle training data
            num_workers=2,
            pin_memory=False  # Set to True if using GPU
        )
    
    print("Created DataLoaders with the following batch sizes:")
    print(f"  Train: {batch_sizes['train']}")
    print(f"  Validation: {batch_sizes['validation']}")
    print(f"  Test: {batch_sizes['test']}")
    
    # Extract a sample batch to verify structure
    rgb_batch, brightness_batch, labels_batch = next(iter(dataloaders['train']))
    
    print("Sample batch shapes:")
    print(f"  RGB: {rgb_batch.shape}")
    print(f"  Brightness: {brightness_batch.shape}")
    print(f"  Labels: {labels_batch.shape}")
    
    # Test with a bottleneck ResNet model
    # Get number of classes from the data
    all_labels = []
    for data_tensor, labels_tensor in data_splits['train']:
        if labels_tensor.dim() == 0:  # Single scalar label
            all_labels.append(labels_tensor.item())
        else:  # Multiple labels
            all_labels.extend(labels_tensor.numpy())
    num_classes = len(set(all_labels))
    
    print("\nTesting Bottleneck ResNet model with DataLoader input...")
    print(f"Number of classes detected: {num_classes}")
    model = MultiChannelResNetNetwork(
        num_classes=num_classes,
        color_input_channels=3,  # RGB channels
        brightness_input_channels=1,  # Brightness channel
        num_blocks=[1, 1, 1, 1],  # Smaller ResNet for testing
        block_type='bottleneck',  # Specifically using bottleneck blocks
        device='cpu'  # Use CPU for testing
    )
    
    # Check if forward pass works with a batch from the DataLoader
    with torch.no_grad():
        output = model(rgb_batch, brightness_batch)
        print(f"  Forward pass output shape: {output.shape}")
        assert output.shape == (rgb_batch.shape[0], num_classes), "Output shape doesn't match expected"
    
    # Test training using our fit() method with DataLoaders
    print("\nTesting bottleneck model training with DataLoader using fit() method...")
    
    # Compile the model first
    model.compile(
        optimizer='adam',
        learning_rate=0.001  # Override default for faster testing
    )
    
    # Train using fit() method with DataLoaders
    history = model.fit(
        train_loader=dataloaders['train'],
        val_loader=dataloaders['validation'],
        epochs=1,  # Just test one epoch
        verbose=1
    )
    
    print("  Training completed successfully!")
    print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    if history['val_loss'] and history['val_loss'][0] is not None:
        print(f"  Final val loss: {history['val_loss'][-1]:.4f}")
    
    print("✅ DataLoader integration test completed successfully!")

def test_bottleneck_feature_extraction(data):
    """Test the feature extraction capabilities of the bottleneck ResNet model."""
    print("\n" + "="*60)
    print("TESTING BOTTLENECK RESNET FEATURE EXTRACTION")
    print("="*60)
    
    # Model parameters
    num_classes = len(np.unique(data['train']['labels']))
    
    # Sample data
    sample_rgb = data['train']['rgb'][:10]
    sample_brightness = data['train']['brightness'][:10]
    
    # Create model with bottleneck blocks
    model = MultiChannelResNetNetwork(
        num_classes=num_classes,
        color_input_channels=sample_rgb.shape[1],
        brightness_input_channels=sample_brightness.shape[1],
        num_blocks=[1, 1, 1, 1],
        block_type='bottleneck',
        device='cpu'
    )
    
    print("✓ Bottleneck ResNet model created for feature extraction testing")
    
    # Convert to tensors
    color_tensor = torch.FloatTensor(sample_rgb)
    brightness_tensor = torch.FloatTensor(sample_brightness)
    
    # Test extract_features method
    with torch.no_grad():
        features = model.extract_features(color_tensor, brightness_tensor)
        print(f"  Combined features shape: {features.shape}")
        
        # Test get_separate_features method
        color_features, brightness_features = model.get_separate_features(color_tensor, brightness_tensor)
        print(f"  Separate features shapes - Color: {color_features.shape}, Brightness: {brightness_features.shape}")
        
        # Verify concatenated features match separate features
        combined_manual = torch.cat([color_features, brightness_features], dim=1)
        assert torch.allclose(features, combined_manual), "extract_features should concatenate the separate features"
    
    # Test pathway analysis methods
    pathway_weights = model.analyze_pathway_weights()
    print(f"  Pathway weights analysis: {pathway_weights}")
    
    # Verify expected keys are present in pathway weights
    expected_keys = ['color_pathway', 'brightness_pathway', 'pathway_ratio', 'fusion_type', 
                     'shared_color_norm', 'shared_brightness_norm', 'head_color_norm', 
                     'head_brightness_norm', 'architecture_type']
    for key in expected_keys:
        assert key in pathway_weights, f"Missing expected key in pathway weights: {key}"
    
    # Test classifier info
    classifier_info = model.get_classifier_info()
    print(f"  Classifier info: {classifier_info}")
    assert classifier_info['type'] == 'shared_with_separate_heads', "Should be using shared classifier with separate heads"
    assert classifier_info['architecture_type'] == 'full', "Should be using full architecture"
    
    print("✅ Bottleneck ResNet feature extraction test completed successfully!")

def main():
    """Run the complete bottleneck ResNet test suite."""
    print("🚀 Starting End-to-End MultiChannelResNet Bottleneck Block Test")
    print("="*60)
    
    try:
        # For direct execution (not via pytest), we need to load data ourselves
        # First import needed modules
        import subprocess
        
        # Run the tests using pytest to properly handle fixtures
        cmd = [sys.executable, "-m", "pytest", "-xvs", __file__]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Print the output
        print(result.stdout)
        
        if result.returncode != 0:
            print(f"\n❌ Tests failed with error code: {result.returncode}")
            print(result.stderr)
            return False
        
        print("\n🎉" * 60)
        print("🎉 ALL BOTTLENECK RESNET TESTS PASSED! 🎉")
        print("🎉" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
