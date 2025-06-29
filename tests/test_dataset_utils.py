"""Test script for dataset_utils.py functions."""

import sys
import torch
from torch.utils.data import TensorDataset
from pathlib import Path

# Add project root to path for imports
project_root = Path('.').resolve()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import the utilities to test
from src.transforms.dataset_utils import process_dataset_to_streams, create_dataloader_with_streams

print("✅ Successfully imported utilities")

# Create a small synthetic dataset for testing
try:
    # Create random data tensors
    num_samples = 100
    input_shape = (3, 32, 32)  # RGB images
    
    # Generate random RGB tensors
    rgb_data = torch.rand(num_samples, *input_shape)
    # Generate random labels
    labels = torch.randint(0, 10, (num_samples,))
    
    # Create a TensorDataset
    test_dataset = TensorDataset(rgb_data, labels)
    
    print(f"✅ Created synthetic test dataset with {len(test_dataset)} samples")
    
    # Test process_dataset_to_streams
    print("\nTesting process_dataset_to_streams...")
    rgb_stream, brightness_stream, labels = process_dataset_to_streams(
        test_dataset, 
        batch_size=20,
        desc="Processing test data"
    )
    
    print(f"RGB stream shape: {rgb_stream.shape}")
    print(f"Brightness stream shape: {brightness_stream.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Test create_dataloader_with_streams
    print("\nTesting create_dataloader_with_streams...")
    dataloader = create_dataloader_with_streams(
        test_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,  # Use 0 workers to avoid multiprocessing issues
        pin_memory=False
    )
    
    # Test a batch from the dataloader
    print("Getting a batch from the dataloader...")
    rgb_batch, brightness_batch, labels_batch = next(iter(dataloader))
    
    print(f"RGB batch shape: {rgb_batch.shape}")
    print(f"Brightness batch shape: {brightness_batch.shape}")
    print(f"Labels batch shape: {labels_batch.shape}")
    
    print("\n✅ All tests passed successfully!")
    
except Exception as e:
    print(f"❌ Error during testing: {e}")
    import traceback
    traceback.print_exc()
