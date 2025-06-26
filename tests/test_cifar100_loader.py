#!/usr/bin/env python3
"""
Test script for the CIFAR-100 loader module.
Demonstrates importing and using the utilities from the project structure.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import the CIFAR-100 utilities
from src.utils.cifar100_loader import (
    get_cifar100_datasets,
    load_cifar100_raw,
    CIFAR100_FINE_LABELS
)

def test_cifar100_loading():
    """Test the CIFAR-100 loading functionality."""
    print("🔍 Testing CIFAR-100 utilities from project module...")
    
    try:
        # Test using the high-level function
        print("\n1️⃣ Testing get_cifar100_datasets()...")
        train_dataset, test_dataset, class_names = get_cifar100_datasets()
        
        # Test a sample
        sample = train_dataset[0]
        image, label = sample
        print(f"\n✅ Sample verification:")
        print(f"   Image shape: {image.shape}")
        print(f"   Label: {label} ({class_names[label]})")
        print(f"   Image range: [{image.min():.3f}, {image.max():.3f}]")
        
        # Test using the raw loading function
        print("\n2️⃣ Testing load_cifar100_raw()...")
        train_data, train_labels, test_data, test_labels, label_names = load_cifar100_raw()
        
        print(f"\n✅ Raw loading verification:")
        print(f"   Train data type: {type(train_data)}")
        print(f"   Train data shape: {train_data.shape}")
        print(f"   Train data dtype: {train_data.dtype}")
        print(f"   First label: {train_labels[0]} ({label_names[train_labels[0]]})")
        
        # Test class names constant
        print("\n3️⃣ Testing CIFAR100_FINE_LABELS constant...")
        print(f"   Total classes: {len(CIFAR100_FINE_LABELS)}")
        print(f"   First 5 classes: {CIFAR100_FINE_LABELS[:5]}")
        print(f"   Last 5 classes: {CIFAR100_FINE_LABELS[-5:]}")
        
        print("\n✅ All CIFAR-100 utilities working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_cifar100_loading()
    
    if success:
        print("\n🎉 CIFAR-100 utilities are ready for use in the notebook!")
        print("💡 You can now import them in the notebook with:")
        print("   from src.utils.cifar100_loader import get_cifar100_datasets, CIFAR100_FINE_LABELS")
    else:
        print("\n❌ CIFAR-100 utilities test failed.")
