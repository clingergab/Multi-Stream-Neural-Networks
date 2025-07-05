#!/usr/bin/env python3
"""
Quick test script to verify MPS (Apple Silicon) functionality.

This script checks if MPS is available and working correctly
before running the full diagnostics.
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def test_device_availability():
    """Test device availability and basic operations."""
    print("🔍 Testing Device Availability")
    print("=" * 40)
    
    # Check MPS availability
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("✅ MPS (Apple Silicon GPU) is available")
        mps_device = torch.device("mps")
    else:
        print("❌ MPS not available")
        mps_device = None
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print("✅ CUDA is available")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        cuda_device = torch.device("cuda")
    else:
        print("❌ CUDA not available")
        cuda_device = None
    
    # CPU is always available
    print("✅ CPU is available")
    cpu_device = torch.device("cpu")
    
    return mps_device, cuda_device, cpu_device


def test_basic_operations(device):
    """Test basic PyTorch operations on the specified device."""
    print(f"\n🧪 Testing Basic Operations on {device}")
    print("-" * 40)
    
    try:
        # Create tensors
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        
        # Matrix multiplication
        z = torch.mm(x, y)
        print("✅ Matrix multiplication successful")
        
        # Gradient computation
        x.requires_grad_(True)
        loss = (x ** 2).sum()
        loss.backward()
        print("✅ Gradient computation successful")
        
        # Memory usage
        if device.type == "mps":
            print(f"✅ MPS operation completed successfully")
        elif device.type == "cuda":
            print(f"✅ CUDA memory: {torch.cuda.memory_allocated(device) / 1024**2:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during operations: {e}")
        return False


def test_model_creation():
    """Test creating and running a simple model."""
    print(f"\n🏗️ Testing Model Creation and Forward Pass")
    print("-" * 40)
    
    # Determine best device
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍎 Using MPS device")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🎮 Using CUDA device")
    else:
        device = torch.device("cpu")
        print("💻 Using CPU device")
    
    try:
        # Create a simple model
        model = torch.nn.Sequential(
            torch.nn.Linear(100, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 10)
        ).to(device)
        
        # Test forward pass
        x = torch.randn(32, 100, device=device)
        output = model(x)
        
        print(f"✅ Model creation successful")
        print(f"✅ Forward pass successful - Output shape: {output.shape}")
        
        # Test backward pass
        loss = output.sum()
        loss.backward()
        print("✅ Backward pass successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during model operations: {e}")
        return False


def test_data_loading():
    """Test data loading optimization for different devices."""
    print(f"\n📊 Testing Data Loading Optimizations")
    print("-" * 40)
    
    try:
        from torch.utils.data import DataLoader, TensorDataset
        
        # Create dummy data
        data = torch.randn(1000, 3, 32, 32)
        labels = torch.randint(0, 10, (1000,))
        dataset = TensorDataset(data, labels)
        
        # Determine best device
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            # MPS optimizations
            num_workers = 0
            pin_memory = False
            print("🍎 Using MPS-optimized DataLoader settings")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            # CUDA optimizations
            num_workers = 4
            pin_memory = True
            print("🎮 Using CUDA-optimized DataLoader settings")
        else:
            device = torch.device("cpu")
            num_workers = 4
            pin_memory = False
            print("💻 Using CPU DataLoader settings")
        
        # Create DataLoader
        loader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        # Test a few batches
        for i, (batch_data, batch_labels) in enumerate(loader):
            if i >= 3:  # Test only 3 batches
                break
                
            # Move to device
            if device.type == "mps":
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
            else:
                batch_data = batch_data.to(device, non_blocking=True)
                batch_labels = batch_labels.to(device, non_blocking=True)
        
        print(f"✅ DataLoader test successful")
        print(f"   Device: {device}")
        print(f"   Num workers: {num_workers}")
        print(f"   Pin memory: {pin_memory}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during data loading test: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 MPS Functionality Test Suite")
    print("=" * 50)
    print(f"PyTorch version: {torch.__version__}")
    print()
    
    # Test device availability
    mps_device, cuda_device, cpu_device = test_device_availability()
    
    # Test operations on available devices
    success_count = 0
    total_tests = 0
    
    for device in [mps_device, cuda_device, cpu_device]:
        if device is not None:
            total_tests += 1
            if test_basic_operations(device):
                success_count += 1
    
    # Test model creation
    total_tests += 1
    if test_model_creation():
        success_count += 1
    
    # Test data loading
    total_tests += 1
    if test_data_loading():
        success_count += 1
    
    # Summary
    print(f"\n📋 Test Summary")
    print("=" * 30)
    print(f"Passed: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("🎉 All tests passed! Ready to run comprehensive diagnostics.")
        
        # Recommend optimal settings
        if mps_device:
            print("\n💡 Recommended settings for your Mac:")
            print("   --device auto")
            print("   --batch-size 16  (start here, increase if memory allows)")
        elif cuda_device:
            print("\n💡 Recommended settings for your system:")
            print("   --device auto")
            print("   --batch-size 32")
        else:
            print("\n💡 Recommended settings for CPU:")
            print("   --device cpu")
            print("   --batch-size 16")
            print("   --epochs 5  (reduce for faster testing)")
    else:
        print("⚠️  Some tests failed. Check error messages above.")
        print("Consider using CPU fallback: --device cpu")


if __name__ == "__main__":
    main()
