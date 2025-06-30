"""
Test script to verify different autocast usage patterns with PyTorch
This will help determine the correct syntax for your specific PyTorch version
"""

import torch
import sys

def test_autocast_variations():
    print(f"PyTorch version: {torch.__version__}")
    print(f"Python version: {sys.version}\n")
    
    # Create a simple tensor to use in our tests
    x = torch.randn(2, 3, 4, 4)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        x = x.to(device)
        print("Running on CUDA")
    else:
        device = torch.device("cpu")
        print("Running on CPU")
    
    print("\nTesting different autocast import and usage patterns:")
    
    # Test 1: torch.amp.autocast with device_type
    print("\nTest 1: torch.amp.autocast with device_type")
    try:
        from torch.amp import autocast
        with autocast(device_type=device.type):
            print("  Success: autocast with device_type=device.type works")
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")
    
    # Test 2: torch.amp.autocast with device type as first arg
    print("\nTest 2: torch.amp.autocast with device type as first arg")
    try:
        from torch.amp import autocast
        with autocast(device.type):
            print("  Success: autocast with device type as first arg works")
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")
    
    # Test 3: torch.amp.autocast with no arguments
    print("\nTest 3: torch.amp.autocast with no arguments")
    try:
        from torch.amp import autocast
        with autocast():
            print("  Success: autocast with no arguments works")
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")
    
    # Test 4: torch.cuda.amp.autocast (legacy)
    print("\nTest 4: torch.cuda.amp.autocast (legacy)")
    try:
        from torch.cuda.amp import autocast
        with autocast():
            print("  Success: torch.cuda.amp.autocast with no arguments works")
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")
    
    # Test 5: torch.cuda.amp.autocast with device_type (legacy mixed with new)
    print("\nTest 5: torch.cuda.amp.autocast with device_type (legacy mixed with new)")
    try:
        from torch.cuda.amp import autocast
        with autocast(device_type=device.type):
            print("  Success: torch.cuda.amp.autocast with device_type works")
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")
    
    # Test 6: String literal for device type
    print("\nTest 6: String literal for device type")
    try:
        from torch.amp import autocast
        with autocast('cuda' if device.type == 'cuda' else 'cpu'):
            print("  Success: autocast with string literal device type works")
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")
    
    # Test 7: String literal with device_type parameter
    print("\nTest 7: String literal with device_type parameter")
    try:
        from torch.amp import autocast
        with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
            print("  Success: autocast with string literal in device_type parameter works")
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")
    
    # Test 8: Direct torch.autocast as mentioned in the docs
    print("\nTest 8: Direct torch.autocast import")
    try:
        with torch.autocast(device.type):
            print("  Success: torch.autocast(device.type) works")
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")
    
    # Test 9: Direct torch.autocast with string literal
    print("\nTest 9: Direct torch.autocast with string literal")
    try:
        with torch.autocast('cuda' if device.type == 'cuda' else 'cpu'):
            print("  Success: torch.autocast with string literal works")
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")
    
    # Test 10: Direct torch.autocast with device_type parameter
    print("\nTest 10: Direct torch.autocast with device_type parameter")
    try:
        with torch.autocast(device_type=device.type):
            print("  Success: torch.autocast with device_type parameter works")
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")
    
    # Test GradScaler variations
    print("\n\nTesting different GradScaler import and usage patterns:")
    
    # Test 11: torch.amp.GradScaler
    print("\nTest 11: torch.amp.GradScaler")
    try:
        from torch.amp import GradScaler
        scaler = GradScaler()
        print("  Success: torch.amp.GradScaler() works")
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")
    
    # Test 12: Direct torch.GradScaler
    print("\nTest 12: Direct torch.GradScaler")
    try:
        scaler = torch.GradScaler()
        print("  Success: torch.GradScaler() works")
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")
    
    # Test 13: torch.GradScaler with device type
    print("\nTest 13: torch.GradScaler with device type")
    try:
        scaler = torch.GradScaler(device_type=device.type)
        print("  Success: torch.GradScaler(device_type=device.type) works")
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")
    
    # Test 14: torch.GradScaler with string literal
    print("\nTest 14: torch.GradScaler with string literal")
    try:
        scaler = torch.GradScaler('cuda' if device.type == 'cuda' else 'cpu')
        print("  Success: torch.GradScaler with string literal works")
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_autocast_variations()
