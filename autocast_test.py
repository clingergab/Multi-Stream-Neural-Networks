import torch

# Test different import patterns for autocast and GradScaler
print(f"PyTorch version: {torch.__version__}")

# Test 1: Import from torch.cuda directly (older pattern)
try:
    from torch.cuda import autocast as cuda_autocast
    from torch.cuda.amp import GradScaler as CudaGradScaler
    print("✅ Can import from torch.cuda.amp")
except ImportError:
    print("❌ Cannot import from torch.cuda.amp")

# Test 2: Import from torch directly (newer pattern)
try:
    from torch import autocast, GradScaler
    print("✅ Can import from torch directly")
except ImportError:
    print("❌ Cannot import from torch directly")

# Test 3: Import from torch.amp (current pattern)
try:
    from torch.amp import autocast as amp_autocast
    from torch.amp import GradScaler as AmpGradScaler
    print("✅ Can import from torch.amp")
except ImportError:
    print("❌ Cannot import from torch.amp")

# Create a dummy tensor and model for testing
x = torch.randn(2, 3, 224, 224)
model = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1)
loss_fn = torch.nn.MSELoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\nUsing device: {device}")

# Test different autocast usages
print("\nTesting autocast usages:")

# Test 1: autocast with no arguments (deprecated)
try:
    from torch import autocast
    with autocast():
        out = model(x)
        loss = loss_fn(out, torch.zeros_like(out))
    print("✅ autocast() with no args works")
except TypeError as e:
    print(f"❌ autocast() with no args fails: {e}")

# Test 2: autocast with device_type
try:
    from torch import autocast
    with autocast(device_type=device.type):
        out = model(x)
        loss = loss_fn(out, torch.zeros_like(out))
    print("✅ autocast(device_type=device.type) works")
except Exception as e:
    print(f"❌ autocast(device_type=device.type) fails: {e}")

# Test 3: autocast with cuda (older style)
try:
    from torch.cuda import autocast as cuda_autocast
    with cuda_autocast():
        out = model(x)
        loss = loss_fn(out, torch.zeros_like(out))
    print("✅ torch.cuda.autocast() works")
except Exception as e:
    print(f"❌ torch.cuda.autocast() fails: {e}")

# Test GradScaler initialization and usage
print("\nTesting GradScaler:")

# Test 1: Direct import
try:
    from torch import GradScaler
    scaler = GradScaler()
    print("✅ GradScaler from torch works")
except Exception as e:
    print(f"❌ GradScaler from torch fails: {e}")

# Test 2: From torch.cuda.amp
try:
    from torch.cuda.amp import GradScaler as CudaGradScaler
    scaler = CudaGradScaler()
    print("✅ GradScaler from torch.cuda.amp works")
except Exception as e:
    print(f"❌ GradScaler from torch.cuda.amp fails: {e}")

# Test 3: From torch.amp
try:
    from torch.amp import GradScaler as AmpGradScaler
    scaler = AmpGradScaler()
    print("✅ GradScaler from torch.amp works")
except Exception as e:
    print(f"❌ GradScaler from torch.amp fails: {e}")

print("\nTest complete.")
