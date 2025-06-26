#!/usr/bin/env python3
import sys
import inspect
sys.path.append('.')

print("🔍 Verifying GPU optimizations...")

# Check imports
try:
    from src.models.basic_multi_channel.base_multi_channel_network import BaseMultiChannelNetwork
    from src.models.basic_multi_channel.multi_channel_resnet_network import MultiChannelResNetNetwork
    print("✅ Imports successful")
except Exception as e:
    print(f"❌ Import error: {e}")
    exit(1)

# Check BaseMultiChannelNetwork optimizations
print("\n📊 BaseMultiChannelNetwork optimizations:")
model = BaseMultiChannelNetwork(color_input_size=784, brightness_input_size=784, num_classes=10)

# Check mixed precision
print(f"   Mixed precision: {model.use_mixed_precision}")
print(f"   Scaler available: {model.scaler is not None}")
print(f"   Device: {model.device}")

# Check fit method signature
fit_sig = inspect.signature(model.fit)
fit_params = list(fit_sig.parameters.keys())
required_params = ['batch_size', 'num_workers', 'pin_memory']
for param in required_params:
    if param in fit_params:
        print(f"   ✅ {param} parameter in fit()")
    else:
        print(f"   ❌ {param} parameter missing")

print("\n📊 MultiChannelResNetNetwork optimizations:")
try:
    resnet_model = MultiChannelResNetNetwork(num_classes=10, num_blocks=[2,2,2,2])
    print(f"   Mixed precision: {resnet_model.use_mixed_precision}")
    print(f"   Device: {resnet_model.device}")
    print("   ✅ ResNet model created successfully")
except Exception as e:
    print(f"   ❌ ResNet error: {e}")

print("\n✅ Optimization verification complete!")
