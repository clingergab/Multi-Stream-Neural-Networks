"""
Test if scheduler is properly created and configured in MCResNet.
"""

import torch
from src.models.multi_channel.mc_resnet import mc_resnet18

print("=" * 80)
print("TESTING SCHEDULER CREATION IN MCRESNET")
print("=" * 80)

# Create and compile model
model = mc_resnet18(
    num_classes=15,
    stream1_input_channels=3,
    stream2_input_channels=1,
    fusion_type='concat',
    device='cpu',
    use_amp=False
)

print("\n1. After model creation:")
print(f"   scheduler = {model.scheduler}")
print(f"   scheduler_type = {model.scheduler_type}")

# Compile
model.compile(
    optimizer='adamw',
    learning_rate=1e-4,
    stream1_lr=2e-4,
    stream2_lr=5e-5,
    weight_decay=1e-3,
    loss='cross_entropy',
    scheduler='cosine'  # Request cosine scheduler
)

print("\n2. After compile:")
print(f"   scheduler = {model.scheduler}")
print(f"   scheduler_type = {model.scheduler_type}")

print("\n" + "=" * 80)
print("ISSUE IDENTIFIED")
print("=" * 80)

if model.scheduler is None:
    print("\n❌ BUG CONFIRMED!")
    print("   Scheduler is None after compile().")
    print("   The scheduler is only created in fit(), NOT in compile()!")
    print("\n📝 This is by design in the current implementation:")
    print("   - compile() stores scheduler_type")
    print("   - fit() creates the actual scheduler")
    print("\n🔍 From abstract_model.py:378:")
    print('   # Scheduler will be configured in fit() with actual training parameters')
    print('   self.scheduler = None')
    print("\n✅ GOOD NEWS:")
    print("   Once scheduler is created in fit(), it WILL update all parameter groups.")
    print("   PyTorch schedulers handle multiple parameter groups correctly.")
else:
    print("\n✅ Scheduler was created in compile()")
    print(f"   Type: {type(model.scheduler)}")

print("\n" + "=" * 80)
