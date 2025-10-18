"""Test gradient flow in ResNet50 to find NaN source."""
import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
from src.models.linear_integration.li_net import li_resnet50

print("Testing ResNet50 gradient flow...")

# Create model
model = li_resnet50(
    num_classes=15,
    stream1_input_channels=3,
    stream2_input_channels=1,
    dropout_p=0.5,
    device='cpu',
    use_amp=False
)
model.train()

# Create optimizer and loss
optimizer = torch.optim.AdamW(model.parameters(), lr=7e-5, weight_decay=2e-4)
criterion = nn.CrossEntropyLoss()

# Small batch
rgb = torch.randn(2, 3, 416, 544)
depth = torch.randn(2, 1, 416, 544)
labels = torch.randint(0, 15, (2,))

print("\nForward pass...")
outputs = model(rgb, depth)
print(f"Outputs: min={outputs.min().item():.4f}, max={outputs.max().item():.4f}")

print("\nComputing loss...")
loss = criterion(outputs, labels)
print(f"Loss: {loss.item():.6f}")

print("\nBackward pass...")
optimizer.zero_grad()
loss.backward()

print("\nChecking gradients in integration weights...")
max_grad = 0
max_grad_name = None

for name, param in model.named_parameters():
    if param.grad is not None and 'integration_from' in name:
        grad_norm = param.grad.norm().item()
        grad_max = param.grad.abs().max().item()
        grad_mean = param.grad.abs().mean().item()

        if grad_max > max_grad:
            max_grad = grad_max
            max_grad_name = name

        if grad_max > 100 or torch.isnan(param.grad).any():
            print(f"\n⚠️  {name}:")
            print(f"    Shape: {param.shape}")
            print(f"    Grad norm: {grad_norm:.6f}")
            print(f"    Grad max: {grad_max:.6f}")
            print(f"    Grad mean: {grad_mean:.6f}")
            print(f"    Has NaN: {torch.isnan(param.grad).any()}")

print(f"\nLargest gradient: {max_grad:.6f} in {max_grad_name}")

if max_grad > 1000:
    print("\n❌ GRADIENT EXPLOSION DETECTED!")
else:
    print("\n✅ Gradients look reasonable")
