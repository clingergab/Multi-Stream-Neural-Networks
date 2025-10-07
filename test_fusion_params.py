from src.models.multi_channel.mc_resnet import mc_resnet18

# Create model with weighted fusion
model = mc_resnet18(
    num_classes=15,
    fusion_type='weighted',
    device='cpu'
)

# Check parameter names
print("All parameter names:")
for name, param in model.named_parameters():
    if 'fusion' in name:
        print(f"  {name}: {param.shape}")

# Compile with stream-specific optimization
model.compile(
    optimizer='adamw',
    learning_rate=1e-4,
    weight_decay=2e-2,
    stream1_lr=5e-5,
    stream2_lr=5e-5,
    stream1_weight_decay=5e-2,
    stream2_weight_decay=5e-2
)

# Check parameter groups
print("\nParameter groups:")
for i, group in enumerate(model.optimizer.param_groups):
    print(f"\nGroup {i}: LR={group['lr']:.2e}, WD={group['weight_decay']:.2e}")
    for param in group['params']:
        # Find parameter name
        for name, p in model.named_parameters():
            if p is param:
                print(f"  - {name}")
                break
