"""Debug scheduler to see if it's stepping correctly."""
import sys
from pathlib import Path
import torch

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models.multi_channel.mc_resnet import mc_resnet18
from src.data_utils.nyu_depth_dataset import create_nyu_dataloaders

print("=" * 80)
print("SCHEDULER DEBUG TEST")
print("=" * 80)

# Load data
print("\n1. Loading dataset...")
train_loader, val_loader = create_nyu_dataloaders(
    h5_file_path='data/content/nyu_depth_v2_labeled.mat',
    batch_size=64,
    num_workers=0,
    num_classes=27
)
print(f"   Train batches: {len(train_loader)}")

# Create model with dropout
print("\n2. Creating model with dropout...")
model = mc_resnet18(
    num_classes=27,
    dropout_p=0.5,
    device='cpu',
    use_amp=False
)
print(f"   Dropout: {model.dropout_p}")

# Compile with label smoothing
print("\n3. Compiling model...")
model.compile(
    optimizer='adamw',
    learning_rate=0.0001,
    weight_decay=2e-2,
    loss='cross_entropy',
    scheduler='cosine',
    label_smoothing=0.2
)

print(f"\n4. Checking scheduler...")
print(f"   Scheduler type: {type(model.scheduler).__name__ if model.scheduler else 'None'}")
print(f"   Scheduler: {model.scheduler}")

if model.scheduler:
    print(f"   Initial LR: {model.optimizer.param_groups[0]['lr']:.6f}")

    # Manually step scheduler to see if it changes
    print("\n5. Testing scheduler stepping...")
    for i in range(5):
        model.scheduler.step()
        lr = model.optimizer.param_groups[0]['lr']
        print(f"   Step {i+1}: LR = {lr:.6f}")

    if hasattr(model.scheduler, 'T_max'):
        print(f"\n   Scheduler T_max: {model.scheduler.T_max}")
        print(f"   Expected T_max: {3 * len(train_loader)} (3 epochs × {len(train_loader)} batches)")
else:
    print("   ❌ NO SCHEDULER CREATED!")
    print("\n   Debugging compile process...")
    print(f"   scheduler_type stored: {model.scheduler_type}")

print("\n" + "=" * 80)
