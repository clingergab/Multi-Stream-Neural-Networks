"""
Verification script for autocast and GradScaler usage from torch.amp
"""

import torch
from torch.amp import autocast, GradScaler

print(f"PyTorch version: {torch.__version__}")

# Create a simple model for testing
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)
        
    def forward(self, x):
        return self.linear(x)

# Initialize model and data
model = SimpleModel()
x = torch.randn(5, 10)
target = torch.randn(5, 1)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Get device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize GradScaler
scaler = GradScaler()

# Test with autocast
print("\nTesting torch.amp.autocast with device_type parameter...")
for epoch in range(3):
    # Zero gradients
    optimizer.zero_grad()
    
    # Forward pass with autocast
    with autocast(device_type=device.type):
        output = model(x)
        loss = criterion(output, target)
        print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")
    
    # Backward pass with GradScaler
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

print("\nVerification completed successfully!")
