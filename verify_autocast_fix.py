"""
Simple test script for autocast and GradScaler usage
"""

import torch
from torch.amp import autocast, GradScaler

# Create a small model
model = torch.nn.Linear(10, 1)
x = torch.randn(5, 10)
target = torch.randn(5, 1)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Get device type
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize GradScaler
scaler = GradScaler()

# Test the autocast and GradScaler
print("Running with autocast and GradScaler...")
for epoch in range(5):
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

print("Test completed successfully.")
