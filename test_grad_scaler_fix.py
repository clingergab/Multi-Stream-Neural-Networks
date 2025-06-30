"""
Test script to verify GradScaler usage is correct
"""

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

# Create a simple model
model = nn.Linear(10, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Create input data
x = torch.randn(8, 10)
y = torch.randn(8, 1)

# Create a gradient scaler
scaler = GradScaler()

print("Testing fixed GradScaler usage...")

# Test the fixed pattern
for epoch in range(3):
    optimizer.zero_grad()
    
    # Forward pass with autocast
    with autocast(device_type='cpu'):  # Use 'cuda' for GPU
        output = model(x)
        loss = criterion(output, y)
    
    # Backward pass with scaler
    scaler.scale(loss).backward()
    
    # Debug mode and gradient clipping (combined in a single unscale_ call)
    debug_mode = True
    gradient_clip = 1.0
    
    if debug_mode or gradient_clip > 0:
        scaler.unscale_(optimizer)
        
        if debug_mode:
            # Track gradient norm
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            print(f"  Epoch {epoch+1}, Gradient norm: {total_norm:.6f}")
        
        if gradient_clip > 0:
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
    
    # Update weights
    scaler.step(optimizer)
    scaler.update()
    
    print(f"  Epoch {epoch+1}, Loss: {loss.item():.6f}")

print("Test completed successfully!")
