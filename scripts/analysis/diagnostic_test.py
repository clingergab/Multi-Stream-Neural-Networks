"""
Diagnostic test to understand why ResNet underperforms compared to dense model.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.builders.model_factory import create_model
from src.data_utils.rgb_to_rgbl import RGBtoRGBL
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

def diagnostic_test():
    """Run diagnostic test to understand ResNet vs Dense performance."""
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"🔍 Running diagnostic test on {device}")
    
    # Create a simple test case with just a few samples
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    mnist_train = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    
    # Very small subset for debugging
    train_subset = Subset(mnist_train, range(100))  # Just 100 samples
    train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
    
    # Create both models
    dense_model = create_model(
        'base_multi_channel',
        color_input_size=28*28*3,
        brightness_input_size=28*28,
        hidden_sizes=[64, 32],  # Smaller for fair comparison
        num_classes=10,
        activation='relu',
        dropout_rate=0.1
    ).to(device)
    
    resnet_model = create_model(
        'multi_channel_resnet18',
        num_classes=10,
        color_input_channels=3,
        brightness_input_channels=1,
        activation='relu'
    ).to(device)
    
    print(f"Dense model params: {sum(p.numel() for p in dense_model.parameters()):,}")
    print(f"ResNet model params: {sum(p.numel() for p in resnet_model.parameters()):,}")
    
    # Test forward pass
    rgb_to_rgbl = RGBtoRGBL()
    
    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)
        color_data, brightness_data = rgb_to_rgbl(data)
        
        print(f"\nBatch {batch_idx}:")
        print(f"Color data shape: {color_data.shape}")
        print(f"Brightness data shape: {brightness_data.shape}")
        
        # Test dense model
        color_flat = color_data.view(color_data.size(0), -1)
        brightness_flat = brightness_data.view(brightness_data.size(0), -1)
        
        with torch.no_grad():
            dense_output = dense_model.forward_combined(color_flat, brightness_flat)
            resnet_output = resnet_model.forward_combined(color_data, brightness_data)
            
            print(f"Dense output shape: {dense_output.shape}")
            print(f"Dense output range: [{dense_output.min():.3f}, {dense_output.max():.3f}]")
            print(f"Dense output std: {dense_output.std():.3f}")
            
            print(f"ResNet output shape: {resnet_output.shape}")
            print(f"ResNet output range: [{resnet_output.min():.3f}, {resnet_output.max():.3f}]")
            print(f"ResNet output std: {resnet_output.std():.3f}")
        
        if batch_idx == 2:  # Just test a few batches
            break
    
    # Test training dynamics
    print(f"\n🧪 Testing training dynamics...")
    
    dense_optimizer = torch.optim.Adam(dense_model.parameters(), lr=0.001)
    resnet_optimizer = torch.optim.Adam(resnet_model.parameters(), lr=0.0001)  # Lower LR for ResNet
    criterion = nn.CrossEntropyLoss()
    
    dense_losses = []
    resnet_losses = []
    
    for epoch in range(3):
        dense_model.train()
        resnet_model.train()
        
        epoch_dense_losses = []
        epoch_resnet_losses = []
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            color_data, brightness_data = rgb_to_rgbl(data)
            
            # Dense model training step
            color_flat = color_data.view(color_data.size(0), -1)
            brightness_flat = brightness_data.view(brightness_data.size(0), -1)
            
            dense_optimizer.zero_grad()
            dense_output = dense_model.forward_combined(color_flat, brightness_flat)
            dense_loss = criterion(dense_output, targets)
            dense_loss.backward()
            dense_optimizer.step()
            epoch_dense_losses.append(dense_loss.item())
            
            # ResNet model training step
            resnet_optimizer.zero_grad()
            resnet_output = resnet_model.forward_combined(color_data, brightness_data)
            resnet_loss = criterion(resnet_output, targets)
            resnet_loss.backward()
            resnet_optimizer.step()
            epoch_resnet_losses.append(resnet_loss.item())
            
            if batch_idx >= 5:  # Just a few batches per epoch
                break
        
        dense_avg_loss = np.mean(epoch_dense_losses)
        resnet_avg_loss = np.mean(epoch_resnet_losses)
        
        dense_losses.append(dense_avg_loss)
        resnet_losses.append(resnet_avg_loss)
        
        print(f"Epoch {epoch+1}: Dense Loss: {dense_avg_loss:.4f}, ResNet Loss: {resnet_avg_loss:.4f}")
    
    print(f"\n📊 Final Analysis:")
    print(f"Dense model learned faster: {dense_losses[-1] < resnet_losses[-1]}")
    print(f"Dense final loss: {dense_losses[-1]:.4f}")
    print(f"ResNet final loss: {resnet_losses[-1]:.4f}")
    

if __name__ == "__main__":
    diagnostic_test()
