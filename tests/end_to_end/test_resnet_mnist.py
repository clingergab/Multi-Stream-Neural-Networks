"""
Quick test to see how MultiChannelResNetNetwork performs on MNIST.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.builders.model_factory import create_model
from src.transforms.rgb_to_rgbl import RGBtoRGBL


def test_resnet_on_mnist():
    """Test ResNet model on MNIST."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Setup MNIST data
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Convert to RGB
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Small subsets
    train_subset = Subset(train_dataset, range(500))
    test_subset = Subset(test_dataset, range(100))
    
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)
    
    # Create ResNet model
    model = create_model(
        'multi_channel_resnet18',
        num_classes=10,
        input_channels=3,
        activation='relu'
    ).to(device)
    
    print(f"ResNet model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train for 2 epochs
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    rgb_to_rgbl = RGBtoRGBL()
    
    model.train()
    for epoch in range(2):
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            # Transform data
            color_data, brightness_data = rgb_to_rgbl(data)
            brightness_data = brightness_data.expand(-1, 3, -1, -1)  # Expand to 3 channels
            
            optimizer.zero_grad()
            outputs = model.forward_combined(color_data, brightness_data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 5 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%")
    
    # Test
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            
            color_data, brightness_data = rgb_to_rgbl(data)
            brightness_data = brightness_data.expand(-1, 3, -1, -1)
            
            outputs = model.forward_combined(color_data, brightness_data)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    print(f"\nFinal Test Accuracy: {100.*correct/total:.2f}% ({correct}/{total})")


if __name__ == "__main__":
    test_resnet_on_mnist()
