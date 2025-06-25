"""
Minimal example of Multi-Stream Neural Network usage

This example demonstrates how to create and use a basic multi-channel model.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# Import working basic multi-channel components
from src.models.basic_multi_channel.multi_channel_model import MultiChannelNetwork, multi_channel_18
from src.models.layers.basic_layers import BasicMultiChannelLayer
from src.transforms.rgb_to_rgbl import RGBtoRGBL


class SimpleMSNN(nn.Module):
    """
    Simplified Multi-Stream Neural Network for demonstration.
    
    This is a minimal implementation to show the basic concept.
    The full implementation will be in the src/models/ directory.
    """
    
    def __init__(self, input_size=(4, 32, 32), hidden_size=512, num_classes=10):
        super().__init__()
        
        # Flatten input for simplicity
        flattened_size = input_size[0] * input_size[1] * input_size[2]
        color_size = 3 * input_size[1] * input_size[2]  # RGB channels
        brightness_size = 1 * input_size[1] * input_size[2]  # Luminance channel
        
        # Separate pathways
        self.color_pathway = nn.Sequential(
            nn.Linear(color_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        self.brightness_pathway = nn.Sequential(
            nn.Linear(brightness_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Simple direct mixing parameters
        self.alpha = nn.Parameter(torch.tensor(1.0))  # Color weight
        self.beta = nn.Parameter(torch.tensor(1.0))   # Brightness weight
        
        # Classifier
        self.classifier = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        """Forward pass through the network."""
        batch_size = x.size(0)
        
        # Extract color and brightness channels
        color_channels = x[:, :3, :, :].view(batch_size, -1)      # RGB
        brightness_channel = x[:, 3:4, :, :].view(batch_size, -1) # Luminance
        
        # Process through separate pathways
        color_features = self.color_pathway(color_channels)
        brightness_features = self.brightness_pathway(brightness_channel)
        
        # Direct mixing with learnable weights
        alpha_reg = torch.clamp(self.alpha, min=0.01)  # Prevent collapse
        beta_reg = torch.clamp(self.beta, min=0.01)
        
        integrated_features = alpha_reg * color_features + beta_reg * brightness_features
        
        # Classification
        output = self.classifier(integrated_features)
        
        return output
    
    def get_pathway_weights(self):
        """Get current pathway mixing weights."""
        return {
            'alpha': self.alpha.item(),
            'beta': self.beta.item()
        }


def rgb_to_rgbl_transform(rgb_tensor):
    """
    Convert RGB tensor to RGB+L tensor by adding luminance channel.
    
    Args:
        rgb_tensor: Tensor of shape (B, 3, H, W) with RGB channels
        
    Returns:
        Tensor of shape (B, 4, H, W) with RGB+L channels
    """
    # ITU-R BT.709 luminance calculation
    luminance = (0.2126 * rgb_tensor[:, 0:1, :, :] + 
                 0.7152 * rgb_tensor[:, 1:2, :, :] + 
                 0.0722 * rgb_tensor[:, 2:3, :, :])
    
    # Concatenate RGB + L
    rgbl_tensor = torch.cat([rgb_tensor, luminance], dim=1)
    
    return rgbl_tensor


def create_dummy_data(batch_size=32, image_size=32, num_samples=1000):
    """Create dummy RGB data for demonstration."""
    # Random RGB images
    rgb_data = torch.randn(num_samples, 3, image_size, image_size)
    rgb_data = torch.clamp(rgb_data, 0, 1)  # Ensure valid RGB range
    
    # Convert to RGB+L
    rgbl_data = rgb_to_rgbl_transform(rgb_data)
    
    # Random labels
    labels = torch.randint(0, 10, (num_samples,))
    
    return rgbl_data, labels


def main():
    """Demonstrate the Multi-Stream Neural Network."""
    print("Multi-Stream Neural Network - Minimal Example")
    print("=" * 50)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = SimpleMSNN(input_size=(4, 32, 32), hidden_size=256, num_classes=10)
    model = model.to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create dummy data
    print("\nCreating dummy data...")
    rgbl_data, labels = create_dummy_data(num_samples=1000)
    
    # Create data loader
    dataset = TensorDataset(rgbl_data, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop (simplified)
    print("\nStarting training...")
    model.train()
    
    for epoch in range(5):  # Just a few epochs for demo
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            epoch_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
        
        # Print epoch results
        accuracy = 100.0 * correct / total
        avg_loss = epoch_loss / len(dataloader)
        
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%")
        
        # Show pathway weights
        weights = model.get_pathway_weights()
        print(f"  Pathway weights - α (color): {weights['alpha']:.3f}, β (brightness): {weights['beta']:.3f}")
    
    print("\nTraining completed!")
    
    # Demonstrate pathway importance
    weights = model.get_pathway_weights()
    total_weight = abs(weights['alpha']) + abs(weights['beta'])
    color_importance = abs(weights['alpha']) / total_weight
    brightness_importance = abs(weights['beta']) / total_weight
    
    print(f"\nFinal pathway importance:")
    print(f"  Color pathway: {color_importance:.1%}")
    print(f"  Brightness pathway: {brightness_importance:.1%}")
    
    print(f"\nThis demonstrates the basic Multi-Stream concept!")
    print(f"The full implementation will include:")
    print(f"  - Multiple integration strategies (channel-wise, dynamic, spatial)")
    print(f"  - Comprehensive evaluation metrics")
    print(f"  - Real dataset integration (CIFAR-10, CIFAR-100, ImageNet)")
    print(f"  - Advanced training features (callbacks, logging, checkpointing)")


if __name__ == "__main__":
    main()
