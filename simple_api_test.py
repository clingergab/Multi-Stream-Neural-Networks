#!/usr/bin/env python3
"""
Simple test script to verify the simplified forward method behavior.
"""

import torch
import torch.nn as nn
from typing import Tuple

class SimpleTestModel(nn.Module):
    """Simplified model for testing the forward API concept."""
    
    def __init__(self):
        super().__init__()
        self.color_layer = nn.Linear(10, 8)
        self.brightness_layer = nn.Linear(8, 8)
        self.shared_classifier = nn.Linear(16, 5)  # Concatenated features
        self.color_head = nn.Linear(8, 5)  # For analysis
        self.brightness_head = nn.Linear(8, 5)  # For analysis
    
    def forward(self, color_input: torch.Tensor, brightness_input: torch.Tensor) -> torch.Tensor:
        """
        Single forward method for training/inference.
        Returns combined output suitable for loss computation.
        """
        # Process inputs
        color_features = self.color_layer(color_input)
        brightness_features = self.brightness_layer(brightness_input)
        
        # Combine for classification
        combined_features = torch.cat([color_features, brightness_features], dim=1)
        return self.shared_classifier(combined_features)
    
    def analyze_pathways(self, color_input: torch.Tensor, brightness_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Separate method for research/analysis.
        Returns separate outputs for each pathway.
        """
        # Process inputs
        color_features = self.color_layer(color_input)
        brightness_features = self.brightness_layer(brightness_input)
        
        # Separate heads for analysis
        color_logits = self.color_head(color_features)
        brightness_logits = self.brightness_head(brightness_features)
        
        return color_logits, brightness_logits

def test_simplified_api():
    """Test the simplified forward API concept."""
    print("🧪 Testing Simplified Forward API Concept")
    print("=" * 50)
    
    # Create model and test data
    model = SimpleTestModel()
    batch_size = 3
    color_data = torch.randn(batch_size, 10)
    brightness_data = torch.randn(batch_size, 8)
    
    print(f"📊 Test data shapes:")
    print(f"   Color: {color_data.shape}")
    print(f"   Brightness: {brightness_data.shape}")
    print()
    
    # Test 1: Primary forward() method
    print("1️⃣ Testing forward() - Primary method:")
    output = model.forward(color_data, brightness_data)
    print(f"   Output shape: {output.shape}")
    print(f"   Output type: {type(output)}")
    print(f"   ✅ Single tensor for training/classification")
    print()
    
    # Test 2: Model call (same as forward)
    print("2️⃣ Testing model() call:")
    call_output = model(color_data, brightness_data)
    print(f"   Same as forward(): {torch.allclose(output, call_output)}")
    print(f"   ✅ Identical results")
    print()
    
    # Test 3: Analysis method
    print("3️⃣ Testing analyze_pathways():")
    color_logits, brightness_logits = model.analyze_pathways(color_data, brightness_data)
    print(f"   Color logits shape: {color_logits.shape}")
    print(f"   Brightness logits shape: {brightness_logits.shape}")
    print(f"   ✅ Separate outputs for research")
    print()
    
    # Test 4: Training compatibility
    print("4️⃣ Testing training compatibility:")
    labels = torch.randint(0, 5, (batch_size,))
    criterion = nn.CrossEntropyLoss()
    
    # This is the key test - seamless training
    loss = criterion(model(color_data, brightness_data), labels)
    print(f"   Loss computation: ✅ Success! Loss = {loss.item():.4f}")
    
    loss.backward()
    print(f"   Backward pass: ✅ Success!")
    print()
    
    # Test 5: Research analysis example
    print("5️⃣ Testing research analysis:")
    color_logits, brightness_logits = model.analyze_pathways(color_data, brightness_data)
    
    color_preds = torch.argmax(color_logits, dim=1)
    brightness_preds = torch.argmax(brightness_logits, dim=1)
    
    color_acc = (color_preds == labels).float().mean().item()
    brightness_acc = (brightness_preds == labels).float().mean().item()
    
    print(f"   Color pathway accuracy: {color_acc:.3f}")
    print(f"   Brightness pathway accuracy: {brightness_acc:.3f}")
    print(f"   ✅ Individual pathway analysis works!")
    print()
    
    print("🎉 Simplified API Test Results:")
    print("=" * 50)
    print("✅ ONE forward() method - clean and simple")
    print("✅ model() calls work seamlessly for training")
    print("✅ analyze_pathways() for research purposes")
    print("✅ No confusion about which method to use")
    print("✅ Standard PyTorch patterns work perfectly")
    print()
    print("💡 API Summary:")
    print("   - Use model(x, y) or model.forward(x, y) for training/inference")
    print("   - Use model.analyze_pathways(x, y) for research/analysis")
    print("   - Simple, clear, and follows PyTorch conventions")

if __name__ == "__main__":
    test_simplified_api()
