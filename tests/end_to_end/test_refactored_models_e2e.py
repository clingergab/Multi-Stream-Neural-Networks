"""
End-to-End Test for Refactored Multi-Channel Models

This test validates the complete pipeline using our refactored BaseMultiChannelNetwork
and MultiChannelResNetNetwork with real data from MNIST and CIFAR-100.

Tests include:
- Data loading and preprocessing with RGBtoRGBL transform
- Model creation using factory methods
- Training loop with proper loss computation
- Evaluation and metrics calculation
-        print("\n📊 MNIST Results:")
        print("  BaseMultiChannelNetwork:")
        print(f"    Training Loss: {mnist_base_losses[-1]:.4f}, Test Accuracy: {mnist_base_metrics['accuracy']:.2f}%")
        print("  MultiChannelResNetNetwork (OPTIMIZED):")
        print(f"    Training Loss: {mnist_resnet_losses[-1]:.4f}, Test Accuracy: {mnist_resnet_metrics['accuracy']:.2f}%")
        
        print("\n📊 CIFAR-100 Results:")
        print("  BaseMultiChannelNetwork:")
        print(f"    Training Loss: {cifar_base_losses[-1]:.4f}, Test Accuracy: {cifar_base_metrics['accuracy']:.2f}%")
        print("  MultiChannelResNetNetwork (OPTIMIZED):")
        print(f"    Training Loss: {cifar_resnet_losses[-1]:.4f}, Test Accuracy: {cifar_resnet_metrics['accuracy']:.2f}%")
        
        # 8. Architecture Efficiency Analysis
        print("\n🔧 Architecture Efficiency:")tance analysis
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
from typing import Dict, Tuple, List

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.basic_multi_channel import (
    BaseMultiChannelNetwork, 
    MultiChannelResNetNetwork
)
from src.models.builders.model_factory import create_model
from src.transforms.rgb_to_rgbl import RGBtoRGBL


class TestConfig:
    """Test configuration."""
    # Data settings
    batch_size = 32
    num_workers = 0  # Set to 0 to avoid multiprocessing issues
    data_root = './data'
    
    # Training settings
    num_epochs = 2  # Short for testing
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Model settings - Use different input sizes for color vs brightness!
    color_input_size = 28 * 28 * 3    # RGB color data (2352)
    brightness_input_size = 28 * 28   # Brightness data (784) - more efficient!
    num_classes_mnist = 10
    num_classes_cifar = 100


def setup_data_loaders() -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """
    Set up data loaders for MNIST and CIFAR-100 with RGBtoRGBL preprocessing.
    
    Returns:
        Tuple of (mnist_train_loader, mnist_test_loader, cifar_train_loader, cifar_test_loader)
    """
    print("📊 Setting up data loaders...")
    
    # MNIST transforms - convert to RGB then to RGBL
    mnist_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Convert to RGB
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize RGB channels
    ])
    
    # CIFAR-100 transforms - already RGB
    cifar_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize RGB channels
    ])
    
    # Load MNIST
    mnist_train = torchvision.datasets.MNIST(
        root=TestConfig.data_root, train=True, download=True, transform=mnist_transform
    )
    mnist_test = torchvision.datasets.MNIST(
        root=TestConfig.data_root, train=False, download=True, transform=mnist_transform
    )
    
    # Load CIFAR-100
    cifar_train = torchvision.datasets.CIFAR100(
        root=TestConfig.data_root, train=True, download=True, transform=cifar_transform
    )
    cifar_test = torchvision.datasets.CIFAR100(
        root=TestConfig.data_root, train=False, download=True, transform=cifar_transform
    )
    
    # Create smaller subsets for faster testing
    mnist_train_subset = Subset(mnist_train, range(1000))  # 1000 samples
    mnist_test_subset = Subset(mnist_test, range(200))     # 200 samples
    cifar_train_subset = Subset(cifar_train, range(1000))  # 1000 samples
    cifar_test_subset = Subset(cifar_test, range(200))     # 200 samples
    
    # Create data loaders
    mnist_train_loader = DataLoader(
        mnist_train_subset, batch_size=TestConfig.batch_size, 
        shuffle=True, num_workers=TestConfig.num_workers
    )
    mnist_test_loader = DataLoader(
        mnist_test_subset, batch_size=TestConfig.batch_size, 
        shuffle=False, num_workers=TestConfig.num_workers
    )
    cifar_train_loader = DataLoader(
        cifar_train_subset, batch_size=TestConfig.batch_size, 
        shuffle=True, num_workers=TestConfig.num_workers
    )
    cifar_test_loader = DataLoader(
        cifar_test_subset, batch_size=TestConfig.batch_size, 
        shuffle=False, num_workers=TestConfig.num_workers
    )
    
    print(f"✅ Data loaders ready - MNIST: {len(mnist_train_subset)}/{len(mnist_test_subset)}, "
          f"CIFAR-100: {len(cifar_train_subset)}/{len(cifar_test_subset)}")
    
    return mnist_train_loader, mnist_test_loader, cifar_train_loader, cifar_test_loader


def create_models() -> Tuple[BaseMultiChannelNetwork, MultiChannelResNetNetwork, BaseMultiChannelNetwork, MultiChannelResNetNetwork]:
    """
    Create both test models for both datasets using factory methods.
    
    Returns:
        Tuple of (mnist_base_model, mnist_resnet_model, cifar_base_model, cifar_resnet_model)
    """
    print("🏗️  Creating models...")
    
    # Create BaseMultiChannelNetwork for MNIST (dense/tabular-style)
    # Use different input sizes for color vs brightness - more efficient!
    mnist_base_model = create_model(
        'base_multi_channel',
        color_input_size=TestConfig.color_input_size,      # 2352 (28*28*3)
        brightness_input_size=TestConfig.brightness_input_size,  # 784 (28*28)
        hidden_sizes=[128, 64, 32],
        num_classes=TestConfig.num_classes_mnist,
        activation='relu',
        dropout_rate=0.2
    )
    
    # Create OPTIMIZED MultiChannelResNetNetwork for MNIST
    # Use different input channels: 3 for color, 1 for brightness - more efficient!
    mnist_resnet_model = create_model(
        'multi_channel_resnet18',
        num_classes=TestConfig.num_classes_mnist,
        color_input_channels=3,       # RGB color input
        brightness_input_channels=1,  # Single brightness channel - efficient!
        activation='relu'
    )
    
    # Create BaseMultiChannelNetwork for CIFAR-100 (scaled up)
    cifar_base_model = create_model(
        'base_multi_channel',
        color_input_size=32 * 32 * 3,  # CIFAR-100 is 32x32 RGB
        brightness_input_size=32 * 32,  # CIFAR-100 brightness
        hidden_sizes=[256, 128, 64],    # Larger hidden sizes for CIFAR
        num_classes=TestConfig.num_classes_cifar,
        activation='relu',
        dropout_rate=0.3
    )
    
    # Create OPTIMIZED MultiChannelResNetNetwork for CIFAR-100
    cifar_resnet_model = create_model(
        'multi_channel_resnet18',
        num_classes=TestConfig.num_classes_cifar,
        color_input_channels=3,       # RGB color input
        brightness_input_channels=1,  # Single brightness channel - efficient!
        activation='relu'
    )
    
    # Move models to device
    mnist_base_model = mnist_base_model.to(TestConfig.device)
    mnist_resnet_model = mnist_resnet_model.to(TestConfig.device)
    cifar_base_model = cifar_base_model.to(TestConfig.device)
    cifar_resnet_model = cifar_resnet_model.to(TestConfig.device)
    
    print(f"✅ Models created and moved to {TestConfig.device}")
    print(f"   MNIST BaseMultiChannelNetwork params: {sum(p.numel() for p in mnist_base_model.parameters()):,}")
    print(f"   MNIST MultiChannelResNetNetwork params: {sum(p.numel() for p in mnist_resnet_model.parameters()):,}")
    print(f"   CIFAR BaseMultiChannelNetwork params: {sum(p.numel() for p in cifar_base_model.parameters()):,}")
    print(f"   CIFAR MultiChannelResNetNetwork params: {sum(p.numel() for p in cifar_resnet_model.parameters()):,}")
    
    return mnist_base_model, mnist_resnet_model, cifar_base_model, cifar_resnet_model


def train_model(model: nn.Module, train_loader: DataLoader, model_name: str, dataset_name: str, is_dense: bool = False) -> List[float]:
    """
    Train a model for a few epochs.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        model_name: Name for logging
        is_dense: Whether this is a dense model (needs flattening)
        
    Returns:
        List of training losses
    """
    print(f"🚀 Training {model_name} on {dataset_name}...")
    
    # OPTIMIZED: Use different learning rates for different architectures
    if is_dense:
        learning_rate = TestConfig.learning_rate  # 0.001 for dense
    else:
        learning_rate = TestConfig.learning_rate * 0.1  # 0.0001 for ResNet (lower!)
    
    print(f"   Using learning rate: {learning_rate}")
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4 if not is_dense else 0)
    criterion = nn.CrossEntropyLoss()
    rgb_to_rgbl = RGBtoRGBL()  # Use our existing transform
    
    model.train()
    losses = []
    
    for epoch in range(TestConfig.num_epochs):
        epoch_losses = []
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(TestConfig.device), targets.to(TestConfig.device)
            
            # Use our existing RGBtoRGBL transform to get separate color and brightness
            # Now handles batches efficiently: [B, 3, H, W] -> ([B, 3, H, W], [B, 1, H, W])
            color_data, brightness_data = rgb_to_rgbl(data)
            
            # OPTIMIZED: ResNet now supports different input channels!
            # No need to expand brightness - save computation and memory
            if not is_dense:
                # Keep brightness as 1 channel: color=[B,3,H,W], brightness=[B,1,H,W]
                # Our optimized ResNet handles this efficiently!
                pass
            
            # For dense model, flatten the inputs but keep different sizes
            if is_dense:
                color_data = color_data.view(color_data.size(0), -1)      # [B, 2352] (28*28*3)
                brightness_data = brightness_data.view(brightness_data.size(0), -1)  # [B, 784] (28*28*1)
                # Keep different sizes - our architecture supports this!
            
            optimizer.zero_grad()
            
            # Forward pass
            if hasattr(model, 'forward_combined'):
                # ResNet model returns tuple, use combined forward
                outputs = model.forward_combined(color_data, brightness_data)
            else:
                # Dense model returns single output
                outputs = model(color_data, brightness_data)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 10 == 0:
                print(f"   Epoch {epoch+1}/{TestConfig.num_epochs}, "
                      f"Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, "
                      f"Acc: {100.*correct/total:.2f}%")
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        print(f"   Epoch {epoch+1} completed - Avg Loss: {avg_loss:.4f}, Accuracy: {100.*correct/total:.2f}%")
    
    return losses


def evaluate_model(model: nn.Module, test_loader: DataLoader, model_name: str, dataset_name: str, is_dense: bool = False) -> Dict[str, float]:
    """
    Evaluate a model on test data.
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        model_name: Name for logging
        is_dense: Whether this is a dense model (needs flattening)
        
    Returns:
        Dictionary of evaluation metrics
    """
    print(f"📊 Evaluating {model_name} on {dataset_name}...")
    
    model.eval()
    criterion = nn.CrossEntropyLoss()
    rgb_to_rgbl = RGBtoRGBL()  # Use our existing transform
    
    total_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(TestConfig.device), targets.to(TestConfig.device)
            
            # Use our existing RGBtoRGBL transform for batch processing
            color_data, brightness_data = rgb_to_rgbl(data)
            
            # OPTIMIZED: ResNet now supports different input channels!
            if not is_dense:
                # Keep brightness as 1 channel - more efficient!
                pass
            
            # For dense model, flatten inputs but keep different sizes
            if is_dense:
                color_data = color_data.view(color_data.size(0), -1)      # [B, 2352] (28*28*3)
                brightness_data = brightness_data.view(brightness_data.size(0), -1)  # [B, 784] (28*28*1)
                # Keep different sizes - our architecture supports this!
            
            # Forward pass
            if hasattr(model, 'forward_combined'):
                outputs = model.forward_combined(color_data, brightness_data)
            else:
                outputs = model(color_data, brightness_data)
            
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(test_loader)
    
    metrics = {
        'accuracy': accuracy,
        'loss': avg_loss,
        'correct': correct,
        'total': total
    }
    
    print(f"   Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    return metrics


def test_pathway_importance(model: nn.Module, model_name: str):
    """Test pathway importance analysis."""
    print(f"🔍 Testing pathway importance for {model_name}...")
    
    if hasattr(model, 'get_pathway_importance'):
        importance = model.get_pathway_importance()
        print(f"   Pathway importance: {importance}")
    else:
        print(f"   {model_name} doesn't have pathway importance analysis")


def main():
    """Run the complete end-to-end test for all model/dataset combinations."""
    print("🧪 Starting Comprehensive End-to-End Test for Refactored Multi-Channel Models")
    print(f"   Device: {TestConfig.device}")
    print("   Testing: BaseMultiChannelNetwork + MultiChannelResNetNetwork on MNIST + CIFAR-100")
    print("=" * 80)
    
    try:
        # 1. Setup data
        mnist_train_loader, mnist_test_loader, cifar_train_loader, cifar_test_loader = setup_data_loaders()
        
        # 2. Create all models
        mnist_base_model, mnist_resnet_model, cifar_base_model, cifar_resnet_model = create_models()
        
        # 3. Test BaseMultiChannelNetwork on MNIST
        print("\n" + "="*60)
        print("TESTING BaseMultiChannelNetwork with MNIST")
        print("="*60)
        
        mnist_base_losses = train_model(mnist_base_model, mnist_train_loader, "BaseMultiChannelNetwork", "MNIST", is_dense=True)
        mnist_base_metrics = evaluate_model(mnist_base_model, mnist_test_loader, "BaseMultiChannelNetwork", "MNIST", is_dense=True)
        test_pathway_importance(mnist_base_model, "BaseMultiChannelNetwork")
        
        # 4. Test OPTIMIZED MultiChannelResNetNetwork on MNIST
        print("\n" + "="*60)
        print("TESTING OPTIMIZED MultiChannelResNetNetwork with MNIST")
        print("="*60)
        
        mnist_resnet_losses = train_model(mnist_resnet_model, mnist_train_loader, "MultiChannelResNetNetwork", "MNIST", is_dense=False)
        mnist_resnet_metrics = evaluate_model(mnist_resnet_model, mnist_test_loader, "MultiChannelResNetNetwork", "MNIST", is_dense=False)
        test_pathway_importance(mnist_resnet_model, "MultiChannelResNetNetwork")
        
        # 5. Test BaseMultiChannelNetwork on CIFAR-100
        print("\n" + "="*60)
        print("TESTING BaseMultiChannelNetwork with CIFAR-100")
        print("="*60)
        
        cifar_base_losses = train_model(cifar_base_model, cifar_train_loader, "BaseMultiChannelNetwork", "CIFAR-100", is_dense=True)
        cifar_base_metrics = evaluate_model(cifar_base_model, cifar_test_loader, "BaseMultiChannelNetwork", "CIFAR-100", is_dense=True)
        test_pathway_importance(cifar_base_model, "BaseMultiChannelNetwork")
        
        # 6. Test OPTIMIZED MultiChannelResNetNetwork on CIFAR-100
        print("\n" + "="*60)
        print("TESTING OPTIMIZED MultiChannelResNetNetwork with CIFAR-100")
        print("="*60)
        
        cifar_resnet_losses = train_model(cifar_resnet_model, cifar_train_loader, "MultiChannelResNetNetwork", "CIFAR-100", is_dense=False)
        cifar_resnet_metrics = evaluate_model(cifar_resnet_model, cifar_test_loader, "MultiChannelResNetNetwork", "CIFAR-100", is_dense=False)
        test_pathway_importance(cifar_resnet_model, "MultiChannelResNetNetwork")
        
        # 7. Comprehensive Results Summary
        print("\n" + "="*80)
        print("COMPREHENSIVE RESULTS SUMMARY")
        print("="*80)
        
        print("\n📊 MNIST Results:")
        print(f"  BaseMultiChannelNetwork:")
        print(f"    Training Loss: {mnist_base_losses[-1]:.4f}, Test Accuracy: {mnist_base_metrics['accuracy']:.2f}%")
        print(f"  MultiChannelResNetNetwork (OPTIMIZED):")
        print(f"    Training Loss: {mnist_resnet_losses[-1]:.4f}, Test Accuracy: {mnist_resnet_metrics['accuracy']:.2f}%")
        
        print(f"\n📊 CIFAR-100 Results:")
        print(f"  BaseMultiChannelNetwork:")
        print(f"    Training Loss: {cifar_base_losses[-1]:.4f}, Test Accuracy: {cifar_base_metrics['accuracy']:.2f}%")
        print(f"  MultiChannelResNetNetwork (OPTIMIZED):")
        print(f"    Training Loss: {cifar_resnet_losses[-1]:.4f}, Test Accuracy: {cifar_resnet_metrics['accuracy']:.2f}%")
        
        # 8. Architecture Efficiency Analysis
        print(f"\n🔧 Architecture Efficiency:")
        mnist_base_params = sum(p.numel() for p in mnist_base_model.parameters())
        mnist_resnet_params = sum(p.numel() for p in mnist_resnet_model.parameters())
        cifar_base_params = sum(p.numel() for p in cifar_base_model.parameters())
        cifar_resnet_params = sum(p.numel() for p in cifar_resnet_model.parameters())
        
        print(f"  MNIST Dense Model: {mnist_base_params:,} params")
        print(f"  MNIST ResNet (Optimized): {mnist_resnet_params:,} params")
        print(f"  CIFAR Dense Model: {cifar_base_params:,} params") 
        print(f"  CIFAR ResNet (Optimized): {cifar_resnet_params:,} params")
        
        # Check if results are reasonable
        success = True
        failed_tests = []
        
        if mnist_base_metrics['accuracy'] < 50:
            failed_tests.append("MNIST BaseMultiChannelNetwork accuracy too low")
        if mnist_resnet_metrics['accuracy'] < 50:
            failed_tests.append("MNIST MultiChannelResNetNetwork accuracy too low")
        if cifar_base_metrics['accuracy'] < 5:
            failed_tests.append("CIFAR-100 BaseMultiChannelNetwork accuracy too low")
        if cifar_resnet_metrics['accuracy'] < 5:
            failed_tests.append("CIFAR-100 MultiChannelResNetNetwork accuracy too low")
        
        if failed_tests:
            success = False
            print("\n❌ Some tests failed:")
            for failure in failed_tests:
                print(f"   - {failure}")
        else:
            print("\n🎉 All tests passed! Comprehensive multi-channel architecture working correctly.")
            print("✅ Optimized ResNet with different input channels works!")
            print("✅ Both dense and ResNet models work on both datasets")
            print("✅ Data loading with RGBtoRGBL preprocessing works")
            print("✅ Model creation via factory works")
            print("✅ Training and evaluation work for all combinations")
            print("✅ Multi-channel processing is efficient and effective")
        
        return success
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
