"""
Quick local training test for SUN RGB-D dataset.
Tests the full training pipeline with a few epochs.
"""

import torch
from src.models.multi_channel.mc_resnet import mc_resnet18
from src.data_utils.sunrgbd_dataset import get_sunrgbd_dataloaders

def test_local_training():
    print("=" * 60)
    print("SUN RGB-D LOCAL TRAINING TEST")
    print("=" * 60)
    
    # Check CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load dataset
    print("\n1. Loading SUN RGB-D dataset...")
    train_loader, val_loader = get_sunrgbd_dataloaders(
        data_root='data/sunrgbd_15',
        batch_size=16,  # Small batch for local testing
        num_workers=2,
        target_size=(224, 224)  # Small size for speed
    )
    
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Train samples: {len(train_loader.dataset)}")
    print(f"   Val samples: {len(val_loader.dataset)}")
    
    # Test batch loading
    print("\n2. Testing batch loading...")
    rgb, depth, labels = next(iter(train_loader))
    print(f"   RGB shape: {rgb.shape}")
    print(f"   Depth shape: {depth.shape}")
    print(f"   Labels shape: {labels.shape}")
    print(f"   Label range: [{labels.min().item()}, {labels.max().item()}]")
    assert labels.min() >= 0 and labels.max() <= 14, "Labels out of range!"
    print("   ✅ Labels in valid range [0, 14]")
    
    # Create model
    print("\n3. Creating MCResNet model...")
    model = mc_resnet18(
        num_classes=15,
        stream1_input_channels=3,
        stream2_input_channels=1,
        fusion_type='weighted',
        dropout_p=0.3,
        device=device,
        use_amp=False  # Disable AMP for local testing
    )
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Compile
    print("\n4. Compiling model...")
    model.compile(
        optimizer='adamw',
        learning_rate=1e-4,
        weight_decay=2e-2,
        loss='cross_entropy',
        scheduler='cosine'
    )
    print("   ✅ Model compiled")
    
    # Test forward pass
    print("\n5. Testing forward pass...")
    model.eval()
    with torch.no_grad():
        rgb_test = rgb.to(device)
        depth_test = depth.to(device)
        outputs = model(rgb_test, depth_test)
        print(f"   Output shape: {outputs.shape}")
        assert outputs.shape == (rgb.size(0), 15), "Output shape mismatch!"
        print("   ✅ Forward pass successful")
    
    # Test backward pass
    print("\n6. Testing backward pass...")
    model.train()
    rgb_train = rgb[:8].to(device)
    depth_train = depth[:8].to(device)
    labels_train = labels[:8].to(device)
    
    model.optimizer.zero_grad()
    outputs = model(rgb_train, depth_train)
    loss = model.criterion(outputs, labels_train)
    loss.backward()
    model.optimizer.step()
    
    print(f"   Loss: {loss.item():.4f}")
    print("   ✅ Backward pass successful")
    
    # Quick training test (2 epochs)
    print("\n7. Running 2-epoch training test...")
    for epoch in range(2):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Train on subset of batches
        for batch_idx, (rgb, depth, labels) in enumerate(train_loader):
            if batch_idx >= 10:  # Only 10 batches for speed
                break
            
            rgb, depth, labels = rgb.to(device), depth.to(device), labels.to(device)
            
            model.optimizer.zero_grad()
            outputs = model(rgb, depth)
            loss = model.criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            model.optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_loss /= min(10, len(train_loader))
        train_acc = train_correct / train_total
        
        # Validation on subset
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_idx, (rgb, depth, labels) in enumerate(val_loader):
                if batch_idx >= 5:  # Only 5 batches
                    break
                
                rgb, depth, labels = rgb.to(device), depth.to(device), labels.to(device)
                outputs = model(rgb, depth)
                loss = model.criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss /= min(5, len(val_loader))
        val_acc = val_correct / val_total
        
        print(f"   Epoch {epoch+1}/2:")
        print(f"     Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"     Val Loss: {val_loss:.4f}   | Val Acc: {val_acc*100:.2f}%")
    
    print("\n" + "=" * 60)
    print("✅ ALL LOCAL TESTS PASSED!")
    print("=" * 60)
    print("\nSUN RGB-D dataset is ready for Colab training!")
    print("The dataloader, model, and training loop all work correctly.")

if __name__ == "__main__":
    test_local_training()
