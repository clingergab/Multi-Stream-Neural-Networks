"""
Script to run MultiChannel ResNet training with strong regularization
to address overfitting and improve validation accuracy.
"""

import os
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import argparse

# Import project modules
from src.utils.cifar100_loader import get_cifar100_datasets
from src.models.basic_multi_channel.multi_channel_resnet_network import multi_channel_resnet50
from src.utils.augmentation import DualStreamAugmentation


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train MultiChannel ResNet with regularization")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs to train for")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="Weight decay coefficient")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--augment", action="store_true", help="Enable advanced augmentation")
    parser.add_argument("--mixup", type=float, default=0.2, help="Mixup alpha parameter")
    parser.add_argument("--cutmix", type=float, default=1.0, help="CutMix alpha parameter")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing factor")
    parser.add_argument("--save_dir", type=str, default="saved_models", help="Directory to save models")
    parser.add_argument("--eval_interval", type=int, default=1, help="Epoch interval for evaluation")
    return parser.parse_args()


def convert_to_dual_stream(rgb_data):
    """Convert RGB data to RGB + Brightness dual stream format."""
    brightness = (0.299 * rgb_data[:, 0:1] + 
                  0.587 * rgb_data[:, 1:2] + 
                  0.114 * rgb_data[:, 2:3])
    return rgb_data, brightness


def create_data_loaders(args):
    """Create data loaders for training, validation, and testing."""
    print("Loading CIFAR-100 datasets...")
    train_dataset, test_dataset, class_names = get_cifar100_datasets("data/cifar-100")
    
    # Convert datasets to tensors
    train_data = []
    train_labels = []
    for i in range(len(train_dataset)):
        data, label = train_dataset[i]
        train_data.append(data.numpy())
        train_labels.append(label)

    test_data = []
    test_labels = []
    for i in range(len(test_dataset)):
        data, label = test_dataset[i]
        test_data.append(data.numpy())
        test_labels.append(label)

    # Convert to tensors
    train_data = torch.stack([torch.from_numpy(x) for x in train_data])
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_data = torch.stack([torch.from_numpy(x) for x in test_data])
    test_labels = torch.tensor(test_labels, dtype=torch.long)

    # Create validation split (20% of training data)
    val_size = int(0.2 * len(train_data))
    indices = torch.randperm(len(train_data))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    val_data = train_data[val_indices]
    val_labels = train_labels[val_indices]
    train_data = train_data[train_indices]
    train_labels = train_labels[train_indices]

    # Convert to dual-stream format
    rgb_train, brightness_train = convert_to_dual_stream(train_data)
    rgb_val, brightness_val = convert_to_dual_stream(val_data)
    rgb_test, brightness_test = convert_to_dual_stream(test_data)
    
    # Create datasets
    train_dataset = TensorDataset(rgb_train, brightness_train, train_labels)
    val_dataset = TensorDataset(rgb_val, brightness_val, val_labels)
    test_dataset = TensorDataset(rgb_test, brightness_test, test_labels)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size * 2, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size * 2, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, class_names


class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss function."""
    def __init__(self, classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes
        
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        
        if target.dim() == 1:
            # Create one-hot encoding
            with torch.no_grad():
                target = torch.zeros_like(pred).scatter_(1, target.unsqueeze(1), 1)
        
        # Apply label smoothing
        target = target * self.confidence + self.smoothing / self.classes
        
        return torch.mean(torch.sum(-target * pred, dim=-1))


class EMA:
    """Exponential Moving Average of model parameters."""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()
        
    def register(self):
        """Register model parameters for EMA."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update EMA parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Apply EMA parameters to the model for evaluation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original parameters to the model after evaluation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]


def train_model(args):
    """Train the MultiChannel ResNet model with regularization."""
    # Check device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Metal Performance Shaders (MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Create data loaders
    train_loader, val_loader, test_loader, class_names = create_data_loaders(args)
    
    # Create model
    print("Creating model...")
    model = multi_channel_resnet50(
        num_classes=100,  # CIFAR-100
        input_channels_rgb=3,
        input_channels_brightness=1,
        input_size=32,
        dropout_rate=args.dropout,
    )
    model = model.to(device)
    
    # Create optimizer and scheduler
    criterion = LabelSmoothingLoss(classes=100, smoothing=args.label_smoothing)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    # Create EMA
    ema = EMA(model, decay=0.999)
    
    # Create augmentation
    if args.augment:
        augmentation = DualStreamAugmentation(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutout_prob=0.5,
            cutout_size=16
        )
    else:
        augmentation = None
    
    # Training variables
    best_val_acc = 0.0
    patience_counter = 0
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": []
    }
    
    # Create directory for saving models
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Training loop
    print("Starting training...")
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for rgb_data, brightness_data, targets in pbar:
            # Move data to device
            rgb_data = rgb_data.to(device)
            brightness_data = brightness_data.to(device)
            targets = targets.to(device)
            
            # Apply augmentation if enabled
            if augmentation is not None:
                rgb_data, brightness_data, mixed_targets = augmentation.apply_to_batch(
                    rgb_data, brightness_data, targets)
                # If we got mixed targets, use them instead
                if torch.is_tensor(mixed_targets) and mixed_targets.dim() > 1:
                    targets = mixed_targets
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(rgb_data, brightness_data)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Update EMA
            ema.update()
            
            # Calculate metrics
            if targets.dim() > 1:
                # For mixed labels, get predicted class
                _, predicted = outputs.max(1)
                _, targets_idx = targets.max(1)
                train_correct += (predicted == targets_idx).sum().item()
            else:
                # Standard accuracy calculation
                _, predicted = outputs.max(1)
                train_correct += (predicted == targets).sum().item()
                
            train_total += targets.size(0)
            train_loss += loss.item() * targets.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                "loss": loss.item(),
                "acc": 100. * train_correct / train_total
            })
        
        # Calculate epoch metrics
        train_loss = train_loss / train_total
        train_acc = 100. * train_correct / train_total
        
        # Validation phase (only every eval_interval epochs)
        if (epoch + 1) % args.eval_interval == 0:
            model.eval()
            ema.apply_shadow()  # Apply EMA weights for evaluation
            
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
                for rgb_data, brightness_data, targets in pbar:
                    # Move data to device
                    rgb_data = rgb_data.to(device)
                    brightness_data = brightness_data.to(device)
                    targets = targets.to(device)
                    
                    # Forward pass
                    outputs = model(rgb_data, brightness_data)
                    loss = nn.CrossEntropyLoss()(outputs, targets)
                    
                    # Calculate metrics
                    _, predicted = outputs.max(1)
                    val_correct += (predicted == targets).sum().item()
                    val_total += targets.size(0)
                    val_loss += loss.item() * targets.size(0)
                    
                    # Update progress bar
                    pbar.set_postfix({
                        "loss": loss.item(),
                        "acc": 100. * val_correct / val_total
                    })
            
            # Calculate epoch metrics
            val_loss = val_loss / val_total
            val_acc = 100. * val_correct / val_total
            
            # Restore original weights
            ema.restore()
            
            # Print epoch summary
            print(f"Epoch {epoch+1}/{args.epochs} - "
                 f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                 f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
                 f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Check for improvement
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # Save best model
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'ema_state_dict': ema.shadow,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_acc': best_val_acc,
                    'train_acc': train_acc,
                }, f"{args.save_dir}/multichannel_resnet_best.pth")
                print(f"🔥 New best model saved with validation accuracy: {best_val_acc:.2f}%")
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"⏳ No improvement for {patience_counter} epochs. Best: {best_val_acc:.2f}%")
                
                # Check for early stopping
                if patience_counter >= args.patience:
                    print(f"🛑 Early stopping after {epoch+1} epochs")
                    break
        else:
            # For epochs without validation, just print training metrics
            print(f"Epoch {epoch+1}/{args.epochs} - "
                 f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                 f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Update learning rate
        scheduler.step()
        
        # Save history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        if (epoch + 1) % args.eval_interval == 0:
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
        else:
            # If no validation this epoch, duplicate the last value or add None
            if history["val_loss"]:
                history["val_loss"].append(history["val_loss"][-1])
                history["val_acc"].append(history["val_acc"][-1])
            else:
                history["val_loss"].append(None)
                history["val_acc"].append(None)
        history["lr"].append(optimizer.param_groups[0]['lr'])
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'ema_state_dict': ema.shadow,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'train_acc': train_acc,
                'history': history
            }, f"{args.save_dir}/multichannel_resnet_epoch{epoch+1}.pth")
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'ema_state_dict': ema.shadow,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_acc': best_val_acc,
        'train_acc': train_acc,
        'history': history
    }, f"{args.save_dir}/multichannel_resnet_final.pth")
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history["train_acc"], label="Train Accuracy")
    val_epochs = [i for i, acc in enumerate(history["val_acc"]) if acc is not None]
    val_accs = [acc for acc in history["val_acc"] if acc is not None]
    plt.plot(val_epochs, val_accs, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history["train_loss"], label="Train Loss")
    val_losses = [loss for loss in history["val_loss"] if loss is not None]
    plt.plot(val_epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{args.save_dir}/training_history.png")
    plt.close()
    
    # Test final model
    print("\nEvaluating on test set...")
    model.eval()
    ema.apply_shadow()  # Apply EMA weights for final evaluation
    
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for rgb_data, brightness_data, targets in tqdm(test_loader, desc="Testing"):
            # Move data to device
            rgb_data = rgb_data.to(device)
            brightness_data = brightness_data.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(rgb_data, brightness_data)
            loss = nn.CrossEntropyLoss()(outputs, targets)
            
            # Calculate metrics
            _, predicted = outputs.max(1)
            test_correct += (predicted == targets).sum().item()
            test_total += targets.size(0)
            test_loss += loss.item() * targets.size(0)
    
    # Calculate test metrics
    test_loss = test_loss / test_total
    test_acc = 100. * test_correct / test_total
    
    print(f"\nTest Results:")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_acc:.2f}%")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    
    # Return test results
    return {
        "test_loss": test_loss,
        "test_acc": test_acc,
        "best_val_acc": best_val_acc,
        "final_train_acc": train_acc,
        "history": history
    }


if __name__ == "__main__":
    args = parse_args()
    results = train_model(args)
    
    # Save results to JSON
    import json
    with open(f"{args.save_dir}/results.json", "w") as f:
        json.dump({
            "test_loss": results["test_loss"],
            "test_acc": results["test_acc"],
            "best_val_acc": results["best_val_acc"],
            "final_train_acc": results["final_train_acc"],
            "args": vars(args)
        }, f, indent=4)
