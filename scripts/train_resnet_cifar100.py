#!/usr/bin/env python
"""
Train a ResNet50 model on CIFAR-100 dataset
"""

import os
import sys
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# Add the src directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))
from src.models2.core.resnet import resnet50

# Training settings
parser = argparse.ArgumentParser(description='Train ResNet50 on CIFAR-100')
parser.add_argument('--batch-size', type=int, default=128, help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=100, help='input batch size for testing (default: 100)')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay (default: 5e-4)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, help='how many batches to wait before logging training status')
parser.add_argument('--save-model', action='store_true', default=True, help='save the current model')
parser.add_argument('--save-dir', type=str, default='saved_models', help='directory to save model')
parser.add_argument('--resume', type=str, default='', help='resume from checkpoint')

args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Set the random seed for reproducibility
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

# Check if CIFAR-100 dataset exists, if not download it
data_dir = Path(__file__).parent.parent / 'data'
os.makedirs(data_dir, exist_ok=True)

# On macOS, there can be SSL certificate issues when downloading datasets
# Setting a SSL context to handle this
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

trainset = torchvision.datasets.CIFAR100(
    root=str(data_dir), train=True, download=True, transform=transform_train)
trainloader = DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(
    root=str(data_dir), train=False, download=True, transform=transform_test)
testloader = DataLoader(
    testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)

classes = list(range(100))

# Model
print('==> Building model..')
# For CIFAR-100, we need to set num_classes=100
model = resnet50(num_classes=100)

# For CIFAR, which is 32x32 images, we need to adjust the first conv layer
# Original ResNet has 7x7 conv with stride=2 for ImageNet, which is too aggressive for CIFAR
# Replace it with a 3x3 conv with stride=1
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
# Remove the max pooling layer which is also too aggressive for CIFAR
model.maxpool = nn.Identity()

model = model.to(device)

if args.resume:
    # Load checkpoint
    print('==> Resuming from checkpoint..')
    assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['model'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, 
                      momentum=args.momentum, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

# Training
def train(epoch):
    print(f'\nEpoch: {epoch}')
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % args.log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(inputs)}/{len(trainloader.dataset)} '
                  f'({100. * batch_idx / len(trainloader):.0f}%)]\tLoss: {loss.item():.6f} '
                  f'Acc: {100. * correct / total:.2f}% ({correct}/{total})')

    return train_loss / len(trainloader), 100. * correct / total

def test(epoch):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    acc = 100. * correct / total
    print(f'\nTest set: Average loss: {test_loss / len(testloader):.4f}, '
          f'Accuracy: {correct}/{total} ({acc:.2f}%)\n')

    # Save checkpoint
    if acc > best_acc and args.save_model:
        print('Saving..')
        state = {
            'model': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(args.save_dir):
            os.makedirs(args.save_dir)
        torch.save(state, os.path.join(args.save_dir, 'resnet50_cifar100.pth'))
        best_acc = acc
    
    return test_loss / len(testloader), acc

# Main training loop
if __name__ == '__main__':
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        train_loss, train_acc = train(epoch)
        test_loss, test_acc = test(epoch)
        scheduler.step()
        
        # Print epoch summary
        epoch_time = time.time() - start_time
        print(f'Epoch {epoch} completed in {epoch_time:.2f}s | '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
