"""Derived brightness dataset implementations."""

import torchvision
from torchvision import transforms
from .base_dataset import BaseMultiStreamDataset


class DerivedBrightnessDataset(BaseMultiStreamDataset):
    """Dataset that derives brightness from RGB images."""
    
    def __init__(self, root, train=True, download=False, transform=None, 
                 target_transform=None, dataset_type='cifar10'):
        super().__init__(transform, target_transform)
        
        self.dataset_type = dataset_type
        
        # Base transforms for the underlying dataset
        base_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        # Load the underlying dataset
        if dataset_type == 'cifar10':
            self.base_dataset = torchvision.datasets.CIFAR10(
                root=root, train=train, download=download, transform=base_transform
            )
        elif dataset_type == 'cifar100':
            self.base_dataset = torchvision.datasets.CIFAR100(
                root=root, train=train, download=download, transform=base_transform
            )
        elif dataset_type == 'imagenet':
            # For ImageNet, assume it's already downloaded
            split = 'train' if train else 'val'
            self.base_dataset = torchvision.datasets.ImageNet(
                root=root, split=split, transform=base_transform
            )
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        image, target = self.base_dataset[idx]
        
        # Get color and brightness inputs
        color_input = self.get_color_input(image)
        brightness_input = self.get_brightness_input(image)
        
        # Apply transforms if specified
        if self.transform:
            color_input = self.transform(color_input)
            brightness_input = self.transform(brightness_input)
        
        if self.target_transform:
            target = self.target_transform(target)
        
        return {
            'color': color_input,
            'brightness': brightness_input,
            'target': target
        }
