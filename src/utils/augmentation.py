"""
Enhanced data augmentation for multi-stream neural networks.
Implements augmentation strategies that work with dual-stream (RGB + Brightness) inputs.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import random
import numpy as np

class DualStreamAugmentation:
    """
    Augmentation class that applies consistent transformations to both
    RGB and brightness streams.
    """
    
    def __init__(self, 
                 flip_prob=0.5,
                 rotate_degrees=15,
                 color_jitter_brightness=0.2,
                 color_jitter_contrast=0.2,
                 color_jitter_saturation=0.2,
                 random_crop_size=32,
                 random_crop_padding=4,
                 cutout_prob=0.5,
                 cutout_size=8,
                 mixup_alpha=0.2,
                 cutmix_alpha=1.0):
        """
        Initialize augmentation with configurable parameters.
        
        Args:
            flip_prob: Probability of horizontal flip
            rotate_degrees: Max rotation degrees
            color_jitter_*: Color jitter parameters
            random_crop_*: Random crop parameters
            cutout_*: Cutout parameters
            mixup_alpha: Mixup alpha parameter
            cutmix_alpha: CutMix alpha parameter
        """
        self.flip_prob = flip_prob
        self.rotate_degrees = rotate_degrees
        self.color_jitter_brightness = color_jitter_brightness
        self.color_jitter_contrast = color_jitter_contrast
        self.color_jitter_saturation = color_jitter_saturation
        self.random_crop_size = random_crop_size
        self.random_crop_padding = random_crop_padding
        self.cutout_prob = cutout_prob
        self.cutout_size = cutout_size
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        
    def apply_geometric(self, rgb_img, brightness_img):
        """
        Apply consistent geometric transformations to both RGB and brightness images.
        
        Args:
            rgb_img: RGB image tensor [C, H, W]
            brightness_img: Brightness image tensor [1, H, W]
        
        Returns:
            Transformed RGB and brightness images
        """
        # Stack for consistent transforms
        combined = torch.cat([rgb_img, brightness_img], dim=0)
        
        # Random horizontal flip
        if random.random() < self.flip_prob:
            combined = F.hflip(combined)
            
        # Random rotation
        if self.rotate_degrees > 0:
            angle = random.uniform(-self.rotate_degrees, self.rotate_degrees)
            combined = F.rotate(combined, angle)
            
        # Random crop with padding
        if self.random_crop_padding > 0:
            padding = self.random_crop_padding
            combined = F.pad(combined, [padding, padding, padding, padding], padding_mode='reflect')
            
            # Get crop parameters
            height, width = combined.shape[1:]
            top = random.randint(0, height - self.random_crop_size)
            left = random.randint(0, width - self.random_crop_size)
            
            # Apply crop
            combined = F.crop(combined, top, left, self.random_crop_size, self.random_crop_size)
        
        # Split back to RGB and brightness
        rgb_img, brightness_img = combined[:3], combined[3:]
        
        return rgb_img, brightness_img
    
    def apply_cutout(self, rgb_img, brightness_img):
        """Apply cutout to both streams if probability check passes."""
        if random.random() < self.cutout_prob:
            # Generate random cutout parameters
            height, width = rgb_img.shape[1:]
            size = self.cutout_size
            
            # Get random center point
            center_y = random.randint(0, height)
            center_x = random.randint(0, width)
            
            # Calculate box boundaries
            y1 = max(0, center_y - size // 2)
            y2 = min(height, center_y + size // 2)
            x1 = max(0, center_x - size // 2)
            x2 = min(width, center_x + size // 2)
            
            # Apply cutout (set to zeros)
            rgb_img[:, y1:y2, x1:x2] = 0.0
            brightness_img[:, y1:y2, x1:x2] = 0.0
            
        return rgb_img, brightness_img
    
    def apply_mixup(self, rgb_batch, brightness_batch, labels_batch):
        """
        Apply Mixup augmentation to a batch of images.
        
        Args:
            rgb_batch: Batch of RGB images [B, C, H, W]
            brightness_batch: Batch of brightness images [B, 1, H, W]
            labels_batch: Labels [B]
            
        Returns:
            Mixed RGB batch, brightness batch, and mixed labels
        """
        if self.mixup_alpha <= 0:
            return rgb_batch, brightness_batch, labels_batch
            
        batch_size = rgb_batch.size(0)
        
        # Sample lambda from beta distribution
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        
        # Create shuffled indices
        index = torch.randperm(batch_size)
        
        # Mix the images
        mixed_rgb = lam * rgb_batch + (1 - lam) * rgb_batch[index, :]
        mixed_brightness = lam * brightness_batch + (1 - lam) * brightness_batch[index, :]
        
        # Create one-hot encoded labels
        y_onehot = torch.zeros(batch_size, labels_batch.max().item() + 1, 
                              device=labels_batch.device)
        y_onehot.scatter_(1, labels_batch.unsqueeze(1), 1)
        
        # Mix the labels
        mixed_y_onehot = lam * y_onehot + (1 - lam) * y_onehot[index]
        
        return mixed_rgb, mixed_brightness, mixed_y_onehot
    
    def apply_cutmix(self, rgb_batch, brightness_batch, labels_batch):
        """
        Apply CutMix augmentation to a batch of images.
        
        Args:
            rgb_batch: Batch of RGB images [B, C, H, W]
            brightness_batch: Batch of brightness images [B, 1, H, W]
            labels_batch: Labels [B]
            
        Returns:
            Mixed RGB batch, brightness batch, and mixed labels
        """
        if self.cutmix_alpha <= 0:
            return rgb_batch, brightness_batch, labels_batch
            
        batch_size = rgb_batch.size(0)
        
        # Sample lambda from beta distribution
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        
        # Create shuffled indices
        index = torch.randperm(batch_size)
        
        # Get dimensions
        _, _, H, W = rgb_batch.shape
        
        # Calculate box size
        cut_ratio = np.sqrt(1. - lam)
        cut_w = int(W * cut_ratio)
        cut_h = int(H * cut_ratio)
        
        # Get random center point
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        # Calculate box boundaries
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Create mixed images
        mixed_rgb = rgb_batch.clone()
        mixed_brightness = brightness_batch.clone()
        
        # Replace the box region with the mixed images
        mixed_rgb[:, :, bby1:bby2, bbx1:bbx2] = rgb_batch[index, :, bby1:bby2, bbx1:bbx2]
        mixed_brightness[:, :, bby1:bby2, bbx1:bbx2] = brightness_batch[index, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda based on the actual box size
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        # Create one-hot encoded labels
        y_onehot = torch.zeros(batch_size, labels_batch.max().item() + 1, 
                              device=labels_batch.device)
        y_onehot.scatter_(1, labels_batch.unsqueeze(1), 1)
        
        # Mix the labels
        mixed_y_onehot = lam * y_onehot + (1 - lam) * y_onehot[index]
        
        return mixed_rgb, mixed_brightness, mixed_y_onehot
    
    def apply_to_sample(self, rgb_img, brightness_img):
        """
        Apply augmentations to a single sample (non-batch version).
        
        Args:
            rgb_img: RGB image tensor [C, H, W]
            brightness_img: Brightness image tensor [1, H, W]
        
        Returns:
            Augmented RGB and brightness images
        """
        # Apply geometric transformations
        rgb_img, brightness_img = self.apply_geometric(rgb_img, brightness_img)
        
        # Apply cutout
        rgb_img, brightness_img = self.apply_cutout(rgb_img, brightness_img)
        
        return rgb_img, brightness_img
    
    def apply_to_batch(self, rgb_batch, brightness_batch, labels_batch=None, 
                      apply_mixup=True, apply_cutmix=True):
        """
        Apply augmentations to a batch of samples.
        
        Args:
            rgb_batch: Batch of RGB images [B, C, H, W]
            brightness_batch: Batch of brightness images [B, 1, H, W]
            labels_batch: Optional labels for label-mixing augmentations
            apply_mixup: Whether to apply mixup
            apply_cutmix: Whether to apply cutmix
            
        Returns:
            Augmented RGB batch, brightness batch, and possibly mixed labels
        """
        batch_size = rgb_batch.size(0)
        
        # Apply sample-wise augmentations
        for i in range(batch_size):
            rgb_batch[i], brightness_batch[i] = self.apply_to_sample(
                rgb_batch[i], brightness_batch[i])
        
        # Apply batch-wise augmentations if labels are provided
        if labels_batch is not None:
            # Apply either mixup or cutmix with probability 0.5 each
            if apply_mixup and apply_cutmix:
                if random.random() < 0.5:
                    rgb_batch, brightness_batch, labels_batch = self.apply_mixup(
                        rgb_batch, brightness_batch, labels_batch)
                else:
                    rgb_batch, brightness_batch, labels_batch = self.apply_cutmix(
                        rgb_batch, brightness_batch, labels_batch)
            elif apply_mixup:
                rgb_batch, brightness_batch, labels_batch = self.apply_mixup(
                    rgb_batch, brightness_batch, labels_batch)
            elif apply_cutmix:
                rgb_batch, brightness_batch, labels_batch = self.apply_cutmix(
                    rgb_batch, brightness_batch, labels_batch)
        
        return rgb_batch, brightness_batch, labels_batch
