"""Dataset wrappers and utilities."""

from torch.utils.data import Dataset


class MultiStreamWrapper(Dataset):
    """Wrapper to convert single-stream datasets to multi-stream format."""
    
    def __init__(self, base_dataset, color_transform=None, brightness_transform=None):
        self.base_dataset = base_dataset
        self.color_transform = color_transform
        self.brightness_transform = brightness_transform
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        
        if isinstance(item, tuple):
            data, target = item
        elif isinstance(item, dict):
            data = item.get('image', item.get('data'))
            target = item.get('target', item.get('label'))
        else:
            data = item
            target = 0
        
        # Create color and brightness inputs
        color_input = data
        brightness_input = self._rgb_to_brightness(data)
        
        # Apply transforms
        if self.color_transform:
            color_input = self.color_transform(color_input)
        if self.brightness_transform:
            brightness_input = self.brightness_transform(brightness_input)
        
        return {
            'color': color_input,
            'brightness': brightness_input,
            'target': target
        }
    
    def _rgb_to_brightness(self, rgb_tensor):
        """Convert RGB to brightness using luminance formula."""
        if len(rgb_tensor.shape) == 3 and rgb_tensor.shape[0] == 3:
            # Standard RGB to luminance conversion
            brightness = 0.299 * rgb_tensor[0] + 0.587 * rgb_tensor[1] + 0.114 * rgb_tensor[2]
            return brightness.unsqueeze(0)
        return rgb_tensor


class DatasetCollator:
    """Collator for batching multi-stream data."""
    
    def __call__(self, batch):
        """Collate a batch of multi-stream samples."""
        color_batch = []
        brightness_batch = []
        target_batch = []
        
        for item in batch:
            color_batch.append(item['color'])
            brightness_batch.append(item['brightness'])
            target_batch.append(item['target'])
        
        import torch
        return {
            'color': torch.stack(color_batch),
            'brightness': torch.stack(brightness_batch),
            'target': torch.tensor(target_batch)
        }
