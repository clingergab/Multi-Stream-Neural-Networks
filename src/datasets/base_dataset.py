"""Base dataset class for multi-stream neural networks."""

from torch.utils.data import Dataset


class BaseMultiStreamDataset(Dataset):
    """Base class for multi-stream datasets."""
    
    def __init__(self, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        raise NotImplementedError("Subclasses must implement __len__")
    
    def __getitem__(self, idx):
        raise NotImplementedError("Subclasses must implement __getitem__")
    
    def get_color_input(self, data):
        """Extract color input from data. Override if needed."""
        return data
    
    def get_brightness_input(self, data):
        """Extract brightness input from data. Override if needed."""
        # Default: convert to grayscale
        if len(data.shape) == 3 and data.shape[0] == 3:
            # RGB to grayscale using standard weights
            brightness = 0.299 * data[0] + 0.587 * data[1] + 0.114 * data[2]
            return brightness.unsqueeze(0)
        return data
