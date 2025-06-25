"""Custom dataloaders for multi-stream neural networks."""

from torch.utils.data import DataLoader
from ..datasets import DatasetCollator


class MultiStreamDataLoader:
    """DataLoader wrapper for multi-stream datasets."""
    
    def __init__(self, dataset, batch_size=32, shuffle=True, num_workers=4, 
                 collate_fn=None, **kwargs):
        
        if collate_fn is None:
            collate_fn = DatasetCollator()
        
        self.dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            **kwargs
        )
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)


def create_train_dataloader(dataset, batch_size=32, num_workers=4, **kwargs):
    """Create training dataloader with appropriate settings."""
    return MultiStreamDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        **kwargs
    )


def create_val_dataloader(dataset, batch_size=32, num_workers=4, **kwargs):
    """Create validation dataloader with appropriate settings."""
    return MultiStreamDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        **kwargs
    )


def create_test_dataloader(dataset, batch_size=32, num_workers=4, **kwargs):
    """Create test dataloader with appropriate settings."""
    return MultiStreamDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        **kwargs
    )
