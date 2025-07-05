"""
Early stopping utility for neural network training.
"""


class EarlyStopping:
    """
    Early stopping to stop training when validation loss stops improving.
    
    Args:
        patience (int): How many epochs to wait after last time validation loss improved.
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        verbose (bool): If True, prints a message for each validation loss improvement.
        restore_best_weights (bool): If True, restores model weights from the best epoch.
    """
    
    def __init__(self, patience=7, min_delta=0, verbose=False, restore_best_weights=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.restore_best_weights = restore_best_weights
        self.wait = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_weights = None
        
    def __call__(self, val_loss, model=None):
        """
        Call early stopping check.
        
        Args:
            val_loss (float): Current validation loss
            model (torch.nn.Module): Model to save weights from (optional)
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.wait = 0
            if self.verbose:
                print(f'Early stopping: validation loss improved to {val_loss:.6f}')
            if self.restore_best_weights and model is not None:
                self.best_weights = model.state_dict().copy()
        else:
            self.wait += 1
            if self.verbose:
                print(f'Early stopping: validation loss did not improve. Wait: {self.wait}/{self.patience}')
            if self.wait >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f'Early stopping: stopping training as validation loss did not improve for {self.patience} epochs')
                    
    def restore_weights(self, model):
        """
        Restore best weights to model.
        
        Args:
            model (torch.nn.Module): Model to restore weights to
        """
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            if self.verbose:
                print('Early stopping: restored best weights')
        else:
            if self.verbose:
                print('Early stopping: no best weights to restore')
