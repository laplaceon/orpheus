import torch
import numpy as np

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, val_loss_min=None, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = -val_loss_min if val_loss_min is not None else None
        self.early_stop = False
        self.val_loss_min = val_loss_min if val_loss_min is not None else np.Inf
        self.delta = delta
        self.trace_func = trace_func
    
    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_loss, model, opts)

            return True
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
            
            return False
        else:
            self.best_score = score
            # self.save_checkpoint(val_loss, model, opts)
            self.counter = 0

            return True

    def save_checkpoint(self, val_loss, maps, paths):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        for map, path in zip(maps, paths):
            chk = {"loss": val_loss}
            chk.update(map)
            torch.save(chk, path)
        self.val_loss_min = val_loss