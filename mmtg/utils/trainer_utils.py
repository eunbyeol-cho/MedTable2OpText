import logging, os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, 
                patience=7, 
                verbose=True, 
                delta=0.0005, 
                compare='increase',
                metric='auprc'

                ):
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
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.target_metric_min = 0
        self.delta = delta
        self.compare_score = self.increase if compare=='increase' else self.decrease
        self.metric = metric

    def __call__(self, target_metric):
        update_token=False
        score = target_metric

        if self.best_score is None:
            self.best_score = score

        if self.compare_score(score):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.verbose:
                print(f'Validation {self.metric} {self.compare_score.__name__}d {self.target_metric_min:.6f} --> {target_metric:.6f})')
            self.target_metric_min = target_metric
            self.counter = 0
            update_token = True
        
        return update_token

    def increase(self, score):
        if score < self.best_score*(1+self.delta):
           return True
        else:
           return False

    def decrease(self, score):
        if score > self.best_score*(1+self.delta):
            return True
        else:
           return False


def count_parameters(model):
    num_params = sum(p.numel() for p in model.parameters())
    num_params_trained = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params, num_params_trained


def model_save(path, model, optimizer, n_epoch):
    path += '.pkl'
    torch.save(
                {'model_state_dict': model.module.state_dict()  if (
                    isinstance(model, DataParallel) or
                    isinstance(model, DistributedDataParallel)
                    ) else model.state_dict(),
                'n_epoch': n_epoch,
                'optimizer_state_dict': optimizer.state_dict()},
                path
    )
    print(f'model save at : {path}')


def model_load(path, model, optimizer):
    path += '.pkl'
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location='cpu')
        n_epoch = checkpoint['n_epoch'] + 1
        state_dict = checkpoint['model_state_dict']

        if isinstance(model, DataParallel) or isinstance(model, DistributedDataParallel):
            new_state_dict = {f'module.{k}':v for k,v in state_dict.items()}
            state_dict = new_state_dict

        model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return n_epoch, model, optimizer
    else:
        raise AssertionError(f"There is no ckpt at {path}")
    

def log_from_dict(epoch_log, data_type, epoch):
    summary = {'epoch':epoch}
    for key, values in epoch_log.items():
        summary[f'{data_type}/{key}'] = values
        print(f'{epoch}:\t{data_type}/{key} : {values:.3f}')
        
    return summary