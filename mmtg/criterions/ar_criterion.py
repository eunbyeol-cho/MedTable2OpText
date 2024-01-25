import torch
import torch.nn as nn
from .base_criterion import BaseCriterion


class ARCriterion(BaseCriterion):
    def __init__(self, _config):
        super().__init__(_config)
        self.test_only = _config['test_only']
        self.reset()

    def reset(self):
        self.loss = {
            'text':0,
        }
        self.acc = {
            'text':0,
        }


    def __call__(self, criterion,  x, targets):
        iter_log = self.compute_mlm(criterion, x, targets)

        if criterion == 'loss':
            self.loss['text'] += iter_log['text_loss']
        elif criterion == 'acc':
            self.acc['text'] += iter_log['text_acc']
                
        return iter_log

    def get_epoch_dict(self, total_iter):
        epoch_log = {}
        for loss_type in self.loss.keys():
            epoch_log[f'{loss_type}_loss'] = self.loss[loss_type] / total_iter
        for acc_type in self.acc.keys():
            epoch_log[f'{acc_type}_acc'] = self.acc[acc_type] / total_iter
        self.reset()
        return epoch_log


    def compute_mlm(self, criterion,  x, targets):
        device = x.device
        pred = x
        label = targets['text_input_ids'].to(device)
        
        if criterion == 'loss':
            return {
                'text_loss': self.criterion(pred.permute(0,2,1), label) 
            }

        elif criterion == 'acc':
            pred_argmax = torch.argmax(pred, dim=-1)
            pad_mask = label == 0
            correct = (label[~pad_mask]==pred_argmax[~pad_mask]).sum().detach().cpu()
            return {
                'text_acc':int(correct)/(len(label[~pad_mask]))
            }


    @property
    def compare(self):
        return 'decrease' if 'loss' in self.update_target else 'increase'

    @property
    def update_target(self):
        return 'text_acc'