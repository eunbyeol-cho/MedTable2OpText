import torch
import torch.nn as nn


class BaseCriterion():
    def __init__(self, _config):
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    @property
    def compare(self):
        raise NotImplementedError


    def reset(self):
        raise NotImplementedError


    def __call__(self):
        raise NotImplementedError