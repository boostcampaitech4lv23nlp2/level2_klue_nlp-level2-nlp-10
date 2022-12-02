import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
from utils import print_msg

class FocalLoss(nn.Module):
    '''
    Multi-class Focal loss implementation
    '''
    def __init__(self, alpha=1, gamma=2, eps=1e-9, weight=None, reduction='mean', ignore_index: int = -100,):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.weight = weight
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index
        print_msg("Focal Loss를 사용합니다...", "INFO")

    def forward(self, pred, target):
        logpt = F.log_softmax(pred + self.eps, dim=1)
        pt = torch.exp(logpt)
        
        # compute the actual focal loss
        weight = torch.pow(1.0 - pt, self.gamma)
        logpt = self.alpha * weight * logpt
        loss = F.nll_loss(logpt, target, self.weight, ignore_index = self.ignore_index)
        return loss