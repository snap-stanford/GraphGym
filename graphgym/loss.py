import torch
import torch.nn as nn
import torch.nn.functional as F

from graphgym.contrib.loss import *
import graphgym.register as register
from graphgym.config import cfg

def compute_loss(pred, true):
    '''

    :param pred: unnormalized prediction
    :param true: label
    :return: loss, normalized prediction score
    '''
    r = 'mean' if cfg.model.size_average else 'sum'
    bce_loss = nn.BCEWithLogitsLoss(reduction=r)
    mse_loss = nn.MSELoss(reduction=r)

    # default manipulation for pred and true
    # can be skipped if special loss computation is needed
    # if multi task binary classification, treat as flatten binary
    if true.ndim > 1 and cfg.model.loss_fun == 'cross_entropy':
        pred, true = torch.flatten(pred), torch.flatten(true)
    pred = pred.squeeze(-1) if pred.ndim > 1 else pred
    true = true.squeeze(-1) if true.ndim > 1 else true

    # Try to load customized loss
    for func in register.loss_dict.values():
        value = func(pred, true)
        if value is not None:
            return value

    if cfg.model.loss_fun == 'cross_entropy':
        # multiclass
        if pred.ndim > 1:
            pred = F.log_softmax(pred, dim=-1)
            return F.nll_loss(pred, true), pred
        # binary
        else:
            true = true.float()
            return bce_loss(pred, true), torch.sigmoid(pred)
    elif cfg.model.loss_fun == 'mse':
        true = true.float()
        return mse_loss(pred, true), pred
    else:
        raise ValueError('Loss func {} not supported'.
                         format(cfg.model.loss_fun))

