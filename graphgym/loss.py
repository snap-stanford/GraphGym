import torch
import torch.nn as nn
import torch.nn.functional as F

import graphgym.register as register
from graphgym.config import cfg


def compute_loss(pred, true, task_type=None):
    """
    Compute loss and prediction score

    Args:
        pred (torch.tensor): Unnormalized prediction
        true (torch.tensor): Ground truth labels
        task_type (str): User specified task type

    Returns: Loss, normalized prediction score

    """
    bce_loss = nn.BCEWithLogitsLoss(reduction=cfg.model.size_average)
    mse_loss = nn.MSELoss(reduction=cfg.model.size_average)

    # default manipulation for pred and true
    # can be skipped if special loss computation is needed
    pred = pred.squeeze(-1) if pred.ndim > 1 else pred
    true = true.squeeze(-1) if true.ndim > 1 else true

    # Try to load customized loss
    for func in register.loss_dict.values():
        value = func(pred, true)
        if value is not None:
            return value
    if task_type is None:
        if cfg.model.loss_fun == 'cross_entropy':
            # multiclass
            if pred.ndim > 1 and true.ndim == 1:
                pred = F.log_softmax(pred, dim=-1)
                return F.nll_loss(pred, true), pred
            # binary or multilabel
            else:
                true = true.float()
                return bce_loss(pred, true), torch.sigmoid(pred)
        elif cfg.model.loss_fun == 'mse':
            true = true.float()
            return mse_loss(pred, true), pred
        else:
            raise ValueError('Loss func {} not supported'.format(
                cfg.model.loss_fun))
    else:
        if task_type == 'classification_multi':
            pred = F.log_softmax(pred, dim=-1)
            return F.nll_loss(pred, true), pred
        elif 'classification' in task_type and 'binary' in task_type:
            true = true.float()
            return bce_loss(pred, true), torch.sigmoid(pred)
        elif task_type == 'regression':
            true = true.float()
            return mse_loss(pred, true), pred
        else:
            raise ValueError('Task type {} not supported'.format(task_type))
