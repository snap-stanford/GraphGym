import torch
import math
import os
import sys
import logging
from graphgym.config import cfg
from graphgym.utils.io import dict_to_json, dict_to_tb
import pdb

from sklearn.metrics import *
from tensorboardX import SummaryWriter


def setup_printing():
    logging.root.handlers = []
    logging_cfg = {'level': logging.INFO, 'format': '%(message)s'}
    h_file = logging.FileHandler('{}/logging.log'.format(cfg.out_dir))
    h_stdout = logging.StreamHandler(sys.stdout)
    if cfg.print == 'file':
        logging_cfg['handlers'] = [h_file]
    elif cfg.print == 'stdout':
        logging_cfg['handlers'] = [h_stdout]
    elif cfg.print == 'both':
        logging_cfg['handlers'] = [h_file, h_stdout]
    else:
        raise ValueError('Print option not supported')
    logging.basicConfig(**logging_cfg)


class Logger(object):
    def __init__(self, iter_total, num_labels, name='train', task_type=None):
        self._iter_total = iter_total
        self._num_labels = num_labels
        self._epoch_total = cfg.optim.max_epoch
        self._time_total = 0  # won't be reset
        self.name = name

        if task_type == None:
            if cfg.dataset.task_type == 'classification':
                if self._num_labels == 2:
                    self.task_type = 'classification_binary'
                else:
                    self.task_type = 'classification_multi'
            else:
                self.task_type = cfg.dataset.task_type
        else:
            self.task_type = task_type

        self.out_dir = '{}/{}'.format(cfg.out_dir, name)
        os.makedirs(self.out_dir, exist_ok=True)
        if cfg.tensorboard_each_run:
            self.tb_writer = SummaryWriter(self.out_dir)

        self.reset()

    def __getitem__(self, key):
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def reset(self):
        self._iter = 0
        self._size_current = 0
        self._loss = 0
        self._lr = 0
        self._params = 0
        self._time_used = 0
        self._true = []
        self._pred = []

    # basic properties
    def basic(self):
        return {'loss': round(self._loss / self._size_current, cfg.round),
                'lr': round(self._lr, cfg.round),
                'params': self._params,
                'time_iter': round(self.time_iter(), cfg.round)
                }

    def _get_pred_int(self, pred_score):
        if len(pred_score.shape) == 1 or pred_score.shape[1] == 1:
            return (pred_score > cfg.model.thresh).long()
        else:
            return pred_score.max(dim=1)[1]

    # task properties
    def classification_binary(self):
        true, pred_score = torch.cat(self._true), torch.cat(self._pred)
        pred_int = self._get_pred_int(pred_score)
        return {'accuracy': round(accuracy_score(true, pred_int), cfg.round),
                'precision': round(precision_score(true, pred_int), cfg.round),
                'recall': round(recall_score(true, pred_int), cfg.round),
                'f1': round(f1_score(true, pred_int), cfg.round),
                'auc': round(roc_auc_score(true, pred_score), cfg.round),
                }

    def classification_multi(self):
        true, pred_score = torch.cat(self._true), torch.cat(self._pred)
        pred_int = self._get_pred_int(pred_score)
        return {'accuracy': round(accuracy_score(true, pred_int), cfg.round)}

    def regression(self):
        true, pred = torch.cat(self._true), torch.cat(self._pred)
        return {'mae': float(round(mean_absolute_error(true, pred), cfg.round)),
                'mse': float(round(mean_squared_error(true, pred), cfg.round)),
                'rmse': float(
                    round(math.sqrt(mean_squared_error(true, pred)), cfg.round))
                }

    def time_iter(self):
        return self._time_used / self._iter

    # deprecated
    def eta_epoch(self):
        return self.time_iter() * self._iter_total - self._time_used

    # deprecated
    def eta_total(self, epoch_current):
        time_epoch = self.time_iter() * self._iter_total
        return time_epoch - self._time_used + \
               time_epoch * (self._epoch_total - epoch_current)

    def eta(self, epoch_current):
        epoch_current += 1  # since counter starts from 0
        time_per_epoch = self._time_total / epoch_current
        return time_per_epoch * (self._epoch_total - epoch_current)

    def update_stats(self, true, pred, loss, lr, time_used, params):
        assert true.shape[0] == pred.shape[0]
        self._iter += 1
        self._true.append(true)
        self._pred.append(pred)
        batch_size = true.shape[0]
        self._size_current += batch_size
        self._loss += loss * batch_size
        self._lr = lr
        self._params = params
        self._time_used += time_used
        self._time_total += time_used

    def write_iter(self):
        raise NotImplementedError

    def write_epoch(self, cur_epoch):
        basic_stats = self.basic()

        if self.task_type == 'regression':
            task_stats = self.regression()
        elif self.task_type == 'classification_binary':
            task_stats = self.classification_binary()
        elif self.task_type == 'classification_multi':
            task_stats = self.classification_multi()
        else:
            raise ValueError('Task has to be regression or classification')

        epoch_stats = {'epoch': cur_epoch}
        eta_stats = {'eta': round(self.eta(cur_epoch), cfg.round)}

        if self.name == 'train':
            stats = {**epoch_stats, **eta_stats, **basic_stats, **task_stats}
        else:
            stats = {**epoch_stats, **basic_stats, **task_stats}

        # print
        logging.info('{}: {}'.format(self.name, stats))
        # json
        dict_to_json(stats, '{}/stats.json'.format(self.out_dir))
        # tensorboard
        if cfg.tensorboard_each_run:
            dict_to_tb(stats, self.tb_writer, cur_epoch)
        self.reset()

    def close(self):
        if cfg.tensorboard_each_run:
            self.tb_writer.close()


def create_logger(datasets, loaders):
    loggers = []
    names = ['train', 'val', 'test']
    for i, dataset in enumerate(datasets):
        iter_total = len(loaders[i])
        num_label = datasets[0].num_labels
        loggers.append(Logger(iter_total, num_label, name=names[i]))
    return loggers
