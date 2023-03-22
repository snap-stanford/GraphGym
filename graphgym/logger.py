import logging
import math
import sys

import torch

import graphgym.register as register
from graphgym.config import cfg
from graphgym.utils.device import get_current_gpu_usage
from graphgym.utils.io import dict_to_json, dict_to_tb, makedirs


def setup_printing():
    """
    Set up printing options

    """
    logging.root.handlers = []
    logging_cfg = {'level': logging.INFO, 'format': '%(message)s'}
    makedirs(cfg.run_dir)
    h_file = logging.FileHandler('{}/logging.log'.format(cfg.run_dir))
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
    def __init__(self, name='train', task_type=None):
        self.name = name
        self.task_type = task_type

        self._epoch_total = cfg.optim.max_epoch
        self._time_total = 0  # won't be reset

        self.out_dir = '{}/{}'.format(cfg.run_dir, name)
        makedirs(self.out_dir)
        if cfg.tensorboard_each_run:
            from tensorboardX import SummaryWriter
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
        self._custom_stats = {}

    # basic properties
    def basic(self):
        stats = {
            'loss': round(self._loss / self._size_current, cfg.round),
            'lr': round(self._lr, cfg.round),
            'params': self._params,
            'time_iter': round(self.time_iter(), cfg.round),
        }
        gpu_memory = get_current_gpu_usage()
        if gpu_memory > 0:
            stats['gpu_memory'] = gpu_memory
        return stats

    # customized input properties
    def custom(self):
        if len(self._custom_stats) == 0:
            return {}
        out = {}
        for key, val in self._custom_stats.items():
            out[key] = val / self._size_current
        return out

    def _get_pred_int(self, pred_score):
        if len(pred_score.shape) == 1 or pred_score.shape[1] == 1:
            return (pred_score > cfg.model.thresh).long()
        else:
            return pred_score.max(dim=1)[1]

    # task properties
    def classification_binary(self):
        from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                     recall_score, roc_auc_score)

        true, pred_score = torch.cat(self._true), torch.cat(self._pred)
        pred_int = self._get_pred_int(pred_score)
        try:
            r_a_score = roc_auc_score(true, pred_score)
        except ValueError:
            r_a_score = 0.0
        return {
            'accuracy': round(accuracy_score(true, pred_int), cfg.round),
            'precision': round(precision_score(true, pred_int), cfg.round),
            'recall': round(recall_score(true, pred_int), cfg.round),
            'f1': round(f1_score(true, pred_int), cfg.round),
            'auc': round(r_a_score, cfg.round),
        }

    def classification_multi(self):
        from sklearn.metrics import accuracy_score

        true, pred_score = torch.cat(self._true), torch.cat(self._pred)
        pred_int = self._get_pred_int(pred_score)
        return {'accuracy': round(accuracy_score(true, pred_int), cfg.round)}

    def collate_dict(self, pred):
        ''' Collate list of dict'''
        pred_all = pred[0]
        for i in range(1, len(pred)):
            pred_tmp = pred[i]
            for key, val in pred_tmp.items():
                if key in pred_all:
                    pred_all[key] = torch.cat((pred_all[key], val), dim=-1)
                else:
                    pred_all[key] = val
        return pred_all

    def classification_binary_multi(self):
        from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                     recall_score, roc_auc_score)
        true_dict = self.collate_dict(self._true)
        pred_score_dict = self.collate_dict(self._pred)
        count = 0
        stats = None
        for key in true_dict.keys():
            true = true_dict[key]
            if len(torch.unique(true)) == 1:
                continue
            pred_score = pred_score_dict[key]
            pred_int = self._get_pred_int(pred_score_dict[key])
            if stats is None:
                stats = {
                    'accuracy': accuracy_score(true, pred_int),
                    'precision': precision_score(true, pred_int),
                    'recall': recall_score(true, pred_int),
                    'f1': f1_score(true, pred_int),
                    'auc': roc_auc_score(true, pred_score),
                }
            else:
                stats['accuracy'] += accuracy_score(true, pred_int)
                stats['precision'] += precision_score(true, pred_int)
                stats['recall'] += recall_score(true, pred_int)
                stats['f1'] += f1_score(true, pred_int)
                stats['auc'] += roc_auc_score(true, pred_score)
            count += 1
        for key in stats.keys():
            stats[key] = round(stats[key] / count, cfg.round)
        return stats

    def regression(self):
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        true, pred = torch.cat(self._true), torch.cat(self._pred)
        return {
            'mae':
            float(round(mean_absolute_error(true, pred), cfg.round)),
            'mse':
            float(round(mean_squared_error(true, pred), cfg.round)),
            'rmse':
            float(round(math.sqrt(mean_squared_error(true, pred)), cfg.round))
        }

    def time_iter(self):
        return self._time_used / self._iter

    def eta(self, epoch_current):
        epoch_current += 1  # since counter starts from 0
        time_per_epoch = self._time_total / epoch_current
        return time_per_epoch * (self._epoch_total - epoch_current)

    def update_stats(self, true, pred, loss, lr, time_used, params, **kwargs):
        self._iter += 1
        if type(true) is dict and type(pred) is dict:
            self._true.append(true)
            self._pred.append(pred)
            batch_size = sum([val.shape[0] for val in true.values()])
        else:
            self._true.append(true)
            self._pred.append(pred)
            batch_size = true.shape[0]
        self._size_current += batch_size
        self._loss += loss * batch_size
        self._lr = lr
        self._params = params
        self._time_used += time_used
        self._time_total += time_used
        for key, val in kwargs.items():
            if key not in self._custom_stats:
                self._custom_stats[key] = val * batch_size
            else:
                self._custom_stats[key] += val * batch_size

    def write_iter(self):
        raise NotImplementedError

    def write_epoch(self, cur_epoch):
        basic_stats = self.basic()

        # Try to load customized metrics
        task_stats = {}
        for custom_metric in cfg.custom_metrics:
            func = register.metric_dict.get(custom_metric)
            if not func:
                raise ValueError(
                    f'Unknown custom metric function name: {custom_metric}')
            custom_metric_score = func(self._true, self._pred, self.task_type)
            task_stats[custom_metric] = custom_metric_score

        if not task_stats:  # use default metrics if no matching custom metric
            if self.task_type == 'regression':
                task_stats = self.regression()
            elif self.task_type == 'classification_binary':
                task_stats = self.classification_binary()
            elif self.task_type == 'classification_multi':
                task_stats = self.classification_multi()
            elif self.task_type == 'classification_binary_multi':
                task_stats = self.classification_binary_multi()
            else:
                raise ValueError('Task has to be regression or classification')

        epoch_stats = {'epoch': cur_epoch}
        eta_stats = {'eta': round(self.eta(cur_epoch), cfg.round)}
        custom_stats = self.custom()

        if self.name == 'train':
            stats = {
                **epoch_stats,
                **eta_stats,
                **basic_stats,
                **task_stats,
                **custom_stats
            }
        else:
            stats = {
                **epoch_stats,
                **basic_stats,
                **task_stats,
                **custom_stats
            }

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


def infer_task():
    num_label = cfg.share.dim_out
    if cfg.dataset.task_type == 'classification':
        if num_label <= 2:
            if cfg.share.num_task == 1:
                task_type = 'classification_binary'
            else:
                task_type = 'classification_binary_multi'
        else:
            task_type = 'classification_multi'
    else:
        task_type = cfg.dataset.task_type
    return task_type


def create_logger():
    """
    Create logger for the experiment

    Returns: List of logger objects

    """
    loggers = []
    names = ['train', 'val', 'test']
    for i, dataset in enumerate(range(cfg.share.num_splits)):
        loggers.append(Logger(name=names[i], task_type=infer_task()))
    return loggers
