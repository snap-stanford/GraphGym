import logging
import time

import torch

from graphgym.checkpoint import clean_ckpt, load_ckpt, save_ckpt
from graphgym.config import cfg
from graphgym.loss import compute_loss
from graphgym.register import register_train
from graphgym.utils.epoch import is_ckpt_epoch, is_eval_epoch


def train_epoch(logger, loader, model, optimizer, scheduler, motifs=None):
    model.train()
    # Meta-learning preparation
    if cfg.kg.meta:
        targets = list(range(cfg.share.num_task))
        meta_num = int(len(targets) * cfg.kg.meta_ratio)
        keep_target = targets[:-meta_num]
    time_start = time.time()
    for i, batch in enumerate(loader):
        if i % 2 == 0:
            batch_support = batch
            continue
        batch_query = batch

        if cfg.kg.meta:
            batch_support = mask_label(batch_support, keep_target, cfg.kg.aux)
            batch_query = mask_label(batch_query, keep_target, cfg.kg.aux)

        batch_support.to(torch.device(cfg.device))
        batch_query.to(torch.device(cfg.device))

        batch_query, pred_index, true = split_label(
            batch_query,
            setting=cfg.kg.setting,
            keep_ratio=cfg.kg.setting_ratio,
            aux=cfg.kg.aux)

        optimizer.zero_grad()

        batch_query = model.forward_emb(batch_query, motifs)
        batch_support = model.forward_emb(batch_support, motifs=None)

        # concat batch
        batch = concat_batch(batch_query, batch_support)

        pred, data_emb, task_emb = model.forward_pred(batch,
                                                      pred_index=pred_index)
        # todo: see if batch_query need forward_pred

        task_type = 'classification_binary'
        loss, pred_score = compute_loss(pred, true, task_type)

        pred_dict, true_dict = get_multi_task(
            pred=pred_score.detach().cpu(),
            true=true.detach().cpu(),
            pred_index=batch.pred_index.detach().cpu(),
            num_targets=len(targets))

        logger.update_stats(true=true_dict,
                            pred=pred_dict,
                            loss=loss.item(),
                            lr=scheduler.get_last_lr()[0],
                            time_used=time.time() - time_start,
                            params=cfg.params)

        time_start = time.time()

        # 3 update parameters
        loss.backward()  # accumulate gradient from each target
        optimizer.step()

    scheduler.step()


def eval_epoch(logger, loader_train, loader_val, model, motifs=None):
    model.eval()
    if cfg.kg.meta:
        targets = list(range(cfg.share.num_task))
        meta_num = int(len(targets) * cfg.kg.meta_ratio)
        keep_target = [0, 1] + targets[-meta_num:]
    time_start = time.time()

    for i, batch_query in enumerate(loader_val):
        if cfg.kg.meta:
            batch_query = mask_label(batch_query, keep_target, aux=cfg.kg.aux)
        batch_query.to(torch.device(cfg.device))
        batch_query, pred_index, true = split_label(
            batch_query,
            setting=cfg.kg.setting,
            keep_ratio=cfg.kg.setting_ratio,
            aux=cfg.kg.aux)
        batch_query = model.forward_emb(batch_query, motifs)

        pred_all = 0
        for j in range(cfg.kg.repeat):
            batch_support = next(iter(loader_train))

            if cfg.kg.meta:
                batch_support = mask_label(batch_support,
                                           keep_target,
                                           aux=cfg.kg.aux)

            batch_support.to(torch.device(cfg.device))

            batch_support = model.forward_emb(batch_support, motifs=None)

            # concat batch
            batch = concat_batch(batch_query, batch_support)

            pred, data_emb, task_emb = model.forward_pred(
                batch, pred_index=pred_index)
            # todo: see if batch_query need forward_pred
            pred_all += pred
        pred = pred_all / cfg.kg.repeat

        task_type = 'classification_binary'
        loss, pred_score = compute_loss(pred, true, task_type)

        pred_dict, true_dict = get_multi_task(
            pred=pred_score.detach().cpu(),
            true=true.detach().cpu(),
            pred_index=batch.pred_index.detach().cpu(),
            num_targets=len(targets))

        logger.update_stats(true=true_dict,
                            pred=pred_dict,
                            loss=loss.item(),
                            lr=0,
                            time_used=time.time() - time_start,
                            params=cfg.params)

        time_start = time.time()


def train(loggers, loaders, model, optimizer, scheduler, motifs=None):
    start_epoch = 0
    if cfg.train.auto_resume:
        start_epoch = load_ckpt(model, optimizer, scheduler)
    if start_epoch == cfg.optim.max_epoch:
        logging.info('Checkpoint found, Task already done')
    else:
        logging.info('Start from epoch {}'.format(start_epoch))

    num_splits = len(loggers)
    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        train_epoch(loggers[0], loaders[0], model, optimizer, scheduler,
                    motifs)
        loggers[0].write_epoch(cur_epoch)
        if is_eval_epoch(cur_epoch):
            for i in range(1, num_splits):
                eval_epoch(loggers[i], loaders[0], loaders[i], model, motifs)
                loggers[i].write_epoch(cur_epoch)
        if is_ckpt_epoch(cur_epoch):
            save_ckpt(model, optimizer, scheduler, cur_epoch)

    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()

    logging.info('Task done, results saved in {}'.format(cfg.out_dir))


def mask_label(batch, keep_target=[], aux=True):
    if not aux:
        target_aux = [0, 1]
        keep_target = [
            target for target in keep_target if target not in target_aux
        ]
    device = batch.graph_targets_id_batch.device
    keep_mask = torch.zeros_like(batch.graph_targets_id,
                                 dtype=torch.bool,
                                 device=device)
    for target in keep_target:
        keep_mask += batch.graph_targets_id == target
    batch.graph_targets_id_batch = batch.graph_targets_id_batch[keep_mask]
    batch.graph_targets_id = batch.graph_targets_id[keep_mask]
    batch.graph_targets_value = batch.graph_targets_value[keep_mask]
    return batch


def split_label(batch, setting='standard', keep_ratio=0.5, aux=True):
    device = batch.graph_targets_id_batch.device
    if setting == 'standard':
        if aux:
            target_aux = [0, 1]
            keep_mask = torch.zeros_like(batch.graph_targets_id,
                                         dtype=torch.bool,
                                         device=device)
            for target in target_aux:
                keep_mask += batch.graph_targets_id == target
            pred_index = torch.stack((batch.graph_targets_id_batch[~keep_mask],
                                      batch.graph_targets_id[~keep_mask]),
                                     dim=0)
            true = batch.graph_targets_value[~keep_mask]
            batch.graph_targets_id_batch = batch.graph_targets_id_batch[
                keep_mask]
            batch.graph_targets_id = batch.graph_targets_id[keep_mask]
            batch.graph_targets_value = batch.graph_targets_value[keep_mask]
        else:
            pred_index = torch.stack(
                (batch.graph_targets_id_batch, batch.graph_targets_id), dim=0)
            true = batch.graph_targets_value
            batch.graph_targets_id_batch = torch.tensor([], device=device)
            batch.graph_targets_id = torch.tensor([], device=device)
            batch.graph_targets_value = torch.tensor([], device=device)
    elif setting == 'relation':
        if aux:
            target_aux = [0, 1]
            keep_mask = torch.zeros_like(batch.graph_targets_id,
                                         dtype=torch.float,
                                         device=device).uniform_() < keep_ratio
            for target in target_aux:
                keep_mask += batch.graph_targets_id == target
            pred_index = torch.stack((batch.graph_targets_id_batch[~keep_mask],
                                      batch.graph_targets_id[~keep_mask]),
                                     dim=0)
            true = batch.graph_targets_value[~keep_mask]
            batch.graph_targets_id_batch = batch.graph_targets_id_batch[
                keep_mask]
            batch.graph_targets_id = batch.graph_targets_id[keep_mask]
            batch.graph_targets_value = batch.graph_targets_value[keep_mask]
        else:
            keep_mask = torch.zeros_like(batch.graph_targets_id,
                                         dtype=torch.float,
                                         device=device).uniform_() < keep_ratio
            pred_index = torch.stack((batch.graph_targets_id_batch[~keep_mask],
                                      batch.graph_targets_id[~keep_mask]),
                                     dim=0)
            true = batch.graph_targets_value[~keep_mask]
            batch.graph_targets_id_batch = batch.graph_targets_id_batch[
                keep_mask]
            batch.graph_targets_id = batch.graph_targets_id[keep_mask]
            batch.graph_targets_value = batch.graph_targets_value[keep_mask]
    return batch, pred_index, true


def concat_batch(batch_query, batch_support):
    id_bias = batch_query.graph_feature.shape[0]
    batch_query.graph_feature = torch.cat(
        (batch_query.graph_feature, batch_support.graph_feature), dim=0)
    if batch_query.graph_targets_id_batch.shape[0] > 0:
        batch_query.graph_targets_id = torch.cat(
            (batch_query.graph_targets_id, batch_support.graph_targets_id),
            dim=0)
        batch_query.graph_targets_value = torch.cat(
            (batch_query.graph_targets_value,
             batch_support.graph_targets_value),
            dim=0)
        batch_support.graph_targets_id_batch += id_bias
        batch_query.graph_targets_id_batch = torch.cat(
            (batch_query.graph_targets_id_batch,
             batch_support.graph_targets_id_batch),
            dim=0)
    else:
        batch_query.graph_targets_id = batch_support.graph_targets_id
        batch_query.graph_targets_value = batch_support.graph_targets_value
        batch_query.graph_targets_id_batch = \
            batch_support.graph_targets_id_batch

    return batch_query


def get_multi_task(pred, true, pred_index, num_targets):
    pred_index = pred_index[1, :]
    pred_dict = {}
    true_dict = {}
    for i in range(num_targets):
        mask = pred_index == i
        pred_tmp = pred[mask]
        true_tmp = true[mask]
        if pred_tmp.shape[0] > 0:
            pred_dict[i] = pred_tmp
        if true_tmp.shape[0] > 0:
            true_dict[i] = true_tmp

    return pred_dict, true_dict


register_train('metalink', train)
