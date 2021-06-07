"""
The baseline training (non-incremental) training for live-update scheme.
NOTE: this setup requires extensive GPU memory and could lead to OOM error.
"""
import copy
import datetime
import logging
import os
from typing import Dict, List, Optional, Tuple

import deepsnap
import numpy as np
import torch
from graphgym.checkpoint import clean_ckpt
from graphgym.config import cfg
from graphgym.contrib.train import train_utils
from graphgym.loss import compute_loss
from graphgym.optimizer import create_optimizer, create_scheduler
from graphgym.register import register_train
from graphgym.utils.io import makedirs_rm_exist
from graphgym.utils.stats import node_degree
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


@torch.no_grad()
def get_task_batch(dataset: deepsnap.dataset.GraphDataset,
                   today: int, tomorrow: int,
                   prev_node_states: Optional[Dict[str, List[torch.Tensor]]]
                   ) -> deepsnap.graph.Graph:
    """
    Construct batch required for the task (today, tomorrow). As defined in
    batch's get_item method (used to get edge_label and get_label_index),
    edge_label and edge_label_index returned would be different everytime
    get_task_batch() is called.

    Moreover, copy node-memories (node_states and node_cells) to the batch.
    """
    assert today < tomorrow < len(dataset)
    # Get edges for message passing and prediction task.
    batch = dataset[today].clone()
    batch.edge_label = dataset[tomorrow].edge_label.clone()
    batch.edge_label_index = dataset[tomorrow].edge_label_index.clone()

    # Copy previous memory to the batch.
    if prev_node_states is not None:
        for key, val in prev_node_states.items():
            copied = [x.detach().clone() for x in val]
            setattr(batch, key, copied)

    batch = train_utils.move_batch_to_device(batch, cfg.device)
    return batch


@torch.no_grad()
def update_node_states(model, dataset, task: Tuple[int, int],
                       prev_node_states: Optional[
                           Dict[str, List[torch.Tensor]]]
                       ) -> Dict[str, List[torch.Tensor]]:
    """Perform the provided task and keep track of the latest node_states.

    Example: task = (t, t+1),
        the prev_node_states contains node embeddings at time (t-1).
        the model perform task (t, t+1):
            Input: (node embedding at t - 1, edges at t).
            Output: possible transactions at t+1.
        the model also generates node embeddings at t.

    after doing task (t, t+1), node_states contains information
    from snapshot t.
    """
    today, tomorrow = task
    batch = get_task_batch(dataset, today, tomorrow, prev_node_states).clone()
    # Let the model modify batch.node_states (and batch.node_cells).
    _, _ = model(batch)
    # Collect the updated node states.
    out = dict()
    out['node_states'] = [x.detach().clone() for x in batch.node_states]
    if isinstance(batch.node_cells[0], torch.Tensor):
        out['node_cells'] = [x.detach().clone() for x in batch.node_cells]

    return out


def train_step(model, optimizer, scheduler, dataset,
               task: Tuple[int, int]) -> dict:
    """
    After receiving ground truth from a particular task, update the model by
    performing back-propagation.
    For example, on day t, the ground truth of task (t-1, t) has been revealed,
    train the model using G[t-1] for message passing and label[t] as target.
    """
    optimizer.zero_grad()
    torch.cuda.empty_cache()
    model.train()

    today, _ = task

    # get loss over time.
    total_loss_over_time = torch.tensor(0.0).to(torch.device(cfg.device))
    # iterate from the beginning to compute node_states.
    for t in range(today + 1):  # (0, 1), (1, 2), ..., (today, today+1).
        # perform task (t, t+1), use information up to tomorrow.
        new_batch = get_task_batch(dataset, t, t + 1, None).clone()
        if t > 0:  # manually inherit node states and node cells for LSTM.
            new_batch.node_states = batch.node_states
            new_batch.node_cells = batch.node_cells
        batch = new_batch
        pred, true = model(batch)
        loss, _ = compute_loss(pred, true)
        if t > today - cfg.train.tbptt_freq:
            # Perform the truncated version, only accumulate loss for recent
            # snapshots.
            total_loss_over_time += loss
    # get average loss over time.
    total_loss_over_time /= (today + 1)
    # perform back-prop through time.
    total_loss_over_time.backward()
    optimizer.step()

    scheduler.step()
    return {'loss': total_loss_over_time}


@torch.no_grad()
def evaluate_step(model, dataset, task: Tuple[int, int], fast: bool = False
                  ) -> dict:
    """
    Evaluate model's performance on task = (today, tomorrow)
        where today and tomorrow are integers indexing snapshots.
    """
    today, tomorrow = task
    model.eval()

    # Run forward pass to get the latest node states.
    for t in range(today):  # (0, 1), (1, 2), ...(today-1, today)
        # Iterative through snapshots in the past, up to (today-1, today)
        new_batch = get_task_batch(dataset, t, t + 1, None).clone()
        if t > 0:
            new_batch.node_states = batch.node_states
            new_batch.node_cells = batch.node_cells
        batch = new_batch
        # forward pass to update node_states in batch.
        _, _ = model(batch)

    # Evaluation.
    # (today, today+1)
    cur_batch = get_task_batch(dataset, today, tomorrow, None).clone()
    if today > 0:
        cur_batch.node_states = copy.deepcopy(batch.node_states)
        cur_batch.node_cells = copy.deepcopy(batch.node_cells)

    pred, true = model(cur_batch)
    loss, _ = compute_loss(pred, true)

    if fast:
        # skip MRR calculation for internal validation.
        return {'loss': loss.item()}

    mrr_batch = get_task_batch(dataset, today, tomorrow, None).clone()
    if today > 0:
        mrr_batch.node_states = copy.deepcopy(batch.node_states)
        mrr_batch.node_cells = copy.deepcopy(batch.node_cells)

    mrr = train_utils.compute_MRR(
        mrr_batch,
        model,
        num_neg_per_node=cfg.metric.mrr_num_negative_edges,
        method=cfg.metric.mrr_method)

    return {'loss': loss.item(), 'mrr': mrr}


def train_live_update_bptt(loggers, loaders, model, optimizer, scheduler, datasets,
                           **kwargs):
    for dataset in datasets:
        # Sometimes edge degree info is already included in dataset.
        if not hasattr(dataset[0], 'keep_ratio'):
            train_utils.precompute_edge_degree_info(dataset)

    num_splits = len(loggers)  # train/val/test splits.
    # range for today in (today, tomorrow) task pairs.
    task_range = range(len(datasets[0]) - cfg.transaction.horizon)

    t = datetime.datetime.now().strftime('%b%d_%H-%M-%S')

    # directory to store tensorboard files of this run.
    out_dir = cfg.out_dir.replace('/', '\\')
    # dir to store all run outputs for the entire batch.
    run_dir = 'runs_' + cfg.remark

    print(f'Tensorboard directory: {out_dir}')
    # If tensorboard directory exists, this config is in the re-run phase
    # of run_batch, replace logs of previous runs with the new one.
    makedirs_rm_exist(f'./{run_dir}/{out_dir}')
    writer = SummaryWriter(f'./{run_dir}/{out_dir}')

    # save a copy of configuration for later identifications.
    with open(f'./{run_dir}/{out_dir}/config.yaml', 'w') as f:
        cfg.dump(stream=f)

    for t in tqdm(task_range, desc='Snapshot'):
        # current task: t --> t+1.
        # (1) Evaluate model's performance on this task, at this time, the
        # model has seen no information on t+1, this evaluation is fair.
        for i in range(1, num_splits):
            perf = evaluate_step(model, datasets[i], (t, t + 1), fast=False)

            writer.add_scalars('val' if i == 1 else 'test', perf, t)

        # (2) Reveal the ground truth of task (t, t+1) and update the model
        # to prepare for the next task.
        del optimizer, scheduler  # use new optimizers.
        optimizer = create_optimizer(model.parameters())
        scheduler = create_scheduler(optimizer)

        # best model's validation loss, training epochs, and state_dict.
        best_model = {'val_loss': np.inf, 'train_epoch': 0, 'state': None}
        # keep track of how long we have NOT update the best model.
        best_model_unchanged = 0
        # after not updating the best model for `tol` epochs, stop.
        tol = cfg.train.internal_validation_tolerance

        # internal training loop (intra-snapshot cross-validation).
        # choose the best model using current validation set, prepare for
        # next task.

        for i in tqdm(range(cfg.optim.max_epoch + 1), desc='live update',
                      leave=False):
            # Start with the un-trained model (i = 0), evaluate the model.
            internal_val_perf = evaluate_step(model, datasets[1],
                                              (t, t + 1), fast=True)
            val_loss = internal_val_perf['loss']

            if val_loss < best_model['val_loss']:
                # replace the best model with the current model.
                best_model = {'val_loss': val_loss, 'train_epoch': i,
                              'state': copy.deepcopy(model.state_dict())}
                best_model_unchanged = 0
            else:
                # the current best model has dominated for these epochs.
                best_model_unchanged += 1

            if best_model_unchanged >= tol:
                # If the best model has not been updated for a while, stop.
                break
            else:
                # Otherwise, keep training.
                train_perf = train_step(model, optimizer, scheduler,
                                        datasets[0], (t, t + 1))
                writer.add_scalars('train', train_perf, t)

        writer.add_scalar('internal_best_val', best_model['val_loss'], t)
        writer.add_scalar('best epoch', best_model['train_epoch'], t)

        # (3) Actually perform the update on training set to get node_states
        # contains information up to time t.
        # Use the best model selected from intra-snapshot cross-validation.
        model.load_state_dict(best_model['state'])

    writer.close()

    if cfg.train.ckpt_clean:
        clean_ckpt()

    logging.info('Task done, results saved in {}'.format(cfg.out_dir))


register_train('live_update_baseline', train_live_update_bptt)
