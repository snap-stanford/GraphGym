"""
The ROLAND training pipeline with live-update.
"""
import copy
import datetime
import logging
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
    Construct batch required for the task (today, tomorrow).
    For current implementation, we use tomorrow = today + 1.
    As defined in batch's get_item method (used to get edge_label and
    get_label_index), edge_label and edge_label_index returned would be
    different everytime get_task_batch() is called.

    Moreover, copy node-memories (node_states and node_cells) to the batch.

    Lastly, this method moves the created task batch to the appropriate device.
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
    # This operation does not track gradient, so should not affect back-prop.
    _, _ = model(batch)  # Inplace modification on batch.
    # Collect the updated node states.
    out = dict()
    out['node_states'] = [x.detach().clone() for x in batch.node_states]
    # If node cells are also used.
    if isinstance(batch.node_cells[0], torch.Tensor):
        out['node_cells'] = [x.detach().clone() for x in batch.node_cells]

    return out


def train_step(model, optimizer, scheduler, dataset,
               task: Tuple[int, int],
               prev_node_states: Optional[Dict[str, torch.Tensor]]
               ) -> Dict[str, float]:
    """
    After receiving ground truth from a particular task, update the model by
    performing back-propagation.
    For example, on day t, the ground truth of task (t-1, t) has been revealed,
    train the model using G[t-1] for message passing and label[t] as target.
    """
    optimizer.zero_grad()
    torch.cuda.empty_cache()

    today, tomorrow = task
    model.train()
    batch = get_task_batch(dataset, today, tomorrow, prev_node_states).clone()

    pred, true = model(batch)
    loss, pred_score = compute_loss(pred, true)
    loss.backward()
    optimizer.step()

    scheduler.step()
    return {'loss': loss.item()}


@torch.no_grad()
def evaluate_step(model, dataset, task: Tuple[int, int],
                  prev_node_states: Optional[Dict[str, List[torch.Tensor]]],
                  fast: bool = False) -> Dict[str, float]:
    """
    Evaluate model's performance on task = (today, tomorrow)
        where today and tomorrow are integers indexing snapshots.
    """
    today, tomorrow = task
    model.eval()
    batch = get_task_batch(dataset, today, tomorrow, prev_node_states).clone()

    pred, true = model(batch)
    loss, pred_score = compute_loss(pred, true)

    if fast:
        # skip MRR calculation for internal validation.
        return {'loss': loss.item()}

    mrr_batch = get_task_batch(dataset, today, tomorrow,
                               prev_node_states).clone()

    mrr = train_utils.compute_MRR(
        mrr_batch,
        model,
        num_neg_per_node=cfg.metric.mrr_num_negative_edges,
        method=cfg.metric.mrr_method)

    return {'loss': loss.item(), 'mrr': mrr}


def train_live_update(loggers, loaders, model, optimizer, scheduler, datasets,
                      **kwargs):

    for dataset in datasets:
        # Sometimes edge degree info is already included in dataset.
        if not hasattr(dataset[0], 'keep_ratio'):
            train_utils.precompute_edge_degree_info(dataset)

    # if cfg.dataset.premade_datasets == 'fresh_save_cache':
    #     if not os.path.exists(f'{cfg.dataset.dir}/cache/'):
    #         os.mkdir(f'{cfg.dataset.dir}/cache/')
    #     cache_path = '{}/cache/cached_datasets_{}_{}_{}.pt'.format(
    #         cfg.dataset.dir, cfg.dataset.format.replace('.tsv', ''),
    #         cfg.transaction.snapshot_freq,
    #         datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
    #     )
    #     torch.save(datasets, cache_path)

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

    prev_node_states = None  # no previous state on day 0.
    # {'node_states': [Tensor, Tensor], 'node_cells: [Tensor, Tensor]}

    model_meta = None  # the state_dict() object of the meta-model.

    # TODO: How to incorporate logger?

    for t in tqdm(task_range, desc='Snapshot'):
        # current task: t --> t+1.
        # (1) Evaluate model's performance on this task, at this time, the
        # model has seen no information on t+1, this evaluation is fair.
        # TODO: modify here to predict on all edges?
        for i in range(1, num_splits):
            # Validation and test edges.
            perf = evaluate_step(model, datasets[i], (t, t + 1),
                                 prev_node_states)

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

        if cfg.meta.is_meta and (model_meta is not None):
            # For meta-learning, start fine-tuning from the meta-model.
            model.load_state_dict(copy.deepcopy(model_meta))

        # Internal training loop.
        for i in tqdm(range(cfg.optim.max_epoch + 1), desc='live update',
                      leave=False):
            # Start with the un-trained model (i = 0), evaluate the model.
            internal_val_perf = evaluate_step(model, datasets[1],
                                              (t, t + 1),
                                              prev_node_states, fast=True)
            val_loss = internal_val_perf['loss']

            if val_loss < best_model['val_loss']:
                # replace the best model with the current model.
                best_model = {'val_loss': val_loss, 'train_epoch': i,
                              'state': copy.deepcopy(model.state_dict())}
                best_model_unchanged = 0
            else:
                # the current best model has dominated for these epochs.
                best_model_unchanged += 1

            # if (i >= 2 * tol) and (best_model_unchanged >= tol):
            if best_model_unchanged >= tol:
                # If the best model has not been updated for a while, stop.
                break
            else:
                # Otherwise, keep training.
                train_perf = train_step(model, optimizer, scheduler,
                                        datasets[0], (t, t + 1),
                                        prev_node_states)
                writer.add_scalars('train', train_perf, t)

        writer.add_scalar('internal_best_val', best_model['val_loss'], t)
        writer.add_scalar('best epoch', best_model['train_epoch'], t)

        # (3) Actually perform the update on training set to get node_states
        # contains information up to time t.
        # Use the best model selected from intra-snapshot cross-validation.
        model.load_state_dict(best_model['state'])

        if cfg.meta.is_meta:  # update meta-learning's initialization weights.
            if model_meta is None:  # for the first task.
                model_meta = copy.deepcopy(best_model['state'])
            else:  # for subsequent task, update init.
                # (1-alpha)*model_meta + alpha*best_model.
                model_meta = train_utils.average_state_dict(model_meta,
                                                            best_model['state'],
                                                            cfg.meta.alpha)

        prev_node_states = update_node_states(model, datasets[0], (t, t + 1),
                                              prev_node_states)

    writer.close()

    if cfg.train.ckpt_clean:
        clean_ckpt()

    logging.info('Task done, results saved in {}'.format(cfg.out_dir))


register_train('live_update', train_live_update)
