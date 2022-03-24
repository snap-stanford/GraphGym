import logging
import os

import torch
from torch_geometric import seed_everything

from graphgym.cmd_args import parse_args
from graphgym.config import cfg, dump_cfg, load_cfg, set_agg_dir, set_run_dir
from graphgym.loader_pyg import create_loader
from graphgym.logger import create_logger, setup_printing
from graphgym.model_builder_pyg import create_model
from graphgym.optimizer import create_optimizer, create_scheduler
from graphgym.register import train_dict
from graphgym.train_pyg import train
from graphgym.utils.agg_runs import agg_runs
from graphgym.utils.comp_budget import params_count
from graphgym.utils.device import auto_select_device

if __name__ == '__main__':
    # Load cmd line args
    args = parse_args()
    # Load config file
    load_cfg(cfg, args)
    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    dump_cfg(cfg)
    # Repeat for different random seeds
    for i in range(args.repeat):
        set_run_dir(cfg.out_dir, args.cfg_file)
        setup_printing()
        # Set configurations for each run
        cfg.seed = cfg.seed + 1
        seed_everything(cfg.seed)
        auto_select_device()
        # Set machine learning pipeline
        loaders = create_loader()
        loggers = create_logger()
        model = create_model()
        optimizer = create_optimizer(model.parameters())
        scheduler = create_scheduler(optimizer)
        # Print model info
        logging.info(model)
        logging.info(cfg)
        cfg.params = params_count(model)
        logging.info('Num parameters: %s', cfg.params)
        # Start training
        if cfg.train.mode == 'standard':
            train(loggers, loaders, model, optimizer, scheduler)
        else:
            train_dict[cfg.train.mode](loggers, loaders, model, optimizer,
                                       scheduler)
    # Aggregate results from different seeds
    agg_runs(set_agg_dir(cfg.out_dir, args.cfg_file), cfg.metric_best)
    # When being launched in batch mode, mark a yaml as done
    if args.mark_done:
        os.rename(args.cfg_file, f'{args.cfg_file}_done')
