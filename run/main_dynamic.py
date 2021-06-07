import logging
import os
import random
import warnings
from datetime import datetime
from itertools import product

import numpy as np
import torch
from graphgym.cmd_args import parse_args
from graphgym.config import (assert_cfg, cfg, dump_cfg, get_parent_dir,
                             update_out_dir)
from graphgym.contrib.train import *
from graphgym.loader import create_dataset, create_loader
from graphgym.logger import create_logger, setup_printing
from graphgym.model_builder import create_model
from graphgym.optimizer import create_optimizer, create_scheduler
from graphgym.register import train_dict
from graphgym.train import train
from graphgym.utils.agg_runs import agg_runs
from graphgym.utils.comp_budget import params_count
from graphgym.utils.device import auto_select_device

os.environ['MPLCONFIGDIR'] = "/tmp"


if __name__ == '__main__':
    # Load cmd line args
    args = parse_args()
    # Repeat for different random seeds
    for i in range(args.repeat):
        # Load config file
        cfg.merge_from_file(args.cfg_file)
        cfg.merge_from_list(args.opts)
        assert_cfg(cfg)
        # Set Pytorch environment
        torch.set_num_threads(cfg.num_threads)
        out_dir_parent = cfg.out_dir
        cfg.seed = i + 1
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        update_out_dir(out_dir_parent, args.cfg_file)
        dump_cfg(cfg)
        setup_printing()
        auto_select_device()

        # Set learning environment
        datasets = create_dataset()

        cfg.dataset.num_nodes = datasets[0][0].num_nodes
        loaders = create_loader(datasets)
        meters = create_logger(datasets, loaders)

        model = create_model(datasets)
        # breakpoint()
        optimizer = create_optimizer(model.parameters())
        scheduler = create_scheduler(optimizer)
        # Print model info
        logging.info(model)
        logging.info(cfg)
        cfg.params = params_count(model)
        logging.info('Num parameters: {}'.format(cfg.params))
        # Start training
        if cfg.train.mode == 'live_update':
            train_dict[cfg.train.mode](
                meters, loaders, model, optimizer, scheduler, datasets=datasets)

    # When being launched in batch mode, mark a yaml as done
    if args.mark_done:
        os.rename(args.cfg_file, '{}_done'.format(args.cfg_file))
