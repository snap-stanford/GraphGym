import logging
import os

import torch
from torch_geometric import seed_everything
from deepsnap.dataset import GraphDataset
from graphgym.cmd_args import parse_args
from graphgym.config import cfg, dump_cfg, load_cfg, set_run_dir, set_out_dir
from graphgym.loader import create_dataset, create_loader
from graphgym.logger import create_logger, setup_printing
from graphgym.model_builder import create_model
from graphgym.optimizer import create_optimizer, create_scheduler
from graphgym.register import train_dict
from graphgym.train import train
from graphgym.utils.agg_runs import agg_runs
from graphgym.utils.comp_budget import params_count
from graphgym.utils.device import auto_select_device
from graphgym.models.gnn import GNNStackStage
from CytokinesDataSet import CytokinesDataSet
from graphgym.models.layer import GeneralMultiLayer, Linear, GeneralConv
from graphgym.models.gnn import GNNStackStage
import numpy as np


if __name__ == '__main__':
    # Load cmd line args
    args = parse_args()
    # Load config file
    load_cfg(cfg, args)
    set_out_dir(cfg.out_dir, args.cfg_file)
    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    dump_cfg(cfg)
    print("wonder")
    # Repeat for different random seeds
    for i in range(args.repeat):
        set_run_dir(cfg.out_dir)
        setup_printing()
        # Set configurations for each run
        cfg.seed = cfg.seed + 1
        seed_everything(cfg.seed)
        auto_select_device()
        # Set machine learning pipeline
        datasets = create_dataset()
        loaders = create_loader(datasets)
        loggers = create_logger()
        model = create_model()

        total = 0
        

        # Add edge_weights attribute to the datasets so that they can be accessed in batches
        num_edges = len(datasets[0][0].edge_index[0])
        edge_weights = torch.nn.Parameter(torch.ones(num_edges))
        for loader in loaders:
            for dataset in loader.dataset:
                dataset.edge_weights = edge_weights


        #add edge weights to the set of parameters
        newParam = list()
        for param in model.parameters():
            newParam.append(param)
        
        newParam.append(edge_weights)

        optimizer = create_optimizer(newParam)
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
        
        agg_runs(cfg.out_dir, cfg.metric_best)
        # When being launched in batch mode, mark a yaml as done
        if args.mark_done:
            os.rename(args.cfg_file, f'{args.cfg_file}_done')


