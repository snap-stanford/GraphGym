import networkx as nx
import time
import logging
import pickle

from typing import List

import deepsnap
from deepsnap.dataset import GraphDataset
import torch
from torch.utils.data import DataLoader

from torch_geometric.datasets import *
import torch_geometric.transforms as T

from graphgym.config import cfg
import graphgym.models.feature_augment as preprocess
from graphgym.models.transform import (ego_nets, remove_node_feature,
                                       edge_nets, path_len)
from graphgym.contrib.loader import *
import graphgym.register as register

from ogb.graphproppred import PygGraphPropPredDataset
from deepsnap.batch import Batch


def load_pyg(name: str, dataset_dir: str) -> List["Graph"]:
    r"""Loads pytorch-geometric (pyg) format dataset

    Args:
        name: the dataset name.
        dataset_dir: the data directory.

    Returns:
        A list of networkx/deepsnap graphs
    """
    dataset_dir = '{}/{}'.format(dataset_dir, name)
    if name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset_raw = Planetoid(dataset_dir, name)
    elif name[:3] == 'TU_':
        # TU_IMDB doesn't have node features
        if name[3:] == 'IMDB':
            name = 'IMDB-MULTI'
            dataset_raw = TUDataset(dataset_dir, name,
                                    transform=T.Constant())
        else:
            dataset_raw = TUDataset(dataset_dir, name[3:])
        # TU_dataset only has graph-level label
        # The goal is to have synthetic tasks
        # that select smallest 100 graphs that have more than 200 edges
        if cfg.dataset.tu_simple and cfg.dataset.task != 'graph':
            size = []
            for data in dataset_raw:
                edge_num = data.edge_index.shape[1]
                edge_num = 9999 if edge_num < 200 else edge_num
                size.append(edge_num)
            size = torch.tensor(size)
            order = torch.argsort(size)[:100]
            dataset_raw = dataset_raw[order]
    elif name == 'Karate':
        dataset_raw = KarateClub()
    elif 'Coauthor' in name:
        if 'CS' in name:
            dataset_raw = Coauthor(dataset_dir, name='CS')
        else:
            dataset_raw = Coauthor(dataset_dir, name='Physics')
    elif 'Amazon' in name:
        if 'Computers' in name:
            dataset_raw = Amazon(dataset_dir, name='Computers')
        else:
            dataset_raw = Amazon(dataset_dir, name='Photo')
    elif name == 'MNIST':
        dataset_raw = MNISTSuperpixels(dataset_dir)
    elif name == 'PPI':
        dataset_raw = PPI(dataset_dir)
    elif name == 'QM7b':
        dataset_raw = QM7b(dataset_dir)
    else:
        raise ValueError('{} not support'.format(name))
    graphs = GraphDataset.pyg_to_graphs(dataset_raw)
    return graphs


def load_nx(name: str, dataset_dir: str) -> List["Graph"]:
    r"""Loads networkx format dataset.

    Args:
        name: the dataset name.
        dataset_dir: the data directory.
    Returns:
        A list of networkx/deepsnap graphs.
    """
    try:
        with open('{}/{}.pkl'.format(dataset_dir, name), 'rb') as file:
            graphs = pickle.load(file)
    except:
        graphs = nx.read_gpickle('{}.gpickle'.format(dataset_dir, name))
        if not isinstance(graphs, list):
            graphs = [graphs]
    return graphs


def load_dataset() -> List["Graph"]:
    r"""Loads raw datasets.

    Returns:
        A list of networkx/deepsnap graphs, plus additional info if needed.
    """
    format = cfg.dataset.format
    name = cfg.dataset.name
    # dataset_dir = '{}/{}'.format(cfg.dataset.dir, name)
    dataset_dir = cfg.dataset.dir
    # Try to load customized data format
    for func in register.loader_dict.values():
        graphs = func(format, name, dataset_dir)
        if graphs is not None:
            return graphs
    # Load from Pytorch Geometric dataset
    if format == 'PyG':
        graphs = load_pyg(name, dataset_dir)
    # Load from networkx formatted data
    # todo: clean nx dataloader
    elif format == 'nx':
        graphs = load_nx(name, dataset_dir)
    # Load from OGB formatted data
    elif cfg.dataset.format == 'OGB':
        if cfg.dataset.name == 'ogbg-molhiv':
            dataset = PygGraphPropPredDataset(name=cfg.dataset.name)
            graphs = GraphDataset.pyg_to_graphs(dataset)
        # Note this is only used for custom splits from OGB
        split_idx = dataset.get_idx_split()
        return graphs, split_idx
    else:
        raise ValueError('Unknown data format: {}'.format(cfg.dataset.format))
    return graphs


def filter_graphs() -> int:
    r""" Filters graphs by the min number of nodes.

    Returns:
        min number of nodes.
    """
    if cfg.dataset.task == 'graph':
        min_node = 0
    else:
        min_node = 5
    return min_node


def transform_before_split(dataset: deepsnap.dataset.GraphDataset) -> deepsnap.dataset.GraphDataset:
    r"""Transforms the whole dataset before train/val/test split.

    Args:
        dataset: A DeepSNAP dataset object

    Returns:
        A transformed DeepSNAP dataset object
    """
    if cfg.dataset.remove_feature:
        dataset.apply_transform(remove_node_feature,
                                update_graph=True, update_tensor=False)
    augmentation = preprocess.FeatureAugment()
    actual_feat_dims, actual_label_dim = augmentation.augment(dataset)
    if cfg.dataset.augment_label:
        dataset.apply_transform(preprocess._replace_label,
                                update_graph=True, update_tensor=False)
    # Update augmented feature/label dims by real dims (user specified dims
    # may not be realized)
    cfg.dataset.augment_feature_dims = actual_feat_dims
    if cfg.dataset.augment_label:
        cfg.dataset.augment_label_dims = actual_label_dim

    # Temporary for ID-GNN path prediction task
    if cfg.dataset.task == 'edge' and 'id' in cfg.gnn.layer_type:
        dataset.apply_transform(path_len, update_graph=False,
                                update_tensor=False)

    return dataset


def transform_after_split(datasets: List[deepsnap.dataset.GraphDataset]) -> List[deepsnap.dataset.GraphDataset]:
    r"""Transform datasets after train/val/test split

    Args:
        datasets: A list of DeepSNAP dataset objects

    Returns:
        A list of transformed DeepSNAP dataset objects
    """
    if cfg.dataset.transform == 'ego':
        for split_dataset in datasets:
            split_dataset.apply_transform(ego_nets,
                                          radius=cfg.gnn.layers_mp,
                                          update_tensor=True,
                                          update_graph=False)
    elif cfg.dataset.transform == 'edge':
        for split_dataset in datasets:
            split_dataset.apply_transform(edge_nets,
                                          radius=cfg.gnn.layers_mp,
                                          update_tensor=True,
                                          update_graph=False)
            split_dataset.task = 'node'
        cfg.dataset.task = 'node'
    return datasets


def create_dataset() -> List[deepsnap.dataset.GraphDataset]:
    r"""Loads the dataset from configuration file and splits datasets into train, validation, (and test) sets.

    Returns:
        A list of 2 (3) deepsnap.dataset.GraphDataset objects corresponding to train, validation (and test) sets.
    """
    # Load dataset
    time1 = time.time()
    if cfg.dataset.format == 'OGB':
        graphs, splits = load_dataset()
    else:
        graphs = load_dataset()

    # Filter graphs
    time2 = time.time()
    min_node = filter_graphs()

    # Create whole dataset
    dataset = GraphDataset(
        graphs,
        task=cfg.dataset.task,
        edge_train_mode=cfg.dataset.edge_train_mode,
        edge_message_ratio=cfg.dataset.edge_message_ratio,
        edge_negative_sampling_ratio=cfg.dataset.edge_negative_sampling_ratio,
        minimum_node_per_graph=min_node)

    # Transform the whole dataset
    dataset = transform_before_split(dataset)

    # Split dataset
    time3 = time.time()
    # Use custom data splits
    if cfg.dataset.format == 'OGB':
        datasets = []
        datasets.append(dataset[splits['train']])
        datasets.append(dataset[splits['valid']])
        datasets.append(dataset[splits['test']])
    # Use random split, supported by DeepSNAP
    else:
        datasets = dataset.split(
            transductive=cfg.dataset.transductive,
            split_ratio=cfg.dataset.split)

    # Transform each split dataset
    time4 = time.time()
    datasets = transform_after_split(datasets)

    time5 = time.time()
    logging.info('Load: {:.4}s, Before split: {:.4}s, '
                 'Split: {:.4}s, After split: {:.4}s'.format(
        time2 - time1, time3 - time2, time4 - time3, time5 - time4))

    return datasets


def create_loader(
    datasets: List[deepsnap.dataset.GraphDataset]
) -> List[torch.utils.data.DataLoader]:
    r"""Converts the list of graphs into pytorch data loaders.

    Args:
        datasets: A list of 3 lists of deepsnap.graph.Graph objects corresponding to train, validation, and test sets.

    Returns:
        A list of 2 (3) lists of torch.utils.data.DataLoader objects corresponding to
            train, validation, (and test) loaders.
    """
    loader_train = DataLoader(datasets[0], collate_fn=Batch.collate(),
                              batch_size=cfg.train.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=False)

    loaders = [loader_train]
    for i in range(1, len(datasets)):
        loaders.append(DataLoader(datasets[i], collate_fn=Batch.collate(),
                                  batch_size=cfg.train.batch_size,
                                  shuffle=False,
                                  num_workers=cfg.num_workers,
                                  pin_memory=False))

    return loaders
