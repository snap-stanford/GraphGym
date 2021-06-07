"""
A generic loader for the roland project, modify this template to build
loaders for other financial transaction datasets and dynamic graphs.
NOTE: this script is the trimmed version for homogenous graphs only.
Mar. 22, 2021.
# Search for TODO in this file.
"""
import os
from typing import List

import deepsnap
import graphgym.contrib.loader.dynamic_graph_utils as utils
from deepsnap.graph import Graph
from graphgym.config import cfg
from graphgym.register import register_loader


def load_single_dataset(dataset_dir: str) -> Graph:
    # TODO: Load your data here.
    node_feature = None  # (num_nodes, *)
    edge_feature = None  # (num_edges, *)
    edge_index = None  # (2, num_edges)
    # edge time should be unix timestmap integers.
    edge_time = None  # (num_edges)

    graph = Graph(
        node_feature=node_feature,
        edge_feature=edge_feature,
        edge_index=edge_index,
        edge_time=edge_time,
        directed=True
    )

    return graph


def load_generic_dataset(format: str, name: str, dataset_dir: str
                         ) -> List[deepsnap.graph.Graph]:
    """Load the dataset as a list of graph snapshots.

    Args:
        format (str): format of dataset.
        name (str): file name of dataset.
        dataset_dir (str): path of dataset, do NOT include the file name, use
            the parent directory of dataset file.

    Returns:
        List[deepsnap.graph.Graph]: a list of graph snapshots.
    """
    # TODO: change the format name.
    if format == 'generic':
        dataset_dir = os.path.join(dataset_dir, name)
        g_all = load_single_dataset(dataset_dir)
        snapshot_list = utils.make_graph_snapshot(
            g_all,
            snapshot_freq=cfg.transaction.snapshot_freq,
            is_hetero=cfg.dataset.is_hetero)
        return snapshot_list


# TODO: don't forget to register the loader.
register_loader('roland_generic', load_generic_dataset)
