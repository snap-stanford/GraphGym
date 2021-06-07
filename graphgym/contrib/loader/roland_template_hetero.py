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
import torch
from deepsnap.graph import Graph
from graphgym.config import cfg
from graphgym.register import register_loader


def load_single_hetero_dataset(dataset_dir: str, type_info_loc: str) -> Graph:
    # TODO: Load your data from dataset_dir here.
    # Example:
    num_nodes = 500
    num_node_feature = 16
    num_edges = 10000
    num_edge_feature = 32
    node_feature = torch.rand((num_nodes, num_node_feature))
    edge_feature = torch.rand((num_edges, num_edge_feature))
    edge_index = torch.randint(0, num_nodes - 1, (2, num_edges))
    # edge time should be unix timestmap integers.
    # random generate timestamps from 2021-05-01 to 2021-06-01
    edge_time = torch.randint(1619852450, 1622530850, (num_edges,)).sort()[0]

    graph = Graph(
        node_feature=node_feature,
        edge_feature=edge_feature,
        edge_index=edge_index,
        edge_time=edge_time,
        directed=True
    )

    # TODO: additional operations required for heterogeneous graphs.
    # Assume there are 3 types of edges.
    num_edge_types = 3
    edge_type_int = torch.randint(0, num_edge_types - 1, (num_edges,)).float()
    # Assume there are 5 types of nodes.
    num_node_types = 5
    node_type_int = torch.randint(0, num_node_types - 1, (num_nodes,)).float()

    if type_info_loc == 'append':
        graph.edge_feature = torch.cat((graph.edge_feature, edge_type_int),
                                       dim=1)
        graph.node_feature = torch.cat((graph.node_feature, node_type_int),
                                       dim=1)
    elif type_info_loc == 'graph_attribute':
        graph.node_type = node_type_int.reshape(-1, )
        graph.edge_type = edge_type_int.reshape(-1, )
    else:
        raise ValueError(f'Unsupported type info loc: {type_info_loc}')

    # add a list of unique types for reference.
    graph.list_n_type = node_type_int.unique().long()
    graph.list_e_type = edge_type_int.unique().long()

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
    if format == 'YOUR_HETERO_FORMAT_NAME_HERE':
        assert cfg.dataset.is_hetero
        dataset_dir = os.path.join(dataset_dir, name)
        g_all = load_single_hetero_dataset(
            dataset_dir,
            type_info_loc=cfg.dataset.type_info_loc)
        snapshot_list = utils.make_graph_snapshot(
            g_all,
            snapshot_freq=cfg.transaction.snapshot_freq,
            is_hetero=cfg.dataset.is_hetero)
        return snapshot_list


# TODO: don't forget to register the loader.
register_loader('YOUR_HETERO_LOADER_NAME_HERE', load_generic_dataset)
