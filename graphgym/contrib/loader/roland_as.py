"""
Loader for the Autonomous systems AS-733 dataset.
"""
import os
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
import torch
from deepsnap.graph import Graph
from graphgym.config import cfg
from graphgym.register import register_loader
from sklearn.preprocessing import OrdinalEncoder
from tqdm import tqdm


def make_graph_snapshot(g_all: Graph, snapshot_freq: str) -> List[Graph]:
    t = g_all.edge_time.numpy().astype(np.int64)
    snapshot_freq = snapshot_freq.upper()

    period_split = pd.DataFrame(
        {'Timestamp': t,
         'TransactionTime': pd.to_datetime(t, unit='s')},
        index=range(len(g_all.edge_time)))

    freq_map = {'D': '%j',  # day of year.
                'W': '%W',  # week of year.
                'M': '%m'  # month of year.
                }

    period_split['Year'] = period_split['TransactionTime'].dt.strftime(
        '%Y').astype(int)

    period_split['SubYearFlag'] = period_split['TransactionTime'].dt.strftime(
        freq_map[snapshot_freq]).astype(int)

    period2id = period_split.groupby(['Year', 'SubYearFlag']).indices

    periods = sorted(list(period2id.keys()))
    snapshot_list = list()

    for p in periods:
        # unique IDs of edges in this period.
        period_members = period2id[p]
        assert np.all(period_members == np.unique(period_members))

        g_incr = Graph(
            node_feature=g_all.node_feature,
            edge_feature=g_all.edge_feature[period_members, :],
            edge_index=g_all.edge_index[:, period_members],
            edge_time=g_all.edge_time[period_members],
            directed=g_all.directed
        )
        snapshot_list.append(g_incr)

    snapshot_list.sort(key=lambda x: torch.min(x.edge_time))

    return snapshot_list


def file2timestamp(file_name):
    t = file_name.strip('.txt').strip('as')
    ts = int(datetime.strptime(t, '%Y%m%d').timestamp())
    return ts


def load_generic_dataset(format, name, dataset_dir):
    if format == 'as':
        all_files = [x for x in sorted(os.listdir(dataset_dir))
                     if (x.startswith('as') and x.endswith('.txt'))]
        assert len(all_files) == 733
        assert all(x.endswith('.txt') for x in all_files)

        edge_index_lst, edge_time_lst = list(), list()
        all_files = sorted(all_files)
        # if cfg.train.mode in ['baseline', 'baseline_v2', 'live_update_fixed_split']:
        #     # The baseline setting in EvolveGCN paper only uses 100 snapshots.
        #     all_files = all_files[:100]
        for graph_file in tqdm(all_files):
            today = file2timestamp(graph_file)
            graph_file = os.path.join(dataset_dir, graph_file)

            src, dst = list(), list()
            with open(graph_file, 'r') as f:
                for line in f.readlines():
                    if line.startswith('#'):
                        continue
                    line = line.strip('\n')
                    v1, v2 = line.split('\t')
                    src.append(int(v1))
                    dst.append(int(v2))

            edge_index = np.stack((src, dst))
            edge_index_lst.append(edge_index)

            edge_time = np.ones(edge_index.shape[1]) * today
            edge_time_lst.append(edge_time)

        edge_index_raw = np.concatenate(edge_index_lst, axis=1).astype(int)

        num_nodes = len(np.unique(edge_index_raw))

        # encode node indices to consecutive integers.
        node_indices = np.sort(np.unique(edge_index_raw))
        enc = OrdinalEncoder(categories=[node_indices, node_indices])
        edge_index = enc.fit_transform(edge_index_raw.transpose()).transpose()
        edge_index = torch.Tensor(edge_index).long()
        edge_time = torch.Tensor(np.concatenate(edge_time_lst))

        # Use scaled datetime as edge_feature.
        scale = edge_time.max() - edge_time.min()
        base = edge_time.min()
        scaled_edge_time = 2 * (edge_time.clone() - base) / scale
        
        assert cfg.dataset.AS_node_feature in ['one', 'one_hot_id',
                                               'one_hot_degree_global',
                                               'one_hot_degree_local']

        if cfg.dataset.AS_node_feature == 'one':
            node_feature = torch.ones(num_nodes, 1)
        elif cfg.dataset.AS_node_feature == 'one_hot_id':
            # One hot encoding the node ID.
            node_feature = torch.Tensor(np.eye(num_nodes))
        elif cfg.dataset.AS_node_feature == 'one_hot_degree_global':
            # undirected graph, use only out degree.
            _, node_degree = torch.unique(edge_index[0], sorted=True,
                                          return_counts=True)
            node_feature = np.zeros((num_nodes, node_degree.max() + 1))
            node_feature[np.arange(num_nodes), node_degree] = 1
            # 1 ~ 63748 degrees, but only 710 possible levels, exclude all zero
            # columns.
            non_zero_cols = (node_feature.sum(axis=0) > 0)
            node_feature = node_feature[:, non_zero_cols]
            node_feature = torch.Tensor(node_feature)
        else:
            raise NotImplementedError

        g_all = Graph(
            node_feature=node_feature,
            edge_feature=scaled_edge_time.reshape(-1, 1),
            edge_index=edge_index,
            edge_time=edge_time,
            directed=True
        )

        snapshot_list = make_graph_snapshot(g_all,
                                            cfg.transaction.snapshot_freq)

        for g_snapshot in snapshot_list:
            g_snapshot.node_states = [0 for _ in range(cfg.gnn.layers_mp)]
            g_snapshot.node_cells = [0 for _ in range(cfg.gnn.layers_mp)]
            g_snapshot.node_degree_existing = torch.zeros(num_nodes)

        if cfg.dataset.split_method == 'chronological_temporal':
            return snapshot_list
        else:
            # The default split (80-10-10) requires at least 10 edges each
            # snapshot.
            filtered_graphs = list()
            for g in tqdm(snapshot_list):
                if g.num_edges >= 10:
                    filtered_graphs.append(g)
            return filtered_graphs


register_loader('roland_as', load_generic_dataset)
