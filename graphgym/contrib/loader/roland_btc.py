"""
Data loader for bitcoin datasets.
Mar. 27, 2021
"""
import os
from typing import List, Union

import deepsnap
import graphgym.contrib.loader.dynamic_graph_utils as utils
import numpy as np
import pandas as pd
import torch
from deepsnap.graph import Graph
from graphgym.config import cfg
from graphgym.register import register_loader
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder


def load_single_dataset(dataset_dir: str) -> Graph:
    df_trans = pd.read_csv(dataset_dir, sep=',', header=None, index_col=None)
    df_trans.columns = ['SOURCE', 'TARGET', 'RATING', 'TIME']
    # NOTE: 'SOURCE' and 'TARGET' are not consecutive.
    num_nodes = len(
        pd.unique(df_trans[['SOURCE', 'TARGET']].to_numpy().ravel()))

    # bitcoin OTC contains decimal numbers, round them.
    df_trans['TIME'] = df_trans['TIME'].astype(np.int).astype(np.float)
    assert not np.any(pd.isna(df_trans).values)

    time_scaler = MinMaxScaler((0, 2))
    df_trans['TimestampScaled'] = time_scaler.fit_transform(
        df_trans['TIME'].values.reshape(-1, 1))

    edge_feature = torch.Tensor(
        df_trans[['RATING', 'TimestampScaled']].values)  # (E, edge_dim)
    # SOURCE and TARGET IDs are already encoded in the csv file.
    # edge_index = torch.Tensor(
    #     df_trans[['SOURCE', 'TARGET']].values.transpose()).long()  # (2, E)

    node_indices = np.sort(
        pd.unique(df_trans[['SOURCE', 'TARGET']].to_numpy().ravel()))
    enc = OrdinalEncoder(categories=[node_indices, node_indices])
    raw_edges = df_trans[['SOURCE', 'TARGET']].values
    edge_index = enc.fit_transform(raw_edges).transpose()
    edge_index = torch.LongTensor(edge_index)

    # num_nodes = torch.max(edge_index) + 1
    # Use dummy node features.
    node_feature = torch.ones(num_nodes, 1).float()

    edge_time = torch.FloatTensor(df_trans['TIME'].values)

    # TODO: add option here.
    # if cfg.train.mode in ['baseline', 'baseline_v2', 'live_update_fixed_split']:
    #     edge_feature = torch.cat((edge_feature, edge_feature.clone()), dim=0)
    #     reversed_idx = torch.stack([edge_index[1], edge_index[0]]).clone()
    #     edge_index = torch.cat((edge_index, reversed_idx), dim=1)
    #     edge_time = torch.cat((edge_time, edge_time.clone()))

    graph = Graph(
        node_feature=node_feature,
        edge_feature=edge_feature,
        edge_index=edge_index,
        edge_time=edge_time,
        directed=True
    )
    return graph


# def make_graph_snapshot(g_all: Graph, snapshot_freq: str) -> List[Graph]:
#     t = g_all.edge_time.numpy().astype(np.int64)
#     snapshot_freq = snapshot_freq.upper()

#     period_split = pd.DataFrame(
#         {'Timestamp': t,
#          'TransactionTime': pd.to_datetime(t, unit='s')},
#         index=range(len(g_all.edge_time)))

#     freq_map = {'D': '%j',  # day of year.
#                 'W': '%W',  # week of year.
#                 'M': '%m'  # month of year.
#                 }

#     period_split['Year'] = period_split['TransactionTime'].dt.strftime(
#         '%Y').astype(int)

#     period_split['SubYearFlag'] = period_split['TransactionTime'].dt.strftime(
#         freq_map[snapshot_freq]).astype(int)

#     period2id = period_split.groupby(['Year', 'SubYearFlag']).indices

#     periods = sorted(list(period2id.keys()))
#     snapshot_list = list()

#     for p in periods:
#         # unique IDs of edges in this period.
#         period_members = period2id[p]
#         assert np.all(period_members == np.unique(period_members))

#         g_incr = Graph(
#             node_feature=g_all.node_feature,
#             edge_feature=g_all.edge_feature[period_members, :],
#             edge_index=g_all.edge_index[:, period_members],
#             edge_time=g_all.edge_time[period_members],
#             directed=g_all.directed
#         )
#         snapshot_list.append(g_incr)

#     snapshot_list.sort(key=lambda x: torch.min(x.edge_time))

#     return snapshot_list


# def split_by_seconds(g_all, freq_sec: int):
#     # Split the entire graph into snapshots.
#     split_criterion = g_all.edge_time // freq_sec
#     groups = torch.sort(torch.unique(split_criterion))[0]
#     snapshot_list = list()
#     for t in groups:
#         period_members = (split_criterion == t)
#         g_incr = Graph(
#             node_feature=g_all.node_feature,
#             edge_feature=g_all.edge_feature[period_members, :],
#             edge_index=g_all.edge_index[:, period_members],
#             edge_time=g_all.edge_time[period_members],
#             directed=g_all.directed
#         )
#         snapshot_list.append(g_incr)
#     return snapshot_list

# TODO: merge these two method.
def load_snapshots(dataset_dir: str,
                   snapshot: bool = True,
                   snapshot_freq: str = None
                   ) -> Union[deepsnap.graph.Graph,
                              List[deepsnap.graph.Graph]]:
    g_all = load_single_dataset(dataset_dir)
    if not snapshot:
        return g_all

    if snapshot_freq.upper() not in ['D', 'W', 'M']:
        # format: '1200000s'
        # assume split by seconds (timestamp) as in EvolveGCN paper.
        freq = int(snapshot_freq.strip('s'))
        snapshot_list = utils.make_graph_snapshot_by_seconds(g_all, freq)
    else:
        snapshot_list = utils.make_graph_snapshot(g_all, snapshot_freq)
    num_nodes = g_all.edge_index.max() + 1

    for g_snapshot in snapshot_list:
        g_snapshot.node_states = [0 for _ in range(cfg.gnn.layers_mp)]
        g_snapshot.node_cells = [0 for _ in range(cfg.gnn.layers_mp)]
        g_snapshot.node_degree_existing = torch.zeros(num_nodes)

    # check snapshots ordering.
    prev_end = -1
    for g in snapshot_list:
        start, end = torch.min(g.edge_time), torch.max(g.edge_time)
        assert prev_end < start <= end
        prev_end = end

    return snapshot_list


def load_btc_dataset(format: str, name: str, dataset_dir: str):
    if format == 'bitcoin':
        graphs = load_snapshots(os.path.join(dataset_dir, name),
                                snapshot=cfg.transaction.snapshot,
                                snapshot_freq=cfg.transaction.snapshot_freq)
        if cfg.dataset.split_method == 'chronological_temporal':
            return graphs
        else:
            # The default split (80-10-10) requires at least 10 edges each
            # snapshot.
            filtered_graphs = list()
            for g in graphs:
                if g.num_edges >= 10:
                    filtered_graphs.append(g)
            return filtered_graphs


register_loader('roland_btc', load_btc_dataset)
