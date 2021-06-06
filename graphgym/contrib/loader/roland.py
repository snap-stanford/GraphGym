"""
A refined version for loading the roland dataset. This version has the
following key points:

(1) Node's features are determined by their first transaction, so that
    payer and payee information are no longer included as a edge features.

    Node features include:
        company identity, bank, country, region, Skd, SkdL1, SkdL2, Skis,
        SkisL1, SkisL2.

(2) edge features include: # system, currency, scaled amount (EUR), and
    scaled timestamp.

Mar. 31, 2021
"""
import os
from typing import List, Union

import dask.dataframe as dd
import deepsnap
import graphgym.contrib.loader.dynamic_graph_utils as utils
import numpy as np
import pandas as pd
import torch
from dask_ml.preprocessing import OrdinalEncoder
from deepsnap.graph import Graph
from graphgym.config import cfg
from graphgym.register import register_loader
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder as SkOrdinalEncoder

# =============================================================================
# Configure and instantiate the loader here.
# =============================================================================
# Required for all graphs.
SRC_NODE: str = 'Payer'
DST_NODE: str = 'Payee'
TIMESTAMP: str = 'Timestamp'
AMOUNT: str = 'AmountEUR'

# Categorical columns are SRC_NODE+var and DST_NODE+var.
# columns: SRC_NODE + NODE_CATE_VARS, DST_NODE + NODE_CATE_VARS, EDGE_CATE_VARS
# will be encoded using ordinal encoder.
# Note that '' corresponds to columns SRC_NODE and DST_NODE.
NODE_CATE_VARS: List[str] = ['', 'Bank', 'Country', 'Region', 'Skd', 'SkdL1',
                             'SkdL2', 'Skis', 'SkisL1', 'SkisL2']
EDGE_CATE_VARS: List[str] = ['# System', 'Currency']

# contents of graph.edge_feature
EDGE_FEATURE_COLS: List[str] = [AMOUNT, 'TimestampScaled']
# contents of graph.node_feature
NODE_FEATURE_LIST: List[str] = ['Bank', 'Country', 'Region', 'SkdL1', 'SkisL1']

# Required for heterogeneous graphs only.
# Node and edge features used to define node and edge type in hete GNN.
NODE_TYPE_DEFN: List[str] = ['Country']
EDGE_TYPE_DEFN: List[str] = ['# System']


# Required for graphs with node features only.

def get_node_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Extract node features from a transaction dataset.
    """
    temp = list()
    for p in [SRC_NODE, DST_NODE]:
        # require ['Payer', 'PayerBank', 'PayerCountry', ...]
        cols = [p] + [p + var for var in NODE_FEATURE_LIST]
        relevant = df[cols].copy()
        # rename to ['Company', 'Bank', 'Country', ...]
        relevant.columns = ['Company'] + NODE_FEATURE_LIST
        temp.append(relevant)
    df_char = pd.concat(temp, axis=0)

    # get company's information based on its first occurrence.
    df_char = df_char.groupby('Company').first()
    return df_char[NODE_FEATURE_LIST]


def construct_additional_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs additional features of the transaction dataset.
    """
    # for p in ('Payer', 'Payee'):
    #     # %% Location of companies.
    #     mask = (df[p + 'Country'] != 'SI')
    #     out_of_country = np.empty(len(df), dtype=object)
    #     out_of_country[mask] = 'OutOfCountry'
    #     out_of_country[~mask] = 'InCountry'
    #     df[p + 'OutOfCountry'] = out_of_country
    #
    # mask = (df['PayerCountry'] != df['PayeeCountry'])
    # missing_mask = np.logical_or(df['PayerCountry'] == 'missing',
    #                              df['PayeeCountry'] == 'missing')
    # cross_country = np.empty(len(df), dtype=object)
    # cross_country[mask] = 'CrossCountry'
    # cross_country[~mask] = 'WithinCountry'
    # cross_country[missing_mask] = 'Missing'
    # df['CrossCountry'] = cross_country
    #
    # amount_level = np.empty(len(df), dtype=object)
    # mask_small = df['AmountEUR'] < 500
    # mask_medium = np.logical_and(df['AmountEUR'] >= 500,
    #                              df['AmountEUR'] < 1000)
    # mask_large = df['AmountEUR'] >= 1000
    # amount_level[mask_small] = '$<500'
    # amount_level[mask_medium] = '500<=$<1k'
    # amount_level[mask_large] = '$>=1k'
    #
    # df['AmountLevel'] = amount_level
    return df


def load_single_dataset(dataset_dir: str, is_hetero: bool = True,
                        type_info_loc: str = 'append'
                        ) -> Graph:
    """
    Loads a single graph object from tsv file.

    Args:
        dataset_dir: the path of tsv file to be loaded.
        is_hetero: whether to load heterogeneous graph.
        type_info_loc: 'append' or 'graph_attribute'.

    Returns:
        graph: a (homogenous) deepsnap graph object.
    """
    # Load dataset using dask for fast parallel loading.
    df_trans = dd.read_csv(dataset_dir, sep='\t', low_memory=False)
    df_trans = df_trans.fillna('missing')
    df_trans = df_trans.compute()
    df_trans = construct_additional_features(df_trans)
    df_trans.reset_index(drop=True, inplace=True)  # necessary for dask.

    # a unique values of node-level categorical variables.
    node_cat_uniques = dict()  # Dict[str, np.ndarray of str]
    for var in NODE_CATE_VARS:  # for each node level categorical variable.
        # get unique values of this categorical variable.
        relevant = df_trans[[SRC_NODE + var, DST_NODE + var]]
        unique_var = pd.unique(relevant.to_numpy().ravel())
        node_cat_uniques[var] = np.sort(unique_var)
        # convert corresponding columns into pandas categorical variables.
        cate_type = pd.api.types.CategoricalDtype(
            categories=node_cat_uniques[var], ordered=True)
        for p in ['Payer', 'Payee']:
            df_trans[p + var] = df_trans[p + var].astype(cate_type)

    # Convert edge level categorical variables.
    for var in EDGE_CATE_VARS:
        unique_var = np.sort(pd.unique(df_trans[[var]].to_numpy().ravel()))
        cate_type = pd.api.types.CategoricalDtype(categories=unique_var,
                                                  ordered=True)
        df_trans[var] = df_trans[var].astype(cate_type)

    # Encoding categorical variables, the dask_ml.OrdinalEncoder only modify
    # and encode columns of categorical dtype.
    enc = OrdinalEncoder()
    df_encoded = enc.fit_transform(df_trans)
    df_encoded.reset_index(drop=True, inplace=True)
    print('Columns encoded to ordinal:')
    print(list(enc.categorical_columns_))

    # Scaling transaction amounts.
    scaler = MinMaxScaler((0, 2))
    df_encoded[AMOUNT] = scaler.fit_transform(
        df_encoded[AMOUNT].values.reshape(-1, 1))

    # Scaling timestamps.
    time_scaler = MinMaxScaler((0, 2))
    df_encoded['TimestampScaled'] = time_scaler.fit_transform(
        df_encoded[TIMESTAMP].values.reshape(-1, 1))

    # Prepare for output.
    edge_feature = torch.Tensor(df_encoded[EDGE_FEATURE_COLS].values)

    print('feature_edge_int_num',
          [int(torch.max(edge_feature[:, i])) + 1
           for i in range(len(EDGE_FEATURE_COLS) - 2)])

    edge_index = torch.Tensor(
        df_encoded[[SRC_NODE, DST_NODE]].values.transpose()).long()  # (2, E)
    num_nodes = torch.max(edge_index) + 1
    assert num_nodes == len(node_cat_uniques[''])

    df_node_info = get_node_feature(df_encoded)
    print(df_node_info.shape)
    node_feature = torch.Tensor(df_node_info.astype(float).values)

    cfg.transaction.feature_node_int_num = [
        int(torch.max(node_feature[:, i])) + 1
        for i in range(len(NODE_FEATURE_LIST))
    ]

    print('feature_node_int_num: ',
          [int(torch.max(node_feature[:, i])) + 1
           for i in range(len(NODE_FEATURE_LIST))])

    edge_time = torch.FloatTensor(df_encoded[TIMESTAMP].values)

    graph = Graph(
        node_feature=node_feature,
        edge_feature=edge_feature,
        edge_index=edge_index,
        edge_time=edge_time,
        directed=True
    )

    if is_hetero:
        # Construct node type signatures. E.g., 'USA--CA' for country + region.
        df_node_info['NodeType'] = df_node_info[NODE_TYPE_DEFN[0]].astype(str)
        for var in NODE_TYPE_DEFN[1:]:
            df_node_info['NodeType'] += ('--' + df_node_info[var].astype(str))

        node_type_enc = SkOrdinalEncoder()
        # The sklearn ordinal encoder transforms numpy array instead.
        node_type_int = node_type_enc.fit_transform(
            df_node_info['NodeType'].values.reshape(-1, 1))
        node_type_int = torch.FloatTensor(node_type_int)

        # Construct edge type signatures.
        df_trans['EdgeType'] = df_trans[EDGE_TYPE_DEFN[0]].astype(str)
        for var in EDGE_TYPE_DEFN[1:]:
            df_trans['EdgeType'] += ('--' + df_trans[var].astype(str))

        edge_type_enc = SkOrdinalEncoder()
        edge_type_int = edge_type_enc.fit_transform(
            df_trans['EdgeType'].values.reshape(-1, 1))
        edge_type_int = torch.FloatTensor(edge_type_int)

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


# def make_graph_snapshot(g_all: Graph,
#                         snapshot_freq: str,
#                         is_hetero: bool = True) -> list:
#     """
#     Constructs a list of graph snapshots (Graph or HeteroGraph) based
#         on g_all and snapshot_freq.
#
#     Args:
#         g_all: the entire homogenous graph.
#         snapshot_freq: snapshot frequency.
#         is_hetero: if make heterogeneous graphs.
#     """
#     t = g_all.edge_time.numpy().astype(np.int64)
#     snapshot_freq = snapshot_freq.upper()
#
#     period_split = pd.DataFrame(
#         {'Timestamp': t,
#          'TransactionTime': pd.to_datetime(t, unit='s')},
#         index=range(len(g_all.edge_time)))
#
#     freq_map = {'D': '%j',  # day of year.
#                 'W': '%W',  # week of year.
#                 'M': '%m'  # month of year.
#                 }
#
#     period_split['Year'] = period_split['TransactionTime'].dt.strftime(
#         '%Y').astype(int)
#
#     period_split['SubYearFlag'] = period_split['TransactionTime'].dt.strftime(
#         freq_map[snapshot_freq]).astype(int)
#
#     period2id = period_split.groupby(['Year', 'SubYearFlag']).indices
#     # e.g., dictionary w/ key = (2021, 3) and val = array(edges).
#
#     periods = sorted(list(period2id.keys()))  # ascending order.
#     # alternatively, sorted(..., key=lambda x: x[0] + x[1]/1000).
#     snapshot_list = list()
#     for p in periods:
#         # unique IDs of edges in this period.
#         period_members = period2id[p]
#
#         g_incr = Graph(
#             node_feature=g_all.node_feature,
#             edge_feature=g_all.edge_feature[period_members, :],
#             edge_index=g_all.edge_index[:, period_members],
#             edge_time=g_all.edge_time[period_members],
#             directed=g_all.directed,
#             list_n_type=g_all.list_n_type if is_hetero else None,
#             list_e_type=g_all.list_e_type if is_hetero else None,
#         )
#         if is_hetero and hasattr(g_all, 'node_type'):
#             g_incr.node_type = g_all.node_type
#             g_incr.edge_type = g_all.edge_type[period_members]
#         snapshot_list.append(g_incr)
#     return snapshot_list


def load_generic(dataset_dir: str,
                 snapshot: bool = True,
                 snapshot_freq: str = None,
                 is_hetero: bool = False,
                 type_info_loc: str = 'graph_attribute'
                 ) -> Union[deepsnap.graph.Graph, List[deepsnap.graph.Graph]]:
    g_all = load_single_dataset(dataset_dir, is_hetero=is_hetero,
                                type_info_loc=type_info_loc)
    if not snapshot:
        return g_all
    else:
        snapshot_list = utils.make_graph_snapshot(g_all, snapshot_freq, is_hetero)
        num_nodes = g_all.edge_index.max() + 1

        for g_snapshot in snapshot_list:
            g_snapshot.node_states = [0 for _ in range(cfg.gnn.layers_mp)]
            g_snapshot.node_cells = [0 for _ in range(cfg.gnn.layers_mp)]
            g_snapshot.node_degree_existing = torch.zeros(num_nodes)

        return snapshot_list


def load_generic_dataset(format, name, dataset_dir):
    if format == 'roland_bsi_general':
        dataset_dir = os.path.join(dataset_dir, name)
        graphs = load_generic(dataset_dir,
                              snapshot=cfg.transaction.snapshot,
                              snapshot_freq=cfg.transaction.snapshot_freq,
                              is_hetero=cfg.dataset.is_hetero,
                              type_info_loc=cfg.dataset.type_info_loc)
        return graphs


# TODO: change name.
register_loader('roland_bsi_v3', load_generic_dataset)
