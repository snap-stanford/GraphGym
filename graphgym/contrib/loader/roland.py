"""
One single loader for the roland project.
"""
import os
from datetime import datetime
from typing import List

import dask.dataframe as dd
import graphgym.contrib.loader.dynamic_graph_utils as utils
import numpy as np
import pandas as pd
import torch
from dask_ml.preprocessing import OrdinalEncoder as DaskOrdinalEncoder
from deepsnap.graph import Graph
from graphgym.config import cfg
from graphgym.register import register_loader
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder as SkOrdinalEncoder
from tqdm import tqdm

# =============================================================================
# AS-733 Dataset.
# =============================================================================


def load_AS_dataset(dataset_dir: str) -> Graph:
    all_files = [x for x in sorted(os.listdir(dataset_dir))
                 if (x.startswith('as') and x.endswith('.txt'))]
    assert len(all_files) == 733
    assert all(x.endswith('.txt') for x in all_files)

    def file2timestamp(file_name: str) -> int:
        t = file_name.strip('.txt').strip('as')
        ts = int(datetime.strptime(t, '%Y%m%d').timestamp())
        return ts

    edge_index_lst, edge_time_lst = list(), list()
    all_files = sorted(all_files)

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
    enc = SkOrdinalEncoder(categories=[node_indices, node_indices])
    edge_index = enc.fit_transform(edge_index_raw.transpose()).transpose()
    edge_index = torch.Tensor(edge_index).long()
    edge_time = torch.Tensor(np.concatenate(edge_time_lst))

    # Use scaled datetime as edge_feature.
    scale = edge_time.max() - edge_time.min()
    base = edge_time.min()
    scaled_edge_time = 2 * (edge_time.clone() - base) / scale

    assert cfg.dataset.AS_node_feature in ['one', 'one_hot_id',
                                           'one_hot_degree_global']

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

    return g_all


# =============================================================================
# BSI-SVT Dataset
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
    for p in ('Payer', 'Payee'):
        # %% Location of companies.
        mask = (df[p + 'Country'] != 'SI')
        out_of_country = np.empty(len(df), dtype=object)
        out_of_country[mask] = 'OutOfCountry'
        out_of_country[~mask] = 'InCountry'
        df[p + 'OutOfCountry'] = out_of_country

    mask = (df['PayerCountry'] != df['PayeeCountry'])
    missing_mask = np.logical_or(df['PayerCountry'] == 'missing',
                                 df['PayeeCountry'] == 'missing')
    cross_country = np.empty(len(df), dtype=object)
    cross_country[mask] = 'CrossCountry'
    cross_country[~mask] = 'WithinCountry'
    cross_country[missing_mask] = 'Missing'
    df['CrossCountry'] = cross_country

    amount_level = np.empty(len(df), dtype=object)
    mask_small = df['AmountEUR'] < 500
    mask_medium = np.logical_and(df['AmountEUR'] >= 500,
                                 df['AmountEUR'] < 1000)
    mask_large = df['AmountEUR'] >= 1000
    amount_level[mask_small] = '$<500'
    amount_level[mask_medium] = '500<=$<1k'
    amount_level[mask_large] = '$>=1k'

    df['AmountLevel'] = amount_level
    return df


def load_bsi_dataset(dataset_dir: str, is_hetero: bool = False) -> Graph:
    """
    Loads a single graph object from tsv file.

    Args:
        dataset_dir: the path of tsv file to be loaded.
        is_hetero: whether to load heterogeneous graph.

    Returns:
        graph: a (homogenous) deepsnap graph object.
    """
    # Load dataset using dask for fast parallel loading.
    df_trans = dd.read_csv(dataset_dir, sep='\t', low_memory=False)
    df_trans = df_trans.fillna('missing')
    df_trans = df_trans.compute()
    if is_hetero:
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
    enc = DaskOrdinalEncoder()
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

    feature_edge_int_num = [int(torch.max(edge_feature[:, i])) + 1
                            for i in range(len(EDGE_FEATURE_COLS) - 2)]
    cfg.transaction.feature_edge_int_num = feature_edge_int_num
    print('feature_edge_int_num', feature_edge_int_num)

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

        graph.node_type = node_type_int.reshape(-1,)
        graph.edge_type = edge_type_int.reshape(-1,)

        # add a list of unique types for reference.
        graph.list_n_type = node_type_int.unique().long()
        graph.list_e_type = edge_type_int.unique().long()

    return graph

# =============================================================================
# Bitcoin Dataset.
# =============================================================================


def load_bitcoin_dataset(dataset_dir: str) -> Graph:
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

    node_indices = np.sort(
        pd.unique(df_trans[['SOURCE', 'TARGET']].to_numpy().ravel()))
    enc = SkOrdinalEncoder(categories=[node_indices, node_indices])
    raw_edges = df_trans[['SOURCE', 'TARGET']].values
    edge_index = enc.fit_transform(raw_edges).transpose()
    edge_index = torch.LongTensor(edge_index)

    # num_nodes = torch.max(edge_index) + 1
    # Use dummy node features.
    node_feature = torch.ones(num_nodes, 1).float()

    edge_time = torch.FloatTensor(df_trans['TIME'].values)

    graph = Graph(
        node_feature=node_feature,
        edge_feature=edge_feature,
        edge_index=edge_index,
        edge_time=edge_time,
        directed=True
    )
    return graph


# =============================================================================
# Reddit Dataset.
# =============================================================================


def load_reddit_dataset(dataset_dir: str) -> Graph:
    df_trans = dd.read_csv(dataset_dir, sep='\t', low_memory=False)
    df_trans = df_trans.compute()
    assert not np.any(pd.isna(df_trans).values)
    df_trans.reset_index(drop=True, inplace=True)  # required for dask.

    # Encode src and dst node IDs.
    # get unique values of src and dst.
    unique_subreddits = pd.unique(
        df_trans[['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT']].to_numpy().ravel())
    unique_subreddits = np.sort(unique_subreddits)
    cate_type = pd.api.types.CategoricalDtype(categories=unique_subreddits,
                                              ordered=True)
    df_trans['SOURCE_SUBREDDIT'] = df_trans['SOURCE_SUBREDDIT'].astype(
        cate_type)
    df_trans['TARGET_SUBREDDIT'] = df_trans['TARGET_SUBREDDIT'].astype(
        cate_type)
    enc = DaskOrdinalEncoder(columns=['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT'])
    df_encoded = enc.fit_transform(df_trans)
    df_encoded.reset_index(drop=True, inplace=True)

    # Add node feature from the embedding dataset.
    node_embedding_dir = os.path.join(cfg.dataset.dir,
                                      'web-redditEmbeddings-subreddits.csv')

    # index: subreddit name, values: embedding.
    df_node = pd.read_csv(node_embedding_dir, header=None, index_col=0)

    # ordinal encoding follows order in unique_subreddits.
    # df_encoded['SOURCE_SUBREDDIT'] contains encoded integral values.
    # unique_subreddits[df_encoded['SOURCE_SUBREDDIT']]
    # tries to reverse encoded_integer --> original subreddit name.
    # check if recovered sub-reddit name matched the raw data.
    for col in ['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT']:
        assert all(unique_subreddits[df_encoded[col]] == df_trans[col])

    num_nodes = len(cate_type.categories)
    node_feature = torch.ones(size=(num_nodes, 300))
    # for nodes without precomputed embedding, use the average value.
    node_feature = node_feature * np.mean(df_node.values)

    # cate_type.categories[i] is encoded to i, by construction.
    for i, subreddit in enumerate(cate_type.categories):
        if subreddit in df_node.index:
            embedding = df_node.loc[subreddit]
            node_feature[i, :] = torch.Tensor(embedding.values)

    # Original format: df['TIMESTAMP'][0] = '2013-12-31 16:39:18'
    # Convert to unix timestamp (integers).
    df_encoded['TIMESTAMP'] = pd.to_datetime(df_encoded['TIMESTAMP'],
                                             format='%Y-%m-%d %H:%M:%S')
    df_encoded['TIMESTAMP'] = (df_encoded['TIMESTAMP'] - pd.Timestamp(
        '1970-01-01')) // pd.Timedelta('1s')  # now integers.

    # Scale edge time.
    time_scaler = MinMaxScaler((0, 2))
    df_encoded['TimestampScaled'] = time_scaler.fit_transform(
        df_encoded['TIMESTAMP'].values.reshape(-1, 1))

    # Link sentimental representation (86-dimension).
    # comma-separated string: '3.1,5.1,0.0,...'
    senti_str_lst = df_encoded['PROPERTIES'].values
    edge_senti_embedding = [x.split(',') for x in senti_str_lst]
    edge_senti_embedding = np.array(edge_senti_embedding).astype(np.float32)
    # (E, 86)

    ef = df_encoded[['TimestampScaled', 'LINK_SENTIMENT']].values
    edge_feature = np.concatenate([ef, edge_senti_embedding], axis=1)
    edge_feature = torch.Tensor(edge_feature).float()  # (E, 88)

    edge_index = torch.Tensor(
        df_encoded[['SOURCE_SUBREDDIT',
                    'TARGET_SUBREDDIT']].values.transpose()).long()  # (2, E)
    num_nodes = torch.max(edge_index) + 1

    edge_time = torch.FloatTensor(df_encoded['TIMESTAMP'].values)

    graph = Graph(
        node_feature=node_feature,
        edge_feature=edge_feature,
        edge_index=edge_index,
        edge_time=edge_time,
        directed=True
    )

    return graph


# =============================================================================
# College Message Dataset.
# =============================================================================


def load_college_message_dataset(dataset_dir: str) -> Graph:
    df_trans = pd.read_csv(dataset_dir, sep=' ', header=None)
    df_trans.columns = ['SRC', 'DST', 'TIMESTAMP']
    assert not np.any(pd.isna(df_trans).values)
    df_trans.reset_index(drop=True, inplace=True)

    # Node IDs of this dataset start from 1, re-index to 0-based.
    df_trans['SRC'] -= 1
    df_trans['DST'] -= 1

    print('num of edges:', len(df_trans))
    print('num of nodes:', np.max(df_trans[['SRC', 'DST']].values) + 1)

    time_scaler = MinMaxScaler((0, 2))
    df_trans['TimestampScaled'] = time_scaler.fit_transform(
        df_trans['TIMESTAMP'].values.reshape(-1, 1))

    edge_feature = torch.Tensor(
        df_trans[['TimestampScaled']].values).view(-1, 1)
    edge_index = torch.Tensor(
        df_trans[['SRC', 'DST']].values.transpose()).long()  # (2, E)
    num_nodes = torch.max(edge_index) + 1

    node_feature = torch.ones(num_nodes, 1)

    edge_time = torch.FloatTensor(df_trans['TIMESTAMP'].values)

    graph = Graph(
        node_feature=node_feature,
        edge_feature=edge_feature,
        edge_index=edge_index,
        edge_time=edge_time,
        directed=True
    )

    return graph


def load_roland_dataset(format: str, name: str, dataset_dir: str
                        ) -> List[Graph]:
    if format == 'roland':
        # Load the entire graph from specified dataset.
        if name in ['AS-733']:
            g_all = load_AS_dataset(os.path.join(dataset_dir, name))
        elif name in ['bsi_svt_2008.tsv']:
            # NOTE: only BSI dataset supports hetero graph.
            g_all = load_bsi_dataset(os.path.join(dataset_dir, name),
                                     is_hetero=cfg.dataset.is_hetero)
        elif name in ['bitcoinotc.csv', 'bitcoinalpha.csv']:
            g_all = load_bitcoin_dataset(os.path.join(dataset_dir, name))
        elif name in ['reddit-body.tsv', 'reddit-title.tsv']:
            g_all = load_reddit_dataset(os.path.join(dataset_dir, name))
        elif name in ['CollegeMsg.txt']:
            g_all = load_college_message_dataset(
                os.path.join(dataset_dir, name))
        else:
            raise ValueError(f'Unsupported filename')

        # Make the graph snapshots.
        snapshot_freq = cfg.transaction.snapshot_freq
        if snapshot_freq.upper() in ['D', 'W', 'M']:
            # Split snapshot using calendar frequency.
            snapshot_list = utils.make_graph_snapshot(g_all,
                                                      snapshot_freq,
                                                      cfg.dataset.is_hetero)
        elif snapshot_freq.endswith('s'):
            # Split using frequency in terms of seconds.
            assert snapshot_freq.endswith('s')
            snapshot_freq = int(snapshot_freq.strip('s'))
            assert not cfg.dataset.is_hetero, 'Hetero graph is not supported.'
            snapshot_list = utils.make_graph_snapshot_by_seconds(g_all,
                                                                 snapshot_freq)
        else:
            raise ValueError(f'Unsupported frequency type: {snapshot_freq}')

        num_nodes = g_all.edge_index.max() + 1

        for g_snapshot in snapshot_list:
            g_snapshot.node_states = [0 for _ in range(cfg.gnn.layers_mp)]
            g_snapshot.node_cells = [0 for _ in range(cfg.gnn.layers_mp)]
            g_snapshot.node_degree_existing = torch.zeros(num_nodes)

        # Filter small snapshots.
        filtered_graphs = list()
        for g in snapshot_list:
            if g.num_edges >= 10:
                filtered_graphs.append(g)

        return filtered_graphs


register_loader('roland', load_roland_dataset)
