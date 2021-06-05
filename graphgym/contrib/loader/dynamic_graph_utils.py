"""
Helper functions and utilities for dynamic graphs.

Mar. 31, 2021.
"""
from typing import List

import numpy as np
import pandas as pd
import torch
from deepsnap.graph import Graph


def make_graph_snapshot(g_all: Graph,
                        snapshot_freq: str,
                        is_hetero: bool=False) -> List[Graph]:
    """
    Constructs a list of graph snapshots based from g_all using g_all.edge_time
    and provided snapshot_freq (frequency on calendar).

    Args:
        g_all: the entire graph object, g_all must have a edge_time attribute,
            g_all.edge_time consists of unix timestamp of edge time.
        snapshot_freq: snapshot frequency, must be one of
            'D': daily, 'W': weekly, and 'M': monthly.
        is_hetero: whether the graph is heterogeneous.

    Return:
        A list of graph object, each graph snapshot has edge level information
            (edge_feature, edge_time, etc) of only edges in that time period.
            However, every graph snapshot has the same and full node level
            information (node_feature, node_type, etc).
    """
    # Arg check.
    if not hasattr(g_all, 'edge_time'):
        raise KeyError('Temporal graph needs to have edge_time attribute.')

    if snapshot_freq.upper() not in ['D', 'W', 'M']:
        raise ValueError(f'Unsupported snapshot freq: {snapshot_freq}.')

    snapshot_freq = snapshot_freq.upper()
    t = g_all.edge_time.numpy().astype(np.int64)  # all timestamps.

    period_split = pd.DataFrame(
        {'Timestamp': t, 'TransactionTime': pd.to_datetime(t, unit='s')},
        index=range(len(g_all.edge_time))
    )

    freq_map = {'D': '%j',  # day of year.
                'W': '%W',  # week of year.
                'M': '%m'}  # month of year.

    period_split['Year'] = period_split['TransactionTime'].dt.strftime(
        '%Y').astype(int)

    period_split['SubYearFlag'] = period_split['TransactionTime'].dt.strftime(
        freq_map[snapshot_freq]).astype(int)

    period2id = period_split.groupby(['Year', 'SubYearFlag']).indices
    # e.g., dictionary w/ key = (2021, 3) and val = array(edge IDs).

    periods = sorted(list(period2id.keys()))  # ascending order.
    # alternatively, sorted(..., key=lambda x: x[0] + x[1]/1000).
    snapshot_list = list()
    for p in periods:
        # unique IDs of edges in this period.
        period_members = period2id[p]

        g_incr = Graph(
            node_feature=g_all.node_feature,
            edge_feature=g_all.edge_feature[period_members, :],
            edge_index=g_all.edge_index[:, period_members],
            edge_time=g_all.edge_time[period_members],
            directed=g_all.directed,
            list_n_type=g_all.list_n_type if is_hetero else None,
            list_e_type=g_all.list_e_type if is_hetero else None,
        )
        if is_hetero and hasattr(g_all, 'node_type'):
            g_incr.node_type = g_all.node_type
            g_incr.edge_type = g_all.edge_type[period_members]
        snapshot_list.append(g_incr)
    return snapshot_list


def make_graph_snapshot_by_seconds(g_all: Graph,
                                   freq_sec: int) -> List[Graph]:
    """
    Split the entire graph into snapshots by frequency in terms of seconds.
    """
    split_criterion = g_all.edge_time // freq_sec
    groups = torch.sort(torch.unique(split_criterion))[0]
    snapshot_list = list()
    for t in groups:
        period_members = (split_criterion == t)
        g_incr = Graph(
            node_feature=g_all.node_feature,
            edge_feature=g_all.edge_feature[period_members, :],
            edge_index=g_all.edge_index[:, period_members],
            edge_time=g_all.edge_time[period_members],
            directed=g_all.directed
        )
        snapshot_list.append(g_incr)
    return snapshot_list
