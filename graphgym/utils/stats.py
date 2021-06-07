import torch


def node_degree(edge_index, n=None, mode='in'):
    if mode == 'in':
        index = edge_index[0, :]
    elif mode == 'out':
        index = edge_index[1, :]
    else:
        index = edge_index.flatten()
    n = edge_index.max() + 1 if n is None else n
    degree = torch.zeros(n)
    ones = torch.ones(index.shape[0])
    return degree.scatter_add_(0, index, ones)
