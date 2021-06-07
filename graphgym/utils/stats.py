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







# edge_index = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]])

# print(compute_degree(edge_index, mode='in'))
# print(compute_degree(edge_index, mode='out'))
# print(compute_degree(edge_index, mode='both'))
