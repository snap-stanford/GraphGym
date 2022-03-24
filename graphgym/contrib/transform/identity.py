import torch
from torch_geometric.utils import add_remaining_self_loops
from torch_scatter import scatter_add


def norm(edge_index, num_nodes, edge_weight=None, improved=False, dtype=None):
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ),
                                 dtype=dtype,
                                 device=edge_index.device)

    fill_value = 1.0 if not improved else 2.0
    edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight,
                                                       fill_value, num_nodes)

    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


# cpu version
def compute_identity(edge_index, n, k):
    id, value = norm(edge_index, n)
    adj_sparse = torch.sparse.FloatTensor(id, value, torch.Size([n, n]))
    adj = adj_sparse.to_dense()
    diag_all = [torch.diag(adj)]
    adj_power = adj
    for i in range(1, k):
        adj_power = adj_power @ adj
        diag_all.append(torch.diag(adj_power))
    diag_all = torch.stack(diag_all, dim=1)
    return diag_all
