import torch
import torch.nn as nn
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import zeros
from torch_geometric.utils import add_remaining_self_loops
from torch_scatter import scatter_add

from graphgym.config import cfg
from graphgym.register import register_layer


class ResidualEdgeConvLayer(MessagePassing):
    r'''
    A general GNN layer with arbitrary edge features and self residual
    connections.
    '''

    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = False, bias: bool = True,
                 **kwargs):
        super(ResidualEdgeConvLayer, self).__init__(aggr=cfg.gnn.agg, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.normalize = cfg.gnn.normalize_adj
        self.msg_direction = cfg.gnn.msg_direction

        if self.msg_direction == 'single':
            self.linear_msg = nn.Linear(in_channels + cfg.dataset.edge_dim,
                                        out_channels, bias=False)
        elif self.msg_direction == 'both':
            self.linear_msg = nn.Linear(in_channels * 2 + cfg.dataset.edge_dim,
                                        out_channels, bias=False)
        else:
            raise ValueError

        if cfg.gnn.skip_connection == 'affine':
            self.linear_skip = nn.Linear(in_channels, out_channels, bias=True)
        elif cfg.gnn.skip_connection == 'identity':
            assert self.in_channels == self.out_channels

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False,
             dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None, edge_feature=None):
        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            if self.normalize:
                edge_index, norm = self.norm(edge_index, x.size(self.node_dim),
                                             edge_weight, self.improved,
                                             x.dtype)
            else:
                norm = edge_weight
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result
        if cfg.gnn.skip_connection == 'affine':
            skip_x = self.linear_skip(x)
        elif cfg.gnn.skip_connection == 'identity':
            skip_x = x
        else:
            skip_x = 0.0
        return self.propagate(edge_index, x=x, norm=norm,
                              edge_feature=edge_feature) + skip_x

    def message(self, x_i, x_j, norm, edge_feature):
        if self.msg_direction == 'both':
            x_j = torch.cat((x_i, x_j, edge_feature), dim=-1)
        elif self.msg_direction == 'single':
            x_j = torch.cat((x_j, edge_feature), dim=-1)
        else:
            raise ValueError
        x_j = self.linear_msg(x_j)
        return norm.view(-1, 1) * x_j if norm is not None else x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class ResidualEdgeConv(nn.Module):
    '''Wrapper for residual edge conv layer'''

    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(ResidualEdgeConv, self).__init__()
        self.model = ResidualEdgeConvLayer(dim_in, dim_out, bias=bias)

    def forward(self, batch):
        batch.node_feature = self.model(batch.node_feature, batch.edge_index,
                                        edge_feature=batch.edge_feature)
        return batch


register_layer('residual_edge_conv', ResidualEdgeConv)
