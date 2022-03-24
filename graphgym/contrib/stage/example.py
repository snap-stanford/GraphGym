import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from graphgym.config import cfg
from graphgym.register import register_stage


class GNNStackStage(nn.Module):
    '''Simple Stage that stack GNN layers'''

    def __init__(self, dim_in, dim_out, num_layers):
        super(GNNStackStage, self).__init__()
        for i in range(num_layers):
            d_in = dim_in if i == 0 else dim_out
            layer = GCNConv(d_in, dim_out)
            self.add_module('layer{}'.format(i), layer)
        self.dim_out = dim_out

    def forward(self, batch):
        for layer in self.children():
            batch = layer(batch)
        if cfg.gnn.l2norm:
            batch.node_feature = F.normalize(batch.node_feature, p=2, dim=-1)
        return batch


register_stage('example', GNNStackStage)
