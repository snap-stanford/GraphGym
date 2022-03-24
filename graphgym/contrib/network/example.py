import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.graphgym.models import MLP

from graphgym.config import cfg
from graphgym.register import register_network


class GNNNodeHead(nn.Module):
    '''Head of GNN, node prediction'''
    def __init__(self, dim_in, dim_out):
        super(GNNNodeHead, self).__init__()
        self.layer_post_mp = MLP(dim_in,
                                 dim_out,
                                 num_layers=cfg.gnn.layers_post_mp,
                                 bias=True)

    def _apply_index(self, batch):
        if batch.node_label_index.shape[0] == batch.node_label.shape[0]:
            return batch.node_feature[batch.node_label_index], batch.node_label
        else:
            return batch.node_feature[batch.node_label_index], \
                   batch.node_label[batch.node_label_index]

    def forward(self, batch):
        batch = self.layer_post_mp(batch)
        pred, label = self._apply_index(batch)
        return pred, label


class ExampleGNN(torch.nn.Module):
    def __init__(self, dim_in, dim_out, num_layers=2, model_type='GCN'):
        super(ExampleGNN, self).__init__()
        conv_model = self.build_conv_model(model_type)
        self.convs = nn.ModuleList()
        self.convs.append(conv_model(dim_in, dim_in))

        for _ in range(num_layers - 1):
            self.convs.append(conv_model(dim_in, dim_in))

        self.post_mp = GNNNodeHead(dim_in=dim_in, dim_out=dim_out)

    def build_conv_model(self, model_type):
        if model_type == 'GCN':
            return pyg_nn.GCNConv
        elif model_type == 'GAT':
            return pyg_nn.GATConv
        elif model_type == "GraphSage":
            return pyg_nn.SAGEConv
        else:
            raise ValueError("Model {} unavailable".format(model_type))

    def forward(self, batch):
        x, edge_index, x_batch = \
            batch.node_feature, batch.edge_index, batch.batch

        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = pyg_nn.global_add_pool(x, x_batch)
        x = self.post_mp(x)
        x = F.log_softmax(x, dim=1)
        batch.node_feature = x
        return batch


register_network('example', ExampleGNN)
