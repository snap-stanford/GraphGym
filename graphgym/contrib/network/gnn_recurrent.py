import graphgym.register as register
import torch.nn as nn
import torch.nn.functional as F
from graphgym.config import cfg
from graphgym.contrib.stage import *
from graphgym.init import init_weights
from graphgym.models.act import act_dict
from graphgym.models.feature_augment import Preprocess
from graphgym.models.feature_encoder import (edge_encoder_dict,
                                             node_encoder_dict)
from graphgym.models.head import head_dict
from graphgym.models.layer import (BatchNorm1dEdge, BatchNorm1dNode,
                                   GeneralMultiLayer, layer_dict)
from graphgym.models.layer_recurrent import RecurrentGraphLayer
from graphgym.register import register_network


def GNNLayer(dim_in: int, dim_out: int, has_act: bool=True, layer_id: int=0):
    # General constructor for GNN layer.
    return RecurrentGraphLayer(cfg.gnn.layer_type, dim_in, dim_out,
                               has_act, layer_id=layer_id)


def GNNPreMP(dim_in, dim_out):
    r'''Constructs preprocessing layers: dim_in --> dim_out --> dim_out --> ... --> dim_out'''
    return GeneralMultiLayer('linear', cfg.gnn.layers_pre_mp,
                             dim_in, dim_out, dim_inner=dim_out,
                             final_act=True)


class GNNStackStage(nn.Module):
    def __init__(self, dim_in, dim_out, num_layers):
        super(GNNStackStage, self).__init__()
        for i in range(num_layers):
            d_in = dim_in if i == 0 else dim_out
            layer = GNNLayer(d_in, dim_out, layer_id=i)
            self.add_module('layer{}'.format(i), layer)
        self.dim_out = dim_out

    def forward(self, batch):
        for layer in self.children():
            batch = layer(batch)
        if cfg.gnn.l2norm:
            batch.node_feature = F.normalize(batch.node_feature, p=2, dim=-1)
        return batch


stage_dict = {
    'stack': GNNStackStage,
}

stage_dict = {**register.stage_dict, **stage_dict}


class GNNRecurrent(nn.Module):
    r'''The General GNN model'''

    def __init__(self, dim_in, dim_out, **kwargs):
        r'''Initializes the GNN model.

        Args:
            dim_in, dim_out: dimensions of in and out channels.
            Parameters:
            node_encoding_classes - For integer features, gives the number
            of possible integer features to map.
        '''
        super(GNNRecurrent, self).__init__()
        # Stage: inter-layer connections.
        GNNStage = stage_dict[cfg.gnn.stage_type]
        # Head: prediction head, the final layer.
        GNNHead = head_dict[cfg.dataset.task]

        if cfg.dataset.node_encoder:
            # Encode integer node features via nn.Embeddings
            NodeEncoder = node_encoder_dict[cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(cfg.dataset.encoder_dim)
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(cfg.dataset.encoder_dim)
            # Update dim_in to reflect the new dimension fo the node features
            dim_in = cfg.dataset.encoder_dim

        if cfg.dataset.edge_encoder:
            # Encode integer edge features via nn.Embeddings
            EdgeEncoder = edge_encoder_dict[cfg.dataset.edge_encoder_name]
            self.edge_encoder = EdgeEncoder(cfg.dataset.encoder_dim)
            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dEdge(cfg.dataset.edge_dim)

        self.preprocess = Preprocess(dim_in)
        d_in = self.preprocess.dim_out

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(d_in, cfg.gnn.dim_inner)
            d_in = cfg.gnn.dim_inner
        if cfg.gnn.layers_mp >= 1:
            self.mp = GNNStage(dim_in=d_in,
                               dim_out=cfg.gnn.dim_inner,
                               num_layers=cfg.gnn.layers_mp)
            d_in = self.mp.dim_out
        self.post_mp = GNNHead(dim_in=d_in, dim_out=dim_out)

        self.apply(init_weights)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


register_network('gnn_recurrent', GNNRecurrent)
