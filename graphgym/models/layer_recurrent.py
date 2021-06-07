'''
This file contains wrapper layers and constructors for dynamic/recurrent GNNs.
'''
from graphgym.register import register_layer
import torch
import torch.nn as nn
import torch.nn.functional as F
from graphgym.config import cfg
from graphgym.models.act import act_dict
from graphgym.models.layer import layer_dict
from graphgym.models.update import update_dict


class RecurrentGraphLayer(nn.Module):
    '''
    The recurrent graph layer for snapshot-based dynamic graphs.
    This layer requires
        (1): a GNN block for message passing.
        (2): a node embedding/state update module.

    This layer updates node embedding as the following:
        h[l, t] = Update(h[l, t-1], GNN(h[l-1, t])).
    
    This layer corresponds to a particular l-th layer in multi-layer setting,
        the layer id is specified by 'id' in '__init__'.
    '''
    def __init__(self, name: str, dim_in: int, dim_out: int, has_act: bool=True,
                 has_bn: bool=True, has_l2norm: bool=False, layer_id: int=0,
                 **kwargs):
        '''
        Args:
            name (str): The name of GNN layer to use for message-passing.
            dim_in (int): Dimension of input node feature.
            dim_out (int): Dimension of updated embedding.
            has_act (bool, optional): Whether to after message passing.
                Defaults to True.
            has_bn (bool, optional): Whether add batch normalization for
                node embedding. Defaults to True.
            has_l2norm (bool, optional): Whether to add L2-normalization for
                message passing result. Defaults to False.
            layer_id (int, optional): The layer id in multi-layer setting.
                Defaults to 0.
        '''
        super(RecurrentGraphLayer, self).__init__()
        self.has_l2norm = has_l2norm
        if layer_id < 0:
            raise ValueError(f'layer_id must be non-negative, got {layer_id}.')
        self.layer_id = layer_id
        has_bn = has_bn and cfg.gnn.batchnorm
        self.dim_in = dim_in
        self.dim_out = dim_out
        # Construct the internal GNN layer.
        self.layer = layer_dict[name](dim_in, dim_out,
                                      bias=not has_bn, **kwargs)
        layer_wrapper = []
        if has_bn:
            layer_wrapper.append(nn.BatchNorm1d(
                dim_out, eps=cfg.bn.eps, momentum=cfg.bn.mom))
        if cfg.gnn.dropout > 0:
            layer_wrapper.append(nn.Dropout(
                p=cfg.gnn.dropout, inplace=cfg.mem.inplace))
        if has_act:
            layer_wrapper.append(act_dict[cfg.gnn.act])
        self.post_layer = nn.Sequential(*layer_wrapper)
        # self.update = self.construct_update_block(self.dim_in, self.dim_out,
        #                                           self.layer_id)
        self.update = update_dict[cfg.gnn.embed_update_method](self.dim_in,
                                                               self.dim_out,
                                                               self.layer_id)

    def _init_hidden_state(self, batch):
        # Initialize all node-states to zero.
        if not isinstance(batch.node_states[self.layer_id], torch.Tensor):
            batch.node_states[self.layer_id] = torch.zeros(
                batch.node_feature.shape[0], self.dim_out).to(
                batch.node_feature.device)

    def forward(self, batch):
        # Message passing.
        batch = self.layer(batch)
        batch.node_feature = self.post_layer(batch.node_feature)
        if self.has_l2norm:
            batch.node_feature = F.normalize(batch.node_feature, p=2, dim=1)

        self._init_hidden_state(batch)
        # Compute output from updater block.
        batch = self.update(batch)
        # batch.node_states[self.layer_id] = node_states_new
        batch.node_feature = batch.node_states[self.layer_id]
        return batch


register_layer('recurrent_graph_layer', RecurrentGraphLayer)
