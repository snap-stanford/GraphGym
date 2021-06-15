"""Embedding update modules for dynamic graphs."""
import graphgym.register as register
import torch
import torch.nn as nn
from graphgym.models.layer import MLP


class MovingAverageUpdater(nn.Module):
    """
    Moving average updater for node embeddings,
    let h[l, t] denote all nodes' embedding at the l-th layer at snapshot t.
    
    h[l,t] = KeepRatio * h[l,t-1] + (1-KeepRatio) * h[l-1,t]
    
    where the precomputed KeepRatio at current snapshot t is node-specific,
        which depends on the node's degree in all snapshots before t and nodes's
        degree in snapshot at time t.
    """

    def __init__(self, dim_in: int, dim_out: int, layer_id: int) -> None:
        self.layer_id = layer_id
        super(MovingAverageUpdater, self).__init__()

    def forward(self, batch):
        # TODO: check if boardcasting is correct.
        H_prev = batch.node_states[self.layer_id]
        X = batch.node_feature
        H_new = H_prev * batch.keep_ratio + X * (1.0 - batch.keep_ratio)
        batch.node_states[self.layer_id] = H_new
        return batch


class MLPUpdater(nn.Module):
    """
    Node embedding update block using simple MLP.
    
    h[l,t] = MLP(concat(h[l,t-1],h[l-1,t]))
    """

    def __init__(self, dim_in: int, dim_out: int, layer_id: int,
                 num_layers: int):
        """
        Args:
            dim_in (int): dimension of h[l-1, t].
            dim_out (int): dimension of h[l, t-1], node embedding dimension of
                the current layer level.
            layer_id (int): the index of current layer in multi-layer setting.
            num_layers (int): number of layers in MLP.
        """
        super(MLPUpdater, self).__init__()
        self.layer_id = layer_id
        # FIXME:
        # assert num_layers > 1, 'There is a problem with layer=1 now, pending fix.'
        self.mlp = MLP(dim_in=dim_in + dim_out, dim_out=dim_out,
                       num_layers=num_layers)

    def forward(self, batch):
        H_prev = batch.node_states[self.layer_id]
        X = batch.node_feature
        concat = torch.cat((H_prev, X), axis=1)
        H_new = self.mlp(concat)
        batch.node_states[self.layer_id] = H_new
        return batch


class GRUUpdater(nn.Module):
    """
    Node embedding update block using standard GRU.

    h[l,t] = GRU(h[l,t-1], h[l-1,t])
    """
    def __init__(self, dim_in: int, dim_out: int, layer_id: int):
        # dim_in (dim of X): dimension of input node_feature.
        # dim_out (dim of H): dimension of previous and current hidden states.
        # forward(X, H) --> H.
        super(GRUUpdater, self).__init__()
        self.layer_id = layer_id
        self.GRU_Z = nn.Sequential(
            nn.Linear(dim_in + dim_out, dim_out, bias=True),
            nn.Sigmoid())
        # reset gate.
        self.GRU_R = nn.Sequential(
            nn.Linear(dim_in + dim_out, dim_out, bias=True),
            nn.Sigmoid())
        # new embedding gate.
        self.GRU_H_Tilde = nn.Sequential(
            nn.Linear(dim_in + dim_out, dim_out, bias=True),
            nn.Tanh())

    def forward(self, batch):
        H_prev = batch.node_states[self.layer_id]
        X = batch.node_feature
        Z = self.GRU_Z(torch.cat([X, H_prev], dim=1))
        R = self.GRU_R(torch.cat([X, H_prev], dim=1))
        H_tilde = self.GRU_H_Tilde(torch.cat([X, R * H_prev], dim=1))
        H_gru = Z * H_prev + (1 - Z) * H_tilde
        batch.node_states[self.layer_id] = H_gru
        return batch


update_dict = {
    'moving_average': MovingAverageUpdater,
    'mlp': MLPUpdater,
    'gru': GRUUpdater
}

# merge additional update modules in register.update_dict.
update_dict = {**register.update_dict, **update_dict}
