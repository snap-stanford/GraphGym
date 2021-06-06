import deepsnap
import torch
import torch.nn as nn
from graphgym.config import cfg
from graphgym.register import register_edge_encoder, register_node_encoder


class TransactionEdgeEncoder(torch.nn.Module):
    r"""A module that encodes edge features in the transaction graph.

    Example:
        TransactionEdgeEncoder(
          (embedding_list): ModuleList(
            (0): Embedding(50, 32)  # The first integral edge feature has 50 unique values.
                # convert this integral feature to 32 dimensional embedding.
            (1): Embedding(8, 32)
            (2): Embedding(252, 32)
            (3): Embedding(252, 32)
          )
          (linear_amount): Linear(in_features=1, out_features=64, bias=True)
          (linear_time): Linear(in_features=1, out_features=64, bias=True)
        )

        Initial edge feature dimension = 6
        Final edge embedding dimension = 32 + 32 + 32 + 32 + 64 + 64 = 256
    """

    def __init__(self, emb_dim: int):
        # emb_dim is not used here.
        super(TransactionEdgeEncoder, self).__init__()

        self.embedding_list = torch.nn.ModuleList()
        # Note: feature_edge_int_num[i] = len(torch.unique(graph.edge_feature[:, i]))
        # where i-th edge features are integral.
        for num in cfg.transaction.feature_edge_int_num:
            emb = torch.nn.Embedding(num, cfg.transaction.feature_int_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.embedding_list.append(emb)

        # Embed non-integral features.
        self.linear_amount = nn.Linear(1, cfg.transaction.feature_amount_dim)
        self.linear_time = nn.Linear(1, cfg.transaction.feature_time_dim)
        # update edge_dim
        cfg.dataset.edge_dim = len(cfg.transaction.feature_edge_int_num) \
            * cfg.transaction.feature_int_dim \
            + cfg.transaction.feature_amount_dim \
            + cfg.transaction.feature_time_dim

    def forward(self, batch: deepsnap.batch.Batch) -> deepsnap.batch.Batch:
        edge_embedding = []
        for i in range(len(self.embedding_list)):
            edge_embedding.append(
                self.embedding_list[i](batch.edge_feature[:, i].long())
            )
        # By default, edge_feature[:, -2] contains edge amount,
        # edge_feature[:, -1] contains edge time.
        edge_embedding.append(
            self.linear_amount(batch.edge_feature[:, -2].view(-1, 1))
        )
        edge_embedding.append(
            self.linear_time(batch.edge_feature[:, -1].view(-1, 1))
        )
        batch.edge_feature = torch.cat(edge_embedding, dim=1)
        return batch


register_edge_encoder('roland', TransactionEdgeEncoder)


class TransactionNodeEncoder(torch.nn.Module):
    r"""A module that encodes node features in the transaction graph.

    Parameters:
        num_classes - the number of classes for the embedding mapping to learn

    Example:
        3 unique values for the first integral node feature.
        3 unique values for the second integral node feature.

        cfg.transaction.feature_node_int_num = [3, 3]
        cfg.transaction.feature_int_dim = 32

        TransactionNodeEncoder(
          (embedding_list): ModuleList(
            (0): Embedding(3, 32)  # embed the first node feature to 32-dimensional space.
            (1): Embedding(3, 32)  # embed the second node feature to 32-dimensional space.
          )
        )

        Initial node feature dimension = 2
        Final node embedding dimension = 32 + 32 = 256
    """

    def __init__(self, emb_dim: int, num_classes=None):
        super(TransactionNodeEncoder, self).__init__()
        self.embedding_list = torch.nn.ModuleList()
        for i, num in enumerate(cfg.transaction.feature_node_int_num):
            emb = torch.nn.Embedding(num, cfg.transaction.feature_int_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.embedding_list.append(emb)
        # update encoder_dim
        cfg.dataset.encoder_dim = len(cfg.transaction.feature_node_int_num) \
            * cfg.transaction.feature_int_dim

    def forward(self, batch: deepsnap.batch.Batch) -> deepsnap.batch.Batch:
        node_embedding = []
        for i in range(len(self.embedding_list)):
            node_embedding.append(
                self.embedding_list[i](batch.node_feature[:, i].long())
            )
        batch.node_feature = torch.cat(node_embedding, dim=1)
        return batch


register_node_encoder('roland', TransactionNodeEncoder)
