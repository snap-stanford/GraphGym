"""
An improved version of graphgym.models.head.GNNEdgeHead. This head handles
large link prediction tasks by splitting them into chunks to avoid OOM errors.
This is particular useful for computing MRR when a large amount of memory is
needed.

(Not implemented yet) Alternatively, one may implement head for MRR by all
prediction task to CPU, by doing so, we need sepearate heads for training and
inference (training requires everything including head to be on GPU).
"""
import torch
import torch.nn as nn
from graphgym.config import cfg
from graphgym.models.layer import MLP
from graphgym.register import register_head


class LargeGNNEdgeHead(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        # Use dim_in for graph conv, since link prediction dim_out could be
        # binary
        # E.g. if decoder='dot', link probability is dot product between
        # node embeddings, of dimension dim_in
        super(LargeGNNEdgeHead, self).__init__()
        # module to decode edges from node embeddings

        if cfg.model.edge_decoding == 'concat':
            # Only use node features.
            self.layer_post_mp = MLP(dim_in * 2, dim_out,
                                     num_layers=cfg.gnn.layers_post_mp,
                                     bias=True)
            # requires parameter
            self.decode_module = lambda v1, v2: \
                self.layer_post_mp(torch.cat((v1, v2), dim=-1))
        elif cfg.model.edge_decoding == 'edgeconcat':
            # Use both node and edge features.
            self.layer_post_mp = MLP(dim_in * 2 + cfg.dataset.edge_dim, dim_out,
                                     num_layers=cfg.gnn.layers_post_mp,
                                     bias=True)
            # requires parameter
            self.decode_module = lambda v1, v2, edge: \
                self.layer_post_mp(torch.cat((v1, v2, edge), dim=-1))
        else:
            if dim_out > 1:
                raise ValueError(
                    'Binary edge decoding ({})is used for multi-class '
                    'edge/link prediction.'.format(cfg.model.edge_decoding))
            self.layer_post_mp = MLP(dim_in, dim_in,
                                     num_layers=cfg.gnn.layers_post_mp,
                                     bias=True)
            if cfg.model.edge_decoding == 'dot':
                self.decode_module = lambda v1, v2: torch.sum(v1 * v2, dim=-1)
            elif cfg.model.edge_decoding == 'cosine_similarity':
                self.decode_module = nn.CosineSimilarity(dim=-1)
            else:
                raise ValueError('Unknown edge decoding {}.'.format(
                    cfg.model.edge_decoding))

    def _apply_index(self, batch):
        return batch.node_feature[batch.edge_label_index], \
            batch.edge_label

    def forward_pred(self, batch):
        # TODO: consider moving this to config.
        predict_batch_size = 500000  # depends on GPU memroy size.
        num_pred = len(batch.edge_label)
        label = batch.edge_label
        if num_pred >= predict_batch_size:
            # for large prediction tasks, split into chunks.
            num_chunks = num_pred // predict_batch_size + 1
            edge_label_index_chunks = torch.chunk(
                batch.edge_label_index, num_chunks, dim=1)
            gathered_pred = list()

            for edge_label_index in edge_label_index_chunks:
                pred = batch.node_feature[edge_label_index]
                # node features of the source node of each edge.
                nodes_first = pred[0]
                nodes_second = pred[1]
                if cfg.model.edge_decoding == 'edgeconcat':
                    raise NotImplementedError
                else:
                    pred = self.decode_module(nodes_first, nodes_second)
                gathered_pred.append(pred)

            pred = torch.cat(gathered_pred)
        else:
            pred, label = self._apply_index(batch)
            # node features of the source node of each edge.
            nodes_first = pred[0]
            nodes_second = pred[1]
            if cfg.model.edge_decoding == 'edgeconcat':
                edge_feature = torch.index_select(
                    batch.edge_feature, 0, batch.edge_split_index)
                pred = self.decode_module(
                    nodes_first, nodes_second, edge_feature)
            else:
                pred = self.decode_module(nodes_first, nodes_second)
        return pred, label

    def forward(self, batch):
        if cfg.model.edge_decoding != 'concat' and \
                cfg.model.edge_decoding != 'edgeconcat':
            batch = self.layer_post_mp(batch)
        pred, label = self.forward_pred(batch)
        return pred, label


register_head('link_pred_large', LargeGNNEdgeHead)
