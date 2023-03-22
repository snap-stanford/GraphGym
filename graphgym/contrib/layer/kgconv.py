import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros

from graphgym.config import cfg
from graphgym.register import register_layer

act_dict = {
    'relu': nn.ReLU(inplace=cfg.mem.inplace),
    'selu': nn.SELU(inplace=cfg.mem.inplace),
    'prelu': nn.PReLU(),
    'elu': nn.ELU(inplace=cfg.mem.inplace),
    'lrelu_01': nn.LeakyReLU(negative_slope=0.1, inplace=cfg.mem.inplace),
    'lrelu_025': nn.LeakyReLU(negative_slope=0.25, inplace=cfg.mem.inplace),
    'lrelu_05': nn.LeakyReLU(negative_slope=0.5, inplace=cfg.mem.inplace),
}


class HeteConvLayer(MessagePassing):
    r"""
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 edge_channels,
                 improved=False,
                 cached=False,
                 bias=True,
                 loss='bce',
                 **kwargs):
        super(HeteConvLayer, self).__init__(aggr=cfg.kg.agg, **kwargs)
        assert in_channels == out_channels  # todo: see if necessary to relax this constraint
        self.in_channels = in_channels
        self.out_channels = out_channels
        if cfg.kg.edge_feature == 'both':
            edge_channels = 2
        else:
            edge_channels = 1
        self.edge_channels = edge_channels
        self.improved = improved
        self.cached = cached
        self.normalize = False
        self.msg_direction = cfg.kg.msg_direction

        # TODO: Take meta learning explaination. Consider grad in message passing

        if self.msg_direction == 'single':
            self.msg_node = nn.Linear(in_channels, out_channels)
            self.msg_edge = nn.Linear(edge_channels, out_channels)
            self.gate_self = nn.Linear(in_channels, out_channels)  # optional
            self.gate_msg_node = nn.Linear(in_channels,
                                           out_channels)  # optional
            self.gate_msg_edge = nn.Linear(edge_channels,
                                           out_channels)  # optional
        else:
            self.msg_node = nn.Linear(in_channels * 2 + edge_channels,
                                      out_channels)
            self.gate_self = nn.Linear(in_channels, out_channels)
            self.gate_msg = nn.Linear(in_channels * 2 + edge_channels,
                                      out_channels)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        if loss == 'bce':
            self.loss = nn.BCEWithLogitsLoss(
                size_average=cfg.model.size_average, reduce=False)

        self.reset_parameters()

    def reset_parameters(self):
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None

    def forward(self, x, edge_index, edge_weight=None, edge_feature=None):
        pred = self.propagate(edge_index, x=x, edge_feature=edge_feature)

        return pred

    def message(self, x_i, x_j, edge_feature):
        if self.msg_direction == 'both':
            x_j = torch.cat((x_i, x_j, edge_feature), dim=-1)
        else:

            if cfg.kg.edge_feature != 'raw':
                pred = torch.sum(x_i * x_j, dim=-1, keepdim=True)
                loss = self.loss(pred, edge_feature)
                if cfg.kg.edge_feature == 'loss':
                    edge_feature = loss
                elif cfg.kg.edge_feature == 'both':
                    edge_feature = torch.cat((edge_feature, loss), dim=-1)

            msg = self.msg_node(x_j) + self.msg_edge(edge_feature)
            if cfg.kg.gate_msg:
                gate = self.gate_msg_node(x_j) + self.gate_msg_edge(
                    edge_feature)
                gate = torch.sigmoid(gate)
                if cfg.kg.gate_bias:
                    gate = gate - 0.5
                msg = msg * gate
        return msg

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class SelfLayer(nn.Module):
    r"""
    """
    def __init__(self, in_channels, out_channels, **kwargs):
        super(SelfLayer, self).__init__()
        assert in_channels == out_channels  # todo: see if necessary to relax this constraint
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.gate_self = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        gate = torch.sigmoid(self.gate_self(x))
        if cfg.kg.gate_bias:
            gate = gate + 0.5
        return gate * x

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class Postlayer(nn.Module):
    r"""
    """
    def __init__(self,
                 dim_out,
                 bn=False,
                 dropout=0,
                 act=True,
                 l2norm=False,
                 **kwargs):
        super(Postlayer, self).__init__()
        self.l2norm = l2norm
        self.post_layer = []
        if bn:
            self.post_layer.append(
                nn.BatchNorm1d(dim_out, eps=cfg.bn.eps, momentum=cfg.bn.mom))
        if dropout > 0:
            self.post_layer.append(
                nn.Dropout(p=dropout, inplace=cfg.mem.inplace))
        if act:
            self.post_layer.append(act_dict[cfg.gnn.act])
        self.post_layer = nn.Sequential(*self.post_layer)

    def forward(self, x):
        if cfg.kg.pos_l2norm == 'post':
            x = self.post_layer(x)
            if self.l2norm:
                x = F.normalize(x, p=2, dim=1)
        else:
            if self.l2norm:
                x = F.normalize(x, p=2, dim=1)
            x = self.post_layer(x)

        return x


class KGHeteroConv(torch.nn.Module):
    r"""A "wrapper" layer designed for heterogeneous graph layers. It takes a
    heterogeneous graph layer, such as :class:`deepsnap.hetero_gnn.HeteroSAGEConv`, at the initializing stage.
    """
    def __init__(self, convs, selfs=None, posts=None, aggr="add"):
        super(KGHeteroConv, self).__init__()

        self.convs = convs
        self.selfs = selfs  # self computations
        self.posts = posts
        self.modules_convs = torch.nn.ModuleList(convs.values())
        if self.selfs is not None:
            self.modules_selfs = torch.nn.ModuleList(selfs.values())
        if self.posts is not None:
            self.modules_posts = torch.nn.ModuleList(posts.values())

        assert aggr in ["add", "mean", "max", "mul", "concat", None]
        self.aggr = aggr

    def forward(self, node_features, edge_indices, edge_features=None):
        r"""The forward function for `HeteroConv`.

        Args:
            node_features (dict): A dictionary each key is node type and the corresponding
                value is a node feature tensor.
            edge_indices (dict): A dictionary each key is message type and the corresponding
                value is an edge index tensor.
            edge_features (dict): A dictionary each key is edge type and the corresponding
                value is an edge feature tensor. Default is `None`.
        """
        # node embedding computed from each message type
        message_type_emb = {}
        for message_key, message_type in edge_indices.items():
            if message_key not in self.convs:
                continue
            neigh_type, edge_type, self_type = message_key
            node_feature_neigh = node_features[neigh_type]
            node_feature_self = node_features[self_type]
            if edge_features is not None:
                edge_feature = edge_features[edge_type]
            edge_index = edge_indices[message_key]

            message_type_emb[message_key] = (self.convs[message_key](
                x=(node_feature_neigh, node_feature_self),
                edge_index=edge_index,
                edge_feature=edge_feature))

        # aggregate node embeddings from different message types into 1 node
        # embedding for each node
        node_emb = {tail: [] for _, _, tail in message_type_emb.keys()}

        for (_, _, tail), item in message_type_emb.items():
            node_emb[tail].append(item)

        # add self messages
        if self.selfs is not None:
            for node_key, node_type in node_features.items():
                # print(self.selfs[node_key](node_features[node_key]))
                node_emb[node_key].append(self.selfs[node_key](
                    node_features[node_key]))

        # Aggregate multiple embeddings with the same tail.
        for node_type, embs in node_emb.items():
            if len(embs) == 1:
                node_emb[node_type] = embs[0]
            else:
                node_emb[node_type] = self.aggregate(embs)
            if self.posts is not None:
                node_emb[node_type] = self.posts[node_type](
                    node_emb[node_type])

        return node_emb

    def aggregate(self, xs):
        r"""The aggregation for each node type. Currently support `concat`, `add`,
        `mean`, `max` and `mul`.
        """
        if self.aggr == "concat":
            return torch.cat(xs, dim=-1)

        x = torch.stack(xs, dim=-1)
        if self.aggr == "add":
            x = x.sum(dim=-1)
        elif self.aggr == "mean":
            x = x.mean(dim=-1)
        elif self.aggr == "max":
            x = x.max(dim=-1)[0]
        elif self.aggr == "mul":
            x = x.prod(dim=-1)[0]
        return x


class KGHeteConv(nn.Module):
    def __init__(self,
                 dim_in,
                 dim_out,
                 bias=False,
                 dim_edge=None,
                 bn=False,
                 act=True,
                 l2norm=False,
                 **kwargs):
        super(KGHeteConv, self).__init__()
        if cfg.kg.hete:
            convs = {
                ('data', 'property', 'task'):
                HeteConvLayer(dim_in, dim_out, edge_channels=dim_edge),
                ('task', 'property', 'data'):
                HeteConvLayer(dim_in, dim_out, edge_channels=dim_edge)
            }
            selfs = {
                'data': SelfLayer(dim_in, dim_out),
                'task': SelfLayer(dim_in, dim_out)
            }
            posts = {
                'data': Postlayer(dim_out, bn=bn, act=act, l2norm=l2norm),
                'task': Postlayer(dim_out, bn=bn, act=act, l2norm=l2norm)
            }
        else:
            conv = HeteConvLayer(dim_in, dim_out, edge_channels=dim_edge)
            convs = {
                ('data', 'property', 'task'): conv,
                ('task', 'property', 'data'): conv
            }
            self_fun = SelfLayer(dim_in, dim_out)
            selfs = {'data': self_fun, 'task': self_fun}
            post = Postlayer(dim_out, bn=bn, act=act, l2norm=l2norm)
            posts = {'data': post, 'task': post}
        if not cfg.kg.self_trans:
            selfs = None
        self.model = KGHeteroConv(convs, selfs, posts)

    def pred(self, batch):

        # todo: compare with index_select then sparse dot product
        pred = torch.matmul(batch.node_feature['data'],
                            batch.node_feature['task'].transpose(0, 1))
        id_data = batch.pred_index[0, :]
        id_task = batch.pred_index[1, :]
        pred = pred[id_data].gather(1, id_task.view(-1, 1))
        # pdb.set_trace()
        if not hasattr(batch, 'pred'):
            batch.pred = pred
        else:
            batch.pred = batch.pred + pred
        return batch

    def forward(self, batch):
        if cfg.kg.pred == 'every':
            batch = self.pred(batch)
        if cfg.kg.self_msg == 'add':
            node_feature = batch.node_feature
            batch.node_feature = self.model(batch.node_feature,
                                            batch.edge_index,
                                            batch.edge_feature)
            for key in batch.node_feature.keys():
                batch.node_feature[
                    key] = batch.node_feature[key] + node_feature[key]
        else:
            batch.node_feature = self.model(batch.node_feature,
                                            batch.edge_index,
                                            batch.edge_feature)
        return batch


register_layer('kgheteconv', KGHeteConv)
