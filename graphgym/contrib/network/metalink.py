import torch
import torch.nn as nn
import torch.nn.functional as F
from deepsnap.batch import Batch

from graphgym.config import cfg
from graphgym.init import init_weights
from graphgym.models.feature_augment import Preprocess
from graphgym.models.gnn import GNNPreMP, stage_dict
from graphgym.models.layer import MLP, GeneralLayer, layer_dict
from graphgym.models.pooling import pooling_dict
from graphgym.register import register_network


class StackHeteConv(nn.Module):
    '''Simple Stage that stack GNN layers'''
    def __init__(self, dim_in, dim_out, num_layers, layer_name):
        super(StackHeteConv, self).__init__()
        self.num_layers = num_layers
        for i in range(num_layers):
            d_in = dim_in if i == 0 else dim_out
            has_act = cfg.kg.has_act if i < num_layers - 1 else False
            if cfg.kg.has_l2norm == 'no':
                has_l2norm = False
            elif cfg.kg.has_l2norm == 'every':
                has_l2norm = True
            elif cfg.kg.has_l2norm == 'last':
                has_l2norm = False if i < num_layers - 1 else True
            layer = layer_dict[layer_name](d_in,
                                           dim_out,
                                           act=has_act,
                                           bn=cfg.kg.has_bn,
                                           l2norm=has_l2norm,
                                           dim_edge=1)
            self.add_module('layer{}'.format(i), layer)
        self.dim_out = dim_out

    def forward(self, batch):
        if self.num_layers > 0:
            for layer in self.children():
                batch = layer(batch)
        return batch


class GNNStackStageUser(nn.Module):
    '''Simple Stage that stack GNN layers'''
    def __init__(self, dim_in, dim_out, num_layers, layer_name):
        super(GNNStackStageUser, self).__init__()
        self.num_layers = num_layers
        for i in range(num_layers):
            d_in = dim_in if i == 0 else dim_out
            has_act = cfg.kg.has_act if i < num_layers - 1 else False
            layer = GeneralLayer(layer_name,
                                 d_in,
                                 dim_out,
                                 has_act=has_act,
                                 has_bn=cfg.kg.has_bn,
                                 dim_edge=1)
            self.add_module('layer{}'.format(i), layer)
        self.dim_out = dim_out

    def forward(self, batch):
        if self.num_layers > 0:
            for layer in self.children():
                batch = layer(batch)
        return batch


class MetaLinkPool(nn.Module):
    '''Head of MetaLink, graph prediction

    The optional post_mp layer (specified by cfg.gnn.post_mp) is used
    to transform the pooled embedding using an MLP.
    '''
    def __init__(self, dim_in, dim_out):
        super(MetaLinkPool, self).__init__()
        self.layer_post_mp = MLP(dim_in,
                                 dim_out,
                                 num_layers=cfg.gnn.layers_post_mp,
                                 bias=True)
        self.pooling_fun = pooling_dict[cfg.model.graph_pooling]

    def forward(self, batch):
        if cfg.dataset.transform == 'ego':
            graph_emb = self.pooling_fun(batch.node_feature, batch.batch,
                                         batch.node_id_index)
        else:
            graph_emb = self.pooling_fun(batch.node_feature, batch.batch)
        graph_emb = self.layer_post_mp(graph_emb)
        batch.graph_feature = graph_emb
        if cfg.kg.norm_emb:
            batch.graph_feature = F.normalize(batch.graph_feature, p=2, dim=-1)
        return batch


class DirectHead(nn.Module):
    '''Directly predict label

    The optional post_mp layer (specified by cfg.gnn.post_mp) is used
    to transform the pooled embedding using an MLP.
    '''
    def __init__(self, dim_in, dim_out):
        super(DirectHead, self).__init__()
        self.layer = MLP(dim_in,
                         dim_in,
                         num_layers=cfg.kg.layers_mp,
                         bias=True)

        self.task_emb = nn.Parameter(torch.Tensor(dim_in, dim_out))
        self.task_emb.data = nn.init.xavier_uniform_(
            self.task_emb.data, gain=nn.init.calculate_gain('relu'))

    def pred(self, batch, pred):
        id_data = batch.pred_index[0, :]
        id_task = batch.pred_index[1, :]
        pred = pred[id_data].gather(1, id_task.view(-1, 1))
        return pred

    def forward(self, batch, pred_index=None):
        batch.pred_index = pred_index
        graph_feature = self.layer(batch.graph_feature)
        if cfg.kg.norm_emb:
            weight = F.normalize(self.task_emb, p=2, dim=0)
            pred = torch.matmul(graph_feature, weight)
        else:
            pred = torch.matmul(graph_feature, self.task_emb)
        pred = self.pred(batch, pred)
        return pred, graph_feature, self.task_emb


class ConcatHead(nn.Module):
    '''Directly predict label

    The optional post_mp layer (specified by cfg.gnn.post_mp) is used
    to transform the pooled embedding using an MLP.
    '''
    def __init__(self, dim_in, dim_out, dim_aux):
        super(ConcatHead, self).__init__()
        self.dim_aux = dim_aux
        self.layer = MLP(dim_in + dim_aux,
                         dim_in + dim_aux,
                         num_layers=cfg.kg.layers_mp,
                         bias=True)

        self.task_emb = nn.Parameter(torch.Tensor(dim_in + dim_aux, dim_out))
        self.task_emb.data = nn.init.xavier_uniform_(
            self.task_emb.data, gain=nn.init.calculate_gain('relu'))

    def get_aux(self, batch):
        i = torch.stack((batch.graph_targets_id_batch, batch.graph_targets_id),
                        dim=0)
        v = batch.graph_targets_value
        size = torch.Size([batch.graph_feature.shape[0], self.dim_aux])
        aux = torch.sparse.FloatTensor(i, v, size).to_dense()
        return aux

    def pred(self, batch, pred):
        id_data = batch.pred_index[0, :]
        id_task = batch.pred_index[1, :]
        pred = pred[id_data].gather(1, id_task.view(-1, 1))
        return pred

    def forward(self, batch, pred_index=None):
        batch.pred_index = pred_index
        aux = self.get_aux(batch)
        graph_feature = torch.cat((batch.graph_feature, aux), dim=1)
        graph_feature = self.layer(graph_feature)

        if cfg.kg.norm_emb:
            weight = F.normalize(self.task_emb, p=2, dim=0)
            pred = torch.matmul(graph_feature, weight)
        else:
            pred = torch.matmul(graph_feature, self.task_emb)
        pred = self.pred(batch, pred)
        return pred, graph_feature, self.task_emb


class MPHead(nn.Module):
    '''Message pass then predict label
    TODO: Heterogenous message passing

    The optional post_mp layer (specified by cfg.gnn.post_mp) is used
    to transform the pooled embedding using an MLP.
    '''
    def __init__(self, dim_in, dim_out, num_layers=1):
        super(MPHead, self).__init__()
        self.task_emb = nn.Parameter(torch.Tensor(dim_out, dim_in))
        self.task_emb.data = nn.init.xavier_uniform_(
            self.task_emb.data, gain=nn.init.calculate_gain('relu'))
        self.mp = GNNStackStageUser(dim_in,
                                    dim_in,
                                    cfg.kg.layers_mp,
                                    layer_name='kgconvv1')

    def forward(self, batch):
        batch_kg = Batch()
        # create nodes
        if cfg.kg.norm_emb:
            task_emb = F.normalize(self.task_emb, p=2, dim=1)
        else:
            task_emb = self.task_emb
        data_emb = batch.graph_feature

        batch_kg.node_feature = torch.cat((data_emb, task_emb), dim=0)

        # create edges
        n_data = data_emb.shape[0]
        # shift task node id
        graph_targets_id = batch.graph_targets_id + n_data
        edge_index = torch.stack(
            (batch.graph_targets_id_batch, graph_targets_id), dim=0)
        edge_feature = batch.graph_targets_value.unsqueeze(-1)
        batch_kg.edge_index = torch.cat(
            (edge_index, torch.flip(edge_index, [0])), dim=1)
        batch_kg.edge_feature = torch.cat((edge_feature, edge_feature), dim=0)
        batch_kg = self.mp(batch_kg)

        # pred
        data_emb = batch_kg.node_feature[:n_data]
        task_emb = batch_kg.node_feature[n_data:]
        pred = torch.matmul(data_emb, task_emb.transpose(0, 1))

        return pred, data_emb, task_emb


class MPHeteHead(nn.Module):
    '''Heterogenous Message pass then predict label

    The optional post_mp layer (specified by cfg.gnn.post_mp) is used
    to transform the pooled embedding using an MLP.
    '''
    def __init__(self, dim_in, dim_out, num_layers=1):
        super(MPHeteHead, self).__init__()
        self.task_emb = nn.Parameter(torch.Tensor(dim_out, dim_in))
        self.task_emb.data = nn.init.xavier_uniform_(
            self.task_emb.data, gain=nn.init.calculate_gain('relu'))
        self.mp = StackHeteConv(dim_in,
                                dim_in,
                                cfg.kg.layers_mp,
                                layer_name=cfg.kg.layer_type)

    def pred(self, batch):
        pred = torch.matmul(batch.node_feature['data'],
                            batch.node_feature['task'].transpose(0, 1))

        if not hasattr(batch, 'pred'):
            batch.pred = pred
        else:
            batch.pred = batch.pred + pred
        return batch

    def forward(self, batch):
        batch_kg = Batch()
        # create nodes
        if cfg.kg.has_l2norm == 'every':
            task_emb = F.normalize(self.task_emb, p=2, dim=1)
            data_emb = F.normalize(batch.graph_feature, p=2, dim=1)
        else:
            task_emb = self.task_emb
            data_emb = batch.graph_feature
        # create node
        batch_kg.node_feature = {'data': data_emb, 'task': task_emb}

        # create edge
        batch_kg.edge_index, batch_kg.edge_feature = {}, {}
        batch_kg.edge_index[('data', 'property', 'task')] = torch.stack(
            (batch.graph_targets_id_batch, batch.graph_targets_id), dim=0)
        batch_kg.edge_index[('task', 'property', 'data')] = torch.flip(
            batch_kg.edge_index[('data', 'property', 'task')], [0])
        batch_kg.edge_feature[
            'property'] = batch.graph_targets_value.unsqueeze(
                -1)  # todo: diff discrete and continous feature

        batch_kg = self.mp(batch_kg)
        batch_kg = self.pred(batch_kg)

        return batch_kg.pred, data_emb, task_emb


class MPHeteNewHead(nn.Module):
    '''Heterogenous Message pass then predict label

    The optional post_mp layer (specified by cfg.gnn.post_mp) is used
    to transform the pooled embedding using an MLP.
    '''
    def __init__(self, dim_in, dim_out, num_layers=1):
        super(MPHeteNewHead, self).__init__()
        if cfg.kg.meta:  # inductive
            self.task_emb = nn.Parameter(F.normalize(torch.ones(
                dim_out, dim_in),
                                                     p=2,
                                                     dim=1),
                                         requires_grad=False)
        else:  # transductive
            self.task_emb = nn.Parameter(torch.Tensor(dim_out, dim_in))
            self.task_emb.data = nn.init.xavier_uniform_(
                self.task_emb.data, gain=nn.init.calculate_gain('relu'))
        self.mp = StackHeteConv(dim_in,
                                dim_in,
                                cfg.kg.layers_mp,
                                layer_name=cfg.kg.layer_type)
        if cfg.kg.decode == 'concat':
            self.decode = nn.Linear(dim_in * 2, 1)

    def pred(self, batch):
        # todo: compare with index_select then sparse dot product
        if cfg.kg.decode == 'dot':
            pred = torch.matmul(batch.node_feature['data'],
                                batch.node_feature['task'].transpose(0, 1))
        elif cfg.kg.decode == 'concat':
            pred = batch.node_feature['data'].view
        id_data = batch.pred_index[0, :]
        id_task = batch.pred_index[1, :]
        pred = pred[id_data].gather(1, id_task.view(-1, 1))
        if not hasattr(batch, 'pred'):
            batch.pred = pred
        else:
            batch.pred = batch.pred + pred
        return batch

    def forward(self, batch, pred_index=None):
        batch_kg = Batch()
        # create nodes
        if cfg.kg.has_l2norm == 'every':
            task_emb = F.normalize(self.task_emb, p=2, dim=1)
            data_emb = F.normalize(batch.graph_feature, p=2, dim=1)
        else:
            task_emb = self.task_emb
            data_emb = batch.graph_feature
        # create node
        batch_kg.node_feature = {'data': data_emb, 'task': task_emb}

        # create edge
        batch_kg.edge_index, batch_kg.edge_feature = {}, {}
        batch_kg.edge_index[('data', 'property', 'task')] = torch.stack(
            (batch.graph_targets_id_batch, batch.graph_targets_id), dim=0)
        batch_kg.edge_index[('task', 'property', 'data')] = torch.flip(
            batch_kg.edge_index[('data', 'property', 'task')], [0])
        batch_kg.edge_feature[
            'property'] = batch.graph_targets_value.unsqueeze(
                -1)  # todo: diff discrete and continous feature

        batch.pred_index = pred_index
        batch_kg.pred_index = pred_index

        batch_kg = self.mp(batch_kg)
        batch_kg = self.pred(batch_kg)

        return batch_kg.pred, data_emb, task_emb


class PairHead(nn.Module):
    '''Directly predict label

    The optional post_mp layer (specified by cfg.gnn.post_mp) is used
    to transform the pooled embedding using an MLP.
    '''
    def __init__(self, dim_in, dim_out, num_layers=1):
        super(PairHead, self).__init__()
        self.layer = nn.Bilinear(dim_in, dim_in, dim_out, bias=True)

    def forward(self, batch):
        x1 = batch.graph_feature.unsqueeze(0).repeat(
            batch.graph_feature.shape[0], 1, 1)
        x2 = batch.graph_feature.unsqueeze(1).repeat(
            1, batch.graph_feature.shape[0], 1)
        pred = self.layer(x1, x2)
        return pred


class MetaLink(nn.Module):
    '''MetaLink GNN model'''
    def __init__(self, dim_in, dim_out, **kwargs):
        """
            Parameters:
            node_encoding_classes - For integer features, gives the number
            of possible integer features to map.
        """
        super(MetaLink, self).__init__()

        GNNStage = stage_dict[cfg.gnn.stage_type]

        self.preprocess = Preprocess(dim_in)
        d_in = self.preprocess.dim_out
        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(d_in, cfg.gnn.dim_inner)
            d_in = cfg.gnn.dim_inner
        if cfg.gnn.layers_mp > 1:
            self.mp = GNNStage(dim_in=d_in,
                               dim_out=cfg.gnn.dim_inner,
                               num_layers=cfg.gnn.layers_mp)
            d_in = self.mp.dim_out
        self.post_mp = MetaLinkPool(dim_in=d_in, dim_out=d_in)

        # KG mp
        head_type = cfg.kg.head

        if head_type == 'direct':
            self.head = DirectHead(dim_in=d_in, dim_out=dim_out)
        elif head_type == 'mp':
            self.head = MPHead(dim_in=d_in, dim_out=dim_out)
        elif head_type == 'mphete':
            self.head = MPHeteHead(dim_in=d_in, dim_out=dim_out)
        elif head_type == 'mphetenew':
            self.head = MPHeteNewHead(dim_in=d_in, dim_out=dim_out)
        elif head_type == 'pair':
            self.head = PairHead(dim_in=d_in, dim_out=dim_out)
        elif head_type == 'concat':  # todo: generalize dim_aux
            self.head = ConcatHead(dim_in=d_in, dim_out=dim_out, dim_aux=14)

        self.apply(init_weights)

    def kg_edges(self, n, degree=20):
        source_all = []
        target_all = []
        for i in range(n):
            source_all += [i] * degree
            target = range(i + 1, i + degree + 1)
            target_all += [node % n for node in target]
        source_all = torch.tensor(source_all)
        target_all = torch.tensor(target_all)
        edge_index = torch.stack((source_all, target_all), dim=0)
        edge_index.to(torch.device(cfg.device))

    def process_kg(self, batch, motif_emb):
        batch_new = Batch()
        batch_new.mol_emb = batch.graph_feature
        batch_new.task_emb = self.task_emb
        batch_new.motif_emb = motif_emb

        batch_new.edge_indices = {}
        batch_new.edge_features = {}

        batch_new.edge_indices['mol_task'] = torch.stack(
            (batch.graph_targets_id_batch, batch.graph_targets_id), dim=0)
        # todo: normalize feature?
        batch_new.edge_features['mol_task'] = batch.graph_targets_value

        batch_new.edge_indices['mol_motif'] = torch.stack(
            (batch.graph_motifs_id_batch, batch.graph_motifs_id), dim=0)

    def emb_motif(self, batch, motifs):
        motifs_id_all = torch.cat(
            (batch.graph_motifs_id, batch.graph_motifs_neg_id), dim=0)
        motifs_batch_all = torch.cat(
            (batch.graph_motifs_id_batch, batch.graph_motifs_neg_id_batch),
            dim=0)
        unique_id = torch.unique(motifs_id_all)
        motifs_batch = Batch.from_data_list(motifs[unique_id])
        motifs_batch.to(torch.device(cfg.device))
        motifs_emb = self.forward_compute(motifs_batch).graph_feature
        motifs_emb_scatter = torch.zeros((len(motifs), motifs_emb.shape[1]),
                                         device=torch.device(cfg.device))
        motifs_emb_scatter.index_add_(0, unique_id, motifs_emb)
        batch.motif_emb = motifs_emb_scatter
        batch.motif_id_all = motifs_id_all
        batch.motif_batch_all = motifs_batch_all

        label_pos = torch.ones((batch.graph_motifs_id.shape[0]),
                               device=torch.device(cfg.device))
        label_neg = torch.zeros((batch.graph_motifs_neg_id.shape[0]),
                                device=torch.device(cfg.device))
        label = torch.cat((label_pos, label_neg), dim=0)
        batch.motif_label = label
        return batch

    def forward_compute(self, batch):
        batch = self.preprocess(batch)
        batch = self.pre_mp(batch)
        batch = self.mp(batch)
        batch = self.post_mp(batch)

        return batch

    def forward_emb(self, batch, motifs=None):
        batch = self.forward_compute(batch)
        return batch

    def forward_pred(self, batch, **kwargs):
        batch = self.head(batch, **kwargs)
        return batch


register_network('metalink', MetaLink)
