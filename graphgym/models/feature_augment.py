import logging

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from deepsnap.graph import Graph

import graphgym.register as register
from graphgym.config import cfg
from graphgym.contrib.transform.identity import compute_identity


def _key(key, as_label=False):
    return key + '_label' if as_label else key


def create_augment_fun(graph_fun, key):
    def augment_fun(graph, as_label, **kwargs):
        graph[_key(key, as_label=as_label)] = graph_fun(graph, **kwargs)

    return augment_fun


def _replace_label(graph):
    if cfg.dataset.augment_label:
        if not cfg.dataset.augment_label_dims:
            raise ValueError('dataset.label_dims should have the same '
                             'length as dataset.label')
        label_key = _key(cfg.dataset.augment_label, as_label=True)
        label = graph[label_key]
        if cfg.dataset.task == 'node':
            graph.node_label = label
        elif cfg.dataset.task == 'edge' or cfg.dataset.task == 'link_pred':
            graph.edge_label = label
        elif cfg.dataset.task == 'graph':
            graph.graph_label = label
        else:
            raise ValueError('Unknown task type: {}'.format(cfg.dataset.task))


# feature preprocessing
class FeatureAugment(nn.Module):
    def __init__(self):
        super(FeatureAugment, self).__init__()

        # pre-defined feature functions
        def degree_fun(graph, **kwargs):
            # TODO: if bin_edges is None, do regression
            return [d for _, d in graph.G.degree()]

        def centrality_fun(graph, **kwargs):
            # TODO multiple feature dimensions
            centrality = nx.betweenness_centrality(graph.G)
            return [centrality[x] for x in graph.G.nodes]

        def path_len_fun(graph, **kwargs):
            return [
                np.mean(
                    list(nx.shortest_path_length(graph.G, source=x).values()))
                for x in graph.G.nodes
            ]

        def edge_path_len_fun(graph, **kwargs):
            return [
                np.mean(
                    list(nx.shortest_path_length(graph.G, source=x).values()))
                for x in graph.G.nodes
            ]

        def pagerank_fun(graph, **kwargs):
            # TODO multiple feature dimensions
            pagerank = nx.pagerank(graph.G)
            return [pagerank[x] for x in graph.G.nodes]

        def identity_fun(graph, **kwargs):
            if 'feature_dim' not in kwargs:
                raise ValueError('Argument feature_dim not supplied')
            return compute_identity(graph.edge_index, graph.num_nodes,
                                    kwargs['feature_dim'])

        def clustering_coefficient_fun(graph, **kwargs):
            return list(nx.clustering(graph.G).values())

        def const_fun(graph, **kwargs):
            # directly return the torch tensor representation
            return torch.ones(graph.num_nodes)

        def onehot_fun(graph, **kwargs):
            # directly return the torch tensor representation
            # return torch.arange(graph.num_nodes)
            return torch.randperm(graph.num_nodes)

        def graph_laplacian_spectrum_fun(graph, **kwargs):
            # first eigenvalue is 0
            spectrum = nx.laplacian_spectrum(graph.G)[1:]
            feature_dim = kwargs['feature_dim']
            if len(spectrum) > feature_dim:
                spectrum = spectrum[:feature_dim]
            return torch.tensor(spectrum)

        def graph_path_len_fun(graph, **kwargs):
            path = nx.average_shortest_path_length(graph.G)
            return torch.tensor([path])

        def graph_clustering_fun(graph, **kwargs):
            clustering = nx.average_clustering(graph.G)
            return torch.tensor([clustering])

        self.feature_dict = {
            'node_degree': degree_fun,
            'node_betweenness_centrality': centrality_fun,
            'node_path_len': path_len_fun,
            'node_pagerank': pagerank_fun,
            'node_clustering_coefficient': clustering_coefficient_fun,
            'node_identity': identity_fun,
            'node_const': const_fun,
            'node_onehot': onehot_fun,
            'edge_path_len': edge_path_len_fun,
            'graph_laplacian_spectrum': graph_laplacian_spectrum_fun,
            'graph_path_len': graph_path_len_fun,
            'graph_clustering_coefficient': graph_clustering_fun
        }

        self.feature_dict = {
            **register.feature_augment_dict,
            **self.feature_dict
        }

        for key, fun in self.feature_dict.items():
            self.feature_dict[key] = create_augment_fun(
                self.feature_dict[key], key)

    def register_feature_fun(self, name, feature_fun):
        self.feature_dict[name] = create_augment_fun(feature_fun, name)

    @staticmethod
    def _bin_features(graph, key, bin_edges, feature_dim=2, as_label=False):
        """ Used as an apply_transform function to convert temporary node
        features
        into binned node features.
        """
        arr = np.array(graph[key])
        feat = np.digitize(arr, bin_edges) - 1
        assert np.min(feat) >= 0
        assert np.max(feat) <= feature_dim - 1
        graph[key] = FeatureAugment._one_hot_tensor(feat,
                                                    one_hot_dim=feature_dim,
                                                    as_label=as_label)

    @staticmethod
    def _one_hot_tensor(vals, one_hot_dim=1, as_label=False):
        if not isinstance(vals, (torch.Tensor, np.ndarray)):
            raise ValueError(
                "Input to _one_hot_tensor must be tensor or np array.")
        if not vals.ndim == 1:
            raise ValueError("Input to _one_hot_tensor must be 1-D.")
        vals = torch.tensor(vals)
        if as_label:
            if vals.ndim > 1:
                return vals.squeeze()
            else:
                return vals
        one_hot = torch.zeros(len(vals), one_hot_dim)
        one_hot.scatter_(1, vals.unsqueeze(-1), 1.0)
        return one_hot

    @staticmethod
    def _orig_features(graph, key):
        # repeat the feature to have length feature_dims
        if not isinstance(graph[key], torch.Tensor):
            graph[key] = torch.tensor(graph[key])
        if cfg.dataset.task_type == 'regression' and 'label' in key:
            graph[key] = graph[key].float()
        assert graph[key].ndim <= 2
        if graph[key].ndim == 1 and Graph._is_node_attribute(key):
            # n-by-1 tensor for node attributes
            graph[key] = graph[key].unsqueeze(-1)

    @staticmethod
    def _position_features(graph,
                           key,
                           feature_dim=4,
                           scale=1,
                           wavelength=10000):
        # pos = torch.tensor(graph[key]).float()
        if isinstance(graph[key], torch.Tensor):
            pos = graph[key].float()
        else:
            pos = torch.tensor(graph[key]).float()
        assert pos.ndim <= 2
        if pos.ndim == 1:
            pos = pos.unsqueeze(-1)
        batch_size, n_feats = pos.shape
        pos = pos.view(-1)
        pos *= scale

        cycle_range = torch.arange(
            0, feature_dim // 2).float() / (feature_dim // 2)
        sins = torch.sin(
            pos.unsqueeze(-1) / wavelength**cycle_range.unsqueeze(0))
        coss = torch.cos(
            pos.unsqueeze(-1) / wavelength**cycle_range.unsqueeze(0))
        # dim-by-2
        m = torch.cat((coss, sins), dim=-1)
        graph[key] = m.view(batch_size, -1)

    @staticmethod
    def _get_max_value(dataset, feature_key):
        list_scalars = np.concatenate([g[feature_key] for g in dataset])
        return np.max(list_scalars)

    @staticmethod
    def _get_bin_edges(dataset, feature_key, feature_dim, bin_method):
        """
        get bin edges for a particular feature_key (e.g. node_degree)
        TODO: maybe sample for efficiency

        TODO: currently doing in numpy
        pytorch might support bucketization in future
        https://github.com/pytorch/pytorch/pull/34577
        """
        list_scalars = np.concatenate([g[feature_key] for g in dataset])
        if bin_method == 'balanced':
            # bin by ensuring that the labels are roughly
            # balanced (except ties)
            arr = np.array(list_scalars)
            sorted_arr = np.sort(list_scalars)
            bin_indices = np.linspace(0,
                                      len(list_scalars),
                                      num=feature_dim,
                                      endpoint=False).astype(int)
            bins = sorted_arr[bin_indices]
            unique_bins = np.unique(bins)
            if len(unique_bins) < len(bins):
                bins = unique_bins
                feature_dim = len(unique_bins)
                logging.warning('{} dimensions are collapsed due to balanced'
                                ' binning'.format(feature_key))

        elif bin_method == 'equal_width':
            # use bins of equal size
            arr = np.array(list_scalars)
            min_val, max_val = np.min(arr), np.max(arr)
            bins = np.linspace(min_val, max_val, num=feature_dim)

        elif bin_method == 'bounded':
            # (only for integer features) upper-bound the values, and use bin
            # of width 1
            bins = np.arange(feature_dim)
        else:
            raise ValueError('Bin method {} not supported'.format(bin_method))
        return bins

    def _augment_feature(self,
                         dataset,
                         features,
                         feature_dims,
                         as_label=False):
        # The feature dims actually used might differ from user specified
        # feature_dims.
        # This could be due to balanced binning on integers where
        # different bin edges have the same value
        if as_label:
            repr_method = 'balanced' if 'classification' in \
                                        cfg.dataset.task_type else 'original'
        else:
            repr_method = cfg.dataset.augment_feature_repr
        actual_feat_dims = []
        for key, dim in zip(features, feature_dims):
            feat_fun = self.feature_dict[key]
            key = key + '_label' if as_label else key
            if key not in dataset[0]:
                # compute (raw) features
                dataset.apply_transform(feat_fun,
                                        update_graph=False,
                                        update_tensor=False,
                                        as_label=as_label,
                                        feature_dim=dim)
                # feat = dataset[0][key]
                if repr_method == 'original':
                    # use the original feature as is
                    # this ignores the specified config feature_dims
                    dataset.apply_transform(FeatureAugment._orig_features,
                                            update_graph=True,
                                            update_tensor=False,
                                            key=key)
                elif repr_method == 'position':
                    # positional encoding similar to that of transformer
                    scale = dim / 2 / FeatureAugment._get_max_value(
                        dataset, key)
                    dataset.apply_transform(FeatureAugment._position_features,
                                            update_graph=True,
                                            update_tensor=False,
                                            key=key,
                                            feature_dim=dim,
                                            scale=scale)
                else:
                    # Bin edges for one-hot (repr_method = balanced, bounded,
                    # equal_width)
                    # use all features in dataset for computing bin edges
                    bin_edges = self._get_bin_edges(dataset, key, dim,
                                                    repr_method)
                    # apply binning
                    dataset.apply_transform(FeatureAugment._bin_features,
                                            update_graph=True,
                                            update_tensor=False,
                                            key=key,
                                            bin_edges=bin_edges,
                                            feature_dim=len(bin_edges),
                                            as_label=as_label)
            actual_feat_dims.append(dataset[0].get_num_dims(key,
                                                            as_label=as_label))
        return actual_feat_dims

    def augment(self, dataset):
        actual_feat_dims = self._augment_feature(
            dataset, cfg.dataset.augment_feature,
            cfg.dataset.augment_feature_dims)
        if cfg.dataset.augment_label:
            # Currently only support 1 label
            actual_label_dim = self._augment_feature(
                dataset, [cfg.dataset.augment_label],
                [cfg.dataset.augment_label_dims],
                as_label=True)[0]
        else:
            actual_label_dim = None
        return actual_feat_dims, actual_label_dim


class Preprocess(nn.Module):
    def __init__(self, dim_in):
        super(Preprocess, self).__init__()
        self.dim_dict = {
            name: dim
            for name, dim in zip(cfg.dataset.augment_feature,
                                 cfg.dataset.augment_feature_dims)
        }
        self.dim_dict['node_feature'] = dim_in
        self.dim_out = sum(self.dim_dict.values())

    def extra_repr(self):
        repr_str = '\n'.join([
            '{}: dim_out={}'.format(name, dim)
            for name, dim in self.dim_dict.items()
        ] + ['Total: dim_out={}'.format(self.dim_out)])
        return repr_str

    def forward(self, batch):
        batch.node_feature = torch.cat(
            [batch[name].float() for name in self.dim_dict], dim=1)
        return batch
