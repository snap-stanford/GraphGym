import networkx as nx

from graphgym.register import register_feature_augment


def example_node_augmentation_func(graph, **kwargs):
    '''
    compute node clustering coefficient as feature augmentation
    :param graph: deepsnap graph. graph.G is networkx
    :param kwargs: required, in case additional kwargs are provided
    :return: List of node feature values, length equals number of nodes
    Note: these returned values are later processed and treated as node
    features as specified in "cfg.dataset.augment_feature_repr"
    '''
    return list(nx.clustering(graph.G).values())


register_feature_augment('example', example_node_augmentation_func)
