import collections
import os
import pickle

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

dirname = os.path.dirname(__file__)


def degree_dist(G):
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())

    fig, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.80, color='b')

    plt.title("Degree Histogram")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    ax.set_xticks([d + 0.4 for d in deg])
    ax.set_xticklabels(deg)
    plt.show()


def save_syn():
    clustering_bins = np.linspace(0.3, 0.6, 7)
    print(clustering_bins)
    path_bins = np.linspace(1.8, 3.0, 7)
    print(path_bins)

    powerlaw_k = np.arange(2, 12)
    powerlaw_p = np.linspace(0, 1, 101)
    ws_k = np.arange(4, 23, 2)
    ws_p = np.linspace(0, 1, 101)

    counts = np.zeros((8, 8))
    thresh = 4
    graphs = []
    n = 64
    while True:
        k, p = np.random.choice(powerlaw_k), np.random.choice(powerlaw_p)
        g = nx.powerlaw_cluster_graph(n, k, p)
        clustering = nx.average_clustering(g)
        path = nx.average_shortest_path_length(g)
        clustering_id = np.digitize(clustering, clustering_bins)
        path_id = np.digitize(path, path_bins)
        if counts[clustering_id, path_id] < thresh:
            counts[clustering_id, path_id] += 1
            default_feature = torch.ones(1)
            nx.set_node_attributes(g, default_feature, 'node_feature')
            graphs.append(g)
            print(np.sum(counts))
        if np.sum(counts) == 8 * 8 * thresh:
            break

    with open('scalefree.pkl', 'wb') as file:
        pickle.dump(graphs, file)

    counts = np.zeros((8, 8))
    thresh = 4
    graphs = []
    n = 64
    while True:
        k, p = np.random.choice(ws_k), np.random.choice(ws_p)
        g = nx.watts_strogatz_graph(n, k, p)
        clustering = nx.average_clustering(g)
        path = nx.average_shortest_path_length(g)
        clustering_id = np.digitize(clustering, clustering_bins)
        path_id = np.digitize(path, path_bins)
        if counts[clustering_id, path_id] < thresh:
            counts[clustering_id, path_id] += 1
            default_feature = torch.ones(1)
            nx.set_node_attributes(g, default_feature, 'node_feature')
            graphs.append(g)
            print(np.sum(counts))
        if np.sum(counts) == 8 * 8 * thresh:
            break

    with open('smallworld.pkl', 'wb') as file:
        pickle.dump(graphs, file)


def load_syn():
    with open('{}/smallworld.pkl'.format(dirname), 'rb') as file:
        graphs = pickle.load(file)
    for graph in graphs:
        print(nx.average_clustering(graph),
              nx.average_shortest_path_length(graph))
        # degree_dist(graph)
