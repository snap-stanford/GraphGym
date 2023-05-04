import torch
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx

class Visualize:
    def visualize_graph(model, graph, name):
        nodeNameLocation = "datasets\\" + name + "\\processed\\nodeNames.pt"
        nodeNames = torch.load(nodeNameLocation)

        inv_map = {v: k for k, v in nodeNames.items()}

        graph = nx.relabel_nodes(graph, inv_map)


        red_edges = []
        edge_colours = ['black' if not edge in red_edges else 'red'
                        for edge in graph.edges]
        black_edges = [edge for edge in graph.edges if edge not in red_edges]


        val_map = {0: 1.0,
            1 : 0.5714285714285714,
            2 : 0.0}

        values = [val_map.get(node, 0.25) for node in graph.nodes]

        pos = nx.spring_layout(graph)
        nx.draw_networkx_nodes(graph, pos, cmap=plt.get_cmap('jet'), 
                            node_color = 'r', node_size = 1000)
        nx.draw_networkx_labels(graph, pos)
        nx.draw_networkx_edges(graph, pos, edgelist=red_edges, edge_color='r', arrows=False)
        nx.draw_networkx_edges(graph, pos, edgelist=black_edges, arrows=False)
        plt.show()

