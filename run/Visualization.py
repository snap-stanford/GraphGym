import torch
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx

class Visualize:
    def visualize_graph(model):
        # Convert the PyTorch Geometric model to a NetworkX graph
        graph = to_networkx(model, to_undirected=True)

        # Extract the edge weights from the PyTorch Geometric model
        edge_weights = model.edge_index[-1]

        # Normalize the edge weights to be between 0 and 1
        edge_weights = (edge_weights - edge_weights.min()) / (edge_weights.max() - edge_weights.min())

        # Create a dictionary that maps edges to their corresponding weights
        edge_weight_dict = dict(zip(list(zip(model.edge_index[0], model.edge_index[1])), edge_weights))

        # Create a new NetworkX graph with edge weights as attributes
        weighted_graph = nx.Graph()
        weighted_graph.add_nodes_from(graph.nodes())
        weighted_graph.add_edges_from(graph.edges())
        for edge in weighted_graph.edges():
            weighted_graph.edges[edge]['weight'] = edge_weight_dict.get(edge, 0)

        # Draw the graph using matplotlib and networkx
        pos = nx.spring_layout(weighted_graph)
        nx.draw(weighted_graph, pos, with_labels=True, node_size=100, font_size=10, width=2, edge_color=[weighted_graph[u][v]['weight'] for u, v in weighted_graph.edges()])
        plt.show()