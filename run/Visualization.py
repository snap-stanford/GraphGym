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


        sampleSource = [[0]*3, [0.1]*3, [0.2]*3, [0.3]*3, [0.4]*3, [0.5]*3, [0.6]*3, [0.7]*3]
    
        print(inv_map)
        print(graph.nodes)
        

        values = []

        print(len(graph.nodes))

        for i in range(len(graph.nodes)):
            values.append(sampleSource[i])

        pos = nx.spring_layout(graph)
        nx.draw_networkx_nodes(graph, pos, cmap=plt.get_cmap('jet'), 
                            node_color = values, node_size = 1000)
        nx.draw_networkx_labels(graph, pos)
        nx.draw_networkx_edges(graph, pos, edgelist=red_edges, edge_color='r', arrows=False)
        nx.draw_networkx_edges(graph, pos, edgelist=black_edges, arrows=False)
        plt.show()

