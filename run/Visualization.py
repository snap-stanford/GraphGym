import torch
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx

class Visualize:
    def visualize_graph(colorWeights, graph, name, edge_weights):
        nodeNameLocation = "datasets\\" + name + "\\processed\\nodeNames.pt"
        divisionsLocation = "datasets\\" + name + "\\processed\\divisions.pt"
        nodeNames = torch.load(nodeNameLocation)
        divisions = torch.load(divisionsLocation)

        print(len(edge_weights))

        print(graph.edges)
        print(edge_weights)

        largest_edge = max(edge_weights).item()
        smallest_edge = min(edge_weights).item()

        diff_edge = largest_edge - smallest_edge
        print(largest_edge)

        edge_colours = []

        for edge in edge_weights:
            color = [0]*3
            color[1] = (edge.item() - smallest_edge) / diff_edge
            edge_colours.append(color)

        print(edge_colours)


        inv_map = {v: k for k, v in nodeNames.items()}

        graph = nx.relabel_nodes(graph, inv_map)





        sampleSource = Visualize.nodeColourings(colorWeights, divisions)
        

        values = []

        for i in range(len(graph.nodes)):
            values.append(sampleSource[i])

        pos = nx.spring_layout(graph)
        nx.draw_networkx_nodes(graph, pos, cmap=plt.get_cmap('jet'), 
                            node_color = values, node_size = 1000)
        nx.draw_networkx_labels(graph, pos)
        nx.draw_networkx_edges(graph, pos, edgelist=graph.edges, edge_color=edge_colours, arrows=False)
        plt.show()
    
    """
    A function that returns how each node is to be colored. 
    colorWeights is the weights of the layer we are looking at.
    division is an array that tells us which weights belong to which node in the graph
    """
    def nodeColourings(colorWeights, division):
        # not sure if I like a simple average eqaution. 

        num_rows = len(colorWeights)
        num_cols = len(colorWeights[0])


        # stores the average value of the weight of a single gene
        averages = [None] * num_cols
        for col in range(num_cols):
            total = 0
            for row in range(num_rows):
                total += colorWeights[row][col].item()
            
            averages[col] = total / num_rows
            
        # we have an array that has the average of every gene. Now we need to average across nodes
        node_avg = []
        start = 0
        for end in division:
            total = 0
            for i in range(start, end): #average out, then print list, then return
                total += averages[i]
            average = total / (end - start)
            node_avg.append(average)
            start = end
        

        largest_avg = 0

        # find the largest average to use as scaling
        for avg in node_avg:
            largest_avg = max(largest_avg, abs(avg))
        

        sampleSource = []
        for i in range(len(node_avg)):
            color = [0]*3
            
            if(node_avg[i] > 0):
                color[0] = node_avg[i] / largest_avg
            else:
                color[2] = -1 * node_avg[i] / largest_avg

            sampleSource.append(color)





        

        return sampleSource

