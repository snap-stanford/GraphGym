import torch
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
from sklearn.manifold import TSNE

class Visualize:
    def visualize_correlations(name, graph, correlationMatrix):
        nodeNameLocation = "datasets\\" + name + "\\processed\\nodeNames.pt"
        nodeNames = torch.load(nodeNameLocation)
        graphCopy = nx.create_empty_copy(graph)
        cutoff = 0.9

        inv_map = {v: k for k, v in nodeNames.items()}
        num_nodes = len(graphCopy.nodes)


        for row in range(num_nodes):
            for col in range(row + 1, num_nodes):
                if(correlationMatrix[row][col] >= cutoff):
                    graphCopy.add_edge(row, col)


        graphCopy = nx.relabel_nodes(graphCopy, inv_map)


        pos = nx.shell_layout(graphCopy)
        nx.draw_networkx_nodes(graphCopy, pos, cmap=plt.get_cmap('jet'))
        nx.draw_networkx_labels(graphCopy, pos, font_color = "red")
        nx.draw_networkx_edges(graphCopy, pos, edgelist=graphCopy.edges)
        plt.show()
        return

    def visualize_TSNE(matrix, classification):



        # Apply t-SNE to the matrix
        tsne = TSNE(n_components=2)
        embedded_matrix = tsne.fit_transform(matrix)

        # Plot the points based on the classification array
        plt.scatter(embedded_matrix[:, 0], embedded_matrix[:, 1], c=classification)
        plt.colorbar()
        plt.show()
        return

    def visualize_graph(colorWeights, graph, name, edge_weights):
        nodeNameLocation = "datasets\\" + name + "\\processed\\nodeNames.pt"
        divisionsLocation = "datasets\\" + name + "\\processed\\divisions.pt"
        nodeNames = torch.load(nodeNameLocation)
        divisions = torch.load(divisionsLocation)

        largest_edge = max(edge_weights).item()
        smallest_edge = min(edge_weights).item()

        diff_edge = largest_edge - smallest_edge

        edge_colours = []

        for edge in edge_weights:
            color = [0]*3
            if((edge.item() - smallest_edge) / diff_edge > 0.9):
                color[1] = 1.0
            edge_colours.append(color)




        inv_map = {v: k for k, v in nodeNames.items()}

        graph = nx.relabel_nodes(graph, inv_map)
        graph = graph.to_directed()






        sampleSource = Visualize.nodeColourings(colorWeights, divisions)
        num_edges = len(graph.edges)
        
        values = []

        for i in range(len(graph.nodes)):
            values.append(sampleSource[i])

        pos = nx.shell_layout(graph)
        nx.draw_networkx_nodes(graph, pos, cmap=plt.get_cmap('jet'),  node_color = values, node_size = 1000)
        nx.draw_networkx_labels(graph, pos, font_color = "red")
        nx.draw_networkx_edges(graph, pos, edgelist=graph.edges, edge_color=edge_colours[:num_edges], node_size = 1000, arrows=True, connectionstyle='arc3, rad = 0.1')
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
                total += abs(colorWeights[row][col].item())
            
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
            
            color[2] = node_avg[i] / largest_avg

            sampleSource.append(color)





        

        return sampleSource

