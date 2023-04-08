import torch
from torch_geometric.data import Dataset, Data
import numpy as np 
import os

CLASS_NAME = "DISEASE"

class CytokinesDataSet(Dataset):
    def __init__(self, root, filename, graphName, test=False, transform=None, pre_transform=None, patients = None, adjacency = None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        self.test = test
        self.filename = filename
        self.patients = patients
        self.adjacency = adjacency
        self.graphName = graphName
        self.process()

        super(CytokinesDataSet, self).__init__(root, transform, pre_transform)
        
    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        return self.filename

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped. Here, always process

        fileNameArray = []
        if self.test:
            for i in range(len(self.patients)):
                fileNameArray.append(graphName[:-9] + "_data_test_" + str(i) + ".pt")
        else:
            for i in range(len(self.patients)):
                fileNameArray.append(graphName[:-9] + "_data_test_" + str(i) + ".pt")"""
        
        fileNameArray = ["dummy.csv"]
        
        return fileNameArray

    def download(self):
        pass

    def process(self):
        name = self.graphName[:-10]
        self.new_dir = "datasets\\" + name + "\\processed\\"# "datasets\\" + name + "\\processed"   # new processed dir

        # runs once for each patient
        index = 0
        for patient in self.patients:
            # Get node features of a single patient
            node_feats = self._get_node_features(patient)
            # Get edge features of a single patient
            edge_feats = self._get_edge_features()
            # Get adjacency info of a single patient
            edge_index = self._get_adjacency_info()
            # Get labels info of a single patient
            label = self._get_labels(patient[CLASS_NAME]) # todo: read label from patient data


            # Create data object
            data = Data(x=node_feats, 
                        edge_index=edge_index,
                        edge_attr=edge_feats,
                        y=label,
                        ) 
            if self.test:
                torch.save(data, 
                    os.path.join(self.new_dir, 
                                 f'{name}_data_test_{index}.pt'))
            else:
                torch.save(data, 
                    os.path.join(self.new_dir, 
                                 f'{name}_data_{index}.pt'))
            
            index += 1

    def _get_node_features(self, patient):
        """ 
        This will return a matrix / 2d array of the shape
        [Number of Nodes, Node Feature size]
        """
        all_node_feats = np.asarray(patient['data'])


        return torch.tensor(all_node_feats, dtype=torch.float)

    # todo: REturn list of 1 each time
    def _get_edge_features(self):
        """ 
        This will return a matrix / 2d array of the shape
        [Number of edges, Edge Feature size]
        """
        all_edge_feats = []
        for edge in self.adjacency:
            all_edge_feats.append(1)

        all_edge_feats = np.asarray(all_edge_feats)
        return torch.tensor(all_edge_feats, dtype=torch.float)

    # todo, same adjacency
    def _get_adjacency_info(self):
        """
        We could also use rdmolops.GetAdjacencyMatrix(mol)
        but we want to be sure that the order of the indices
        matches the order of the edge features
        """


        edge_indices = []
        for edge in self.adjacency:
            edge_indices += edge

        edge_indices = torch.tensor(edge_indices)
        edge_indices = edge_indices.t().to(torch.long).view(2, -1)
        return edge_indices

    
    def _get_labels(self, label):
        return torch.tensor(float(label), dtype=torch.int64)

    def len(self):
        return len(self.patients)

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        name = self.graphName[:-10]

        if self.test:
            data = torch.load(os.path.join(self.new_dir, 
                                 f'{name}_data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.new_dir, 
                                 f'{name}_data_{idx}.pt'))   
        return data