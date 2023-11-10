import torch
from torch_geometric.data import Dataset, Data
import numpy as np 
import os

CLASS_NAME = "DISEASE"

class CytokinesDataSet(Dataset):
    def __init__(self, root, filename, graphName, test=False, transform=None, pre_transform=None, patients = None, adjacency = None, nodeNames = None, divisions = None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """

        self.test = test
        self.filename = filename
        self.patients = patients
        self.adjacency = adjacency
        self.graphName = graphName
        self.nodeNames = nodeNames
        self.divisions = divisions # 
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
        name = self.graphName
        print(self.graphName)
        self.new_dir = os.path.join("datasets", name, "processed")# "datasets\\" + name + "\\processed"   # new processed dir
        train_data = Data()
        train_slices = dict()
        test_data = Data()
        test_slices = dict()
        all_data = Data()
        all_slices = dict()
        train_tuple = (train_data, train_slices)
        test_tuple = (test_data, test_slices)
        all_tuple = (all_data, all_slices)

        num_patients = len(self.patients)
        num_training = int(0.8*num_patients) # when index <= num_training, the patient goes into training data. Else, testing
        num_testing = num_patients - num_training
        num_nodes = len(self._get_node_features(self.patients[0]))
        num_edges = len(self.adjacency)
        
        
        # prepare the slices
        train_y_slice = []
        train_x_slice = []
        train_edge_index_slice = []

        test_y_slice = []
        test_x_slice = []
        test_edge_index_slice = []

        all_y_slice = []
        all_x_slice = []
        all_edge_index_slice = []

        for i in range(num_training + 1):
            train_y_slice.append(i)
            train_x_slice.append(num_nodes*i)
            train_edge_index_slice.append(num_edges * i)

        for i in range(num_testing + 1):
            test_y_slice.append(i)
            test_x_slice.append(num_nodes*i)
            test_edge_index_slice.append(num_edges * i)

        for i in range(num_patients + 1):
            all_y_slice.append(i)
            all_x_slice.append(num_nodes*i)
            all_edge_index_slice.append(num_edges * i)
        
        train_slices['y'] = torch.tensor(train_y_slice)
        train_slices['x'] = torch.tensor(train_x_slice)
        train_slices['edge_index'] = torch.tensor(train_edge_index_slice)

        test_slices['y'] = torch.tensor(test_y_slice)
        test_slices['x'] = torch.tensor(test_x_slice)
        test_slices['edge_index'] = torch.tensor(test_edge_index_slice)


        all_slices['y'] = torch.tensor(all_y_slice)
        all_slices['x'] = torch.tensor(all_x_slice)
        all_slices['edge_index'] = torch.tensor(all_edge_index_slice)

        
        node_vector_len = len(self._get_node_features(self.patients[0]).numpy()[0]) # the length of a node vector
        # preparing data
        train_x_tensor = torch.empty((0, node_vector_len), dtype=torch.float32)
        train_edge_index_tensor = torch.empty((2, 0), dtype=torch.int32)
        train_y_list = [] # It's just easier to make a list than figure out how to do 1d concats

        test_x_tensor = torch.empty((0, node_vector_len), dtype=torch.float32)
        test_edge_index_tensor = torch.empty((2, 0), dtype=torch.int32)
        test_y_list = [] # It's just easier to make a list than figure out how to do 1d concats


        all_x_tensor = torch.empty((0, node_vector_len), dtype=torch.float32)
        all_edge_index_tensor = torch.empty((2, 0), dtype=torch.int32)
        all_y_list = [] # It's just easier to make a list than figure out how to do 1d concats


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
            label = self._get_labels(patient[CLASS_NAME])

            if index <= num_training:
                train_x_tensor = torch.cat((train_x_tensor,node_feats), 0)
                train_edge_index_tensor = torch.cat((train_edge_index_tensor,edge_index), 1)
                train_y_list.append(int(patient[CLASS_NAME]))
            else:
                test_x_tensor = torch.cat((test_x_tensor,node_feats), 0)
                test_edge_index_tensor = torch.cat((test_edge_index_tensor,edge_index), 1)
                test_y_list.append(int(patient[CLASS_NAME]))
            
            all_x_tensor = torch.cat((all_x_tensor,node_feats), 0)
            all_edge_index_tensor = torch.cat((all_edge_index_tensor,edge_index), 1)
            all_y_list.append(int(patient[CLASS_NAME]))

            index += 1

        # turn y_lists into tensors
        train_y_tensor = torch.tensor(train_y_list)
        test_y_tensor = torch.tensor(test_y_list)
        all_y_tensor = torch.tensor(all_y_list)

        train_data.x = train_x_tensor
        train_data.y = train_y_tensor
        train_data.edge_index = train_edge_index_tensor


        test_data.x = test_x_tensor
        test_data.y = test_y_tensor
        test_data.edge_index = test_edge_index_tensor


        all_data.x = all_x_tensor
        all_data.y = all_y_tensor
        all_data.edge_index = all_edge_index_tensor

        torch.save(self.divisions, os.path.join(self.new_dir, 'divisions.pt')) # states which genes belong to which cytokine
        torch.save(self.nodeNames, os.path.join(self.new_dir, 'nodeNames.pt'))
        torch.save(train_tuple, os.path.join(self.new_dir, 'train_data.pt'))
        torch.save(test_tuple, os.path.join(self.new_dir, 'test_data.pt'))
        torch.save(all_tuple, os.path.join(self.new_dir, 'all_data.pt'))


        # last step, just save everything in the right .pt files
        """
                return ['train_data.pt', 'test_data.pt']

                        torch.save(data, 
                    os.path.join(self.processed_dir, 
                                 f'{name}_data_{index}.pt'))
        """

        

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
            edge_indices.append(edge[0])

        for edge in self.adjacency:
            edge_indices.append(edge[1])

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