import logging
import os

import torch
from torch_geometric import seed_everything
from deepsnap.dataset import GraphDataset
from graphgym.custom_dataset import custom_dataset
from graphgym.cmd_args import parse_args
from graphgym.config import cfg, dump_cfg, load_cfg, set_run_dir, set_out_dir
from graphgym.loader import create_dataset, create_local_dataset, create_loader
from graphgym.logger import create_logger, setup_printing
from graphgym.model_builder import create_model
from graphgym.optimizer import create_optimizer, create_scheduler
from graphgym.register import train_dict
from graphgym.train import train
from graphgym.utils.agg_runs import agg_runs
from graphgym.utils.comp_budget import params_count
from graphgym.utils.device import auto_select_device
from graphgym.models.gnn import GNNStackStage
from iCYPRESS.CytokinesDataSet import CytokinesDataSet
from iCYPRESS.Visualization import Visualize
from graphgym.models.layer import GeneralMultiLayer, Linear, GeneralConv
from graphgym.models.gnn import GNNStackStage
import numpy as np


class Cypress:
    def __init__(self, patients = None, eset = None, blood_only=True, active_cyto_list = ['CCL1'], batch_size = 80, 
                 eval_period = 20, layers_pre_mp = 2, layers_mp = 6, layers_post_mp = 2, 
                 dim_inner = 137, max_epoch = 400):
        
        if eset is None:
            self.makeExistingConfigFile()
            self.custom = False
            return

        self.custom = True
        # get current config file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        curr_cfg_file = os.path.join(current_dir, "configs", "example_custom.yaml")
        curr_cfg_gen = os.path.join(current_dir, "configs_gen.py")
        self.write_lines_to_file(curr_cfg_file, os.path.join( "configs", "example_custom.yaml"))
        self.write_lines_to_file(curr_cfg_gen, "configs_gen.py")

        self.current_dir = os.path.dirname(os.path.abspath(__file__))

        self.eset = eset

        patient_dict, patient_list = self.process_patients(patients) # a dict that matches a patient name to their classification

        
        cyto_list,cyto_adjacency_dict,cyto_tissue_dict  = self.process_graphs(blood_only) # list of cytokines, maps a cytokine's name to their adjacency matrix, maps a cytokine's name to the tissues they need


        tissue_gene_dict, gene_set = self.process_tissues(blood_only) # dict that matches tissues to the genes associated with them, a set of all genes we have


        gene_to_patient, active_tissue_gene_dict = self.process_eset(eset, gene_set, patient_dict, tissue_gene_dict) # 2 layer deep dict. First layer maps gene name to a dict. Second layer matches patient code to gene expresion data of the given gene.
        

        # now convert these to vectors like needed
        # probably best to only do this for ONE cytokine at a time? Yeah, accept a gene, then move.
        # also, make a function that prints out everu available cytokine.
        for cyto in active_cyto_list:
            if cyto not in cyto_tissue_dict.keys():
                raise(ValueError("{} is not a valid cytokine.".format(cyto)))
            self.create_cyto_database(cyto, eset, cyto_tissue_dict, active_tissue_gene_dict, patient_list, 
                                     patient_dict, gene_to_patient, cyto_adjacency_dict)
           
            name = cyto + "_" + eset[:eset.index(".")]  
            self.name = name
            self.makeConfigFile(name, batch_size, eval_period, layers_pre_mp, layers_mp, 
                       layers_post_mp, dim_inner, max_epoch)
        
            self.active_cyto_list = active_cyto_list

    def makeExistingConfigFile(self):
        if (not os.path.exists(os.path.abspath("configs"))):
            os.makedirs(os.path.abspath("configs"))
        
        # get current config file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        curr_cfg_file = os.path.join(current_dir, "configs", "example_custom.yaml")
        self.write_lines_to_file(curr_cfg_file, os.path.join("configs", "example_custom.yaml"))

    def make_bash_scripts(self, cyto):
        scipt_name = cyto + "_" + self.eset[:self.eset.index(".")]

        # check if parallel.sh exists. If it doesn't create it.

        # generate the sh file for this particular cytokine and GSE

        # run the bash script

    def create_cyto_database(self, cyto, eset, cyto_tissue_dict, active_tissue_gene_dict, patient_list, 
                             patient_dict, gene_to_patient, cyto_adjacency_dict):

        # creates graphname
        graphName = cyto + "_" + eset[:eset.index(".")]

        #create patientArray
        patientArray = []

        # count the number of active genes in each tissue.
        tissues = cyto_tissue_dict[cyto]

        gene_count = []
        for tissue in tissues:
            count = len(active_tissue_gene_dict[tissue])
            
            gene_count.append(count)
        
        total_genes = sum(gene_count)


        for patient in patient_list:
            patient_data = {}

            patient_data["DISEASE"] = str(patient_dict[patient])


            data = []
            # create the information that goes into each node
            for i, tissue in enumerate(tissues):
                tissue_data = [0]*total_genes
                start = sum(gene_count[:i])

                tissue_genes = active_tissue_gene_dict[tissue]
                
                offset = 0
                for gene in tissue_genes:                        
                    tissue_data[start + offset] = gene_to_patient[gene][patient] / 20
                    offset +=  1
                
                data.append(tissue_data)

            patient_data["data"] = data
            patientArray.append(patient_data)

                
        nodeList = cyto_tissue_dict[cyto]
        graphAdjacency = cyto_adjacency_dict[cyto]

        nodeIntMap = {}
        i = 0

        for node in nodeList:
            nodeIntMap[node] = i
            i += 1

        intAdjacency = []
        # turn the adjacency names into int
        for edge in graphAdjacency:
            newEdge = [nodeIntMap[edge[0]], nodeIntMap[edge[1]]]
            intAdjacency.append(newEdge)

        try:
            os.mkdir(os.path.join("datasets"))
        except OSError:
            pass
    
        try:
            os.mkdir(os.path.join("datasets", graphName))
        except OSError:
            pass

        try:
            os.mkdir(os.path.join("datasets", graphName ,"raw"))
        except OSError:
            pass

        try:
            os.mkdir(os.path.join("datasets", graphName, "processed"))
        except OSError:
            pass
            
        full_dataset = CytokinesDataSet(root="data/", graphName=graphName, filename="full.csv", 
                                        test=True, patients=patientArray, adjacency=intAdjacency, 
                                        nodeNames = nodeIntMap, divisions = gene_count)
        

        torch.save(full_dataset, (os.path.join("datasets", graphName, "raw",  graphName + ".pt")))



    def process_eset(self, eset, gene_set, patient_dict, tissue_gene_dict):
        eset_file = open(eset, 'r')

        eset_lines = eset_file.read().splitlines()

        
        # read the first line, and see if it matches with the patient file provided
        patients = eset_lines[0].replace("\"", "").split(",")[2:]
        
        patient_set = set(patient_dict.keys())

        for patient in patients:
            try:
                patient_set.remove(patient)
            except(KeyError):
                raise(ValueError("{} is not found in the patients file.".format(patient)))


        if (len(patient_set) != 0):
            raise(ValueError("The eset file does not contain {}".format(patient_set)))


        gene_to_patient = dict() # maps the name of a gene to a dict of patients
        for line_num in range(1, len(eset_lines)):
            line = eset_lines[line_num].replace("\"", "")
            parts = line.split(",")
            new_gene = parts[1]


            if (new_gene not in gene_set):
                continue
            # get all the gene expression numbers, and then normalize them
            gene_nums = parts[2:]
            gene_nums = [float(gene_num) for gene_num in gene_nums]
            gene_nums = self.normalize_vector(gene_nums)


            
            patient_gene_data_dict = dict() # maps the patients code to their gene expression data of this one specific gene
            for index, patient in enumerate(patients):
                patient_gene_data_dict[patient] = gene_nums[index]

            gene_to_patient[new_gene] = patient_gene_data_dict
            
        # make a new tissue_gene_dict

        active_tissue_gene_dict = dict()

        for tissue in tissue_gene_dict.keys():
                gene_array = []

                original_genes = tissue_gene_dict[tissue]

                for gene in original_genes:
                    if gene in gene_to_patient.keys():
                        gene_array.append(gene)
                
                active_tissue_gene_dict[tissue] = gene_array
        return (gene_to_patient, active_tissue_gene_dict)

    def normalize_vector(self, vector):
        min_val = np.min(vector)
        max_val = np.max(vector)
        normalized_vector = (vector - min_val) / (max_val - min_val)
        return normalized_vector    
    
    def process_patients(self, patients):
        patient_file = open(patients, 'r')
        patient_dict = dict()
        patient_list = []

        for line in patient_file.read().splitlines():
            parts = line.split(",")
            patient_dict[parts[0]] = int(parts[1])
            patient_list.append(parts[0])
        
        return patient_dict, patient_list
        

    def process_graphs(self, blood_only):
        if(blood_only):
            graph_folder_path = os.path.join(self.current_dir,"Modified Graphs")
        else:
            graph_folder_path = os.path.join(self.current_dir,"Graphs")
        
        cyto_list = []
        cyto_adjacency_dict = dict() # maps a cytokine's name to their adjacency matrix
        cyto_tissue_dict = dict() # maps a cytokine's name to the tissues they need

        for filename in os.listdir(graph_folder_path):
            cyto_name = filename[:-10]
            cyto_list.append(cyto_name) # drop the _graph.csv
            graph_file_path = os.path.join(graph_folder_path, filename)
            if (filename == '__pycache__'):
                continue
            graphAdjacency = []
            tissue_set = set()

            f = open(graph_file_path, 'r')
            graphLines = f.read().splitlines()
            
            for line in graphLines:
                parts = line.upper().split(",") # remove newline, capitalize, and remove spaces
                graphAdjacency.append(parts)
                newParts = [parts[1], parts[0]]
                tissue_set.update(newParts)

                graphAdjacency.append(newParts)
            

            # put the tissues into a list, and then sort them
            tissue_list = []
            for tissue in tissue_set:
                tissue_list.append(tissue)
            
            tissue_list.sort()

            cyto_adjacency_dict[cyto_name] = graphAdjacency
            cyto_tissue_dict[cyto_name] = tissue_list


        return cyto_list,cyto_adjacency_dict,cyto_tissue_dict

    def process_tissues(self, blood_only):
        tissue_gene_dict = dict() # maps tissues to the genes associated with them

        gene_set = set()

        tissue_file = open(os.path.join(self.current_dir, "GenesToTissues.csv"))

        tissue_lines = tissue_file.read().splitlines()

        for i in range(0, len(tissue_lines), 2):
            tissue_line = tissue_lines[i]
            
            tissue_line_arr = tissue_line.split(",")

            if (blood_only and (tissue_line_arr[1] == "N")):
                continue

            tissue = tissue_line_arr[0]
            genes_array = tissue_lines[i + 1].split(',')

            tissue_gene_dict[tissue] = genes_array

            gene_set.update(genes_array)

        return tissue_gene_dict, gene_set
    
    def makeGridFile(self, name, layers_pre_mp = [1,2], layers_mp =[2,4,6,8], layers_post_mp = [2,3],
                     agg_type = ['add','mean']) :
        if (not os.path.exists(os.path.abspath("grids"))):
            os.makedirs(os.path.abspath("grids"))

        with open(os.path.join('grids', name + ".txt"), 'w') as file:
            file.write("# Format for each row: name in config.py; alias; range to search\n")
            file.write("# No spaces, except between these 3 sfields\n")
            file.write("# Line breaks are used to union different grid search spaces\n")
            file.write("# Feel free to add '#' to add comments\n")
            file.write("\n")
            file.write("# (1) dataset configurations\n")
            file.write("dataset.format format ['PyG']\n")
            file.write("dataset.name dataset ['" + name + "']\n")
            file.write("dataset.task task ['graph']\n")
            file.write("dataset.transductive trans [False]\n")
            file.write("dataset.augment_feature feature [[]]\n")
            file.write("dataset.augment_label label ['']\n")
            file.write("# (2) The recommended GNN design space, 96 models in total\n")
            file.write("gnn.layers_pre_mp l_pre " + str(layers_pre_mp) + "\n")
            file.write("gnn.layers_mp l_mp " + str(layers_mp) + "\n")
            file.write("gnn.layers_post_mp l_post " + str(layers_post_mp) + "\n")
            file.write("gnn.stage_type stage ['skipsum','skipconcat']\n")
            file.write("gnn.agg agg " + str(agg_type))


    def makeConfigFile(self, name, batch_size, eval_period, layers_pre_mp, layers_mp, 
                       layers_post_mp, dim_inner, max_epoch):
        if (not os.path.exists(os.path.abspath("configs"))):
            os.makedirs(os.path.abspath("configs"))

        # using with statement
        with open(os.path.join('configs', name + ".yaml"), 'w') as file:
            file.write('out_dir: results\n')
            file.write('dataset:\n')
            file.write(' format: PyG\n')
            file.write(' name: ' + name + '\n')
            file.write(' task: graph\n')
            file.write(' task_type: classification\n')
            file.write(' transductive: True\n')
            file.write(' split: [0.8, 0.2]\n')
            file.write(' augment_feature: []\n')
            file.write(' augment_feature_dims: [0]\n')
            file.write(' augment_feature_repr: position\n')
            file.write(' augment_label: \'\'\n')
            file.write(' augment_label_dims: 0\n')
            file.write(' transform: none\n')
            file.write('train:\n')
            file.write(' batch_size: ' + str(batch_size) + '\n' )
            file.write(' eval_period: ' + str(eval_period) + '\n')
            file.write(' ckpt_period: 100\n')
            file.write('model:\n')
            file.write(' type: gnn\n')
            file.write(' loss_fun: cross_entropy\n')
            file.write(' edge_decoding: dot\n')
            file.write(' graph_pooling: add\n')
            file.write('gnn:\n')
            file.write(' layers_pre_mp: ' + str(layers_pre_mp) + '\n')
            file.write(' layers_mp: ' + str(layers_mp) + '\n')
            file.write(' layers_post_mp: ' + str(layers_post_mp) + '\n')
            file.write(' dim_inner: ' + str(dim_inner) + '\n')
            file.write(' layer_type: generalconv\n')
            file.write(' stage_type: skipsum\n')
            file.write(' batchnorm: True\n')
            file.write(' act: prelu\n')
            file.write(' dropout: 0.0\n')
            file.write(' agg: add\n')
            file.write(' normalize_adj: False\n')
            file.write('optim:\n')
            file.write(' optimizer: adam\n')
            file.write(' base_lr: 0.01\n')
            file.write(' max_epoch: ' +  str(max_epoch) + '\n')


    def write_lines_to_file(self, input_file, output_file_name):
        try:
            with open(input_file, 'r') as infile:
                # Read all lines from the input file
                lines = infile.readlines()

            # Remove any leading/trailing whitespaces from the output file name
            output_file_name = output_file_name.strip()

            # Write the lines to the output file with the provided name (overwriting if it exists)
            with open(output_file_name, 'w') as outfile:
                for line in lines:
                    outfile.write(line)

        except FileNotFoundError:
            print(f"Error: Input file '{input_file}' not found.")
        except Exception as e:
            print(f"Error: An unexpected error occurred - {e}")
    
    def parallel(self, cyto = None):
        self.makeGridFile(self.name)
        fake_sh_file = self.make_fake_sh(self.name)

        try:
            os.mkdir(os.path.join("results"))
            print(os.path.join("results"))
            print("here")
            1/0
        except OSError:
            pass

        
        os.system(fake_sh_file)
        return

    def make_fake_sh(self, name):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parallel_location = os.path.join(current_dir, "parallel.sh")
        agg_batch_location = os.path.join(current_dir, "agg_batch.py")
        max_jobs = 3
        repeat = 1



        try:
            os.mkdir(os.path.join("results", name))
        except OSError:
            pass

        fake_sh_file = (
            "set CONFIG="
            + name
            + " && set GRID="
            + name
            + " && set REPEAT=1 && set MAX_JOBS=3 && python configs_gen.py --config configs/" + name + ".yaml "
            "--config_budget configs/" + name + ".yaml --grid grids/" + name + ".txt --out_dir configs && "
            + "bash " + parallel_location + " configs/" + name + "_grid_" + name + " "+ str(repeat) + " " + str(max_jobs) + " && "
            + "python " + agg_batch_location + " --dir results/" + name + "_grid_" + name + ""
        )

        

        return fake_sh_file

    def train(self, cyto = None):


        if (self.custom):
            # check if cytokine has been initalized
            if cyto not in self.active_cyto_list:
                raise(ValueError("{} has not been initalized as a cytokine".format(cyto)))

            name = cyto + "_" + self.eset[:self.eset.index(".")]  

            args = parse_args(name + ".yaml")
        else:
            name = None
            args = parse_args()
            
        # Load config file
        load_cfg(cfg, args)
        
        set_out_dir(cfg.out_dir, args.cfg_file)
        # Set Pytorch environment
        torch.set_num_threads(cfg.num_threads)
        dump_cfg(cfg)
        # Repeat for different random seeds
        for i in range(args.repeat):
            set_run_dir(cfg.out_dir)
            setup_printing()
            # Set configurations for each run
            cfg.seed = cfg.seed + 1
            seed_everything(cfg.seed)
            auto_select_device()
            # Set machine learning pipeline
            # to fix this, copy create_dataset, only with formed graphs already created.

            if (self.custom):
                dataset_raw = custom_dataset(root=os.path.join("datasets",  name), name=name, url="null")
                graphs = GraphDataset.pyg_to_graphs(dataset_raw)
                datasets = create_local_dataset(graphs)
            else:
                datasets = create_dataset()
            loaders = create_loader(datasets)
            loggers = create_logger()
            model = create_model()





            # Add edge_weights attribute to the datasets so that they can be accessed in batches
            num_edges = len(datasets[0][0].edge_index[0])
            edge_weights = torch.nn.Parameter(torch.ones(num_edges))
            for loader in loaders:
                for dataset in loader.dataset:
                    dataset.edge_weights = edge_weights


            #add edge weights to the set of parameters
            newParam = list()
            for param in model.parameters():
                newParam.append(param)
            
            newParam.append(edge_weights)

            optimizer = create_optimizer(newParam)
            scheduler = create_scheduler(optimizer)
            # Print model info
            logging.info(model)
            logging.info(cfg)
            cfg.params = params_count(model)
            logging.info('Num parameters: %s', cfg.params)


            # Start training
            if cfg.train.mode == 'standard':
                train(loggers, loaders, model, optimizer, scheduler)
            else:
                train_dict[cfg.train.mode](loggers, loaders, model, optimizer,
                                        scheduler)
            # Aggregate results from different seeds
            agg_runs(cfg.out_dir, cfg.metric_best)
            # When being launched in batch mode, mark a yaml as done
            if args.mark_done:
                os.rename(args.cfg_file, f'{args.cfg_file}_done')

        

    def get_cyto_list():
        return ['CCL1', 'CCL2', 'CCL3', 'CCL3L1', 'CCL4', 'CCL4L2', 'CCL5', 'CCL7', 
                'CCL8', 'CCL13', 'CCL17', 'CCL19', 'CCL20', 'CCL23', 'CCL24', 'CCL25', 
                'CCL26', 'CCL27', 'CD40LG', 'CD70', 'CKLF', 'CSF1', 'CSF2', 'CXCL2', 'CXCL8', 
                'CXCL9', 'CXCL10', 'CXCL11', 'CXCL12', 'CXCL13', 'CXCL14', 'EGF', 'FASLG', 
                'FLT3LG', 'HGF', 'IFNG', 'IFNL1', 'IL2', 'IL3', 'IL4', 'IL5', 'IL6', 
                'IL7', 'IL11', 'IL12A', 'IL12B', 'IL18', 'IL19', 'IL1B', 'IL1RN', 'IL23A', 
                'IL26', 'LIF', 'OSM', 'PDGFB', 'PF4', 'PPBP', 'SPP1', 'TGFA', 'TGFB1', 
                'TGFB3', 'TNFSF9', 'TNFSF10', 'TNFSF11', 'TNFSF13', 'TNFSF13B', 'TNFSF14', 
                'TNF', 'XCL1', 'XCL2']