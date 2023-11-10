import os
import torch
import numpy as np
from CytokinesDataSet import CytokinesDataSet
import sys

# A file that is meant to generate all the files needed to run the underlying graphgym.

# things we need to be provided

eset = sys.argv[1]
eset = os.path.join("rawData", eset + ".csv")

patients = sys.argv[2]

patients = os.path.join("rawData", patients + ".csv")
cyto = sys.argv[3]
grid = sys.argv[4]

if grid[0] == "F":
    grid = False
else:
    grid = bool(grid)


# general configs that we can keep as they are, unless changed.
blood_only=True
batch_size = 80
eval_period = 20
layers_pre_mp = 2
layers_mp = 6
layers_post_mp = 2
dim_inner = 137
max_epoch = 400

def makeConfigFile(name, batch_size, eval_period, layers_pre_mp, layers_mp, 
                    layers_post_mp, dim_inner, max_epoch):
    if (not os.path.exists(os.path.abspath("configs"))):
        os.makedirs(os.path.abspath("configs"))

    # using with statement
    with open(os.path.join('configs', name + ".yaml"), 'w') as file:
        file.write('out_dir: results\n')
        file.write('dataset:\n')
        file.write(' format: PyG\n')
        file.write(' name: Custom,' + name + ',,\n')
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
    
    return name + ".yaml"



def create_cyto_database(cyto, eset, cyto_tissue_dict, active_tissue_gene_dict, patient_list, 
                            patient_dict, gene_to_patient, cyto_adjacency_dict):

    # creates graphname
    graphName = cyto + "_" + eset

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


def normalize_vector(vector):
    min_val = np.min(vector)
    max_val = np.max(vector)
    normalized_vector = (vector - min_val) / (max_val - min_val)
    return normalized_vector    

def process_tissues(blood_only):
    tissue_gene_dict = dict() # maps tissues to the genes associated with them

    gene_set = set()

    tissue_file = open("GenesToTissues.csv")

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

def process_eset(eset, gene_set, patient_dict, tissue_gene_dict):
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
        gene_nums = normalize_vector(gene_nums)


        
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

def process_graphs(blood_only):
    if(blood_only):
        graph_folder_path = "Modified Graphs"
    else:
        graph_folder_path = "Graphs"
    
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

def process_patients(patients):
        patient_file = open(patients, 'r')
        patient_dict = dict()
        patient_list = []

        for line in patient_file.read().splitlines():
            parts = line.split(",")
            patient_dict[parts[0]] = int(parts[1])
            patient_list.append(parts[0])
        
        return patient_dict, patient_list

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


def make_grid(name):
    if (not os.path.exists(os.path.abspath("grids"))):
        os.makedirs(os.path.abspath("grids"))

    # using with statement
    with open(os.path.join('grids', name + ".txt"), 'w') as file:
        file.write("dataset.format format ['PyG']\n")
        file.write("dataset.name dataset ['Custom," + name + ",','Custom," + name + ",']\n")
        file.write("dataset.task task ['graph']\n")
        file.write("dataset.transductive trans [False]\n")
        file.write("dataset.augment_feature feature [[]]\n")
        file.write("dataset.augment_label label ['']\n")
        file.write("gnn.layers_pre_mp l_pre [1,2]\n")
        file.write("gnn.layers_mp l_mp [2,4,6,8]\n")
        file.write("gnn.layers_post_mp l_post [2,3]\n")
        file.write("gnn.stage_type stage ['skipsum','skipconcat']\n")
        file.write("gnn.agg agg ['add','mean']\n")

def make_grid_sh(eset_name, cyto, name):
    with open("run_custom_batch_" + eset_name + "_" + cyto + ".sh", 'w') as file:
        file.write("#!/usr/bin/env bash\n")
        file.write("\n")
        file.write("CONFIG=" + name + "\n")
        file.write("GRID=" + name + "\n")
        file.write("REPEAT=1\n")
        file.write("MAX_JOBS=20\n")
        file.write("\n")
        file.write("# generate configs (after controlling computational budget)\n")
        file.write("# please remove --config_budget, if don't control computational budget\n")
        file.write("python configs_gen.py --config configs/${CONFIG}.yaml \\\n")
        file.write(" --config_budget configs/${CONFIG}.yaml \\\n")
        file.write(" --grid grids/${GRID}.txt \\\n")
        file.write(" --out_dir configs\n")
        file.write("#python configs_gen.py --config configs/ChemKG/${CONFIG}.yaml --config_budget configs/ChemKG/${CONFIG}.yaml --grid grids/ChemKG/${GRID}.txt --out_dir configs\n")
        file.write("# run batch of configs\n")
        file.write("# Args: config_dir, num of repeats, max jobs running\n")
        file.write("bash parallel.sh configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS\n")
        file.write("# rerun missed / stopped experiments\n")
        file.write("bash parallel.sh configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS\n")
        file.write("# rerun missed / stopped experiments\n")
        file.write("bash parallel.sh configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS\n")
        file.write("\n")
        file.write("# aggregate results for the batch\n")
        file.write("python agg_batch.py --dir results/${CONFIG}_grid_${GRID}\n")


def make_single_sh(eset_name, cyto, config_name):
    # using with statement
    with open("run_custom_" + eset_name + "_" + cyto + ".sh", 'w') as file:
        file.write('#!/usr/bin/env bash\n')
        file.write('\n')
        escaped_path = os.path.join("configs",config_name).replace("\\", "/")
        file.write('python main.py --cfg ' + escaped_path + ' --repeat 1')

#MAIN


#get patient data
patient_dict, patient_list = process_patients(patients) # a dict that matches a patient name to their classification

# process graph data
cyto_list,cyto_adjacency_dict,cyto_tissue_dict  = process_graphs(blood_only) # list of cytokines, maps a cytokine's name to their adjacency matrix, maps a cytokine's name to the tissues they need

tissue_gene_dict, gene_set = process_tissues(blood_only) # dict that matches tissues to the genes associated with them, a set of all genes we have


gene_to_patient, active_tissue_gene_dict = process_eset(eset, gene_set, patient_dict, tissue_gene_dict) # 2 layer deep dict. First layer maps gene name to a dict. Second layer matches patient code to gene expresion data of the given gene.

eset_name = sys.argv[1]

create_cyto_database(cyto, eset_name, cyto_tissue_dict, active_tissue_gene_dict, patient_list, 
                            patient_dict, gene_to_patient, cyto_adjacency_dict)

name = cyto + "_" + eset_name


config_name = makeConfigFile(name, batch_size, eval_period, layers_pre_mp, layers_mp, 
                             layers_post_mp, dim_inner, max_epoch)



if (grid) :
    make_grid_sh(sys.argv[1], cyto, name)
    make_grid(name)
else:
    make_single_sh(sys.argv[1], cyto, config_name)
#also need to make the grid file and the sh file