import os
import numpy as np

# A file that is meant to generate all the files needed to run the underlying graphgym.

# things we need to be provided
eset = os.path.join("rawData", "GSE40240_ESET.csv")
patients = os.path.join("rawData", "GSE40240_patients.csv")
CYTOKINE = "CCL2"


# general configs that we can keep as they are, unless changed.
blood_only=True
batch_size = 80
eval_period = 20
layers_pre_mp = 2
layers_mp = 6
layers_post_mp = 2
dim_inner = 137
max_epoch = 400

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


#MAIN


#get patient data
patient_dict, patient_list = process_patients(patients) # a dict that matches a patient name to their classification

# process graph data
cyto_list,cyto_adjacency_dict,cyto_tissue_dict  = process_graphs(blood_only) # list of cytokines, maps a cytokine's name to their adjacency matrix, maps a cytokine's name to the tissues they need

tissue_gene_dict, gene_set = process_tissues(blood_only) # dict that matches tissues to the genes associated with them, a set of all genes we have


gene_to_patient, active_tissue_gene_dict = process_eset(eset, gene_set, patient_dict, tissue_gene_dict) # 2 layer deep dict. First layer maps gene name to a dict. Second layer matches patient code to gene expresion data of the given gene.


