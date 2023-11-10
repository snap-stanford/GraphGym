import os


# A file that is meant to generate all the files needed to run the underlying graphgym.

# things we need to be provided
GSE = os.path.join("rawData", "GSE40240_ESET")
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
