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

print(patient_dict)
print(patient_list)