import os
from Bio import SeqIO
import pickle
import numpy as np
import argparse
import pdb

def hot_encode(sequence):
    seq_encoded = np.zeros((len(sequence), 4))
    dict_nuc = {
        "A": 0,
        "C": 1,
        "G": 2,
        "T": 3
    }
    i = 0
    for l in sequence:
        if(l.upper() in dict_nuc.keys()):
            seq_encoded[i][dict_nuc[l.upper()]] = 1
            i = i+1
        else:
            return []
    return seq_encoded

def PseTNC(sequence):
    dic = dict()
    PseTNC_encoded = []
    lst = ['A', 'C', 'G', 'T']
    for i in range(4):
        for j in range(4):
            for k in range(4):
                dic[lst[i] + lst[j] + lst[k]] = 0
    for i in range(len(sequence) - 2):
        s = sequence[i].upper() + sequence[i + 1].upper() + sequence[i + 2].upper()
        dic[s] += 1
    data = dict(sorted(dic.items(), key=lambda x: x[0]))
    for k in data.values():
        k /= 145
        PseTNC_encoded.append([k])
    return PseTNC_encoded

def parse_shape_file(filepath):
    """
    Parse a shape file and return list of shape arrays (one per sequence)
    Each shape array is 147 columns x 1 channel (flattened from 5 lines of ~30 columns each)
    """
    shape_data_list = []
    current_shape = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # New sequence starts
                if current_shape:
                    # Convert accumulated data to numpy array
                    shape_array = np.array(current_shape, dtype=np.float32).flatten()
                    # Ensure exactly 147 elements by padding or trimming
                    if len(shape_array) < 147:
                        shape_array = np.pad(shape_array, (0, 147 - len(shape_array)), mode='constant', constant_values=0)
                    elif len(shape_array) > 147:
                        shape_array = shape_array[:147]
                    shape_data_list.append(shape_array)
                    current_shape = []
            else:
                # Data line - parse comma-separated values
                try:
                    values = [float(v) if v.strip().upper() != 'NA' else 0.0 for v in line.split(',')]
                    current_shape.extend(values)
                except ValueError:
                    # Handle parsing errors
                    continue
        
        # Don't forget the last sequence
        if current_shape:
            shape_array = np.array(current_shape, dtype=np.float32).flatten()
            if len(shape_array) < 147:
                shape_array = np.pad(shape_array, (0, 147 - len(shape_array)), mode='constant', constant_values=0)
            elif len(shape_array) > 147:
                shape_array = shape_array[:147]
            shape_data_list.append(shape_array)
    
    return shape_data_list

parser = argparse.ArgumentParser(description='Generating Pickle encoded File from Fasta with Shape Data')
parser.add_argument('-p', '--path', dest='path', type=str, default=r"D:\data\setting1",
                    help='Fasta File Path')
parser.add_argument('-sp', '--shapepath', dest='shapePath', type=str, default=r"D:\data\shapes",
                    help='Shape File Directory Path')
parser.add_argument('-f', '-fas', dest='fasName', type=str, default="nucleosomes_vs_linkers_elegans.fas",
                    help='Fasta filename')
parser.add_argument('-o',  '--out', dest='outDir', type=str, default=r"D:\data\setting1/pickle_H",
                    help='Output file Path')

args = parser.parse_args()
inPath = args.path
shapeDir = args.shapePath
outPath = args.outDir
fasName = args.fasName

del args

if(not os.path.isdir(outPath)):
    os.mkdir(outPath)

# Initialize lists for nucleosomal and linker sequences
nucList = []
linkList = []
nuc_one_hot_List = []
link_one_hot_List = []
nuc_PseTNC_List = []
link_PseTNC_List = []
nuc_shape_List = []
link_shape_List = []

# Parse FASTA file
fastaSequences = SeqIO.parse(open(os.path.join(inPath, fasName)), 'fasta')
fasta_list = list(fastaSequences)

print(f"Read {len(fasta_list)} sequences from FASTA file")

# Parse shape files (EP, HelT, MGW, ProT, Roll)
shape_files = ['EP', 'HelT', 'MGW', 'ProT', 'Roll']
shape_data_dict = {}

for shape_type in shape_files:
    shape_filepath = os.path.join(shapeDir, f"{fasName}.{shape_type}")
    if os.path.exists(shape_filepath):
        print(f"Parsing {shape_type} shape file...")
        shape_data_dict[shape_type] = parse_shape_file(shape_filepath)
    else:
        print(f"Warning: Shape file not found: {shape_filepath}")

# Process each FASTA sequence
seq_idx = 0
for fasta in fasta_list:
    name, sequence = fasta.id, str(fasta.seq)
    
    # Encode sequence
    one_hot = hot_encode(sequence)
    pse_tnc = PseTNC(sequence)
    
    if one_hot is None or pse_tnc is None or len(one_hot) == 0 or len(pse_tnc) == 0:
        continue
    
    # Collect shape data for this sequence (5 shape types x 147 columns)
    shape_features = []
    for shape_type in shape_files:
        if shape_type in shape_data_dict and seq_idx < len(shape_data_dict[shape_type]):
            shape_features.append(shape_data_dict[shape_type][seq_idx])
    
    # Stack shapes into (5, 147) array then transpose to (147, 5)
    if len(shape_features) == len(shape_files):
        combined_shapes = np.array(shape_features, dtype=np.float32).T  # Shape: (147, 5)
    else:
        # Fallback if shape data is incomplete
        combined_shapes = np.zeros((147, 5), dtype=np.float32)
    
    # Classify and store
    if "nucleosomal" in name:
        nucList.append(sequence)
        nuc_one_hot_List.append(one_hot)
        nuc_PseTNC_List.append(pse_tnc)
        nuc_shape_List.append(combined_shapes)
    else:
        linkList.append(sequence)
        link_one_hot_List.append(one_hot)
        link_PseTNC_List.append(pse_tnc)
        link_shape_List.append(combined_shapes)
    
    seq_idx += 1

print(f"Nucleosomal sequences: {len(nucList)}")
print(f"Linker sequences: {len(linkList)}")

# Save pickle files
with open(os.path.join(outPath, "nuc.pickle"), "wb") as fp:
    pickle.dump(nucList, fp)
with open(os.path.join(outPath, "link.pickle"), "wb") as fp:
    pickle.dump(linkList, fp)

with open(os.path.join(outPath, "one_hot_nuc.pickle"), "wb") as fp:
    pickle.dump(nuc_one_hot_List, fp)
with open(os.path.join(outPath, "one_hot_link.pickle"), "wb") as fp:
    pickle.dump(link_one_hot_List, fp)

with open(os.path.join(outPath, "PseTNC_nuc.pickle"), "wb") as fp:
    pickle.dump(nuc_PseTNC_List, fp)
with open(os.path.join(outPath, "PseTNC_link.pickle"), "wb") as fp:
    pickle.dump(link_PseTNC_List, fp)

with open(os.path.join(outPath, "shape_nuc.pickle"), "wb") as fp:
    pickle.dump(nuc_shape_List, fp)
with open(os.path.join(outPath, "shape_link.pickle"), "wb") as fp:
    pickle.dump(link_shape_List, fp)

print("Encoding complete! Pickle files saved.")
