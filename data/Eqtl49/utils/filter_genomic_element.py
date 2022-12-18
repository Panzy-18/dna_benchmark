from tqdm import tqdm
import numpy as np
import json

def filter_genomic_element(
    input_file: str, # rows are {"seq_index": [chr, start, end, ...], ...}
    gtf_file: str,
    output_file: str
):
    chr_sizes = {
        '1': 248956422, '2': 242193529, '3': 198295559, '4': 190214555,
        '5': 181538259, '6': 170805979, '7': 159345973, 'X': 156040895,
        '8': 145138636, '9': 138394717, '11': 135086622, '10': 133797422,
        '12': 133275309, '13': 114364328, '14': 107043718, '15': 101991189,
        '16': 90338345, '17': 83257441, '18': 80373285, '20': 64444167,
        '19': 58617616, 'Y': 57227415, '22': 50818468, '21': 46709983
    }

    output = open(output_file, 'w')

    for chr in tqdm(chr_sizes):
        chr_element = np.zeros(chr_sizes[chr], dtype='int8')
        # load coverage
        with open(gtf_file, 'r') as f:
            while True:
                line = f.readline()
                if line == '':
                    break
                line = line.strip().split('\t')
                chr1, start, end = line[0], int(line[3]) - 1, int(line[4])
                if chr1 == chr:
                    for i in range(start, end):
                        chr_element[i] = 1
        # filter out snps in genes of this chromosome
        with open(input_file, 'r') as f:
            while True:
                line = f.readline()
                if line == '':
                    break
                line = line.strip()
                data = json.loads(line)
                if data['seq_index'][0] != ('chr' + chr):
                    continue
                if np.sum(chr_element[data['seq_index'][1]:data['seq_index'][2]]) == 0:
                    output.write(line + '\n')
    
    output.close()