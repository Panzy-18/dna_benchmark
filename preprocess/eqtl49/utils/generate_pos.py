import numpy as np
import json
from tqdm import tqdm


def generate_pos(
    upstream_delta: int = -499,
    downstream_delta: int = 500,
    pip_threshold: float = 0.9,
    by_tissue_files: list[str] = None
):
    '''
    Generate (positive) variant effect prediction data from positive bed files
    Args:
        output_file         (str): path of the output file
        upstream_delta       (int): suppose the snp is at x, the location of the start of the sequence. 0-based inclusive
        downstream_delta         (int): suppose the snp is at x, the location of the end of the sequence. 0-based inclusive
        mut_column_number   (int): which column in the bed file indicates the mutation allele? 0-based inclusive
        pip_column_number   (int): which column in the bed file indicates the pip (post inclusion probability) of the snp? 0-based inclusive
        pip_threshold:      (float): pip cutoff of the positve functional snp. The snp will be included in the output only its pip is bigger than this cutoff
        by_tissue_files           (list[str]): names of files containing the information of the positive snps. Each file represents a single tissue
    Returns:
        None, write data in the given output file.
        Data format example: {"seq_index": ["chrX", 279168, 279296], "mut": "A", "label": [0, 1]}
    '''

    data = {}
    print('Start processing data...')
    '''
    store information in the hash table
    {
        [chromosome, position]: [], #length is # of all files
    }
    '''
    for file_idx, bed_file in tqdm(enumerate(by_tissue_files)):
        with open(bed_file, 'r') as input:
            line = ''
            while True:
                line = input.readline()
                if line == '':
                    break
                
                line = line.strip().split('\t')
                chr, start, mutation, pip = line[0], line[1], line[3].upper(), line[5]
                start = int(start)
                pip = float(pip)
                
                if len(mutation) > 1 or pip < pip_threshold:
                    continue
                
                if json.dumps([chr, start, mutation]) not in data:
                    data[json.dumps([chr, start, mutation])] = np.zeros((len(by_tissue_files),), int)                    
                data[json.dumps([chr, start, mutation])][file_idx] = 1

    pos_data = []
    for d in tqdm(data):
        d1 = json.loads(d)
        sample = {
            "index": [d1[0], d1[1]+upstream_delta, d1[1]+downstream_delta+1, '+'], 
            "alter": d1[2], 
            "label": list(data[d].tolist())
        }
        pos_data.append(sample)

    return pos_data

# if __name__ == '__main__':
#     with open('../source/by_tissue/all_tissues.txt', 'r') as f:
#         all_tissues = json.loads(f.readline().strip())
#     all_tissues = [('../source/by_tissue/'+tissue.lower()+'.txt') for tissue in all_tissues]

#     generate_pos(
#         upstream_delta=0,
#         downstream_delta=0,
#         output_file='./pos.txt',
#         by_tissue_files=all_tissues
#     )
