import random
import json
import linecache
import os
import numpy as np
from tqdm import tqdm

seed = 123456

def generate_neg(
    output_file: str,
    pos_data_path: str,
    neg_pool_path: str,
    upstream_delta: int,
    downstream_delta: int,
    neg_sample_fold_size: int = 1
):
    with open(pos_data_path, 'r') as input:
        pos_data = input.readlines()
        len_label = len(json.loads((pos_data[0]).strip())['label'])
    pos_data = [(json.loads(line.strip()))['seq_index'] for line in pos_data if line != '']
    

    random.seed(seed)
    with open(neg_pool_path, 'r') as f:
        all_lines = len(f.readlines())
    linos = random.sample(range(1, all_lines+1), k=int(neg_sample_fold_size*len(pos_data)))

    with open(output_file, 'w') as output:
        for lino in tqdm(linos):
            line = linecache.getline(neg_pool_path, lineno=lino)
            line = line.strip().split('\t')
            neg_example = line[4]
            neg_example = neg_example.strip().split('_')
            start = int(neg_example[1])-1+upstream_delta
            end = int(neg_example[1])-1+downstream_delta+1
            if [neg_example[0], start, end] in pos_data:
                continue
            if len(neg_example[3]) > 1:
                continue
            to_be_written = {
                'seq_index': [neg_example[0], start, end],
                'alt': neg_example[3].upper(),
                'label': np.zeros((len_label,), dtype='int8').tolist()
            }
            output.write(json.dumps(to_be_written) + '\n')