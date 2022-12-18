import random
import json
import linecache
import os
import numpy as np
from tqdm import tqdm


def generate_neg(
    pos_data: list[dict],
    neg_pool_path: str,
    upstream_delta: int,
    downstream_delta: int,
    neg_sample_fold_size: int = 1
):

    len_label = len(pos_data[0]['label'])
    pos_index = [d['index'] for d in pos_data]
    

    with open(neg_pool_path, 'r') as f:
        all_lines = len(f.readlines())
    linos = random.sample(range(1, all_lines+1), k=int(neg_sample_fold_size*len(pos_data)))

    neg_data = []
    for lino in tqdm(linos):
        line = linecache.getline(neg_pool_path, lineno=lino)
        line = line.strip().split('\t')
        neg_example = line[4]
        neg_example = neg_example.strip().split('_')
        start = int(neg_example[1])-1+upstream_delta
        end = int(neg_example[1])-1+downstream_delta+1
        if [neg_example[0], start, end] in pos_index:
            continue
        if len(neg_example[3]) > 1:
            continue
        sample = {
            'index': [neg_example[0], start, end, '+'],
            'alter': neg_example[3].upper(),
            'label': np.zeros((len_label,), dtype='int8').tolist()
        }
        neg_data.append(sample)
    
    return neg_data