import random
import json
import numpy as np
from tqdm import tqdm
seed = 123456

def generate_positive(
    output_path: str,
    upstream_delta: int = -250,
    downstream_delta: int = +49,
    seq_length: int = 500,
    epdnew_bed_path: str = './tss_epdnew.bed',
):
    with open(epdnew_bed_path, 'r') as input:
        pos_bed = input.readlines()
    pos_bed = [line.strip().split('\t') for line in pos_bed if line != '']
    
    random.seed(seed)
    '''
    In .gtf format, tss, pos range is [tss+upstream_delta, tss+downstream_delta]
    Convert it to .bed format, it is [tss+upstream_delta-1, tss+downstream_delta)
    => [start, start+seq_length) contains [tss+upstream_delta-1, tss+downstream_delta)
    => tss+downstream_delta-seq_length <= start <= tss+upstream_delta-1
    => downstream_delta-seq_length <= start - tss <= upstream_delta-1
    '''
    start_deltas = random.choices(range(downstream_delta - seq_length, upstream_delta), k=len(pos_bed))
    with open(output_path, 'w') as output:
        for idx, line in tqdm(enumerate(pos_bed)):
            tss = int(line[1])
            start_delta = start_deltas[idx]
            to_be_written = {
                'seq_index': [line[0], tss+start_delta, tss+start_delta+seq_length, line[5]],
                'label': 1
            }
            output.write(json.dumps(to_be_written)+'\n')

def generate_negative(
    output_path: str,
    neg_size_fold: int = 3,
    upstream_delta: int = -250,
    downstream_delta: int = +49,
    seq_length: int = 500,
    epdnew_bed_path = './tss_epdnew.bed'
):
    output = open(output_path, 'w')
    chr_sizes = {
        '1': 248956422, '2': 242193529, '3': 198295559, '4': 190214555,
        '5': 181538259, '6': 170805979, '7': 159345973, 'X': 156040895,
        '8': 145138636, '9': 138394717, '11': 135086622, '10': 133797422,
        '12': 133275309, '13': 114364328, '14': 107043718, '15': 101991189,
        '16': 90338345, '17': 83257441, '18': 80373285, '20': 64444167,
        '19': 58617616, 'Y': 57227415, '22': 50818468, '21': 46709983
    }

    with open(epdnew_bed_path, 'r') as input:
        pos_bed = input.readlines()
    pos_bed = [line.strip().split('\t') for line in pos_bed if line != '']

    '''
    Fix chromosome-specific seeds
    '''
    random.seed(seed)
    neg_seeds = random.sample(range(100000, 1000000), k=len(chr_sizes))

    for idx, chr in tqdm(enumerate(chr_sizes)):
        chr_seed = neg_seeds[idx]
        neg_size = 0
        pos_range = np.zeros((chr_sizes[chr]), dtype='int8')
        for line in pos_bed:
            if line[0][3:] == chr:
                neg_size += 1
                tss = int(line[1])
                for x in range(tss+upstream_delta-1, tss+downstream_delta):
                    pos_range[x] = 1
        neg_size *= neg_size_fold

        random.seed(chr_seed)
        pos_starts = random.sample(range(chr_sizes[chr]), k=neg_size)
        for start in pos_starts:
            if np.sum(pos_range[start: start+seq_length]) > 0:
                continue
            strand = ''
            if start / chr_sizes[chr] > 0.5:
                strand = '+'
            else:
                strand = '-'
            to_be_written = {
                'seq_index': ['chr'+chr, start, start+seq_length, strand],
                'label': 0
            }
            output.write(json.dumps(to_be_written)+'\n')
    output.close()


if __name__ == '__main__':
    generate_positive(output_path='../prom300_pos.txt', seq_length=300)
    generate_negative(output_path='../prom300_neg.txt', seq_length=300)
    generate_positive('../prom500_varying_tss_pos.txt')
    generate_negative('../prom500_varying_tss_neg.txt')