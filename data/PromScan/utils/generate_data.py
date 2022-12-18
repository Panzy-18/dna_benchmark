import numpy as np
import os
import json
import random
from tqdm import tqdm

seed = 123456
chr_sizes = {
    '1': 248956422, '2': 242193529, '3': 198295559, '4': 190214555,
    '5': 181538259, '6': 170805979, '7': 159345973, 'X': 156040895,
    '8': 145138636, '9': 138394717, '11': 135086622, '10': 133797422,
    '12': 133275309, '13': 114364328, '14': 107043718, '15': 101991189,
    '16': 90338345, '17': 83257441, '18': 80373285, '20': 64444167,
    '19': 58617616, 'Y': 57227415, '22': 50818468, '21': 46709983
}

def calc_pos_range(
    tss_gtf_file: str,
    chromosome: str,
    upstream_delta: int = -250,
    downstream_delta: int = +49
):
    print(f'\nCalculate positive range...')
    forward_coverage = np.zeros((chr_sizes[chromosome],), dtype='int8')
    reverse_coverage = np.zeros((chr_sizes[chromosome],), dtype='int8')
    with open(tss_gtf_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line1 = line.strip().split('\t')
            if line1[0] == chromosome:
                
                if line1[6] == '+':
                    tss = int(line1[3])
                    for x in range(tss+upstream_delta-1, tss+downstream_delta):
                        forward_coverage[x] = 1
                elif line1[6] == '-':
                    tss = int(line1[4])
                    for x in range(tss-downstream_delta-1, tss-upstream_delta):
                        reverse_coverage[x] = 1
    print('Calculation Done.')
    return forward_coverage, reverse_coverage

def do_sliding_window(
    tss_gtf_file: str,
    output_folder: str,
    train_chrs: list[str] = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '14', '15', '16', '17', '18', '19', '20', '21', '22'],
    valid_chrs: list[str] = ['12', '13'],
    test_chrs: list[str] = ['10', '11', 'X', 'Y'],
    window_size: int = 300,
    step_size: int = 50,
    upstream_delta = -250,
    downstream_delta = +49,
    overlap_threshold: float = 0.8
):
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    sets = {'train': train_chrs, 'valid': valid_chrs, 'test': test_chrs}
    for set_ in tqdm(sets):
        data = []
        for chr in tqdm(sets[set_]):
            forward_coverage, reverse_coverage = calc_pos_range(tss_gtf_file, chr, upstream_delta, downstream_delta)
            for idx in tqdm(range(chr_sizes[chr]//step_size)):
                start = idx * step_size
                end = start + window_size
                label = 0
                strand = (lambda x: '+' if x % 2 == 0 else '-')(idx)
                if np.sum(reverse_coverage[start: end]) >= overlap_threshold * window_size:
                    label = 1
                    strand = '-'
                if np.sum(forward_coverage[start: end]) >= overlap_threshold * window_size:
                    label = 1
                    strand = '+'
                to_be_written = {
                    'seq_index': ['chr'+chr,start, end, strand],
                    'label': label
                }
                data.append(json.dumps(to_be_written)+'\n')
        print(f'\nWriting into file...')
        random.seed(seed)
        random.shuffle(data)
        with open(output_folder+'/'+set_+'.txt', 'w') as output:
            output.writelines(data)