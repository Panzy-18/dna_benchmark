import random
import json
import numpy as np
from tqdm import tqdm
import argparse
import pathlib

# hg38

def generate_positive(
    upstream_delta: int = -250,
    downstream_delta: int = +49,
    seq_length: int = 500,
    epdnew_bed_path: str = './tss_epdnew.bed',
) -> list:
    with open(epdnew_bed_path, 'r') as input:
        pos_bed = input.readlines()
    pos_bed = [line.strip().split('\t') for line in pos_bed if line != '']

    '''
    In .gtf format, tss, pos range is [tss+upstream_delta, tss+downstream_delta]
    Convert it to .bed format, it is [tss+upstream_delta-1, tss+downstream_delta)
    => [start, start+seq_length) contains [tss+upstream_delta-1, tss+downstream_delta)
    => tss+downstream_delta-seq_length <= start <= tss+upstream_delta-1
    => downstream_delta-seq_length <= start - tss <= upstream_delta-1
    '''
    start_deltas = random.choices(range(downstream_delta - seq_length, upstream_delta), k=len(pos_bed))
    pos_data = []
    for idx, line in tqdm(enumerate(pos_bed)):
        tss = int(line[1])
        start_delta = start_deltas[idx]
        sample = {
            'index': [line[0], tss+start_delta, tss+start_delta+seq_length, line[5]],
            'tss': -start_delta,
            'label': [1]
        }
        pos_data.append(sample)
    return pos_data

def generate_negative(
    neg_size_fold: int = 3,
    upstream_delta: int = -250,
    downstream_delta: int = +49,
    seq_length: int = 500,
    epdnew_bed_path = './tss_epdnew.bed'
):

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
    neg_seeds = random.sample(range(100000, 1000000), k=len(chr_sizes))
    neg_data = []
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
        pos_starts = random.sample(range(chr_sizes[chr] - seq_length), k=neg_size)
        for start in pos_starts:
            if np.sum(pos_range[start: start+seq_length]) > 0:
                continue
            strand = ''
            if start / chr_sizes[chr] > 0.5:
                strand = '+'
            else:
                strand = '-'
            sample = {
                'index': ['chr'+chr, start, start+seq_length, strand],
                'tss': -1,
                'label': [0],
            }
            neg_data.append(sample)
    return neg_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate promoter detection dataset from edp')
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--data-root', type=str, default='../../data')
    parser.add_argument('--seq-length', type=int, default=500)
    parser.add_argument('--valid-ratio', type=float, default=0.1)
    parser.add_argument('--test-ratio', type=float, default=0.1)
    args = parser.parse_args()
    random.seed(args.seed)
    output_dir = pathlib.Path(args.data_root) / 'promoter'
    output_dir.mkdir(exist_ok=True)
    
    pos_data = generate_positive(seq_length=args.seq_length)
    neg_data = generate_negative(seq_length=args.seq_length)
    print(f'Generate {len(pos_data)} positive.')
    print(f'Generate {len(neg_data)} negative.')
    all_data = pos_data + neg_data
    random.shuffle(all_data)
    train_end = int(len(all_data) * (1-args.valid_ratio-args.test_ratio))
    valid_end = int(len(all_data) * (1-args.test_ratio))
    data = {}
    data['train'] = all_data[:train_end]
    data['valid'] = all_data[train_end: valid_end]
    data['test'] = all_data[valid_end:]
    for split in ['valid', 'test', 'train']:
        output_file = output_dir / f'{split}.json'
        with open(output_file, 'w') as f:
            for sample in data[split]:
                print(json.dumps(sample), file=f)
        print(f'File {str(output_file)} with {len(data[split])} samples.')
    
    