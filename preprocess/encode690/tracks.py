import numpy as np
import json
import math
from tqdm import tqdm
from typing import List
from glob import glob
import pdb
import argparse
import h5py
import pathlib
import pyfastx

chr_to_num = {
        '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, '10':10, '11':11, '12':12,
        '13':13, '14':14, '15':15, '16':16, '17':17, '18':18, '19':19, '20':20, '21':21, '22':22, 'X':23, 'Y':24
    }
hg19 = pyfastx.Fasta('../../data/genome/hg19.fa')

def bed2json(
    list_of_bed_files: List[str],
    window_size: int,
    output_file: str,
    genome: str,
    chromosomes: List[str]=None,
    at_least_one_target: List[int]=None,
):
    '''
    Get sliding window examples from bed files
    .bed format:
    chromosome  start(0-based, inclusive)   end(0-based, exclusive) optional_columns

    Args:
        list_of_bed_files list[str]:  bed files containing track information
        window_size int: sliding window size
        overlap_ratio_threshold float: when the overlap ratio of a sliding window exceeds this value, it is labeled as 1
        output_file str: output file path
        genome str: reference genome, hg19 or hg38
        chromosomes list[str]: (optional) which chromosomes are processed, default is all. chromosome name should be chr1, chr2, ..., chrX, chrY
        at_least_one_target list[int]: list of bed file indices. a slide will be contained in the output only when there is at least one positive label for these files.
    Returns:
        None, write the output examples in output_file. example format: {"seq_index": ["chrX", 20320896, 20321024], "label": [1, 0]} note that the start index is 0-based inclusive and the end index is 0-based exclusive.
    '''

    def bed2json_chr(
        list_of_bed_files: List[str],
        window_size: int,
        genome: str,
        chromosome: str,
        chromosome_num: int,
        at_least_one_target: List[int]=None
    ):
        '''
        process a single chromosome
        '''
        
        chr_sizes = {
            'hg19': {
                'chr1': 249250621, 'chr2': 243199373, 'chr3': 198022430, 'chr4': 191154276,
                'chr5': 180915260, 'chr6': 171115067, 'chr7': 159138663, 'chrX': 155270560,
                'chr8': 146364022, 'chr9': 141213431, 'chr10': 135534747, 'chr11': 135006516,
                'chr12': 133851895, 'chr13': 115169878, 'chr14': 107349540, 'chr15': 102531392,
                'chr16': 90354753, 'chr17': 81195210, 'chr18': 78077248, 'chr20': 63025520,
                'chrY': 59373566, 'chr19': 59128983, 'chr22': 51304566, 'chr21': 48129895
            },
            'hg38': {
                'chr1': 248956422, 'chr2': 242193529, 'chr3': 198295559, 'chr4': 190214555,
                'chr5': 181538259, 'chr6': 170805979, 'chr7': 159345973, 'chrX': 156040895,
                'chr8': 145138636, 'chr9': 138394717, 'chr11': 135086622, 'chr10': 133797422,
                'chr12': 133275309, 'chr13': 114364328, 'chr14': 107043718, 'chr15': 101991189,
                'chr16': 90338345, 'chr17': 83257441, 'chr18': 80373285, 'chr20': 64444167,
                'chr19': 58617616, 'chrY': 57227415, 'chr22': 50818468, 'chr21': 46709983
            }
        }
        if genome not in chr_sizes.keys():
            raise Exception('geneome must be either hg19 or hg38')
        if chromosome not in chr_sizes[genome].keys():
            raise Exception('chromosome must be in chr1, chr2, ..., chr22, chrX and chrY')
        chr_size = chr_sizes[genome][chromosome]
        num_windows = math.ceil(chr_size / window_size)
        N = len(list_of_bed_files)

        tracks = np.zeros((N, num_windows), dtype=np.bool_) 
        for file_idx in tqdm(range(N), desc=f'Reading {N} files'):
            with open(list_of_bed_files[file_idx], 'r') as f:
                lines = [line for line in f.readlines() if line.strip().split('\t')[0] == chromosome]
                for line in lines:
                    start, end = int(line.strip().split('\t')[1]), int(line.strip().split('\t')[2])
                    start /= window_size
                    end /= window_size
                    start = int(start)
                    end = math.ceil(end)
                    tracks[file_idx][start: end] = 1
        
        if at_least_one_target is None:
            at_least_one_target = np.arange(start=0, stop=N, step=1)
        else:
            at_least_one_target = np.array(at_least_one_target)

        filtered_index = []
        filtered_label = []
        for i in tqdm(range(num_windows), desc=f'Ranging from windows'):
            label = tracks[:, i]
            if np.sum(label[at_least_one_target]) == 0:
                continue
            real_seq = hg19.fetch(chromosome, (i * window_size, (i+1) * window_size))
            nn = 0
            for c in real_seq:
                if c == 'N':
                    nn += 1
            if nn / window_size > 0.4:
                continue
            filtered_index.append(np.array([chromosome_num, i * window_size, (i+1) * window_size, 1], dtype=np.int_)) # num_data
            filtered_label.append(label) # N

        return filtered_index, filtered_label
        
    if chromosomes == None:
        chrs = [
            '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
            '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X', 'Y'
        ]
    else:
        chrs = chromosomes
    
    bar = tqdm(chrs)
    index = []
    label = []
    for chr in bar:
        bar.set_description(f'Handling chromosome {chr}')
        i, l = bed2json_chr(
            list_of_bed_files,
            window_size,
            genome,
            'chr'+chr,
            chr_to_num[chr],
            at_least_one_target
        )
        index.extend(i)
        label.extend(l)
    label = np.stack(label, axis=0)
    index = np.stack(index, axis=0)
    print(f'File {str(output_file)} with {len(label)} samples.')
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('index', data=index, compression='gzip')
        f.create_dataset('label', data=label, compression='gzip')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate chromatin feature dataset from track file')
    parser.add_argument('--target-dataset', type=str, choices=['encode690', 'track1925'], default='encode690')
    parser.add_argument('--window-size', type=int, default=200)
    parser.add_argument('--data-root', type=str, default='../../data')
    parser.add_argument('--valid-chrs', nargs='+', default=['7'])
    parser.add_argument('--test-chrs', nargs='+', default=['8', '9'])
    args = parser.parse_args()
    
    train_chrs = [
            '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
            '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X', 'Y'
        ]
    valid_chrs = args.valid_chrs
    test_chrs = args.test_chrs
    for chr in valid_chrs + test_chrs:
        train_chrs.remove(chr)
    chrs = {
        'train': train_chrs,
        'valid': valid_chrs,
        'test': test_chrs,
    }
    # website:
    # https://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeAwgDnaseUniform/
    # https://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeAwgTfbsUniform/
    # https://personal.broadinstitute.org/anshul/projects/roadmap/peaks/consolidated/narrowPeak/
    if args.target_dataset == 'encode690':
        files = sorted(glob('resources/wgEncodeAwgTfbsUniform/*'))
    elif args.target_dataset == 'track1925':
        files = sorted(glob('resources/wgEncodeAwgTfbsUniform/*')) + \
            sorted(glob('resources/wgEncodeAwgDnaseUniform/*')) + \
            sorted(glob('resources/Roadmap/*'))
    
    output_dir = pathlib.Path(args.data_root) / args.target_dataset
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / 'peakFile.txt', 'w') as f:
        for file in files:
            print(file, file=f)
    with open(output_dir / 'chr_to_num.json', 'w') as f:
        json.dump(chr_to_num, f)

    for split in ['valid', 'train', 'test']:
        bed2json(
            list_of_bed_files=files,
            window_size=args.window_size,
            output_file=output_dir / f'{split}.h5',
            genome='hg19',
            chromosomes=chrs[split],
            at_least_one_target=list(range(690))
        )
