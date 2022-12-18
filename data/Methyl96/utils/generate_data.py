import os
import json
import numpy as np
from tqdm import tqdm

def custom_calc_percent(total, divident):
    if total == 0:
        return 0
    else:
        return divident / 100 / total

def generate(
    output_folder: str,
    source_folder: str = '../source',
    total_groups: int = 96,
    train_chrs: list[str] = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '14', '15', '16', '17', '18', '19', '20', '21', '22'],
    valid_chrs: list[str] = ['12', '13'],
    test_chrs: list[str] = ['10', '11', 'X', 'Y']
):
    files = os.listdir(source_folder)
    files = [f for f in files if f[-4:]=='.bed']
    D, D_TRAIN, D_VALID, D_TEST = None, {}, {}, {}
    
    for group_idx in tqdm(range(total_groups)):
        l = len(str(group_idx))
        files1 = [f1 for f1 in files if f1[:l]==str(group_idx) and f1[l+1]=='_']
        for f2 in tqdm(files1):
            with open(source_folder+'/'+f2) as input:
                lines = (input.readlines())[1:]
            for line in tqdm(lines):
                d = line.strip().split('\t')
                if d[0][3:] in train_chrs:
                    D = D_TRAIN
                elif d[0][3:] in valid_chrs:
                    D = D_VALID
                elif d[0][3:] in test_chrs:
                    D = D_TEST
                else:
                    continue
                start, end, strand, total_reads, pos_rate = int(d[1]), int(d[2]), d[5], int(d[9]), int(d[10])
                seq_index = json.dumps([d[0], start, end, strand])
                if seq_index not in D:
                    D[seq_index] = np.zeros((total_groups, 2), dtype='int') # [sum of all reads, sum of methylated reads]
                D[seq_index][group_idx][0] += total_reads
                D[seq_index][group_idx][1] += total_reads * pos_rate # pos_rate is in percent here
    

    sets = ['train', 'valid', 'test']
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    for i, D in enumerate([D_TRAIN, D_VALID, D_TEST]):
        set_ = sets[i]
        with open(output_folder + '/' + set_ + '.txt', 'w') as output:
            for seq_index in tqdm(D):
                methyl_stat = np.zeros((total_groups,))
                for i in range(total_groups):
                    methyl_stat[i] = custom_calc_percent(D[seq_index][i][0], D[seq_index][i][1])
                to_be_written = {
                    'seq_index': json.loads(seq_index),
                    'label': methyl_stat.tolist()
                }
                output.write(json.dumps(to_be_written)+'\n')
