import json
import os
import argparse
import random
import pathlib
import pdb
from utils import generate_pos, generate_neg, filter_genomic_element

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate chromatin feature dataset from track file')
    parser.add_argument('--data-root', type=str, default='../../data')
    parser.add_argument('--pip-th', type=float, default=0.5)
    parser.add_argument('--neg-fold', type=float, default=2)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--valid-ratio', type=float, default=0.1)
    parser.add_argument('--test-ratio', type=float, default=0.2)
    args = parser.parse_args()
    random.seed(args.seed)
    output_dir = pathlib.Path(args.data_root) / 'eqtl49'
    output_dir.mkdir(exist_ok=True)
    
    with open('./source/all_tissues.txt', 'r') as f:
        all_tissues = json.loads(f.readline().strip())
    all_tissue_files = [('./source/by_tissue/'+tissue+'.txt') for tissue in all_tissues]
    
    pos_data = generate_pos.generate_pos(
        upstream_delta=0,
        downstream_delta=0,
        by_tissue_files=all_tissue_files,
        pip_threshold=args.pip_th,
    )    
    neg_data = generate_neg.generate_neg(
        pos_data=pos_data,
        neg_pool_path='./source/dapg_sorted.txt',
        upstream_delta=0,
        downstream_delta=0,
        neg_sample_fold_size=args.neg_fold
    )
    pos_data = filter_genomic_element.filter_genomic_element(
        datas = pos_data,
        gtf_file = './source/exons.gtf',
    )
    print(f'Total pos data: {len(pos_data)} samples')
    neg_data = filter_genomic_element.filter_genomic_element(
        datas = neg_data,
        gtf_file = './source/exons.gtf',
    )
    print(f'Total neg data: {len(neg_data)} samples')
    # 首先将所有的数据保存
    with open(output_dir/'all_sample.json', 'w') as f:
        for data in pos_data+neg_data:
            print(json.dumps(data), file=f)
    
    # do random split
    for i, tissue in enumerate(all_tissues):
        output_file = output_dir/f'{i}:{tissue}:data_index.json'
        
        pos_index_for_cur_tissue = [j for j, l in enumerate(pos_data) if l['label'][i] == 1]
        neg_index_for_cur_tissue = random.sample(
            list(range(len(neg_data))),
            k = int(len(pos_index_for_cur_tissue) * args.neg_fold)
        )
        neg_index_for_cur_tissue = [i + len(pos_data) for i in neg_index_for_cur_tissue]
        all_index_for_cur_tissue = pos_index_for_cur_tissue + neg_index_for_cur_tissue
        random.shuffle(all_index_for_cur_tissue)
        
        print(f'File {str(output_file)} with {len(all_index_for_cur_tissue)} samples.')
        train_end = int(len(all_index_for_cur_tissue) * (1-args.valid_ratio-args.test_ratio))
        valid_end = int(len(all_index_for_cur_tissue) * (1-args.test_ratio))
        with open(output_file, 'w') as f:
            json.dump(dict(
                train=all_index_for_cur_tissue[:train_end],
                valid=all_index_for_cur_tissue[train_end:valid_end],
                test=all_index_for_cur_tissue[valid_end:]
            ), f)

        
        
        
    
    