import numpy as np
import pandas as pd
import argparse
import random
import json
import pathlib

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate CAGE expression dataset from track file')
    parser.add_argument('--data-root', type=str, default='../../data')
    parser.add_argument('--valid-chrs', nargs='+', default=['X', 'Y'])
    parser.add_argument('--test-chrs', nargs='+', default=['8'])
    parser.add_argument('--seed', type=int, default=3407)
    args = parser.parse_args()
    random.seed(args.seed)
    output_dir = pathlib.Path(args.data_root) / 'expression218'
    output_dir.mkdir(exist_ok=True)
    index_df = pd.read_csv('resources/geneanno.csv')
    exp_df = pd.read_csv('resources/geneanno.exp.csv')
    
    datas = []
    for i in range(len(index_df)):
        if index_df.iloc[i, -1] == 'rRNA':
            continue
        start = int(index_df.iloc[i, 5])
        start = (start // 200) * 200
        end = start + 200
        index = [index_df.iloc[i, 2], start, end, index_df.iloc[i, 3]]
        exp = exp_df.iloc[i, 1:].astype(float).tolist()
        datas.append({
            "index": index,
            "label": exp,
        })
    
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
    for split in ['valid', 'test', 'train']:
        output_file = output_dir / f'{split}.json'
        with open(output_file, 'w') as f:
            for sample in datas:
                if sample['index'][0][3:] in chrs[split]:
                    print(json.dumps(sample), file=f)