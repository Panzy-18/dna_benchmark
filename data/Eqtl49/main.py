import json
import os
from utils import generate_pos, generate_neg, filter_genomic_element, split_by_proportion

with open('./source/by_tissue/all_tissues.txt', 'r') as f:
        all_tissues = json.loads(f.readline().strip())
all_tissues = [('./source/by_tissue/'+tissue.lower()+'.txt') for tissue in all_tissues]

print(f'\nGenerate postive examples...')
generate_pos.generate_pos(
    output_file='pos',
    upstream_delta=0,
    downstream_delta=0,
    by_tissue_files=all_tissues,
    pip_threshold=.5
)

print(f'\nGenerate negative examples...')
generate_neg.generate_neg(
    output_file='neg',
    pos_data_path='pos',
    neg_pool_path='./source/dapg_sorted.txt',
    upstream_delta=0,
    downstream_delta=0,
    neg_sample_fold_size=1
)

print(f'\nFilter out examples that falls in exon regions...')
os.system('cat pos >> data')
os.system('cat neg >> data')
filter_genomic_element.filter_genomic_element(
    input_file='data',
    gtf_file='./source/exons.gtf',
    output_file='data_filtered'
)

print(f'\nSplitting data...')
if not os.path.isdir('./eqtl49'):
    os.mkdir('./eqtl49')
split_by_proportion.split_by_proportion(
    output_folder='./eqtl49',
    files=['data_filtered']
)

print(f'\nFinishing...')
os.system('rm pos neg data data_filtered')