import os
from utils import generate_data, split

if not os.path.isfile('./tss_epdnew.bed'):
    os.system('bash download_tss_epdnew_bed.sh')

if not os.path.isdir('./prom300'):
    os.mkdir('./prom300')
if not os.path.isdir('./prom500'):
    os.mkdir('./prom500')

generate_data.generate_positive(output_path='./prom300_pos.txt', seq_length=300)
generate_data.generate_negative(output_path='./prom300_neg.txt', seq_length=300)
split.split_by_proportion(
    output_folder='./prom300',
    files=['prom300_neg.txt', 'prom300_pos.txt']
)

generate_data.generate_positive('./prom500_varying_tss_pos.txt')
generate_data.generate_negative('./prom500_varying_tss_neg.txt')
split.split_by_proportion(
    output_folder='./prom500',
    files=['prom500_varying_tss_neg.txt', 'prom500_varying_tss_pos.txt']
)

os.system('rm *.txt')