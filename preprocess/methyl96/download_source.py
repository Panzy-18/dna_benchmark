import os
from tqdm import tqdm

reads_count_threshold = 10

with open('./filenames.txt', 'r') as input:
    lines = input.readlines()
files = [line.strip().split('\t') for line in lines if line != '']

os.system('mkdir -p sources')
# os.system('rm *.gz')
for file in tqdm(files):
    encode_name, download_name = file[0], file[1]
    os.system(f'wget https://www.encodeproject.org/files/{encode_name}/@@download/{encode_name}.bed.gz -O sources/{download_name}.bed.gz')
    print(f'\nUnzip file...')
    os.system(f'gunzip sources/{download_name}.bed.gz')

    print('Filter low read count records...')
    with open(f'sources/{download_name}.bed', 'r') as input:
        lines1 = input.readlines()
    with open(f'sources/{download_name}_filtered.bed', 'w') as output:
        for idx, line in tqdm(enumerate(lines1)):
            if idx == 0:
                output.write(line.strip()+'\n')
                continue
            reads_count = (line.strip().split('\t'))[-2]
            reads_count = int(reads_count)
            if reads_count >= reads_count_threshold:
                output.write(line.strip()+'\n')
    os.system(f'rm sources/{download_name}.bed')