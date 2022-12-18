import random
seed = 123456

def split_by_proportion(
    output_folder: str,
    files: list[str],
    ratios: list[float] = [.7, .1, .2]
):
    sets = ['train', 'valid', 'test']

    lines = []
    for f in files:
        with open(f, 'r') as input:
            lines += input.readlines()
    
    lines = [line for line in lines if line != '']
    random.seed(seed)
    random.shuffle(lines)

    t = len(lines)
    cutoff = 0
    for idx, p in enumerate(ratios):
        with open(output_folder+'/'+sets[idx]+'.txt', 'w') as output:
            output.writelines(lines[int(t*cutoff):int(t*(cutoff+p))])
            cutoff += p

if __name__ == '__main__':
    split_by_proportion(
        output_folder='../prom300',
        files=['prom300_neg.txt', 'prom300_pos.txt']
    )
    split_by_proportion(
        output_folder='../prom500',
        files=['prom500_varying_tss_neg.txt', 'prom500_varying_tss_pos.txt']
    )