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