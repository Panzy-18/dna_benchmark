import os
import json

if not os.path.isdir(os.getcwd() + '/by_tissue'):
    os.mkdir(os.getcwd() + '/by_tissue')

sorted_file = 'dapg_sorted.txt'
tissues = []

with open(sorted_file, 'r') as input:
    curr_tissue, prev_tissue = '', ''
    curr_gene, prev_gene = '', ''
    curr_cluster, prev_cluster = -1, -1
    output = None


    while True:
        line = input.readline()
        if line == '':
            break
        
        [curr_tissue, curr_gene, curr_cluster, cluster_p, var, var_p] = line.strip().split('\t')
        curr_cluster, cluster_p, var_p = int(curr_cluster), float(cluster_p), float(var_p)
        
        # {chr}_{pos_first_ref_base}_{ref_seq}_{alt_seq}_b38
        [chr, loc, ref, mut] = var.strip().split('_')[:4]
        loc = int(loc)-1 # 1-based to 0-based


        do_write = (curr_tissue != prev_tissue) or (curr_gene != prev_gene) or (curr_cluster != prev_cluster)

        if curr_tissue != prev_tissue:
            prev_tissue = curr_tissue
            tissues.append(curr_tissue)

            if output is not None:
                output.close()
            output = open(os.getcwd() + f'/by_tissue/{curr_tissue.lower()}.txt', 'w')
                    
        if curr_gene != prev_gene:
            prev_gene = curr_gene
        if curr_cluster != prev_cluster:
            prev_cluster = curr_cluster

        if (len(ref) == 1):
            if do_write:
                output.write(f'{chr}\t{loc}\t{loc+1}\t{mut}\t{cluster_p}\t{var_p}\n')

with open(os.getcwd() + '/by_tissue/all_tissues.txt', 'w') as output1:
    output1.write(json.dumps(tissues))