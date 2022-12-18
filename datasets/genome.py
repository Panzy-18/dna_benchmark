import sys
sys.path.append('..')
from tools.logging import get_logger
import pyfastx
from typing import Union, List, Tuple
from pathlib import Path
import numpy as np
from copy import deepcopy

logger = get_logger(__name__)

class Genome(object):
    
    num_to_chr = {
        1: 'chr1', 2:'chr2', 3:'chr3', 4:'chr4',
        5: 'chr5', 6:'chr6', 7:'chr7', 8:'chr8',
        9: 'chr9', 10:'chr10', 11:'chr11', 12:'chr12',
        13: 'chr13', 14:'chr14', 15:'chr15', 16:'chr16',
        17: 'chr17', 18:'chr18', 19:'chr19', 20:'chr20',
        21: 'chr21', 22:'chr22', 23:'chrX', 24:'chrY',
    }
    
    @staticmethod
    def get_antisense(seq: str):
        reverse_map = {'A':'T', 'G':'C', 'C':'G', 'T':'A', 'N':'N'}
        reverse_seq = ''.join([reverse_map[x] for x in seq[::-1]])
        return reverse_seq
    
    def __init__(self,
                 data_root: Union[str, Path],
                 ref_file: Union[str, Path],
    ) -> None:
        ref_file = Path(data_root) / ref_file
        fa = pyfastx.Fasta(str(ref_file), uppercase=True)
        self.fa = {}
        for chr in self.num_to_chr.values():
            self.fa[chr] = fa.fetch(chr, (1, len(fa[chr])))
    
    def fetch(self,
              chrom: str,
              start: int,
              end: int,
    ):
        l, r = '', ''
        if start < 0:
            logger.warning(f'Try to index {chrom} at {start}. You may check the data.')
            l = 'N' * (0 - start)
            start = 0
        if end > len(self.fa[chrom]):
            logger.warning(f'Try to index {chrom} at {end}. You may check the data.')
            r = 'N' * (end - len(self.fa[chrom]))
            end = len(self.fa[chrom])
        return l + self.fa[chrom][start: end] + r
    
    def flank(self,
              chrom: str,
              start: int,
              end: int,
              flank_length: int,
    ):
        if flank_length == 0:
            return ('', '')
        
        l_start = max(start - flank_length, 0)
        r_end = min(end + flank_length, len(self.fa[chrom]))
        return (self.fa[chrom][l_start: start], self.fa[chrom][end: r_end])
    
    def get_sequence(self,
                     index: Union[List, Tuple, np.ndarray],
                     flank_length: int = 0
    ) -> str:
        # index 有两种形式, 0-based, 左闭右开
        # list: ['chr1', 0, 200, '-']
        # np: (1, 0, 200, 0)
        if isinstance(index, np.ndarray):
            index = index.tolist()
        index = deepcopy(index)
        chrom = index[0] if isinstance(index[0], str) else self.num_to_chr[index[0]]
        start = int(index[1])
        end = int(index[2])
        forward = 0 if index[3] == '-' or index[3] == 0 else 1
        # 检查是否越界
        
        sequence = self.fetch(
            chrom=chrom,
            start=start-flank_length,
            end=end+flank_length
        )
            
        if not forward:
            sequence = self.get_antisense(sequence)
        
        return sequence

        