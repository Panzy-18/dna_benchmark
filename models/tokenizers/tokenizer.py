import pdb
from typing import *
from copy import copy
import numpy as np
import random
import torch

class BaseTokenizer:
    
    method = 'base'
    
    pad_token = '<PAD>'
    mask_token = '<MASK>'
    unknown_token = '<UNK>'
    cls_token = '<CLS>'
    
    pad_token_id = 0
    mask_token_id = 1
    unknown_token_id = 2
    cls_token_id = 3
    
    vocab = {}
    
    max_input_bp_seq_len = 1024
    
    def __init__(self, max_input_bp_seq_len: int) -> None:
        self.max_input_bp_seq_len = max_input_bp_seq_len
    
    @staticmethod
    def apply_random_mask_on_token_ids(
        ids: List[int],
        mask_ratio: float,
        mask_token_id: int,
        replace_token_ids: Sequence,
    ):
        indice = list(range(len(ids)))
        random.shuffle(indice)
        indice = indice[:int(len(ids) * mask_ratio + 0.5)]
        
        masked_ids = copy(ids)
        for i in indice:
            randice = random.random()
            if randice < 0.85:
                masked_ids[i] = mask_token_id
            else:
                masked_ids[i] = random.choice(replace_token_ids)
        
        return masked_ids
    
    @staticmethod
    def apply_random_mask_on_dnaseq(
        seq: str, 
        mask_ratio: float,
        mask_bp: str = 'N',
        replace_bps: list = ['A','G','C','T'],
    ) -> str:
        '''
        the mask strategy is that:
            80% changed to mask (with N)
            10% single base change
        '''
        def random_bps(length=1) -> str:
                s = ''
                for _ in range(length):
                    s += random.choice(replace_bps)
                return s

        # 进行迭代式的
        noised_seq = ''
        for bp in seq:
            if random.random() < mask_ratio:
                # add noise
                randice = random.random()
                if randice < 0.85:
                    noised_seq += mask_bp
                else:
                    noised_seq += random_bps(1)
            else:
                noised_seq += bp

        return noised_seq
    
    def apply_random_mask(
        self,
        seq: str,
        mask_ratio: float,
        ids: List[str] = None,
        return_type: str = 'py',
    ) -> Sequence:
        ...
        
    def encode(self, seq: str) -> List[int]:
        ...
        
    def __call__(self, 
                 seq: Union[str, List[str], Tuple[str]],
                 return_type: str = 'py',
    ) -> Sequence:
        if isinstance(seq, str):
            if len(seq) > self.max_input_bp_seq_len:
                seq = seq[:self.max_input_bp_seq_len]
            ids = self.encode(seq)
            if return_type == 'np':
                ids = np.array(ids)
            elif return_type == 'pt':
                ids = torch.tensor(ids)
            return ids
        elif isinstance(seq, Iterable):
            ids = [self(x, return_type=return_type) for x in seq]
            if return_type == 'py':
                return ids
            else:
                return pad_sequences(
                    sequences=ids,
                    pad_value=self.pad_token_id
                )
        else:
            raise ValueError
    

def pad_sequences(
    sequences: Union[List[np.ndarray], List[torch.Tensor]],
    pad_value = 0,
    target_shape: Union[int, List[int], Tuple[int]] = None
):
    '''
    pad 1d or more, input shape: [bsz, ...]
    '''
    batch_size = len(sequences)
    if target_shape is None:
        shape = [batch_size] + np.max([seq.shape for seq in sequences], 0).tolist()
    else:
        if isinstance(target_shape, int):
            target_shape = [target_shape]
        else:
            target_shape = list(target_shape)
        shape = [batch_size] + target_shape
    
    dtype = sequences[0].dtype
    if isinstance(sequences[0], np.ndarray):
        array = np.full(shape, pad_value, dtype=dtype)
    elif isinstance(sequences[0], torch.Tensor):
        array = torch.full(shape, pad_value, dtype=dtype)
    
    for arr, seq in zip(array, sequences):
        arrslice = tuple(slice(dim) for dim in seq.shape)
        arr[arrslice] = seq
        
    return array
