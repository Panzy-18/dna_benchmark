from .tokenizer import BaseTokenizer
from collections import OrderedDict
import itertools
from typing import *

class OneHotTokenizer(BaseTokenizer):
    
    method = 'onehot'
    
    def __init__(self, max_input_bp_seq_len: int) -> None:
        super().__init__(max_input_bp_seq_len)
        self.mask_token_id = 0
        self.unknown_token_id = 0
        self.vocab = OrderedDict([
            ('A', 1),
            ('T', 2),
            ('G', 3),
            ('C', 4),
            ('N', 0),
        ])
    
    def __len__(self):
        return 5

    def apply_random_mask(
        self,
        ids: List[str],
        mask_ratio: float,
        seq: str = None,
        return_type: str = 'py',
    ):
        masked_ids = self.apply_random_mask_on_token_ids(
            ids = ids,
            mask_ratio = mask_ratio,
            mask_token_id = self.mask_token_id,
            replace_token_ids = range(1, 5)
        )
        return super().wrap_to_type(masked_ids, return_type)
    
    def encode(self, seq: str) -> List[int]:
        ids = []
        for x in seq:
            if x not in 'AGCTN':
                ids.append(self.unknown_token_id)
            else:
                ids.append(self.vocab[x])
        
        return ids
