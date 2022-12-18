from .tokenizer import BaseTokenizer
from collections import OrderedDict
import itertools
from typing import *

class KmerTokenizer(BaseTokenizer):
    
    method = 'kmer'
    seq_elements = 'NAGCT'
    pad_bp = 'N'
    
    def __init__(self, kmer: int, max_input_bp_seq_len: int):
        super().__init__(max_input_bp_seq_len)
        self.vocab = OrderedDict([
            (self.pad_token, self.pad_token_id),
            (self.mask_token, self.mask_token_id),
            (self.unknown_token, self.unknown_token_id),
            (self.cls_token, self.cls_token_id),
        ])
        self.kmer = kmer
        seq_elements = tuple(self.seq_elements)
        
        repeat = tuple([seq_elements] * self.kmer)
        for seq in itertools.product(*repeat):
            seq = ''.join(seq)
            self.vocab[seq] = len(self.vocab)
        
        self.id_to_token = OrderedDict()
        for k, v in self.vocab.items():
            self.id_to_token[v] = k
    
    def __len__(self):
        return len(self.vocab)

    def tokenize(self, seq: str) -> List[str]:
        if not seq:
            return []
        
        tokenized_seq = []
        for i in range(len(seq) - self.kmer + 1):
            tokenized_seq.append(seq[i : i + self.kmer])
        return tokenized_seq

    def convert_token_to_id(self, token: str) -> int:
        if token not in self.vocab:
            return self.unknown_token_id
        else:
            return self.vocab[token]
    
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.convert_token_to_id(token) for token in tokens]
    
    def encode(self, seq: str) -> List[int]:
        tokens = self.tokenize(seq)
        return self.convert_tokens_to_ids(tokens)

    def apply_random_mask(
        self,
        seq: str,
        mask_ratio: float,
        ids: List[str] = None,
        return_type: str = 'py',
    ):
        noised_seq = self.apply_random_mask_on_dnaseq(seq, mask_ratio)
        return self(noised_seq, return_type)