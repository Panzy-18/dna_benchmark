from .tokenizer import BaseTokenizer
from .kmer import KmerTokenizer
from .bpe import BPETokenizer
from .patch import PatchTokenizer
from .onehot import OneHotTokenizer
import os

def get_tokenizer(
    method: str,
    max_input_bp_seq_len: int,
    **kwargs,
) -> BaseTokenizer:
    if method.startswith('kmer'):
        _, kmer = method.split('=')
        return KmerTokenizer(
            kmer=int(kmer), 
            max_input_bp_seq_len=max_input_bp_seq_len,
            **kwargs
        )
    elif method == 'bpe':
        cur_dir = os.path.dirname(__file__)
        return BPETokenizer(
            vocab_file = cur_dir + '/bpe_vocab.json',
            merge_file = cur_dir + '/bpe_merges.txt',
            max_input_bp_seq_len=max_input_bp_seq_len,
            **kwargs
        )
    elif method == 'onehot':
        return OneHotTokenizer(
            max_input_bp_seq_len=max_input_bp_seq_len,
            **kwargs
        )
    elif method.startswith('patch'):
        _, patch_size = method.split('=')
        return PatchTokenizer(
            patch_size=int(patch_size), 
            max_input_bp_seq_len=max_input_bp_seq_len,
            **kwargs
        )
    else:
        raise NotImplementedError