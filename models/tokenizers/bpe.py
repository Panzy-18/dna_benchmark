from tokenizers import SentencePieceBPETokenizer
from .tokenizer import BaseTokenizer
from typing import *

class BPETokenizer(BaseTokenizer):
    
    method = 'bpe'
    
    def __init__(self,
                 vocab_file: str,
                 merge_file: str,
                 max_input_bp_seq_len: int,
    ) -> None:
        super().__init__(max_input_bp_seq_len)
        self._tokenizer = SentencePieceBPETokenizer.from_file(
            vocab_file,
            merge_file,
            unk_token=self.unknown_token,
            add_prefix_space=False,
        )
    
    def __len__(self):
        return self._tokenizer.get_vocab_size()
    
    def encode(self, seq: str) -> List[int]:
        return self._tokenizer.encode(seq).ids
    
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
            replace_token_ids = range(5, len(self))
        )
        return super().wrap_to_type(masked_ids, return_type)
