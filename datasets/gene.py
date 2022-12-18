from typing import Callable, Union
from itertools import chain
from pathlib import Path
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
from .base import DNATaskDataset, Genome, HDF5Dataset
from .metric import *
from .utils import pad_sequences, sigmoid
import bmtrain as bmt
import numpy as np
import torch
import random
from tqdm import tqdm
import sys
sys.path.append("..")
from tools import get_args
import pdb

__all__ = [
    'PretrainDataset',
    'ExpressionDataset',
    'QTLDataset',
    'ExpressionFeatureDataset',
]

class PretrainDataset(Dataset):
    
    def __init__(self,
                 genome: Genome,
                 max_length: int,
                 mask_ratio: float,
                 tokenizer: Callable,
                 **kwargs
    ) -> None:
        super().__init__()
        self.dataset_name = 'pretrain'
        self.max_length = max_length
        self.mask_ratio = mask_ratio
        self.genome = genome
        self.tokenizer = tokenizer
        self.data = self.generate_data()
    
    def generate_data(self):
        all_chrs = list(self.genome.num_to_chr.values())
        data = []
        for chr in all_chrs:
            chr_length = len(self.genome.fa[chr])
            start = random.randint(0, self.max_length)
            end = start
            while start < chr_length:
                end = start + self.max_length
                if end > chr_length:
                    break
                data.append((chr, start, end, '+'))
                start = end
        return data

    def __len__(self):
        if get_args().debugging:
            return min(1000, len(self.data))
        
        return len(self.data)
    
    def __getitem__(self, index):
        index = self.data[index]
        sequence = self.genome.get_sequence(index = index)
        return dict(
            sequence = sequence,
            antisense_sequence = Genome.get_antisense(sequence)
        )
    
    def collate_fn(self, batch):
        '''
        交替正链与反链 [0-forward, 1-antisense], [2-forward, 3-antisense] ...
        '''
        batch = default_collate(batch)
        sequences = list(chain.from_iterable(zip(
            batch['sequence'], batch['antisense_sequence']
        )))
        label_ids = []
        masked_ids = []
        for seq in sequences:
            id = self.tokenizer(seq, return_type='py')
            masked_id = self.tokenizer.apply_random_mask(
                seq=seq,
                ids=id,
                mask_ratio=self.mask_ratio,
                return_type='py',
            )
            label_ids.append(torch.tensor(id))
            masked_ids.append(torch.tensor(masked_id))
        
        input_ids = pad_sequences(masked_ids, self.tokenizer.pad_token_id)
        labels = pad_sequences(label_ids, self.tokenizer.pad_token_id)
        return dict(
            input_ids = input_ids,
            labels = labels
        )


class QTLDataset(DNATaskDataset):
    
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        sequence = sample['sequence']
        sample['alter_sequence'] = sequence[:self.flank_length] + sample['alter'] + sequence[self.flank_length + 1:]
        return sample
    
    def collate_fn(self, batch):
        batch = {
            key: [i[key] for i in batch] for key in batch[0]
        }
        batch['labels'] = torch.from_numpy(np.array(batch['label'])).to(torch.float)
        sequences = list(chain.from_iterable(zip(batch['sequence'], batch['alter_sequence'])))
        batch['input_ids'] = self.tokenizer(sequences, return_type='pt')
        batch['id'] = torch.tensor(batch['id'])
        
        batch.pop('index')
        batch.pop('alter')
        batch.pop('sequence')
        batch.pop('alter_sequence')
        batch.pop('label')
        
        return batch
        

class ExpressionDataset(DNATaskDataset):
    
    def __init__(self, 
                 data_root: Union[str, Path], 
                 data_file: Union[str, Path], 
                 dataset_name: str = '',
                 pseudocount: float = 0.0001, 
                 chunk_length: int = 200,
                 track_flank_length: int = 400,
                 split: str = '', 
                 genome: Genome = None, 
                 flank_length: int = 0, 
                 tokenizer: Callable = None, 
                 doublet: bool = False, 
                 **kwargs) -> None:
        self.pseudocount = pseudocount
        super().__init__(data_root, data_file, dataset_name, split, genome, flank_length, tokenizer, doublet, **kwargs)
        self.chunk_length = chunk_length
        self.track_flank_length = track_flank_length
        self.track_total_length = chunk_length + 2 * track_flank_length
    
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        # pdb.set_trace()
        label = np.log(np.array(sample['label']) + self.pseudocount)
        label = torch.from_numpy(label)
        label = torch.nan_to_num(label, nan=0, posinf=0, neginf=0)
        sample['logp_label'] = label.tolist()
        return sample
    
    def collate_fn(self, batch):
        assert len(batch) == 1
        item = batch[0]
        sequence = item['sequence']
        n_chunk = (len(sequence) - self.track_flank_length * 2) // self.chunk_length
        chunks = [sequence[i*self.chunk_length: (i+1)*self.chunk_length + self.track_total_length] for i in range(n_chunk)]
        
        input_ids = self.tokenizer(chunks, return_type='pt')
        labels = torch.tensor(item['logp_label'])
        id = torch.tensor(item['id'])
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            id=id,
        )

class ExpressionFeatureDataset(Dataset):
    
    def __init__(self,
                 data_root: Union[str, Path],
                 data_file: Union[str, Path],
                 **kwargs
    ) -> None:
        super().__init__()
        data_root = Path(data_root)
        data_file = str(data_file)
        if not data_file.endswith('.h5'):
            raise ValueError('Only support h5, please use customized dataset.')
        self.data = HDF5Dataset(data_file=data_root/data_file)
        for k, v in kwargs.items():
            setattr(self, k, v)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        input_embeddings = torch.tensor(item['feature'])
        labels = torch.tensor(item['label'])
        return dict(
            input_embeddings=input_embeddings,
            labels=labels
        )
    
    def collate_fn(self, batch):
        return default_collate(batch)
    