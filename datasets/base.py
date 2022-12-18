from typing import Callable, Union
from torch.utils.data import Dataset
from pathlib import Path
import sys
sys.path.append('..')
from tools import get_logger, get_args
import time
from tqdm import tqdm
import json
import h5py
import torch
import bmtrain as bmt
import numpy as np
from .genome import Genome
from .metric import STR2METRIC
from copy import deepcopy
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

logger = get_logger(__name__)

class JSONDataset(Dataset):
    
    def __init__(self,
                 data_file: Union[str, Path],
    ) -> None:
        super().__init__()
        self._data = []
        with open(data_file) as f:
            for line in tqdm(f, desc=f"loading data from {data_file}", disable=bmt.rank()!=0):
                self._data.append(json.loads(line.strip()))
    
    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, index) -> dict:
        return self._data[index]
    
class HDF5Dataset(Dataset):
    
    def __init__(self,
                 data_file: Union[str, Path],
    ) -> None:
        super().__init__()
        self.h5 = h5py.File(data_file, mode='r')
        self.fields = self.h5.keys()
        self.length = self.h5['label'].shape[0]
        self.buffer_size = int(0.05*self.length) if self.length > 100000 else self.length
        self.buffer_range = [0, min(self.buffer_size, self.length)]
        self.buffer_data = self.get_buffer()
    
    def get_buffer(self):
        return {
            field: self.h5[field][self.buffer_range[0]:self.buffer_range[1], ...] for field in self.fields
        }

    def __len__(self):
        return self.length
    
    def __getitem__(self, index) -> dict:
        if index >= self.buffer_range[1] or index < self.buffer_range[0]:
            self.buffer_range = [index, min(index + self.buffer_size, self.length)]
            self.buffer_data = self.get_buffer()
        
        item = {}
        for field in self.fields:
            item[field] = self.buffer_data[field][index - self.buffer_range[0]]
        return item
    
class DNATaskDataset(Dataset):
    
    def __init__(self,
                 data_root: Union[str, Path],
                 data_file: Union[str, Path],
                 dataset_name: str = '',
                 split: str = '',
                 genome: Genome = None,
                 flank_length: int = None,
                 tokenizer: Callable = None,
                 doublet: bool = None,
                 metrics: list = None,
                 **kwargs
    ) -> None:
        super().__init__()
        data_root = Path(data_root)
        data_file = str(data_file)
        start = time.time()
        if data_file.endswith('.json'):
            self.data = JSONDataset(data_file=data_root/data_file)
        elif data_file.endswith('.h5'):
            self.data = HDF5Dataset(data_file=data_root/data_file)
        else:
            raise ValueError('File format unknown, please use customized dataset')
        end = time.time()
        logger.info(f'load {split} data time consuming: {end-start}s')
        self.dataset_name = dataset_name
        self.split = split
        self.genome = genome
        self.flank_length = 0 if flank_length is None else flank_length
        self.tokenizer = tokenizer
        self.doublet = False if doublet is None else doublet
        self.metrics = {}
        if metrics:
            for metric_name in metrics:
                if STR2METRIC.get(metric_name) is not None:
                    self.metrics[metric_name] = STR2METRIC.get(metric_name)
        
        for k, v in kwargs.items():
            setattr(self, k, v)
        
        logger.info(
            '{}_{} contain {} samples, loading from {}\n'.format(self.dataset_name, split, len(self), str(data_root/data_file)) + \
            'sample from it:\n' + \
            '{}'.format(self[0])
        )
    
    def __len__(self):
        if get_args().debugging:
            return min(1000, len(self.data))
        
        if not self.doublet:
            return len(self.data)
        else:
            return len(self.data) * 2
    
    def process_sample(self, 
                       sample: dict,
                       antisense: bool = False,
    ) -> dict:
        # 将index转换为序列
        sample = deepcopy(sample)
        index_to_seq = {}
        for k, v in sample.items():
            if 'index' in k:
                sequence = self.genome.get_sequence(
                    index=v,
                    flank_length=self.flank_length
                )
                index_to_seq[k.replace("index", "sequence")] = sequence
            else:
                index_to_seq[k] = v
        
        # 判断序列是否需要转换为反义链
        if antisense:
            for k, v in index_to_seq.items():
                if 'sequence' in k:
                    anti_v = Genome.get_antisense(v)
                    index_to_seq[k] = anti_v

        return index_to_seq
    
    def __getitem__(self, index):
        sample = {}
        sample['id'] = str(index)
        if self.doublet:
            real_index = index // 2
            antisense = index % 2 == 1
            sample.update(self.process_sample(self.data[real_index], antisense=antisense))
        else:
            sample.update(self.process_sample(self.data[index]))
        
        return sample
    
    def collate_fn(self, batch):
        batch = {
            key: [i[key] for i in batch] for key in batch[0]
        }
        sequences = batch['sequence']
        batch['labels'] = torch.from_numpy(np.array(batch['label'])).to(torch.float)
        batch['input_ids'] = self.tokenizer(sequences, return_type='pt')
        
        batch.pop('sequence')
        batch.pop('label')
        
        return batch
    
    def metric_fn(self,
                  preds: np.ndarray,
                  targets: np.ndarray,
                  **kwargs
    ) -> dict[str, float]:
        if not self.metrics:
            return None
        
        if len(targets.shape) == 1:
            targets = np.expand_dims(targets, axis=1)
            preds = np.expand_dims(preds, axis=1)
        
        scores = {metric_name: [] for metric_name in self.metrics.keys()}
        for i in tqdm(range(targets.shape[-1]), desc='Calculating metric for each column', disable=bmt.rank()!=0):
            for metric_name, metric_func in self.metrics.items():
                scores[metric_name].append(metric_func(targets=targets[:, i], preds=preds[:, i]))
        
        avg_scores = {k: np.nanmean(v) for k, v in scores.items()}
        for k, v in avg_scores.items():
            avg_scores['main'] = v
            break
        
        return avg_scores