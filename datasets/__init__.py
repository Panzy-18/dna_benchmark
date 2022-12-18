from typing import Callable
from .metadata import DatasetMetadata
from .base import DNATaskDataset
import sys
sys.path.append("..")
from tools import get_args
from .gene import *
from .genome import Genome
from .dataloader import DistributedDataLoader


def get_dataset(metadata: DatasetMetadata, 
                tokenizer: Callable,
                splits: list = None,
                **kwargs
):
    data_root = get_args().data_root
    genome = Genome(
        data_root=data_root,
        ref_file=metadata.dataset_args.get('ref_file')
    ) if metadata.dataset_args.get('ref_file') is not None else None
    
    dataset = {}
    splits = ['valid', 'test', 'train'] if splits is None else splits
    for split in splits:
        data_file = metadata.dataset_args.get(f'{split}_file')
        if data_file is None:
            continue
        dataset[split] = eval(metadata.dataset_args['dataset_class'])(
            data_root=data_root,
            data_file=data_file,
            dataset_name=metadata.dataset_name,
            split=split,
            genome=genome,
            tokenizer=tokenizer,
            metrics=metadata.metrics,
            **metadata.dataset_args,
            **kwargs
        )
    
    return dataset