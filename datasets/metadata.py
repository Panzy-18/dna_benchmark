import sys
sys.path.append("..")
from tools import get_logger
from dataclasses import asdict, dataclass, field
import json

logger = get_logger(__name__)

@dataclass
class DatasetMetadata:
    dataset_name: str = None
    dataset_args: dict = field(default_factory=dict)
    # fields: 
    # dataset_class, ref_file, train/valid/test_file, flank_length, doublet ...
    # all key-value in dataset_args will be passed when initializing dataset object
    model_args: dict = field(default_factory=dict)
    # fields:
    # task, final_dim, loss_fn(dict)
    # all key-value in model_args will be stored in config.
    metrics: list = None
    
    description: str = None # Option
    extra: dict = None # Option
    # fields: any
    # all key-value in extra will be passed when initializing dataset object
    
    def __post_init__(self):
        for k in list(self.dataset_args.keys()):
            if k in ['data_root']:
                logger.warning(f'please indicate {k} in argsparser. This will be ignored in dataset_args.')
                self.dataset_args.pop(k)
            if k in ['data_file']:
                logger.warning(f'please specify train/valid/test_file. {k} will be ignored in metadata.')
                self.dataset_args.pop(k)
            if k in ['dataset_name',' genome', 'split', 'tokenizer']:
                logger.warning(f'{k} will be ignored in dataset_args.')
                self.dataset_args.pop(k)
        if self.extra is not None:
            self.dataset_args.update(self.extra)

    @classmethod
    def load(cls, file):
        logger.info(f'Load metadata from {file}')
        with open(file, 'r') as f:
            return cls(**json.load(f))
    
    def save(self, file):
        logger.info(f'Save metadata to {file}')
        with open(file, 'w') as f:
            kwargs = asdict(self)
            json.dump(kwargs, f)
