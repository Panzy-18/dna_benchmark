import os
from .global_var import GLOBAL_VAR
from .logging import get_logger
import os
from glob import glob

logger = get_logger(__name__)

__all__ = [
    'get_config',
    'get_metadata',
]

from models import Config
from datasets import DatasetMetadata

def get_config(
    config_path: str = None
):
    if GLOBAL_VAR._config is not None:
        return GLOBAL_VAR._config
    
    if config_path is not None:
        config = Config.load(config_path)
    
    else:
        args = GLOBAL_VAR._args
        config_path = args.config_path
        if args.load_dir is not None:
            _json_list = glob(os.path.join(args.load_dir, '*.json'))
            if len(_json_list) == 1:
                config_path = _json_list[0]
            else:
                raise ValueError('please specify the config path.')
        config = Config.load(config_path)
    
    if args.tscam:
        config.tscam = True
    
    if GLOBAL_VAR._metadata is not None:
        config.update_from_metadata(GLOBAL_VAR._metadata)
    else:
        if args.load_dir is None:
            logger.warning('Config dose not update from metadata. It may loss model construction args.')
    
    if not GLOBAL_VAR._args.no_log:
        config.save(os.path.join(args.save_dir, 'config.json'))
    GLOBAL_VAR._config = config

    return config

def get_metadata(
    metadata_path: str = None
):
    if GLOBAL_VAR._metadata is not None:
        return GLOBAL_VAR._metadata
    
    if metadata_path is not None:
        metadata = DatasetMetadata.load(metadata_path)
    else:
        args = GLOBAL_VAR._args
        metadata_path = os.path.join(args.data_root, args.dataset_dir, 'metadata.json')
        metadata = DatasetMetadata.load(metadata_path)
    GLOBAL_VAR._metadata = metadata
    
    return metadata
    