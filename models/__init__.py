import sys
sys.path.append("..")
from tools import get_args, get_logger
from .base import BaseModel
from .config import BaseConfig as Config
from .tokenizers import BaseTokenizer as Tokenizer
from .tokenizers import get_tokenizer
from .task import *
import bmtrain as bmt
import os
from glob import glob

logger = get_logger(__name__)

def get_model(
    config: Config,
    model_path: str = None,
) -> BaseModel:
    if config.task is None:
        raise ValueError('please assign task or update from metadata.')
    model = eval(config.task)(config=config)
    
    if model_path is not None:
        bmt.init_parameters(model)
        model.load(model_path)
    
    args = get_args()
    if args.load_dir is None:
        logger.info('init model from scratch')
        bmt.init_parameters(model)
    else:
        if args.ckpt_path is None:
            model_path = sorted(glob(os.path.join(args.load_dir, '*.pt')), key=os.path.getmtime)[-1]
            args.ckpt_path = os.path.basename(model_path)
        else:
            model_path = os.path.join(args.load_dir, args.ckpt_path)
        bmt.init_parameters(model)
        model.load(model_path)
        
    logger.info(
        '\n' + bmt.inspect.format_summary(bmt.inspect.inspect_model(model, '*')) +
        '\nmodel params : {}'.format(model.param_num(ignore_key=[]) * bmt.config["world_size"]), 
    )
    
    return model