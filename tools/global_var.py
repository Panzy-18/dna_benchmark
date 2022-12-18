from easydict import EasyDict as edict
import os
import bmtrain as bmt
from .envs import parse_args, init_bmt

args = parse_args()
init_bmt(args)
GLOBAL_VAR = edict({
    "_logger": {},
    "_writer": None,
    "_args": args,
    "_config": None,
    "_metadata": None,
})

def modify_args_by_kv(kv: dict):
    for k, v in kv.items():
        if not hasattr(GLOBAL_VAR._args, k):
            raise ValueError(f'invalid key: {k}')
        setattr(GLOBAL_VAR._args, k, v)

def clean_file():
    os.system('rm {}'.format(os.path.join(GLOBAL_VAR._args.save_dir, 'args_kw')))
    os.system('rm {}'.format(os.path.join(GLOBAL_VAR._args.save_dir, 'config.json')))
    os.system('rm {}'.format(GLOBAL_VAR._args.log_file))

def get_args():
    return GLOBAL_VAR._args