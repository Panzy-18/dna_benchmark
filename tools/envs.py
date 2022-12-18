import argparse
import os
import json
from datetime import datetime
import sys 
import bmtrain as bmt
import random
import warnings

def add_model_config_args(parser: argparse.ArgumentParser):
    """Model arguments"""

    group = parser.add_argument_group('model', 'model configuration')

    group.add_argument('--load-dir', type=str, default=None,
                       help='Path to a directory containing a model checkpoint.')
    group.add_argument('--save-dir', type=str, default=None,
                       help='Output directory to save checkpoints to.')
    group.add_argument('--config-path', type=str, default='config/default_attention_config.json', 
                       help='name of model configuration file')
    group.add_argument('--ckpt-path', type=str, default=None, 
                       help='model checkpoint path')
    group.add_argument('--half', action='store_true') 
    return parser

def add_training_args(parser: argparse.ArgumentParser):
    """Training arguments."""

    group = parser.add_argument_group('train', 'training configurations')

    group.add_argument('--data-root', type=str, default='data',
                        help='the directory where data of all task stores at')
    group.add_argument('--dataset-dir', type=str, default=None,
                       help='Directory of the dataset')
    group.add_argument('--batch-size', type=int, default=32,
                       help='Data Loader batch size')
    group.add_argument('--clip-grad', type=float, default=1.0,
                       help='gradient clipping')
    group.add_argument('--seed', type=int, default=3407,
                       help='random seed for reproducibility')
    group.add_argument('--gradient-accumulation', type=int, default=1,)

    # group.add_argument('--train-iters', type=int, default=None,
    #                    help='total number of iterations to train over all training runs')
    group.add_argument('--epochs', type=int, default=20,
                       help='total number of epochs to train over all training runs. \
                             When setting epochs, overload train-iters.')
    # group.add_argument('--eval-strategy', type=str, default='iters', 
    #                    choices=['iters', 'epoch'],
    #                    help='strategy of evaluate and saving: iters/epoch')
    # group.add_argument('--eval-iters', type=int, default=None,
    #                    help='number of iterations between saves')

    # Learning rate.
    group.add_argument('--lr', type=float, default=1.0e-4,
                       help='initial learning rate')
    group.add_argument('--weight-decay', type=float, default=1.0e-2,
                       help='weight-decay')
    group.add_argument('--loss-scale', type=float, default=128,
                       help='loss scale')
    group.add_argument('--warmup-iters', type=float, default=0,)
    group.add_argument('--lr-decay-iters', type=int, default=1000000,
                       help='number of iterations to decay LR over,'
                       ' If None defaults to `--train-iters`*`--epochs`')
    group.add_argument('--lr-decay-style', type=str, default='constant',
                       choices=['constant', 'linear', 'cosine', 'exponential', 'noam'],
                       help='learning rate decay function')
    group.add_argument('--local_rank', type=int, default=None,
                       help='local rank passed from distributed launcher')
    
    return parser


def add_other_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('other', 'other configurations')
    
    # finetuning
    group.add_argument('--debugging', action='store_true',
                       help='to set a series of minimun paramters')
    group.add_argument('--eval-only', action='store_true',
                       help='not to train, just valid and test.')
    group.add_argument('--save-pred', action='store_true',
                       help='save all the preds and targets')
    group.add_argument('--early-stop', type=int, default=100000,
                       help='stop training after evaluating result not rising x times')
    group.add_argument('--tscam', action='store_true',
                       help='to set for tscam')
    
    # other
    group.add_argument('--no-log', action='store_true', 
                       help='clean log/config/args file when noe need to save')

    return parser

def save_args(args, file):
    with open(os.path.join(file), 'w') as f:
        json.dump(args.__dict__, f)

def parse_args():
    parser = argparse.ArgumentParser()
    parser = add_model_config_args(parser)
    parser = add_training_args(parser)
    parser = add_other_args(parser)
    args = parser.parse_args()
    
    # 一些基于参数的检查
    if args.load_dir in ['none', 'null', 'None']:
        args.load_dir = None
    if args.ckpt_path in ['none', 'null', 'None']:
        args.ckpt_path = None
    if args.save_dir is None:
        if os.environ.get('LOCAL_RANK') == str(0):
            warnings.warn('please assign save_dir. set save_dir to ./tmp')
        args.save_dir = 'tmp'
        args.no_log = True
    if args.eval_only:
        args.epochs = 0
    
    if not args.no_log:
        os.makedirs(args.save_dir, exist_ok=True)
        with open(os.path.join(args.save_dir, 'args_kw'), 'w') as f:
            json.dump(vars(args), f)
    
    args.save = lambda file: save_args(args, file)
    nowtime = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    args.log_file = os.path.join(args.save_dir, 'experiment_{}.log'.format(nowtime))

    return args

def init_bmt(args):
    if os.environ.get('LOCAL_RANK') is None:
        opts = ' '.join(sys.argv[1:])
        dist_opts = "--nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port {}".format(random.randint(27000, 28000))
        cmd = 'torchrun {} {} {}'.format(
            dist_opts, sys.argv[0], opts
        )
        print(cmd)
        os.system(cmd)
        sys.exit()
    else:
        bmt.init_distributed(seed=args.seed)