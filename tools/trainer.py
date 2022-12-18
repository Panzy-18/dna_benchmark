import sys
sys.path.append('..')
import os
from models import BaseModel
from torch.utils.data import Dataset
from datasets import DistributedDataLoader
from tqdm import tqdm
import torch
import bmtrain as bmt
import time
from torch.optim import AdamW, RMSprop
import json
import numpy as np
import pdb
from enum import Enum
from torch.utils.tensorboard import SummaryWriter
from .logging import get_logger
from .global_var import GLOBAL_VAR

logger = get_logger(__name__)

class EvalFlag(Enum):
    BEST_MODEL = 'best_model'
    EARYL_STOP = 'early_stop'
    KEEP_TRAIN = 'keep_train'
    NO_RESULT = 'no_result'

class TrainerWriter:
    
    def __init__(self) -> None:
        self.logger = get_logger('Trainer')
        if bmt.rank() == 0:
            self.writer = SummaryWriter(log_dir=GLOBAL_VAR._args.save_dir)
        else:
            self.writer = None
        self.score = {}
        self.early_stop = GLOBAL_VAR._args.early_stop
        GLOBAL_VAR._writer = self
    
    def add_scalar(self, *args, **kwargs):
        if self.writer:
            self.writer.add_scalar(*args, **kwargs)
    
    def info(self, msg):
        self.logger.info(msg)
    
    def warning(self, msg):
        self.logger.warning(msg)
    
    def error(self, msg):
        self.logger.error(msg)
    
    def update_evaluation(self, 
                          result: dict=None, 
                          split: str='valid',
                          epoch: int=None,
    ):
        if result is None:
            self.warning('evaluation result is null, do nothing.')
            return EvalFlag.NO_RESULT
        
        if split not in self.score:
            self.score[split] = {}
        main_score = result['main']
        self.score[split][epoch] = main_score
        
        self.info(f'{split} epoch[{epoch}]: {result}.')
        for k, v in result.items():
            if isinstance(v, float):
                self.add_scalar(f'valid/{k}', v, epoch)
        
        # find max
        max_epoch, max_score = max(self.score[split].items(), key=lambda x:x[1])
        if max_epoch == epoch:
            self.info(f'find best model at epoch[{epoch}].')
            return EvalFlag.BEST_MODEL
        if epoch - max_epoch > self.early_stop:
            self.info(f'trigger early stop at epoch[{epoch}].')
            return EvalFlag.EARYL_STOP
        return EvalFlag.KEEP_TRAIN
    
    def save_evaluation(self,
                        result: dict,
                        preds: np.ndarray,
                        targets: np.ndarray,
                        ids: np.ndarray = None,
                        **kwargs
    ):
        args = GLOBAL_VAR._args
        self.info(f'final test: {result}.')
        if bmt.rank() == 0:
            with open(os.path.join(args.save_dir, 'test.json'), 'w') as f:
                json.dump(result, f)
        if args.save_pred and bmt.rank() == 0:
            np.save(os.path.join(args.save_dir, 'preds.npy'), preds)
            np.save(os.path.join(args.save_dir, 'targets.npy'), targets)
            np.save(os.path.join(args.save_dir, 'idss.npy'), ids)
            

class Trainer:
    
    def __init__(self) -> None:
        self.writer = TrainerWriter()
        self.iter = 0
    
    @staticmethod
    def init_optimizer_and_lr_scheduler(model: BaseModel):
        args = GLOBAL_VAR._args
        no_decay = ['bias', 'norm']
        optimizer_grouped_parameters = [
            {
                'params': [
                    p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                'weight_decay': args.weight_decay
            },
            {
                'params': [
                    p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                'weight_decay': 0.0
            },
        ]
        if model.config.model_name == 'attention':
            if next(model.parameters()).dtype == torch.half:
                raise ValueError('fp16 may cause unknown error, abandon.')
                optimizer = bmt.optim.AdamOffloadOptimizer(optimizer_grouped_parameters, 
                                                           lr=args.lr,
                                                           scale=args.loss_scale)
            else:
                optimizer = AdamW(optimizer_grouped_parameters,
                                  lr=args.lr)
        else:
            optimizer = RMSprop(model.parameters(),
                                lr=args.lr)
        ### LR SCHEDULAR
        if args.lr_decay_iters is None:
            args.lr_decay_iters = args.train_iters
        if args.lr_decay_style == "noam":
            lr_scheduler = bmt.lr_scheduler.Noam(optimizer, 
                                                start_lr = args.lr,
                                                warmup_iter = args.warmup_iters, 
                                                end_iter = args.lr_decay_iters,
                                                num_iter = 0)
        elif args.lr_decay_style == "constant":
            lr_scheduler = bmt.lr_scheduler.NoDecay(optimizer, 
                                                start_lr = args.lr,
                                                warmup_iter = args.warmup_iters, 
                                                end_iter = -1,
                                                num_iter = 0)
        elif args.lr_decay_style == "linear":
            lr_scheduler = bmt.lr_scheduler.Linear(optimizer, 
                                                start_lr = args.lr,
                                                warmup_iter = args.warmup_iters, 
                                                end_iter = args.lr_decay_iters,
                                                num_iter = 0)
        elif args.lr_decay_style == "exponential":
            lr_scheduler = bmt.lr_scheduler.Exponential(optimizer, 
                                                start_lr = args.lr,
                                                warmup_iter = args.warmup_iters, 
                                                end_iter = args.lr_decay_iters,
                                                num_iter = 0)
        elif args.lr_decay_style == "cosine":
            lr_scheduler = bmt.lr_scheduler.Cosine(optimizer, 
                                                start_lr = args.lr,
                                                warmup_iter = args.warmup_iters, 
                                                end_iter = args.lr_decay_iters,
                                                num_iter = 0)

        else:
            raise ValueError(f"lr_scheduler of type {args.lr_decay_style} is not supported yet.")

        return optimizer, lr_scheduler
    
    def fit(self,
            model: BaseModel,
            dataset: dict[str, Dataset],
            save_name: str = 'model_epoch_{}.pt'
    ):
        args = GLOBAL_VAR._args
        optimizer, lr_scheduler = self.init_optimizer_and_lr_scheduler(model)
        
        do_train = not args.eval_only
        logger.info(f'do train: {do_train}')
        do_valid = dataset.get('valid') is not None
        logger.info(f'do valid: {do_valid}')
        do_test = dataset.get('test') is not None
        logger.info(f'do test: {do_test}')
        best_path = os.path.join(args.save_dir, 'model_best.pt')
        epoch_path = os.path.join(args.save_dir, save_name)
        
        if do_train:
            for epoch in range(args.epochs):
                if do_valid:
                    evaluation = self.evaluate(model, dataset['valid'])
                    flag = self.writer.update_evaluation(evaluation['result'], 'valid', epoch)
                    if flag == EvalFlag.BEST_MODEL:
                        model.save(best_path)
                    elif flag == EvalFlag.EARYL_STOP:
                        break
                bmt.synchronize()
                self.train(model, dataset['train'], optimizer, lr_scheduler, epoch)
                model.save(epoch_path.format(epoch))
                bmt.synchronize()
        
        if do_valid:
            evaluation = self.evaluate(model, dataset['valid'])
            flag = self.writer.update_evaluation(evaluation['result'], 'valid', args.epochs)
            if flag == EvalFlag.BEST_MODEL:
                model.save(best_path)
        
        if do_test:
            model.load(best_path)
            evaluation = self.evaluate(model, dataset['test'])  
            self.writer.save_evaluation(**evaluation)    
            
        
    def train(self,
              model: BaseModel,
              dataset: Dataset,
              optimizer: torch.optim.Optimizer,
              lr_scheduler: bmt.lr_scheduler.WarmupLRScheduler,
              epoch: int,
    ):
        start = time.time()
        def bp(model, optimizer, lr_scheduler, clip_grad):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad, norm_type=2)
            bmt.optim_step(optimizer, lr_scheduler)
            optimizer.zero_grad()
            
        args = GLOBAL_VAR._args
        model.train()
        dataloader = DistributedDataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            collate_fn=dataset.collate_fn,
            shuffle=dataset.dataset_name not in ['track7878'],
            drop_last=True,
            pin_memory=True,
            num_workers=0,
            # prefetch_factor=4,
        )
        dataloader = tqdm(dataloader, desc = f'Training Epoch[{epoch}]', disable = bmt.rank() != 0)
        optimizer.zero_grad()
        avg_loss = 0.
        
        for iter, batch in enumerate(dataloader):
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            model_output = model.train_forward(**batch)
            loss = model_output.loss
            loss /= args.gradient_accumulation
            # 先在这里对全部的节点处理一下inf和nan的事情
            iter_loss = bmt.distributed.all_reduce(loss, 'avg')
            if torch.isnan(iter_loss).sum() or torch.isinf(iter_loss).sum():
                self.writer.error(
                    'Finding loss nan or inf, ignore this batch.'
                )
                continue
            loss.backward()
            avg_loss += iter_loss.item()
            
            # BP
            if iter % args.gradient_accumulation == 0:
                bp(model, optimizer, lr_scheduler, args.clip_grad)
                dataloader.set_postfix(
                    loss='{:4f}'.format(avg_loss), 
                    lr=lr_scheduler.current_lr, 
                    iter=self.iter
                )
                self.writer.add_scalar('loss', avg_loss, self.iter)
                avg_loss = 0.
                self.iter += 1
        optimizer.zero_grad()
        end = time.time()
        logger.info(f'Epoch [{epoch}] finished. Time consuming: {end-start}s')
    
    def evaluate(self,
                 model: BaseModel,
                 dataset: Dataset,
    ):
        args = GLOBAL_VAR._args
        model.eval()
        dataloader = DistributedDataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            collate_fn=dataset.collate_fn,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=0,
            # prefetch_factor=4,
        )
        dataloader = tqdm(dataloader, desc = f'Evaluating', disable = bmt.rank() != 0)
        preds = []
        targets = []
        with torch.no_grad():
            for batch in dataloader:
                # pdb.set_trace()
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                labels = batch['labels']
                output = model.eval_forward(**batch)
                values = output.logits_or_values
                # gather
                labels = bmt.distributed.all_gather(labels)
                values = bmt.distributed.all_gather(values)
                labels = torch.flatten(labels, start_dim=0, end_dim=1) # 把 world_size / batch_size 展平
                values = torch.flatten(values, start_dim=0, end_dim=1)
                preds.append(values.detach().cpu().to(torch.half))
                targets.append(labels.detach().cpu().to(torch.half))
        
        preds = torch.cat(preds, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()
        result = dataset.metric_fn(
            targets=targets,
            preds=preds,
        ) if dataset.metric_fn is not None else None
        bmt.synchronize()
        
        return dict(
            preds=preds,
            targets=targets,
            result=result
        )
        
    def predict(self,
                model: BaseModel,
                dataset: Dataset,
    ):
        if bmt.rank() != 0:
            raise ValueError('Not support for DDP')
        
        args = GLOBAL_VAR._args
        model.eval()
        dataloader = DistributedDataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            collate_fn=dataset.collate_fn,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=0,
        )
        dataloader = tqdm(dataloader, desc = f'Predicting', disable = bmt.rank() != 0)
        all_datas = None
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i == 0:
                    all_datas = {key: [] for key in batch.keys()}
                    all_datas['preds'] = []
                # forward
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                output = model.eval_forward(**batch)
                # store
                for k, v in batch.items():
                    all_datas[k].append(v.cpu().to(torch.half) if isinstance(v, torch.Tensor) else v)
                
                for k, v in output.items():
                    if all_datas.get(k) is None:
                        all_datas[k] = []
                    all_datas[k].append(v.cpu().to(torch.half) if isinstance(v, torch.Tensor) else v)
        
        return all_datas