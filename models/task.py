import numpy as np
from .base import *
from .tokenizers import BaseTokenizer
from .layer import Head
from .config import BaseConfig
from .backbone import get_low_backbone
from .loss import CSELoss, get_criterion
from .visualize import VisualHead
import torch
from easydict import EasyDict as edict
import pdb

__all__ = [
    'ModelForPretrain',
    'ModelForSequenceTask'
]

class ModelForPretrain(BaseModel):
    
    def __init__(self, 
                 config: BaseConfig, 
                 tokenizer: BaseTokenizer = None,
                 **kwargs,
    ):
        super().__init__(config, tokenizer)
        self.low_backbone = get_low_backbone(config)
        if config.tokenization != 'onehot':
            self.lm_head = Head(
                dim_in = config.model_dim,
                dim_ff = config.dim_ff,
                linear_init_method = config.linear_init_method,
                dim_out = config.vocab_size,
                bias = config.bias,
                dtype = config.dtype,
                activation = config.activation
            )
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.cse_loss = CSELoss() if config.train_cse else lambda x: 0.
    
    def forward(self,
                input_ids: torch.Tensor,
    ):
        return self.low_backbone(input_ids=input_ids)

    def train_forward(self,
                      input_ids: torch.Tensor,
                      labels: torch.Tensor,
                      *args, **kwargs,
    ):
        backbone_output = self.forward(input_ids)
        mlm_loss = 0.
        if self.config.tokenization != 'onehot':
            logits = self.lm_head(backbone_output.rep_token)
            mlm_loss += self.ce_loss(logits.view(-1, logits.shape[-1]), labels.view(-1))
        cse_loss = self.cse_loss(rep_seq=backbone_output.rep_seq)
        loss = mlm_loss + cse_loss
        
        return edict(
            loss = loss
        )
    
    def eval_forward(self, *args, **kwargs):
        raise NotImplementedError


class ModelForSequenceTask(BaseModel):
    
    def __init__(self, 
                 config: BaseConfig, 
                 tokenizer: BaseTokenizer = None,
                 **kwargs
    ):
        super().__init__(config, tokenizer)
        self.low_backbone = get_low_backbone(config)
        self.head = Head(
            dim_out = config.final_dim,
            dim_in = config.model_dim,
            dim_ff = config.dim_ff,
            linear_init_method = config.linear_init_method,
            bias = config.bias,
            dtype = config.dtype,
            activation = config.activation
        ) if config.with_head else torch.nn.Identity()
        self.visual_head = VisualHead(
            model_dim = config.model_dim,
            num_classes = config.final_dim,
            dtype = config.dtype,
            for_each_class = config.tscam
        ) 
        self.criterion = get_criterion(config.loss_fn)
    
    def forward(self,
                input_ids: torch.Tensor = None,
                input_embeddings: torch.Tensor = None,
    ):  
        backbone_output = self.low_backbone(
            input_ids=input_ids,
            input_embeddings=input_embeddings,
        )
        logits_or_values = self.head(backbone_output.rep_seq)
        logits_or_values, heat_map = self.visual_head(
            backbone_output.rep_token,
            logits_or_values,
            backbone_output.pooling_score
        )
        backbone_output.logits_or_values = logits_or_values
        backbone_output.heat_map = heat_map
        return backbone_output
    
    def train_forward(self,
                      labels: torch.Tensor,
                      input_ids: torch.Tensor = None,
                      input_embeddings: torch.Tensor = None,
                      **kwargs,
    ):
        backbone_output = self.forward(
            input_ids=input_ids,
            input_embeddings=input_embeddings,
        )
        loss = self.criterion(
            backbone_output.logits_or_values.to(torch.float), 
            labels.reshape_as(backbone_output.logits_or_values).to(torch.float)
        )
        backbone_output.loss = loss
        return backbone_output
    
    def eval_forward(self,
                     input_ids: torch.Tensor = None,
                     input_embeddings: torch.Tensor = None,
                     *args, **kwargs
    ):
        with torch.no_grad():
            return self.forward(
                    input_ids=input_ids,
                    input_embeddings=input_embeddings,
                )

