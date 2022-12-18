import sys
sys.path.append("..")
from tools import get_logger
import torch
from typing import Union
import bmtrain as bmt
import torch.nn.functional as F
from .tokenizers import BaseTokenizer, get_tokenizer
import pdb

logger = get_logger(__name__)

__all__ = [
    'Linear', 
    'GatedActivation', 
    'RMSNorm', 
    'Embedding', 
    'OneHotEmbeddingForDNA',
    'BaseModel',
    'ACT2FN'
]

ACT2FN = {
    "none": torch.nn.Identity(),
    "gelu": F.gelu,
    "relu": F.relu,
    "swish": F.silu,
}

class Linear(bmt.DistributedModule):
    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 dtype: torch.dtype,
                 init_method: dict,
                 bias: bool,
    ):
        super().__init__()
        self.weight = bmt.DistributedParameter(
            torch.empty((dim_out, dim_in), dtype=dtype),
            init_method=bmt.ParameterInitializer(**init_method)
        )
        if bias:
            self.bias = bmt.DistributedParameter(
                torch.empty((dim_out,), dtype=dtype),
                init_method=bmt.ParameterInitializer(func = torch.nn.init.constant_, val = 0)
            )
        else:
            self.bias = None
    
    def forward(self, x):
        x = F.linear(x, self.weight)
        if self.bias is not None:
            x += self.bias
        return x
    
class GatedActivation(bmt.DistributedModule):
    
    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 activation: str,
                 dtype: torch.dtype,
                 init_method: dict,
                 bias: bool,
    ):
        super().__init__()
        self.v_proj = Linear(
            dim_in = dim_in,
            dim_out = dim_out,
            dtype = dtype,
            init_method = init_method,
            bias = bias,
        )
        self.u_proj = Linear(
            dim_in = dim_in,
            dim_out = dim_out,
            dtype = dtype,
            init_method = init_method,
            bias = bias,
        )
        self.activation = ACT2FN[activation]
    
    def forward(self, x):
        weight = self.activation(self.u_proj(x))
        x = torch.mul(self.v_proj(x), weight)
        return x
    
class RMSNorm(bmt.DistributedModule):

    def __init__(self,
                 dim: int,
                 dtype: torch.dtype,
    ):
        super().__init__()
        self.eps = 1e-8
        self.weight = bmt.DistributedParameter(torch.ones(dim, dtype=dtype))

    def forward(self, x):
        dtype = x.dtype
        var = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        x = (x * torch.rsqrt(var + self.eps)).to(dtype)
        return torch.mul(x, self.weight)

class Embedding(bmt.DistributedModule):

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 init_method: dict,
                 dtype: torch.dtype,
    ):
        super().__init__()
        self.weight = bmt.DistributedParameter(
            torch.empty(num_embeddings, embedding_dim, dtype=dtype),
            init_method=bmt.ParameterInitializer(**init_method)
        )
        self.embedding_dim = embedding_dim
    
    def forward(self, ids):
        return F.embedding(ids, self.weight)
    
    def project(self, x):
        return F.linear(x, self.weight)
    
class OneHotEmbeddingForDNA(bmt.DistributedModule):
    
    def __init__(self,
                 dtype: torch.dtype,
    ) -> None:
        super().__init__()
        one_hot_matrix = torch.eye(4, dtype=dtype) # [4, 4]
        one_hot_matrix_with_pad = torch.cat([torch.zeros((1, 4), dtype=dtype), one_hot_matrix], dim=0)
        self.weight = bmt.DistributedParameter(
            data = one_hot_matrix_with_pad,
            requires_grad = False,
        )
        self.embedding_dim = 4
    
    def forward(self, ids):
        return F.embedding(ids, self.weight)


class BaseModel(torch.nn.Module):
    
    def __init__(self,
                 config, 
                 tokenizer: BaseTokenizer = None,
    ):
        super().__init__()
        self.config = config
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = get_tokenizer(method=config.tokenization, max_input_bp_seq_len=config.max_input_bp_seq_len)
        self.config.vocab_size = len(self.tokenizer)
        
    
    def load(self, file):        
        try:
            bmt.load(self, file, strict=False)
        except:
            from bmtrain.store import DistributedStateDictWrapper
            if bmt.rank() == 0:
                state_dict = torch.load(file)
                _about_to_move = []
                for k in state_dict.keys():
                    if any(x in k for x in ['head', 'class']):
                        _about_to_move.append(k)
                for k in _about_to_move:
                    state_dict.pop(k)
                state_dict = DistributedStateDictWrapper(state_dict)
            else:
                state_dict = DistributedStateDictWrapper({})
            self.load_state_dict(state_dict, strict=False)
            torch.cuda.synchronize()
        
        logger.info(f'load model from {file}')
        return self

    def save(self, file):
        logger.info(f'save model to {file}')
        bmt.save(self, file)

    def param_num(self,
                  ignore_key: list = ['head', 'class']
    ):
        total = 0
        for name, param in self.named_parameters():
            if not any(key in name for key in ignore_key):
                total += param.numel()
        return total
    
    def train_forward(self,
                      input_ids: torch.Tensor = None,
                      input_embeddings: torch.Tensor = None,
                      labels: torch.Tensor = None,
                      **kwargs,
    ):
        ...
    
    def eval_forward(self,
                     input_ids: torch.Tensor = None,
                     input_embeddings: torch.Tensor = None,
                     *args, **kwargs
    ):
        ...
    
    def predict(self,
                sequences: Union[str, list[str]] = None,
                features: torch.Tensor = None,
                **kwargs
    ):
        if self.training:
            self.eval()
            
        if sequences is not None:
            if isinstance(sequences, str):
                sequences = [sequences]
            input_ids = self.tokenizer(sequences, return_type='pt').cuda()
            return self.eval_forward(input_ids=input_ids)
        if features is not None:
            features = features.cuda()
            return self.eval_forward(input_embeddings=features)
    
    