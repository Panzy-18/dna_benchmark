from .base import *
import bmtrain as bmt
import torch.nn.functional as F
import torch
from typing import Union
from easydict import EasyDict
import math
import pdb

def parse_conv_layer(layer_desc: dict, dtype, dropout=0.):
        '''
        根据参数添加卷积层
        default format
        OrderedDict: [
            ("Conv1d", 512, 512, 26, "valid", 1, "gelu")
            ("MaxPooling1d", 13, 13)
        ]
        '''
        if layer_desc[0].lower() == 'conv1d':
            return Conv1d(
                dim_in=layer_desc[1],
                dim_out=layer_desc[2],
                kernel_size=layer_desc[3],
                padding=layer_desc[4],
                stride=layer_desc[5],
                activation=layer_desc[6],
                dtype=dtype,
                dropout=dropout,
            )
        elif layer_desc[0].lower() == 'maxpooling1d':
            return MaxPooling1d(
                kernel_size=layer_desc[1],
                stride=layer_desc[2],
            )
        else:
            raise ValueError(f'Unknown conv layer name: {layer_desc}')

class FFN(bmt.DistributedModule):
    
    def __init__(self,
                 dim_in: int,
                 dim_ff: int,
                 dtype:  torch.dtype,
                 linear_init_method: dict,
                 bias: bool,
                 activation: str,
                 dim_out: int = None,
    ) -> None:
        super().__init__()
        if dim_out is None:
            dim_out = dim_in
        self.proj_ff = Linear(
            dim_in = dim_in,
            dim_out = dim_ff,
            dtype = dtype,
            init_method = linear_init_method,
            bias = bias,
        )
        self.proj_out = Linear(
            dim_in = dim_ff,
            dim_out = dim_out,
            dtype = dtype,
            init_method = linear_init_method,
            bias = bias,
        )
        if activation.startswith('gated_'):
            act_fn = activation.split('_')[-1]
            self.activation = GatedActivation(
                dim_in = dim_ff,
                dim_out = dim_ff,
                activation = act_fn,
                dtype = dtype,
                init_method = linear_init_method,
                bias = bias,
            )
        else:
            self.activation = ACT2FN[activation]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_ff(x)
        x = self.activation(x)
        return self.proj_out(x)
    
class Conv1d(bmt.DistributedModule):

    def __init__(self,
                 dim_in : int,
                 dim_out : int,
                 kernel_size: int,
                 padding: str,
                 stride: int,
                 dtype: torch.dtype,
                 activation: str = "none",
                 dropout: float = 0.
    ):
        super().__init__()
        init_u = (1 / dim_in * kernel_size) ** 0.5
        init_u = 1 if init_u > 1 else init_u
        self.padding = padding
        self.stride = stride
        if activation.startswith('gated_'):
            raise ValueError('Not support gated activation in conv layer.')
        else:
            self.activation = ACT2FN[activation]
        self.filters = bmt.DistributedParameter(
            torch.empty((dim_out, dim_in, kernel_size), dtype=dtype),
            init_method=bmt.ParameterInitializer(torch.nn.init.uniform_, a=-init_u, b=init_u)
        )
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        '''
        F.conv1d:
        >>> inputs = torch.randn(33, 16, 30)
        >>> filters = torch.randn(20, 16, 5)
        >>> F.conv1d(inputs, filters)
        the conv-dim should be at dim==-2
        '''
        x = x.transpose(dim, -1)
        x = F.conv1d(x, self.filters, padding=self.padding, stride=self.stride)
        x = x.transpose(dim, -1)
        x = self.activation(x)
        x = self.dropout(x)
        return x
    
class MaxPooling1d(bmt.DistributedModule):
    
    def __init__(self,  
                 kernel_size: int,
                 stride: int,
                 padding: int = None,
                 **kwargs
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        if padding is None:
            self.padding = kernel_size // 2
        else:
            self.padding = padding
    
    def forward(self, x, dim=-2):
        x = x.transpose(dim, -1)
        x = F.max_pool1d(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        x = x.transpose(dim, -1)
        return x

class Pooling(bmt.DistributedModule):
    '''
    cls_embedding: [1, model_dim]
    '''
    
    def __init__(self,
                 pooling_method: str,
                 model_dim: int,
                 dim_head: int,
                 dim_ff: int,
                 linear_init_method: dict,
                 embedding_init_method: dict,
                 bias: bool,
                 dtype: torch.dtype,
                 dim_out: int = None,
                 **kwargs,
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.pooling_method = pooling_method
        if dim_out is None:
            dim_out = model_dim
        if pooling_method == "attention":
            self.cls_embeddings = bmt.DistributedParameter(
                torch.empty(1 , model_dim, dtype = dtype),
                init_method=bmt.ParameterInitializer(**embedding_init_method)
            )
            self.proj_kv = Linear(
                dim_in = model_dim,
                dim_out = dim_ff + dim_head,
                dtype = dtype,
                init_method = linear_init_method,
                bias = bias,
            )
            self.proj_uq = Linear(
                dim_in = model_dim,
                dim_out = dim_head + dim_ff,
                dtype = dtype,
                init_method = linear_init_method,
                bias = bias,
            )
            self.project_out = Linear(
                dim_in = dim_ff,
                dim_out = dim_out,
                dtype = dtype,
                init_method = linear_init_method,
                bias = bias,
            )
        assert pooling_method in ['cls', 'mean', 'attention']
        
    def forward(self,
                hidden_states: torch.Tensor,
                attention_scores: torch.Tensor, # n_layer x n_batch x n_head x n_token x n_token
    ):
        score = None
        
        att_weights = torch.mean(attention_scores, dim=2) # n_layer x n_batch x n_token x n_token
        residual_att = torch.eye(att_weights.size(2)).unsqueeze(0).unsqueeze(1).to(att_weights.get_device())
        aug_att_mat = att_weights + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1) 
        joint_atts = torch.zeros(att_weights.size()).to(att_weights.get_device())
        joint_atts[0] = aug_att_mat[0]
        for n in range(1, aug_att_mat.size()[0]):
            joint_atts[n] = torch.matmul(aug_att_mat[n], joint_atts[n-1])
        total_attention_score = joint_atts[-1] # n_batch x n_token x n_token
        
        if self.pooling_method == "cls":
            rep_seq = hidden_states[:, 0, ...]
            rep_token = hidden_states[:, 1:, ...]
            pooling_score = total_attention_score[:, 0, 1:]
        if self.pooling_method == "mean":
            rep_seq = hidden_states.mean(dim=-2)
            rep_token = hidden_states
            pooling_score = total_attention_score.mean(dim=1)
        if self.pooling_method == "attention":
            k, v = torch.split(F.gelu(self.proj_kv(hidden_states)), [self.dim_head, self.dim_ff], dim = -1)
            u, q = torch.split(F.gelu(self.proj_uq(self.cls_embeddings)), [self.dim_ff, self.dim_head], dim = -1)
            score = torch.einsum('nr,...mr->...nm', q, k) * (self.model_dim ** -0.5)
            score = F.softmax(score, dim=-1) # n_batch x 1 x n_token
            rep_seq = torch.mul(u, torch.matmul(score, v))
            rep_seq = self.project_out(rep_seq)
            rep_token = hidden_states
            pooling_score = torch.matul(score, total_attention_score).sequeeze(1)

        return rep_seq, rep_token, pooling_score

class Head(bmt.DistributedModule):
    
    def __init__(self,
                 dim_out: int,
                 dim_in: int,
                 dim_ff: int,
                 linear_init_method: dict,
                 bias: bool,
                 dtype: torch.dtype,
                 activation: str,
                 **kwargs,
    ) -> None:
        super().__init__()
        self.dense = Linear(
            dim_in = dim_in,
            dim_out = dim_ff,
            init_method = linear_init_method,
            bias = bias,
            dtype = dtype
        )
        if activation.startswith('gated_'):
            act_fn = activation.split('_')[-1]
            self.activation = GatedActivation(
                dim_in = dim_ff,
                dim_out = dim_ff,
                activation = act_fn,
                dtype = dtype,
                init_method = linear_init_method,
                bias = bias,
            )
        else:
            self.activation = ACT2FN[activation]
        self.decoder = Linear(
            dim_in = dim_ff,
            dim_out = dim_out,
            init_method = linear_init_method,
            bias = bias,
            dtype = dtype
        )
    
    def forward(self,
                hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        return self.decoder(hidden_states)

