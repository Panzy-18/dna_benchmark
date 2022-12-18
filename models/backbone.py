from dataclasses import dataclass, asdict
from .attention import QuadraticSelfAttentionWithFFNLayer, QuadraticSelfAttention
from .config import BaseConfig
from .input import InputEmbeddingLayer, PositionEmbeddingLayer
from .layer import FFN, Pooling, Conv1d, MaxPooling1d, parse_conv_layer
from .base import OneHotEmbeddingForDNA, RMSNorm, Linear
import torch
import bmtrain as bmt
import math
import pdb

def get_low_backbone(config: BaseConfig):
    if config.model_name == 'attention':
        return AttentionLowLevelBackbone(**config.to_dict())
    elif config.model_name == 'convolution':
        return CNNLowLevelBackbone(**config.to_dict())

@dataclass
class BackboneOutput:
    rep_token: torch.Tensor = None # [bsz, seq_len, dim]
    rep_seq: torch.Tensor = None # [bsz, dim]
    logits_or_values: torch.Tensor = None
    attention_scores: torch.Tensor = None
    pooling_score: torch.Tensor = None 
    heat_map: torch.Tensor = None

class AttentionEncoder(torch.nn.Module):
    
    def __init__(self,
                 num_layers: int,
                 model_dim: int,
                 dim_head: int,
                 num_heads: int,
                 dim_ff: int,
                 dtype: torch.dtype,
                 linear_init_method: dict,
                 dropout: float,
                 bias: bool,
                 activation: str,
                 **kwargs
    ) -> None:
        super().__init__()
        self.layers = bmt.TransformerBlockList([
            bmt.CheckpointBlock(
                QuadraticSelfAttentionWithFFNLayer(
                    model_dim = model_dim,
                    dim_ff = dim_ff,
                    dim_head = dim_head,
                    num_heads = num_heads,
                    dropout = dropout,
                    linear_init_method = linear_init_method,
                    bias = bias,
                    dtype = dtype,
                    activation = activation,
                )
            )
            for _ in range(num_layers)
        ])
    
    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor = None,
                relative_position_embeddings: torch.Tensor = None, 
                rotary_position_embeddings: torch.nn.Module = None,
    ) -> torch.Tensor:
        last_hidden_states = self.layers(hidden_states, attention_mask, relative_position_embeddings, rotary_position_embeddings)
        attention_scores = []
        for name, module in self.layers.named_modules():
            if isinstance(module, QuadraticSelfAttention):
                # score: batch_size, self.num_heads, seq_len, seq_len
                attention_scores.append(module.attention_score)
        attention_scores = torch.stack(attention_scores, dim=0)
        
        # score: num_layer, batch_size, num_heads, seq_len, seq_len
        return last_hidden_states, attention_scores
    
class AttentionLowLevelBackbone(torch.nn.Module):
    
    def __init__(self,
                 vocab_size: int,
                 model_dim: int,
                 tokenization: str,
                 max_attention_seq_len: int,
                 dtype: torch.dtype,
                 embedding_init_method: dict,
                 conv_layers: list,
                 dropout: float,
                 dim_head: int,
                 abs_pos: bool,
                 rel_pos: bool,
                 rot_pos: bool,
                 num_layers: int,
                 num_heads: int,
                 dim_ff: int,
                 linear_init_method: dict,
                 bias: bool,
                 pooling_method: str,
                 activation: str,
                 input_embedding_dim: int = None,
                 **kwargs
    ):
        super().__init__()
        self.pooling_method = pooling_method
        self.input_embedding = InputEmbeddingLayer(
            vocab_size = vocab_size,
            model_dim = model_dim,
            tokenization = tokenization,
            max_attention_seq_len = max_attention_seq_len,
            dtype = dtype,
            embedding_init_method = embedding_init_method,
            conv_layers = conv_layers,
            pooling_method = pooling_method
        )
        self.position_embedding = PositionEmbeddingLayer(
            max_attention_seq_len = max_attention_seq_len,
            model_dim = model_dim,
            dtype = dtype,
            dropout = dropout,
            dim_head = dim_head,
            abs_pos = abs_pos,
            rel_pos = rel_pos,
            rot_pos = rot_pos,
            embedding_init_method = embedding_init_method,
        )
        self.proj_feature = Linear(
            dim_in = input_embedding_dim,
            dim_out = model_dim,
            dtype = dtype,
            init_method = linear_init_method,
            bias = bias
        ) if input_embedding_dim is not None else torch.nn.Identity()
        self.encoder = AttentionEncoder(
            num_layers = num_layers,
            model_dim = model_dim,
            dim_head = dim_head,
            num_heads = num_heads,
            dim_ff = dim_ff,
            dtype = dtype,
            linear_init_method = linear_init_method,
            dropout = dropout,
            bias = bias,
            activation = activation,
        )
        self.pooling = Pooling(
            pooling_method = pooling_method,
            model_dim = model_dim,
            dim_head = dim_head,
            dim_ff = dim_ff,
            linear_init_method = linear_init_method,
            embedding_init_method = embedding_init_method,
            bias = bias,
            dtype = dtype
        )
    
    def forward(self,
                input_ids: torch.Tensor = None,
                input_embeddings: torch.Tensor = None,
    ):
        attention_mask = None
        
        if input_embeddings is None:
            hidden_states = self.input_embedding(input_ids)
        else:
            hidden_states = self.proj_feature(input_embeddings)
        hidden_states = self.position_embedding(hidden_states)

        last_hidden_states, attention_scores = self.encoder(
            hidden_states, 
            attention_mask, 
            self.position_embedding.relative_position_embeddings, 
            self.position_embedding.rotary_position_embeddings 
        )
        rep_seq, rep_token, pooling_score = self.pooling(
            last_hidden_states,
            attention_scores,
        )

        return BackboneOutput(
            rep_token = rep_token,
            rep_seq = rep_seq,
            attention_scores = attention_scores,
            pooling_score = pooling_score
        )

class CNNLowLevelBackbone(torch.nn.Module):
    '''
    baseline for all task. Assuming that input_bp_length == max_input_bp_seq_len,
    sequence that do not fit this length will be paded or truncated.
    '''
    def __init__(self,
                 model_dim: int,
                 flatten_dim: int,
                 linear_init_method: dict,
                 bias: bool,
                 max_input_bp_seq_len: int,
                 dtype: torch.dtype,
                 conv_layers: list,
                 activation: str,
                 dropout: float,
                 **kwargs
    ) -> None:
        super().__init__()
        self.max_input_bp_seq_len = max_input_bp_seq_len
        self.input_embedding = OneHotEmbeddingForDNA(dtype=dtype)
        self.model_dim = model_dim
        self.dtype = dtype
        self.activation = activation
        self.conv_layers = torch.nn.ModuleList()
        self.tot_stride = 1
        self.dropout = torch.nn.Dropout(p=dropout)
        self.conv_layers = torch.nn.ModuleList([parse_conv_layer(layer_desc) for layer_desc in conv_layers])
        self.pooling = Linear(
            dim_in=flatten_dim,
            dim_out=model_dim,
            dtype=dtype,
            init_method=linear_init_method,
            bias=bias
        )

    def forward(self,
                input_ids: torch.Tensor = None,
    ):
        # if pad
        batch_size, seq_len = input_ids.shape[:2]
        if seq_len > self.max_input_bp_seq_len:
            input_ids = input_ids[:, :self.max_input_bp_seq_len]
        if seq_len < self.max_input_bp_seq_len:
            pad_len = self.max_input_bp_seq_len - seq_len
            pad_left = int(pad_len / 2)
            pad_right = math.ceil(pad_len / 2)
            input_ids = torch.cat([
                torch.zeros(batch_size, pad_left, dtype=input_ids.dtype, device=input_ids.device),
                input_ids,
                torch.zeros(batch_size, pad_right, dtype=input_ids.dtype, device=input_ids.device),
            ], dim=1)
        
        hidden_states = self.input_embedding(input_ids.long())
        for _layer in self.conv_layers:
            hidden_states = _layer(hidden_states)
        rep_seq = self.pooling(hidden_states.reshape(batch_size, -1))
        
        return BackboneOutput(
            rep_seq = rep_seq,
        )
