from .base import *
from .layer import parse_conv_layer
import bmtrain as bmt
import torch

class RoPE(bmt.DistributedModule):
    
    def __init__(self, 
                 dim: int,
                 max_attention_seq_len: int,
                 dtype: torch.dtype,
                 ) -> None:
        super().__init__()
        self.head_size = dim
        self.max_attention_seq_len = max_attention_seq_len

        position = torch.arange(0, max_attention_seq_len).to(torch.float32).unsqueeze(-1) # [max_seq_len, 1]
        half_d = dim // 2
        freq_seq = torch.arange(0, half_d).to(torch.float32) / half_d
        inv_freq = 10000 ** -freq_seq
        sinusoid = torch.matmul(position, inv_freq.unsqueeze(0))
        
        self._sin = bmt.DistributedParameter(
            data = torch.sin(sinusoid).repeat_interleave(2, dim=-1).to(dtype),
            requires_grad = False,
        )
        self._cos = bmt.DistributedParameter(
            data = torch.cos(sinusoid).repeat_interleave(2, dim=-1).to(dtype),
            requires_grad = False,
        )
        # self.register_buffer('_sin', torch.sin(sinusoid).repeat_interleave(2, dim=-1).to(device).to(dtype)) # [max_seq_len, head_size]
        # self.register_buffer('_cos', torch.cos(sinusoid).repeat_interleave(2, dim=-1).to(device).to(dtype)) # [cos(p0), cos(p0), cos(p1), cos(p1) ...]
        
    def forward(self, qk_states: torch.Tensor, dim=-2): # 只有一个head
        assert qk_states.shape[-1] == self.head_size
        if isinstance(dim, int):
            if dim < 0:
                dim += qk_states.dim()
            dim = [dim]

        h2 = torch.stack((qk_states[..., 1::2] * -1, qk_states[..., ::2])).reshape(qk_states.shape) # [-q1, q0, -q3, q2 ...]

        tot_seq_len = 1
        for i in dim:
            tot_seq_len *= qk_states.shape[i]
        chunk_shape = []
        for i, dim_size in enumerate(qk_states.shape):
            if i in dim or i == qk_states.dim() - 1:
                chunk_shape.append(dim_size)
            else:
                chunk_shape.append(1)

        _cos = self._cos[:tot_seq_len, :].reshape(chunk_shape)
        _sin = self._sin[:tot_seq_len, :].reshape(chunk_shape)
        return torch.mul(qk_states, _cos) + torch.mul(h2, _sin)
        
class LearnableScaledSinusoid(bmt.DistributedModule):

    def __init__(self,
                 dim: int,
                 max_attention_seq_len: int,
                 dtype: torch.dtype,
                 ) -> None:
        super().__init__()
        
        position = torch.arange(0, max_attention_seq_len).to(torch.float32).unsqueeze(-1) # [max_seq_len, 1]
        half_d = dim // 2
        freq_seq = torch.arange(0, half_d).to(torch.float32) / half_d
        inv_freq = 10000 ** -freq_seq
        sinusoid = torch.matmul(position, inv_freq.unsqueeze(0))
        _sinusoid = torch.cat([torch.sin(sinusoid), torch.cos(sinusoid)], dim=-1) * (1 / dim ** 0.5)

        self.sinusoid = bmt.DistributedParameter(_sinusoid).to(dtype)
    
    def forward(self, hidden_states: torch.Tensor, dim=-2):
        length = hidden_states.shape[dim]  
        return hidden_states + self.sinusoid[:length, :].unsqueeze(0)

class InputEmbeddingLayer(bmt.DistributedModule):
    
    
    def __init__(self,
                 vocab_size: int,
                 model_dim: int,
                 tokenization: str,
                 max_attention_seq_len: int,
                 dtype: torch.dtype,
                 embedding_init_method: dict,
                 conv_layers: list,
                 pooling_method: str,
                 **kwargs
    ) -> None:
        super().__init__()
        self.tokenization = tokenization
        self.model_dim = model_dim
        self.dtype = dtype
        self.max_attention_seq_len = max_attention_seq_len
        if tokenization == 'onehot':
            self.input_embedding = OneHotEmbeddingForDNA(
                dtype = dtype
            )
            if not conv_layers:
                self.input_embedding = Embedding(
                    num_embeddings = 5,
                    embedding_dim = model_dim,
                    init_method = embedding_init_method,
                    dtype = dtype
                )
        else:
            self.input_embedding = Embedding(
                num_embeddings = vocab_size,
                embedding_dim = model_dim,
                init_method = embedding_init_method,
                dtype = dtype
            )
        if pooling_method == 'cls':
            self.cls_embedding = Embedding(
                num_embeddings = 1,
                embedding_dim = model_dim,
                init_method = embedding_init_method,
                dtype = dtype
            )
        self.pooling_method = pooling_method
        self.num_conv_layers = len(conv_layers)
        for i, layer_desc in enumerate(conv_layers):
            setattr(self, f'conv_{i}', parse_conv_layer(layer_desc, dtype))
            
    def forward(self, 
                input_ids: torch.Tensor,
    ):
        assert len(input_ids.shape) == 2
        hidden_states = self.input_embedding(input_ids.long())
        for i in range(self.num_conv_layers):
            conv_layer = getattr(self, f'conv_{i}')
            hidden_states = conv_layer(hidden_states)
        
        if self.pooling_method == 'cls':
            cls_vector = self.cls_embedding(torch.zeros(input_ids.shape[0], 1).to(input_ids.device).long())
            hidden_states = torch.cat([cls_vector, hidden_states], dim=1)
        
        if hidden_states.shape[1] > self.max_attention_seq_len:
            # 剪裁至最允许值
            hidden_states = hidden_states[:, :self.max_attention_seq_len, ...]
        return hidden_states

class PositionEmbeddingLayer(bmt.DistributedModule):
    
    def __init__(self,
                 max_attention_seq_len: int,
                 model_dim: int,
                 dtype: torch.dtype,
                 dropout: float,
                 dim_head: int,
                 abs_pos: bool,
                 rel_pos: bool,
                 rot_pos: bool,
                 embedding_init_method: dict,
                 **kwargs
    ) -> None:
        super().__init__()
        self.abs_pos = abs_pos
        self.rel_pos = rel_pos
        self.rot_pos = rot_pos
        if abs_pos:
            self.absolute_position_embeddings = LearnableScaledSinusoid(
                dim = model_dim,
                max_attention_seq_len = max_attention_seq_len,
                dtype = dtype,
            )
        else:
            self.absolute_position_embeddings = None
        if rel_pos:
            self.relative_position_embeddings = bmt.DistributedParameter(
                torch.empty(max_attention_seq_len * 2, model_dim, dtype=dtype),
                init_method=bmt.ParameterInitializer(**embedding_init_method)
            )
        else:
            self.relative_position_embeddings = None
        if rot_pos:
            self.rotary_position_embeddings = RoPE(
                dim = dim_head,
                max_attention_seq_len = max_attention_seq_len,
                dtype = dtype,
            )
        else:
            self.rotary_position_embeddings = None
            
        self.norm = RMSNorm(
            dim = model_dim,
            dtype = dtype
        )
        
        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None
    
    def forward(self,
                hidden_states: torch.Tensor,
    ):
        if self.abs_pos:
            hidden_states = self.absolute_position_embeddings(hidden_states)
        if self.dropout:
            hidden_states = self.dropout(hidden_states)
        return self.norm(hidden_states)