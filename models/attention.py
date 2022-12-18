from functools import lru_cache
from .base import *
from .layer import FFN
import bmtrain as bmt
import torch.nn.functional as F
import torch

class QuadraticSelfAttention(bmt.DistributedModule):
    
    '''
    Quadratic Attention Implementation
    '''
    
    @classmethod
    @lru_cache(maxsize=128)
    def build_relative_relation_index(cls, 
                                      main_size=None, 
                                      query_size=None, 
                                      key_size=None, 
                                      device=None):
        if main_size is not None:
            query_size = main_size
            key_size = main_size
        q_ids = torch.arange(0, query_size, dtype=torch.long)
        k_ids = torch.arange(0, key_size, dtype=torch.long)
        relative_relation_matrix = q_ids.unsqueeze(-1) - torch.tile(k_ids, (query_size, 1))
        return relative_relation_matrix.to(device)

    def __init__(self,
                 dim_in: int,
                 dim_head: int,
                 num_heads: int,
                 dtype: torch.dtype,
                 linear_init_method: dict,
                 dropout: float,
                 bias: bool,
                 dim_out: int = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.dim_mediate = num_heads * dim_head
        if dim_out is None:
            dim_out = dim_in
        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None
        
        self.q_proj = Linear(
            dim_in = dim_in,
            dim_out = self.dim_mediate,
            dtype = dtype,
            init_method = linear_init_method,
            bias = bias,
        )
        self.k_proj = Linear(
            dim_in = dim_in,
            dim_out = self.dim_mediate,
            dtype = dtype,
            init_method = linear_init_method,
            bias = bias,
        )
        self.v_proj = Linear(
            dim_in = dim_in,
            dim_out = self.dim_mediate,
            dtype = dtype,
            init_method = linear_init_method,
            bias = bias,
        )
        self.out_proj = Linear(
            dim_in = self.dim_mediate,
            dim_out = dim_out,
            dtype = dtype,
            init_method = linear_init_method,
            bias = bias,
        )
        self.attention_score = None
    
    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor = None,
                relative_position_embeddings: torch.Tensor = None, # 全部[2 * max_seq_len, dim]
                rotary_position_embeddings: torch.nn.Module = None,
    ) -> torch.Tensor:
        batch_size, seq_len = hidden_states.shape[:2]
        q, k, v = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
        q = q.view(batch_size, seq_len, self.num_heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        k = k.view(batch_size, seq_len, self.num_heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        v = v.view(batch_size, seq_len, self.num_heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        
        score = 0
        score_scalar = 0
        
        if relative_position_embeddings is not None:
            max_seq_len = relative_position_embeddings.shape[0] // 2
            redundency = max_seq_len - seq_len
            # 进行剪裁, 只需要中间这些部分以增加运算速度.
            relative_position_embeddings = relative_position_embeddings[redundency: 2 * max_seq_len - redundency, ...]
            q_pos, k_pos = self.q_proj(relative_position_embeddings), self.k_proj(relative_position_embeddings)
            q_pos = q_pos.view(-1, self.num_heads, self.dim_head).transpose(0, 1).contiguous()
            k_pos = k_pos.view(-1, self.num_heads, self.dim_head).transpose(0, 1).contiguous()
            
            rel_pos = self.build_relative_relation_index(query_size=q.shape[-2], key_size=k.shape[-2], device=q.device)
            c2p_index = rel_pos + seq_len
            c2p_score = torch.einsum('...hmr,hnr->...hmn', q, k_pos)
            score += torch.gather(c2p_score, dim=-1, index=c2p_index.unsqueeze(0).unsqueeze(0))
            p2c_index = -rel_pos + seq_len
            p2c_score = torch.einsum('hnr,...hmr->...hnm', q_pos, k)
            score += torch.gather(p2c_score, dim=-2, index=p2c_index.unsqueeze(0).unsqueeze(0))
            
            score_scalar += 2
        
        if rotary_position_embeddings is not None:
            q = rotary_position_embeddings(q)
            k = rotary_position_embeddings(k)
        
        score = torch.matmul(q, k.transpose(-2, -1)) + score
        score_scalar += 1
        
        score /= score_scalar
        score = torch.pow(F.relu(score), 2)
        score /= (seq_len * self.dim_head)
        
        if attention_mask is not None:
            score = torch.masked_fill(
                score,
                attention_mask.unsqueeze(1) == False,
                torch.scalar_tensor(0, device=score.device, dtype=score.dtype)
            )
        
        if self.dropout:
            score = self.dropout(score)
        
        # score: batch_size, self.num_heads, seq_len, seq_len
        self.attention_score = score.detach()
        
        hidden_states = torch.matmul(score, v)
        hidden_states = hidden_states.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        return self.out_proj(hidden_states)


class QuadraticSelfAttentionWithFFNLayer(bmt.DistributedModule):
    
    def __init__(self,
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
        self.attention = QuadraticSelfAttention(
            dim_in = model_dim,
            dim_head = dim_head,
            num_heads = num_heads,
            dtype = dtype,
            linear_init_method = linear_init_method,
            dropout = dropout,
            dim_out = model_dim,
            bias = bias
        )
        self.ffn = FFN(
            dim_in = model_dim,
            dim_ff = dim_ff,
            dtype = dtype,
            dim_out = model_dim,
            linear_init_method = linear_init_method,
            bias = bias,
            activation = activation
        )
        self.attention_norm = RMSNorm(
            dim = model_dim,
            dtype = dtype
        )
        self.ffn_norm = RMSNorm(
            dim = model_dim,
            dtype = dtype
        )
        
        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None
            
    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor = None,
                relative_position_embeddings: torch.Tensor = None, # 全部[2 * max_seq_len, dim]
                rotary_position_embeddings: torch.nn.Module = None,
    ) -> torch.Tensor:
        shortcut = hidden_states
        hidden_states = self.attention(hidden_states, attention_mask, relative_position_embeddings, rotary_position_embeddings)
        hidden_states = self.attention_norm(shortcut + hidden_states)
        
        shortcut = hidden_states
        hidden_states = self.ffn(hidden_states)
        hidden_states = self.ffn_norm(shortcut + hidden_states)
        
        if self.dropout:
            hidden_states = self.dropout(hidden_states)
        
        return hidden_states