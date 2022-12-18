from dataclasses import asdict, dataclass, field
import json
import torch

LINEAR_INIT = {'func': torch.nn.init.trunc_normal_, 'mean': 0., 'std': 0.02, 'a': -0.04, 'b': 0.04}
EMBED_INIT = {'func': torch.nn.init.trunc_normal_, 'mean': 0., 'std': 1}

@dataclass(frozen=False)
class BaseConfig:
    """
    handle model architecture:
        - embedding
        - convolution layers
        - transformer layers
        - task specific
    """
    # attention part
    model_name: str = "attention"
    model_size: str = "mini"
    abs_pos: bool = True 
    rel_pos: bool = False
    rot_pos: bool = False
    num_layers: int = 4
    model_dim: int = 512
    dim_ff: int = 2048
    dim_head: int = 64
    num_heads: int = 8
    pooling_method: str = "cls" # cls/mean/attention
    activation: str = "gated_gelu"
    
    # tokenization&embedding part
    vocab_size: str = 5
    tokenization: str = "onehot"
    max_input_bp_seq_len: int = 1024
    max_attention_seq_len: int = 1025 # cls-token
    conv_layers: list = field(default_factory=list)
    flatten_dim: int = None
    
    # other model hyper-parameters
    dropout: float = 0.15
    bias: bool = False
    half: bool = False
    
    # task specific (head)
    task: str = None
    with_head: bool = True
    input_embedding_dim: int = None
    final_dim: int = None
    loss_fn: dict = field(default_factory=dict)
    tscam: bool = False
    
    # about training
    
    def __post_init__(self):
        self._extra = {}
        self.dtype = torch.half if self.half else torch.float
        self.linear_init_method = LINEAR_INIT
        self.embedding_init_method = EMBED_INIT
        self._extra["dtype"] = self.dtype
        self._extra["linear_init_method"] = self.linear_init_method
        self._extra["embedding_init_method"] = self.embedding_init_method
        
        if self.tscam and self.model_name != 'attention':
            raise ValueError('visualize attention heatmap only when model is attention.')
        if not self.with_head:
            self.final_dim = self.model_dim
    
    def to_dict(self):
        '''
        与asdict方法不同的是, post_init的值也会被包括进
        '''
        ret = asdict(self)
        ret.update(self._extra)
        return ret
    
    def update_from_metadata(self, metadata):
        for k, v in metadata.model_args.items():
            setattr(self, k, v)
            if k not in self.__annotations__.keys():
                self._extra[k] = v
        if self.final_dim == -1:
            self.with_head = False
    
    @classmethod
    def load(cls, file):
        with open(file, 'r') as f:
            return cls(**json.load(f))
    
    def save(self, file):
        with open(file, 'w') as f:
            kwargs = asdict(self)
            json.dump(kwargs, f)
        