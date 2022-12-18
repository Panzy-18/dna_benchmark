import sys
sys.path.append("..")
from tools import get_logger
import torch
from .layer import Conv1d
import pdb

logger = get_logger(__name__)

class VisualHead(torch.nn.Module):
    
    def __init__(self,
                 model_dim: int,
                 num_classes: int,
                 dtype: torch.dtype,
                 for_each_class: bool = False
    ) -> None:
        super().__init__()
        self.for_each_class = for_each_class
        self.num_classes = num_classes
        if for_each_class:
            self.conv = Conv1d(
                dim_in=model_dim,
                dim_out=num_classes,
                kernel_size=11,
                padding="same",
                stride=1,
                dtype=dtype,
                activation='relu'
            )
        else:
            logger.info('No class specific visualization.')

    def forward(self,
                rep_token: torch.Tensor,
                logits_or_values: torch.Tensor,
                pooling_score: torch.Tensor,
    ):
        
        if self.for_each_class:
            # 并不往回传梯度
            rep_token = rep_token.detach()
            # convert rep_token to semantic-aware map
            feature_map: torch.Tensor = self.conv(rep_token) # [bsz, n_token, n_classes]
            if self.training:
                logits_or_values += feature_map.mean(dim=-2)
            tscam = pooling_score.unsqueeze(-1) * feature_map # [bsz, n_token, -1]
        else:
            tscam = pooling_score.unsqueeze(-1).expand(-1, -1, self.num_classes)
        
        return logits_or_values, tscam