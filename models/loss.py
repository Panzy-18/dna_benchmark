import torch
import torch.nn.functional as F
import bmtrain as bmt
import os
import pdb

class OHEMBCELoss(torch.nn.Module):
    
    def __init__(self, alpha=0.5, **kwargs) -> None:
        super().__init__()
        self.alpha = alpha
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        
    def forward(self, 
                logits: torch.Tensor, 
                targets: torch.Tensor
    ) -> torch.Tensor:
        loss = self.criterion(logits, targets).view(-1)
        sorted_loss, idx = torch.sort(loss, descending=True)
        keep_num = int(sorted_loss.shape[0] * self.alpha + 0.5)
        keep_idx = idx[:keep_num]
        loss = loss[keep_idx].mean()
        return loss

class FocalBCELoss(torch.nn.Module):
    
    def __init__(self, alpha=0.8, gamma=2, scale=4, **kwargs):
        '''
        alpha: 正例的alpha权重
        gamma: 损失的衰减指数
        '''
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.scale = scale
    
    def forward(self, 
                logits: torch.Tensor, # [B, C]
                targets: torch.Tensor # [B, C]
    ) -> torch.Tensor:
        eps = 1e-8
        probs = logits.sigmoid()
        probs = torch.clamp(probs, min=eps, max=1-eps)
        confidence = torch.where(targets == 1, probs, 1 - probs) 
        pos_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        loss = -torch.mul(torch.pow(1 - confidence, self.gamma), confidence.log())
        loss = torch.mul(pos_weight, loss) * self.scale
        return loss.mean()

class CSELoss(torch.nn.Module):
    
    def __init__(self, alpha=0.05, **kwargs) -> None:
        super().__init__()
        self.alpha = alpha
    
    def forward(self, 
                rep_seq: torch.Tensor = None,
                rep_seq1: torch.Tensor = None, 
                rep_seq2: torch.Tensor = None,
    ):
        if rep_seq is None:
            device = rep_seq1.device
            dtype = rep_seq1.dtype
            rep_seq = torch.empty((rep_seq1.shape[0] * 2,) + rep_seq1.shape[1:], device=device, dtype=dtype)
            rep_seq[::2, ...] = rep_seq1
            rep_seq[1::2, ...] = rep_seq2
        else:
            device = rep_seq.device
            dtype = rep_seq.dtype
        idx = torch.arange(0, rep_seq.shape[0], device=device)
        idx1 = idx[None, :]
        idx2 = (idx + 1 - idx % 2 * 2)[:, None]
        y_true = (idx1 == idx2).to(dtype)
        sims = F.cosine_similarity(rep_seq.unsqueeze(1), rep_seq.unsqueeze(0), dim=-1)
        sims -= torch.eye(rep_seq.shape[0], device=device) * 1e12
        sims /= self.alpha
        loss = F.cross_entropy(sims, y_true)
        return loss

def get_criterion(
    loss_fn: dict
):
    if loss_fn is None:
        return None
    name = loss_fn['name']
    loss_fn.pop('name')
    return eval(name)(**loss_fn)