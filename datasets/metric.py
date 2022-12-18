from typing import Union
from sklearn.metrics import (precision_recall_curve,
                             roc_curve,
                             auc,
                             accuracy_score)
import numpy as np
import torch
from .utils import sigmoid
from scipy.stats import spearmanr

__all__ = [
    'get_acc',
    'get_auroc',
    'get_aupr',
    'get_f_score',
    'get_spearmanr',
    'get_mse',
    'STR2METRIC'
]

def turn_to_np(a):
    if not isinstance(a, np.ndarray):
        return np.array(a)
    return a

def turn_to_tensor(a):
    if not isinstance(a, torch.Tensor):
        return torch.tensor(a)
    return a

def get_mse(targets: Union[np.ndarray, torch.Tensor],
            preds: Union[np.ndarray, torch.Tensor],
):
    targets = turn_to_tensor(targets)
    preds = turn_to_tensor(preds)
    return torch.nn.functional.mse_loss(preds, targets).item()

def get_spearmanr(targets: Union[np.ndarray, torch.Tensor],
                  preds: Union[np.ndarray, torch.Tensor],
):
    targets = turn_to_np(targets)
    preds = turn_to_np(preds)
    return spearmanr(preds, targets).correlation

def get_acc(targets: Union[np.ndarray, torch.Tensor],
            preds: Union[np.ndarray, torch.Tensor],
):
    '''
    for binary classification
    '''
    preds = turn_to_np(preds)
    targets = turn_to_np(targets)
    if preds.max() > 1 or preds.min() < 0:
        preds = sigmoid(preds)
    preds = np.where(preds < 0.5, 0, 1)
    return accuracy_score(preds, targets)

def get_auroc(targets: Union[np.ndarray, torch.Tensor],
              preds: Union[np.ndarray, torch.Tensor],
):
    '''
    for binary classification
    '''
    preds = turn_to_np(preds)
    targets = turn_to_np(targets)
    if preds.max() > 1 or preds.min() < 0:
        preds = sigmoid(preds)
    fpr, tpr, thresholds = roc_curve(targets, preds, drop_intermediate=False)
    return auc(fpr, tpr)

def get_aupr(targets: Union[np.ndarray, torch.Tensor],
             preds: Union[np.ndarray, torch.Tensor],
):
    '''
    for binary classification
    '''
    preds = turn_to_np(preds)
    targets = turn_to_np(targets)
    if preds.max() > 1 or preds.min() < 0:
        preds = sigmoid(preds)
    precisions, recalls, ths = precision_recall_curve(targets, preds)
    return auc(recalls, precisions)

def get_f_score(targets: Union[np.ndarray, torch.Tensor],
                preds: Union[np.ndarray, torch.Tensor],
                beta: float = 1.    
):
    '''
    for binary classification
    '''
    preds = turn_to_np(preds)
    targets = turn_to_np(targets)
    if preds.max() > 1 or preds.min() < 0:
        preds = sigmoid(preds)
    precisions, recalls, ths = precision_recall_curve(targets, preds)
    f_score = (1 + beta ** 2) * (precisions * recalls) / ((beta ** 2) * precisions + recalls + 1e-8)
    return np.max(f_score)

STR2METRIC = {
    'acc': get_acc,
    'auroc': get_auroc,
    'aupr': get_aupr,
    'f1': get_f_score,
    'spearmanr': get_spearmanr,
    'mse': get_mse
}