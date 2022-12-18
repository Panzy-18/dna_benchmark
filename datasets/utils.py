from typing import Union, List, Tuple, Any, Dict
import torch
import numpy as np
import json
from dataclasses import dataclass, field

def sigmoid(a: np.ndarray):
    return torch.tensor(a).to(torch.float).sigmoid().numpy().astype(np.float16)

def pad_sequences(
    sequences: Union[List[np.ndarray], List[torch.Tensor]],
    pad_value = 0,
    target_shape: Union[int, List[int], Tuple[int]] = None
):
    '''
    pad 1d or more, input shape: [bsz, ...]
    '''
    batch_size = len(sequences)
    if target_shape is None:
        shape = [batch_size] + np.max([seq.shape for seq in sequences], 0).tolist()
    else:
        if isinstance(target_shape, int):
            target_shape = [target_shape]
        else:
            target_shape = list(target_shape)
        shape = [batch_size] + target_shape
    
    dtype = sequences[0].dtype
    if isinstance(sequences[0], np.ndarray):
        array = np.full(shape, pad_value, dtype=dtype)
    elif isinstance(sequences[0], torch.Tensor):
        array = torch.full(shape, pad_value, dtype=dtype)
    
    for arr, seq in zip(array, sequences):
        arrslice = tuple(slice(dim) for dim in seq.shape)
        arr[arrslice] = seq
        
    return array