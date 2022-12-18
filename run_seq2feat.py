from tools import (
    get_args,
    get_config,
    get_metadata,
    Trainer,
)
from tools.logging import get_logger
from models import get_model
from datasets import get_dataset
import bmtrain as bmt
import torch
import numpy as np
import os
import h5py
import json

logger = get_logger(__name__)

def main():
    assert bmt.rank() == 0
    feature_dim = get_features()
    metadata_for_next = {
        "dataset_name": "expression218_feature",
        "description": "Predict expression level given sequence.",
        "dataset_args": {
            "dataset_class": "ExpressionFeatureDataset",
            "train_file": "expression218_feature/train.h5",
            "valid_file": "expression218_feature/valid.h5",
            "test_file": "expression218_feature/test.h5",
        },
        "model_args": {
            "task": "ModelForSequenceTask",
            "final_dim": 218,
            "loss_fn": {
                "name": "torch.nn.HuberLoss"
            },
            "input_embedding_dim": feature_dim
        },
        "metric": "spearmanr",
        "extra": None
    }
    with open(os.path.join('data/expression218_feature', 'metadata.json'), 'w') as f:
        json.dump(metadata_for_next, f)
    logger.info('metadata for expression218_feature done.')

    os.system('rm {}'.format(os.path.join(get_args().save_dir, 'events.out.*')))
    os.system('rm {}'.format(os.path.join(get_args().save_dir, '*.log')))

def get_features():
    args = get_args()
    metadata = get_metadata()
    config = get_config()
    model = get_model(config)
    dataset = get_dataset(metadata, model.tokenizer)
    trainer = Trainer()
    for split, splitset in dataset.items():
        if os.path.exists(os.path.join(args.save_dir, f'{split}.h5')):
            logger.info((f'{split} feature exists.'))
            continue
        datas = trainer.predict(model, splitset)
        # ['id', 'labels', 'input_ids', 'preds']
        assert datas['id'][0] == str(0)
        assert datas['id'][-1] == str(len(splitset) - 1)
        preds = torch.stack(datas['logits_or_values'], dim=0).numpy().astype(np.float16) #[num, n_chunk, dim]
        labels = torch.stack(datas['labels'], dim=0).numpy().astype(np.float16) #[num, n_tissue]
        with h5py.File(os.path.join(args.save_dir, f'{split}.h5'), 'w') as f:
            f.create_dataset('feature', data=preds, compression='gzip')
            f.create_dataset('label', data=labels, compression='gzip')
            logger.info((f'Split {split} with {len(preds)} samples.'))
    
    feat_dim = preds.shape[-1]
    return feat_dim

if __name__ == '__main__':
    main()
