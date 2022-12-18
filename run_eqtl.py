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
import pdb
import torch
import numpy as np
import os
from glob import glob
import json
import xgboost as xgb

logger = get_logger(__name__)

def main():
    
    assert bmt.rank() == 0
    args = get_args()
    if args.load_dir is None:
        Warning('Please indicate the load_dir of model. Initialzing model from scratch is not recommended.')
    
    final_features, labels = get_features()
    get_scores(final_features, labels)
    
def get_score(features, labels, data_index, column_index):
    dtrain = xgb.DMatrix(
        data=features[data_index['train'], ...],
        label=labels[data_index['train'], column_index]
    )
    dvalid = xgb.DMatrix(
        data=features[data_index['valid'], ...],
        label=labels[data_index['valid'], column_index]
    )
    dtest = xgb.DMatrix(
        data=features[data_index['test'], ...],
        label=labels[data_index['test'], column_index]
    )
    
    param = {
        'eta': 0.3, 
        'max_depth': 3,  
        'n_estimators': 100,
        'eval_metric': ['aucpr', 'auc'],
        'early_stopping_rounds': 10,
    }
    clf = xgb.XGBClassifier(
        objective='binary:logistic',
        **param
    )
    clf.fit(
        dtrain.get_data(), dtrain.get_label(),
        eval_set=[(dtest.get_data(), dtest.get_label()), (dvalid.get_data(), dvalid.get_label()), ]
    )
    
    result = clf.evals_result()
    aupr = result['validation_0']['aucpr'][clf.best_iteration]
    auroc = result['validation_0']['auc'][clf.best_iteration]
    return clf, dict(
        aupr=aupr,
        auroc=auroc,
    )
    
def get_scores(features, labels):
    
    args = get_args()
    split_files = os.path.join(args.data_root, args.dataset_dir, '*:data_index.json')
    split_files = sorted(glob(split_files), key=lambda x: int(os.path.basename(x).split(':')[0]))
    auprs = []
    aurocs = []
    results = []
    for i, file in enumerate(split_files):
        tissue = os.path.basename(file).split(':')[1]
        logger.info(f'training {i}th tissue: {tissue}')
        with open(file) as f:
            data_index = json.load(f)
        model, result = get_score(features, labels, data_index, i)
        logger.info(f'tissue:{tissue} {result}')
        model.save_model(os.path.join(args.save_dir, 'model_for_tissue', f'{i}:tissue.json'))
        result['tissue'] = tissue
        results.append(result)
        auprs.append(result['aupr'])
        aurocs.append(result['auroc'])
    
    aupr = np.mean(auprs)
    auroc = np.mean(aurocs)
    logger.info(f'final: aupr={aupr} auroc={auroc}')
    with open('results.json', 'w') as f:
        for r in results:
            print(r, file=f)
    
def get_features():
    args = get_args()
    metadata = get_metadata()
    config = get_config()
    model = get_model(config)
    dataset = get_dataset(metadata, model.tokenizer)
    # only test
    testset = dataset['test']
    trainer = Trainer()
    all_datas = trainer.predict(model, testset)
    # ['id', 'labels', 'input_ids', 'preds']
    assert all_datas['id'][0][0] == str(0)
    assert all_datas['id'][-1][-1] == str(len(testset) - 1)
    preds = torch.cat(all_datas['logits_or_values'], dim=0)
    labels = torch.cat(all_datas['labels'], dim=0).numpy().astype(int)
    
    sequence_features = preds[::2, ...]
    alter_sequence_features = preds[1::2, ...]
    final_features = torch.cat([sequence_features, alter_sequence_features], dim=-1).numpy()
    np.save(os.path.join(args.save_dir, 'snp_features.npy'), final_features)
    np.save(os.path.join(args.save_dir, 'snp_labels.npy'), labels)
    
    return final_features, labels

if __name__ == '__main__':
    main()
