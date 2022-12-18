## Overview

This repo contains 6 DNA task and scripts for quick experiment.

**Note:** We strongly recommend that you browse the overall structure of our code at first. If you have any question, feel free to contact us.

## Model Construction

In this frameowork, we support transformer and convolution based model. You can change model hyper-parameters by simply modifying the model config. Examples are listed in `./config`

## Benchmark

| Name          | Objective          | Input_length     | Output            | Dataset                                         | Main Metric |
| ------------- | ------------------ | ---------------- | ----------------- | ----------------------------------------------- | ----------- |
| promoter      | Promoter           | 500              | Probability, 1    | Train: 94159, Valid: 11770, Test: 11770         | AUROC       |
| encode690     | TFBS               | 200+800(flank)   | Probability, 690  | Train: 4472062, Valid: 279494, Test: 454380    | AUPR        |
| methyl96      | Methyl Probability | 1+1000(flank)    | Probability, 96   | Train: 1172067, Valid: 93046, Test: 181329   | SpearmanR   |
| track7878     | TF/DNase/Histone   | 200+800(flank)   | Probability, 7878 | Train: 26880666, Valid: 280006, Test: 840022 | AUPR        |
| eqtl49        | Casual SNP         | 1+1000(flank)    | Probability, 49   | SNP number: 216330                              | AUROC       |
| expression218 | Gene Expression    | 200+40800(flank) | log RPKM, 218     | Train: 21840, Valid: 987, Test: 990             | SpearmanR   |

More detailed description is in `./data/$dataset/metadata.json`. Preprocessed pipeline in `./preprocess`. 

## Usage

#### Environment

```shell
pip install torch
pip install -r requirements.txt
```

#### Pretrained Model and Dataset

Download pre-trained models from the following links.

- [Unsupervised Pretrained Model on HS1](https://drive.google.com/file/d/19oWVyIWrEG3wBpFihEdCRcXSpns4crhV/view?usp=sharing)
- Pretrained Model on Track7878

All the datasets are processed from open resources. Download and preprocessing scripts are listed in `./preprocess`. Run the scripts to generate data in your local environments. You can also download data in this [link](https://drive.google.com/file/d/19RyxTCcFzdST4-l42STol0ExNgTAXdKl/view?usp=sharing).

Promoter dataset is an example. This required hg38.fa in `./data/genome`

#### Run

Run experiment simply by default setting:

```
python run_task.py --dataset-dir [directory_in_data_root] \
	--save-dir [directory_to_save_experiment] 
```

For example, you can train your promoter model from scratch by:

```
python run_task.py --dataset-dir promoter --save-dir experiment/promoter_default
```

Run `python run_task.py -h` to check all the arguments. Check more examples `./scripts` folder

#### Customize Dataset

We provide a easy-to-use customized data pipeline. If you want to start experiment on your own dataset, organize your file:

```
- data_root
    - customized_data
	- metadata.json
	- train.json
	- ...
```

In metadata, you must specify fields like: (check more examples in `./data` folder)

```json
{
    "dataset_name": ...,
    "dataset_args": {
        "dataset_class": "DNATaskDataset",
        "ref_file": "genome/hg19.fa", # if do not need, use 'null'
        "train_file": "customized_data/train.json", # pass train_file to make trainer train model
	"valid_file": ..., # pass valid_file to make trainer evaluate model after each epoch for training
	...
    },
    "model_args": {
        "task": "ModelForSequenceTask",
        "final_dim": ...,
        "loss_fn": {
            "name": ...,
        }
    },
    "metrics": ..., # the first metric will be the main score to save best model
}
```

We support JSON or HDF5 format data file. In JSON, a sample is structured like:

```json
{"sequence": "ATGGCTC", "label": [1, 0]} 
or 
{"index": ["chr1", 0, 7, "+"], "label": [1, 0]}
```

In HDF5(for huge dataset storation), a sample is structured by two fields: `index` and `label`.

```json
index: np.array([1, 0, 7, 1]) # (chr_num, start_pos, end_pos(exclusive), is_forward)
label: np.array([1, 0])
```

#### Visualize

We support two modes of visualization. You can get roll-out attention score without any modification to model, however, this method sometimes does not perform well. (See `./visual_result/without_tscam`)

We used [TS-CAM](https://github.com/vasgaowei/TS-CAM) to enhance visualization in transformer-based model. To utilize its advantages, you should pass `--tscam` when training model on a certain dataset. Then, model will provide informative class-specific visualization result. See example code in `visualize.py`

```
python visualize.py --load-dir experiment/promoter --save-dir visual_result/promoter
```

#### Prediction

Use `model.predict` method to infer for short sequence task (<1024bp) on your own data. For example:

```python
# example for promoter detection.
# run by 'python $file --load-dir experiment/promoter'
from tools import get_config
from models import get_model

config = get_config()
model = get_model(config)
sequences = ['ATTCATCCAACTCTCCGTGAGCTCCCCTGGGTAGGAGTACAGTGGCAGCCAGTGTCCCCAGAAAACTGGCGCCTCCCCCCTCGCCGTGCGGGGCTAATTAACTCTTAGCCGGCGGGACCCTCCTCCTCCTCGGAGGTTGGCCAGGAGCAGCGCGGCATCCCAGGCGTTCCTGTCTGATGTCATAGGCTGCCGGCGATTGCGGAGAATCGCCACCACGCCTTTATGAAGGTCCCAACTTTGCCATCTGATACCCTTTACTACTGACAGGCGCTCAGCCAATCAGGAGCGGCGAGCGGGGTCTGGGGACCCGGAGCCGCCGAAGCCGTCTCGGGAACCGGCTCTTAACTCTTTGCGGCGGGCCCCGCAGCCGCCGAGGCACAGAGGGCGGGAGCAGGGCCAGGGGTCGGGAATCTGGGAGAGGGGCGCGAGCTAAAGAGCGGATGCCCGGAGGAAAGAAGGAAGGGCTGCGACGCCGCGGGGCTTGCAGGTGGTTCGCGGGG',
            'ATGAAATACACATAAAAAACACACACATTAAATATTAATATATGCTTATTATTGTATTATGAATGAGGAAATAAAATATAACTTGGAATTTTTTTAAAACTTAAAAAAATACAATGGACTGAGCACTGAAATCAGAATATGCAGCTTATTTAGAACAAAATTCTACTTTTTCCCCTAAACTGTCCCTTAACATTGTCATCTCTCCTGCTAATCCTGCATTACCCTGGATCCTTCCTTTTTGTCTCTGCCTCCACTCACTGCTGCCTCTGCCATAAGCCTTCATACTCCAGCTGCTACACACTGCTGCTTCTATCCCTGAGGATTCCACGAGCATCCTTATTCTTCTGTCACTGATATGGTTCCTATTGGCATATCAAAAGTTATAGCCATATGAAGAAAAATCTAGGGATGCAGCAGCAGCAGCAGCAGTAGCAGTAGCAGCAACAGTCTATCAAGATGTTTTAATCTGGAATAAATTTCAGAATAGATCAATTCAGCAT'
            ] # 2 positive sample
model_output = model.predict(sequences)
print(model_output.logits_or_values)

```

For expression task, see example in `predict_expression.py`
