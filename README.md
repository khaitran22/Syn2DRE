# Syn2DRE

Implementation of **Synergistic Dual Teacher-Student Framework for Document-level Relation Extraction**

## Installation
```
  conda create --name syn2dre python=3.11
  conda activate syn2dre
  pip install -r requirements.txt
```
## Dataset
We perform experiments on [DocRED](https://github.com/thunlp/DocRED) and [Re-DocRED](https://github.com/tonytan48/Re-DocRED).
```
root
└── dataset
    ├── dev.json
    ├── dev_revised.json
    ├── meta
    │   ├── ner2id.json
    │   └── rel2id.json
    ├── test.json
    ├── test_revised.json
    ├── train_annotated.json
    ├── train_distant.json
    └── train_revised.json
```

## Training & Evaluation

### Pre-training Phase
Pre-training Phase can be reproduced by following steps:
- Fork the Github repository of [KD-DocRE](https://github.com/tonytan48/KD-DocRE/tree/main) and [UGDRE](https://github.com/QiSun123/UGDRE).
- Copy `pretrain_kddocre.sh` and `pretrain_ugdre.sh` under `/scripts` folder to corresponding Github repos.
- Follow their instructions in specifying `PATH` for dataset setup.

#### KD-DocRE
```
>> bash pretrain_kddocre.sh
```
#### UGDRE
```
>> bash pretrain_ugdre.sh
```

### Fine-tuning Phase
To reproduce the Fine-tuning Phase:
- Move pre-trained KD-DocRE and UGDRE models to the `/checkpoints` folder

#### Training on DocRED dataset
```
>> bash scripts/train_docred.sh
```

#### Training on Re-DocRED dataset
```
>> bash scripts/train_redocred.sh
```