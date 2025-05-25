# Syn2DRE

Implementation of **Synergistic Dual Teacher-Student Framework for Document-level Relation Extraction**

## Environment Setup
```sh
>> conda create --name syn2dre python=3.11
>> conda activate syn2dre
>> pip install -r requirements.txt
```
## Dataset
We perform experiments on [DocRED](https://github.com/thunlp/DocRED) and [Re-DocRED](https://github.com/tonytan48/Re-DocRED).

Please note that the `test.json` in `docred` folder does not contain ground-truth labels.

## Folder Structure
Due to large size, we do not include the dataset here. You can download the `dataset` folder from this [link](https://drive.google.com/file/d/1OdPWZpP_0PkBluohPkjKC8W17LPcOHh1/view?usp=sharing). After downloading, upzip and place `dataset` under the `Syn2DRE` folder as structure below:
```
Syn2DRE
├── dataset
│   ├── ...
├── FT-stage
│   ├── ...
├── KD-DocRE
│   ├── ...
└── UGDRE
    ├── ...
```

## Training & Evaluation
### Pre-training Phase
#### Pre-training KD-DocRE
To pre-train KD-DocRE, run:
```sh
>> bash Syn2DRE/KD-DocRE/scripts/pretrain_kddocre.sh
```

#### Pre-training UGDRE
To pre-train UGDRE, run:
```sh
>> bash Syn2DRE/UGDRE/scripts/pretrain_ugdre.sh
```

### Fine-tuning Phase
To reproduce the Fine-tuning Phase, run:

#### DocRED dataset
```sh
>> bash Syn2DRE/FT-stage/scripts/train_docred.sh
```

#### Re-DocRED dataset
```sh
>> bash Syn2DRE/FT-stage/scripts/train_redocred.sh
```