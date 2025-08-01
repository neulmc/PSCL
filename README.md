# Patch-Sampled Contrastive Learning for Dense Prediction Pretraining in Metallographic Images

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.7+](https://img.shields.io/badge/pytorch-1.7+-red.svg)](https://pytorch.org/)

This repository contains the implementation of our paper "*Patch-Sampled Contrastive Learning for Dense Prediction Pretraining in Metallographic Images*", introducing a novel self-supervised learning framework specifically designed for microstructure analysis in metallographic images.

## Key Features
- **Dual-level Contrastive Learning**: Combines image-level and patch-level contrastive learning
- **Multi-scale Strategy**: Enhances feature learning across different scales
- **Smart Patch Sampling**: Feature similarity-based sampling for discriminative learning

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.7+
- CUDA 11.0+ (for GPU acceleration)

### Quick Start
1. Clone the repository:
   ```bash
   git clone https://github.com/neulmc/PSCL.git
   cd PSCL
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   
### Dataset Preparation
#### Aluminum Alloy Microstructure Dataset
Download our dataset from Baidu Drive: https://pan.baidu.com/s/18fBqMlGDj1s3bQGm41yycg?pwd=845d. (extraction code: 845d)

#### Dataset Structure:
```
dataset/
    ├── aluminum/               # Metallographic dataset
       ├── train_numf/          # Training dataset (full supervision)
           ├── img/
           ├── gt/
       ├── test_numf/           # Test dataset
           ├── img/
           ├── gt/
       ├── sup_num1_0/          # One supervised image (used for patch sampling)
           ├── img/             # Randomly obtained from the training set
           ├── gt/
       ├── ... 
```
#### Important Notes:
- 198 metallographic images (376×376 pixels)
- Covers holding times: 15/30 min (66), 45/60 min (60), 90/120 min (72)
- Academic Use Only - Commercial use prohibited

### Training Pipeline
#### 1. Self-Supervised Pretraining
Configure parameters in config.py and run:
```bash
python train.py -stages self
```
Key configurable parameters in train.py:
```
self_mode = simclr          # moco: PSCL-MoCo; simclr: PSCL-SimCLR
global_local_ratio = 0.7    # Image/patch loss ratio
stage = self                # self: pre-training; fine: fine-tuning; sup: supervised
```
#### 2. Supervised Fine-Tuning
After pretraining, run fine-tuning:
```bash
python train.py -stages fine
```
#### 3. Evaluation
The evaluation runs automatically after fine-tuning, generating:
- Prediction visualizations in {method_name}/1_0/epoch_xx/filename.png
- Performance metrics in {method_name}/1_0/log_fine.txt

### Reproducing Paper Results
Our implementation supports two variants:
- PSCL-MoCo: Momentum Contrast version
- PSCL-SimCLR: Simple Contrastive Learning version

### Results on Metallographic Dataset
| Method                 | Dice(Mean) | Dice(Std) | ACC(Mean) | ACC(Std) | 
|:-----------------------|:----------:|:---------:|:---------:|:--------:| 
| Baseline               |   0.5134   |  0.0609   |  0.9167   |  0.0373  |
| MoCo                   |   0.5696   |  0.0683   |  0.9448   |  0.0181  | 
| SimCLR                 |   0.5712   |  0.0301   |  0.9329   |  0.0368  | 
| BYOL                   |   0.5437   |  0.0743   |  0.9339   |  0.0232  | 
| DenseCL                |   0.5622   |  0.0377   |  0.9356   |  0.0227  | 
| PSCL-MoCo (proposed)   |   0.6284   |  0.0316   |  0.9546   |  0.0108  | 
| PSCL-SimCLR (proposed) |   0.6296   |  0.0522   |  0.9574   |  0.0086  | 

### Final models
This is the pre-trained model and log file in our paper. We used this model for fine-turning and evaluation. You can download by:
https://pan.baidu.com/s/19cvVEM4Bz-i4gHZrnQA5tw?pwd=bxm8 code: bxm8.

### Code Structure
```
PSCL-github/
├── config.py             # Configuration parameters
├── config_run.py         # Execute code
│   ├── Supervised         
│   ├── SelfSupervised       
│   └── Finetune       
├── data.py               # Data loading and augmentation
├── model.py              
│   ├── PSCL_MoCo         # MoCo variant implementation
│   └── PSCL_SimCLR       # SimCLR variant implementation
├── networks.py           # Networks(UNet) used in model
├── train.py              # Main training script
├── utils                 # Utility functions
└── requirements.txt      # Required packages
```

### Custom Dataset Training
To train on your own dataset:
1. Prepare images and corresponding annotations
2. Create directory structure matching our dataset
3. Update paths "data_dir" in config.py or train.py
4. Run the training pipeline

**Note**: This repository is designed for metallographic image segmentation tasks. The image naming convention is "ProductionBatch(aging times)-sampleID(id)-x-x". The defined dataloader will identify the aging times, IDs, and other information, and feed this information into the PSCL-MoCo queue for comparative learning. 
If using PSCL-SimCLR, this information is not involved in model training.

**Examples**: As a minimal implementation example on other datasets, 
you can run the proposed **PSCL-SimCLR** by running python train_defect.py. 
This dataset is a publicly available defect detection dataset (**<a href="https://ieeexplore.ieee.org/document/8930292"> NEU-Seg </a>**) used for segmentation tasks. 
We randomly selected <a href="https://pan.baidu.com/s/1hxAv0htQYrhFoR_-NHHcSA?pwd=8uqb"> 8 images </a> as a simple demo. 

This demo is implemented without changing the source code.
For a specific and user-defined dataset, appropriate modifications are necessary. 
For example, the number of dataset categories should be adjusted in the model.py 
and config.py files (the current dataset has 4 microstructure categories).

### References
[1] <a href="https://github.com/facebookresearch/moco">MoCo: Improved baselines with momentum contrastive learning.</a>

[2] <a href="https://github.com/google-research/simclr">SimCLR: A simple framework for contrastive learning of visual representations.</a>

[3] <a href="https://github.com/WXinlong/DenseCL">DenseCL: Dense Contrastive Learning for Self-Supervised Visual Pre-Training.</a>

[4] <a href="https://github.com/cthoyt/autoreviewer"> AutoReviewer </a>

[5] <a href="https://ieeexplore.ieee.org/document/8930292"> NEU-Seg dataset </a>

