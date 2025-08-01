"""
PSCL (Method Name) Training Script

This script handles the training pipeline for the PSCL model, including:
- Self-supervised pre-training
- Supervised fine-tuning
- Multi-round experiments with different data configurations

Usage:
1. Configure parameters in the main block
2. Run: python train.py
"""

import config
import argparse

name = 'PSCL-defect-demo'
data_dir = 'dataset/NEU_Seg-demo/'
stages = 'self-fine'
# stage: 'self-fine' means running pre-training and fine-tuning in turn
# (self): pre-training;
# (fine): fine-tuning
# (sup): without pre-training
self_mode = 'simclr'
# self_mode
# moco: PSCL-MoCo; simclr: PSCL-SimCLR
data_size = 'f'
# data_size:
# 1(only one supervised image);
# 5(five supervised images)
# f(normal supervised learning)
multi_round = True
# multi_round:
# True: Repeated experiments multiple times with different samples;
# False: Perform only one experiment
temperature = 0.5
# hyper-parameters
global_local_ratio = 0.7
# hyper-parameters
GPU_env = '0'
# NVIDIA GPU ID selection

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-stages", default=stages, type=str, required=False, help="Specify the model stage")
    args = parser.parse_args()
    stages = args.stages

    if multi_round:
        RepeatID = '01234' # Run 5 rounds for statistical properties (mean Dice & std Dice)
    else:
        RepeatID = '0'

    file_name = name + '_' + self_mode
    stage = stages.split('-')
    # top_k: number of used annotated samples
    if self_mode == 'moco':
        top_k = 30
    elif self_mode == 'simclr':
        top_k = 4
    for stage_ in stage:
        config.self2fine(tt=name, method=stage_, env=GPU_env, RepeatID=RepeatID, mode=data_size, selfmode=self_mode,
                         temperature=temperature, moco_denseloss_ratio=global_local_ratio, data_dir = data_dir, top_k = top_k)