"""
Configuration and Training Pipeline for PSCL
This file contains:
1. Main configuration class with all hyperparameters
2. Training loops for supervised, self-supervised, and fine-tuning phases (now in config_run.py)
3. Utility functions for model evaluation and logging
"""

import numpy as np
import os
from model import UNet, MoCo_DenseModel
import torch
import random
from config_run import Supervised, SelfSupervised, Finetune

# This class covers almost all hyperparameters,
# including Model and method configuration, Loss weights, learning strategies, data augmentations, etc.
# In fact, most of them are default, but for flexibility, we can modify them again
class config():
    def __init__(self, tt, method, backbone, moco_mode, moco_denseloss_ratio, dice_bce_ratio, load_moco_ep, weight,
                 dice_weight, sup_num, epoch_map, batch_size, self_batch_size, self_max_epoch, self_save_epoch,
                 jitter_d, random_c, super_lr, self_lr, queue_size, queue_momentum, temperature,
                 lab_size_decay, env, seed, print_freq, mapping, patch_num, verb,
                 drop, channel, opti, sche, adamw_decay, dense_weight, fineturn_lr_en, fineturn_lr_de, sample_num,
                 self_scale_weight, sample_ratio, confea_num, hidfea_num, top_k, patch_lab_decay, patch_size_decay,
                 multihead,
                 IncNorm, DownNorm, UpNorm, HeadNrom, global_weight, Patch_sup, selfmode, simclrinner, data_dir):

        # Model and method configuration
        self.tt = tt  # Experiment tag
        self.method = method  # 'sup', 'self', or 'fine'
        self.backbone = backbone  # Model architecture (UNet)
        self.moco_mode = moco_mode  # our PSCL or normal MoCo

        # Loss parameters
        self.moco_denseloss_ratio = moco_denseloss_ratio  # [0-1] dense loss ratio
        self.dice_bce_ratio = dice_bce_ratio  # Dice vs BCE loss ratio
        self.load_moco_ep = load_moco_ep  # Epoch to load pre-trained model
        self.weight = weight # Class weights
        self.dice_weight = dice_weight # Dice weights
        self.dense_weight = dense_weight # Dense contrast loss weight
        self.global_weight = global_weight # Global contrast loss weight

        # Model Architecture
        self.drop = drop  # dropout ratio
        self.channel = channel  # channel number
        self.confea_num = confea_num # final feature number in Denseproj block
        self.hidfea_num = hidfea_num # hidden feature number in Denseproj block
        self.multihead = multihead # head number in Denseproj block
        self.IncNorm = IncNorm # Normalization (BN, LN, ...)
        self.DownNorm = DownNorm # Normalization (BN, LN, ...)
        self.UpNorm = UpNorm # Normalization (BN, LN, ...)
        self.HeadNrom = HeadNrom # Normalization (BN, LN, ...)

        # Runtime setting
        self.super_lr = super_lr # Supervised LR
        self.self_lr = self_lr # Self-supervised LR
        self.fineturn_lr_en = fineturn_lr_en # Fine-turning LR
        self.fineturn_lr_de = fineturn_lr_de # Fine-turning LR
        self.opti = opti # Optimizer
        self.sche = sche # Scheduler for LR
        self.adamw_decay = adamw_decay # decay for adamw
        self.sup_num = sup_num  # Supervised data amount ('1'-one annotation,'5'-five annotations,'f'-'full')
        self.epoch_map = epoch_map # Number of epochs under different amounts of supervision
        self.batch_size = batch_size # batch size
        self.self_batch_size = self_batch_size # batch size in self-supervised learning
        self.self_max_epoch = self_max_epoch # max epoch
        self.self_save_epoch = self_save_epoch # save model interval
        if '1_' in sup_num:
            self.supv_max_epoch = 1000
            self.fine_max_epoch = 1000
            self.batch_size = 1
        elif '5_' in sup_num:
            self.supv_max_epoch = 500
            self.fine_max_epoch = 500
        elif sup_num == 'f':
            self.supv_max_epoch = 50
            self.fine_max_epoch = 50

        # Patch sampling setting
        self.top_k = top_k # number of used annotated samples
        self.patch_lab_decay = patch_lab_decay # weight decay for labels
        self.patch_size_decay = patch_size_decay # weight decay for labels
        self.sample_num = sample_num # sample number
        self.self_scale_weight = self_scale_weight # weights of different scales in multi-scale Strategy
        self.sample_ratio = sample_ratio # sample ratio
        self.lab_size_decay = lab_size_decay  # label decay
        self.patch_num = patch_num # patch num for contrast learning

        # Data setting
        self.jitter_d = jitter_d  # Jitter intensity
        self.random_c = random_c  # Random crop ratio

        # Used for ablation study
        self.Patch_sup = Patch_sup # patch supervised sampling or not

        # MoCo specific
        self.queue_size = queue_size  # Size of Memory Queue, Must be Divisible by batch_size. 256
        self.queue_momentum = queue_momentum  # Momentum for the Key Encoder Update.
        self.temperature = temperature  # Contrastive temperature
        self.selfmode = selfmode # build self-supervised mode for 'moco','simclr', '...'
        self.simclrinner = simclrinner # if 'simclr', it works

        # Environment
        os.environ["CUDA_VISIBLE_DEVICES"] = env # GPU selection
        # torch.set_num_threads(2)  # no used
        self.seed = seed  # Random seed

        # Printer setting
        self.print_freq = print_freq
        self.test_freq = int(100 / self.epoch_map[self.sup_num.split('_')[0]])
        self.mapping = mapping
        self.verb = verb

        # build work dir
        self.tmp = self.method + '_' + self.backbone.__name__ + '_' + self.tt + '/'
        if not os.path.exists(self.tmp):
            os.makedirs(self.tmp)
        self.dataset_sup = data_dir + '/sup' + '_num' + self.sup_num
        self.dataset_train = data_dir + '/train' + '_numf'
        self.dataset_test = data_dir + '/test' + '_num' + self.sup_num.split('_')[0]

# This function will call the training function in config_run according to the specified mode,
# such as: fine-tuning, pre-training, etc. by arg "method".
def runner_preheat(tt='typical1', method='fine', backbone=UNet, moco_mode=MoCo_DenseModel,
                   dice_bce_ratio=0.5, load_moco_ep='200', weight=np.array([1, 1, 10, 10]),  # load 100
                   dice_weight=np.array([0, 1, 1, 1]),
                   sup_num='f', epoch_map={'1': 0.5, '5': 1, 'f': 10}, batch_size=6, self_batch_size=8,
                   self_max_epoch=200,
                   self_save_epoch=10, jitter_d=0.2, random_c=0.2, super_lr=1e-3, self_lr=1e-4,  #
                   queue_momentum=0.99, temperature=0.07, env='1', seed=10, print_freq=1000,
                   mapping={0: 0, 1: 128, 2: 64, 3: 255}, patch_num=4, verb=True,
                   drop=0, channel=[32, 64, 128, 256, ], opti='adam', sche=False, adamw_decay=1e-3,
                   fineturn_lr_en=1e-4, fineturn_lr_de=1e-3, sample_num=50, self_scale_weight=[1, 1, 1, 1],
                   sample_ratio=2,
                   confea_num=64, hidfea_num=128, IncNorm=['BN', 'BN'], DownNorm=['BN', 'BN'], UpNorm=['BN', 'LN'],
                   HeadNrom=['LN'],
                   global_weight=1, dense_weight=1, multihead=4, Patch_sup=True,  # ablation
                   patch_size_decay=0.5, patch_lab_decay=0.2, lab_size_decay=0.2,  # decay
                   queue_size=504, top_k=30,  # moco top_k=30
                   moco_denseloss_ratio=0.5,  # loss ratio
                   selfmode='moco',  # 'moco','simclr','byol'
                   simclrinner = True,
                   data_dir = ''
                   ):
    """Main training runner that initializes config and starts training"""
    config_lmc = config(tt, method, backbone, moco_mode, moco_denseloss_ratio, dice_bce_ratio, load_moco_ep, weight,
                        dice_weight, sup_num, epoch_map, batch_size, self_batch_size, self_max_epoch, self_save_epoch,
                        jitter_d, random_c, super_lr, self_lr, queue_size, queue_momentum, temperature,
                        lab_size_decay, env, seed, print_freq, mapping, patch_num, verb, drop, channel, opti, sche,
                        adamw_decay,
                        dense_weight, fineturn_lr_en, fineturn_lr_de, sample_num, self_scale_weight, sample_ratio,
                        confea_num, hidfea_num, top_k, patch_lab_decay, patch_size_decay, multihead, IncNorm, DownNorm,
                        UpNorm, HeadNrom,
                        global_weight, Patch_sup, selfmode, simclrinner, data_dir)
    # Set random seed
    seed_torch(config_lmc.seed)
    # Start appropriate training phase
    if config_lmc.method == 'sup':
        Supervised(config_lmc)
    elif config_lmc.method == 'self':
        SelfSupervised(config_lmc)
    elif config_lmc.method == 'fine':
        Finetune(config_lmc)

    return

# This function is used to set the random seed, but it is still limited by the torch library version
# Random initialization results may vary on different torch versions and operating systems.
def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)

# It is used for analyzing the performance of the model from log file (self-supervised mode).
def analyse_log(tt, RepeatID, mode):
    def read_log(file):
        mIoU = -1
        ACC = 0
        loss = 1e6
        mIoU_epoch = -1
        with open(file, "r") as f:
            lines = f.readlines()
            for line in lines:
                if 'Epoch: [' in line:
                    epoch = int(line.split('[')[1].split('/')[0])
                    loss_now = float(line.split('Loss ')[1].split(' (')[0])
                    if loss_now < loss:
                        loss = loss_now
                        Loss_epoch = epoch
                elif 'ACC:' in line:
                    ACC_now = float(line.split(':')[1])
                elif 'mIoU:' in line:
                    mIoU_now = float(line.split(':')[1])
                    if mIoU < mIoU_now:
                        mIoU = mIoU_now
                        mIoU_epoch = epoch
                        ACC = ACC_now
        return mIoU, ACC, mIoU_epoch, loss, Loss_epoch

    root_dir = 'self_UNet_' + tt
    data1_dir = 'fine_UNet_' + tt + '/_Num1'
    data5_dir = 'fine_UNet_' + tt + '/_Num5'
    dataf_dir = 'fine_UNet_' + tt + '/_Numf'
    mIoU1s = []
    mIoU5s = []
    ACC1s = []
    ACC5s = []
    with open(root_dir + '/' + 'eval.txt', 'w') as f:
        f.writelines('\n' + 'Final self:')
        if '1' in mode:
            f.writelines('\n' + 'Num1:')
            for i in range(5):
                if str(i) in RepeatID:
                    mIoU, ACC, mIoU_epoch, loss, Loss_epoch = read_log(data1_dir + '/1_' + str(i) + '/log_fine.txt')
                    f.writelines('\n' + str(i) + ': [mIoU]: ' + str(mIoU) + ' [ACC]: ' + str(ACC) +  '[mIoU_epoch]: ' + str(mIoU_epoch) +
                                 '[loss]: ' + str(loss) + '[Loss_epoch]: ' + str(Loss_epoch))
                    mIoU1s.append(mIoU)
                    ACC1s.append(ACC)
            f.writelines('\n' + 'Total' + ': [mIoU]: ' + str(np.mean(np.array(mIoU1s))) + ' [ACC]: ' + str(np.mean(np.array(ACC1s))))
        if '5' in mode:
            f.writelines('\n' + 'Num5:')
            for i in range(5):
                if str(i) in RepeatID:
                    mIoU, ACC, mIoU_epoch, loss, Loss_epoch = read_log(data5_dir + '/5_' + str(i) + '/log_fine.txt')
                    f.writelines('\n' + str(i) + ': [mIoU]: ' + str(mIoU) + ' [ACC]: ' + str(ACC) + '[mIoU_epoch]: ' + str(mIoU_epoch) +
                                 '[loss]: ' + str(loss) + '[Loss_epoch]: ' + str(Loss_epoch))
                    mIoU5s.append(mIoU)
                    ACC5s.append(ACC)
            f.writelines('\n' + 'Total' + ': [mIoU]: ' + str(np.mean(np.array(mIoU5s))) + ' [ACC]: ' + str(np.mean(np.array(ACC5s))))
        if 'f' in mode:
            i = 1
            f.writelines('\n' + 'Numf:')
            mIoU, ACC, mIoU_epoch, loss, Loss_epoch = read_log(dataf_dir + '/f/log_fine.txt')
            f.writelines('\n' + str(i) + ': [mIoU]: ' + str(mIoU) + ' [ACC]: ' + str(ACC) + '[mIoU_epoch]: ' + str(mIoU_epoch) +
                         '[loss]: ' + str(loss) + '[Loss_epoch]: ' + str(Loss_epoch))
            f.writelines('\n' + 'Total' + ': [mIoU]: ' + str(mIoU) + ' [ACC]: ' + str(ACC))

# It is used for analyzing the performance of the model from log file.
def analyse_sup_log(tt):
    def read_log(file):
        mIoU = -1
        loss = 1e6
        with open(file, "r") as f:
            lines = f.readlines()
            for line in lines:
                if 'Epoch: [' in line:
                    epoch = int(line.split('[')[1].split('/')[0])
                    loss_now = float(line.split('Loss ')[1].split(' (')[0])
                    if loss_now < loss:
                        loss = loss_now
                        Loss_epoch = epoch
                elif 'mIoU:' in line:
                    mIoU_now = float(line.split(':')[1])
                    if mIoU < mIoU_now:
                        mIoU = mIoU_now
                        mIoU_epoch = epoch
        return mIoU, mIoU_epoch, loss, Loss_epoch

    root_dir = tt
    data1_dir = tt + '/_1'
    data5_dir = tt + '/_5'
    dataf_dir = tt + '/_f'
    mIoU1s = []
    mIoU5s = []
    with open(root_dir + '/' + 'eval.txt', 'w') as f:
        f.writelines('\n' + 'Num1:')
        for i in range(5):
            mIoU, mIoU_epoch, loss, Loss_epoch = read_log(data1_dir + '/1_' + str(i) + '/log_sup.txt')
            f.writelines('\n' + str(i) + ': [mIoU]: ' + str(mIoU) + '[mIoU_epoch]: ' + str(mIoU_epoch) +
                         '[loss]: ' + str(loss) + '[Loss_epoch]: ' + str(Loss_epoch))
            mIoU1s.append(mIoU)
        f.writelines('\n' + 'Total' + ': [mIoU]: ' + str(np.mean(np.array(mIoU1s))))
        f.writelines('\n' + 'Num5:')
        for i in range(5):
            mIoU, mIoU_epoch, loss, Loss_epoch = read_log(data5_dir + '/5_' + str(i) + '/log_sup.txt')
            f.writelines('\n' + str(i) + ': [mIoU]: ' + str(mIoU) + '[mIoU_epoch]: ' + str(mIoU_epoch) +
                         '[loss]: ' + str(loss) + '[Loss_epoch]: ' + str(Loss_epoch))
            mIoU5s.append(mIoU)
        f.writelines('\n' + 'Total' + ': [mIoU]: ' + str(np.mean(np.array(mIoU5s))))
        f.writelines('\n' + 'Numf:')
        mIoU, mIoU_epoch, loss, Loss_epoch = read_log(dataf_dir + '/f/log_sup.txt')
        f.writelines('\n' + str(i) + ': [mIoU]: ' + str(mIoU) + '[mIoU_epoch]: ' + str(mIoU_epoch) +
                     '[loss]: ' + str(loss) + '[Loss_epoch]: ' + str(Loss_epoch))
        f.writelines('\n' + 'Total' + ': [mIoU]: ' + str(mIoU))

# This function is actually used for our experiments.
# In fact, training can also be performed through train.py
def self2fine(tt, method='fine', env='0', RepeatID='01234', mode='15f',
              global_weight=1, dense_weight=1, multihead=4, Patch_sup=True,  # ablation
              patch_size_decay=0.5, patch_lab_decay=0.2, lab_size_decay=0.2,  # decay
              queue_size=504, top_k=4,  # moco top_k=30
              moco_denseloss_ratio=0.7, self_scale_weight=[1, 1, 1, 1], HeadNrom=['LN',''],
              queue_momentum=0.99, temperature=0.5, selfmode='moco', simclrinner = True, data_dir = ''):
    # method: 'self': pre-training; 'fine': fine-tuning; 'sup': without pre-training
    # RepeatID: Repeat the experiment up to 5 times;
    # mode: '15f' means run the method under 1,5,all annotations in turn;
    # These experiments were performed independently and did not affect each other.
    # The definitions of more args are explained in "runner_preheat" function.
    # Repeat the experiment 5 times using one annotation in training set
    for i in range(5):
        if ('1' in mode) and (str(i) in RepeatID):
            runner_preheat(tt=tt, method=method, env=env, sup_num='1_' + str(i), global_weight=global_weight,
                           dense_weight=dense_weight, multihead=multihead, Patch_sup=Patch_sup,
                           patch_size_decay=patch_size_decay, patch_lab_decay=patch_lab_decay,
                           lab_size_decay=lab_size_decay, self_scale_weight=self_scale_weight, HeadNrom = HeadNrom,
                           queue_size=queue_size, top_k=top_k, moco_denseloss_ratio=moco_denseloss_ratio,
                           queue_momentum=queue_momentum, temperature=temperature, selfmode=selfmode, simclrinner = simclrinner, data_dir = data_dir)
    # Repeat the experiment 5 times using 5 annotations in training set
    for i in range(5):
        if ('5' in mode) and (str(i) in RepeatID):
            runner_preheat(tt=tt, method=method, env=env, sup_num='5_' + str(i), global_weight=global_weight,
                           dense_weight=dense_weight, multihead=multihead, Patch_sup=Patch_sup,
                           patch_size_decay=patch_size_decay, patch_lab_decay=patch_lab_decay,
                           lab_size_decay=lab_size_decay, self_scale_weight=self_scale_weight, HeadNrom = HeadNrom,
                           queue_size=queue_size, top_k=top_k, moco_denseloss_ratio=moco_denseloss_ratio,
                           queue_momentum=queue_momentum, temperature=temperature, selfmode=selfmode, simclrinner = simclrinner, data_dir = data_dir)
    # Run the experiment using all annotations in training set
    if 'f' in mode:
        runner_preheat(tt=tt, method=method, env=env, sup_num='f', global_weight=global_weight,
                       dense_weight=dense_weight, multihead=multihead, Patch_sup=Patch_sup,
                       patch_size_decay=patch_size_decay, patch_lab_decay=patch_lab_decay,
                       lab_size_decay=lab_size_decay, self_scale_weight=self_scale_weight, HeadNrom = HeadNrom,
                       queue_size=queue_size, top_k=top_k, moco_denseloss_ratio=moco_denseloss_ratio,
                       queue_momentum=queue_momentum, temperature=temperature, selfmode=selfmode, simclrinner =simclrinner, data_dir = data_dir)
    # Analyse Finetuning log
    if method == 'fine':
        analyse_log(tt=tt, RepeatID=RepeatID, mode=mode)
    # Analyse Supervised log
    elif method == 'sup':
        root_dir = 'sup_UNet_' + tt
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        analyse_sup_log(tt=root_dir)

if __name__ == '__main__':
    self2fine(tt='PSCL-MoCo', method='self', env='0', RepeatID='01234', mode='1', selfmode='moco', temperature=0.5, moco_denseloss_ratio=0.7, data_dir = '')
    self2fine(tt='PSCL-MoCo', method='fine', env='0', RepeatID='01234', mode='1', selfmode='moco', temperature=0.5, moco_denseloss_ratio=0.7, data_dir = '')