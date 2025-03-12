import config

name = 'PSCL'
data_dir = 'dataset/aluminum/'
stages = 'self-fine'
# stage
# self: pre-training;
# fine: fine-tuning
# sup: without pre-training
self_mode = 'moco'
# self_mode: moco; simclr
data_size = '1'
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
    if multi_round:
        RepeatID = '01234'
    else:
        RepeatID = '0'

    file_name = name + '_' + self_mode
    stage = stages.split('-')
    for stage_ in stage:
        config.self2fine(tt=name, method=stage_, env=GPU_env, RepeatID=RepeatID, mode=data_size, selfmode=self_mode,
                         temperature=temperature, moco_denseloss_ratio=global_local_ratio, data_dir = data_dir)