"""
Utility Functions for PSCL Training
Includes:
1. Logging utilities
2. Training helpers (checkpoint, average meter)
3. Model initialization functions
4. Optimizer configuration
"""

import os, sys
import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio

# class 'logger' is used for recording metrics
class Logger(object):
    """Dual logger that writes to both console and file

    Usage:
    with Logger('log.txt') as logger:
        logger.write("Training started\n")
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        #self.console.close()
        self.console = None
        if self.file is not None:
            self.file.close()

# class 'Averagvalue' is used for calculating the average value
class Averagvalue(object):
    """Computes and stores the average and current value
    Useful for tracking metrics like loss etc.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# this funtion is used for saving trained model.
def save_checkpoint(state, filename='checkpoint.pth'):
    """Save training checkpoint
    Args:
        state: Dict containing model state, optimizer etc.
        filename: Output filename
    """
    torch.save(state, filename)

# load pretrained model
def load_pretrained(model, fname, optimizer=None):
    """Load pretrained model from checkpoint

    Args:
        model: Model to load weights into
        fname: Checkpoint filename
        optimizer: Optional optimizer to also load

    Returns:
        Loaded model and optionally optimizer
    """
    if os.path.isfile(fname):
        print("=> loading checkpoint '{}'".format(fname))
        checkpoint = torch.load(fname)
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            return model, optimizer, checkpoint['epoch']
        else:
            return model, checkpoint['epoch']
    else:
        print("=> no checkpoint found at '{}'".format(fname))

# load vgg16 parameters;
# In fact, this is deprecated for the unet model
def load_vgg16pretrain(model, vggmodel='vgg16convs.mat'):
    """Load VGG16 pretrained weights from .mat file
    For compatibility with legacy models
    """
    vgg16 = sio.loadmat(vggmodel)
    torch_params = model.state_dict()
    for k in vgg16.keys():
        name_par = k.split('-')
        size = len(name_par)
        if size == 2:
            name_space = name_par[0] + '.' + name_par[1]
            data = np.squeeze(vgg16[k])
            torch_params[name_space] = torch.from_numpy(data)
    model.load_state_dict(torch_params)

# load vgg16 parameters;
# In fact, this is deprecated for the unet model
def load_vgg16pretrain_half(model, vggmodel='vgg16convs.mat'):
    vgg16 = sio.loadmat(vggmodel)
    torch_params = model.state_dict()
    for k in vgg16.keys():
        name_par = k.split('-')
        size = len(name_par)
        if size == 2:
            name_space = name_par[0] + '.' + name_par[1]
            data = np.squeeze(vgg16[k])
            shape = data.shape
            index = int(shape[0] / 2)
            if len(shape) == 1:
                data = data[:index]
            else:
                data = data[:index, :, :, :]
            torch_params[name_space] = torch.from_numpy(data)
    model.load_state_dict(torch_params)

# Parameter initialization
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        # xavier(m.weight.data)
        m.weight.data.normal_(0, 0.01)
        if m.weight.data.shape == torch.Size([1, 4, 1, 1]):
            torch.nn.init.constant_(m.weight, 0.25)
        if m.bias is not None:
            m.bias.data.zero_()