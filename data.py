"""
Data Loading and Augmentation for PSCL

This file contains:
1. Custom dataset classes for supervised and self-supervised learning
2. Data augmentation pipelines
3. Utility functions for one-hot encoding and image preprocessing
"""

import torch
from torchvision import transforms
from torch.utils import data
import os
from os.path import join
from PIL import Image
import numpy as np
import cv2
import random

# Convert labels to one-hot form
def one_hot(labels, num_classes=-1):
    """Convert label map to one-hot encoded tensor"""
    if num_classes == -1:
        num_classes = int(labels.max()) + 1
    one_hot_tensor = torch.zeros(labels.size() + (num_classes,), dtype=torch.int64)
    one_hot_tensor.scatter_(-1, labels.unsqueeze(-1).to(torch.int64), 1)
    return one_hot_tensor

# Convert labels to one-hot form for segmentation networks
def get_one_hot(labels, num_classes=-1):
    """One-hot encoding for segmentation networks

    Args:
        labels: Input label map
        num_classes: Number of classes

    Returns:
        One-hot encoded tensor with shape (*labels.shape, num_classes)
    """
    labels = torch.as_tensor(labels)
    ones = one_hot(labels, num_classes)
    return ones.view(*labels.size(), num_classes)

# Preprocess image for CV2 compatibility
# In fact, the image here is a grayscale image, so this function is not very necessary
def prepare_image_cv2(im):
    """Preprocess image for CV2 compatibility
    1. Subtract mean values
    2. Transpose dimensions to (C x H x W)
    """
    im -= np.array((104.00698793,116.66876762,122.67891434))
    im = np.transpose(im, (2, 0, 1)) # (H x W x C) to (C x H x W)
    return im

# Supervised Dataset Loader for Aluminum Segmentation
# It is used for fineturning and supervision.
class Data_preheat(data.Dataset):
    """Supervised Dataset Loader for Aluminum Segmentation
    Features:
    - Train/test split handling
    - Different augmentation for train vs test
    - One-hot encoding of labels
    - Random rotations and flips
    """
    def __init__(self, root='../dataset/cast/split_images', split = 'train'):  # fuz: use fuz, fuz add: use fuz label
        self.split = split
        if split == 'train':
            jitter_d = 0.1
            jitter_p = 0.8
            color_jitter = transforms.ColorJitter(0.8 * jitter_d, 0.8 * jitter_d, 0.8 * jitter_d, 0.2 * jitter_d)
            rnd_color_jitter = transforms.RandomApply([color_jitter], p=jitter_p)
            transforms_sup = transforms.Compose([
                transforms.ToPILImage(),
                rnd_color_jitter,
                transforms.ToTensor(),
                transforms.Normalize((0.49139968, 0.48215841, 0.44653091),
                                     (0.24703223, 0.24348513, 0.26158784))])
        else:
            transforms_sup = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize((0.49139968, 0.48215841, 0.44653091),
                                 (0.24703223, 0.24348513, 0.26158784))])
        self.transform = transforms_sup
        self.root = root + '/img'
        self.filelist = os.listdir(self.root)

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        if self.split == 'train':
            img_file = self.filelist[index]
            gt = cv2.imread(join(self.root.replace('/img', '/gt'), img_file.replace('.jpg','.png')), 0)
            #gt[gt == 255] = 1
            #gt = torch.tensor(gt)#.unsqueeze(0)
            gt_new = get_one_hot(gt, num_classes=4)
            gt_new = gt_new.permute(2, 0, 1).float()
            #gtt0 = gt_new.cpu().numpy()[0]
            #gtt1 = gt_new.cpu().numpy()[1]
            #gtt2 = gt_new.cpu().numpy()[2]
            #gtt3 = gt_new.cpu().numpy()[3]
            image = Image.open(join(self.root, img_file)).convert('RGB')
            image = np.array(image)
            img1 = self.transform(image)

            rot_k = random.randint(1, 4)
            filp = random.randint(1, 2)

            if filp == 1:
                img1 = torch.flip(img1, dims=[1])
                gt_new = torch.flip(gt_new, dims=[1])
            img1 = torch.rot90(img1, k=rot_k, dims=[1, 2])
            gt_new = torch.rot90(gt_new, k=rot_k, dims=[1, 2])

            return img1, gt_new
        else:
            img_file = self.filelist[index]
            gt = cv2.imread(join(self.root.replace('/img', '/gt'), img_file.replace('.jpg','.png')), 0)
            gt_new = get_one_hot(gt, num_classes=4)
            gt_new = gt_new.permute(2, 0, 1).float()
            image = Image.open(join(self.root, img_file)).convert('RGB')
            image = np.array(image)
            img1 = self.transform(image)
            return img1, gt_new, img_file.split('/')[-1]

# Dataset Loader used in pre-training Learning
# During the training process, "MoCoData_preheat" here and "MoCoData_preheat_sup" below
# are used at the same time.
class MoCoData_preheat(data.Dataset):
    """Dataset Loader for Self-Supervised Learning

    Features:
    - Two augmented views of each image
    - Random crops and color jitter
    - Consistent ID tracking for contrastive learning
    """

    def __init__(self, root='dataset/cast/split_images', jitter_d = 0.2, jitter_p = 0.8, grey_p = 0.2, random_c = 0.1,
                 crop_dim = 376, two_crop = True):  # fuz: use fuz, fuz add: use fuz label
        """Initialize MoCo dataset
        Args:
            root: Dataset root path
            jitter_d: Color jitter intensity
            jitter_p: Color jitter probability
            grey_p: Grayscale probability
            random_c: Random crop ratio
            crop_dim: Crop dimension
            two_crop: Whether to generate two crops
        """
        color_jitter = transforms.ColorJitter(0.8 * jitter_d, 0.8 * jitter_d, 0.8 * jitter_d, 0.2 * jitter_d)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=jitter_p)
        rnd_grey = transforms.RandomGrayscale(p=grey_p)
        self.random_c = random_c
        transforms_moco = transforms.Compose([
            transforms.ToPILImage(),
            rnd_color_jitter,
            #rnd_grey,
            #transforms.RandomResizedCrop((crop_dim, crop_dim)),
            #transforms.RandomRotation(180),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.49139968, 0.48215841, 0.44653091),
                                 (0.24703223, 0.24348513, 0.26158784))])
        self.transform = transforms_moco
        self.two_crop = two_crop
        self.root = root + '/img'
        self.filelist = os.listdir(self.root)

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        """Get two augmented views for contrastive learning
        Returns:
            Concatenated augmented views
            Image ID for memory bank
            Augmentation parameters (rotation, flip, crop coords)
        """
        img_file = self.filelist[index]
        name = img_file.split('_')
        id = name[0] + '_' + name[1]
        image = Image.open(join(self.root, img_file)).convert('RGB')
        image = np.array(image)
        #test1 = np.zeros([376,376,3]).astype(np.uint8)
        #test1[:200,:200] = 255
        #test1[:,:200] = 255
        #image = test1
        img1 = self.transform(image)
        img2 = self.transform(image)
        rot_k = random.randint(1, 4)
        filp = random.randint(1, 2)
        r1 = 1 - random.uniform(0, self.random_c)
        r2 = 1 - random.uniform(0, self.random_c)
        r3 = random.uniform(0, 1 - r1)
        r4 = random.uniform(0, 1 - r2)
        if filp == 1:
            img2 = torch.flip(img2, dims=[1])
        img2 = torch.rot90(img2, k=rot_k, dims=[1, 2])
        _, h, w = img2.shape
        img2 = img2[:,int(r3*h):int(r3*h+r1*h),int(r4*w):int(r4*w+r2*w)]
        img2 = transforms.Resize([h, w])(img2)
        img12 = torch.cat([img1, img2], dim=0)
        return img12, id, rot_k, filp, r1, r2, r3, r4

# Dataset Loader used in pre-training Learning.
# It is used to obtain supervised samples and to implement sampling of unlabeled patches.
# Patch sampling is the main difference between us and conventional pre-training methods.
class MoCoData_preheat_sup(data.Dataset):
    """Initialize dataset
    """
    def __init__(self, root='dataset/cast/split_images', jitter_d = 0.2, jitter_p = 0.8, grey_p = 0.2, random_c = 0.1,
                 crop_dim = 376, two_crop = True):  # fuz: use fuz, fuz add: use fuz label
        color_jitter = transforms.ColorJitter(0.8 * jitter_d, 0.8 * jitter_d, 0.8 * jitter_d, 0.2 * jitter_d)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=jitter_p)
        rnd_grey = transforms.RandomGrayscale(p=grey_p)
        self.random_c = random_c
        transforms_moco = transforms.Compose([
            transforms.ToPILImage(),
            rnd_color_jitter,
            transforms.ToTensor(),
            transforms.Normalize((0.49139968, 0.48215841, 0.44653091),
                                 (0.24703223, 0.24348513, 0.26158784))])
        self.transform = transforms_moco
        self.two_crop = two_crop
        self.root = root + '/img'
        self.filelist = os.listdir(self.root)

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        """Get two augmented views for contrastive learning
        Returns:
            Concatenated augmented views (with annotations)
            Image ID for memory bank
            Augmentation parameters (rotation, flip, crop coords)
        """
        img_file = self.filelist[index]
        name = img_file.split('_')
        gt = cv2.imread(join(self.root.replace('/img', '/gt'), img_file.replace('.jpg','.png')), 0)
        gt_new = get_one_hot(gt, num_classes=4) * 255
        gt_new = gt_new.permute(2, 0, 1).float()
        id = name[0] + '_' + name[1]
        image = Image.open(join(self.root, img_file)).convert('RGB')
        image = np.array(image)
        img1 = self.transform(image)
        img2 = self.transform(image)

        rot_k = random.randint(1, 4)
        filp = random.randint(1, 2)
        r1 = 1 - random.uniform(0, self.random_c)
        r2 = 1 - random.uniform(0, self.random_c)
        r3 = random.uniform(0, 1 - r1)
        r4 = random.uniform(0, 1 - r2)

        if filp == 1:
            img2 = torch.flip(img2, dims = [1])
            gt_new = torch.flip(gt_new, dims = [1])
        img2 = torch.rot90(img2, k=rot_k, dims=[1, 2])
        gt_new = torch.rot90(gt_new, k=rot_k, dims=[1, 2])
        _, h, w = img2.shape
        img2 = img2[:,int(r3*h):int(r3*h+r1*h),int(r4*w):int(r4*w+r2*w)]
        img2 = transforms.Resize([h, w])(img2)
        gt_new = gt_new[:,int(r3*h):int(r3*h+r1*h),int(r4*w):int(r4*w+r2*w)]
        gt_new = transforms.Resize([h, w])(gt_new)

        img12G = torch.cat([img1, img2, gt_new], dim=0)
        return img12G, id, rot_k, filp, r1, r2, r3, r4
