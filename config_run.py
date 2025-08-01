"""
Training Pipeline for PSCL
This file contains training loops for supervised, self-supervised, and fine-tuning phases
"""

import numpy as np
import os
from data import Data_preheat, MoCoData_preheat, MoCoData_preheat_sup
from torch.utils.data import DataLoader
from utils import Averagvalue, Logger
import shutil
import sys
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

# Supervised training loop
def Supervised(config_lmc):
    # Training function definition
    def train(model, train_loader, optimizer, epoch, max_epoch):
        """Training process
        Args:
            model: Model to train
            train_loader: Training data loader
            optimizer: Optimizer
            epoch: Current epoch number
            max_epoch: Maximum epoch number
        """

        def dice(predict, target):
            # Calculate Dice coefficient loss
            predict = predict.contiguous().view(predict.shape[0], -1)
            target = target.contiguous().view(target.shape[0], -1)
            num = torch.sum(torch.mul(predict, target), dim=1) + 1e-4 # Intersection
            den = torch.sum(predict + target, dim=1) + 1e-4 # Union
            loss = 1 - 2 * num / den # Dice coefficient
            return loss.mean()

        def multidiceLoss(predict, target):
            """Multi-class Dice loss calculation
            Args:
                predict: Prediction tensor [B, C, H, W]
                target: Target tensor [B, C, H, W]
            Returns:
                List of Dice losses per class
            """
            b, c, h, w = predict.shape
            predict = predict.view(b, c, h * w)
            target = target.view(b, c, h * w)
            assert predict.shape == target.shape, 'predict & target shape do not match'
            total_loss = []
            # Softmax normalization
            predict = torch.nn.functional.softmax(predict, dim=1)
            # Calculate Dice loss for each class
            for i in range(target.shape[1]):
                dice_loss = dice(predict[:, i], target[:, i])
                dice_loss *= config_lmc.dice_weight[i] # Apply class weight
                total_loss.append(dice_loss)
            return total_loss
        # Initialize CrossEntropy loss with class weights
        criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(config_lmc.weight).cuda())

        model.train()
        # Initialize average meters for tracking metrics
        losses_bce = Averagvalue() # BCE loss tracker
        losses_dice = Averagvalue() # Combined Dice loss tracker
        losses_dice1 = Averagvalue() # Class 1 Dice tracker
        losses_dice2 = Averagvalue() # Class 2 Dice tracker
        losses_dice3 = Averagvalue() # Class 3 Dice tracker
        losses = Averagvalue() # Total loss tracker
        for i, (image, label) in enumerate(train_loader):
            optimizer.zero_grad()
            image, label = image.cuda(), label.cuda()
            # Forward pass
            pred = model(image)
            # Calculate losses
            loss_bce = criterion(pred, label) # Cross-entropy loss
            loss_dices = multidiceLoss(pred, label) # Dice losses per class
            loss_diceSum = (loss_dices[1] + loss_dices[2] + loss_dices[3]) / 3
            # Combined loss
            loss = loss_bce * (1 - config_lmc.dice_bce_ratio) + loss_diceSum * config_lmc.dice_bce_ratio
            # Backward pass
            loss.backward()
            optimizer.step()
            # Update metrics
            losses_bce.update(loss_bce.item(), image.size(0))
            losses_dice1.update(loss_dices[1].item(), image.size(0))
            losses_dice2.update(loss_dices[2].item(), image.size(0))
            losses_dice3.update(loss_dices[3].item(), image.size(0))
            losses_dice.update(loss_diceSum.item(), image.size(0))
            losses.update(loss.item(), image.size(0))
            # Logging
            if config_lmc.verb: # Verbose mode
                if i == len(train_loader) - 1:
                    info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, max_epoch, i, len(train_loader)) + \
                           'Loss {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses) + \
                           'BCE {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses_bce) + \
                           'DICE {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses_dice) + \
                           '[DICE1 (avg:{loss.avg:f}) '.format(loss=losses_dice1) + \
                           'DICE2 (avg:{loss.avg:f}) '.format(loss=losses_dice2) + \
                           'DICE3 (avg:{loss.avg:f})]'.format(loss=losses_dice3)
                    print(info)
            else:
                if i % config_lmc.print_freq == 0 or i == len(train_loader) - 1:
                    info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, max_epoch, i, len(train_loader)) + \
                           'Loss {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses) + \
                           'BCE {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses_bce) + \
                           'DICE {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses_dice) + \
                           '[DICE1 (avg:{loss.avg:f}) '.format(loss=losses_dice1) + \
                           'DICE2 (avg:{loss.avg:f}) '.format(loss=losses_dice2) + \
                           'DICE3 (avg:{loss.avg:f})]'.format(loss=losses_dice3)
                    print(info)

    # Testing function definition
    def test(model, test_loader, epoch):
        """Evaluation process
        Args:
            model: Trained model
            test_loader: Test data loader
            epoch: Current epoch number
        Returns:
            Mean accuracy
        """
        # Create evaluation directory
        eval_dir = config_lmc.tmp + '/epoch_' + str(epoch)
        if not os.path.exists(eval_dir):
            os.mkdir(eval_dir)
        model.eval()
        # Initialize metric trackers
        accs = [] # Accuracy tracker
        inte_c1s = []  # Class 1 intersection tracker
        inte_c2s = []  # Class 2 intersection tracker
        inte_c3s = []  # Class 3 intersection tracker
        oute_c1s = []  # Class 1 union tracker
        oute_c2s = []  # Class 2 union tracker
        oute_c3s = []  # Class 3 union tracker

        for i, (image, label, filename) in enumerate(test_loader):
            image, label = image.cuda(), label.cuda()
            # Forward pass
            pred = model(image)
            # Convert to probabilities and numpy
            pred_prob = torch.nn.functional.softmax(pred, dim=1)
            pred_np = pred_prob.detach().cpu().numpy()[0]
            pred_cls = np.argmax(pred_np, axis=0) # Predicted class map
            label_np = label.detach().cpu().numpy()[0]
            label_cls = np.argmax(label_np, axis=0) # Ground truth class map
            # Save prediction with color mapping
            filename_ = filename[0]
            pred_eval = pred_cls.copy()
            for color in range(4):
                pred_eval[pred_eval == color] = config_lmc.mapping[color]
            cv2.imwrite(eval_dir + '/' + filename_, pred_eval)
            # Calculate accuracy
            acc_ = 1 * np.equal(pred_cls, label_cls)
            acc = np.mean(acc_)
            # Calculate IoU metrics per class
            pred_c1 = np.zeros([pred_np.shape[1], pred_np.shape[2]])
            pred_c2 = np.zeros([pred_np.shape[1], pred_np.shape[2]])
            pred_c3 = np.zeros([pred_np.shape[1], pred_np.shape[2]])
            pred_c1[pred_cls == 1] = 1 # Class 1 predictions
            pred_c2[pred_cls == 2] = 1 # Class 2 predictions
            pred_c3[pred_cls == 3] = 1 # Class 3 predictions
            # Intersection and union calculations
            inte_c1 = 2 * np.logical_and(pred_c1, label_np[1]) # Class 1 intersection
            inte_c2 = 2 * np.logical_and(pred_c2, label_np[2]) # Class 2 intersection
            inte_c3 = 2 * np.logical_and(pred_c3, label_np[3]) # Class 3 intersection
            oute_c1 = pred_c1.sum() + label_np[1].sum() # Class 1 union
            oute_c2 = pred_c2.sum() + label_np[2].sum() # Class 2 union
            oute_c3 = pred_c3.sum() + label_np[3].sum() # Class 3 union
            # Record metrics
            accs.append(acc)
            inte_c1s.append(inte_c1)
            inte_c2s.append(inte_c2)
            inte_c3s.append(inte_c3)
            oute_c1s.append(oute_c1)
            oute_c2s.append(oute_c2)
            oute_c3s.append(oute_c3)
        # Calculate final metrics
        accs = np.array(accs)
        iou1 = np.array(inte_c1s).sum() / np.array(oute_c1s).sum()
        iou2 = np.array(inte_c2s).sum() / np.array(oute_c2s).sum()
        iou3 = np.array(inte_c3s).sum() / np.array(oute_c3s).sum()
        # We need to point out that although the output here is iou,
        # it is actually calculated according to dice, as shown in the paper
        print('test')
        print('ACC:' + str(np.mean(accs)))
        print('mIoU:' + str((iou1 + iou2 + iou3) / 3))
        print('IoU1:' + str(iou1))
        print('IoU2:' + str(iou2))
        print('IoU3:' + str(iou3))

        return np.mean(accs)
    # Model saving function
    def save(model, epoch, acc):
        """Save model checkpoint
        Args:
            model: Model to save
            epoch: Current epoch number
            acc: Current accuracy
        """
        state = {
            # 'args': args,
            'super': model.state_dict(),
            'epoch': epoch,
            'acc': acc
        }
        torch.save(state, config_lmc.tmp + '/' + 'sup.pt')
        print('save file with acc: ' + str(acc))
    # Main execution block
    # Create output directory
    if not os.path.exists(config_lmc.tmp + '_' + config_lmc.sup_num.split('_')[0]):
        os.makedirs(config_lmc.tmp + '_' + config_lmc.sup_num.split('_')[0])
    config_lmc.tmp = config_lmc.tmp + '_' + config_lmc.sup_num.split('_')[0] + '/' + config_lmc.sup_num
    if not os.path.exists(config_lmc.tmp):
        os.makedirs(config_lmc.tmp)
    # Set up logging
    shutil.copy(os.path.realpath(__file__), config_lmc.tmp + '/config_sup')
    log = Logger(os.path.join(config_lmc.tmp, 'log_sup.txt'))
    raw_std = sys.stdout
    sys.stdout = log
    # Initialize datasets and loaders
    train_dataset = Data_preheat(root=config_lmc.dataset_sup, split='train')
    test_dataset = Data_preheat(root=config_lmc.dataset_test, split='test')
    train_loader = DataLoader(train_dataset, batch_size=config_lmc.batch_size, drop_last=True, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, drop_last=True, shuffle=False)
    # Initialize model and optimizer
    model = config_lmc.backbone(config_lmc.drop, config_lmc.channel,
                                IncNorm=config_lmc.IncNorm, DownNorm=config_lmc.DownNorm, UpNorm=config_lmc.UpNorm)
    model = model.cuda()
    acc_best = 0 # Track best accuracy
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config_lmc.super_lr)
    # Training loop
    for epoch in range(config_lmc.supv_max_epoch):
        train(model, train_loader, optimizer, epoch, config_lmc.supv_max_epoch)
        # Periodic evaluation
        if (epoch + 1) % config_lmc.test_freq == 0:
            acc = test(model, test_loader, epoch)
            if acc > acc_best: # Save best model
                acc_best = acc
                save(model, epoch, acc)
    # Clean up
    log.close()
    sys.stdout = raw_std
    return

# Self-supervised training using contrastive learning
def SelfSupervised(config_lmc):
    # Self-supervised training pipeline using contrastive learning (PSCL-MoCo/SimCLR)
    # Validate dense loss ratio parameter
    if config_lmc.moco_denseloss_ratio > 1 or config_lmc.moco_denseloss_ratio < 0:
        print('error')
        return
    # 1. Setup directories and logging
    # Create output directory with numbered suffix
    if not os.path.exists(config_lmc.tmp + '_Num' + config_lmc.sup_num.split('_')[0]):
        os.makedirs(config_lmc.tmp + '_Num' + config_lmc.sup_num.split('_')[0])
    # Set up temporary directory path
    config_lmc.tmp = config_lmc.tmp + '_Num' + config_lmc.sup_num.split('_')[0] + '/' + config_lmc.sup_num
    if not os.path.exists(config_lmc.tmp):
        os.makedirs(config_lmc.tmp)
    # Copy config and model files for reproducibility
    shutil.copy(os.path.realpath(__file__), config_lmc.tmp + '/config_self')
    shutil.copy('model.py', config_lmc.tmp + '/model_self')
    # Initialize logging
    log = Logger(os.path.join(config_lmc.tmp, 'log_self.txt'))
    raw_std = sys.stdout
    sys.stdout = log
    # 2. Data preparation
    # Create datasets with augmentations
    train_dataset = MoCoData_preheat(root=config_lmc.dataset_train, jitter_d=config_lmc.jitter_d,
                                     random_c=config_lmc.random_c)
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config_lmc.self_batch_size, drop_last=True, shuffle=True)
    # Supervised dataset for contrastive learning
    sup_dataset = MoCoData_preheat_sup(root=config_lmc.dataset_sup, jitter_d=config_lmc.jitter_d,
                                       random_c=config_lmc.random_c)
    sup_loader = DataLoader(sup_dataset, batch_size=1, drop_last=True, shuffle=True)
    # 3. Model initialization
    MoComodel = config_lmc.moco_mode(
        backbone=config_lmc.backbone,              # UNet is used here
        queue_size=config_lmc.queue_size,          # Size of memory bank for MoCo
        momentum=config_lmc.queue_momentum,        # Momentum for key encoder updates
        temperature=config_lmc.temperature,        # Temperature for contrastive loss
        lab_size_decay=config_lmc.lab_size_decay,  # decay factor
        patch_num=config_lmc.patch_num,            # Number of patches
        drop=config_lmc.drop,                      # Dropout rate
        channel=config_lmc.channel,                # Channel dimensions
        self_scale_weight=config_lmc.self_scale_weight,  # Weight for multi-scale features
        sample_ratio=config_lmc.sample_ratio,      # Ratio for patch sampling
        confea_num=config_lmc.confea_num,          # Feature dimensions in proj-head function
        hidfea_num=config_lmc.hidfea_num,          # Hidden feature dimensions in proj-head function
        top_k=config_lmc.top_k,                    # Top-k patches to select
        patch_lab_decay=config_lmc.patch_lab_decay,  # Patch label decay
        patch_size_decay=config_lmc.patch_size_decay,  # Patch size decay
        multihead=config_lmc.multihead,            # Number of attention heads
        IncNorm=config_lmc.IncNorm,                # Normalization for input
        DownNorm=config_lmc.DownNorm,              # Normalization for downsampling
        UpNorm=config_lmc.UpNorm,                  # Normalization for upsampling
        HeadNrom=config_lmc.HeadNrom,              # Normalization for head
        Patch_sup=config_lmc.Patch_sup,            # Whether to use patch supervision
        selfmode=config_lmc.selfmode               # Mode: 'moco' or 'simclr'
    )
    MoComodel = MoComodel.cuda()
    # 4. Optimizer setup
    if config_lmc.opti == 'adam':
        optimiser = torch.optim.Adam(params=MoComodel.parameters(), lr=config_lmc.self_lr)
    elif config_lmc.opti == 'adamW':
        optimiser = torch.optim.AdamW(params=MoComodel.parameters(), lr=config_lmc.self_lr,
                                      weight_decay=config_lmc.adamw_decay)
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimiser, [int(config_lmc.self_max_epoch * 0.5),
                                                                 int(config_lmc.self_max_epoch * 0.75)], gamma=0.1)
    # Loss function
    criterion = nn.CrossEntropyLoss()
    # 5. Contrastive loss functions
    def nt_xent_loss_d(z1, z2, temperature=0.5, deep=0, bat=1):
        """Modified NT-Xent loss for dense (patch-level) contrastive learning
        Args:
            z1, z2: Positive pair feature embeddings
            temperature: Softmax temperature
            deep: Scale depth (0,1,2 for different scales)
            bat: Batch size multiplier
        """
        # Construct 4x4 block matrix (one block per class)
        muban_1 = np.ones(
            [config_lmc.top_k * config_lmc.sample_ratio ** deep, config_lmc.top_k * config_lmc.sample_ratio ** deep])
        muban_0 = np.zeros(
            [config_lmc.top_k * config_lmc.sample_ratio ** deep, config_lmc.top_k * config_lmc.sample_ratio ** deep])
        muban_p0 = np.zeros([4 * config_lmc.top_k * config_lmc.sample_ratio ** deep,
                             4 * config_lmc.top_k * config_lmc.sample_ratio ** deep])
        muban_h1 = np.hstack([muban_0, muban_1, muban_1, muban_1])
        muban_h2 = np.hstack([muban_1, muban_0, muban_1, muban_1])
        muban_h3 = np.hstack([muban_1, muban_1, muban_0, muban_1])
        muban_h4 = np.hstack([muban_1, muban_1, muban_1, muban_0])
        muban_t = np.vstack([muban_h1, muban_h2, muban_h3, muban_h4])
        # Tile for batch dimension
        muban_t1 = np.hstack([muban_t, muban_t])
        muban_t2 = np.hstack([muban_t, muban_t])
        muban = np.vstack([muban_t1, muban_t2])
        muban_new = np.tile(muban, (bat, bat))
        # Remove self-similarities if not using SimCLR-style inner loss
        if not config_lmc.simclrinner:
            lenth = 4 * config_lmc.top_k * config_lmc.sample_ratio ** deep
            for ii in range(bat*2):
                muban_new[int(ii*lenth):int((ii+1)*lenth), int(ii*lenth):int((ii+1)*lenth)] = 0

        # Normalize features
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        # Concatenate all features
        N, Z = z1.shape
        device = z1.device
        representations = torch.cat([z1, z2], dim=0)
        # Compute similarity matrix
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=-1)
        # Get positive pairs (diagonals)
        l_pos = torch.diag(similarity_matrix, N)
        r_pos = torch.diag(similarity_matrix, -N)
        positives = torch.cat([l_pos, r_pos]).view(2 * N, 1)
        diag = torch.eye(2 * N, dtype=torch.bool, device=device)
        diag[N:, :N] = diag[:N, N:] = diag[:N, :N]
        # Get negative pairs using our mask
        negatives = similarity_matrix[muban_new == 1].view(2 * N, -1)
        # Compute logits and loss
        logits = torch.cat([positives, negatives], dim=1)
        logits /= temperature
        labels = torch.zeros(2 * N, device=device, dtype=torch.int64)
        loss = F.cross_entropy(logits, labels, reduction='sum')
        return loss / (2 * N)

    def nt_xent_loss(z1, z2, temperature=0.5):
        """Standard NT-Xent loss for global contrastive learning"""
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        N, Z = z1.shape
        device = z1.device
        representations = torch.cat([z1, z2], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=-1)
        # Positive pairs are the diagonal elements
        l_pos = torch.diag(similarity_matrix, N)
        r_pos = torch.diag(similarity_matrix, -N)
        positives = torch.cat([l_pos, r_pos]).view(2 * N, 1)
        # Negative pairs are all other elements
        diag = torch.eye(2 * N, dtype=torch.bool, device=device)
        diag[N:, :N] = diag[:N, N:] = diag[:N, :N]
        negatives = similarity_matrix[~diag].view(2 * N, -1)

        logits = torch.cat([positives, negatives], dim=1)
        logits /= temperature
        labels = torch.zeros(2 * N, device=device, dtype=torch.int64)
        loss = F.cross_entropy(logits, labels, reduction='sum')
        return loss / (2 * N)
    # 6. Training loop
    for epoch in range(config_lmc.self_max_epoch + 1):
        # Train models
        MoComodel.train()
        # Initialize metric trackers
        losses = Averagvalue() # Total loss
        losses_c = Averagvalue() # Global contrastive loss
        losses_dense = Averagvalue() # Dense contrastive loss
        losses_dense1 = Averagvalue() # Scale 1 dense loss
        losses_dense2 = Averagvalue() # Scale 2 dense loss
        losses_dense3 = Averagvalue() # Scale 3 dense loss
        # Save checkpoint periodically
        state = {
            # 'args': args,
            'moco': MoComodel.state_dict(),
            'epoch': epoch,
        }

        if epoch % config_lmc.self_save_epoch == 0:
            torch.save(state, config_lmc.tmp + '/' + 'moco.pt'.replace('.pt', str(epoch) + '.pt'))

        for i, (inputs, ids, rot_k, filp, r1, r2, r3, r4) in enumerate(train_loader):
            inputs = inputs.cuda(non_blocking=True)
            # Forward pass
            optimiser.zero_grad()
            # Split into two views (augmented versions)
            x_i, x_j = torch.split(inputs, [3, 3], dim=1)
            # Get supervised batch
            inputs_sup, ids_sup, rot_k_sup, filp_sup, r1_sup, r2_sup, r3_sup, r4_sup = next(iter(sup_loader))
            inputs_sup = inputs_sup.cuda(non_blocking=True)
            x_i_sup, x_j_sup, y_i_sup = torch.split(inputs_sup, [3, 3, 4], dim=1)
            # Combine supervised and unsupervised batches
            x_i = torch.cat([x_i_sup, x_i], dim=0)
            x_j = torch.cat([x_j_sup, x_j], dim=0)
            ids = list(ids_sup) + list(ids)
            rot_k = torch.cat([rot_k_sup, rot_k], dim=0)
            filp = torch.cat([filp_sup, filp], dim=0)
            r1 = torch.cat([r1_sup, r1], dim=0)
            r2 = torch.cat([r2_sup, r2], dim=0)
            r3 = torch.cat([r3_sup, r3], dim=0)
            r4 = torch.cat([r4_sup, r4], dim=0)
            # Forward pass based on mode
            if config_lmc.selfmode == 'moco':
                # MoCo mode forward pass
                logit, label, logits_dense, labels_dense = MoComodel(x_i, x_j, y_i_sup, ids, rot_k, filp, r1, r2, r3,
                                                                     r4)
                # Calculate losses
                loss_c = criterion(logit, label)  # Global contrastive loss
                loss_dense = criterion(logits_dense, labels_dense) # Dense contrastive
                # Split dense loss by scale
                base_num1 = config_lmc.top_k * 4
                base_num2 = config_lmc.top_k * 4 * config_lmc.sample_ratio + base_num1
                base_num3 = config_lmc.top_k * 4 * config_lmc.sample_ratio ** 2 + base_num2
                loss_dense1 = criterion(logits_dense[0:base_num1], labels_dense[0:base_num1])
                loss_dense2 = criterion(logits_dense[base_num1:base_num2], labels_dense[base_num1:base_num2])
                loss_dense3 = criterion(logits_dense[base_num2:base_num3], labels_dense[base_num2:base_num3])
                # Combined loss
                loss = loss_c * config_lmc.global_weight * (1 - config_lmc.moco_denseloss_ratio) + \
                       loss_dense * config_lmc.moco_denseloss_ratio * config_lmc.dense_weight
                loss.backward()
                optimiser.step()
                # Update metrics
                losses.update(loss.item(), inputs.size(0))
                losses_c.update(loss_c.item(), inputs.size(0))
                losses_dense.update(loss_dense.item(), inputs.size(0))
                losses_dense1.update(loss_dense1.item(), inputs.size(0))
                losses_dense2.update(loss_dense2.item(), inputs.size(0))
                losses_dense3.update(loss_dense3.item(), inputs.size(0))
                if config_lmc.verb:
                    if i == len(train_loader) - 1:
                        info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, config_lmc.self_max_epoch, i,
                                                                   len(train_loader)) + \
                               'Loss {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses) + \
                               'LossC {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses_c) + \
                               'LossDense {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses_dense) + \
                               'Lr: ' + str(optimiser.state_dict()['param_groups'][0]['lr']) + \
                               'LossDense1 {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses_dense1) + \
                               'LossDense2 {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses_dense2) + \
                               'LossDense3 {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses_dense3)
                        print(info)
                else:
                    if i % config_lmc.print_freq == 0 or i == len(train_loader) - 1:
                        info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, config_lmc.self_max_epoch, i,
                                                                   len(train_loader)) + \
                               'Loss {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses) + \
                               'LossC {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses_c) + \
                               'LossDense {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses_dense)
                        print(info)
            elif config_lmc.selfmode == 'simclr':
                q, k, q_dense1, k_dense1, q_dense2, k_dense2, q_dense3, k_dense3 = MoComodel(x_i, x_j, y_i_sup, ids,
                                                                                             rot_k, filp, r1, r2, r3,
                                                                                             r4)
                # Calculate losses at different scales
                loss_dense1 = nt_xent_loss_d(q_dense1, k_dense1, config_lmc.temperature, deep=0, bat=q.shape[0])
                loss_dense2 = nt_xent_loss_d(q_dense2, k_dense2, config_lmc.temperature, deep=1, bat=q.shape[0])
                loss_dense3 = nt_xent_loss_d(q_dense3, k_dense3, config_lmc.temperature, deep=2, bat=q.shape[0])
                # Global contrastive loss
                loss_c = nt_xent_loss(q, k, config_lmc.temperature)
                # Weighted dense loss
                loss_dense = (loss_dense1 * config_lmc.self_scale_weight[0] +
                              loss_dense2 * config_lmc.self_scale_weight[1] +
                              loss_dense3 * config_lmc.self_scale_weight[2]) / 3
                # Combined loss
                loss = loss_c * config_lmc.global_weight * (1 - config_lmc.moco_denseloss_ratio) + \
                       loss_dense * config_lmc.moco_denseloss_ratio * config_lmc.dense_weight
                loss.backward()
                optimiser.step()
                # Update metrics
                losses.update(loss.item(), inputs.size(0))
                losses_c.update(loss_c.item(), inputs.size(0))
                losses_dense.update(loss_dense.item(), inputs.size(0))
                losses_dense1.update(loss_dense1.item(), inputs.size(0))
                losses_dense2.update(loss_dense2.item(), inputs.size(0))
                losses_dense3.update(loss_dense3.item(), inputs.size(0))
                # Logging
                if config_lmc.verb:
                    if i == len(train_loader) - 1:
                        info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, config_lmc.self_max_epoch, i,
                                                                   len(train_loader)) + \
                               'Loss {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses) + \
                               'LossC {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses_c) + \
                               'LossDense {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses_dense) + \
                               'Lr: ' + str(optimiser.state_dict()['param_groups'][0]['lr']) + \
                               'LossDense1 {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses_dense1) + \
                               'LossDense2 {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses_dense2) + \
                               'LossDense3 {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses_dense3)
                        print(info)
                else:
                    if i % config_lmc.print_freq == 0 or i == len(train_loader) - 1:
                        info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, config_lmc.self_max_epoch, i,
                                                                   len(train_loader)) + \
                               'Loss {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses) + \
                               'LossC {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses_c) + \
                               'LossDense {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses_dense)
                        print(info)
        # Step learning rate scheduler
        if config_lmc.sche:
            scheduler.step()
    # Clean up logging
    log.close()
    sys.stdout = raw_std
    return

# Fine-tuning pre-trained model
def Finetune(config_lmc):
    # The code here is almost the same as "Supervised" function,
    # and the comments are not repeated
    # The main difference is that the pre-trained model is loaded here

    def train(model, train_loader, optimizer, epoch, max_epoch):

        def dice(predict, target):
            predict = predict.contiguous().view(predict.shape[0], -1)
            target = target.contiguous().view(target.shape[0], -1)
            num = torch.sum(torch.mul(predict, target), dim=1) + 1e-4
            den = torch.sum(predict + target, dim=1) + 1e-4
            loss = 1 - 2 * num / den
            return loss.mean()

        def multidiceLoss(predict, target):
            b, c, h, w = predict.shape
            predict = predict.view(b, c, h * w)
            target = target.view(b, c, h * w)
            assert predict.shape == target.shape, 'predict & target shape do not match'
            total_loss = []
            predict = torch.nn.functional.softmax(predict, dim=1)
            for i in range(target.shape[1]):
                dice_loss = dice(predict[:, i], target[:, i])
                dice_loss *= config_lmc.dice_weight[i]
                total_loss.append(dice_loss)
            return total_loss

        criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(config_lmc.weight).cuda())

        model.train()
        losses_bce = Averagvalue()
        losses_dice = Averagvalue()
        losses_dice1 = Averagvalue()
        losses_dice2 = Averagvalue()
        losses_dice3 = Averagvalue()
        losses = Averagvalue()
        for i, (image, label) in enumerate(train_loader):
            optimizer.zero_grad()
            image, label = image.cuda(), label.cuda()
            pred = model(image)
            loss_bce = criterion(pred, label)
            loss_dices = multidiceLoss(pred, label)
            loss_diceSum = (loss_dices[1] + loss_dices[2] + loss_dices[3]) / 3
            loss = loss_bce * (1 - config_lmc.dice_bce_ratio) + loss_diceSum * config_lmc.dice_bce_ratio
            loss.backward()
            optimizer.step()
            losses_bce.update(loss_bce.item(), image.size(0))
            losses_dice1.update(loss_dices[1].item(), image.size(0))
            losses_dice2.update(loss_dices[2].item(), image.size(0))
            losses_dice3.update(loss_dices[3].item(), image.size(0))
            losses_dice.update(loss_diceSum.item(), image.size(0))
            losses.update(loss.item(), image.size(0))
            if config_lmc.verb:
                if i == len(train_loader) - 1:
                    info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, max_epoch, i, len(train_loader)) + \
                           'Loss {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses) + \
                           'BCE {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses_bce) + \
                           'DICE {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses_dice) + \
                           '[DICE1 (avg:{loss.avg:f}) '.format(loss=losses_dice1) + \
                           'DICE2 (avg:{loss.avg:f}) '.format(loss=losses_dice2) + \
                           'DICE3 (avg:{loss.avg:f})]'.format(loss=losses_dice3)
                    print(info)
            else:
                if i % config_lmc.print_freq == 0 or i == len(train_loader) - 1:
                    info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, max_epoch, i, len(train_loader)) + \
                           'Loss {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses) + \
                           'BCE {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses_bce) + \
                           'DICE {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses_dice) + \
                           '[DICE1 (avg:{loss.avg:f}) '.format(loss=losses_dice1) + \
                           'DICE2 (avg:{loss.avg:f}) '.format(loss=losses_dice2) + \
                           'DICE3 (avg:{loss.avg:f})]'.format(loss=losses_dice3)
                    print(info)

    def test(model, test_loader, epoch):
        eval_dir = config_lmc.tmp + '/epoch_' + str(epoch)
        if not os.path.exists(eval_dir):
            os.mkdir(eval_dir)
        model.eval()
        accs = []
        inte_c1s = []
        inte_c2s = []
        inte_c3s = []
        oute_c1s = []
        oute_c2s = []
        oute_c3s = []

        for i, (image, label, filename) in enumerate(test_loader):
            image, label = image.cuda(), label.cuda()
            pred = model(image)
            '''
            pred, pred_inner = model(image)
            for idxx, pred_inner_ in enumerate(pred_inner):
                _, c, h, w = pred_inner_.shape
                pred_inner_ = pred_inner_[0].detach().cpu()
                pred_inner_ = pred_inner_.reshape([c, h * w]).permute(1, 0)
                min_vals, _ = torch.min(pred_inner_, dim=1, keepdim=True)
                max_vals, _ = torch.max(pred_inner_, dim=1, keepdim=True)
                pred_inner_np = pred_inner_.detach().cpu().numpy()
                pred_inner_np_idx = np.random.choice(np.arange(pred_inner_np.shape[0]), size=100, )
                pred_inner_np_std = np.std(pred_inner_np[pred_inner_np_idx], axis=0)
                pred_inner_ = (pred_inner_ - min_vals) / (max_vals - min_vals)
                pred_inner_ = pred_inner_.numpy()
                from sklearn.cluster import KMeans
                import matplotlib.pyplot as plt
                y_pred = KMeans(n_clusters=4, random_state=9).fit_predict(pred_inner_)
                s1 = y_pred.reshape([h, w])
                s1 = cv2.resize(s1, dsize=(376, 376), interpolation=cv2.INTER_NEAREST)
                filename_ = filename[0]
                plt.imsave(eval_dir + '/' + str(idxx) + '_' + filename_, s1)
            '''
            pred_prob = torch.nn.functional.softmax(pred, dim=1)
            pred_np = pred_prob.detach().cpu().numpy()[0]
            pred_cls = np.argmax(pred_np, axis=0)
            label_np = label.detach().cpu().numpy()[0]
            label_cls = np.argmax(label_np, axis=0)
            # save prediction
            filename_ = filename[0]
            pred_eval = pred_cls.copy()
            for color in range(4):
                pred_eval[pred_eval == color] = config_lmc.mapping[color]
            cv2.imwrite(eval_dir + '/' + filename_, pred_eval)
            # index
            acc_ = 1 * np.equal(pred_cls, label_cls)
            acc = np.mean(acc_)
            # iou
            pred_c1 = np.zeros([pred_np.shape[1], pred_np.shape[2]])
            pred_c2 = np.zeros([pred_np.shape[1], pred_np.shape[2]])
            pred_c3 = np.zeros([pred_np.shape[1], pred_np.shape[2]])
            pred_c1[pred_cls == 1] = 1
            pred_c2[pred_cls == 2] = 1
            pred_c3[pred_cls == 3] = 1
            inte_c1 = 2 * np.logical_and(pred_c1, label_np[1])
            inte_c2 = 2 * np.logical_and(pred_c2, label_np[2])
            inte_c3 = 2 * np.logical_and(pred_c3, label_np[3])
            oute_c1 = pred_c1.sum() + label_np[1].sum()
            oute_c2 = pred_c2.sum() + label_np[2].sum()
            oute_c3 = pred_c3.sum() + label_np[3].sum()
            # record
            accs.append(acc)
            inte_c1s.append(inte_c1)
            inte_c2s.append(inte_c2)
            inte_c3s.append(inte_c3)
            oute_c1s.append(oute_c1)
            oute_c2s.append(oute_c2)
            oute_c3s.append(oute_c3)

        accs = np.array(accs)
        iou1 = np.array(inte_c1s).sum() / np.array(oute_c1s).sum()
        iou2 = np.array(inte_c2s).sum() / np.array(oute_c2s).sum()
        iou3 = np.array(inte_c3s).sum() / np.array(oute_c3s).sum()

        print('test')
        print('ACC:' + str(np.mean(accs)))
        print('mIoU:' + str((iou1 + iou2 + iou3) / 3))
        print('IoU1:' + str(iou1))
        print('IoU2:' + str(iou2))
        print('IoU3:' + str(iou3))

        return np.mean(accs)

    def save(model, epoch, acc):
        state = {
            # 'args': args,
            'finetune': model.state_dict(),
            'epoch': epoch,
            'acc': acc
        }
        torch.save(state, config_lmc.tmp + '/' + 'fine.pt')
        print('save file with acc: ' + str(acc))

    if not os.path.exists(config_lmc.tmp + '_Num' + config_lmc.sup_num.split('_')[0]):
        os.makedirs(config_lmc.tmp + '_Num' + config_lmc.sup_num.split('_')[0])
    config_lmc.tmp = config_lmc.tmp + '_Num' + config_lmc.sup_num.split('_')[0] + '/' + config_lmc.sup_num
    if not os.path.exists(config_lmc.tmp):
        os.makedirs(config_lmc.tmp)

    shutil.copy(os.path.realpath(__file__), config_lmc.tmp + '/config_fine')
    log = Logger(os.path.join(config_lmc.tmp, 'log_fine.txt'))
    raw_std = sys.stdout
    sys.stdout = log

    train_dataset = Data_preheat(root=config_lmc.dataset_sup, split='train')
    test_dataset = Data_preheat(root=config_lmc.dataset_test, split='test')

    train_loader = DataLoader(train_dataset, batch_size=config_lmc.batch_size, drop_last=True, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, drop_last=True, shuffle=False)
    model = config_lmc.backbone(config_lmc.drop, config_lmc.channel,
                                IncNorm=config_lmc.IncNorm, DownNorm=config_lmc.DownNorm, UpNorm=config_lmc.UpNorm)
    model = model.cuda()
    # The main difference between this function "Finetune" and "Supervised"
    model = load_moco(model, config_lmc.tmp.replace('fine', 'self') + '/' + 'moco.pt'.replace('.pt',
                                                                                              config_lmc.load_moco_ep + '.pt'))
    acc_best = 0
    if config_lmc.opti == 'adam':
        '''
        params_dict = [{'params': model.encode.parameters(), 'lr': config_lmc.fineturn_lr_en},
                       {'params': model.decode.parameters(), 'lr': config_lmc.fineturn_lr_de}]
        '''
        params_dict = [{'params': model.encode.inc.parameters(), 'lr': config_lmc.fineturn_lr_en},
                       {'params': model.encode.down1.parameters(), 'lr': config_lmc.fineturn_lr_en},
                       {'params': model.encode.down2.parameters(), 'lr': config_lmc.fineturn_lr_en},
                       {'params': model.encode.down3.parameters(), 'lr': config_lmc.fineturn_lr_en},
                       # {'params': model.encode.down4.parameters(), 'lr': config_lmc.fineturn_lr_en},  ## haha
                       {'params': model.encode.up1.parameters(), 'lr': config_lmc.fineturn_lr_de},
                       {'params': model.encode.up2.parameters(), 'lr': config_lmc.fineturn_lr_de},
                       {'params': model.encode.up3.parameters(), 'lr': config_lmc.fineturn_lr_de},
                       # {'params': model.encode.up4.parameters(), 'lr': config_lmc.fineturn_lr_de},,  ## haha
                       {'params': model.decode.parameters(), 'lr': config_lmc.fineturn_lr_de}]
        optimizer = torch.optim.Adam(params_dict)
    elif config_lmc.opti == 'adamW':
        # params_dict = [{'params': model.encode.parameters(), 'lr': config_lmc.fineturn_lr_en, "weight_decay":config_lmc.adamw_decay},
        #               {'params': model.decode.parameters(), 'lr': config_lmc.fineturn_lr_de, "weight_decay":config_lmc.adamw_decay}]
        params_dict = [{'params': model.encode.inc.parameters(), 'lr': config_lmc.fineturn_lr_en,
                        "weight_decay": config_lmc.adamw_decay},
                       {'params': model.encode.down1.parameters(), 'lr': config_lmc.fineturn_lr_en,
                        "weight_decay": config_lmc.adamw_decay},
                       {'params': model.encode.down2.parameters(), 'lr': config_lmc.fineturn_lr_en,
                        "weight_decay": config_lmc.adamw_decay},
                       {'params': model.encode.down3.parameters(), 'lr': config_lmc.fineturn_lr_en,
                        "weight_decay": config_lmc.adamw_decay},
                       {'params': model.encode.down4.parameters(), 'lr': config_lmc.fineturn_lr_en,
                        "weight_decay": config_lmc.adamw_decay},
                       {'params': model.encode.up1.parameters(), 'lr': config_lmc.fineturn_lr_de,
                        "weight_decay": config_lmc.adamw_decay},
                       {'params': model.encode.up2.parameters(), 'lr': config_lmc.fineturn_lr_de,
                        "weight_decay": config_lmc.adamw_decay},
                       {'params': model.encode.up3.parameters(), 'lr': config_lmc.fineturn_lr_de,
                        "weight_decay": config_lmc.adamw_decay},
                       {'params': model.encode.up4.parameters(), 'lr': config_lmc.fineturn_lr_de,
                        "weight_decay": config_lmc.adamw_decay},
                       {'params': model.decode.parameters(), 'lr': config_lmc.fineturn_lr_de,
                        "weight_decay": config_lmc.adamw_decay}]
        optimizer = torch.optim.AdamW(params_dict)

    for epoch in range(config_lmc.fine_max_epoch):
        train(model, train_loader, optimizer, epoch, config_lmc.fine_max_epoch)
        if (epoch + 1) % config_lmc.test_freq == 0:
            acc = test(model, test_loader, epoch)
            if acc > acc_best:
                acc_best = acc
                save(model, epoch, acc)

    log.close()
    sys.stdout = raw_std
    return

# This function is used by "Finetune" to load pre-trained parameters
def load_moco(base_encoder, load_checkpoint_dir):
    """ Loads the pre-trained MoCo model parameters.

        Applies the loaded pre-trained params to the base encoder used in Linear Evaluation,
         freezing all layers except the Linear Evaluation layer/s.

    Args:
        base_encoder (model): Randomly Initialised base_encoder.

        args (dict): Program arguments/commandline arguments.
    Returns:
        base_encoder (model): Initialised base_encoder with parameters from the MoCo query_encoder.
    """
    print("\n\nLoading the model: {}\n\n".format(load_checkpoint_dir))

    # Load the pretrained model
    checkpoint = torch.load(load_checkpoint_dir, map_location="cpu")

    # rename moco pre-trained keys
    state_dict = checkpoint['moco']
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('encoder_q') and not k.startswith('encoder_q.decode'):
            # remove prefix
            state_dict[k[len("encoder_q."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    # Load the encoder parameters
    base_encoder.load_state_dict(state_dict, strict=False)

    return base_encoder