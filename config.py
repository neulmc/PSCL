import numpy
import numpy as np
import os
from data import Data_preheat, MoCoData_preheat, MoCoData_preheat_sup
from model import UNet, load_moco, MoCo_DenseModel
from torch.utils.data import DataLoader
from utils import Averagvalue, Logger
import shutil
import sys
import cv2
import torch
import random
import torch.nn as nn
import torch.nn.functional as F


class config():
    def __init__(self, tt, method, backbone, moco_mode, moco_denseloss_ratio, dice_bce_ratio, load_moco_ep, weight,
                 dice_weight, sup_num, epoch_map, batch_size, self_batch_size, self_max_epoch, self_save_epoch,
                 jitter_d, random_c, super_lr, self_lr, queue_size, queue_momentum, temperature,
                 lab_size_decay, env, seed, print_freq, mapping, patch_num, verb,
                 drop, channel, opti, sche, adamw_decay, dense_weight, fineturn_lr_en, fineturn_lr_de, sample_num,
                 self_scale_weight, sample_ratio, confea_num, hidfea_num, top_k, patch_lab_decay, patch_size_decay,
                 multihead,
                 IncNorm, DownNorm, UpNorm, HeadNrom, global_weight, Patch_sup, selfmode, simclrinner, data_dir):

        # dir & method
        self.tt = tt
        self.method = method  # sup, self, fine
        self.backbone = backbone  # UNet, VGG
        self.moco_mode = moco_mode  # MoCo_Model, MoCo_DenseModel
        self.moco_denseloss_ratio = moco_denseloss_ratio  # dense ratio [0-1]
        self.dice_bce_ratio = dice_bce_ratio  # dense ratio [0-1]
        self.load_moco_ep = load_moco_ep  # 0, 10, 20, 30, 40, 50, 60, 70, 80, 90,
        self.weight = weight
        self.dice_weight = dice_weight
        self.patch_num = patch_num
        self.verb = verb
        self.drop = drop
        self.channel = channel
        self.opti = opti
        self.sche = sche
        self.adamw_decay = adamw_decay
        self.dense_weight = dense_weight
        self.sample_num = sample_num
        self.self_scale_weight = self_scale_weight
        self.sample_ratio = sample_ratio
        self.confea_num = confea_num
        self.hidfea_num = hidfea_num
        self.top_k = top_k
        self.patch_lab_decay = patch_lab_decay
        self.patch_size_decay = patch_size_decay
        # data
        self.sup_num = sup_num  # '1','5','f'
        self.epoch_map = epoch_map
        self.batch_size = batch_size
        self.self_batch_size = self_batch_size
        self.self_max_epoch = self_max_epoch
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
        self.self_save_epoch = self_save_epoch
        self.jitter_d = jitter_d
        self.random_c = random_c
        self.multihead = multihead
        self.IncNorm = IncNorm
        self.DownNorm = DownNorm
        self.UpNorm = UpNorm
        self.HeadNrom = HeadNrom
        self.global_weight = global_weight
        self.Patch_sup = Patch_sup
        # hyper-parameters
        self.super_lr = super_lr
        self.self_lr = self_lr
        self.queue_size = queue_size  # Size of Memory Queue, Must be Divisible by batch_size. 256
        self.queue_momentum = queue_momentum  # Momentum for the Key Encoder Update.
        self.temperature = temperature
        self.selfmode = selfmode
        self.simclrinner = simclrinner

        # hyper-parameters(Ours)
        self.lab_size_decay = lab_size_decay  # label decay 02

        # environment
        os.environ["CUDA_VISIBLE_DEVICES"] = env
        # torch.set_num_threads(2)  # gai
        self.seed = seed

        # print
        self.print_freq = print_freq
        self.test_freq = int(100 / self.epoch_map[self.sup_num.split('_')[0]])
        self.mapping = mapping

        # build dir

        self.tmp = self.method + '_' + self.backbone.__name__ + '_' + self.tt + '/'
        if not os.path.exists(self.tmp):
            os.makedirs(self.tmp)

        self.dataset_sup = data_dir + '/sup' + '_num' + self.sup_num
        self.dataset_train = data_dir + '/train' + '_numf'
        self.dataset_test = data_dir + '/test' + '_num' + self.sup_num.split('_')[0]

        self.fineturn_lr_en = fineturn_lr_en
        self.fineturn_lr_de = fineturn_lr_de


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
    config_lmc = config(tt, method, backbone, moco_mode, moco_denseloss_ratio, dice_bce_ratio, load_moco_ep, weight,
                        dice_weight, sup_num, epoch_map, batch_size, self_batch_size, self_max_epoch, self_save_epoch,
                        jitter_d, random_c, super_lr, self_lr, queue_size, queue_momentum, temperature,
                        lab_size_decay, env, seed, print_freq, mapping, patch_num, verb, drop, channel, opti, sche,
                        adamw_decay,
                        dense_weight, fineturn_lr_en, fineturn_lr_de, sample_num, self_scale_weight, sample_ratio,
                        confea_num, hidfea_num, top_k, patch_lab_decay, patch_size_decay, multihead, IncNorm, DownNorm,
                        UpNorm, HeadNrom,
                        global_weight, Patch_sup, selfmode, simclrinner, data_dir)

    seed_torch(config_lmc.seed)

    if config_lmc.method == 'sup':
        Supervised(config_lmc)
    elif config_lmc.method == 'self':
        SelfSupervised(config_lmc)
    elif config_lmc.method == 'fine':
        Finetune(config_lmc)

    return


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)


def Supervised(config_lmc):
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
            pred,pred_inner = model(image)
            pred_inner = pred_inner[0].detach().cpu()
            pred_inner = pred_inner.reshape([1024,23*23]).permute(1,0)
            pred_inner = pred_inner.numpy()
            from sklearn.cluster import KMeans
            import matplotlib.pyplot as plt
            y_pred = KMeans(n_clusters=5, random_state=9).fit_predict(pred_inner)
            s1 = y_pred.reshape([23,23])
            s1 = cv2.resize(s1, dsize=(376,376), interpolation=cv2.INTER_NEAREST )
            filename_ = filename[0]
            plt.imsave(eval_dir + '/1_' + filename_, s1)
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
            'super': model.state_dict(),
            'epoch': epoch,
            'acc': acc
        }
        torch.save(state, config_lmc.tmp + '/' + 'sup.pt')
        print('save file with acc: ' + str(acc))

    if not os.path.exists(config_lmc.tmp + '_' + config_lmc.sup_num.split('_')[0]):
        os.makedirs(config_lmc.tmp + '_' + config_lmc.sup_num.split('_')[0])
    config_lmc.tmp = config_lmc.tmp + '_' + config_lmc.sup_num.split('_')[0] + '/' + config_lmc.sup_num
    if not os.path.exists(config_lmc.tmp):
        os.makedirs(config_lmc.tmp)

    shutil.copy(os.path.realpath(__file__), config_lmc.tmp + '/config_sup')
    log = Logger(os.path.join(config_lmc.tmp, 'log_sup.txt'))
    raw_std = sys.stdout
    sys.stdout = log

    train_dataset = Data_preheat(root=config_lmc.dataset_sup, split='train')
    test_dataset = Data_preheat(root=config_lmc.dataset_test, split='test')
    # test_dataset = Data_preheat(root='E:/selfsupervised/dataset/preheat/train_numf', split='test')  # hahaha
    # test_dataset = Data_preheat(root=config_lmc.dataset_sup, split='test')  # hahaha
    train_loader = DataLoader(train_dataset, batch_size=config_lmc.batch_size, drop_last=True, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, drop_last=True, shuffle=False)
    model = config_lmc.backbone(config_lmc.drop, config_lmc.channel,
                                IncNorm=config_lmc.IncNorm, DownNorm=config_lmc.DownNorm, UpNorm=config_lmc.UpNorm)
    model = model.cuda()
    acc_best = 0
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config_lmc.super_lr)
    for epoch in range(config_lmc.supv_max_epoch):
        train(model, train_loader, optimizer, epoch, config_lmc.supv_max_epoch)
        if (epoch + 1) % config_lmc.test_freq == 0:
            acc = test(model, test_loader, epoch)
            if acc > acc_best:
                acc_best = acc
                save(model, epoch, acc)

    log.close()
    sys.stdout = raw_std
    return


def SelfSupervised(config_lmc):
    if config_lmc.moco_denseloss_ratio > 1 or config_lmc.moco_denseloss_ratio < 0:
        print('error')
        return

    if not os.path.exists(config_lmc.tmp + '_Num' + config_lmc.sup_num.split('_')[0]):
        os.makedirs(config_lmc.tmp + '_Num' + config_lmc.sup_num.split('_')[0])
    config_lmc.tmp = config_lmc.tmp + '_Num' + config_lmc.sup_num.split('_')[0] + '/' + config_lmc.sup_num
    if not os.path.exists(config_lmc.tmp):
        os.makedirs(config_lmc.tmp)

    shutil.copy(os.path.realpath(__file__), config_lmc.tmp + '/config_self')
    shutil.copy('model.py', config_lmc.tmp + '/model_self')
    log = Logger(os.path.join(config_lmc.tmp, 'log_self.txt'))
    raw_std = sys.stdout
    sys.stdout = log

    train_dataset = MoCoData_preheat(root=config_lmc.dataset_train, jitter_d=config_lmc.jitter_d,
                                     random_c=config_lmc.random_c)
    train_loader = DataLoader(train_dataset, batch_size=config_lmc.self_batch_size, drop_last=True, shuffle=True)
    sup_dataset = MoCoData_preheat_sup(root=config_lmc.dataset_sup, jitter_d=config_lmc.jitter_d,
                                       random_c=config_lmc.random_c)
    sup_loader = DataLoader(sup_dataset, batch_size=1, drop_last=True, shuffle=True)

    MoComodel = config_lmc.moco_mode(backbone=config_lmc.backbone, id_num=len(train_dataset),
                                     queue_size=config_lmc.queue_size, momentum=config_lmc.queue_momentum,
                                     temperature=config_lmc.temperature,
                                     lab_size_decay=config_lmc.lab_size_decay, patch_num=config_lmc.patch_num,
                                     drop=config_lmc.drop, channel=config_lmc.channel,
                                     sample_num=config_lmc.sample_num, self_scale_weight=config_lmc.self_scale_weight,
                                     sample_ratio=config_lmc.sample_ratio,
                                     confea_num=config_lmc.confea_num, hidfea_num=config_lmc.hidfea_num,
                                     top_k=config_lmc.top_k, patch_lab_decay=config_lmc.patch_lab_decay,
                                     patch_size_decay=config_lmc.patch_size_decay, multihead=config_lmc.multihead,
                                     IncNorm=config_lmc.IncNorm, DownNorm=config_lmc.DownNorm, UpNorm=config_lmc.UpNorm,
                                     HeadNrom=config_lmc.HeadNrom, Patch_sup=config_lmc.Patch_sup,
                                     selfmode=config_lmc.selfmode)
    MoComodel = MoComodel.cuda()
    if config_lmc.opti == 'adam':
        optimiser = torch.optim.Adam(params=MoComodel.parameters(), lr=config_lmc.self_lr)
    elif config_lmc.opti == 'adamW':
        optimiser = torch.optim.AdamW(params=MoComodel.parameters(), lr=config_lmc.self_lr,
                                      weight_decay=config_lmc.adamw_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimiser, [int(config_lmc.self_max_epoch * 0.5),
                                                                 int(config_lmc.self_max_epoch * 0.75)], gamma=0.1)

    criterion = nn.CrossEntropyLoss()

    def nt_xent_loss_d(z1, z2, temperature=0.5, deep=0, bat=1):
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
        # muban_t1 = np.hstack([muban_p0, muban_t])
        # muban_t2 = np.hstack([muban_t, muban_p0])
        muban_t1 = np.hstack([muban_t, muban_t])
        muban_t2 = np.hstack([muban_t, muban_t])
        muban = np.vstack([muban_t1, muban_t2])
        muban_new = np.tile(muban, (bat, bat))
        # shapes = muban_new.shape[0]
        # unique_random_numbers = random.sample(range(0, shapes), int(shapes*(9/10)))  # sample
        # muban_new[:,unique_random_numbers] = 0
        if not config_lmc.simclrinner:
            lenth = 4 * config_lmc.top_k * config_lmc.sample_ratio ** deep
            for ii in range(bat*2):
                muban_new[int(ii*lenth):int((ii+1)*lenth), int(ii*lenth):int((ii+1)*lenth)] = 0

        """ NT-Xent loss """
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        N, Z = z1.shape
        device = z1.device
        representations = torch.cat([z1, z2], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=-1)
        l_pos = torch.diag(similarity_matrix, N)
        r_pos = torch.diag(similarity_matrix, -N)
        # similarity_matrix_ = similarity_matrix.detach().cpu().numpy()
        # l_pos_ = l_pos.detach().cpu().numpy()
        # r_pos_ = r_pos.detach().cpu().numpy()
        positives = torch.cat([l_pos, r_pos]).view(2 * N, 1)
        diag = torch.eye(2 * N, dtype=torch.bool, device=device)
        diag[N:, :N] = diag[:N, N:] = diag[:N, :N]
        negatives = similarity_matrix[muban_new == 1].view(2 * N, -1)
        # positives_ = positives.detach().cpu().numpy()
        # negatives_ = negatives.detach().cpu().numpy()
        logits = torch.cat([positives, negatives], dim=1)
        logits /= temperature
        labels = torch.zeros(2 * N, device=device, dtype=torch.int64)
        loss = F.cross_entropy(logits, labels, reduction='sum')
        return loss / (2 * N)

    def nt_xent_loss(z1, z2, temperature=0.5):
        """ NT-Xent loss """
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        N, Z = z1.shape
        device = z1.device
        representations = torch.cat([z1, z2], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=-1)
        l_pos = torch.diag(similarity_matrix, N)
        r_pos = torch.diag(similarity_matrix, -N)
        # similarity_matrix_ = similarity_matrix.detach().cpu().numpy()
        # l_pos_ = l_pos.detach().cpu().numpy()
        # r_pos_ = r_pos.detach().cpu().numpy()
        positives = torch.cat([l_pos, r_pos]).view(2 * N, 1)
        diag = torch.eye(2 * N, dtype=torch.bool, device=device)
        diag[N:, :N] = diag[:N, N:] = diag[:N, :N]
        negatives = similarity_matrix[~diag].view(2 * N, -1)
        # positives_ = positives.detach().cpu().numpy()
        # negatives_ = negatives.detach().cpu().numpy()
        logits = torch.cat([positives, negatives], dim=1)
        logits /= temperature
        labels = torch.zeros(2 * N, device=device, dtype=torch.int64)
        loss = F.cross_entropy(logits, labels, reduction='sum')
        return loss / (2 * N)

    for epoch in range(config_lmc.self_max_epoch + 1):
        # Train models
        MoComodel.train()
        losses = Averagvalue()
        losses_c = Averagvalue()
        losses_dense = Averagvalue()
        losses_dense1 = Averagvalue()
        losses_dense2 = Averagvalue()
        losses_dense3 = Averagvalue()
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
            # retrieve the 2 views
            x_i, x_j = torch.split(inputs, [3, 3], dim=1)

            inputs_sup, ids_sup, rot_k_sup, filp_sup, r1_sup, r2_sup, r3_sup, r4_sup = next(iter(sup_loader))

            inputs_sup = inputs_sup.cuda(non_blocking=True)
            x_i_sup, x_j_sup, y_i_sup = torch.split(inputs_sup, [3, 3, 4], dim=1)

            x_i = torch.cat([x_i_sup, x_i], dim=0)
            x_j = torch.cat([x_j_sup, x_j], dim=0)
            ids = list(ids_sup) + list(ids)
            rot_k = torch.cat([rot_k_sup, rot_k], dim=0)
            filp = torch.cat([filp_sup, filp], dim=0)
            r1 = torch.cat([r1_sup, r1], dim=0)
            r2 = torch.cat([r2_sup, r2], dim=0)
            r3 = torch.cat([r3_sup, r3], dim=0)
            r4 = torch.cat([r4_sup, r4], dim=0)
            '''
            a = np.array([0.49139968, 0.48215841, 0.44653091])
            b = np.array([0.24703223, 0.24348513, 0.26158784])
            x_i_ = (x_i.permute(0, 2, 3, 1).detach().cpu().numpy() * b + a) * 255
            x_j_ = (x_j.permute(0, 2, 3, 1).detach().cpu().numpy() * b + a) * 255
            cv2.imwrite('sb/xi1.png', x_i_[0])
            cv2.imwrite('sb/xi2.png', x_i_[1])
            cv2.imwrite('sb/xi3.png', x_i_[2])
            cv2.imwrite('sb/xi4.png', x_i_[3])
            cv2.imwrite('sb/xj1.png', x_j_[0])
            cv2.imwrite('sb/xj2.png', x_j_[1])
            cv2.imwrite('sb/xj3.png', x_j_[2])
            cv2.imwrite('sb/xj4.png', x_j_[3])
            sd = x_i_[0][:,:,0]
            '''
            # Get the encoder representation
            if config_lmc.selfmode == 'moco':
                logit, label, logits_dense, labels_dense = MoComodel(x_i, x_j, y_i_sup, ids, rot_k, filp, r1, r2, r3,
                                                                     r4)
                loss_c = criterion(logit, label)
                loss_dense = criterion(logits_dense, labels_dense)
                base_num1 = config_lmc.top_k * 4
                base_num2 = config_lmc.top_k * 4 * config_lmc.sample_ratio + base_num1
                base_num3 = config_lmc.top_k * 4 * config_lmc.sample_ratio ** 2 + base_num2
                loss_dense1 = criterion(logits_dense[0:base_num1], labels_dense[0:base_num1])
                loss_dense2 = criterion(logits_dense[base_num1:base_num2], labels_dense[base_num1:base_num2])
                loss_dense3 = criterion(logits_dense[base_num2:base_num3], labels_dense[base_num2:base_num3])
                loss = loss_c * config_lmc.global_weight * (1 - config_lmc.moco_denseloss_ratio) + \
                       loss_dense * config_lmc.moco_denseloss_ratio * config_lmc.dense_weight
                loss.backward()
                optimiser.step()
                # torch.cuda.synchronize()
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
                loss_dense1 = nt_xent_loss_d(q_dense1, k_dense1, config_lmc.temperature, deep=0, bat=q.shape[0])
                loss_dense2 = nt_xent_loss_d(q_dense2, k_dense2, config_lmc.temperature, deep=1, bat=q.shape[0])
                loss_dense3 = nt_xent_loss_d(q_dense3, k_dense3, config_lmc.temperature, deep=2, bat=q.shape[0])
                loss_c = nt_xent_loss(q, k, config_lmc.temperature)
                loss_dense = (loss_dense1 * config_lmc.self_scale_weight[0] +
                              loss_dense2 * config_lmc.self_scale_weight[1] +
                              loss_dense3 * config_lmc.self_scale_weight[2]) / 3
                loss = loss_c * config_lmc.global_weight * (1 - config_lmc.moco_denseloss_ratio) + \
                       loss_dense * config_lmc.moco_denseloss_ratio * config_lmc.dense_weight
                loss.backward()
                optimiser.step()
                # torch.cuda.synchronize()
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
        if config_lmc.sche:
            scheduler.step()
    log.close()
    sys.stdout = raw_std
    return


def Finetune(config_lmc):
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
        train(model, train_loader, optimizer, epoch, config_lmc.fine_max_epoch)  # hahaha
        if (epoch + 1) % config_lmc.test_freq == 0:
            acc = test(model, test_loader, epoch)
            if acc > acc_best:
                acc_best = acc
                save(model, epoch, acc)

    log.close()
    sys.stdout = raw_std
    return


def analyse_log(tt, RepeatID, mode):
    def read_log(file):
        mIoU = -1
        ACC = 0
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
            f.writelines('\n' + 'Numf:')
            mIoU, ACC, mIoU_epoch, loss, Loss_epoch = read_log(dataf_dir + '/f/log_fine.txt')
            f.writelines('\n' + str(i) + ': [mIoU]: ' + str(mIoU) + ' [ACC]: ' + str(ACC) + '[mIoU_epoch]: ' + str(mIoU_epoch) +
                         '[loss]: ' + str(loss) + '[Loss_epoch]: ' + str(Loss_epoch))
            f.writelines('\n' + 'Total' + ': [mIoU]: ' + str(mIoU) + ' [ACC]: ' + str(ACC))


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


def self2fine(tt, method='fine', env='0', RepeatID='01234', mode='15f',
              global_weight=1, dense_weight=1, multihead=4, Patch_sup=True,  # ablation
              patch_size_decay=0.5, patch_lab_decay=0.2, lab_size_decay=0.2,  # decay
              queue_size=504, top_k=4,  # moco top_k=30
              moco_denseloss_ratio=0.7, self_scale_weight=[1, 1, 1, 1], HeadNrom=['LN',''],
              queue_momentum=0.99, temperature=0.5, selfmode='moco', simclrinner = True, data_dir = ''):  # loss ratio

    for i in range(5):
        if ('1' in mode) and (str(i) in RepeatID):
            runner_preheat(tt=tt, method=method, env=env, sup_num='1_' + str(i), global_weight=global_weight,
                           dense_weight=dense_weight, multihead=multihead, Patch_sup=Patch_sup,
                           patch_size_decay=patch_size_decay, patch_lab_decay=patch_lab_decay,
                           lab_size_decay=lab_size_decay, self_scale_weight=self_scale_weight, HeadNrom = HeadNrom,
                           queue_size=queue_size, top_k=top_k, moco_denseloss_ratio=moco_denseloss_ratio,
                           queue_momentum=queue_momentum, temperature=temperature, selfmode=selfmode, simclrinner = simclrinner, data_dir = data_dir)
    for i in range(5):
        if ('5' in mode) and (str(i) in RepeatID):
            runner_preheat(tt=tt, method=method, env=env, sup_num='5_' + str(i), global_weight=global_weight,
                           dense_weight=dense_weight, multihead=multihead, Patch_sup=Patch_sup,
                           patch_size_decay=patch_size_decay, patch_lab_decay=patch_lab_decay,
                           lab_size_decay=lab_size_decay, self_scale_weight=self_scale_weight, HeadNrom = HeadNrom,
                           queue_size=queue_size, top_k=top_k, moco_denseloss_ratio=moco_denseloss_ratio,
                           queue_momentum=queue_momentum, temperature=temperature, selfmode=selfmode, simclrinner = simclrinner, data_dir = data_dir)
    if 'f' in mode:
        runner_preheat(tt=tt, method=method, env=env, sup_num='f', global_weight=global_weight,
                       dense_weight=dense_weight, multihead=multihead, Patch_sup=Patch_sup,
                       patch_size_decay=patch_size_decay, patch_lab_decay=patch_lab_decay,
                       lab_size_decay=lab_size_decay, self_scale_weight=self_scale_weight, HeadNrom = HeadNrom,
                       queue_size=queue_size, top_k=top_k, moco_denseloss_ratio=moco_denseloss_ratio,
                       queue_momentum=queue_momentum, temperature=temperature, selfmode=selfmode, simclrinner =simclrinner, data_dir = data_dir)

    if method == 'fine':
        analyse_log(tt=tt, RepeatID=RepeatID, mode=mode)
    elif method == 'sup':
        root_dir = 'sup_UNet_' + tt
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        analyse_sup_log(tt=root_dir)


if __name__ == '__main__':
    self2fine(tt='PSCL-MoCo', method='self', env='0', RepeatID='01234', mode='1', selfmode='moco', temperature=0.5, moco_denseloss_ratio=0.7, data_dir = '')
    self2fine(tt='PSCL-MoCo', method='fine', env='0', RepeatID='01234', mode='1', selfmode='moco', temperature=0.5, moco_denseloss_ratio=0.7, data_dir = '')