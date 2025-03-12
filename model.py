# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import random
import cv2

def load_model(base_encoder, load_checkpoint_dir):
    # Load the pretrained model
    checkpoint = torch.load(load_checkpoint_dir, map_location="cpu")

    # rename moco pre-trained keys
    state_dict = checkpoint['finetune']

    # Load the encoder parameters
    base_encoder.load_state_dict(state_dict, strict=True)

    return base_encoder

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

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        # Different model for smaller image size
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)  # For CIFAR

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=0)  # For CIFAR

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x

class projection_MLP(nn.Module):
    def __init__(self):
        '''Projection head for the pretraining of the resnet encoder.

            - Uses the dataset and model size to determine encoder output
                representation dimension.
            - Outputs to a dimension of 128, and uses non-linear activation
                as described in SimCLR paper: https://arxiv.org/pdf/2002.05709.pdf
        '''
        super(projection_MLP, self).__init__()

        n_channels = 512

        self.projection_head = nn.Sequential()

        self.projection_head.add_module('W1', nn.Linear(
            n_channels, n_channels))
        self.projection_head.add_module('ReLU', nn.ReLU())
        self.projection_head.add_module('W2', nn.Linear(
            n_channels, 128))

    def forward(self, x):
        return self.projection_head(x)

class projection_UNET_MLP(nn.Module):
    def __init__(self):
        super(projection_UNET_MLP, self).__init__()

        n_channels = 1024
        self.avg_pooling = torch.nn.AdaptiveMaxPool2d((1, 1))
        #self.avg_pooling = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.projection_head = nn.Sequential()

        self.projection_head.add_module('W1', nn.Linear(
            n_channels, n_channels))
        self.projection_head.add_module('ReLU', nn.ReLU())
        self.projection_head.add_module('W2', nn.Linear(
            n_channels, 128))

    def forward(self, x):
        B, C, H, W = x[0].size()
        x = self.avg_pooling(x[0]).view(B, -1)
        return self.projection_head(x)

class MoCo_DenseModel(nn.Module):
    def __init__(self, backbone, id_num, queue_size=65536, momentum=0.999, temperature=0.07,
                 lab_size_decay=0.2, patch_num = 4, drop = 0, channel = 0, sample_num = 100,
                 self_scale_weight = [1, 1, 1], sample_ratio = 4, confea_num = 64, hidfea_num = 128, top_k = 15,
                 patch_lab_decay = 0.2, patch_size_decay = 0.2, multihead = 4, IncNorm =[], DownNorm=[], UpNorm=[], HeadNrom = [],
                 Patch_sup = True, selfmode = 'moco'):

        super(MoCo_DenseModel, self).__init__()
        self.id_num = id_num
        self.info_ids = dict()
        self.info_size = dict()
        self.lab_size = np.zeros((queue_size, 6))
        self.patch_size = np.zeros((queue_size, 6))
        self.lab_size_decay = lab_size_decay
        self.patch_num = patch_num  # old --> class num
        self.sample_num = sample_num
        self.self_scale_weight = self_scale_weight
        self.sample_ratio = sample_ratio
        self.patch_lab_decay = patch_lab_decay   # new
        self.patch_size_decay = patch_size_decay  # new
        self.top_k = top_k # new
        self.multihead = multihead
        self.Patch_sup = Patch_sup
        self.selfmode = selfmode

        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature

        # Load model
        self.encoder_q = backbone(drop=drop, channel=channel, IncNorm=IncNorm, DownNorm=DownNorm, UpNorm=UpNorm)  # Query Encoder
        self.encoder_k = backbone(drop=drop, channel=channel, IncNorm=IncNorm, DownNorm=DownNorm, UpNorm=UpNorm)  # Key Encoder

        # Add the mlp head
        self.encoder_q.decode = Denseproj_UNET_MLP(channel=channel, hidfea_num = hidfea_num, confea_num = confea_num, multihead = multihead, normName = HeadNrom)
        self.encoder_k.decode = Denseproj_UNET_MLP(channel=channel, hidfea_num = hidfea_num, confea_num = confea_num, multihead = multihead, normName = HeadNrom)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # Create the queue to store negative samples
        self.register_buffer("queue", torch.randn(self.queue_size, confea_num))
        self.register_buffer("queue2", torch.randn(len(self_scale_weight), self.queue_size * patch_num, confea_num)) # 4个尺度
        #self.register_buffer("queue2", torch.randn(len(self_scale_weight), (self.queue_size + 100) * patch_num, confea_num))

        # Create pointer to store current position in the queue when enqueue and dequeue
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        patch_labels = []
        self.patch_lab_tag = torch.zeros(patch_num*self.queue_size).cuda()
        for jj in range(4):
            c1_label = torch.from_numpy(np.array([1, 0, 0, 0])).repeat(top_k * (sample_ratio **jj), self.queue_size).cuda()
            c2_label = torch.from_numpy(np.array([0, 1, 0, 0])).repeat(top_k * (sample_ratio **jj), self.queue_size).cuda()
            c3_label = torch.from_numpy(np.array([0, 0, 1, 0])).repeat(top_k * (sample_ratio **jj), self.queue_size).cuda()
            c4_label = torch.from_numpy(np.array([0, 0, 0, 1])).repeat(top_k * (sample_ratio **jj), self.queue_size).cuda()
            s_label = torch.cat([c1_label, c2_label, c3_label, c4_label], dim=0)
            patch_labels.append(s_label)
        self.patch_labels = patch_labels

    @torch.no_grad()
    def momentum_update(self):
        '''
        Update the key_encoder parameters through the momentum update:


        key_params = momentum * key_params + (1 - momentum) * query_params

        '''

        # For each of the parameters in each encoder
        for p_q, p_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            p_k.data = p_k.data * self.momentum + p_q.detach().data * (1. - self.momentum)

    @torch.no_grad()
    def shuffled_idx(self, batch_size):
        '''
        Generation of the shuffled indexes for the implementation of ShuffleBN.

        https://github.com/HobbitLong/CMC.

        args:
            batch_size (Tensor.int()):  Number of samples in a batch

        returns:
            shuffled_idxs (Tensor.long()): A random permutation index order for the shuffling of the current minibatch

            reverse_idxs (Tensor.long()): A reverse of the random permutation index order for the shuffling of the
                                            current minibatch to get back original sample order

        '''

        # Generate shuffled indexes
        shuffled_idxs = torch.randperm(batch_size).long().cuda()

        reverse_idxs = torch.zeros(batch_size).long().cuda()

        value = torch.arange(batch_size).long().cuda()

        reverse_idxs.index_copy_(0, shuffled_idxs, value)

        return shuffled_idxs, reverse_idxs

    @torch.no_grad()
    def update_queue(self, k, k_patch_list, size_label):

        batch_size = k.size(0)
        ptr = int(self.queue_ptr)

        # replace the keys at ptr (dequeue and enqueue)
        self.patch_lab_tag[ptr*self.patch_num :(ptr + batch_size)*self.patch_num] = 1
        self.queue[ptr:ptr + batch_size, :] = k
        for batch_size_i in range(batch_size):
            for scale_i in range(len(k_patch_list)):
                self.queue2[scale_i][(ptr + batch_size_i) * self.patch_num :(ptr + batch_size_i + 1) * self.patch_num] \
                    = k_patch_list[scale_i][batch_size_i]
        self.lab_size[ptr:ptr + batch_size, :] = np.eye(6)[size_label]
        self.patch_size[ptr:ptr + batch_size, :] = np.eye(6)[size_label]

        ptr = (ptr + batch_size) % self.queue_size

        # Store queue pointer as register_buffer
        self.queue_ptr[0] = ptr

        # weight decay for labels
        self.lab_size = self.lab_size * self.lab_size_decay
        self.patch_size = self.patch_size * self.patch_size_decay
        self.patch_lab_tag = self.patch_lab_tag * self.patch_lab_decay

    def InfoNCE_logits(self, q, k, y_i_sup, q_grid_list, k_grid_list, size_label, rot_k, filp, batch_size, r1, r2, r3, r4, fh_list, fw_list):

        def tensor2logit(pos, neg, l_dense_weight = 0, dense_batch = 0, scale_num = 0, cut_lists = []):
            logits = torch.cat((pos, neg), dim=1)
            logits /= self.temperature
            labels = torch.zeros(logits.shape, dtype=torch.float).cuda()
            # pos label
            labels[:, 0] = 1
            # history lable
            if dense_batch == 0:
                label_size = np.eye(6)[size_label]
                label_add = np.matmul(label_size, self.lab_size.T)
                labels[:, 1:] = torch.from_numpy(label_add)
            else:
                #l_dense_weight_ = l_dense_weight.detach().cpu().numpy()
                #l_dense_weight
                label_size = np.eye(6)[size_label]
                label_add = np.matmul(label_size, self.patch_size.T)
                label_add = np.repeat(label_add, self.patch_num, axis=1)
                label_add = torch.from_numpy(label_add).cuda()
                for i_b in range(dense_batch):
                    for j_s in range(scale_num):
                        labels[cut_lists[i_b][j_s]:cut_lists[i_b][j_s + 1], 1:self.queue_size * self.patch_num + 1] += self.patch_labels[j_s] * label_add[i_b]
                        labels[cut_lists[i_b][j_s]:cut_lists[i_b][j_s + 1], 1:self.queue_size*self.patch_num+1] += self.patch_labels[j_s] * self.patch_lab_tag
                        labels[cut_lists[i_b][j_s]:cut_lists[i_b][j_s + 1], :] *= self.self_scale_weight[j_s]
                labels = labels * l_dense_weight.unsqueeze(-1).detach() # weight new
            return logits, labels

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,kc->nk', [q, self.queue.clone().detach()])

        filp = filp.detach().cpu().numpy()
        rot_k = rot_k.detach().cpu().numpy()
        cut_lists = []
        template_k_values = []
        template_k_labels = []

        for batch_i in range(batch_size):
            cut_list = []
            if batch_i == 0:
                cut_list.append(0)
            else:
                cut_list.append(cut_lists[-1][-1])
            for scale_i in range(len(q_grid_list)):
                template_q = torch.from_numpy(np.array(range(fh_list[scale_i] * fw_list[scale_i])).reshape([fh_list[scale_i], fw_list[scale_i]]))
                template_k = torch.from_numpy(np.array(range(fh_list[scale_i] * fw_list[scale_i])).reshape([fh_list[scale_i], fw_list[scale_i]]))
                if filp[batch_i] == 1:
                    template_q = torch.flip(template_q, dims=[0])
                template_q = torch.rot90(template_q, k=rot_k[batch_i], dims=[0, 1])
                template_q = template_q[int(r3[batch_i] * fh_list[scale_i]): int(r3[batch_i] * fh_list[scale_i] + r1[batch_i] * fh_list[scale_i]),
                             int(r4[batch_i] * fw_list[scale_i]): int(r4[batch_i] * fw_list[scale_i] + r2[batch_i] * fw_list[scale_i])]
                fh_q, fw_q = template_q.shape
                template_q = torch.reshape(template_q, (-1,)).long().cuda()
                template_k = torchvision.transforms.Resize([fh_q, fw_q], interpolation=torchvision.transforms.InterpolationMode.NEAREST)\
                    (template_k.unsqueeze(0))
                template_k = torch.reshape(template_k, (-1,)).long().cuda()
                # sample it
                if batch_i == 0:
                    y_i_sup_resize = torchvision.transforms.Resize([fh_q, fw_q],)(y_i_sup)
                    y_i_sup_resize = torch.reshape(y_i_sup_resize, (4,-1))
                    epss = torch.rand(y_i_sup_resize.shape).cuda() * 0.1
                    history = torch.zeros(y_i_sup_resize.shape).cuda()
                    if not self.Patch_sup:
                        y_i_sup_resize = torch.zeros(y_i_sup_resize.shape).cuda()
                    class0_v, class0_indice = torch.topk(y_i_sup_resize[0] + epss[0] + history[0], k=self.top_k*(self.sample_ratio ** scale_i), largest=True)
                    history[:,class0_indice] += -torch.max(y_i_sup_resize)
                    class1_v, class1_indice = torch.topk(y_i_sup_resize[1] + epss[1] + history[1], k=self.top_k*(self.sample_ratio ** scale_i), largest=True)
                    history[:,class1_indice] += -torch.max(y_i_sup_resize)
                    class2_v, class2_indice = torch.topk(y_i_sup_resize[2] + epss[2] + history[2], k=self.top_k*(self.sample_ratio ** scale_i), largest=True)
                    history[:,class2_indice] += -torch.max(y_i_sup_resize)
                    class3_v, class3_indice = torch.topk(y_i_sup_resize[3] + epss[3] + history[3], k=self.top_k*(self.sample_ratio ** scale_i), largest=True)
                    L1 = torch.cat([class0_indice, class1_indice, class2_indice, class3_indice])

                    #class0_v_ = class0_v.detach().cpu().numpy()
                    #class1_v_ = class1_v.detach().cpu().numpy()
                    #class2_v_ = class2_v.detach().cpu().numpy()
                    #class3_v_ = class3_v.detach().cpu().numpy()

                    template_q = template_q[L1]
                    template_k = template_k[L1]
                    indexed_q_grid = torch.gather(q_grid_list[scale_i][batch_i], 1,
                                                  template_q.expand(q_grid_list[scale_i].size(1), -1))  # NxCxS^2
                    indexed_k_grid = torch.gather(k_grid_list[scale_i][batch_i], 1,
                                                  template_k.expand(k_grid_list[scale_i].size(1), -1))  # NxCxS^2

                    densecl_sim_q = (indexed_q_grid * indexed_k_grid).sum(0)  # NxS^2
                    indexed_q_grid = indexed_q_grid.permute(1, 0)
                    cut_list.append(cut_list[-1] + len(L1))

                    class0_v = class0_v / torch.sum(class0_v)
                    class1_v = class1_v / torch.sum(class1_v)
                    class2_v = class2_v / torch.sum(class2_v)
                    class3_v = class3_v / torch.sum(class3_v)

                    ratio_now = self.sample_ratio ** scale_i
                    patch_cls0 = torch.sum(indexed_k_grid[:, 0:self.top_k*ratio_now] * class0_v, dim = 1, keepdim=True)
                    patch_cls1 = torch.sum(indexed_k_grid[:, self.top_k*ratio_now:self.top_k*2*ratio_now] * class1_v, dim=1, keepdim=True)
                    patch_cls2 = torch.sum(indexed_k_grid[:, self.top_k*2*ratio_now:self.top_k*3*ratio_now] * class2_v, dim=1, keepdim=True)
                    patch_cls3 = torch.sum(indexed_k_grid[:, self.top_k*3*ratio_now:] * class3_v, dim=1, keepdim=True)

                    #indexed_k_grid_ = indexed_k_grid.detach().cpu().numpy()
                    #patch_cls3_ = patch_cls3.detach().cpu().numpy()

                    template_k_values.append(indexed_k_grid)
                    template_k_labels.append([class0_v,class1_v,class2_v,class3_v])

                    if scale_i == 0:
                        k1_patch = torch.cat([patch_cls0, patch_cls1, patch_cls2, patch_cls3], dim=1)
                    elif scale_i == 1:
                        k2_patch = torch.cat([patch_cls0, patch_cls1, patch_cls2, patch_cls3], dim=1)
                    elif scale_i == 2:
                        k3_patch = torch.cat([patch_cls0, patch_cls1, patch_cls2, patch_cls3], dim=1)

                else:
                    f_indexed_k_grid = torch.gather(k_grid_list[scale_i][batch_i], 1,
                                                  template_k.expand(k_grid_list[scale_i].size(1), -1))
                    densecl_sim_k = torch.einsum('cn,ck->nk', [template_k_values[scale_i].clone().detach(), f_indexed_k_grid])
                    tag = torch.max(densecl_sim_k)
                    for idxx in range(template_k_values[scale_i].shape[1]):
                        if idxx == 0:
                            L1 = torch.argmax(densecl_sim_k[idxx], keepdim=True)
                            densecl_sim_k[:, L1] += -tag
                        else:
                            L1_ = torch.argmax(densecl_sim_k[idxx], keepdim=True)
                            L1 = torch.cat([L1, L1_])
                            densecl_sim_k[:, L1_] += -tag
                    template_q = template_q[L1]
                    template_k = template_k[L1]
                    indexed_q_grid = torch.gather(q_grid_list[scale_i][batch_i], 1,
                                                  template_q.expand(q_grid_list[scale_i].size(1), -1))  # NxCxS^2
                    indexed_k_grid = torch.gather(k_grid_list[scale_i][batch_i], 1,
                                                  template_k.expand(k_grid_list[scale_i].size(1), -1))  # NxCxS^2
                    densecl_sim_q = (indexed_q_grid * indexed_k_grid).sum(0)  # NxS^2
                    indexed_q_grid = indexed_q_grid.permute(1, 0)
                    cut_list.append(cut_list[-1] + len(L1))

                    class0_v, class1_v, class2_v, class3_v = template_k_labels[scale_i]
                    ratio_now = self.sample_ratio ** scale_i
                    patch_cls0 = torch.sum(indexed_k_grid[:, 0:self.top_k*ratio_now] * class0_v, dim = 1, keepdim=True)
                    patch_cls1 = torch.sum(indexed_k_grid[:, self.top_k*ratio_now:self.top_k*2*ratio_now] * class1_v, dim=1, keepdim=True)
                    patch_cls2 = torch.sum(indexed_k_grid[:, self.top_k*2*ratio_now:self.top_k*3*ratio_now] * class2_v, dim=1, keepdim=True)
                    patch_cls3 = torch.sum(indexed_k_grid[:, self.top_k*3*ratio_now:] * class3_v, dim=1, keepdim=True)
                    if scale_i == 0:
                        k1_patch = torch.cat([k1_patch, patch_cls0, patch_cls1, patch_cls2, patch_cls3], dim=1)
                    elif scale_i == 1:
                        k2_patch = torch.cat([k2_patch, patch_cls0, patch_cls1, patch_cls2, patch_cls3], dim=1)
                    elif scale_i == 2:
                        k3_patch = torch.cat([k3_patch, patch_cls0, patch_cls1, patch_cls2, patch_cls3], dim=1)

                if batch_i == 0 and scale_i == 0:
                    l_pos_dense = densecl_sim_q.unsqueeze(-1)
                    l_neg_dense = torch.einsum('nc,kc->nk', [indexed_q_grid, self.queue2[scale_i].clone().detach()])
                    l_dense_weight = torch.cat([class0_v, class1_v, class2_v, class3_v], dim=0) * self.top_k*ratio_now
                else:
                    l_pos_dense_add = densecl_sim_q.unsqueeze(-1)
                    l_neg_dense_add = torch.einsum('nc,kc->nk', [indexed_q_grid, self.queue2[scale_i].clone().detach()])
                    l_dense_weight_add = torch.cat([class0_v, class1_v, class2_v, class3_v], dim=0) * self.top_k*ratio_now
                    l_pos_dense = torch.cat([l_pos_dense, l_pos_dense_add], dim=0)
                    l_neg_dense = torch.cat([l_neg_dense, l_neg_dense_add], dim=0)
                    l_dense_weight = torch.cat([l_dense_weight, l_dense_weight_add], dim=0)
            cut_lists.append(cut_list)

        logits_dense, labels_dense = tensor2logit(l_pos_dense, l_neg_dense, l_dense_weight = l_dense_weight, dense_batch = batch_size, scale_num = len(q_grid_list), cut_lists = cut_lists)
        logits, labels = tensor2logit(l_pos, l_neg)

        return logits, labels, logits_dense, labels_dense, k1_patch, k2_patch, k3_patch

    def simclr_logits(self, y_i_sup, q_grid_list, k_grid_list, rot_k, filp, batch_size, r1, r2, r3, r4, fh_list, fw_list):

        filp = filp.detach().cpu().numpy()
        rot_k = rot_k.detach().cpu().numpy()
        template_k_values = []

        for batch_i in range(batch_size):
            for scale_i in range(len(q_grid_list)):
                template_q = torch.from_numpy(np.array(range(fh_list[scale_i] * fw_list[scale_i])).reshape([fh_list[scale_i], fw_list[scale_i]]))
                template_k = torch.from_numpy(np.array(range(fh_list[scale_i] * fw_list[scale_i])).reshape([fh_list[scale_i], fw_list[scale_i]]))
                if filp[batch_i] == 1:
                    template_q = torch.flip(template_q, dims=[0])
                template_q = torch.rot90(template_q, k=rot_k[batch_i], dims=[0, 1])
                template_q = template_q[int(r3[batch_i] * fh_list[scale_i]): int(r3[batch_i] * fh_list[scale_i] + r1[batch_i] * fh_list[scale_i]),
                             int(r4[batch_i] * fw_list[scale_i]): int(r4[batch_i] * fw_list[scale_i] + r2[batch_i] * fw_list[scale_i])]
                fh_q, fw_q = template_q.shape
                template_q = torch.reshape(template_q, (-1,)).long().cuda()
                template_k = torchvision.transforms.Resize([fh_q, fw_q], interpolation=torchvision.transforms.InterpolationMode.NEAREST)\
                    (template_k.unsqueeze(0))
                template_k = torch.reshape(template_k, (-1,)).long().cuda()
                # sample it
                if batch_i == 0:
                    y_i_sup_resize = torchvision.transforms.Resize([fh_q, fw_q],)(y_i_sup)
                    y_i_sup_resize = torch.reshape(y_i_sup_resize, (4,-1))
                    epss = torch.rand(y_i_sup_resize.shape).cuda() * 0.1
                    history = torch.zeros(y_i_sup_resize.shape).cuda()
                    if not self.Patch_sup:
                        y_i_sup_resize = torch.zeros(y_i_sup_resize.shape).cuda()
                    class0_v, class0_indice = torch.topk(y_i_sup_resize[0] + epss[0] + history[0], k=self.top_k*(self.sample_ratio ** scale_i), largest=True)
                    history[:,class0_indice] += -torch.max(y_i_sup_resize)
                    class1_v, class1_indice = torch.topk(y_i_sup_resize[1] + epss[1] + history[1], k=self.top_k*(self.sample_ratio ** scale_i), largest=True)
                    history[:,class1_indice] += -torch.max(y_i_sup_resize)
                    class2_v, class2_indice = torch.topk(y_i_sup_resize[2] + epss[2] + history[2], k=self.top_k*(self.sample_ratio ** scale_i), largest=True)
                    history[:,class2_indice] += -torch.max(y_i_sup_resize)
                    class3_v, class3_indice = torch.topk(y_i_sup_resize[3] + epss[3] + history[3], k=self.top_k*(self.sample_ratio ** scale_i), largest=True)
                    L1 = torch.cat([class0_indice, class1_indice, class2_indice, class3_indice])
                    template_q = template_q[L1]
                    template_k = template_k[L1]
                    indexed_q_grid = torch.gather(q_grid_list[scale_i][batch_i], 1,
                                                  template_q.expand(q_grid_list[scale_i].size(1), -1))  # NxCxS^2
                    indexed_k_grid = torch.gather(k_grid_list[scale_i][batch_i], 1,
                                                  template_k.expand(k_grid_list[scale_i].size(1), -1))  # NxCxS^2
                    template_k_values.append(indexed_k_grid)
                else:
                    f_indexed_k_grid = torch.gather(k_grid_list[scale_i][batch_i], 1,
                                                  template_k.expand(k_grid_list[scale_i].size(1), -1))
                    densecl_sim_k = torch.einsum('cn,ck->nk', [template_k_values[scale_i].clone().detach(), f_indexed_k_grid]).detach()
                    tag = torch.max(densecl_sim_k)
                    for idxx in range(template_k_values[scale_i].shape[1]):
                        if idxx == 0:
                            L1 = torch.argmax(densecl_sim_k[idxx], keepdim=True)
                            densecl_sim_k[:, L1] += -tag
                        else:
                            L1_ = torch.argmax(densecl_sim_k[idxx], keepdim=True)
                            L1 = torch.cat([L1, L1_])
                            densecl_sim_k[:, L1_] += -tag
                    template_q = template_q[L1]
                    template_k = template_k[L1]
                    indexed_q_grid = torch.gather(q_grid_list[scale_i][batch_i], 1,
                                                  template_q.expand(q_grid_list[scale_i].size(1), -1))  # NxCxS^2
                    indexed_k_grid = torch.gather(k_grid_list[scale_i][batch_i], 1,
                                                  template_k.expand(k_grid_list[scale_i].size(1), -1))  # NxCxS^2

                if batch_i == 0 and scale_i == 0:
                    q_dense1 = indexed_q_grid
                    k_dense1 = indexed_k_grid
                elif batch_i == 0 and scale_i == 1:
                    q_dense2 = indexed_q_grid
                    k_dense2 = indexed_k_grid
                elif batch_i == 0 and scale_i == 2:
                    q_dense3 = indexed_q_grid
                    k_dense3 = indexed_k_grid
                elif batch_i > 0 and scale_i == 0:
                    q_dense1 = torch.cat((q_dense1, indexed_q_grid), dim=1)
                    k_dense1 = torch.cat((k_dense1, indexed_k_grid), dim=1)
                elif batch_i > 0 and scale_i == 1:
                    q_dense2 = torch.cat((q_dense2, indexed_q_grid), dim=1)
                    k_dense2 = torch.cat((k_dense2, indexed_k_grid), dim=1)
                elif batch_i > 0 and scale_i == 2:
                    q_dense3 = torch.cat((q_dense3, indexed_q_grid), dim=1)
                    k_dense3 = torch.cat((k_dense3, indexed_k_grid), dim=1)
        q_dense1 = q_dense1.permute(1, 0)
        k_dense1 = k_dense1.permute(1, 0)
        q_dense2 = q_dense2.permute(1, 0)
        k_dense2 = k_dense2.permute(1, 0)
        q_dense3 = q_dense3.permute(1, 0)
        k_dense3 = k_dense3.permute(1, 0)
        return q_dense1, k_dense1, q_dense2, k_dense2, q_dense3, k_dense3

    # return id
    def get_label(self, ids):
        ids_label = []
        size_label = []
        for ids_each in ids:
            if ids_each not in self.info_ids.keys():
                self.info_ids[ids_each] = len(self.info_ids.keys())
            ids_label.append(self.info_ids[ids_each])
            size_ids_each = ids_each.split('_')[0]
            if size_ids_each not in self.info_size.keys():
                self.info_size[size_ids_each] = len(self.info_size.keys())
            size_label.append(self.info_size[size_ids_each])
        return np.array(ids_label), np.array(size_label)

    def forward(self, x_q, x_k, y_i_sup, ids, rot_k, filp, r1, r2, r3, r4):
        '''
        x_q = x_q[0:1]
        x_k = x_k[0:1]
        y_i_sup = y_i_sup[0:1]
        ids = ids[0:1]
        rot_k = rot_k[0:1]
        filp = filp[0:1]
        r1 = r1[0:1]
        r2 = r2[0:1]
        r3 = r3[0:1]
        r4 = r4[0:1]
        '''

        if self.selfmode == 'moco':
            ids_label, size_label = self.get_label(ids)
            batch_size, _, _, _ = x_q.shape

            q_b = self.encoder_q.encode(x_q)  # backbone features
            q, qup1_grid, qup2_grid, qup3_grid = self.encoder_q.decode(q_b)  # queries: NxC; NxCxS^2

            _, _, f1h, f1w = q_b[1].shape
            _, _, f2h, f2w = q_b[2].shape
            _, _, f3h, f3w = q_b[3].shape

            q = nn.functional.normalize(q, dim=1)
            qup1_grid = nn.functional.normalize(qup1_grid, dim=1)
            qup2_grid = nn.functional.normalize(qup2_grid, dim=1)
            qup3_grid = nn.functional.normalize(qup3_grid, dim=1)

            shuffled_idxs, reverse_idxs = self.shuffled_idx(batch_size)

            with torch.no_grad():
                # Update the key encoder
                self.momentum_update()

                # Shuffle minibatch
                x_k = x_k[shuffled_idxs]

                # Feature representations of the shuffled key view from the key encoder
                k_b = self.encoder_k.encode(x_k)
                k, kup1_grid, kup2_grid, kup3_grid = self.encoder_k.decode(k_b)  # keys: NxC; NxCxS^2

                k = nn.functional.normalize(k, dim=1)
                kup1_grid = nn.functional.normalize(kup1_grid, dim=1)
                kup2_grid = nn.functional.normalize(kup2_grid, dim=1)
                kup3_grid = nn.functional.normalize(kup3_grid, dim=1)

                k = k[reverse_idxs]
                kup1_grid = kup1_grid[reverse_idxs]
                kup2_grid = kup2_grid[reverse_idxs]
                kup3_grid = kup3_grid[reverse_idxs]

            # Compute the logits for the InfoNCE contrastive loss.
            logits, labels, logits_dense, labels_dense, k1_patch, k2_patch, k3_patch = \
                self.InfoNCE_logits(q, k, y_i_sup, [qup1_grid, qup2_grid, qup3_grid], [kup1_grid, kup2_grid, kup3_grid],
                                    size_label, rot_k, filp, batch_size, r1, r2, r3, r4,
                                    [f1h, f2h, f3h], [f1w, f2w, f3w])

            # Update the queue/memory with the current key_encoder minibatch.
            k1_patch = k1_patch.transpose(1, 0).reshape([batch_size, 4, -1])
            k2_patch = k2_patch.transpose(1, 0).reshape([batch_size, 4, -1])
            k3_patch = k3_patch.transpose(1, 0).reshape([batch_size, 4, -1])

            self.update_queue(k, [k1_patch, k2_patch, k3_patch], size_label, )

            return logits, labels, logits_dense, labels_dense

        elif self.selfmode == 'simclr':
            batch_size, _, _, _ = x_q.shape

            q_b, k_b = self.encoder_q.encode(x_q), self.encoder_q.encode(x_k)  # backbone features
            q, qup1_grid, qup2_grid, qup3_grid = self.encoder_q.decode(q_b)  # queries: NxC; NxCxS^2
            k, kup1_grid, kup2_grid, kup3_grid = self.encoder_q.decode(k_b)  # keys: NxC; NxCxS^2

            _, _, f1h, f1w = q_b[1].shape
            _, _, f2h, f2w = q_b[2].shape
            _, _, f3h, f3w = q_b[3].shape

            # Compute the logits for the InfoNCE contrastive loss.

            q_dense1, k_dense1, q_dense2, k_dense2, q_dense3, k_dense3 = \
                self.simclr_logits(y_i_sup, [qup1_grid, qup2_grid, qup3_grid], [kup1_grid, kup2_grid, kup3_grid],
                                    rot_k, filp, batch_size, r1, r2, r3, r4,
                                    [f1h, f2h, f3h], [f1w, f2w, f3w])

            return q, k, q_dense1, k_dense1, q_dense2, k_dense2, q_dense3, k_dense3

class Denseproj_UNET_MLP(nn.Module):
    def __init__(self, channel=[64, 128, 256, 512], hidfea_num = 128, confea_num = 64, multihead = 4, normName = ['BN']):
        super(Denseproj_UNET_MLP, self).__init__()
        self.normName = normName
        n_hide = hidfea_num
        n_out = confea_num

        self.avg_pooling = torch.nn.AdaptiveAvgPool2d((1, 1))
        if normName[1] == 'gBN':
            self.mlp = nn.Sequential(
                nn.Linear(channel[-1], min(n_hide, channel[-1])),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(min(n_hide, channel[-1])),  ## haha
                nn.Linear(min(n_hide, channel[-1]), n_out))
        else:
            self.mlp = nn.Sequential(
                nn.Linear(channel[-1], min(n_hide, channel[-1])),
                nn.ReLU(inplace=True),
                nn.Linear(min(n_hide, channel[-1]), n_out))

        self.conv_x4up1_conv1 = nn.Conv2d(channel[-2], min(n_hide, channel[-2]), 1, groups=multihead)
        if normName[0] == 'BN':
            self.conv_x4up1_norm1 = nn.BatchNorm2d(min(n_hide, channel[-2]))
        elif normName[0] == 'LN':
            self.conv_x4up1_norm1 = nn.LayerNorm([min(n_hide, channel[-2])])
        self.conv_x4up1_act = nn.ReLU(inplace=True)
        self.conv_x4up1_conv2 = nn.Conv2d(min(n_hide, channel[-2]), n_out, 1, groups=multihead)

        self.conv_x4up2_conv1 = nn.Conv2d(channel[-3], min(n_hide, channel[-3]), 1, groups=multihead)
        if normName[0] == 'BN':
            self.conv_x4up2_norm1 = nn.BatchNorm2d(min(n_hide, channel[-3]))
        elif normName[0] == 'LN':
            self.conv_x4up2_norm1 = nn.LayerNorm([min(n_hide, channel[-3])])
        self.conv_x4up2_act = nn.ReLU(inplace=True)
        self.conv_x4up2_conv2 = nn.Conv2d(min(n_hide, channel[-3]), n_out, 1, groups=multihead)

        self.conv_x4up3_conv1 = nn.Conv2d(channel[-4], min(n_hide, channel[-4]), 1, groups=multihead)
        if normName[0] == 'BN':
            self.conv_x4up3_norm1 = nn.BatchNorm2d(min(n_hide, channel[-4]))
        elif normName[0] == 'LN':
            self.conv_x4up3_norm1 = nn.LayerNorm([min(n_hide, channel[-4])])
        self.conv_x4up3_act = nn.ReLU(inplace=True)
        self.conv_x4up3_conv2 = nn.Conv2d(min(n_hide, channel[-4]), n_out, 1, groups=multihead)

    def forward(self, xlist):
        x4, x4up1, x4up2, x4up3 = xlist

        B, _, _, _ = x4.size()

        avgpooled_x = self.avg_pooling(x4).view(B, -1)
        avgpooled_x = self.mlp(avgpooled_x)

        x4up1 = self.conv_x4up1_conv1(x4up1)
        if self.normName[0] == 'BN':
            x4up1 = self.conv_x4up1_norm1(x4up1)
        elif self.normName[0] == 'LN':
            B, cc, H, W = x4up1.shape
            x4up1 = x4up1.view(B, cc, H*W).permute(0, 2, 1)
            x4up1 = self.conv_x4up1_norm1(x4up1)
            x4up1 = x4up1.permute(0, 2, 1).view(B, cc, H, W)
        x4up1 = self.conv_x4up1_act(x4up1)
        x4up1 = self.conv_x4up1_conv2(x4up1)
        x4up1 = x4up1.view(x4up1.size(0), x4up1.size(1), -1) # bxdxs^2

        x4up2 = self.conv_x4up2_conv1(x4up2)
        if self.normName[0] == 'BN':
            x4up2 = self.conv_x4up2_norm1(x4up2)
        elif self.normName[0] == 'LN':
            B, cc, H, W = x4up2.shape
            x4up2 = x4up2.view(B, cc, H * W).permute(0, 2, 1)
            x4up2 = self.conv_x4up2_norm1(x4up2)
            x4up2 = x4up2.permute(0, 2, 1).view(B, cc, H, W)
        x4up2 = self.conv_x4up2_act(x4up2)
        x4up2 = self.conv_x4up2_conv2(x4up2)
        x4up2 = x4up2.view(x4up2.size(0), x4up2.size(1), -1) # bxdxs^2

        x4up3 = self.conv_x4up3_conv1(x4up3)
        if self.normName[0] == 'BN':
            x4up3 = self.conv_x4up3_norm1(x4up3)
        elif self.normName[0] == 'LN':
            B, cc, H, W = x4up3.shape
            x4up3 = x4up3.view(B, cc, H * W).permute(0, 2, 1)
            x4up3 = self.conv_x4up3_norm1(x4up3)
            x4up3 = x4up3.permute(0, 2, 1).view(B, cc, H, W)
        x4up3 = self.conv_x4up3_act(x4up3)
        x4up3 = self.conv_x4up3_conv2(x4up3)
        x4up3 = x4up3.view(x4up3.size(0), x4up3.size(1), -1) # bxdxs^2

        return [avgpooled_x, x4up1, x4up2, x4up3]

def resnet18(**kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        args: arguments
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

'''
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, droprate = None, size = 0):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            #nn.LayerNorm([size, size]),
            #nn.ReLU(inplace=True),  ## haha
            nn.LeakyReLU(inplace=True), ## haha
            #nn.Dropout(p=droprate), # new
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            #nn.LayerNorm([size, size]),
            #nn.ReLU(inplace=True), ## haha
            nn.LeakyReLU(inplace=True), ## haha
        )

    def forward(self, x):
        return self.double_conv(x)
'''

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, normName = ['BN','LN'], droprate = None, size = 0):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.normName = normName
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.con1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        if normName[0] == 'BN':
            self.norm1 = nn.BatchNorm2d(mid_channels)
        elif normName[0] == 'LN':
            self.norm1 = nn.LayerNorm([mid_channels])
        self.act1 = nn.LeakyReLU(inplace=True)
        self.con2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        if normName[1] == 'BN':
            self.norm2 = nn.BatchNorm2d(out_channels)
        elif normName[1] == 'LN':
            self.norm2 = nn.LayerNorm([out_channels])
        self.act2 = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        B, _, H, W = x.shape
        x = self.con1(x)
        if self.normName[0] == 'BN':
            x = self.norm1(x)
        elif self.normName[0] == 'LN':
            x = x.view(B, self.mid_channels, H*W).permute(0, 2, 1)
            x = self.norm1(x)
            x = x.permute(0, 2, 1).view(B, self.mid_channels, H, W)
        x = self.act1(x)
        x = self.con2(x)
        if self.normName[1] == 'BN':
            x = self.norm2(x)
        elif self.normName[1] == 'LN':
            x = x.view(B, self.out_channels, H*W).permute(0, 2, 1)
            x = self.norm2(x)
            x = x.permute(0, 2, 1).view(B, self.out_channels, H, W)
        x = self.act2(x)
        return x

class SingleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, droprate = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, ConvBlock = DoubleConv, droprate = 0.5, size = 0, normName =[]):
        super().__init__()
        self.down = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_channels, out_channels, droprate = droprate, size = size, normName = normName)

    def forward(self, x):
        return self.conv(self.down(x))

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, ConvBlock = SingleConv, droprate = 0.5, size = 0, normName =[]):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        bilinear = False
        if bilinear:
            #self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                    nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, padding=0, bias=False))
            self.conv = ConvBlock(in_channels, out_channels, droprate = droprate, size = size, normName = normName)
        else:
            #self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=3, stride=2, padding=1)
            self.conv = ConvBlock(in_channels, out_channels, droprate = droprate, size = size, normName = normName)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.fc1 = nn.Linear(in_channels, out_channels)
        #self.relu1 = nn.ReLU(inplace=True)
        #self.fc2 = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        #x = self.relu1(self.fc1(x))
        #x = self.fc2(x)
        x = self.fc1(x)
        x = x.permute(0, 3, 1, 2)
        return x

class UNet_decode(nn.Module):
    def __init__(self, drop, channel, n_classes = 4, bilinear=False, droprate = 0.5):
        super(UNet_decode, self).__init__()
        self.outc = OutConv(channel[0], n_classes)

    def forward(self, xlist):
        x4, x4up1, x4up2, x4up3 = xlist
        x = self.outc(x4up3)
        return x

class UNet_encode(nn.Module):
    def __init__(self, drop, channel, n_channels = 3, n_classes = 4, bilinear=False, droprate = 0, IncNorm = ['',''],
                 DownNorm = ['',''], UpNorm = ['','']):
        super(UNet_encode, self).__init__()
        ConvBlock = DoubleConv
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = ConvBlock(n_channels, channel[0], droprate=drop, size=376, normName = IncNorm)
        self.down1 = Down(channel[0], channel[1], ConvBlock, droprate = drop, size = 188, normName = DownNorm)
        self.down2 = Down(channel[1], channel[2], ConvBlock, droprate = drop, size = 94, normName = DownNorm)
        self.down3 = Down(channel[2], channel[3], ConvBlock, droprate = drop, size = 47, normName = DownNorm)
        self.up1 = Up(channel[3], channel[2], bilinear, ConvBlock, droprate = drop, size = 94, normName = UpNorm)
        self.up2 = Up(channel[2], channel[1], bilinear, ConvBlock, droprate = drop, size = 188, normName = UpNorm)
        self.up3 = Up(channel[1], channel[0], bilinear, ConvBlock, droprate = drop, size = 376, normName = UpNorm)

    def forward(self, x, sample = True):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x4up1 = self.up1(x4, x3)
        x4up2 = self.up2(x4up1, x2)
        x4up3 = self.up3(x4up2, x1)

        return [x4, x4up1, x4up2, x4up3]

class UNet(nn.Module):
    def __init__(self, drop, channel, n_channels = 3, n_classes = 4, bilinear=False, droprate = 0, IncNorm = ['',''],
                 DownNorm = ['',''], UpNorm = ['','']):
        super(UNet, self).__init__()
        self.encode = UNet_encode(drop, channel, n_channels = n_channels, n_classes = n_classes, bilinear=bilinear, droprate = droprate,
                                  IncNorm = IncNorm, DownNorm = DownNorm, UpNorm = UpNorm)
        self.decode = UNet_decode(drop, channel, n_classes = n_classes, bilinear=bilinear, droprate = droprate)

    def forward(self, x, sample = True):
        [x4, x4up1, x4up2, x4up3] = self.encode(x)
        x = self.decode([x4, x4up1, x4up2, x4up3])
        #return x, [x4, x4up1, x4up2, x4up3]
        return x
