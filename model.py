# -*- coding: utf-8 -*-
"""
PSCL Model Architecture

Contains:
1. UNet implementation with modifications for contrastive learning
2. MoCo (Momentum Contrast) framework
3. Projection heads for contrastive learning (now in file 'networks.py')
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision
from networks import Denseproj_UNET_MLP, UNet_encode, UNet_decode

# Our PSCL framework with dense contrastive learning
# Its name is "MoCo_DenseModel", but in fact we implement the PSCL-MoCo and PSCL-SimCLR
# in the paper through hyperparameters "selfmode": "moco" or "simclr"
class MoCo_DenseModel(nn.Module):
    """Our PSCL framework with dense contrastive learning

    Key features:
    - Momentum encoder for stable contrastive learning
    - Multi-scale dense contrastive learning
    - Memory bank for samples
    """
    def __init__(self, backbone, queue_size=65536, momentum=0.999, temperature=0.07,
                 lab_size_decay=0.2, patch_num = 4, drop = 0, channel = 0,
                 self_scale_weight = [1, 1, 1], sample_ratio = 4, confea_num = 64, hidfea_num = 128, top_k = 15,
                 patch_lab_decay = 0.2, patch_size_decay = 0.2, multihead = 4, IncNorm =[], DownNorm=[], UpNorm=[], HeadNrom = [],
                 Patch_sup = True, selfmode = 'moco'):

        super(MoCo_DenseModel, self).__init__()

        self.multihead = multihead # number of multihead
        self.Patch_sup = Patch_sup # supervised assisted patch-sampling strategy or normal sampling
        self.selfmode = selfmode # moco: PSCL-MoCo; simclr: PSCL-SimCLR

        self.patch_num = patch_num # the number of patches for contrast learning
        self.self_scale_weight = self_scale_weight # weights of different scales in multi-scale Strategy
        self.sample_ratio = sample_ratio  # sample ratio
        self.top_k = top_k # the supervised patch number

        self.queue_size = queue_size # queue size to save history features
        self.momentum = momentum # it is used for updating parameters
        self.temperature = temperature # it is used to control smoothness

        # These dictionaries and numpy matrix are used to store production batch of "metallographic images".
        # The image filename consists of ProductionBatch(size)-sampleID(ids)-x-x.
        self.info_ids = dict()
        self.info_size = dict()
        self.lab_size = np.zeros((queue_size, 6))  # 6 means the six different production batches for "metallographic images".
        self.patch_size = np.zeros((queue_size, 6)) # Specifically, the holding time form 15/30 min, 45/60 and 90/120 min
        # We expect the same production batch to have higher correlation with label decay in for contrast learning
        self.lab_size_decay = lab_size_decay # global decay
        self.patch_lab_decay = patch_lab_decay # local decay
        self.patch_size_decay = patch_size_decay # local decay

        # Load model; the backbone is actually UNet
        # Query Encoder
        self.encoder_q = backbone(drop=drop, channel=channel, IncNorm=IncNorm, DownNorm=DownNorm, UpNorm=UpNorm)
        # Key Encoder
        self.encoder_k = backbone(drop=drop, channel=channel, IncNorm=IncNorm, DownNorm=DownNorm, UpNorm=UpNorm)

        # Add the head funtion for contrastive learning
        # It is executed after backbone(UNet)
        self.encoder_q.decode = Denseproj_UNET_MLP(channel=channel, hidfea_num = hidfea_num, confea_num = confea_num, multihead = multihead, normName = HeadNrom)
        self.encoder_k.decode = Denseproj_UNET_MLP(channel=channel, hidfea_num = hidfea_num, confea_num = confea_num, multihead = multihead, normName = HeadNrom)

        # Parameter initialization mechanism
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # Create the queue to store negative samples
        # queue is the set to save global features of samples;
        # queue2 is the set to save lacal(patch) features, we use multi-scale strategy, so it has an additional dimensions
        self.register_buffer("queue", torch.randn(self.queue_size, confea_num))
        self.register_buffer("queue2", torch.randn(len(self_scale_weight), self.queue_size * patch_num, confea_num))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long)) # queue_ptr is used to erase obsolete samples

        # patch_labels is used to record the category of supervised patches
        # and to assist in selecting unsupervised samples for comparative learning
        patch_labels = []
        self.patch_lab_tag = torch.zeros(patch_num*self.queue_size).cuda()
        for jj in range(4): # there are 4 kinds of microstructures in "metallographic images"
            c1_label = torch.from_numpy(np.array([1, 0, 0, 0])).repeat(top_k * (sample_ratio **jj), self.queue_size).cuda()
            c2_label = torch.from_numpy(np.array([0, 1, 0, 0])).repeat(top_k * (sample_ratio **jj), self.queue_size).cuda()
            c3_label = torch.from_numpy(np.array([0, 0, 1, 0])).repeat(top_k * (sample_ratio **jj), self.queue_size).cuda()
            c4_label = torch.from_numpy(np.array([0, 0, 0, 1])).repeat(top_k * (sample_ratio **jj), self.queue_size).cuda()
            s_label = torch.cat([c1_label, c2_label, c3_label, c4_label], dim=0)
            patch_labels.append(s_label)
        self.patch_labels = patch_labels

    # Update the key_encoder parameters
    @torch.no_grad()
    def momentum_update(self):
        '''
        Update the key_encoder parameters through the momentum update:
        key_params = momentum * key_params + (1 - momentum) * query_params
        '''

        # For each of the parameters in each encoder
        for p_q, p_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            p_k.data = p_k.data * self.momentum + p_q.detach().data * (1. - self.momentum)

    # Generation of the shuffled indexes
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

    # update queue by the current samples
    @torch.no_grad()
    def update_queue(self, k, k_patch_list, size_label):
        # k: global feature; k_patch_list: local feature; size_label: procession batch
        batch_size = k.size(0)
        ptr = int(self.queue_ptr)
        # replace the keys at ptr (dequeue and enqueue)
        self.patch_lab_tag[ptr*self.patch_num :(ptr + batch_size)*self.patch_num] = 1
        # 'self.queue' is the global feature queue
        self.queue[ptr:ptr + batch_size, :] = k
        # 'self.queue2' is the local feature queue with multiple scales
        for batch_size_i in range(batch_size):
            for scale_i in range(len(k_patch_list)):
                self.queue2[scale_i][(ptr + batch_size_i) * self.patch_num :(ptr + batch_size_i + 1) * self.patch_num] \
                    = k_patch_list[scale_i][batch_size_i]
        # 'size_label' is specific for "metallographic images" indicating procession batch
        self.lab_size[ptr:ptr + batch_size, :] = np.eye(6)[size_label]
        self.patch_size[ptr:ptr + batch_size, :] = np.eye(6)[size_label]
        # updata ptr
        ptr = (ptr + batch_size) % self.queue_size

        # Store queue pointer as register_buffer
        self.queue_ptr[0] = ptr

        # weight decay for labels
        self.lab_size = self.lab_size * self.lab_size_decay
        self.patch_size = self.patch_size * self.patch_size_decay
        self.patch_lab_tag = self.patch_lab_tag * self.patch_lab_decay

    # InfoNCE_logits is used for PSCL-MoCo contrastive learning
    def InfoNCE_logits(self, q, k, y_i_sup, q_grid_list, k_grid_list, size_label, rot_k, filp, batch_size, r1, r2, r3, r4, fh_list, fw_list):

        def tensor2logit(pos, neg, l_dense_weight = 0, dense_batch = 0, scale_num = 0, cut_lists = []):
            logits = torch.cat((pos, neg), dim=1) # Concatenate pos and neg
            logits /= self.temperature # Apply temperature scaling
            # Initialize labels (all zeros)
            labels = torch.zeros(logits.shape, dtype=torch.float).cuda()
            labels[:, 0] = 1 # Positive samples are labeled 1
            # Handle dense labels if needed
            if dense_batch == 0:
                # global contrastive learning
                label_size = np.eye(6)[size_label]
                label_add = np.matmul(label_size, self.lab_size.T) # Introducing production batch dependencies
                labels[:, 1:] = torch.from_numpy(label_add)
            else:
                # local(patch) contrastive learning
                label_size = np.eye(6)[size_label]
                label_add = np.matmul(label_size, self.patch_size.T)
                label_add = np.repeat(label_add, self.patch_num, axis=1)
                label_add = torch.from_numpy(label_add).cuda()
                # Introducing production batch dependencies for each patch
                for i_b in range(dense_batch):
                    for j_s in range(scale_num):
                        labels[cut_lists[i_b][j_s]:cut_lists[i_b][j_s + 1], 1:self.queue_size * self.patch_num + 1] += self.patch_labels[j_s] * label_add[i_b]
                        labels[cut_lists[i_b][j_s]:cut_lists[i_b][j_s + 1], 1:self.queue_size*self.patch_num+1] += self.patch_labels[j_s] * self.patch_lab_tag
                        labels[cut_lists[i_b][j_s]:cut_lists[i_b][j_s + 1], :] *= self.self_scale_weight[j_s]
                labels = labels * l_dense_weight.unsqueeze(-1).detach() # weight new
            return logits, labels

        # Global contrastive learning (original MoCo)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)  # Positive pairs: q and k
        l_neg = torch.einsum('nc,kc->nk', [q, self.queue.clone().detach()]) # Negative pairs: q and queue

        # Convert augmentation parameters to numpy
        filp = filp.detach().cpu().numpy()
        rot_k = rot_k.detach().cpu().numpy()
        # Initialize lists for dense contrastive learning
        cut_lists = [] #  Store indices for different scales/batches
        template_k_values = []
        template_k_labels = []

        # supervised assisted patch-sampling strategy
        # Process each sample in batch
        for batch_i in range(batch_size):
            cut_list = [] # Indices for current sample
            if batch_i == 0:
                cut_list.append(0)
            else:
                cut_list.append(cut_lists[-1][-1])
            # Process each scale
            for scale_i in range(len(q_grid_list)):
                # Create template grids for current scale
                template_q = torch.from_numpy(np.array(range(fh_list[scale_i] * fw_list[scale_i])).reshape([fh_list[scale_i], fw_list[scale_i]]))
                template_k = torch.from_numpy(np.array(range(fh_list[scale_i] * fw_list[scale_i])).reshape([fh_list[scale_i], fw_list[scale_i]]))
                # Apply augmentations to query template
                if filp[batch_i] == 1:
                    template_q = torch.flip(template_q, dims=[0])
                template_q = torch.rot90(template_q, k=rot_k[batch_i], dims=[0, 1])
                # Crop template based on region parameters
                template_q = template_q[int(r3[batch_i] * fh_list[scale_i]): int(r3[batch_i] * fh_list[scale_i] + r1[batch_i] * fh_list[scale_i]),
                             int(r4[batch_i] * fw_list[scale_i]): int(r4[batch_i] * fw_list[scale_i] + r2[batch_i] * fw_list[scale_i])]
                fh_q, fw_q = template_q.shape
                # Flatten and move to GPU
                template_q = torch.reshape(template_q, (-1,)).long().cuda()
                template_k = torchvision.transforms.Resize([fh_q, fw_q], interpolation=torchvision.transforms.InterpolationMode.NEAREST)\
                    (template_k.unsqueeze(0))
                template_k = torch.reshape(template_k, (-1,)).long().cuda()
                # supervised assisted patch-sampling strategy (core)
                # Sample patches based on supervised labels (first sample in batch only)
                if batch_i == 0:
                    # Resize supervised labels to current scale
                    y_i_sup_resize = torchvision.transforms.Resize([fh_q, fw_q],)(y_i_sup)
                    y_i_sup_resize = torch.reshape(y_i_sup_resize, (4,-1))
                    # Add small noise for sampling
                    epss = torch.rand(y_i_sup_resize.shape).cuda() * 0.1
                    history = torch.zeros(y_i_sup_resize.shape).cuda()
                    if not self.Patch_sup:
                        y_i_sup_resize = torch.zeros(y_i_sup_resize.shape).cuda()
                    # Sample top-k patches for each class
                    class0_v, class0_indice = torch.topk(y_i_sup_resize[0] + epss[0] + history[0], k=self.top_k*(self.sample_ratio ** scale_i), largest=True)
                    history[:,class0_indice] += -torch.max(y_i_sup_resize)
                    class1_v, class1_indice = torch.topk(y_i_sup_resize[1] + epss[1] + history[1], k=self.top_k*(self.sample_ratio ** scale_i), largest=True)
                    history[:,class1_indice] += -torch.max(y_i_sup_resize)
                    class2_v, class2_indice = torch.topk(y_i_sup_resize[2] + epss[2] + history[2], k=self.top_k*(self.sample_ratio ** scale_i), largest=True)
                    history[:,class2_indice] += -torch.max(y_i_sup_resize)
                    class3_v, class3_indice = torch.topk(y_i_sup_resize[3] + epss[3] + history[3], k=self.top_k*(self.sample_ratio ** scale_i), largest=True)
                    # Combine indices from all classes
                    L1 = torch.cat([class0_indice, class1_indice, class2_indice, class3_indice])
                    # Get corresponding query and key features
                    template_q = template_q[L1]
                    template_k = template_k[L1]
                    # Gather features using sampled indices
                    indexed_q_grid = torch.gather(q_grid_list[scale_i][batch_i], 1,
                                                  template_q.expand(q_grid_list[scale_i].size(1), -1))  # NxCxS^2
                    indexed_k_grid = torch.gather(k_grid_list[scale_i][batch_i], 1,
                                                  template_k.expand(k_grid_list[scale_i].size(1), -1))  # NxCxS^2
                    # Compute dense similarity
                    densecl_sim_q = (indexed_q_grid * indexed_k_grid).sum(0)  # NxS^2
                    indexed_q_grid = indexed_q_grid.permute(1, 0)
                    cut_list.append(cut_list[-1] + len(L1)) # Update indices
                    # Normalize class weights
                    class0_v = class0_v / torch.sum(class0_v)
                    class1_v = class1_v / torch.sum(class1_v)
                    class2_v = class2_v / torch.sum(class2_v)
                    class3_v = class3_v / torch.sum(class3_v)
                    # Compute weighted patch features
                    ratio_now = self.sample_ratio ** scale_i
                    patch_cls0 = torch.sum(indexed_k_grid[:, 0:self.top_k*ratio_now] * class0_v, dim = 1, keepdim=True)
                    patch_cls1 = torch.sum(indexed_k_grid[:, self.top_k*ratio_now:self.top_k*2*ratio_now] * class1_v, dim=1, keepdim=True)
                    patch_cls2 = torch.sum(indexed_k_grid[:, self.top_k*2*ratio_now:self.top_k*3*ratio_now] * class2_v, dim=1, keepdim=True)
                    patch_cls3 = torch.sum(indexed_k_grid[:, self.top_k*3*ratio_now:] * class3_v, dim=1, keepdim=True)
                    # Store template values and labels
                    template_k_values.append(indexed_k_grid)
                    template_k_labels.append([class0_v,class1_v,class2_v,class3_v])
                    # Store patch features by scale
                    if scale_i == 0:
                        k1_patch = torch.cat([patch_cls0, patch_cls1, patch_cls2, patch_cls3], dim=1)
                    elif scale_i == 1:
                        k2_patch = torch.cat([patch_cls0, patch_cls1, patch_cls2, patch_cls3], dim=1)
                    elif scale_i == 2:
                        k3_patch = torch.cat([patch_cls0, patch_cls1, patch_cls2, patch_cls3], dim=1)

                else:
                    # For subsequent samples in batch, sampling patches to the labeled templates
                    f_indexed_k_grid = torch.gather(k_grid_list[scale_i][batch_i], 1,
                                                  template_k.expand(k_grid_list[scale_i].size(1), -1))
                    # Using similarity for sampling
                    densecl_sim_k = torch.einsum('cn,ck->nk', [template_k_values[scale_i].clone().detach(), f_indexed_k_grid])
                    tag = torch.max(densecl_sim_k)
                    # Sample matching indices
                    for idxx in range(template_k_values[scale_i].shape[1]):
                        if idxx == 0:
                            L1 = torch.argmax(densecl_sim_k[idxx], keepdim=True)
                            densecl_sim_k[:, L1] += -tag
                        else:
                            L1_ = torch.argmax(densecl_sim_k[idxx], keepdim=True)
                            L1 = torch.cat([L1, L1_])
                            densecl_sim_k[:, L1_] += -tag
                    # Get matched features
                    template_q = template_q[L1]
                    template_k = template_k[L1]
                    indexed_q_grid = torch.gather(q_grid_list[scale_i][batch_i], 1,
                                                  template_q.expand(q_grid_list[scale_i].size(1), -1))  # NxCxS^2
                    indexed_k_grid = torch.gather(k_grid_list[scale_i][batch_i], 1,
                                                  template_k.expand(k_grid_list[scale_i].size(1), -1))  # NxCxS^2
                    # Compute similarities
                    densecl_sim_q = (indexed_q_grid * indexed_k_grid).sum(0)  # NxS^2
                    indexed_q_grid = indexed_q_grid.permute(1, 0)
                    cut_list.append(cut_list[-1] + len(L1))
                    # Get class weights
                    class0_v, class1_v, class2_v, class3_v = template_k_labels[scale_i]
                    ratio_now = self.sample_ratio ** scale_i
                    # Compute weighted patch features
                    patch_cls0 = torch.sum(indexed_k_grid[:, 0:self.top_k*ratio_now] * class0_v, dim = 1, keepdim=True)
                    patch_cls1 = torch.sum(indexed_k_grid[:, self.top_k*ratio_now:self.top_k*2*ratio_now] * class1_v, dim=1, keepdim=True)
                    patch_cls2 = torch.sum(indexed_k_grid[:, self.top_k*2*ratio_now:self.top_k*3*ratio_now] * class2_v, dim=1, keepdim=True)
                    patch_cls3 = torch.sum(indexed_k_grid[:, self.top_k*3*ratio_now:] * class3_v, dim=1, keepdim=True)
                    # Accumulate patch features
                    if scale_i == 0:
                        k1_patch = torch.cat([k1_patch, patch_cls0, patch_cls1, patch_cls2, patch_cls3], dim=1)
                    elif scale_i == 1:
                        k2_patch = torch.cat([k2_patch, patch_cls0, patch_cls1, patch_cls2, patch_cls3], dim=1)
                    elif scale_i == 2:
                        k3_patch = torch.cat([k3_patch, patch_cls0, patch_cls1, patch_cls2, patch_cls3], dim=1)
                # Build dense contrastive learning pairs
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

        # Generate final logits and labels
        logits_dense, labels_dense = tensor2logit(l_pos_dense, l_neg_dense, l_dense_weight = l_dense_weight, dense_batch = batch_size, scale_num = len(q_grid_list), cut_lists = cut_lists)
        logits, labels = tensor2logit(l_pos, l_neg)

        return logits, labels, logits_dense, labels_dense, k1_patch, k2_patch, k3_patch

    # simclr_logits is used for PSCL-SimCLR contrastive learning
    def simclr_logits(self, y_i_sup, q_grid_list, k_grid_list, rot_k, filp, batch_size, r1, r2, r3, r4, fh_list, fw_list):
        # Convert augmentation parameters to numpy
        filp = filp.detach().cpu().numpy()
        rot_k = rot_k.detach().cpu().numpy()
        # Store template key features for matching
        template_k_values = []
        # Process each sample in batch
        for batch_i in range(batch_size):
            # Process each scale level
            for scale_i in range(len(q_grid_list)):
                # Create coordinate grids for current scale
                template_q = torch.from_numpy(np.array(range(fh_list[scale_i] * fw_list[scale_i])).reshape([fh_list[scale_i], fw_list[scale_i]]))
                template_k = torch.from_numpy(np.array(range(fh_list[scale_i] * fw_list[scale_i])).reshape([fh_list[scale_i], fw_list[scale_i]]))
                # Apply augmentations to query template
                if filp[batch_i] == 1:
                    template_q = torch.flip(template_q, dims=[0])
                # Crop template based on region parameters
                template_q = torch.rot90(template_q, k=rot_k[batch_i], dims=[0, 1])
                template_q = template_q[int(r3[batch_i] * fh_list[scale_i]): int(r3[batch_i] * fh_list[scale_i] + r1[batch_i] * fh_list[scale_i]),
                             int(r4[batch_i] * fw_list[scale_i]): int(r4[batch_i] * fw_list[scale_i] + r2[batch_i] * fw_list[scale_i])]
                fh_q, fw_q = template_q.shape
                # Flatten and move to GPU
                template_q = torch.reshape(template_q, (-1,)).long().cuda()
                template_k = torchvision.transforms.Resize([fh_q, fw_q], interpolation=torchvision.transforms.InterpolationMode.NEAREST)\
                    (template_k.unsqueeze(0))
                template_k = torch.reshape(template_k, (-1,)).long().cuda()
                # supervised assisted patch-sampling strategy (core)
                # Sample patches based on supervised labels (first sample in batch only)
                if batch_i == 0:
                    # Resize supervised labels to current scale
                    y_i_sup_resize = torchvision.transforms.Resize([fh_q, fw_q],)(y_i_sup)
                    y_i_sup_resize = torch.reshape(y_i_sup_resize, (4,-1))
                    # Add small noise for diversity
                    epss = torch.rand(y_i_sup_resize.shape).cuda() * 0.1
                    history = torch.zeros(y_i_sup_resize.shape).cuda()
                    if not self.Patch_sup:
                        y_i_sup_resize = torch.zeros(y_i_sup_resize.shape).cuda()
                    # Sample top-k patches for each class
                    class0_v, class0_indice = torch.topk(y_i_sup_resize[0] + epss[0] + history[0], k=self.top_k*(self.sample_ratio ** scale_i), largest=True)
                    history[:,class0_indice] += -torch.max(y_i_sup_resize)
                    class1_v, class1_indice = torch.topk(y_i_sup_resize[1] + epss[1] + history[1], k=self.top_k*(self.sample_ratio ** scale_i), largest=True)
                    history[:,class1_indice] += -torch.max(y_i_sup_resize)
                    class2_v, class2_indice = torch.topk(y_i_sup_resize[2] + epss[2] + history[2], k=self.top_k*(self.sample_ratio ** scale_i), largest=True)
                    history[:,class2_indice] += -torch.max(y_i_sup_resize)
                    class3_v, class3_indice = torch.topk(y_i_sup_resize[3] + epss[3] + history[3], k=self.top_k*(self.sample_ratio ** scale_i), largest=True)
                    # Combine indices from all classes
                    L1 = torch.cat([class0_indice, class1_indice, class2_indice, class3_indice])
                    # Get corresponding query and key features
                    template_q = template_q[L1]
                    template_k = template_k[L1]
                    # Gather features using sampled indices
                    indexed_q_grid = torch.gather(q_grid_list[scale_i][batch_i], 1,
                                                  template_q.expand(q_grid_list[scale_i].size(1), -1))  # NxCxS^2
                    indexed_k_grid = torch.gather(k_grid_list[scale_i][batch_i], 1,
                                                  template_k.expand(k_grid_list[scale_i].size(1), -1))  # NxCxS^2
                    # Store template key features for matching
                    template_k_values.append(indexed_k_grid)
                else:
                    # For subsequent samples, sample patches to labeled templates
                    f_indexed_k_grid = torch.gather(k_grid_list[scale_i][batch_i], 1,
                                                  template_k.expand(k_grid_list[scale_i].size(1), -1))
                    # Using similarity for sampling
                    densecl_sim_k = torch.einsum('cn,ck->nk', [template_k_values[scale_i].clone().detach(), f_indexed_k_grid]).detach()
                    tag = torch.max(densecl_sim_k)
                    # Sample matching indices
                    for idxx in range(template_k_values[scale_i].shape[1]):
                        if idxx == 0:
                            L1 = torch.argmax(densecl_sim_k[idxx], keepdim=True)
                            densecl_sim_k[:, L1] += -tag # Avoid re-selection ** important **
                        else:
                            L1_ = torch.argmax(densecl_sim_k[idxx], keepdim=True)
                            L1 = torch.cat([L1, L1_])
                            densecl_sim_k[:, L1_] += -tag
                    # Get matched features
                    template_q = template_q[L1]
                    template_k = template_k[L1]
                    indexed_q_grid = torch.gather(q_grid_list[scale_i][batch_i], 1,
                                                  template_q.expand(q_grid_list[scale_i].size(1), -1))  # NxCxS^2
                    indexed_k_grid = torch.gather(k_grid_list[scale_i][batch_i], 1,
                                                  template_k.expand(k_grid_list[scale_i].size(1), -1))  # NxCxS^2
                # Accumulate features by scale
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
        # Reshape features to [N, C] format
        q_dense1 = q_dense1.permute(1, 0)
        k_dense1 = k_dense1.permute(1, 0)
        q_dense2 = q_dense2.permute(1, 0)
        k_dense2 = k_dense2.permute(1, 0)
        q_dense3 = q_dense3.permute(1, 0)
        k_dense3 = k_dense3.permute(1, 0)
        return q_dense1, k_dense1, q_dense2, k_dense2, q_dense3, k_dense3

    # return image ProductionBatch&sampleID information from filename
    def get_label(self, ids):
        # The image filename consists of ProductionBatch(size)-sampleID(ids)-x-x.
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
        # When the ProductionBatch are the same, the corresponding categories in "size_label" are the same.
        # When the ProductionBatch and sampleID are the same, the corresponding categories in "ids_label" are the same.
        return np.array(ids_label), np.array(size_label)

    # forward function when feeding samples to the model
    def forward(self, x_q, x_k, y_i_sup, ids, rot_k, filp, r1, r2, r3, r4):
        # x_q, x_k are raw image ang its augmented image
        # the first samples x_q[0], x_k[0] in batch are with annotations y_i_sup
        # y_i_sup is the annotation for patch sampling
        # the augmentation style "rot_k, filp, ..." are also required for research corresponding patch

        # PSCL-MoCo framework
        if self.selfmode == 'moco':
            ids_label, size_label = self.get_label(ids)
            batch_size, _, _, _ = x_q.shape
            # get global features (image) and local features (patch) in query network
            q_b = self.encoder_q.encode(x_q)  # backbone features
            q, qup1_grid, qup2_grid, qup3_grid = self.encoder_q.decode(q_b)  # queries: NxC; NxCxS^2

            _, _, f1h, f1w = q_b[1].shape
            _, _, f2h, f2w = q_b[2].shape
            _, _, f3h, f3w = q_b[3].shape

            q = nn.functional.normalize(q, dim=1) # global feature
            qup1_grid = nn.functional.normalize(qup1_grid, dim=1) # scale-1 local feature
            qup2_grid = nn.functional.normalize(qup2_grid, dim=1) # scale-2 local feature
            qup3_grid = nn.functional.normalize(qup3_grid, dim=1) # scale-3 local feature

            shuffled_idxs, reverse_idxs = self.shuffled_idx(batch_size)

            # get global features (image) and local features (patch) in key network
            with torch.no_grad():
                # Update the key encoder
                self.momentum_update()
                # Shuffle minibatch
                x_k = x_k[shuffled_idxs]
                # Feature representations of the shuffled key view from the key encoder
                k_b = self.encoder_k.encode(x_k)
                k, kup1_grid, kup2_grid, kup3_grid = self.encoder_k.decode(k_b)  # keys: NxC; NxCxS^2
                k = nn.functional.normalize(k, dim=1)
                kup1_grid = nn.functional.normalize(kup1_grid, dim=1) # scale-1 local feature (key encoder)
                kup2_grid = nn.functional.normalize(kup2_grid, dim=1) # scale-2 local feature (key encoder)
                kup3_grid = nn.functional.normalize(kup3_grid, dim=1) # scale-3 local feature (key encoder)
                # Restore original order
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

        # PSCL-SimCLR framework
        # For the typical MoCo framework, it has Dynamic Queues and Momentum Encoders.
        # However, the typical SimCLR framework, Dynamic Queues and Momentum Encoders are not necessary,
        # so its implementation is simpler.
        elif self.selfmode == 'simclr':
            batch_size, _, _, _ = x_q.shape
            # get global features (image) and local features (patch)
            # for SimCLR framework, it has not key encoder
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
            # for SimCLR framework, it has not queue/memory to update.
            return q, k, q_dense1, k_dense1, q_dense2, k_dense2, q_dense3, k_dense3

# We use this network to segment aluminum metallographic images,
# which is the benchmark model of our PSCL
class UNet(nn.Module):
    """Modified UNet architecture

    Key features:
    - Encoder-decoder structure with skip connections
    - Configurable normalization layers
    - Multi-scale feature extraction
    """
    def __init__(self, drop, channel, n_channels = 3, n_classes = 4, bilinear=False, droprate = 0, IncNorm = ['',''],
                 DownNorm = ['',''], UpNorm = ['','']):
        super(UNet, self).__init__()
        # the UNet_encode, UNet_decode code are in networks
        self.encode = UNet_encode(drop, channel, n_channels = n_channels, n_classes = n_classes, bilinear=bilinear, droprate = droprate,
                                  IncNorm = IncNorm, DownNorm = DownNorm, UpNorm = UpNorm)
        self.decode = UNet_decode(drop, channel, n_classes = n_classes, bilinear=bilinear, droprate = droprate)

    def forward(self, x, sample = True):
        [x4, x4up1, x4up2, x4up3] = self.encode(x)
        x = self.decode([x4, x4up1, x4up2, x4up3])
        #return x, [x4, x4up1, x4up2, x4up3]
        return x