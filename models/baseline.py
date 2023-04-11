#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/1/2022 9:29 PM
# @Author  : Guofeng Mei
# @Email   : Guofeng.Mei@student.uts.edu.au
# @File    : baseline.py
# @Software: PyCharm
import torch
import torch.nn.functional as F
from torch import nn

from models.utils import get_module_device


class ConCriterion(nn.Module):
    """
    Taken from: https://github.com/google-research/simclr/blob/master/objective.py
    Converted to pytorch, and decomposed for a clearer understanding.
    batch_size (integer): Number of data_samples per batch.
    normalize (bool, optional): Whether to normalise the representation. (Default: True)
    temperature (float, optional): The temperature parameter of the NT_Xent loss. (Default: 1.0)
    z_i (Tensor): Representation of view 'i'
    z_j (Tensor): Representation of view 'j'
    Returns: loss (Tensor): NT_Xent loss between z_i and z_j
    """

    def __init__(self, temperature=0.1):
        super().__init__()

        self.temperature = temperature

    def forward(self, z_i, z_j, normalize=True):
        device = z_i.device
        batch_size = z_i.shape[0]
        labels = torch.zeros(batch_size * 2).long().to(device)
        mask = torch.ones((batch_size, batch_size), dtype=bool).fill_diagonal_(0).to(device)
        if normalize:
            z_i_norm = F.normalize(z_i, p=2, dim=-1)
            z_j_norm = F.normalize(z_j, p=2, dim=-1)
        else:
            z_i_norm = z_i
            z_j_norm = z_j
        bsz = z_i_norm.size(0)
        ''' Note: **
        Cosine similarity matrix of all samples in batch: a = z_i, b = z_j
         ____ ____
        | aa | ab |
        |____|____|
        | ba | bb |
        |____|____|
        Positives: Diagonals of ab and ba '\'
        Negatives: All values that do not lie on leading diagonals of aa, bb, ab, ba.
        '''
        # Cosine similarity between all views
        logits_aa = torch.mm(z_i_norm, z_i_norm.t()) / self.temperature
        logits_bb = torch.mm(z_j_norm, z_j_norm.t()) / self.temperature
        logits_ab = torch.mm(z_i_norm, z_j_norm.t()) / self.temperature
        logits_ba = torch.mm(z_j_norm, z_i_norm.t()) / self.temperature
        # Compute Positive Logits
        logits_ab_pos = logits_ab[torch.logical_not(mask)]
        logits_ba_pos = logits_ba[torch.logical_not(mask)]
        # Compute Negative Logits
        logit_aa_neg = logits_aa[mask].reshape(bsz, -1)
        logit_bb_neg = logits_bb[mask].reshape(bsz, -1)
        logit_ab_neg = logits_ab[mask].reshape(bsz, -1)
        logit_ba_neg = logits_ba[mask].reshape(bsz, -1)
        # Positive Logits over all samples
        pos = torch.cat((logits_ab_pos, logits_ba_pos)).unsqueeze(1)
        # Negative Logits over all samples
        neg_a = torch.cat((logit_aa_neg, logit_ab_neg), dim=1)
        neg_b = torch.cat((logit_ba_neg, logit_bb_neg), dim=1)
        neg = torch.cat([neg_a, neg_b], dim=0)
        # Compute cross entropy
        logits = torch.cat([pos, neg], dim=1)
        loss = F.cross_entropy(logits, labels)

        return loss


class MLP(nn.Module):
    def __init__(self, in_size=512, out_size=256, hidden_size=1024, used='proj'):
        super().__init__()
        if used == 'proj':
            self.net = nn.Sequential(
                nn.Linear(in_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, out_size)
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(in_size, hidden_size),
                # nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, out_size)
            )

    def forward(self, x):
        return self.net(x)


class CONV(nn.Module):
    def __init__(self, in_size=512, out_size=256, hidden_size=1024, used='proj'):
        super().__init__()
        if used == 'proj':
            self.net = nn.Sequential(
                nn.Conv1d(in_size, hidden_size, 1),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden_size, hidden_size, 1),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden_size, out_size, 1)
            )
        else:
            self.net = nn.Sequential(
                nn.Conv1d(in_size, hidden_size, 1),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden_size, out_size, 1)
            )

    def forward(self, x):
        return self.net(x)


class SimCLR(nn.Module):
    def __init__(self,
                 backbone,
                 projector=None,
                 tau=0.01):
        super().__init__()
        self.backbone = backbone
        self.projector = projector
        self.criterion = ConCriterion(temperature=tau)
        device = get_module_device(backbone)
        self.to(device)

    def forward(self, x1, x2=None, return_embedding=False):
        """
        :param x1: [bz, dim, num]
        :param x2: [bz, dim, num]
        :param return_embedding:
        :return:
        """
        if return_embedding or x2 is None:
            return self.backbone(x1)
        x1_out = self.backbone(x1)
        x2_out = self.backbone(x2)
        if len(x1_out) == 2:
            out1, wise_x1 = x1_out
            out2, wise_x2 = x2_out
        else:
            out1, wise_x1, trans1 = x1_out
            out2, wise_x2, trans2 = x2_out
        z1 = self.projector(out1)
        z2 = self.projector(out2)
        loss_sim = 10 * self.criterion(z1, z2)

        return loss_sim
