#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/15/2022 11:05 AM
# @Author  : Guofeng Mei
# @Email   : Guofeng.Mei@student.uts.edu.au
# @File    : model.py
# @Software: PyCharm
import torch
import torch.nn.functional as F
from torch import nn

from models.utils import sinkhorn, square_distance, feature_transform_regularizer, get_module_device


def ot_assign(x, y, epsilon=1e-3, thresh=1e-3, max_iter=30, dst='fe'):
    device = x.device
    batch_size, dim, num_x = x.shape
    num_y = y.shape[-1]
    # both marginals are fixed with equal weights
    p = torch.empty(batch_size, num_x, dtype=torch.float,
                    requires_grad=False, device=device).fill_(1.0 / num_x).squeeze()
    q = torch.empty(batch_size, num_y, dtype=torch.float,
                    requires_grad=False, device=device).fill_(1.0 / num_y).squeeze()
    if dst == 'eu':
        cost = square_distance(x.transpose(-1, -2), y.transpose(-1, -2))
    else:
        cost = 2.0 - 2.0 * torch.einsum('bdn,bdm->bnm', x, y)
    gamma, loss = sinkhorn(cost, p, q, epsilon, thresh, max_iter)
    return gamma, loss


def dis_assign(x, y, tau=0.01, dst='fe'):
    """
    :param x:
    :param y: cluster center
    :param tau:
    :param dst:
    :return:
    """
    if dst == 'eu':
        cost = square_distance(x.transpose(-1, -2), y.transpose(-1, -2))
        cost_mean = torch.mean(cost, dim=-1, keepdim=True)
        cost = cost_mean - cost
    else:
        cost = 2.0 * torch.einsum('bdn,bdj->bnj', x, y)
    gamma = F.softmax(cost / tau, dim=-1)
    return gamma.transpose(-1, -2), cost


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


def regular(center, reg=0.0001):
    bs, dim, num = center.shape
    identity = torch.eye(num).to(center).unsqueeze(0)
    loss = reg * torch.abs(torch.einsum('bdm,bdn->bmn', center, center) - identity).mean()
    return loss


class PointCluOT(nn.Module):

    def __init__(self, num_clusters=32, dim=1024, ablation='all'):
        """
        num_clusters: int The number of clusters
        dim: int Dimension of descriptors
        alpha: float Parameter of initialization. Larger value is harder assignment.
        normalize_input: bool If true, descriptor-wise L2 normalization is applied to input.
        """
        super().__init__()
        self.num_clusters = num_clusters
        self.conv = CONV(in_size=dim, out_size=num_clusters, hidden_size=dim // 2, used='proj')
        self.dim = dim
        self.ablation = ablation

    def forward(self, feature, xyz):
        bs, dim, num = feature.shape
        # soft-assignment
        log_score = self.conv(feature).view(bs, self.num_clusters, -1)
        score = F.softmax(log_score, dim=1)  # [b, k, n]
        pi = score.sum(-1).clip(min=1e-4).unsqueeze(1).detach()  # [b, 1, k]
        if self.ablation in ['all', 'xyz']:
            mu_xyz = torch.einsum('bkn,bdn->bdk', score, xyz) / pi  # [b, d, k]
            reg_xyz = 0.001 * regular(mu_xyz)
            with torch.no_grad():
                assign_xyz, dis = ot_assign(xyz, mu_xyz.detach(), max_iter=25, dst='eu')
                assign_xyz = num * assign_xyz.transpose(-1, -2)  # [b, k, n]
        else:
            assign_xyz = torch.zeros_like(score).to(xyz)
            reg_xyz = torch.tensor(0.0).to(xyz)
        if self.ablation in ['all', 'fea']:
            mu_fea = torch.einsum('bkn,bdn->bdk', score, feature) / pi  # [b, d, k]
            n_feature = F.normalize(feature, dim=1, p=2)
            n_mu = F.normalize(mu_fea, dim=1, p=2)
            reg_fea = regular(n_mu)
            with torch.no_grad():
                assign_fea, dis = ot_assign(n_feature.detach(), n_mu.detach(), max_iter=25)
                assign_fea = num * assign_fea.transpose(-1, -2)
        else:
            assign_fea = torch.zeros_like(score).to(xyz)
            reg_fea = torch.tensor(0.0).to(xyz)
        loss_xyz = -torch.mean(torch.sum(assign_xyz.detach() * F.log_softmax(log_score, dim=1), dim=1))
        loss_fea = -torch.mean(torch.sum(assign_fea.detach() * F.log_softmax(log_score, dim=1), dim=1))
        return loss_xyz + loss_fea + reg_fea + reg_xyz


class PointCluDS(nn.Module):
    def __init__(self, num_clusters=32, dim=1024, ablation='all'):
        """
        num_clusters: int The number of clusters
        dim: int Dimension of descriptors
        alpha: float Parameter of initialization. Larger value is harder assignment.
        normalize_input: bool If true, descriptor-wise L2 normalization is applied to input.
        """
        super().__init__()
        self.num_clusters = num_clusters
        self.conv = CONV(in_size=dim, out_size=num_clusters, hidden_size=dim // 2, used='proj')
        self.dim = dim
        self.ablation = ablation

    def forward(self, feature, xyz):
        bs, dim, num = feature.shape
        # soft-assignment
        log_score = self.conv(feature).view(bs, self.num_clusters, -1)
        score = F.softmax(log_score, dim=1)  # [b, k, n]
        pi = score.sum(-1).clip(min=1e-4).unsqueeze(1).detach()  # [b, 1, k]
        if self.ablation in ['all', 'xyz']:
            mu_xyz = torch.einsum('bkn,bdn->bdk', score, xyz) / pi  # [b, d, k]
            reg_xyz = 0.001 * regular(mu_xyz)
            with torch.no_grad():
                assign_xyz, dis = dis_assign(xyz, mu_xyz.detach(), dst='eu')
        else:
            assign_xyz = torch.zeros_like(score).to(xyz)
            reg_xyz = torch.tensor(0.0).to(xyz)
        if self.ablation in ['all', 'fea']:
            mu_fea = torch.einsum('bkn,bdn->bdk', score, feature) / pi  # [b, d, k]
            n_feature = F.normalize(feature, dim=1, p=2)
            n_mu = F.normalize(mu_fea, dim=1, p=2)
            reg_fea = regular(n_mu)
            with torch.no_grad():
                assign_fea, dis = dis_assign(n_feature.detach(), n_mu.detach())
        else:
            assign_fea = torch.zeros_like(score).to(xyz)
            reg_fea = torch.tensor(0.0).to(xyz)
        loss_xyz = -torch.mean(torch.sum(assign_xyz.detach() * F.log_softmax(log_score, dim=1), dim=1))
        loss_fea = -torch.mean(torch.sum(assign_fea.detach() * F.log_softmax(log_score, dim=1), dim=1))
        return loss_xyz + loss_fea + reg_fea + reg_xyz


class ClusterNet(nn.Module):
    def __init__(self,
                 backbone,
                 dim=1024,
                 num_clus=64,
                 num_clus1=None,
                 ablation='all',
                 c_type='ot'):
        super().__init__()
        self.backbone = backbone
        if c_type == 'ot':
            self.cluster = PointCluOT(num_clusters=num_clus, dim=dim, ablation=ablation)
            if num_clus1 is not None:
                self.cluster1 = PointCluOT(num_clusters=num_clus1, dim=dim, ablation=ablation)
        else:
            self.cluster = PointCluDS(num_clusters=num_clus, dim=dim, ablation=ablation)
            if num_clus1 is not None:
                self.cluster1 = PointCluDS(num_clusters=num_clus1, dim=dim, ablation=ablation)
        self.num_clus1 = num_clus1
        device = get_module_device(backbone)
        self.to(device)

    def forward(self, x, return_embedding=False):
        """
        :param x: [bz, dim, num]
        :param return_embedding:
        :return:
        """
        if return_embedding:
            return self.backbone(x, True)
        out = self.backbone(x)
        trans_loss = torch.tensor(0.0, requires_grad=True)
        if len(out) == 2:
            feature, wise = out
        else:
            feature, wise, trans = out
            if trans is not None:
                trans_loss = 0.001 * feature_transform_regularizer(trans)
        loss_rq = self.cluster(wise, x)
        loss_rq1 = torch.tensor(0.0, requires_grad=True)
        if self.num_clus1 is not None:
            loss_rq1 = self.cluster1(wise, x)

        return loss_rq + loss_rq1, trans_loss
