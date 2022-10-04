#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/14/2022 11:54 PM
# @Author  : Guofeng Mei
# @Email   : Guofeng.Mei@student.uts.edu.au
# @File    : encoder.py
# @Software: PyCharm
import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch import nn
from torch.autograd import Variable

from models.utils import get_graph_feature


class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = nn.Conv1d(channel, 64, 1)  # in-channel, out-channel, kernel size
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        B = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=False)[0]  # global descriptors

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        iden = Variable(torch.eye(3, dtype=torch.float32, device=x.device)).view(1, 9).repeat(B, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        B = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=False)[0]

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(
            torch.eye(self.k, dtype=torch.float32,
                      device=x.device)).view(1, self.k ** 2).repeat(B, 1)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNet(nn.Module):
    def __init__(self, dims=1024, is_normal=False, feature_transform=False, feat_type='global'):
        super().__init__()
        channel = 6 if is_normal else 3
        self.stn = STN3d(channel)  # Batch * 3 * 3
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)
        self.conv1 = nn.Conv1d(channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, dims, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(dims)
        self.feat_type = feat_type

    def forward(self, x):
        _, D, N = x.size()
        trans = self.stn(x)
        x = x.transpose(2, 1)
        # ids = farthest_point_sample(x, self.npoint)
        feature = None
        if D > 3:
            x, feature = x.split([3, D - 3], dim=2)
        if self.feature_transform:
            x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        out1 = self.bn1(self.conv1(x))
        x = F.relu(out1)

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None
        point_feat = x
        out2 = self.bn2(self.conv2(x))
        x = F.relu(out2)
        out3 = self.bn3(self.conv3(x))
        x = torch.max(out3, 2, keepdim=False)[0]
        if self.feat_type == 'global':
            return x, out3, trans_feat
        elif self.feat_type == 'detailed':
            return x, out1, out2, out3
        else:  # concatenate global and local feature together
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, point_feat], 1), out3, trans_feat


class DGCNN(nn.Module):
    def __init__(self, emb_dims, k, dropout=0.5, num_cls=-1):
        super(DGCNN, self).__init__()
        self.k = k
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, emb_dims, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(emb_dims),
                                   nn.LeakyReLU(negative_slope=0.2))

        if num_cls != -1:
            self.linear1 = nn.Linear(emb_dims * 2, 512, bias=False)
            self.bn6 = nn.BatchNorm1d(512)
            self.dp1 = nn.Dropout(p=dropout)
            self.linear2 = nn.Linear(512, 256)
            self.bn7 = nn.BatchNorm1d(256)
            self.dp2 = nn.Dropout(p=dropout)
            self.linear3 = nn.Linear(256, num_cls)
        self.cls = num_cls

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x.contiguous())
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)
        feat = x
        # x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        # x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        # x = torch.cat((x1, x2), 1)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        if self.cls != -1:
            x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
            x = self.dp1(x)
            x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
            x = self.dp2(x)
            x = self.linear3(x)

        return x, feat


class PointNetPartSeg(nn.Module):
    def __init__(self, feature_transform=True, channel=3):
        super(PointNetPartSeg, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = nn.Conv1d(channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 128, 1)
        self.conv4 = nn.Conv1d(128, 512, 1)
        self.conv5 = nn.Conv1d(512, 2048, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(2048)

        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=128)

    def forward(self, point_cloud, label):
        B, D, N = point_cloud.size()

        trans = self.stn(point_cloud)
        point_cloud = point_cloud.transpose(2, 1)
        if D > 3:
            point_cloud, feature = point_cloud.split(3, dim=2)
        point_cloud = torch.bmm(point_cloud, trans)
        if D > 3:
            point_cloud = torch.cat([point_cloud, feature], dim=2)
        point_cloud = point_cloud.transpose(2, 1)

        out1 = F.relu(self.bn1(self.conv1(point_cloud)))
        out2 = F.relu(self.bn2(self.conv2(out1)))
        out3 = F.relu(self.bn3(self.conv3(out2)))

        if self.feature_transform:
            trans_feat = self.fstn(out3)
            net_transformed = torch.bmm(out3.transpose(2, 1), trans_feat)
            out3 = net_transformed.transpose(2, 1)

        out4 = F.relu(self.bn4(self.conv4(out3)))
        out5 = self.bn5(self.conv5(out4))

        out_max = torch.max(out5, 2, keepdim=False)[0]
        out_max = torch.cat([out_max, label.squeeze(1)], 1)
        expand = out_max.view(-1, 2048 + 16, 1).repeat(1, 1, N)
        concat = torch.cat([expand, out1, out2, out3, out4, out5], 1)

        if self.feature_transform:
            return concat, trans_feat
        return concat


class TransformNet(nn.Module):
    def __init__(self, in_dims, out_dims):
        super(TransformNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_dims, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn4 = nn.BatchNorm1d(256)

        self.transform = nn.Linear(256, out_dims * out_dims)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(out_dims, out_dims))

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = self.conv3(x)  # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 1024, num_points) -> (batch_size, 1024)

        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)  # (batch_size, 1024) -> (batch_size, 512)
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)  # (batch_size, 512) -> (batch_size, 256)

        x = self.transform(x)  # (batch_size, 256) -> (batch_size, 3*3)
        x = x.view(batch_size, 3, 3)  # (batch_size, 3*3) -> (batch_size, 3, 3)

        return x


class DGCNNPartSeg(nn.Module):
    def __init__(self, emb_dims, k, dropout=0.5, seg_num_all=None, pretrain=True):
        super(DGCNNPartSeg, self).__init__()
        self.seg_num_all = seg_num_all
        self.k = k
        self.pretrain = pretrain
        self.transform_net = TransformNet(6, 3)
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, emb_dims, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(emb_dims),
                                   nn.LeakyReLU(negative_slope=0.2))

        if not self.pretrain:
            self.conv7 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                       nn.BatchNorm1d(64),
                                       nn.LeakyReLU(negative_slope=0.2))
            self.conv8 = nn.Sequential(nn.Conv1d(1280, 256, kernel_size=1, bias=False),
                                       nn.BatchNorm1d(256),
                                       nn.LeakyReLU(negative_slope=0.2))
            self.dp1 = nn.Dropout(p=dropout)
            self.conv9 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                       nn.BatchNorm1d(256),
                                       nn.LeakyReLU(negative_slope=0.2))
            self.dp2 = nn.Dropout(p=dropout)
            self.conv10 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(128),
                                        nn.LeakyReLU(negative_slope=0.2))
            self.conv11 = nn.Conv1d(128, self.seg_num_all, kernel_size=1, bias=False)

    def forward(self, x, l=None):
        batch_size = x.size(0)
        num_points = x.size(2)

        x0 = get_graph_feature(x, k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        t = self.transform_net(x0)  # (batch_size, 3, 3)
        x = x.transpose(2, 1)  # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
        x = torch.bmm(x, t)  # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
        x = x.transpose(2, 1)  # (batch_size, num_points, 3) -> (batch_size, 3, num_points)

        x = get_graph_feature(x, k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)  # (batch_size, 64*3, num_points)

        x = self.conv6(x)  # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        feat = x
        x = x.max(dim=-1, keepdim=True)[0]  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        if self.pretrain:
            return x.squeeze(-1), feat
        else:
            l = l.view(batch_size, -1, 1)  # (batch_size, num_categoties, 1)
            l = self.conv7(l)  # (batch_size, num_categoties, 1) -> (batch_size, 64, 1)
            x = torch.cat((x, l), dim=1)  # (batch_size, 1088, 1)
            x = x.repeat(1, 1, num_points)  # (batch_size, 1088, num_points)
            x = torch.cat((x, x1, x2, x3), dim=1)  # (batch_size, 1088+64*3, num_points)
            x = self.conv8(x)  # (batch_size, 1088+64*3, num_points) -> (batch_size, 256, num_points)
            x = self.dp1(x)
            x = self.conv9(x)  # (batch_size, 256, num_points) -> (batch_size, 256, num_points)
            x = self.dp2(x)
            x = self.conv10(x)  # (batch_size, 256, num_points) -> (batch_size, 128, num_points)
            x = self.conv11(x)  # (batch_size, 256, num_points) -> (batch_size, seg_num_all, num_points)

            return x


class DGCNNSemSeg(nn.Module):
    def __init__(self, emb_dims, k, dropout=0.5, num_channel=9, num_class=40, pretrain=False):
        super(DGCNNSemSeg, self).__init__()
        self.k = k
        self.pretrain = pretrain
        self.conv1 = nn.Sequential(nn.Conv2d(num_channel * 2, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, emb_dims, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(emb_dims),
                                   nn.LeakyReLU(negative_slope=0.2))
        if not self.pretrain:
            self.conv7 = nn.Sequential(nn.Conv1d(1216, 512, kernel_size=1, bias=False),
                                       nn.BatchNorm1d(512),
                                       nn.LeakyReLU(negative_slope=0.2))
            self.conv8 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                       nn.BatchNorm1d(256),
                                       nn.LeakyReLU(negative_slope=0.2))
            self.dp1 = nn.Dropout(p=dropout)
            self.conv9 = nn.Conv1d(256, num_class, kernel_size=1, bias=False)

    def forward(self, x):
        batch_size, _, num_points = x.shape
        x = get_graph_feature(x, self.k, extra_dim=True)
        x = self.conv1(x)
        x = self.conv2(x)
        x1 = x.max(dim=-1, keepdim=False)[0]
        x = get_graph_feature(x1, self.k)
        x = self.conv3(x)
        x = self.conv4(x)
        x2 = x.max(dim=-1, keepdim=False)[0]
        x = get_graph_feature(x2, self.k)
        x = self.conv5(x)
        x3 = x.max(dim=-1, keepdim=False)[0]
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.conv6(x)
        feat = x
        x = x.max(dim=-1, keepdim=True)[0]
        if self.pretrain:
            return x.squeeze(-1), feat
        else:
            x = x.repeat(1, 1, num_points)
            x = torch.cat((x, x1, x2, x3), dim=1)
            x = self.conv7(x)
            x = self.conv8(x)
            x = self.dp1(x)
            x = self.conv9(x)
            return x


if __name__ == '__main__':
    x = torch.rand(4, 9, 1024)
    dgsg = DGCNNSemSeg(1024, 20, num_class=40, num_channel=9, pretrain=False)
    print(dgsg(x).shape)
