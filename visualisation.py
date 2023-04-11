#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/24/2022 6:48 PM
# @Author  : Guofeng Mei
# @Email   : Guofeng.Mei@student.uts.edu.au
# @File    : visualisation.py
# @Software: PyCharm
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from data.modelnet import ModelNet40
from data.shapenet_part import ShapeNetPart

try:
    from sklearn.manifold import TSNE

    HAS_SK = True
except Exception as E:
    HAS_SK = False
    print('Please install sklearn for layer visualization {}', E)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'data'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'lib'))
from models.encoder import DGCNNPartSeg, PointNet, DGCNN
from models.model import ClusterNet
from models.utils import svm_data
from lib.lib_utils import get_color_map, make_point_cloud


def parse_args():
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    # dataset
    parser.add_argument('--root', type=str,
                        default='/data/gmei/data', help="dataset path")
    parser.add_argument('--dataset', type=str, default='shapenetpart', metavar='N',
                        choices=['shapenetpart', 'modelnet'], help='Dataset to use')
    parser.add_argument('--subset10', action='store_true', default=False,
                        help='Whether to use ModelNet10 [default: False]')
    parser.add_argument('--batch_size', type=int, default=12, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    # model
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N', choices=['pointnet, dgcnn,dgcnn_seg'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--ablation', type=str, default='all',
                        help='[xyz, all, fea]')
    parser.add_argument('--c_type', type=str, default='ot',
                        help='[ot, ds]')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--num_clus', type=int, default=64, metavar='N',
                        help='Num of clusters to use')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    # Training settings
    parser.add_argument('--use_sgd', action="store_true", help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--decay_rate', type=float, default=1e-4,
                        help='decay rate [default: 1e-4]')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--exp_name', type=str, default='dgcnn_part_seg', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train')
    parser.add_argument('--start_epoch', type=int, default=0, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--class_choice', type=str, default=None, metavar='N',
                        choices=['airplane', 'bag', 'cap', 'car', 'chair',
                                 'earphone', 'guitar', 'knife', 'lamp', 'laptop',
                                 'motor', 'mug', 'pistol', 'rocket', 'skateboard', 'table'])
    parser.add_argument('--resume', action="store_true", help='resume from checkpoint')
    parser.add_argument('--save_freq', type=int, default=50, help='save frequency')
    parser.add_argument('--print_freq', type=int, default=50, help='print frequency')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    return parser.parse_args()


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid[np.newaxis, :]
    m = np.max(np.sqrt(np.sum(np.power(pc, 2), axis=1)), axis=0)
    pc = pc / m[np.newaxis, np.newaxis]
    return pc


def data_depict(args):
    test_dataset = ShapeNetPart(args.root, partition='test', num_points=args.num_points,
                                class_choice=args.class_choice)
    loader = DataLoader(test_dataset, batch_size=1,
                        num_workers=args.workers, shuffle=True, drop_last=False)
    seg_num_all = loader.dataset.seg_num_all
    model = DGCNNPartSeg(args.emb_dims, args.num_clus, args.dropout, seg_num_all, pretrain=False).cuda()
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(f'checkpoints/{args.exp_name}/models/model.t7'))

    model1 = DGCNNPartSeg(args.emb_dims, args.num_clus, args.dropout, seg_num_all, pretrain=False).cuda()
    model1 = nn.DataParallel(model1)
    model1.load_state_dict(torch.load(f'checkpoints/dgcnn_part_seg/models/model.t7'))
    with torch.no_grad():
        model.eval()
        model1.eval()
        labels = list()
        for i, data in enumerate(loader):
            points, cls, seg = data
            ids = cls.view(-1).item()
            skip = 0  # Skip every n points
            label_one_hot = np.zeros((cls.shape[0], 16))
            for idx in range(cls.shape[0]):
                label_one_hot[idx, cls[idx]] = 1
            label_one_hot = torch.from_numpy(label_one_hot.astype(np.float32))
            if ids not in labels:
                labels.append(ids)
                points, label_one_hot = points.cuda(), label_one_hot.cuda()
                seg_pred = model(points.transpose(-1, -2), label_one_hot)
                seg_pred = seg_pred.permute(0, 2, 1).contiguous()
                pred = seg_pred.max(dim=2)[1][skip].view(-1).cpu().numpy()
                pred = pred - pred.min()

                label = seg[skip].view(-1).numpy()
                label = label - label.min()

                seg_pred1 = model1(points.transpose(-1, -2), label_one_hot)
                seg_pred1 = seg_pred1.permute(0, 2, 1).contiguous()
                pred1 = seg_pred1.max(dim=2)[1][skip].view(-1).cpu().numpy()
                pred1 = pred1 - pred1.min()

                print(set(list(label)))
                print(set(list(pred)))
                print(set(list(pred1)))
                print("==================")
                gt_color = get_color_map(1.0 * label / (label.max() + 0.1))
                pred_color = get_color_map(1.0 * pred / (label.max() + 0.1))
                pred_color1 = get_color_map(1.0 * pred1 / (label.max() + 0.1))
                point = points[skip].cpu().numpy()
                point[:, :3] = pc_normalize(point[:, :3])
                make_point_cloud(point, f'figures/gt{ids}.ply', color=gt_color)
                make_point_cloud(point, f'figures/cross{ids}.ply', color=pred_color)
                # make_point_cloud(point, f'figures/soft{ids}.ply', color=pred_color1)


def visual_tsne(args, data):
    """MODEL LOADING"""
    # Try to load models
    if args.model == 'dgcnn':
        net = DGCNN(args.emb_dims, args.k, args.dropout, -1).cuda()
    elif args.model == 'dgcnn_seg':
        net = DGCNNPartSeg(args.emb_dims, args.k, args.dropout).cuda()
    elif args.model == 'pointnet':
        net = PointNet(args.emb_dims, is_normal=False, feature_transform=True, feat_type='global').cuda()
    else:
        raise Exception("Not implemented")
    point_model = ClusterNet(net, dim=args.emb_dims, num_clus=args.num_clus, ablation='all', c_type='ot')
    model_path = os.path.join(f'checkpoints/{args.exp_name}/models/', 'best_model.pth')
    try:
        point_model.load_state_dict(torch.load(model_path))
    except Exception as e:
        print('No existing model, starting training from scratch {}'.format(e))
    features, labels = svm_data(data, point_model)
    # t-SNE 是一种非线性降维算法，非常适用于高维数据降维到2维或者3维，进行可视化
    tsne = TSNE(n_components=2, perplexity=100)
    # fit_transform函数把last_layer的Tensor降低至2个特征量,即3个维度(2个维度的坐标系)
    low_dim_embs = tsne.fit_transform(features)
    plot_with_labels(low_dim_embs, labels)


def plot_with_labels(low_d_weights, labels, num_classes=40, dsname='ModelNet40'):
    if dsname == 'ModelNet10':
        classnames = [
            'bathtub',
            'bed',
            'chair',
            'desk',
            'dresser',
            'monitor',
            'night_stand',
            'sofa',
            'table',
            'toilet'
        ]
    else:
        classnames = []
    plt.cla()  # clear当前活动的坐标轴
    # fig = plt.figure(figsize=(8, 8))
    X, Y = low_d_weights[:, 0], low_d_weights[:, 1]
    cmap = plt.cm.get_cmap("tab20", num_classes)
    # 把Tensor的第1列和第2列,也就是TSNE之后的前两个特征提取出来,作为X,Y
    fig = plt.scatter(X, Y, c=labels + 1, cmap=cmap, s=20)
    cbar = plt.colorbar(fig)
    cbar.set_ticks(list(range(1, num_classes + 1)))
    cbar.set_ticklabels(classnames)
    cbar.ax.tick_params(labelsize=15)
    plt.clim(0.5, num_classes + 0.5)
    # plt.title('Ours')
    plt.axis('off')
    plt.savefig('/home/gmei/Data/data/pscnn.pdf')
    plt.show()


if __name__ == '__main__':
    args = parse_args()
    test_loader = DataLoader(ModelNet40(args.root, args.num_points, False, False, True, True, True, False))
    # visual_tsne(args, test_loader)
    data_depict(args)
