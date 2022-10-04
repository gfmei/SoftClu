#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/25/2022 7:25 PM
# @Author  : Guofeng Mei
# @Email   : Guofeng.Mei@student.uts.edu.au
# @File    : train_semseg.py
# @Software: PyCharm
import argparse
import glob
import os
import shutil
import sys

import torch
import numpy as np
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.common import points_sampler

sys.path.append('models')
sys.path.append('data')
from data.S3DIS import S3DISDatasetHDF5, S3DISDatasetWholeScene, S3DISDataset
from models.encoder import DGCNNSemSeg, DGCNNPartSeg
from models.utils import weights_init, bn_momentum_adjust
from tqdm import tqdm
from lib.lib_utils import TrainLogger, cal_loss, get_colored_point_cloud_pca, get_colored_point_cloud_pca_sep, read_ply
from models.model import ClusterNet

classes = ['ceiling', 'floor', 'wall', 'beam', 'column',
           'window', 'door', 'table', 'chair', 'sofa',
           'bookcase', 'board', 'clutter']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

num_classes = len(seg_label_to_cat)


def vis(args):
    data_root = '/data/gmei/data/stanford_indoor3d/'
    num_point, test_area, block_size, sample_rate = 4096, 6, 1.0, 0.01

    test_data = S3DISDataset(split='test', data_root=data_root, num_point=num_point, test_area=test_area,
                              block_size=block_size, sample_rate=sample_rate, transform=None)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False,
                                                   num_workers=4)
    net = DGCNNSemSeg(args.emb_dims, args.k, dropout=0.5, num_channel=9, num_class=40, pretrain=True).cuda()
    # net = nn.DataParallel(net)
    point_model = ClusterNet(net, dim=args.emb_dims, num_clus=args.num_clus)
    model_path = os.path.join(f'checkpoints/{args.exp_name}/models/', args.pretrained_path)
    point_model.load_state_dict(torch.load(model_path))
    point_model.eval()
    with torch.no_grad():
        feat_list = []
        xyz_list = list()
        num = 0
        count = 0
        for points, target in tqdm(test_data_loader, total=len(test_data_loader), smoothing=0.9):
            xyz = points
            num = xyz.shape[1]
            if count % 5 == 0:
                points = points.transpose(2, 1).float().cuda()
                _, geometric_feats = point_model.backbone(points)
                geometric_feats = geometric_feats.squeeze(0).transpose(1, 0)
                feat_list.append(geometric_feats)
                xyz_list.append(xyz.squeeze(0).cpu().numpy())
            count += 1
        features = torch.cat(feat_list, dim=0).cpu().numpy()
        get_colored_point_cloud_pca(xyz_list, features, f'dgsem', num)


def parse_args():
    parser = argparse.ArgumentParser(description='Point Cloud Semantic Segmentation')
    parser.add_argument('--root', default='/home/gmei/Data/data/indoor3d_sem_seg_hdf5_data',
                        type=str, help='log path [default: ]')
    parser.add_argument('--log_dir', type=str, default='dgcnn_semseg', help='log path [default: ]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU [default: 0]')
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size [default: 24]')
    parser.add_argument('--test_area', type=int, default=5, help='test area, 1-6 [default: 5]')
    parser.add_argument('--epoch', default=100, type=int, help='training epochs [default: 100]')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate [default: 0.001]')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum [default: 0.9]')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='lr decay rate [default: 0.5]')
    parser.add_argument('--restore', action='store_true', default=True,
                        help='restore the weights [default: True]')
    parser.add_argument('--pretrained_path', type=str, default='ckpt_epoch_200.pth', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--exp_name', type=str, default='dgcnn_sem_S64', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--num_clus', type=int, default=64, metavar='N',
                        help='Num of clusters to use')
    parser.add_argument('--restore_path', type=str, default='dgcnn_sem_S64',
                        help='path to pre-saved model weights [default: ]')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate in FCs [default: 0.5]')
    parser.add_argument('--bn_decay', action='store_true', help='use BN Momentum Decay [default: False]')
    parser.add_argument('--xavier_init', action='store_true', help='Xavier weight init [default: False]')
    parser.add_argument('--emb_dims', type=int, default=1024, help='embedding dimensions [default: 1024]')
    parser.add_argument('--k', type=int, default=20, help='num of nearest neighbors to use [default: 20]')
    parser.add_argument('--step_size', type=int, default=40, help='lr decay steps [default: every 40 epochs]')
    parser.add_argument('--scheduler', type=str, default='cos', help='lr decay scheduler [default: cos, step]')
    parser.add_argument('--model', type=str, default='dgcnn_sem', help='model [default: pointnet_semseg]')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimiser [default: adam, otherwise sgd]')

    return parser.parse_args()


def class_pred(seg_pred, target, num_classes, criterion, trans_feat=None):
    seg_pred = seg_pred.permute(0, 2, 1).contiguous()
    seg_pred = seg_pred.contiguous().view(-1, num_classes)
    loss = criterion(seg_pred, target, trans_feat)
    return loss, seg_pred


def main(args):
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    root = args.root
    train_dataset = S3DISDatasetHDF5(root=root, split='train', test_area=args.test_area)
    test_dataset = S3DISDatasetHDF5(root=root, split='test', test_area=args.test_area)
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    my_logger = TrainLogger(args, name=args.model.upper(), subfold='semseg',
                            cls2name=class2label, filename=args.mode + '_log')
    my_logger.logger.info("The number of training data is: %d" % len(train_dataset))
    my_logger.logger.info("The number of testing data is: %d" % len(test_dataset))
    ''' === Model Loading === '''
    shutil.copy(os.path.abspath(__file__), my_logger.log_dir)
    writer = SummaryWriter(os.path.join(my_logger.experiment_dir, 'runs'))

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = DGCNNSemSeg(args.emb_dims, 20, num_class=num_classes, num_channel=9,
                      dropout=args.dropout).cuda()
    net = torch.nn.DataParallel(net)
    print('=' * 27)
    print('Using %d GPU,' % torch.cuda.device_count(), 'Indices: %s' % args.gpu)
    print('=' * 27)
    point_model = ClusterNet(net, dim=args.emb_dims, num_clus=args.num_clus)
    model_path = os.path.join(f'checkpoints/{args.restore_path}/models/', args.pretrained_path)
    if args.restore:
        # checkpoint = torch.load(model_path)
        # point_model = copy_parameters(point_model, checkpoint, verbose=True)
        point_model.load_state_dict(torch.load(model_path), strict=False)
        classifier = point_model.backbone
        my_logger.logger.info('Use pre-trained weights from %s' % args.restore_path)
    else:
        my_logger.logger.info('No pre-trained weights, start training from scratch...')
        classifier = net.apply(weights_init)
        my_logger.logger.info("Using Xavier weight initialisation")
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=1e-4)
        my_logger.logger.info("Using Adam optimiser")
    else:
        optimizer = torch.optim.SGD(
            classifier.parameters(),
            lr=args.lr * 100,
            momentum=args.momentum)
        my_logger.logger.info("Using SGD optimiser")
    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=1e-3)
    else:
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.lr_decay)

    learning_rate_clip = 1e-5
    momentum_original = 0.1
    momentum_decay = 0.5
    momentum_decay_step = args.step_size
    criterion = cal_loss

    ''' === Testing then Exit === '''
    if args.mode == 'test':
        with torch.no_grad():
            classifier.eval()
            my_logger.epoch_init(training=False)

            for points, target in tqdm(test_data_loader, total=len(test_data_loader), smoothing=0.9):
                points, target = points.transpose(2, 1).float().cuda(), target.view(-1, 1)[:, 0].long().cuda()
                trans_feat = None
                if args.model == 'pointnet_sem':
                    seg_pred, trans_feat = classifier(points)
                else:
                    seg_pred = classifier(points)
                loss, seg_pred = class_pred(seg_pred, target, num_classes, criterion, trans_feat)
                my_logger.step_update(seg_pred.data.max(1)[1].cpu().numpy(),
                                      target.long().cpu().numpy(),
                                      loss.cpu().detach().numpy())

            my_logger.epoch_summary(writer=writer, training=False, mode='semseg')
        sys.exit("Test Finished")

    for epoch in range(my_logger.epoch, args.epoch + 1):
        ''' === Training === '''
        # scheduler.step()
        my_logger.epoch_init()
        for points, target in tqdm(train_data_loader, total=len(train_data_loader), smoothing=0.9):
            writer.add_scalar('learning rate', scheduler.get_lr()[-1], my_logger.step)
            points, target = points.float().transpose(2, 1).cuda(), target.view(-1, 1)[:, 0].long().cuda()
            classifier.train()
            optimizer.zero_grad()
            trans_feat = None
            if args.model == 'pointnet_sem':
                seg_pred, trans_feat = classifier(points)
            else:
                seg_pred = classifier(points)
            loss, seg_pred = class_pred(seg_pred, target, num_classes, criterion, trans_feat)
            loss.backward()
            optimizer.step()

            my_logger.step_update(seg_pred.data.max(1)[1].cpu().numpy(),
                                  target.long().cpu().numpy(),
                                  loss.cpu().detach().numpy())
        my_logger.epoch_summary(writer=writer, training=True, mode='semseg')

        '''=== Evaluating ==='''
        with torch.no_grad():
            classifier.eval()
            my_logger.epoch_init(training=False)

            for points, target in tqdm(test_data_loader, total=len(test_data_loader), smoothing=0.9):
                points, target = points.transpose(2, 1).float().cuda(), target.view(-1, 1)[:, 0].long().cuda()
                trans_feat = None
                if args.model == 'pointnet_sem':
                    seg_pred, trans_feat = classifier(points)
                else:
                    seg_pred = classifier(points)
                loss, seg_pred = class_pred(seg_pred, target, num_classes, criterion, trans_feat)
                my_logger.step_update(seg_pred.data.max(1)[1].cpu().numpy(),
                                      target.long().cpu().numpy(),
                                      loss.cpu().detach().numpy())

            my_logger.epoch_summary(writer=writer, training=False, mode='semseg')
            if my_logger.save_model:
                state = {
                    'step': my_logger.step,
                    'miou': my_logger.best_miou,
                    'epoch': my_logger.best_miou_epoch,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, my_logger.savepath)

        scheduler.step()
        if args.scheduler == 'step':
            for param_group in optimizer.param_groups:
                if optimizer.param_groups[0]['lr'] < learning_rate_clip:
                    param_group['lr'] = learning_rate_clip
        if args.bn_decay:
            momentum = momentum_original * (momentum_decay ** (epoch // momentum_decay_step))
            if momentum < 0.01:
                momentum = 0.01
            print('BN momentum updated to: %f' % momentum)
            classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))

    my_logger.train_summary(mode='semseg')


if __name__ == '__main__':
    args = parse_args()
    # main(args)
    vis(args)
