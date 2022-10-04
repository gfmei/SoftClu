#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/15/2022 11:20 AM
# @Author  : Guofeng Mei
# @Email   : Guofeng.Mei@student.uts.edu.au
# @File    : modelnet.py
# @Software: PyCharm
import glob
import os
import sys

import h5py
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../data'))
from data import common
from data.common import pc_normalize, points_sampler

trans = transforms.Compose(
    [
        common.PointcloudSphereCrop(0.85),
        common.PointcloudToTensor(),
        common.PointcloudRandomInputDropout(p=1),
        common.PointcloudScale(lo=0.5, hi=2, p=1),
        common.PointcloudRotate(),
        common.PointcloudTranslate(0.5, p=1),
        common.PointcloudJitter(p=1)
        # common.PointcloudNormalize(True)
    ])


def load_scanobjectnn(root, partition):
    data_dir = os.path.join(root, 'ScanObjectNN', 'main_split')
    h5_name = os.path.join(data_dir, f'{partition}.h5')
    f = h5py.File(h5_name, 'r')
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')

    return data, label


def load_modelnet_data(data_dir, partition):
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(data_dir, 'modelnet40_ply_hdf5_2048',
                                          'ply_data_%s*.h5' % partition)):
        f = h5py.File(h5_name, 'r')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


class ModelNet40SVM(Dataset):
    def __init__(self, root, num_points, partition='train'):
        self.data, self.label = load_modelnet_data(root, partition)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        points = self.data[item]
        points = points_sampler(points, self.num_points)
        points[:, 0:3] = pc_normalize(points[:, 0:3])
        label = self.label[item]
        return points, label

    def __len__(self):
        return self.data.shape[0]


class ModelNet40(Dataset):
    def __init__(self, root, num_points, transform=True, train=True, normalize=True,
                 xyz_only=True, uniform=False, subset10=False, cache_size=15000):
        self.root = os.path.join(root, 'modelnet40_normal_resampled')
        self.num_points = num_points
        self.uniform = uniform
        self.normalize = normalize
        self.transform = transform
        if subset10:
            name = 'modelnet10_shape_names.txt'
        else:
            name = 'modelnet40_shape_names.txt'
        self.catfile = os.path.join(self.root, name)

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.xyz_only = xyz_only
        if subset10:
            train_name = 'modelnet10_train.txt'
            test_name = 'modelnet10_test.txt'
        else:
            train_name = 'modelnet40_train.txt'
            test_name = 'modelnet40_test.txt'
        shape_ids = {'train': [line.rstrip() for line in open(os.path.join(self.root, train_name))],
                     'test': [line.rstrip() for line in open(os.path.join(self.root, test_name))]}
        if train:
            split = 'train'
        else:
            split = 'test'
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d' % (split, len(self.datapath)))
        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if index in self.cache:
            point_set, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            cls = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
            if self.transform:
                point_set = trans(point_set).numpy()
            point_set = points_sampler(point_set, self.num_points, self.uniform)
            if self.normalize:
                point_set[:, :3] = pc_normalize(point_set[:, :3])
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls)
        if self.xyz_only:
            point_set = point_set[:, 0:3]

        return point_set, cls

    def __getitem__(self, index):
        return self._get_item(index)


def _build_dataloader(dset, mode, batch_size=32, num_workers=4):
    return DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=mode == "train",
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True, )


def mnet_dataloader(root, num_points, batch_size, transform=True, normalize=True,
                    xyz_only=True, uniform=False, subset10=True, num_workers=4):
    train_dset = ModelNet40(root, num_points, transform, True, normalize, xyz_only,
                            uniform, subset10)
    test_dset = ModelNet40(root, num_points, transform, False, normalize, xyz_only,
                           uniform, subset10)
    train_loader = _build_dataloader(train_dset, mode="train", batch_size=batch_size,
                                     num_workers=num_workers)
    test_loader = _build_dataloader(test_dset, mode="val", batch_size=batch_size,
                                    num_workers=num_workers)
    return train_loader, test_loader
