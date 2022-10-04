#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/14/2022 10:59 PM
# @Author  : Guofeng Mei
# @Email   : Guofeng.Mei@student.uts.edu.au
# @File    : shapenet.py
# @Software: PyCharm
import glob
import os
import sys

import numpy as np
import torch
import torchvision.transforms as transforms
from plyfile import PlyData
from torch.utils.data import Dataset

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
        common.PointcloudJitter(p=1),
        # common.PointcloudNormalize(True)
    ])


def load_ply(file_name: str, with_faces: bool = False, with_color: bool = False):
    ply_data = PlyData.read(file_name)
    points = ply_data['vertex']
    points = np.vstack([points['x'], points['y'], points['z']]).T
    ret_val = [points]

    if with_faces:
        faces = np.vstack(ply_data['face']['vertex_indices'])
        ret_val.append(faces)

    if with_color:
        r = np.vstack(ply_data['vertex']['red'])
        g = np.vstack(ply_data['vertex']['green'])
        b = np.vstack(ply_data['vertex']['blue'])
        color = np.hstack((r, g, b))
        ret_val.append(color)

    if len(ret_val) == 1:  # Unwrap the list
        ret_val = ret_val[0]

    return ret_val


def load_shapenet_path(data_dir):
    all_filepath = []

    for cls in glob.glob(os.path.join(data_dir, 'ShapeNet/*')):
        pcs = glob.glob(os.path.join(cls, '*'))
        all_filepath += pcs

    return all_filepath


class ShapeNetClr(Dataset):
    def __init__(self, root, n_points=1024, transform=False, fps=False):
        self.data = load_shapenet_path(root)
        self.n_points = n_points
        self.fps = fps
        self.transform = transform

    def __getitem__(self, item):
        pcd_path = self.data[item]
        points = load_ply(pcd_path)
        points1 = trans(points).numpy()
        points2 = trans(points).numpy()
        num1 = points1.shape[0]
        if num1 > self.n_points:
            points1 = points_sampler(points1, self.n_points)
        num2 = points2.shape[0]
        if num2 > self.n_points:
            points2 = points_sampler(points2, self.n_points)
        points1[:, 0:3] = pc_normalize(points1[:, 0:3])
        points2[:, 0:3] = pc_normalize(points2[:, 0:3])
        return points1, points2

    def __len__(self):
        return len(self.data)


class ShapeNet(Dataset):
    def __init__(self, root, n_points=1024, transform=False, fps=False):
        self.data = load_shapenet_path(root)
        self.n_points = n_points
        self.fps = fps
        self.transform = transform

    def __getitem__(self, item):
        pcd_path = self.data[item]
        points = load_ply(pcd_path)
        if self.transform:
            points = trans(points).numpy()
        num = points.shape[0]
        if num > self.n_points:
            points = points_sampler(points, self.n_points)
        points[:, 0:3] = pc_normalize(points[:, 0:3])
        return points

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    root = '/data1/gmei/data'
    dataset = ShapeNet(root, 1024, transform=True)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=6, shuffle=True)
    for i, data in enumerate(train_loader):
        print(data.shape)
        print(data)
        break
