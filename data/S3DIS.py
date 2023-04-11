#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/25/2022 7:19 PM
# @Author  : Guofeng Mei
# @Email   : Guofeng.Mei@student.uts.edu.au
# @File    : S3DIS.py
# @Software: PyCharm
import h5py
import numpy as np
import os
import sys

from torch.utils.data import Dataset, DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)

# 13 classes, as noted in the meta/s3dis/class_names.txt
num_per_class = np.array([3370714, 2856755, 4919229, 318158, 375640, 478001, 974733,
                          650464, 791496, 88727, 1284130, 229758, 2272837], dtype=np.int32)
num_per_class_dict = {}
for cls, num_cls in enumerate(num_per_class):
    num_per_class_dict[cls] = num_cls


class S3DISDatasetHDF5(Dataset):
    """Chopped Scene"""

    def __init__(self, root, split='train', test_area=5):
        if root is None:
            root = 'data/indoor3d_sem_seg_hdf5_data'
        self.root = root
        self.all_files = self.getDataFiles(os.path.join(self.root, 'all_files.txt'))
        self.room_filelist = self.getDataFiles(os.path.join(self.root, 'room_filelist.txt'))
        self.scene_points_list = []
        self.semantic_labels_list = []
        for h5_filename in self.all_files:
            data_batch, label_batch = self.loadh5DataFile(os.path.join(self.root, h5_filename))
            self.scene_points_list.append(data_batch)
            self.semantic_labels_list.append(label_batch)
        self.data_batches = np.concatenate(self.scene_points_list, 0)
        self.label_batches = np.concatenate(self.semantic_labels_list, 0)
        test_area = 'Area_' + str(test_area)
        train_idxs, test_idxs = [], []
        for i, room_name in enumerate(self.room_filelist):
            if test_area in room_name:
                test_idxs.append(i)
            else:
                train_idxs.append(i)
        assert split in ['train', 'test']
        if split == 'train':
            self.data_batches = self.data_batches[train_idxs, ...]
            self.label_batches = self.label_batches[train_idxs]
        else:
            self.data_batches = self.data_batches[test_idxs, ...]
            self.label_batches = self.label_batches[test_idxs]

    @staticmethod
    def getDataFiles(list_filename):
        return [line.rstrip() for line in open(list_filename)]

    @staticmethod
    def loadh5DataFile(PathtoFile):
        f = h5py.File(PathtoFile, 'r')
        return f['data'][:], f['label'][:]

    def __getitem__(self, index):
        points = self.data_batches[index, :]
        labels = self.label_batches[index].astype(np.int32)
        return points, labels

    def __len__(self):
        return len(self.data_batches)


class S3DISDataset(Dataset):
    def __init__(self, split='train', data_root='trainval_fullarea', num_point=4096, test_area=5,
                 block_size=1.0, sample_rate=1.0, transform=None):
        super().__init__()
        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform
        rooms = sorted(os.listdir(data_root))
        rooms = [room for room in rooms if 'Area_' in room]
        if split == 'train':
            rooms_split = [room for room in rooms if not 'Area_{}'.format(test_area) in room]
        else:
            rooms_split = [room for room in rooms if 'Area_{}'.format(test_area) in room]

        self.room_points, self.room_labels = [], []
        self.room_coord_min, self.room_coord_max = [], []
        num_point_all = []
        labelweights = np.zeros(13)

        for room_name in rooms_split:
            room_path = os.path.join(data_root, room_name)
            room_data = np.load(room_path)  # xyzrgbl, N*7
            points, labels = room_data[:, 0:6], room_data[:, 6]  # xyzrgb, N*6; l, N
            tmp, _ = np.histogram(labels, range(14))
            labelweights += tmp
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.room_points.append(points), self.room_labels.append(labels)
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)
            num_point_all.append(labels.size)
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
        print(self.labelweights)
        sample_prob = num_point_all / np.sum(num_point_all)
        num_iter = int(np.sum(num_point_all) * sample_rate / num_point)
        room_idxs = []
        for index in range(len(rooms_split)):
            room_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))
        self.room_idxs = np.array(room_idxs)
        print("Totally {} samples in {} set.".format(len(self.room_idxs), split))

    def __getitem__(self, idx):
        room_idx = self.room_idxs[idx]
        points = self.room_points[room_idx]   # N * 6
        labels = self.room_labels[room_idx]   # N
        N_points = points.shape[0]

        while True:
            center = points[np.random.choice(N_points)][:3]
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
            point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
            if point_idxs.size > 1024:
                break

        if point_idxs.size >= self.num_point:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)

        # normalize
        selected_points = points[selected_point_idxs, :]  # num_point * 6
        current_points = np.zeros((self.num_point, 9))  # num_point * 9
        current_points[:, 6] = selected_points[:, 0] / self.room_coord_max[room_idx][0]
        current_points[:, 7] = selected_points[:, 1] / self.room_coord_max[room_idx][1]
        current_points[:, 8] = selected_points[:, 2] / self.room_coord_max[room_idx][2]
        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        selected_points[:, 3:6] /= 255.0
        current_points[:, 0:6] = selected_points
        current_labels = labels[selected_point_idxs]
        if self.transform is not None:
            current_points, current_labels = self.transform(current_points, current_labels)
        return current_points, current_labels

    def __len__(self):
        return len(self.room_idxs)


class S3DISDatasetWholeScene(Dataset):
    def __init__(self, root, block_points=8192, split='val', test_area=5, with_rgb=True, use_weight=True,
                 block_size=1.5, stride=1.5, padding=0.001):
        self.npoints = block_points
        self.block_size = block_size
        self.padding = padding
        self.stride = stride
        self.root = root
        self.with_rgb = with_rgb
        self.split = split
        assert split in ['train', 'test']
        if self.split == 'train':
            self.file_list = [d for d in os.listdir(root) if d.find('Area_%d' % test_area) == -1]
        else:
            self.file_list = [d for d in os.listdir(root) if d.find('Area_%d' % test_area) != -1]
        self.scene_points_list = []
        self.semantic_labels_list = []
        for file in self.file_list:
            data = np.load(root + file, allow_pickle=True)
            self.scene_points_list.append(data[:, :6])
            self.semantic_labels_list.append(data[:, 6])
        assert len(self.scene_points_list) == len(self.semantic_labels_list)
        print('Number of scene: ', len(self.scene_points_list))
        if split == 'train' and use_weight:
            labelweights = np.zeros(13)
            for seg in self.semantic_labels_list:
                tmp, _ = np.histogram(seg, range(14))
                labelweights += tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights / np.sum(labelweights)
            self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
        else:
            self.labelweights = np.ones(13)

        print(self.labelweights)

    def __getitem__(self, index):
        if self.with_rgb:
            point_set_ini = self.scene_points_list[index]
            point_set_ini[:, 3:] = 2 * point_set_ini[:, 3:] / 255.0 - 1
        else:
            point_set_ini = self.scene_points_list[index][:, 0:3]
        semantic_seg_ini = self.semantic_labels_list[index].astype(np.int32)
        coordmax = np.max(point_set_ini[:, 0:3], axis=0)
        coordmin = np.min(point_set_ini[:, 0:3], axis=0)
        nsubvolume_x = np.ceil((coordmax[0] - coordmin[0]) / self.block_size).astype(np.int32)
        nsubvolume_y = np.ceil((coordmax[1] - coordmin[1]) / self.block_size).astype(np.int32)
        point_sets = list()
        semantic_segs = list()
        sample_weights = list()
        for i in range(nsubvolume_x):
            for j in range(nsubvolume_y):
                curmin = coordmin + [i * self.block_size, j * self.block_size, 0]
                curmax = coordmin + [(i + 1) * self.block_size, (j + 1) * self.block_size, coordmax[2] - coordmin[2]]
                curchoice = np.sum(
                    (point_set_ini[:, 0:3] >= (curmin - 0.2)) * (point_set_ini[:, 0:3] <= (curmax + 0.2)), axis=1) == 3
                cur_point_set = point_set_ini[curchoice, 0:3]
                cur_point_full = point_set_ini[curchoice, :]
                cur_semantic_seg = semantic_seg_ini[curchoice]
                if len(cur_semantic_seg) == 0:
                    continue
                mask = np.sum((cur_point_set >= (curmin - self.padding)) * (cur_point_set <= (curmax + self.padding)),
                              axis=1) == 3
                choice = np.random.choice(len(cur_semantic_seg), self.npoints, replace=True)
                point_set = cur_point_full[choice, :]  # Nx3/6
                semantic_seg = cur_semantic_seg[choice]  # N
                mask = mask[choice]

                sample_weight = self.labelweights[semantic_seg]
                sample_weight *= mask  # N
                point_sets.append(np.expand_dims(point_set, 0))  # 1xNx3
                semantic_segs.append(np.expand_dims(semantic_seg, 0))  # 1xN
                sample_weights.append(np.expand_dims(sample_weight, 0))  # 1xN
        point_sets = np.concatenate(tuple(point_sets), axis=0)
        semantic_segs = np.concatenate(tuple(semantic_segs), axis=0)
        sample_weights = np.concatenate(tuple(sample_weights), axis=0)
        return point_sets, semantic_segs, sample_weights

    def __len__(self):
        return len(self.scene_points_list)


if __name__ == '__main__':
    data_root = '/data/gmei/data/stanford_indoor3d/'
    num_point, test_area, block_size, sample_rate = 4096, 5, 1.0, 0.01

    point_data = S3DISDataset(split='train', data_root=data_root, num_point=num_point, test_area=test_area,
                              block_size=block_size, sample_rate=sample_rate, transform=None)
    print('point data size:', point_data.__len__())
    print('point data 0 shape:', point_data.__getitem__(0)[0].shape)
    print('point label 0 shape:', point_data.__getitem__(0)[1].shape)
    import torch, time, random
    manual_seed = 123
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    def worker_init_fn(worker_id):
        random.seed(manual_seed + worker_id)
    train_loader = torch.utils.data.DataLoader(point_data, batch_size=16, shuffle=True, num_workers=16,
                                               pin_memory=True, worker_init_fn=worker_init_fn)
    for idx in range(4):
        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            print('time: {}/{}--{}'.format(i+1, len(train_loader), time.time() - end))
            end = time.time()
