#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/26/2022 4:52 PM
import copy
import os
import pickle
import random
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'data'))
from data.common import points_sampler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    from plyfile import PlyData, PlyElement
except:
    print("Please install the module 'plyfile' for PLY i/o, e.g.")
    print("pip install plyfile")
    sys.exit(-1)


def write_ply(points, filename, text=True):
    """ input: Nx3, write points to filename as PLY format. """
    points = [(points[i, 0], points[i, 1], points[i, 2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(filename)


class ScannetWholeScene(Dataset):
    def __init__(self, root, num_points, transforms=None, train=True):
        self.npoints = num_points
        self.transforms = transforms
        self.root = os.path.join(root, "scannet")
        if train:
            self.split = "train"
        else:
            self.split = "test"
        self.data_filename = os.path.join(self.root, 'scannet_%s.pickle' % (self.split))
        with open(self.data_filename, 'rb') as fp:
            self.scene_points_list = pickle.load(fp, encoding="bytes")

    def __getitem__(self, index):
        point_set = self.scene_points_list[index]

        if self.transforms is not None:
            point_set = self.transforms(point_set)
        point_set = points_sampler(point_set, self.npoints)

        return point_set

    def __len__(self):
        return len(self.scene_points_list)


class ScannetWholeSceneHeight(Dataset):
    def __init__(self, num_points, transforms=None, transforms_2=None, train=True, no_height=True):
        self.npoints = num_points
        self.transforms = transforms
        self.transforms_2 = transforms_2
        self.no_height = no_height
        self.root = os.path.join(BASE_DIR, "scannet")
        if train:
            self.split = "train"
        else:
            self.split = "test"
        self.data_filename = os.path.join(self.root, 'scannet_%s.pickle' % (self.split))
        with open(self.data_filename, 'rb') as fp:
            self.scene_points_list = pickle.load(fp, encoding="bytes")

    def __getitem__(self, index):
        point_set = self.scene_points_list[index]
        if self.transforms is not None:
            point_set = self.transforms(point_set)
        point_set = point_set.numpy()
        if not self.no_height:
            floor = np.percentile(point_set[:, 2], 0.99)
        point_set = points_sampler(point_set, self.npoints)
        if not self.no_height:
            height = point_set[:, 2] - floor
            height = torch.unsqueeze(height, 1)
            point_set = torch.cat([point_set, height], 1)

        return point_set

    def __len__(self):
        return len(self.scene_points_list)


class ScanNetFrameContrast(Dataset):
    def __init__(self, num_points, transforms_1=None, transforms_2=None, no_height=True, mode="temporal"):
        self.npoints = num_points
        self.transforms_1 = transforms_1
        self.transforms_2 = transforms_2
        self.no_height = no_height
        self.root_path = os.path.join(BASE_DIR, "scannet", "scannet_frames_25k")
        self.mode = mode
        assert mode in ["spatial", "temporal", "both"]

        self.load_filenames()

    def load_filenames(self):
        self.scenes = os.listdir(self.root_path)
        self.scenes.sort()
        self.depth_files = {}  # {scene: [depth_files]}
        self.cam_pose_files = {}  # {scene: [cam_pose_files]}
        self.frame_num = {}  # {scene: frame_num}
        self.total_num = 0
        self.frame_idx = []  # [(scene, id_in_scene)]

        for scene in self.scenes:
            scene_dir = os.path.join(self.root_path, scene)
            depth_dir = os.path.join(scene_dir, "depth")
            self.depth_files[scene] = os.listdir(depth_dir)  # 000100.png, 000200.png...
            self.depth_files[scene].sort(key=lambda f: int(f.split('.')[0]))
            cam_pose_dir = os.path.join(scene_dir, "pose")
            self.cam_pose_files[scene] = os.listdir(cam_pose_dir)  # 000100.txt, 000200.txt...
            self.cam_pose_files[scene].sort(key=lambda f: int(f.split('.')[0]))
            self.frame_num[scene] = len(self.depth_files[scene])
            self.total_num += self.frame_num[scene]

            for idx in range(self.frame_num[scene]):
                self.frame_idx.append((scene, idx))

    def get_adjacent(self, scene, frameidx, index, both=False):
        if self.frame_num[scene] == 1:
            scene_adj, frameidx_adj = self.frame_idx[index]
        elif frameidx == self.frame_num[scene] - 1:
            if both:
                scene_adj, frameidx_adj = random.choice([self.frame_idx[index - 1], self.frame_idx[index]])
            else:
                scene_adj, frameidx_adj = self.frame_idx[index - 1]
        elif frameidx == 0:
            if both:
                scene_adj, frameidx_adj = random.choice([self.frame_idx[index + 1], self.frame_idx[index]])
            else:
                scene_adj, frameidx_adj = self.frame_idx[index + 1]
        else:
            if both:
                scene_adj, frameidx_adj = random.choice(
                    [self.frame_idx[index - 1], self.frame_idx[index + 1], self.frame_idx[index]])
            else:
                scene_adj, frameidx_adj = random.choice([self.frame_idx[index - 1], self.frame_idx[index + 1]])
        return scene_adj, frameidx_adj

    def __len__(self):
        return self.total_num

    def __getitem__(self, index):
        scene, idx = self.frame_idx[index]

        cam_pose_file = self.cam_pose_files[scene][idx]
        cam_pose_file = os.path.join(self.root_path, scene, "pose", cam_pose_file)
        cam_pose = np.loadtxt(cam_pose_file)
        global_translation = cam_pose[:3, 3]

        if self.mode == "temporal":
            scene_adj, idx_adj = self.get_adjacent(scene, idx, index)
            point_set = self.get_point_cloud(scene, idx, global_translation=global_translation)
            point_set_ = self.get_point_cloud(scene_adj, idx_adj, global_translation=global_translation)
        elif self.mode == "spatial":
            point_set = self.get_point_cloud(scene, idx, global_translation=global_translation)
            point_set_ = copy.deepcopy(point_set)
        else:
            assert self.mode == "both"
            scene_adj, idx_adj = self.get_adjacent(scene, idx, index, both=True)
            point_set = self.get_point_cloud(scene, idx, global_translation=global_translation)
            point_set_ = self.get_point_cloud(scene_adj, idx_adj, global_translation=global_translation)

        if self.transforms_1 is not None:
            point_set = self.transforms_1(point_set)
            point_set_ = self.transforms_1(point_set_)

        point_set = point_set.numpy()
        point_set_ = point_set_.numpy()

        if not self.no_height:
            floor = np.percentile(point_set[:, 2], 0.99)
            floor_ = np.percentile(point_set_[:, 2], 0.99)

        if self.transforms_2 is not None:
            point_set = self.transforms_2(point_set)
            point_set_ = self.transforms_2(point_set_)

        point_set = points_sampler(point_set, self.npoints)
        point_set_ = points_sampler(point_set_, self.npoints)

        if not self.no_height:
            height = point_set[:, 2] - floor
            height_ = point_set_[:, 2] - floor_

            height = torch.unsqueeze(height, 1)
            height_ = torch.unsqueeze(height_, 1)
            point_set = torch.cat([point_set, height], 1)
            point_set_ = torch.cat([point_set_, height_], 1)
        return point_set, point_set_

    def get_point_cloud(self, scene, frameidx, global_translation, depth_scale=1000):
        depth_file = self.depth_files[scene][frameidx]
        depth_file = os.path.join(self.root_path, scene, "depth", depth_file)
        cam_pose_file = self.cam_pose_files[scene][frameidx]
        cam_pose_file = os.path.join(self.root_path, scene, "pose", cam_pose_file)
        intrinsics_file = os.path.join(self.root_path, scene, "intrinsics_depth.txt")
        depth_map = np.asarray(Image.open(depth_file))
        # depth_map = cv2.imread(depth_file, cv2.IMREAD_GRAYSCALE)
        cam_pose = np.loadtxt(cam_pose_file)
        # cam_pose[:3, :]
        cam_pose[:3, 3] -= global_translation
        depth_cam_matrix = np.loadtxt(intrinsics_file)

        fx, fy = depth_cam_matrix[0, 0], depth_cam_matrix[1, 1]
        cx, cy = depth_cam_matrix[0, 2], depth_cam_matrix[1, 2]
        h, w = np.mgrid[0:depth_map.shape[0], 0:depth_map.shape[1]]
        z = depth_map / depth_scale
        x = (w - cx) * z / fx
        y = (h - cy) * z / fy
        xyz = np.dstack((x, y, z))
        pad = np.ones((xyz.shape[0], xyz.shape[1], 1))
        xyz = np.concatenate([xyz, pad], 2)
        height, width, _ = xyz.shape
        xyz = xyz.transpose([2, 0, 1])

        xyz = xyz.reshape(4, -1)
        xyz = np.matmul(cam_pose, xyz)
        # xyz = np.matmul(cam_pose_inverse, xyz)
        xyz = xyz.reshape(4, height, width)

        xyz = xyz.transpose([1, 2, 0])

        pc = xyz[:, :, :3]
        pc = pc.reshape(-1, 3)
        return pc


class ScannetDatasetWholeScene(Dataset):
    # prepare to give prediction on each points
    def __init__(self, root, block_points=4096, split='test', test_area=5, stride=0.5, block_size=1.0, padding=0.001):
        self.block_points = block_points
        self.block_size = block_size
        self.padding = padding
        self.root = root
        self.split = split
        self.stride = stride
        self.scene_points_num = []
        assert split in ['train', 'test']
        if self.split == 'train':
            self.file_list = [d for d in os.listdir(root) if d.find('Area_%d' % test_area) is -1]
        else:
            self.file_list = [d for d in os.listdir(root) if d.find('Area_%d' % test_area) is not -1]
        self.scene_points_list = []
        self.semantic_labels_list = []
        self.room_coord_min, self.room_coord_max = [], []
        for file in self.file_list:
            data = np.load(root + file)
            points = data[:, :3]
            self.scene_points_list.append(data[:, :6])
            self.semantic_labels_list.append(data[:, 6])
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)
        assert len(self.scene_points_list) == len(self.semantic_labels_list)

        labelweights = np.zeros(13)
        for seg in self.semantic_labels_list:
            tmp, _ = np.histogram(seg, range(14))
            self.scene_points_num.append(seg.shape[0])
            labelweights += tmp
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)

    def __getitem__(self, index):
        point_set_ini = self.scene_points_list[index]
        points = point_set_ini[:,:6]
        labels = self.semantic_labels_list[index]
        coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
        grid_x = int(np.ceil(float(coord_max[0] - coord_min[0] - self.block_size) / self.stride) + 1)
        grid_y = int(np.ceil(float(coord_max[1] - coord_min[1] - self.block_size) / self.stride) + 1)
        data_room, label_room, sample_weight, index_room = np.array([]), np.array([]), np.array([]),  np.array([])
        for index_y in range(0, grid_y):
            for index_x in range(0, grid_x):
                s_x = coord_min[0] + index_x * self.stride
                e_x = min(s_x + self.block_size, coord_max[0])
                s_x = e_x - self.block_size
                s_y = coord_min[1] + index_y * self.stride
                e_y = min(s_y + self.block_size, coord_max[1])
                s_y = e_y - self.block_size
                point_idxs = np.where(
                    (points[:, 0] >= s_x - self.padding) & (points[:, 0] <= e_x + self.padding) & (points[:, 1] >= s_y - self.padding) & (
                                points[:, 1] <= e_y + self.padding))[0]
                if point_idxs.size == 0:
                    continue
                num_batch = int(np.ceil(point_idxs.size / self.block_points))
                point_size = int(num_batch * self.block_points)
                replace = False if (point_size - point_idxs.size <= point_idxs.size) else True
                point_idxs_repeat = np.random.choice(point_idxs, point_size - point_idxs.size, replace=replace)
                point_idxs = np.concatenate((point_idxs, point_idxs_repeat))
                np.random.shuffle(point_idxs)
                data_batch = points[point_idxs, :]
                normlized_xyz = np.zeros((point_size, 3))
                normlized_xyz[:, 0] = data_batch[:, 0] / coord_max[0]
                normlized_xyz[:, 1] = data_batch[:, 1] / coord_max[1]
                normlized_xyz[:, 2] = data_batch[:, 2] / coord_max[2]
                data_batch[:, 0] = data_batch[:, 0] - (s_x + self.block_size / 2.0)
                data_batch[:, 1] = data_batch[:, 1] - (s_y + self.block_size / 2.0)
                data_batch[:, 3:6] /= 255.0
                data_batch = np.concatenate((data_batch, normlized_xyz), axis=1)
                label_batch = labels[point_idxs].astype(int)
                batch_weight = self.labelweights[label_batch]

                data_room = np.vstack([data_room, data_batch]) if data_room.size else data_batch
                label_room = np.hstack([label_room, label_batch]) if label_room.size else label_batch
                sample_weight = np.hstack([sample_weight, batch_weight]) if label_room.size else batch_weight
                index_room = np.hstack([index_room, point_idxs]) if index_room.size else point_idxs
        data_room = data_room.reshape((-1, self.block_points, data_room.shape[1]))
        label_room = label_room.reshape((-1, self.block_points))
        sample_weight = sample_weight.reshape((-1, self.block_points))
        index_room = index_room.reshape((-1, self.block_points))
        return data_room, label_room, sample_weight, index_room

    def __len__(self):
        return len(self.scene_points_list)

