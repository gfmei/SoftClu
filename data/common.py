#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/14/2022 11:08 PM
# @Author  : Guofeng Mei
# @Email   : Guofeng.Mei@student.uts.edu.au
# @File    : common.py
# @Software: PyCharm
import numpy as np
import torch


def angle_axis(angle, axis):
    r"""Returns a 4x4 rotation matrix that performs a rotation around axis by angle
    Parameters
    ----------
    angle : float
        Angle to rotate by
    axis: np.ndarray
        Axis to rotate about
    Returns
    -------
    torch.Tensor
    3x3 rotation matrix
    """
    u = axis / np.linalg.norm(axis)
    cosval, sinval = np.cos(angle), np.sin(angle)

    # yapf: disable
    cross_prod_mat = np.array([[0.0, -u[2], u[1]],
                               [u[2], 0.0, -u[0]],
                               [-u[1], u[0], 0.0]])

    R = torch.from_numpy(
        cosval * np.eye(3)
        + sinval * cross_prod_mat
        + (1.0 - cosval) * np.outer(u, u)
    )
    return R.float()


class PointcloudScale(object):
    def __init__(self, lo=0.8, hi=1.25, p=1):
        self.lo, self.hi = lo, hi
        self.p = p

    def __call__(self, points):
        if np.random.uniform(0, 1) > self.p:
            return points
        scaler = np.random.uniform(self.lo, self.hi)
        points[:, 0:3] *= scaler
        return points


class PointcloudRotate(object):
    def __init__(self, axis=np.array([0.0, 1.0, 0.0]), p=1):
        self.axis = axis
        self.p = p

    def __call__(self, points):
        if np.random.uniform(0, 1) > self.p:
            return points
        if self.axis is None:
            angles = np.random.uniform(size=3) * 2 * np.pi
            Rx = angle_axis(angles[0], np.array([1.0, 0.0, 0.0]))
            Ry = angle_axis(angles[1], np.array([0.0, 1.0, 0.0]))
            Rz = angle_axis(angles[2], np.array([0.0, 0.0, 1.0]))

            rotation_matrix = torch.matmul(torch.matmul(Rz, Ry), Rx)
        else:
            rotation_angle = np.random.uniform() * 2 * np.pi
            rotation_matrix = angle_axis(rotation_angle, self.axis)

        normals = points.size(1) > 3
        if not normals:
            return torch.matmul(points, rotation_matrix.t())
        else:
            pc_xyz = points[:, 0:3]
            pc_normals = points[:, 3:]
            points[:, 0:3] = torch.matmul(pc_xyz, rotation_matrix.t())
            points[:, 3:] = torch.matmul(pc_normals, rotation_matrix.t())

            return points


class PointcloudRotatePerturbation(object):
    def __init__(self, angle_sigma=0.06, angle_clip=0.18, p=1):
        self.angle_sigma, self.angle_clip = angle_sigma, angle_clip
        self.p = p

    def _get_angles(self):
        angles = np.clip(
            self.angle_sigma * np.random.randn(3), -self.angle_clip, self.angle_clip
        )

        return angles

    def __call__(self, points):
        if np.random.uniform(0, 1) > self.p:
            return points
        angles = self._get_angles()
        Rx = angle_axis(angles[0], np.array([1.0, 0.0, 0.0]))
        Ry = angle_axis(angles[1], np.array([0.0, 1.0, 0.0]))
        Rz = angle_axis(angles[2], np.array([0.0, 0.0, 1.0]))
        rotation_matrix = torch.matmul(torch.matmul(Rz, Ry), Rx)
        normals = points.size(1) > 3
        if not normals:
            return torch.matmul(points, rotation_matrix.t())
        else:
            pc_xyz = points[:, 0:3]
            pc_normals = points[:, 3:]
            points[:, 0:3] = torch.matmul(pc_xyz, rotation_matrix.t())
            points[:, 3:] = torch.matmul(pc_normals, rotation_matrix.t())

            return points


class PointcloudJitter(object):
    def __init__(self, std=0.001, clip=0.0025, p=1):
        self.std, self.clip = std, clip
        self.p = p

    def __call__(self, points):
        if np.random.uniform(0, 1) > self.p:
            return points
        jittered_data = (
            points.new(points.size(0), 3).normal_(mean=0.0, std=self.std).clamp_(-self.clip, self.clip)
        )
        points[:, 0:3] += jittered_data
        return points


class Jitter(object):
    def __init__(self, sigma=0.001, clip=0.005):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, points):
        N, C = points.shape
        points += np.clip(self.sigma * np.random.randn(N, C), -1 * self.clip, self.clip)
        return points


class PointcloudTranslate(object):
    def __init__(self, translate_range=0.1, p=1):
        self.translate_range = translate_range
        self.p = p

    def __call__(self, points):
        if np.random.uniform(0, 1) > self.p:
            return points

        points = points.numpy()
        coord_min = np.min(points[:, :3], axis=0)
        coord_max = np.max(points[:, :3], axis=0)
        coord_diff = coord_max - coord_min
        translation = np.random.uniform(-self.translate_range, self.translate_range, size=(3)) * coord_diff
        points[:, 0:3] += translation
        return torch.from_numpy(points).float()


class PointcloudToTensor(object):
    def __call__(self, points):
        return torch.from_numpy(points).float()


class PointcloudRandomInputDropout(object):
    def __init__(self, max_dropout_ratio=0.875, p=1):
        assert 0 <= max_dropout_ratio < 1
        self.max_dropout_ratio = max_dropout_ratio
        self.p = p

    def __call__(self, points):
        if np.random.uniform(0, 1) > self.p:
            return points
        pc = points.numpy()

        dropout_ratio = np.random.random() * self.max_dropout_ratio  # 0~0.875
        drop_idx = np.where(np.random.random((pc.shape[0])) <= dropout_ratio)[0]
        if len(drop_idx) > 0:
            pc[drop_idx] = pc[0]  # set to the first point

        return torch.from_numpy(pc).float()


class PointcloudSample(object):
    def __init__(self, num_pt=4096, fs=False):
        self.num_points = num_pt
        self.fs = fs

    def __call__(self, points):
        pc = points.numpy()
        pc = points_sampler(pc, self.num_points, self.fs)
        return torch.from_numpy(pc).float()


def torch_pc_normalize(tensor, zero_mean=True):
    if zero_mean:
        m = tensor.mean(dim=0, keepdim=True)  # [N, D] -> [1, D]
        v = tensor - m
    else:
        v = tensor
    nn = v.norm(p=2, dim=1)  # [N, D] -> [N]
    nmax = torch.max(nn)
    return v / nmax


class PointcloudRemoveInvalid(object):
    def __init__(self, invalid_value=0):
        self.invalid_value = invalid_value

    def __call__(self, points):
        pc = points.numpy()
        valid = np.sum(pc, axis=1) != self.invalid_value
        pc = pc[valid, :]
        return torch.from_numpy(pc).float()


class PointcloudRandomCrop(object):
    def __init__(self, x_min=0.6, x_max=1.1, ar_min=0.75, ar_max=1.33, p=1,
                 min_num_points=4096, max_try_num=10):
        self.x_min = x_min
        self.x_max = x_max

        self.ar_min = ar_min
        self.ar_max = ar_max

        self.p = p

        self.max_try_num = max_try_num
        self.min_num_points = min_num_points

    def __call__(self, points):
        if np.random.uniform(0, 1) > self.p:
            return points
        points = points.numpy()

        isvalid = False
        try_num = 0
        while not isvalid:
            coord_min = np.min(points[:, :3], axis=0)
            coord_max = np.max(points[:, :3], axis=0)
            coord_diff = coord_max - coord_min
            # resampling later, so only consider crop here
            new_coord_range = np.zeros(3)
            new_coord_range[0] = np.random.uniform(self.x_min, self.x_max)
            ar = np.random.uniform(self.ar_min, self.ar_max)
            # new_coord_range[1] = np.random.uniform(self.ar_min, self.ar_max) * new_coord_range[0]
            # new_coord_range[2] = np.random.uniform(self.ar_min, self.ar_max) * new_coord_range[0]
            new_coord_range[1] = new_coord_range[0] * ar
            new_coord_range[2] = new_coord_range[0] / ar
            # new_coord_range = np.where(new_coord_range>1, 1, new_coord_range)

            new_coord_min = np.random.uniform(0, 1 - new_coord_range)
            new_coord_max = new_coord_min + new_coord_range

            new_coord_min = coord_min + coord_diff * new_coord_min
            new_coord_max = coord_min + coord_diff * new_coord_max

            new_indices = (points[:, :3] > new_coord_min) & (points[:, :3] < new_coord_max)
            new_indices = np.sum(new_indices, axis=1) == 3
            new_points = points[new_indices]

            # other_num = points.shape[0] - new_points.shape[0]
            # if new_points.shape[0] > 0:
            #     isvalid = True
            if self.min_num_points <= new_points.shape[0] < points.shape[0]:
                isvalid = True

            try_num += 1
            if try_num > self.max_try_num:
                return torch.from_numpy(points).float()
        return torch.from_numpy(new_points).float()


class PointcloudRandomCutout(object):
    def __init__(self, ratio_min=0.3, ratio_max=0.6, p=1, min_num_points=4096, max_try_num=10):
        self.ratio_min = ratio_min
        self.ratio_max = ratio_max
        self.p = p
        self.min_num_points = min_num_points
        self.max_try_num = max_try_num

    def __call__(self, points):
        if np.random.uniform(0, 1) > self.p:
            return points
        points = points.numpy()
        try_num = 0
        valid = False
        while not valid:
            coord_min = np.min(points[:, :3], axis=0)
            coord_max = np.max(points[:, :3], axis=0)
            coord_diff = coord_max - coord_min

            cut_ratio = np.random.uniform(self.ratio_min, self.ratio_max, 3)
            new_coord_min = np.random.uniform(0, 1 - cut_ratio)
            new_coord_max = new_coord_min + cut_ratio

            new_coord_min = coord_min + new_coord_min * coord_diff
            new_coord_max = coord_min + new_coord_max * coord_diff

            cut_indices = (points[:, :3] > new_coord_min) & (points[:, :3] < new_coord_max)
            cut_indices = np.sum(cut_indices, axis=1) == 3

            # print(np.sum(cut_indices))
            # other_indices = (points[:, :3] < new_coord_min) | (points[:, :3] > new_coord_max)
            # other_indices = np.sum(other_indices, axis=1) == 3
            try_num += 1

            if try_num > self.max_try_num:
                return torch.from_numpy(points).float()

            # cut the points, sampling later

            if points.shape[0] - np.sum(cut_indices) >= self.min_num_points and np.sum(cut_indices) > 0:
                # print (np.sum(cut_indices))
                points = points[cut_indices == False]
                valid = True

        # if np.sum(other_indices) > 0:
        #     comp_indices = np.random.choice(np.arange(np.sum(other_indices)), np.sum(cut_indices))
        #     points[cut_indices] = points[comp_indices]
        return torch.from_numpy(points).float()


class PointcloudNormalize(object):
    def __init__(self, tensor=True):
        self.tensor = tensor

    def __call__(self, pc):
        if self.tensor:
            pc = torch_pc_normalize(pc)
        else:
            pc = torch.from_numpy(pc_normalize(pc)).float()
        return pc


def uniform_2_sphere(num: int = None):
    """Uniform sampling on a 2-sphere
    Source: https://gist.github.com/andrewbolster/10274979
    Args:
        num: Number of vectors to sample (or None if single)
    Returns:
        Random Vector (np.ndarray) of size (num, 3) with norm 1.
        If num is None returned value will have size (3,)
    """
    if num is not None:
        phi = np.random.uniform(0.0, 2 * np.pi, num)
        cos_theta = np.random.uniform(-1.0, 1.0, num)
    else:
        phi = np.random.uniform(0.0, 2 * np.pi)
        cos_theta = np.random.uniform(-1.0, 1.0)

    theta = np.arccos(cos_theta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return np.stack((x, y, z), axis=-1)


class PointcloudSphereCrop(object):
    """Randomly crops the *source* point cloud, approximately retaining half the points
    A direction is randomly sampled from S2, and we retain points which lie within the
    half-space oriented in this direction.
    If p_keep != 0.5, we shift the plane until approximately p_keep points are retained
    """

    def __init__(self, p_keep=0.85):
        self.p_keep = p_keep

    @staticmethod
    def crop(points, p_keep):
        rand_xyz = uniform_2_sphere()
        centroid = np.mean(points[:, :3], axis=0)
        points_centered = points[:, :3] - centroid

        dist_from_plane = np.dot(points_centered, rand_xyz)
        if p_keep == 0.5:
            mask = dist_from_plane > 0
        else:
            mask = dist_from_plane > np.percentile(dist_from_plane, (1.0 - p_keep) * 100)

        return points[mask, :]

    def __call__(self, sample):

        if self.p_keep >= 1.0:
            return sample  # No need crop
        return self.crop(sample, self.p_keep)


class PointcloudUpSampling(object):
    def __init__(self, max_num_points, nsample=5, centroid="random"):
        self.max_num_points = max_num_points
        self.centroid = centroid
        self.nsample = nsample

    def __call__(self, points):
        p_num = points.shape[0]
        if p_num > self.max_num_points:
            return points
        c_num = self.max_num_points - p_num
        if self.centroid == "random":
            cids = np.random.choice(np.arange(p_num), c_num)
        else:
            assert self.centroid == "fps"
            fps_num = c_num / self.nsample
            fps_ids = farthest_point_sample(points, fps_num)
            cids = np.random.choice(fps_ids, c_num)
        xyzs = points[:, :3]
        loc_matmul = torch.matmul(xyzs, xyzs.t())
        loc_norm = xyzs * xyzs
        r = torch.sum(loc_norm, -1, keepdim=True)
        r_t = r.t()
        dist = r - 2 * loc_matmul + r_t
        # adj_matrix = torch.sqrt(dist + 1e-6)
        dist = dist[cids]
        # adj_sort = torch.argsort(adj_matrix, 1)
        adj_topk = torch.topk(dist, k=self.nsample * 2, dim=1, largest=False)[1]
        uniform = np.random.uniform(0, 1, (cids.shape[0], self.nsample * 2))
        median = np.median(uniform, axis=1, keepdims=True)
        # choice = adj_sort[:, 0:self.nsample*2][uniform > median]  # (c_num, n_samples)
        choice = adj_topk[uniform > median]  # (c_num, n_samples)
        choice = choice.reshape(-1, self.nsample)
        sample_points = points[choice]  # (c_num, n_samples, 3)
        new_points = torch.mean(sample_points, dim=1)
        new_points = torch.cat([points, new_points], 0)

        return new_points


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def points_sampler(points, num, fs=False):
    if fs:
        return farthest_point_sample(points, num)
    pt_idxs = np.arange(0, points.shape[0])
    np.random.shuffle(pt_idxs)
    points = points[pt_idxs[0:num], :]
    return points


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point
