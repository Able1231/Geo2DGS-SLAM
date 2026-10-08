import os
import numpy as np
import torch
import lietorch
import droid_backends
import matplotlib.pyplot as plt

from torch.multiprocessing import Value
from lietorch import SE3
from src.modules.droid_net import cvx_upsample
import src.geom.projective_ops as pops
from src.utils.pose_transform import quaternion_to_transform_noBatch



class DepthVideo:
    def __init__(self, config, buffer=1024, stereo=False, device="cuda:0"):
        self.config = config
        # current keyframe count
        self.counter = Value('i', 0)
        self.ready = Value('i', 0)
        self.ht = ht = 240
        self.wd = wd = 320

        self.ht_8 = self.ht // 8
        self.wd_8 = self.wd // 8

        self.coords0 = pops.coords_grid(self.ht_8, self.wd_8, device=device)

        ### state attributes ###
        self.tstamp = torch.zeros(buffer, device="cuda", dtype=torch.float).share_memory_()
        self.images = torch.zeros(buffer, 3, ht, wd, device="cuda", dtype=torch.uint8)
        self.dirty = torch.zeros(buffer, device="cuda", dtype=torch.bool).share_memory_()
        self.red = torch.zeros(buffer, device="cuda", dtype=torch.bool).share_memory_()
        self.poses = torch.zeros(buffer, 7, device="cuda", dtype=torch.float).share_memory_()
        self.disps = torch.ones(buffer, ht // 8, wd // 8, device="cuda", dtype=torch.float).share_memory_()
        self.disps_sens = torch.zeros(buffer, ht // 8, wd // 8, device="cuda", dtype=torch.float).share_memory_()
        self.disps_up = torch.zeros(buffer, ht, wd, device="cuda", dtype=torch.float).share_memory_()
        self.depths_gt = torch.zeros(buffer, ht, wd, device="cuda", dtype=torch.float).share_memory_()
        self.intrinsics = torch.zeros(buffer, 4, device="cuda", dtype=torch.float).share_memory_()
        self.poses_gt = torch.zeros(buffer, 7, device="cuda", dtype=torch.float32).share_memory_()
        self.zeros = torch.zeros(buffer, ht//8, wd//8, device="cuda", dtype=torch.float).share_memory_()
        self.stereo = stereo
        c = 1 if not self.stereo else 2

        ### feature attributes ###
        self.fmaps = torch.zeros(buffer, c, 128, ht // 8, wd // 8, dtype=torch.half, device="cuda").share_memory_()
        self.nets = torch.zeros(buffer, 128, ht // 8, wd // 8, dtype=torch.half, device="cuda").share_memory_()
        self.inps = torch.zeros(buffer, 128, ht // 8, wd // 8, dtype=torch.half, device="cuda").share_memory_()

        # initialize poses to identity transformation
        self.poses[:] = torch.as_tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device="cuda")


    def get_lock(self):
        return self.counter.get_lock()

    def __item_setter(self, index, item):
        if isinstance(index, int) and index >= self.counter.value:
            self.counter.value = index + 1

        elif isinstance(index, torch.Tensor) and index.max().item() > self.counter.value:
            self.counter.value = index.max().item() + 1

        self.tstamp[index] = item[0]
        self.images[index] = item[1]

        if item[2] is not None:
            self.poses[index] = item[2]

        if item[3] is not None:
            if len(item[3].shape) > 2:
                depth = item[3][:, 3::8, 3::8]
            else:
                depth = item[3][3::8, 3::8]
            self.disps[index] = torch.where(depth > 0, 1.0 / depth, depth)

        if item[4] is not None:
            if len(item[4].shape) > 2:
                depth = item[4][:, 3::8, 3::8]
            else:
                depth = item[4][3::8, 3::8]
            self.disps_sens[index] = torch.where(depth > 0, 1.0 / depth, depth)

            self.depths_gt[index] = item[4]

        if item[5] is not None:
            self.intrinsics[index] = item[5]

        if len(item) > 6:
            self.fmaps[index] = item[6]

        if len(item) > 7:
            if item[7] is not None:
                self.nets[index] = item[7]

        if len(item) > 8:
            if item[8] is not None:
                self.inps[index] = item[8]

        if len(item) > 9:
            self.poses_gt[index] = item[9]

    def __setitem__(self, index, item):
        with self.get_lock():
            self.__item_setter(index, item)

    def __getitem__(self, index):
        """ index the depth video """

        with self.get_lock():
            # support negative indexing
            if isinstance(index, int) and index < 0:
                index = self.counter.value + index

            item = (
                self.poses[index],
                self.disps[index],
                self.intrinsics[index],
                self.fmaps[index],
                self.nets[index],
                self.inps[index])

        return item

    def append(self, *item):
        with self.get_lock():
            self.__item_setter(self.counter.value, item)

    @staticmethod
    def format_indicies(ii, jj):
        """ to device, long, {-1} """

        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii)

        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj)

        ii = ii.to(device="cuda", dtype=torch.long).reshape(-1)
        jj = jj.to(device="cuda", dtype=torch.long).reshape(-1)

        return ii, jj

    def upsample(self, ix, mask):
        """ upsample disparity """

        disps_up = cvx_upsample(self.disps[ix].unsqueeze(-1), mask)
        self.disps_up[ix] = disps_up.squeeze()

    def reproject(self, ii, jj):
        """ project points from ii -> jj """
        ii, jj = DepthVideo.format_indicies(ii, jj)
        Gs = lietorch.SE3(self.poses[None])

        coords, valid_mask = \
            pops.projective_transform(Gs, self.disps[None], self.intrinsics[None], ii, jj)

        return coords, valid_mask

    def distance(self, ii=None, jj=None, beta=0.3, bidirectional=True):
        """ frame distance metric """

        return_matrix = False
        if ii is None:
            return_matrix = True
            N = self.counter.value
            ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N))

        ii, jj = DepthVideo.format_indicies(ii, jj)

        if bidirectional:

            poses = self.poses[:self.counter.value].clone()

            d1 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[0], ii, jj, beta, 0)

            d2 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[0], jj, ii, beta, 0)

            d = .5 * (d1 + d2)

        else:
            d = droid_backends.frame_distance(
                self.poses, self.disps, self.intrinsics[0], ii, jj, beta, 0)

        if return_matrix:
            return d.reshape(N, N)

        return d

    def ba(self, target, weight, eta, ii, jj, t0=1, t1=None, itrs=2, lm=1e-4, ep=0.1, motion_only=False,
           use_mask=False):
        """ dense bundle adjustment (DBA) """

        with self.get_lock():
            # [t0, t1] window of bundle adjustment optimization
            if t1 is None:
                t1 = max(ii.max().item(), jj.max().item()) + 1

            droid_backends.ba(self.poses, self.disps, self.intrinsics[0], self.disps_sens,
                              target, weight, eta, ii, jj, t0, t1, itrs, 0, lm, ep, motion_only, self.config.get("adaptive_intri", False))
            self.intrinsics[:] = self.intrinsics[0].repeat(self.intrinsics.shape[0], 1)
            self.disps.clamp_(min=1e-5)

    def set_dirty(self, index_start, index_end):
        self.dirty[index_start:index_end] = True

    def get_pose(self, index, device):
        w2c = lietorch.SE3(self.poses[index].clone()).to(device)  # Tw(droid)_to_c
        c2w = w2c.inv().matrix()  # [4, 4]
        return c2w

    def get_depth_and_pose(self, index, device):
        with self.get_lock():
            c2w = self.get_pose(index, device)
        return c2w

    def get_pose_and_index(self):
        poses = []
        timestamps = []
        for i in range(self.counter.value):
            pose = self.get_pose(i, 'cpu')
            timestamp = self.tstamp[i].cpu()
            poses.append(pose)
            timestamps.append(timestamp)
        poses = torch.stack(poses, dim=0).numpy()

        timestamps = torch.stack(timestamps, dim=0).numpy()
        return timestamps, poses

    def save_video(self, path: str):
        poses = []
        timestamps = []

        for i in range(self.counter.value):
            pose = self.get_depth_and_pose(i, 'cpu')
            timestamp = self.tstamp[i].cpu()
            poses.append(pose)
            timestamps.append(timestamp)
        poses = torch.stack(poses, dim=0).numpy()

        timestamps = torch.stack(timestamps, dim=0).numpy()

        np.savez(path, poses=poses, timestamps=timestamps)
        print(f"Saved final depth video: {path}")

