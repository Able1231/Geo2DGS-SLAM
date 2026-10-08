import cv2
import numpy as np
import torch

import contextlib
import io
import sys
import math
from einops import *

from lightglue import LightGlue, SuperPoint, DISK, SIFT
from lightglue.utils import load_image, rbd
from sympy.printing.precedence import precedence_Integer


class Light:
    def __init__(self):
        self.extractor = self.initialize_extractor()
        self.matcher = self.initialize_matcher()

    def initialize_extractor(self):
        """Initialize the feature extractor based on the given type."""
        extractor = SuperPoint(max_num_keypoints=600).eval().cuda()
        return extractor

    def initialize_matcher(self, matcher="LightGlue"):
        """Initialize the matcher based on the given type."""
        matcher = LightGlue(features='superpoint').eval().cuda()
        return matcher

    def extract_features(self, image, extractor):
        """Extract features from the image using a specified extractor."""
        return extractor.extract(image)

    def get_matched_points(self, current_image, referred_image, cv2_macher):
        """Match points between two images using specified feature extractor and matcher."""
        extractor = self.extractor
        matcher = self.matcher
        feats0 = self.extract_features(current_image, extractor)
        feats1 = self.extract_features(referred_image, extractor)

        matches01 = matcher({'image0': feats0, 'image1': feats1})

        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # Remove batch dimension
        matches = matches01['matches']
        points0 = feats0['keypoints'][matches[..., 0]]
        points1 = feats1['keypoints'][matches[..., 1]]

        return points0.cpu(), points1.cpu()


def depths_to_points(depthmap, c2w, w, h, intrins):
    c2w = c2w
    W, H = w, h

    intrins = intrins.to(dtype=torch.float32)
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float(), torch.arange(H, device='cuda').float(),
                                    indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = points @ intrins.inverse().T @ c2w[:3, :3].T
    rays_o = c2w[:3, 3]
    # rays_d = points @ intrins.inverse().T
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points


def compute_3d_points(W, H, depth, intrins, c2w, pts):
    device = depth.device

    # pts -> tensor
    if not torch.is_tensor(pts):
        pts_tensor = torch.tensor(pts, dtype=torch.float32, device=device)
    else:
        pts_tensor = pts.to(device).float()

    ones = torch.ones((pts_tensor.shape[0], 1), device=device)
    points_homo = torch.cat([pts_tensor, ones], dim=-1)  # N x 3

    # intrinsics and c2w to same device
    intrins = intrins.to(device=device, dtype=torch.float32)
    c2w = c2w.to(device=device, dtype=torch.float32)

    intrins_inv = torch.inverse(intrins)
    rays_d = points_homo @ intrins_inv.T @ c2w[:3, :3].T

    x_coords = pts_tensor[:, 0].long().clamp(0, W - 1)
    y_coords = pts_tensor[:, 1].long().clamp(0, H - 1)
    depths = depth[y_coords, x_coords].reshape(-1, 1)

    rays_o = c2w[:3, 3]
    points_3d = depths * rays_d + rays_o

    return points_3d


def GetBA_loss_3D(pts1, pts2, depthmap1, depthmap2, c2w_sign, c2w_cur, w, h, intrins):
   
    points1 = compute_3d_points(w, h, depthmap1, intrins, c2w_sign, pts1)  # (N, 3)
    points2 = compute_3d_points(w, h, depthmap2, intrins, c2w_cur, pts2)   # (N, 3)

    distances = torch.norm(points1 - points2, dim=1)

    mask = distances < 10.0
    total_distance = distances[mask].sum()

    return total_distance


def getProjectionMatrix2(znear, zfar, cx, cy, fx, fy, W, H):
    left = ((2 * cx - W) / W - 1.0) * W / 2.0
    right = ((2 * cx - W) / W + 1.0) * W / 2.0
    top = ((2 * cy - H) / H + 1.0) * H / 2.0
    bottom = ((2 * cy - H) / H - 1.0) * H / 2.0
    left = znear / fx * left
    right = znear / fx * right
    top = znear / fy * top
    bottom = znear / fy * bottom
    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)

    return P


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def getWorld2View2(R, t, translate=torch.tensor([0.0, 0.0, 0.0]), scale=1.0):
    translate = translate.to(R.device)
    Rt = torch.zeros((4, 4), device=R.device)
    # Rt[:3, :3] = R.transpose()
    Rt[:3, :3] = R
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = torch.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = torch.linalg.inv(C2W)
    return Rt