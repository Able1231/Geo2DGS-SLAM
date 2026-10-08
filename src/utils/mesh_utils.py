#
# Copyright (C) 2024, ShanghaiTech
# SVIP research group, https://github.com/svip-lab
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  huangbb@shanghaitech.edu.cn
#

import torch
import numpy as np
import os
import math
from tqdm import tqdm
from scipy.ndimage import median_filter
from functools import partial
import open3d as o3d
import trimesh
from src.utils.utils import torch2np

def post_process_mesh(mesh, cluster_to_keep=1000):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy
    print("post processing the mesh to have {} clusterscluster_to_kep".format(cluster_to_keep))
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50)  # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0


def to_cam_open3d(width, height, opengl_proj, estimate_w2c, Fx, Fy, Cx, Cy):
    camera_traj = []
    for i in range(len(opengl_proj)):
        proj_matrix = opengl_proj[i].cpu().numpy()  # (4x4)
        extrinsic = estimate_w2c[i].cpu().numpy()

        fx = Fx
        fy = Fy
        cx = Cx
        cy = Cy

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy
        )
        camera = o3d.camera.PinholeCameraParameters()
        camera.intrinsic = intrinsic
        camera.extrinsic = extrinsic
        camera_traj.append(camera)
    return camera_traj

def filter_depth_outliers(depth_map, kernel_size=3, threshold=1.0):
    median_filtered = median_filter(depth_map, size=kernel_size)
    abs_diff = np.abs(depth_map - median_filtered)
    outlier_mask = abs_diff > threshold
    depth_map_filtered = np.where(outlier_mask, median_filtered, depth_map)
    return depth_map_filtered

class GaussianExtractor(object):
    def __init__(self, gaussians, render, render_setting, w, h, Fx, Fy, Cx, Cy):
        """
        a class that extracts attributes a scene presented by 2DGS

        Usage example:
        >>> gaussExtrator = GaussianExtractor(gaussians, render, pipe)
        >>> gaussExtrator.reconstruction(view_points)
        >>> mesh = gaussExtractor.export_mesh_bounded(...)
        """
        self.gaussians = gaussians
        self.render = render
        self.render_setting = render_setting
        self.width = w
        self.height = h
        self.Fx = Fx
        self.Fy = Fy
        self.Cx = Cx
        self.Cy = Cy
        self.clean()

    @torch.no_grad()
    def clean(self):
        self.depthmaps = []
        # self.alphamaps = []
        self.rgbmaps = []
        # self.normals = []
        # self.depth_normals = []
        self.estimate_w2c = []
        self.opengl_proj = []
        self.full_proj_matrix = []

    @torch.no_grad()
    def reconstruction(self, estimate_w2c, full_proj_matrix, opengl_proj):
        """
        reconstruct radiance field given cameras
        """
        self.clean()
        self.estimate_w2c = estimate_w2c
        self.full_proj_matrix = full_proj_matrix
        self.opengl_proj = opengl_proj
        estimate_c2w = [torch.linalg.inv(w2c) for w2c in self.estimate_w2c]
        for i in tqdm(range(len(self.estimate_w2c)), desc="reconstruct radiance fields"):
            intric = torch.tensor([
                [self.Fx, 0.0, self.Cx],
                [0.0, self.Fy, self.Cy],
                [0.0, 0.0, 1.0]
            ], dtype=torch.float32)
            #intric = intric.to("cuda")
            render_setting = self.render_setting(self.width, self.height, intric,
                                                 self.estimate_w2c[i])

            render_pkg = self.render(self.gaussians, render_setting)
            rgb = render_pkg['color']
            alpha = render_pkg['alpha']
            # normal = torch.nn.functional.normalize(render_pkg['normal'], dim=0)
            depth = render_pkg['depth']
            #depth = render_pkg['depth'][0].detach()
            #rendered_depth = torch2np(depth)
            #depth = filter_depth_outliers(
            #    rendered_depth, kernel_size=20, threshold=0.1)
            #depth = torch.from_numpy(depth).unsqueeze(0)

            # depth_normal = render_pkg['surf_normal']
            self.rgbmaps.append(rgb.cpu())
            self.depthmaps.append(depth.cpu())
            # self.alphamaps.append(alpha.cpu())
            # self.normals.append(normal.cpu())
            # self.depth_normals.append(depth_normal.cpu())

        # self.rgbmaps = torch.stack(self.rgbmaps, dim=0)
        # self.depthmaps = torch.stack(self.depthmaps, dim=0)
        # self.alphamaps = torch.stack(self.alphamaps, dim=0)
        # self.depth_normals = torch.stack(self.depth_normals, dim=0)
        self.estimate_bounding_sphere(estimate_c2w)

    def estimate_bounding_sphere(self, c2ws):
        """
        Estimate the bounding sphere given camera pose
        """
        from src.utils.utils import transform_poses_pca, focus_point_fn

        torch.cuda.empty_cache()
        c2ws = [tensor.cpu().numpy() for tensor in c2ws]

        c2ws = np.array(c2ws)
        poses = c2ws[:, :3, :] @ np.diag([1, -1, -1, 1])
        center = (focus_point_fn(poses))
        self.radius = np.linalg.norm(c2ws[:, :3, 3] - center, axis=-1).min()
        self.center = torch.from_numpy(center).float().cuda()
        print(f"The estimated bounding radius is {self.radius:.2f}")
        print(f"Use at least {2.0 * self.radius:.2f} for depth_trunc")

    @torch.no_grad()
    def extract_mesh_bounded(self, voxel_size=0.004, sdf_trunc=0.02, depth_trunc=3, mask_backgrond=True):
        """
        Perform TSDF fusion given a fixed depth range, used in the paper.

        voxel_size: the voxel size of the volume
        sdf_trunc: truncation value
        depth_trunc: maximum depth range, should depended on the scene's scales
        mask_backgrond: whether to mask backgroud, only works when the dataset have masks

        return o3d.mesh
        """
        print("Running tsdf volume integration ...")
        print(f'voxel_size: {voxel_size}')
        print(f'sdf_trunc: {sdf_trunc}')
        print(f'depth_truc: {depth_trunc}')

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        for i, cam_o3d in tqdm(enumerate(to_cam_open3d(self.width, self.height, self.opengl_proj, self.estimate_w2c,
                                                       self.Fx, self.Fy, self.Cx, self.Cy),),
                               desc="TSDF integration progress"):
            rgb = self.rgbmaps[i]
            depth = self.depthmaps[i]

            # make open3d rgbd
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(
                    np.asarray(np.clip(rgb.permute(1, 2, 0).cpu().numpy(), 0.0, 1.0) * 255, order="C", dtype=np.uint8)),
                o3d.geometry.Image(np.asarray(depth.permute(1, 2, 0).cpu().numpy(), order="C")),
                depth_trunc=depth_trunc, convert_rgb_to_intensity=False,
                depth_scale=1.0
            )

            volume.integrate(rgbd, intrinsic=cam_o3d.intrinsic, extrinsic=cam_o3d.extrinsic)

        mesh = volume.extract_triangle_mesh()
        return mesh

    @torch.no_grad()
    def extract_mesh_unbounded(self, resolution=1024):
        """
        Experimental features, extracting meshes from unbounded scenes, not fully test across datasets.
        return o3d.mesh
        """

        def contract(x):
            mag = torch.linalg.norm(x, ord=2, dim=-1)[..., None]
            return torch.where(mag < 1, x, (2 - (1 / mag)) * (x / mag))

        def uncontract(y):
            mag = torch.linalg.norm(y, ord=2, dim=-1)[..., None]
            return torch.where(mag < 1, y, (1 / (2 - mag) * (y / mag)))

        def compute_sdf_perframe(i, points, depthmap, rgbmap, full_proj_transform):
            """
                compute per frame sdf
            """
            new_points = torch.cat([points, torch.ones_like(points[..., :1])],
                                   dim=-1) @ full_proj_transform
            z = new_points[..., -1:]
            pix_coords = (new_points[..., :2] / new_points[..., -1:])
            mask_proj = ((pix_coords > -1.) & (pix_coords < 1.) & (z > 0)).all(dim=-1)
            sampled_depth = torch.nn.functional.grid_sample(depthmap.cuda()[None], pix_coords[None, None],
                                                            mode='bilinear', padding_mode='border',
                                                            align_corners=True).reshape(-1, 1)
            sampled_rgb = torch.nn.functional.grid_sample(rgbmap.cuda()[None], pix_coords[None, None], mode='bilinear',
                                                          padding_mode='border', align_corners=True).reshape(3, -1).T
            sdf = (sampled_depth - z)
            return sdf, sampled_rgb, mask_proj

        def compute_unbounded_tsdf(samples, inv_contraction, voxel_size, return_rgb=False):
            """
                Fusion all frames, perform adaptive sdf_funcation on the contract spaces.
            """
            if inv_contraction is not None:
                mask = torch.linalg.norm(samples, dim=-1) > 1
                # adaptive sdf_truncation
                sdf_trunc = 5 * voxel_size * torch.ones_like(samples[:, 0])
                sdf_trunc[mask] *= 1 / (2 - torch.linalg.norm(samples, dim=-1)[mask].clamp(max=1.9))
                samples = inv_contraction(samples)
            else:
                sdf_trunc = 5 * voxel_size

            tsdfs = torch.ones_like(samples[:, 0]) * 1
            rgbs = torch.zeros((samples.shape[0], 3)).cuda()

            weights = torch.ones_like(samples[:, 0])
            for i in tqdm(range(len(self.estimate_w2c)), desc="TSDF integration progress"):
                sdf, rgb, mask_proj = compute_sdf_perframe(i, samples,
                                                           depthmap=self.depthmaps[i],
                                                           rgbmap=self.rgbmaps[i],
                                                           full_proj_transform=self.full_proj_matrix[i],
                                                           )

                # volume integration
                sdf = sdf.flatten()
                mask_proj = mask_proj & (sdf > -sdf_trunc)
                sdf = torch.clamp(sdf / sdf_trunc, min=-1.0, max=1.0)[mask_proj]
                w = weights[mask_proj]
                wp = w + 1
                tsdfs[mask_proj] = (tsdfs[mask_proj] * w + sdf) / wp
                rgbs[mask_proj] = (rgbs[mask_proj] * w[:, None] + rgb[mask_proj]) / wp[:, None]
                # update weight
                weights[mask_proj] = wp

            if return_rgb:
                return tsdfs, rgbs

            return tsdfs

        normalize = lambda x: (x - self.center) / self.radius
        unnormalize = lambda x: (x * self.radius) + self.center
        inv_contraction = lambda x: unnormalize(uncontract(x))

        N = resolution
        voxel_size = (self.radius * 2 / N)
        print(f"Computing sdf gird resolution {N} x {N} x {N}")
        print(f"Define the voxel_size as {voxel_size}")
        sdf_function = lambda x: compute_unbounded_tsdf(x, inv_contraction, voxel_size)
        from src.utils.mcube_utils import marching_cubes_with_contraction
        print(f"self.xyz: {self.gaussians.get_xyz}")

        R = contract(normalize(self.gaussians.get_xyz())).norm(dim=-1).cpu().numpy()
        R = np.quantile(R, q=0.95)
        R = min(R + 0.01, 1.9)

        mesh = marching_cubes_with_contraction(
            sdf=sdf_function,
            bounding_box_min=(-R, -R, -R),
            bounding_box_max=(R, R, R),
            level=0,
            resolution=N,
            inv_contraction=inv_contraction,
        )

        # coloring the mesh
        torch.cuda.empty_cache()
        mesh = mesh.as_open3d
        print("texturing mesh ... ")
        _, rgbs = compute_unbounded_tsdf(torch.tensor(np.asarray(mesh.vertices)).float().cuda(), inv_contraction=None,
                                         voxel_size=voxel_size, return_rgb=True)
        mesh.vertex_colors = o3d.utility.Vector3dVector(rgbs.cpu().numpy())
        return mesh

