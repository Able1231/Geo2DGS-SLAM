""" This module is responsible for merging submaps. """
from argparse import ArgumentParser

import faiss
import numpy as np
import open3d as o3d
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from torchvision.utils import save_image
from src.entities.arguments import OptimizationParams
from src.entities.gaussian_model import GaussianModel,BasicPointCloud
from src.entities.losses import isotropic_loss, l1_loss, ssim, ScaleRegularizationMetricsModuleMixin, TVLoss, get_depth_ranking_loss, get_loss_mapping_rgbd
from src.utils.utils import (batch_search_faiss, get_render_settings, render_gaussian_model,
                             np2ptcloud,  torch2np, depth_to_normal)
#from src.utils.refine_utils import render
import matplotlib.pyplot as plt
import cv2
import os
from pathlib import Path
import imageio
from src.entities.gaussian_model import build_scaling_rotation
import random

class RenderFrames(Dataset):
    """A dataset class for loading keyframes along with their estimated camera poses and render settings."""
    def __init__(self, video_traj, video_timestamps, dataset, render_poses: np.ndarray, height: int, width: int, fx: float, fy: float, intrinsics=None, crop: int = 0):
        self.dataset = dataset
        print(len(self.dataset))
        self.render_poses = render_poses
        self.height = height
        self.width = width
        self.fx = fx
        self.fy = fy
        self.crop = crop
        self.intrinsics = intrinsics if intrinsics is not None else dataset.intrinsics
        self.device = "cuda"
        self.stride = 1

        self.video_timestamps = video_timestamps
        print(self.video_timestamps)
        self.video_traj = video_traj
        self.valid_indices = list(range(len(video_timestamps)))
        
    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx):
        traj_idx = random.choice(self.valid_indices)
        idx = int(self.video_timestamps[traj_idx])
  
        color_np = self.dataset[idx][1]
        depth_np = self.dataset[idx][2]
        if self.crop > 0:
            color_np = color_np[self.crop:-self.crop, self.crop:-self.crop]
            depth_np = depth_np[self.crop:-self.crop, self.crop:-self.crop]

        color = (torch.from_numpy(color_np) / 255.0).float().to(self.device)
        depth = torch.from_numpy(depth_np).float().to(self.device)
        estimate_c2w = self.video_traj[traj_idx]
        estimate_w2c = np.linalg.inv(estimate_c2w)

        frame = {
            "frame_id": idx,
            "color": color,
            "depth": depth,
            "render_settings": get_render_settings(
                self.width, self.height, self.intrinsics, estimate_w2c),
            "w2c": torch.from_numpy(estimate_w2c).to(self.device),
            "w": self.width,
            "h": self.height,
            "K": self.intrinsics
        }
        color = color.cpu().numpy().transpose(1, 2, 0)

        return frame


def merge_submaps(submaps_paths: list, radius: float = 0.0001, device: str = "cuda") -> o3d.geometry.PointCloud:
    """ Merge submaps into a single point cloud, which is then used for global map refinement.
    Args:
        segments_paths (list): Folder path of the submaps.
        radius (float, optional): Nearest neighbor distance threshold for adding a point. Defaults to 0.0001.
        device (str, optional): Defaults to "cuda".

    Returns:
        o3d.geometry.PointCloud: merged point cloud
    """
    pts_index = faiss.IndexFlatL2(3)
    if device == "cuda":
        pts_index = faiss.index_cpu_to_gpu(
            faiss.StandardGpuResources(),
            0,
            faiss.IndexIVFFlat(faiss.IndexFlatL2(3), 3, 500, faiss.METRIC_L2))
        pts_index.nprobe = 5
    merged_pts = []
    print("Merging segments")
    for submap_path in tqdm(submaps_paths):
        gaussian_params = torch.load(submap_path)["gaussian_params"]
        current_pts = gaussian_params["xyz"].to(device).float()
        pts_index.train(current_pts)
        distances, _ = batch_search_faiss(pts_index, current_pts, 8)
        neighbor_num = (distances < radius).sum(axis=1).int()
        ids_to_include = torch.where(neighbor_num == 0)[0]
        pts_index.add(current_pts[ids_to_include])
        merged_pts.append(current_pts[ids_to_include])
    pts = torch2np(torch.vstack(merged_pts))
    pt_cloud = np2ptcloud(pts, np.zeros_like(pts))

    # Downsampling if the total number of points is too large
    if len(pt_cloud.points) > 1_000_000:
        voxel_size = 0.02
        pt_cloud = pt_cloud.voxel_down_sample(voxel_size)
        print(f"Downsampled point cloud to {len(pt_cloud.points)} points")
    filtered_pt_cloud, _ = pt_cloud.remove_statistical_outlier(nb_neighbors=40, std_ratio=3.0)
    del pts_index
    return filtered_pt_cloud

def refined_merge_submaps(submaps_paths: list, opt_args):
    """ Merge submaps from agents in a coarse manner: Section 3.4
    Args:
        agents_submaps: A dictionary of agent submaps.
        agents_kf_ids: A dictionary of agent keyframe IDs.
        agents_opt_kf_c2ws: A dictionary of agent optimized camera-to-world matrices.
        opt_args: The optimization arguments.
    Returns:
        merged_map: The merged Gaussian model.
    """
    merged_map = GaussianModel(0)
    merged_map.training_setup(opt_args)
    device = "cuda"

    print("Merging submaps")

    for submap_path in tqdm(submaps_paths):
        submap = torch.load(submap_path)

        xyz = submap["gaussian_params"]["xyz"].to(device)
        rotations = submap["gaussian_params"]["rotation"].to(device)
        features_dc = submap["gaussian_params"]["features_dc"].to(device)
        features_rest = submap["gaussian_params"]["features_rest"].to(device)
        opacity = submap["gaussian_params"]["opacity"].to(device)
        scaling = submap["gaussian_params"]["scaling"].to(device)

        merged_map.densification_postfix(
            xyz,
            features_dc,
            features_rest,
            opacity,
            scaling,
            rotations)

    return merged_map
    
def refine_global_map(pt_cloud, submaps_paths, training_frames: list, max_iterations: int, output_dir, len_frames=None, o3d_intrinsic=None) -> GaussianModel:
    """Refines a global map based on the merged point cloud and training keyframes frames.
    Args:
        pt_cloud (o3d.geometry.PointCloud): The merged point cloud used for refinement.
        training_frames (list): A list of training frames for map refinement.
        max_iterations (int): The maximum number of iterations to perform for refinement.
    Returns:
        GaussianModel: The refined global map as a Gaussian model.
    """
    opt_params = OptimizationParams(ArgumentParser(description="Training script parameters"))

    #gaussian_model = refined_merge_submaps(submaps_paths, opt_params)
    gaussian_model = GaussianModel(0)
    gaussian_model.active_sh_degree = 0
    if pt_cloud is None:
         output_mesh = output_dir / "mesh" / "test_mesh.ply"
         output_mesh = o3d.io.read_triangle_mesh(str(output_mesh))
         pcd = o3d.geometry.PointCloud()
         pcd.points = output_mesh.vertices
         pcd.colors = output_mesh.vertex_colors
         pcd = pcd.voxel_down_sample(voxel_size=0.02)
         pcd = BasicPointCloud(points=np.asarray(pcd.points),
                             colors=np.asarray(pcd.colors))
         gaussian_model.create_from_pcd(pcd, 1.0)
         gaussian_model.training_setup(opt_params)
         
    else:
         gaussian_model.training_setup(opt_params)
         gaussian_model.add_points(pt_cloud)

    # Scene extent used by the densification clone/split thresholds (percent_dense * extent).
    with torch.no_grad():
        xyz = gaussian_model.get_xyz().detach()
        if xyz.shape[0] > 0:
            centroid = xyz.mean(dim=0, keepdim=True)
            scene_extent = (xyz - centroid).norm(dim=-1).max().item()
        else:
            scene_extent = 1.0
        scene_extent = max(scene_extent, 1.0)

    iteration = 0
    Metrics = ScaleRegularizationMetricsModuleMixin()
    for iteration in tqdm(range(max_iterations), desc="Refinement"):
        training_frame = next(training_frames)
        if training_frame is None:
            continue   
        #xyz_lr = gaussian_model.update_learning_rate(iteration)

        #if iteration > 0 and iteration % 1000 == 0:
        #     gaussian_model.oneupSHdegree()
        idx, gt_color, gt_depth, render_settings, w2c= (
            training_frame["frame_id"],
            training_frame["color"].squeeze(0),
            training_frame["depth"].squeeze(0),
            training_frame["render_settings"],
            training_frame["w2c"].squeeze(0))

            
        render_dict = render_gaussian_model(gaussian_model, render_settings)

        # Densification inputs: projected 2D means (with retained grad) and visible gaussians.
        viewspace_point_tensor = render_dict["means2D"]
        radii = render_dict["radii"]
        visibility_filter = radii > 0

        # renders, render_alphas, info = render(gaussian_model, training_frame["w"], training_frame["h"],
        #                                       training_frame["K"], training_frame['w2c'])
        #
        # rendered_color = torch.clamp(renders[..., 0:3], 0.0, 1.0).squeeze(0)  # [1, H, W, 3]
        # rendered_depth = renders[..., 3:4].squeeze(0).permute(2, 0, 1)  # [1, H, W, 1]

        metrics, pbar = Metrics.get_train_metrics(gaussian_model, iteration, batch=None)

        if metrics:
            scale_loss = metrics["loss"]

        else:
            scale_loss = 0
            max_scale = mean_scale = max_ratio = mean_ratio = 0

        rendered_color, rendered_depth = (render_dict["color"].permute(1, 2, 0), render_dict["depth"])
     
        render_alpha = render_dict["alpha"]
        render_normal1 = render_dict['normal']

        #render_normal1 = (render_normal1.permute(1, 2, 0) @ (w2c[:3, :3].T)).permute(2, 0, 1)
        render_normal = depth_to_normal(rendered_depth, torch.linalg.inv(w2c), render_settings.image_width.item(), render_settings.image_height.item(), render_settings.tanfovx.item(),          render_settings.tanfovy.item())
        render_normal = render_normal.permute(2,0,1)
        # remember to multiply with accum_alpha since render_normal is unnormalized.
        render_normal = render_normal * (render_alpha).detach()

        #surf_normal = depth_to_normal(gt_depth.unsqueeze(0), torch.linalg.inv(w2c), render_settings.image_width.item(), render_settings.image_height.item(), render_settings.tanfovx.item(),          render_settings.tanfovy.item())
        #surf_normal = surf_normal.permute(2,0,1)
        # remember to multiply with accum_alpha since render_normal is unnormalized.
        #surf_normal = surf_normal * (render_alpha).detach()

        lambda_normal = 0.005 

        normal_error = (1 - (render_normal * render_normal1).sum(dim=0))[None]
      
        normal_mask = render_normal.norm(dim=0) < 0.2  # 计算范数，使其形状为 [H, W]
        normal_error[:, normal_mask] = 0  # 这里 normal_error 是 [1, H, W]，匹配成功

        normal_loss = lambda_normal * (normal_error).mean()

        reg_loss = isotropic_loss(gaussian_model.get_scaling())
        depth_mask = (gt_depth > 0)
        color_loss = (1.0 - opt_params.lambda_dssim) * l1_loss(
            rendered_color[depth_mask, :], gt_color[depth_mask, :]
        ) + opt_params.lambda_dssim * (1.0 - ssim(rendered_color, gt_color))
        depth_loss = l1_loss(
            rendered_depth[:, depth_mask], gt_depth[depth_mask])

        #dsmooth_loss = TVLoss(rendered_depth, gt_depth.unsqueeze(0))
        #rank_loss = get_depth_ranking_loss(rendered_depth, gt_depth)
        total_loss = get_loss_mapping_rgbd(rendered_color.permute(2, 0, 1), rendered_depth, gt_color.permute(2, 0, 1), gt_depth) + color_loss + depth_loss + normal_loss

        total_loss.backward()

        with torch.no_grad():
        #    if iteration % 500 == 0:
        #        prune_mask = (gaussian_model.get_opacity() < 0.005).squeeze()
       #         gaussian_model.prune_points(prune_mask)

            # Densification: track visible gaussians and accumulate the gradient norm of
            # their projected 2D positions, then periodically clone / split / prune.
            gaussian_model.max_radii2D[visibility_filter] = torch.max(
                gaussian_model.max_radii2D[visibility_filter], radii[visibility_filter])
            if viewspace_point_tensor.grad is not None:
                gaussian_model.add_densification_stats(
                    viewspace_point_tensor, visibility_filter)

            if (iteration < opt_params.densify_until_iter
                    and iteration > opt_params.densify_from_iter
                    and iteration % 500 == 0):
                gaussian_model.densify_and_prune(
                    opt_params.densify_grad_threshold,
                    min_opacity=0.1,
                    extent=scene_extent,
                    max_screen_size=20)

            # Optimizer step
            gaussian_model.optimizer.step()
            gaussian_model.optimizer.zero_grad(set_to_none=True)

        iteration += 1
    export_refine_mesh = True
    try:
        if export_refine_mesh:
            output_dir = output_dir / "mesh" / "refined_mesh.ply"
            scale = 1.0
            volume = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=5.0 * scale / 512.0,
                sdf_trunc=0.04 * scale,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
            for i in tqdm(range(len_frames), desc="Integrating mesh"):  # one cycle
                training_frame = next(training_frames)
                gt_color, gt_depth, render_settings, estimate_w2c = (
                    training_frame["color"].squeeze(0),
                    training_frame["depth"].squeeze(0),
                    training_frame["render_settings"],
                    training_frame["w2c"])

                render_dict = render_gaussian_model(gaussian_model, render_settings)
                rendered_color, rendered_depth = (
                    render_dict["color"].permute(1, 2, 0), render_dict["depth"])
                rendered_color = torch.clamp(rendered_color, min=0.0, max=1.0)

                rendered_color = (
                    torch2np(rendered_color) * 255).astype(np.uint8)
                rendered_depth = torch2np(rendered_depth.squeeze())
      
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(np.ascontiguousarray(rendered_color)),
                    o3d.geometry.Image(rendered_depth),
                    depth_scale=scale,
                    depth_trunc=30,
                    convert_rgb_to_intensity=False)
                volume.integrate(
                    rgbd, o3d_intrinsic, estimate_w2c.squeeze().cpu().numpy().astype(np.float64))

            o3d_mesh = volume.extract_triangle_mesh()
            o3d.io.write_triangle_mesh(str(output_dir), o3d_mesh)
            print(f"Refined mesh saved to {output_dir}")

    except Exception as e:
        print(f"Error export_refine_mesh in refine_global_map:\n {e}")
    return gaussian_model

