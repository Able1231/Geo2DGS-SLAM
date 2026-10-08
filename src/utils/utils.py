import os
import random

import numpy as np
import open3d as o3d
import torch
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from typing import List, Mapping, Optional, Text, Tuple, Union
import copy
from PIL import Image
import mediapy as media
from matplotlib import cm
from tqdm import tqdm
import math


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def torch2np(tensor: torch.Tensor) -> np.ndarray:
    """ Converts a PyTorch tensor to a NumPy ndarray.
    Args:
        tensor: The PyTorch tensor to convert.
    Returns:
        A NumPy ndarray with the same data and dtype as the input tensor.
    """
    return tensor.detach().cpu().numpy()


def np2torch(array: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """Converts a NumPy ndarray to a PyTorch tensor.
    Args:
        array: The NumPy ndarray to convert.
        device: The device to which the tensor is sent. Defaults to 'cpu'.

    Returns:
        A PyTorch tensor with the same data as the input array.
    """
    return torch.from_numpy(array).float().to(device)


def np2ptcloud(pts: np.ndarray, rgb=None) -> o3d.geometry.PointCloud:
    """converts numpy array to point cloud
    Args:
        pts (ndarray): point cloud
    Returns:
        (PointCloud): resulting point cloud
    """
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts)
    if rgb is not None:
        cloud.colors = o3d.utility.Vector3dVector(rgb)
    return cloud


def dict2device(dict: dict, device: str = "cpu") -> dict:
    """Sends all tensors in a dictionary to a specified device.
    Args:
        dict: The dictionary containing tensors.
        device: The device to send the tensors to. Defaults to 'cpu'.
    Returns:
        The dictionary with all tensors sent to the specified device.
    """
    for k, v in dict.items():
        if isinstance(v, torch.Tensor):
            dict[k] = v.to(device)
    return dict


def get_render_settings(w, h, intrinsics, w2c, near=0.01, far=100, sh_degree=0):
    """
    Constructs and returns a GaussianRasterizationSettings object for rendering,
    configured with given camera parameters.

    Args:
        width (int): The width of the image.
        height (int): The height of the image.
        intrinsic (array): 3*3, Intrinsic camera matrix.
        w2c (array): World to camera transformation matrix.
        near (float, optional): The near plane for the camera. Defaults to 0.01.
        far (float, optional): The far plane for the camera. Defaults to 100.

    Returns:
        GaussianRasterizationSettings: Configured settings for Gaussian rasterization.
    """
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1,
    1], intrinsics[0, 2], intrinsics[1, 2]
    w2c = torch.tensor(w2c).cuda().float()
    cam_center = torch.inverse(w2c)[:3, 3]
    viewmatrix = w2c.transpose(0, 1)
    opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                                [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                [0.0, 0.0, far /
                                 (far - near), -(far * near) / (far - near)],
                                [0.0, 0.0, 1.0, 0.0]], device='cuda').float().transpose(0, 1)
    full_proj_matrix = viewmatrix.unsqueeze(
        0).bmm(opengl_proj.unsqueeze(0)).squeeze(0)
    return GaussianRasterizationSettings(
        image_height=h,
        image_width=w,
        tanfovx=w / (2 * fx),
        tanfovy=h / (2 * fy),
        bg=torch.tensor([0, 0, 0], device='cuda').float(),
        scale_modifier=1.0,
        viewmatrix=viewmatrix,
        projmatrix=full_proj_matrix,
        sh_degree=sh_degree,
        campos=cam_center,
        prefiltered=False,
        debug=False)


def render_gaussian_model(gaussian_model, render_settings,
                          override_means_3d=None, override_means_2d=None,
                          override_scales=None, override_rotations=None,
                          override_opacities=None, override_colors=None):
    """
    Renders a Gaussian model with specified rendering settings, allowing for
    optional overrides of various model parameters.

    Args:
        gaussian_model: A Gaussian model object that provides methods to get
            various properties like xyz coordinates, opacity, features, etc.
        render_settings: Configuration settings for the GaussianRasterizer.
        override_means_3d (Optional): If provided, these values will override
            the 3D mean values from the Gaussian model.
        override_means_2d (Optional): If provided, these values will override
            the 2D mean values. Defaults to zeros if not provided.
        override_scales (Optional): If provided, these values will override the
            scale values from the Gaussian model.
        override_rotations (Optional): If provided, these values will override
            the rotation values from the Gaussian model.
        override_opacities (Optional): If provided, these values will override
            the opacity values from the Gaussian model.
        override_colors (Optional): If provided, these values will override the
            color values from the Gaussian model.
    Returns:
        A dictionary containing the rendered color, depth, radii, and 2D means
        of the Gaussian model. The keys of this dictionary are 'color', 'depth',
        'radii', and 'means2D', each mapping to their respective rendered values.
    """
    renderer = GaussianRasterizer(raster_settings=render_settings)

    if override_means_3d is None:
        means3D = gaussian_model.get_xyz()
    else:
        means3D = override_means_3d

    if override_means_2d is None:
        means2D = torch.zeros_like(
            means3D, dtype=means3D.dtype, requires_grad=True, device="cuda")
        means2D.retain_grad()
    else:
        means2D = override_means_2d

    if override_opacities is None:
        opacities = gaussian_model.get_opacity()
    else:
        opacities = override_opacities

    shs, colors_precomp = None, None
    if override_colors is not None:
        colors_precomp = override_colors
    else:
        shs = gaussian_model.get_features()

    render_args = {
        "means3D": means3D,
        "means2D": means2D,
        "opacities": opacities,
        "colors_precomp": colors_precomp,
        "shs": shs,
        "scales": gaussian_model.get_scaling() if override_scales is None else override_scales,
        "rotations": gaussian_model.get_rotation() if override_rotations is None else override_rotations,
        "cov3D_precomp": None
    }
    rendered_image, radii, allmap = renderer(**render_args)

    render_alpha = allmap[1:2]
    render_normal = allmap[2:5]
    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
    return {"color": rendered_image, "depth": render_depth_expected, "radii": radii, "means2D": means2D,
            "alpha": render_alpha, "normal": render_normal}


def batch_search_faiss(indexer, query_points, k):
    """
    Perform a batch search on a IndexIVFFlat indexer to circumvent the search size limit of 65535.

    Args:
        indexer: The FAISS indexer object.
        query_points: A tensor of query points.
        k (int): The number of nearest neighbors to find.

    Returns:
        distances (torch.Tensor): The distances of the nearest neighbors.
        ids (torch.Tensor): The indices of the nearest neighbors.
    """
    split_pos = torch.split(query_points, 65535, dim=0)
    distances_list, ids_list = [], []

    for split_p in split_pos:
        distance, id = indexer.search(split_p.float(), k)
        distances_list.append(distance.clone())
        ids_list.append(id.clone())
    distances = torch.cat(distances_list, dim=0)
    ids = torch.cat(ids_list, dim=0)

    return distances, ids


def get_world_points(depth, intrinsics, extrinsics):
    '''
    Args:
        depthmap: H*W
        intrinsics: 3*3 or 4*4
        extrinsics: 4*4, world to camera
    Return:
        points: N*3, in world space
    '''
    if intrinsics.shape[0] == 4:
        intrinsics = intrinsics[:3, :3]

    height, width = depth.shape

    x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))

    # valid_points = np.ma.masked_greater(depth, 0.0).mask
    # x, y, depth = x[valid_points], y[valid_points], depth[valid_points]

    x = x.reshape((1, height * width))
    y = y.reshape((1, height * width))
    depth = depth.reshape((1, height * width))

    xyz_ref = np.matmul(np.linalg.inv(intrinsics),
                        np.vstack((x, y, np.ones_like(x))) * depth)
    xyz_world = np.matmul(np.linalg.inv(extrinsics),
                          np.vstack((xyz_ref, np.ones_like(x))))[:3]
    xyz_world = xyz_world.transpose((1, 0))

    return xyz_world


def get_camera_origins(poses_homo):
    '''
    Args:
        poses_homo: world to camera poses
    '''
    if not isinstance(poses_homo, np.ndarray):
        poses_homo = np.array(poses_homo)
    cam_centers = []
    poses_homo = np.array(poses_homo)
    num_cams = poses_homo.shape[0]
    for i in range(num_cams):
        rot = poses_homo[i, :3, :3]
        trans = poses_homo[i, :3, 3]
        trans = trans.reshape(3, 1)
        cam_center = - np.linalg.inv(rot) @ trans
        cam_centers.append(cam_center)
    cam_centers = np.array(cam_centers).squeeze(axis=-1)
    return cam_centers


def calculate_normalmap_from_depthmap(depthmap, intrin, extrin, num_nearest_neighbors=100):
    '''
    Args:
        depthmap: H*W. depth in image plane
        extrin: word to cam
    Return:
        normalmap: H*W*3
    '''
    pts = get_world_points(depthmap, intrin, extrin)
    cam_center = get_camera_origins([extrin])[0]
    H, W = depthmap.shape

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=num_nearest_neighbors))
    normals = np.array(pcd.normals)

    # check normal direction: if ray dir and normal angle is smaller than 90, reverse normal
    ray_dir = pts - cam_center.reshape(1, 3)
    normal_dir_not_correct = (ray_dir * normals).sum(axis=-1) > 0
    normals[normal_dir_not_correct] = -normals[normal_dir_not_correct]

    return pts, normals.reshape(H, W, 3)


def normalize(x: np.ndarray) -> np.ndarray:
    """Normalization helper function."""
    return x / np.linalg.norm(x)


def pad_poses(p: np.ndarray) -> np.ndarray:
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def unpad_poses(p: np.ndarray) -> np.ndarray:
    """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
    return p[..., :3, :4]


def recenter_poses(poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Recenter poses around the origin."""
    cam2world = average_pose(poses)
    transform = np.linalg.inv(pad_poses(cam2world))
    poses = transform @ pad_poses(poses)
    return unpad_poses(poses), transform


def average_pose(poses: np.ndarray) -> np.ndarray:
    """New pose using average position, z-axis, and up vector of input poses."""
    position = poses[:, :3, 3].mean(0)
    z_axis = poses[:, :3, 2].mean(0)
    up = poses[:, :3, 1].mean(0)
    cam2world = viewmatrix(z_axis, up, position)
    return cam2world


def viewmatrix(lookdir: np.ndarray, up: np.ndarray,
               position: np.ndarray) -> np.ndarray:
    """Construct lookat view matrix."""
    vec2 = normalize(lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m


def focus_point_fn(poses: np.ndarray) -> np.ndarray:
    """Calculate nearest point to all focal axes in poses."""
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt


def transform_poses_pca(poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Transforms poses so principal components lie on XYZ axes.

    Args:
      poses: a (N, 3, 4) array containing the cameras' camera to world transforms.

    Returns:
      A tuple (poses, transform), with the transformed poses and the applied
      camera_to_world transforms.
    """
    t = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t = t - t_mean

    eigval, eigvec = np.linalg.eig(t.T @ t)
    # Sort eigenvectors in order of largest to smallest eigenvalue.
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot

    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    poses_recentered = unpad_poses(transform @ pad_poses(poses))
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

    # Flip coordinate system if z component of y-axis is negative
    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
        transform = np.diag(np.array([1, -1, -1, 1])) @ transform

    return poses_recentered, transform
    # points = np.random.rand(3,100)
    # points_h = np.concatenate((points,np.ones_like(points[:1])), axis=0)
    # (poses_recentered @ points_h)[0]
    # (transform @ pad_poses(poses) @ points_h)[0,:3]
    # import pdb; pdb.set_trace()

    # # Just make sure it's it in the [-1, 1]^3 cube
    # scale_factor = 1. / np.max(np.abs(poses_recentered[:, :3, 3]))
    # poses_recentered[:, :3, 3] *= scale_factor
    # transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform

    # return poses_recentered, transform


def generate_ellipse_path(poses: np.ndarray,
                          n_frames: int = 120,
                          const_speed: bool = True,
                          z_variation: float = 0.,
                          z_phase: float = 0.) -> np.ndarray:
    """Generate an elliptical render path based on the given poses."""
    # Calculate the focal point for the path (cameras point toward this).
    center = focus_point_fn(poses)
    # Path height sits at z=0 (in middle of zero-mean capture pattern).
    offset = np.array([center[0], center[1], 0])

    # Calculate scaling for ellipse axes based on input camera positions.
    sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0)
    # Use ellipse that is symmetric about the focal point in xy.
    low = -sc + offset
    high = sc + offset
    # Optional height variation need not be symmetric
    z_low = np.percentile((poses[:, :3, 3]), 10, axis=0)
    z_high = np.percentile((poses[:, :3, 3]), 90, axis=0)

    def get_positions(theta):
        # Interpolate between bounds with trig functions to get ellipse in x-y.
        # Optionally also interpolate in z to change camera height along path.
        return np.stack([
            low[0] + (high - low)[0] * (np.cos(theta) * .5 + .5),
            low[1] + (high - low)[1] * (np.sin(theta) * .5 + .5),
            z_variation * (z_low[2] + (z_high - z_low)[2] *
                           (np.cos(theta + 2 * np.pi * z_phase) * .5 + .5)),
        ], -1)

    theta = np.linspace(0, 2. * np.pi, n_frames + 1, endpoint=True)
    positions = get_positions(theta)

    # if const_speed:

    # # Resample theta angles so that the velocity is closer to constant.
    # lengths = np.linalg.norm(positions[1:] - positions[:-1], axis=-1)
    # theta = stepfun.sample(None, theta, np.log(lengths), n_frames + 1)
    # positions = get_positions(theta)

    # Throw away duplicated last position.
    positions = positions[:-1]

    # Set path's up vector to axis closest to average of input pose up vectors.
    avg_up = poses[:, :3, 1].mean(0)
    avg_up = avg_up / np.linalg.norm(avg_up)
    ind_up = np.argmax(np.abs(avg_up))
    up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])

    return np.stack([viewmatrix(p - center, up, p) for p in positions])


def generate_path(viewpoint_cameras, n_frames=480):
    c2ws = np.array(
        [np.linalg.inv(np.asarray((cam.world_view_transform.T).cpu().numpy())) for cam in viewpoint_cameras])
    pose = c2ws[:, :3, :] @ np.diag([1, -1, -1, 1])
    pose_recenter, colmap_to_world_transform = transform_poses_pca(pose)

    # generate new poses
    new_poses = generate_ellipse_path(poses=pose_recenter, n_frames=n_frames)
    # warp back to orignal scale
    new_poses = np.linalg.inv(colmap_to_world_transform) @ pad_poses(new_poses)

    traj = []
    for c2w in new_poses:
        c2w = c2w @ np.diag([1, -1, -1, 1])
        cam = copy.deepcopy(viewpoint_cameras[0])
        cam.image_height = int(cam.image_height / 2) * 2
        cam.image_width = int(cam.image_width / 2) * 2
        cam.world_view_transform = torch.from_numpy(np.linalg.inv(c2w).T).float().cuda()
        cam.full_proj_transform = (
            cam.world_view_transform.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0))).squeeze(0)
        cam.camera_center = cam.world_view_transform.inverse()[3, :3]
        traj.append(cam)

    return traj


def depths_to_points(depthmap, c2w, w, h, tanfovx, tanfovy):
    c2w = c2w
    W, H = w, h
    fx = W / (2 * math.tan(tanfovx / 2.))
    fy = H / (2 * math.tan(tanfovy / 2.))

    intrins = torch.tensor([[fx, 0., W / 2.], [0., fy, H / 2.], [0., 0., 1.0]]).float().cuda()

    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float()+ 0.5, torch.arange(H, device='cuda').float()+ 0.5,
                                    indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = points @ intrins.inverse().T @ c2w[:3, :3].T
    rays_o = c2w[:3, 3]
    # rays_d = points @ intrins.inverse().T
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points


def depth_to_normal(depth, c2w, w, h, tanfovx, tanfovy):
    """
        view: view camera
        depth: depthmap
    """
    points = depths_to_points(depth, c2w, w, h, tanfovx, tanfovy).reshape(*depth.shape[1:], 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output

