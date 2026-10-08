from math import exp

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from typing import Tuple, Dict, Any
import torchvision.transforms as transforms

transform1 = transforms.CenterCrop((576, 768))
transform2 = transforms.CenterCrop((544, 736))

def get_loss_mapping_rgbd(image, depth, gt_image, gt_depth):
    alpha = 0.95
    rgb_boundary_threshold = 0.01

    gt_image = gt_image.cuda()
    gt_depth = gt_depth.to(dtype=torch.float32, device=image.device)[None]

    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*depth.shape)
    depth_pixel_mask = torch.logical_and(gt_depth > 0.01, depth > 0.01).view(*depth.shape)

    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask).mean()
    l1_depth = torch.abs(1./depth * depth_pixel_mask - 1./gt_depth * depth_pixel_mask).mean()
    return alpha * l1_rgb + (1 - alpha) * l1_depth * 5
    
def l1_loss(network_output: torch.Tensor, gt: torch.Tensor, agg="mean") -> torch.Tensor:
    """
    Computes the L1 loss, which is the mean absolute error between the network output and the ground truth.

    Args:
        network_output: The output from the network.
        gt: The ground truth tensor.
        agg: The aggregation method to be used. Defaults to "mean".
    Returns:
        The computed L1 loss.
    """
    l1_loss = torch.abs(network_output - gt)
    if agg == "mean":
        return l1_loss.mean()
    elif agg == "sum":
        return l1_loss.sum()
    elif agg == "none":
        return l1_loss
    else:
        raise ValueError("Invalid aggregation method.")


def gaussian(window_size: int, sigma: float) -> torch.Tensor:
    """
    Creates a 1D Gaussian kernel.

    Args:
        window_size: The size of the window for the Gaussian kernel.
        sigma: The standard deviation of the Gaussian kernel.

    Returns:
        The 1D Gaussian kernel.
    """
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 /
                         float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size: int, channel: int) -> Variable:
    """
    Creates a 2D Gaussian window/kernel for SSIM computation.

    Args:
        window_size: The size of the window to be created.
        channel: The number of channels in the image.

    Returns:
        A 2D Gaussian window expanded to match the number of channels.
    """
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(
        _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(
        channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, size_average: bool = True) -> torch.Tensor:
    """
    Computes the Structural Similarity Index (SSIM) between two images.

    Args:
        img1: The first image.
        img2: The second image.
        window_size: The size of the window to be used in SSIM computation. Defaults to 11.
        size_average: If True, averages the SSIM over all pixels. Defaults to True.

    Returns:
        The computed SSIM value.
    """
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1: torch.Tensor, img2: torch.Tensor, window: Variable, window_size: int,
          channel: int, size_average: bool = True) -> torch.Tensor:
    """
    Internal function to compute the Structural Similarity Index (SSIM) between two images.

    Args:
        img1: The first image.
        img2: The second image.
        window: The Gaussian window/kernel for SSIM computation.
        window_size: The size of the window to be used in SSIM computation.
        channel: The number of channels in the image.
        size_average: If True, averages the SSIM over all pixels.

    Returns:
        The computed SSIM value.
    """
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window,
                         padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window,
                         padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window,
                       padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
        ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def isotropic_loss(scaling: torch.Tensor) -> torch.Tensor:
    """
    Computes loss enforcing isotropic scaling for the 3D Gaussians
    Args:
        scaling: scaling tensor of 3D Gaussians of shape (n, 3)
    Returns:
        The computed isotropic loss
    """
    mean_scaling = scaling.mean(dim=1, keepdim=True)
    isotropic_diff = torch.abs(scaling - mean_scaling * torch.ones_like(scaling))
    return isotropic_diff.mean()


class ScaleRegularizationMetricsModuleMixin:
    def __init__(self):
        self.scale_reg_from = 0
        self.scale_ratio_reg_lambda = 0
        self.scale_reg_lambda = 0.005
        self.max_scale = 0.05
        self.max_scale_ratio = 100

    def get_scale_regularization_metrics(self, gaussian_model, metrics, pbar):
        scales = gaussian_model.get_scaling()
        sorted_scales = torch.sort(scales, dim=-1).values

        max_scales = sorted_scales[:, -1]
        mid_scales = sorted_scales[:, -2]

        n_over_scales = 0
        over_scale_loss = 0.
        is_over_scales = None
        if self.scale_reg_lambda > 0.:
            is_over_scales = scales.detach() > self.max_scale
            n_over_scales = is_over_scales.sum()
            over_scale_loss = (scales * is_over_scales).sum() / (n_over_scales + 1) * self.scale_reg_lambda

        n_over_ratios = 0
        over_ratio_loss = 0
        is_over_ratios = None
        if self.scale_ratio_reg_lambda > 0.:
            scale_ratios = max_scales / (mid_scales + 1e-8)
            is_over_ratios = scale_ratios.detach() > self.max_scale_ratio
            n_over_ratios = is_over_ratios.sum()
            over_ratio_loss = (scale_ratios * is_over_ratios).sum() / (n_over_ratios + 1) * self.scale_ratio_reg_lambda
        else:
            n_over_ratios = 0
            over_ratio_loss = 0

        if "loss" not in metrics:
            metrics["loss"] = 0.0
        metrics["loss"] = metrics["loss"] + over_scale_loss + over_ratio_loss
        metrics["scale_reg"] = over_scale_loss
        metrics["scale_ratio_reg"] = over_ratio_loss
        metrics["n_over_scales"] = n_over_scales
        metrics["n_over_ratios"] = n_over_ratios
        # with torch.no_grad():
        #     metrics["max_scale"] = max_scales.max()
        #     metrics["max_ratio"] = scale_ratios.max()
        #     metrics["mean_ratio"] = scale_ratios.mean()
        #     metrics["mean_scale"] = max_scales.mean()

        pbar["scale_reg"] = False
        pbar["scale_ratio_reg"] = False
        pbar["n_over_scales"] = False
        pbar["n_over_ratios"] = False
        pbar["max_scale"] = False
        pbar["max_ratio"] = False
        pbar["mean_ratio"] = False

        return sorted_scales, is_over_scales, is_over_ratios

    def get_train_metrics(
            self,
            gaussian_model,
            step: int,
            batch,
    ) -> Tuple[Dict[str, Any], Dict[str, bool]]:

        metrics = {}
        pbar = {}
        if step >= self.scale_reg_from:
            self.get_scale_regularization_metrics(gaussian_model, metrics, pbar)

        return metrics, pbar

def TVLoss(network_output, pred_output, edge_margin=1e-2, margin=1e-4):
    """Total variation loss for a 2D image. input is expected to be of shape (channel, h, w)"""
    h_diff = torch.max((network_output[:, 1:, :] - network_output[:, :-1, :]).abs() - margin,
                       torch.zeros_like(network_output[:, 1:, :])) * ((pred_output[:, 1:, :] - pred_output[:, :-1, :]).abs() < edge_margin).float()
    w_diff = torch.max((network_output[:, :, 1:] - network_output[:, :, :-1]).abs() - margin,
                       torch.zeros_like(network_output[:, :, 1:])) * ((pred_output[:, :, 1:] - pred_output[:, :, :-1]).abs() < edge_margin).float()
    return torch.mean(h_diff) + torch.mean(w_diff)

def patchify(img, patch_size):
    img = img.unsqueeze(0)
    img = F.unfold(img, patch_size, stride=patch_size)
    img = img.transpose(2, 1).contiguous()
    return img.view(-1, patch_size, patch_size)

def patched_depth_ranking_loss(surf_depth, mono_depth, patch_size=-1, margin=1e-4):
    if patch_size > 0:
        surf_depth_patches = patchify(surf_depth, patch_size).view(-1, patch_size * patch_size) # [N, P*P]
        mono_depth_patches = patchify(mono_depth, patch_size).view(-1, patch_size * patch_size)
    else:
        surf_depth_patches = surf_depth.reshape(-1).unsqueeze(0)
        mono_depth_patches = mono_depth.reshape(-1).unsqueeze(0)

    length = (surf_depth_patches.shape[1]) // 2 * 2
    rand_indices = torch.randperm(length)
    surf_depth_patches_rand = surf_depth_patches[:, rand_indices]
    mono_depth_patches_rand = mono_depth_patches[:, rand_indices]

    patch_rank_loss = torch.max(
        torch.sign(mono_depth_patches_rand[:, :length // 2] - mono_depth_patches_rand[:, length // 2:]) * \
            (surf_depth_patches_rand[:, length // 2:] - surf_depth_patches_rand[:, :length // 2]) + margin,
        torch.zeros_like(mono_depth_patches_rand[:, :length // 2], device=mono_depth_patches_rand.device)
    ).mean()

    return patch_rank_loss

def get_depth_ranking_loss(surf_depth, mono_depth, object_mask=None):
    depth_rank_loss = 0.0

    for transform in [transform1, transform2]:
        surf_depth_crop = transform(surf_depth)
        mono_depth_crop = transform(mono_depth.unsqueeze(0))

        object_mask_crop = None
        if object_mask is not None:
            object_mask_crop = transform(object_mask)
            surf_depth_crop[object_mask_crop.float() < 0.5] = -1e-4
            mono_depth_crop[object_mask_crop.float() < 0.5] = -1e-4

        depth_rank_loss += 0.5 * patched_depth_ranking_loss(surf_depth_crop, mono_depth_crop, patch_size=32)

    return depth_rank_loss
