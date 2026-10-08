# from gsplat.rendering import rasterization, rasterization_2dgs
# from gsplat.strategy import MCMCStrategy
# import torch
# from sympy import false
#
# from src.entities.gaussian_model import GaussianModel
# from gaussian_rasterizer import GaussianRasterizationSettings, GaussianRasterizer
#
# def render(gaussian_model:GaussianModel, w, h, intrinsics, w2c, near=0.01, far=100, sh_degree=3):
#     #intrinsics = torch.from_numpy(intrinsics).cuda().float()
#     intrinsics = intrinsics.cuda().float()
#     cam_center = torch.inverse(w2c)[:3, 3]
#     viewmatrix = w2c
#
#     render_colors,render_alphas,render_normals,normals_from_depth,render_distort,render_median,info = rasterization_2dgs(
#         means=gaussian_model.get_xyz(),
#         quats=gaussian_model.get_rotation(),  # F.normalize is fused into the kernel
#         scales=gaussian_model.get_scaling(),
#         opacities=gaussian_model.get_opacity().view(-1),
#         colors=gaussian_model.get_features(),
#         viewmats=viewmatrix,
#         Ks=intrinsics,
#         width=w,
#         height=h,
#         sh_degree=sh_degree,
#         render_mode="RGB+ED",
#     )
#     return render_colors, render_alphas, info
#
# def get_render_settings_3d(w, h, intrinsics, w2c, near=0.01, far=100, sh_degree=0):
#     """
#     Constructs and returns a GaussianRasterizationSettings object for rendering,
#     configured with given camera parameters.
#
#     Args:
#         width (int): The width of the image.
#         height (int): The height of the image.
#         intrinsic (array): 3*3, Intrinsic camera matrix.
#         w2c (array): World to camera transformation matrix.
#         near (float, optional): The near plane for the camera. Defaults to 0.01.
#         far (float, optional): The far plane for the camera. Defaults to 100.
#
#     Returns:
#         GaussianRasterizationSettings: Configured settings for Gaussian rasterization.
#     """
#     fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1,
#                                                   1], intrinsics[0, 2], intrinsics[1, 2]
#     w2c = torch.tensor(w2c).cuda().float()
#     cam_center = torch.inverse(w2c)[:3, 3]
#     viewmatrix = w2c.transpose(0, 1)
#     opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
#                                 [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
#                                 [0.0, 0.0, far /
#                                     (far - near), -(far * near) / (far - near)],
#                                 [0.0, 0.0, 1.0, 0.0]], device='cuda').float().transpose(0, 1)
#     full_proj_matrix = viewmatrix.unsqueeze(
#         0).bmm(opengl_proj.unsqueeze(0)).squeeze(0)
#     return GaussianRasterizationSettings(
#         image_height=h,
#         image_width=w,
#         tanfovx=w / (2 * fx),
#         tanfovy=h / (2 * fy),
#         bg=torch.tensor([0, 0, 0], device='cuda').float(),
#         scale_modifier=1.0,
#         viewmatrix=viewmatrix,
#         projmatrix=full_proj_matrix,
#         sh_degree=sh_degree,
#         campos=cam_center,
#         prefiltered=False,
#         debug=False)
#
# def render_gaussian_model_3d(gaussian_model, render_settings,
#                           override_means_3d=None, override_means_2d=None,
#                           override_scales=None, override_rotations=None,
#                           override_opacities=None, override_colors=None):
#     """
#     Renders a Gaussian model with specified rendering settings, allowing for
#     optional overrides of various model parameters.
#
#     Args:
#         gaussian_model: A Gaussian model object that provides methods to get
#             various properties like xyz coordinates, opacity, features, etc.
#         render_settings: Configuration settings for the GaussianRasterizer.
#         override_means_3d (Optional): If provided, these values will override
#             the 3D mean values from the Gaussian model.
#         override_means_2d (Optional): If provided, these values will override
#             the 2D mean values. Defaults to zeros if not provided.
#         override_scales (Optional): If provided, these values will override the
#             scale values from the Gaussian model.
#         override_rotations (Optional): If provided, these values will override
#             the rotation values from the Gaussian model.
#         override_opacities (Optional): If provided, these values will override
#             the opacity values from the Gaussian model.
#         override_colors (Optional): If provided, these values will override the
#             color values from the Gaussian model.
#     Returns:
#         A dictionary containing the rendered color, depth, radii, and 2D means
#         of the Gaussian model. The keys of this dictionary are 'color', 'depth',
#         'radii', and 'means2D', each mapping to their respective rendered values.
#     """
#     renderer = GaussianRasterizer(raster_settings=render_settings)
#
#     if override_means_3d is None:
#         means3D = gaussian_model.get_xyz()
#     else:
#         means3D = override_means_3d
#
#     if override_means_2d is None:
#         means2D = torch.zeros_like(
#             means3D, dtype=means3D.dtype, requires_grad=True, device="cuda")
#         means2D.retain_grad()
#     else:
#         means2D = override_means_2d
#
#     if override_opacities is None:
#         opacities = gaussian_model.get_opacity()
#     else:
#         opacities = override_opacities
#
#     shs, colors_precomp = None, None
#     if override_colors is not None:
#         colors_precomp = override_colors
#     else:
#         shs = gaussian_model.get_features()
#
#     render_args = {
#         "means3D": means3D,
#         "means2D": means2D,
#         "opacities": opacities,
#         "colors_precomp": colors_precomp,
#         "shs": shs,
#         "scales": gaussian_model.get_scaling() if override_scales is None else override_scales,
#         "rotations": gaussian_model.get_rotation() if override_rotations is None else override_rotations,
#         "cov3D_precomp": None
#     }
#     color, depth, alpha, radii = renderer(**render_args)
#
#     return {"color": color, "depth": depth, "radii": radii, "means2D": means2D, "alpha": alpha}