#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import os
import csv
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
import json
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from torchvision.utils import save_image


DROP_LOG_PATH = "drop_rate_log.json"


def save_drop_rate(iteration, drop_rates):
    """
    Save drop_rate statistics into JSON file.
    drop_rates: tensor of shape [N]
    """

    mean_v = float(drop_rates.mean())
    max_v = float(drop_rates.max())
    min_v = float(drop_rates.min())

    # init file if missing
    if not os.path.exists(DROP_LOG_PATH):
        init_data = {
            "iterations": [],
            "mean_drop_rate": [],
            "max_drop_rate": [],
            "min_drop_rate": []
        }
        with open(DROP_LOG_PATH, "w") as f:
            json.dump(init_data, f, indent=4)

    with open(DROP_LOG_PATH, "r") as f:
        data = json.load(f)

    # append
    data["iterations"].append(int(iteration))
    data["mean_drop_rate"].append(mean_v)
    data["max_drop_rate"].append(max_v)
    data["min_drop_rate"].append(min_v)

    with open(DROP_LOG_PATH, "w") as f:
        json.dump(data, f, indent=4)


def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0, \
           override_color=None, is_train=False, iteration=None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    if is_train:

        with torch.no_grad():
            opacity_values = opacity.squeeze()
            importance = (opacity_values - opacity_values.min()) / (
                        opacity_values.max() - opacity_values.min() + 1e-8)
            drop_bias = 1.0 - importance  
            opacity_mean = opacity_values.mean() 
            high_opacity_ratio = (opacity_values > 0.3).float().mean()  

            points = pc.get_xyz 
            if 'points' in locals() and points is not None and points.shape[0] > 0:
                min_coords = points.min(dim=0).values
                max_coords = points.max(dim=0).values
                bbox_size = (max_coords - min_coords).prod() + 1e-8
                point_density = points.shape[0] / bbox_size
                density_norm = torch.sigmoid(torch.log(point_density + 1e-8))  
            else:
                density_norm = 0.5  

            base_factor = (1.0 - opacity_mean) * (1.0 + density_norm) * (1.0 - high_opacity_ratio)

            base_factor = torch.clamp(base_factor, 0.1, 0.8)  

        drop_rates = torch.zeros(scales.shape[0], device=scales.device, dtype=scales.dtype)

        if iteration < 600:
            pass
        elif iteration < 2000:
            phase_progress = (iteration - 600) / (2000 - 600)  
            base_drop = base_factor * phase_progress  
            drop_rates = base_drop * drop_bias
        else:
            phase_progress = (iteration - 2000) / (10000 - 2000)  
            base_drop = base_factor * phase_progress * 0.7  
            drop_rates = base_drop * drop_bias

        if torch.any(drop_rates > 0):
            rand_values = torch.rand_like(drop_rates)
            drop_mask = rand_values < drop_rates
            compensation = torch.where(drop_mask,
                                        torch.zeros_like(drop_rates),
                                        torch.ones_like(drop_rates) / (1 - drop_rates + 1e-8))
            opacity = opacity * compensation[:, None]


    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    # 调用 rasterizer 并获取 depth
    rendered_image, radii, depth, alpha = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)

    out = {
        "render": rendered_image.clamp(0, 1),
        "viewspace_points": screenspace_points,
        "visibility_filter": (radii > 0).nonzero(),
        "radii": radii,
    }


    return out






