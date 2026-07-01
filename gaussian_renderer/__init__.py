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

    confidence = pc.confidence if pipe.use_confidence else torch.ones_like(pc.confidence)
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
        confidence=confidence  # ✅ 加上这行
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



    # # DropGaussian
    # if is_train:
    #
    #     if iteration in [2000, 9999]:
    #         with torch.no_grad():
    #             import matplotlib.pyplot as plt
    #             import numpy as np  # 确保导入numpy
    #
    #             op_np = opacity.squeeze().detach().cpu().numpy()
    #             point_count = op_np.size  # 获取点的数量
    #
    #             # 保存直方图图片（原有逻辑）
    #             plt.figure(figsize=(6, 4))
    #             plt.hist(op_np, bins=50, range=(0, 1))
    #             plt.title(f"Opacity Distribution @ iter {iteration} (Total Points: {point_count})")
    #             plt.xlabel("Opacity")
    #             plt.ylabel("Count")
    #             plt.tight_layout()
    #             save_path = f"opacity_hist_iter_{iteration}.png"
    #             plt.savefig(save_path)
    #             plt.close()
    #             print(f"[Opacity Histogram Saved] {save_path}")
    #
    #             # 保存不透明度原始数据（核心需求）
    #             data_save_path = f"opacity_data_iter_{iteration}.npy"
    #             np.save(data_save_path, op_np)
    #             print(f"[Opacity Data Saved] {data_save_path}")
    #             print(f"[Data Info] Total points: {point_count} (不透明度数组形状: {op_np.shape})")
    #
    #     opacity_values = opacity.squeeze()
    #
    #     # Create initial compensation factor (1 for each Gaussian)
    #     compensation = torch.ones(opacity.shape[0], dtype=torch.float32, device="cuda")
    #
    #     # Apply DropGaussian with compensation
    #     drop_rate = 0.2 * (iteration / 10000)
    #
    #     # save_drop_rate(iteration, drop_rate)
    #
    #
    #     d = torch.nn.Dropout(p=drop_rate)
    #     compensation = d(compensation)
    #
    #     # Apply to opacity
    #     opacity = opacity * compensation[:, None]


    if is_train:

        # if iteration in [2000, 9999]:
        #     with torch.no_grad():
        #
        #         op_np = opacity.squeeze().detach().cpu().numpy()
        #         point_count = op_np.size  # 获取点的数量
        #
        #         # 保存直方图图片（原有逻辑）
        #         plt.figure(figsize=(6, 4))
        #         plt.hist(op_np, bins=50, range=(0, 1))
        #         plt.title(f"Opacity Distribution @ iter {iteration} (Total Points: {point_count})")
        #         plt.xlabel("Opacity")
        #         plt.ylabel("Count")
        #         plt.tight_layout()
        #         save_path = f"opacity_hist_iter_{iteration}.png"
        #         plt.savefig(save_path)
        #         plt.close()
        #         print(f"[Opacity Histogram Saved] {save_path}")
        #
        #         # 保存不透明度原始数据（核心需求）
        #         data_save_path = f"opacity_data_iter_{iteration}.npy"
        #         np.save(data_save_path, op_np)
        #         print(f"[Opacity Data Saved] {data_save_path}")
        #         print(f"[Data Info] Total points: {point_count} (不透明度数组形状: {op_np.shape})")

        with torch.no_grad():
            # 使用当前不透明度值作为重要性指标
            opacity_values = opacity.squeeze()
            importance = (opacity_values - opacity_values.min()) / (
                        opacity_values.max() - opacity_values.min() + 1e-8)
            drop_bias = 1.0 - importance  # 重要性越低，drop倾向越高

            # --------------------------
            # 计算点云特性（用于动态调整base_drop）
            # --------------------------
            # 1. 不透明度均值（反映整体重要性：均值高则整体更重要）
            opacity_mean = opacity_values.mean()  # 范围[0,1]（假设opacity已归一化）

            # 2. 高不透明度点的比例（反映关键信息占比）
            high_opacity_ratio = (opacity_values > 0.3).float().mean()  # 不透明度>0.5的点占比

            # 3. 点云密度（假设points是点云坐标，shape为[N,3]）
            points = pc.get_xyz  # 确保pc.get_xyz返回的是[N,3]的tensor
            if 'points' in locals() and points is not None and points.shape[0] > 0:
                min_coords = points.min(dim=0).values
                max_coords = points.max(dim=0).values
                bbox_size = (max_coords - min_coords).prod() + 1e-8
                point_density = points.shape[0] / bbox_size
                density_norm = torch.sigmoid(torch.log(point_density + 1e-8))  # 对数归一化
            else:
                density_norm = 0.5  # 若无点坐标，用默认值

            base_factor = (1.0 - opacity_mean) * (1.0 + density_norm) * (1.0 - high_opacity_ratio)
            # base_factor = (1.0 - opacity_mean) * (1.0 + density_norm)

            base_factor = torch.clamp(base_factor, 0.1, 0.8)  # 限制范围，避免极端值

            # save_dir = "./output_point_clouds"
            # os.makedirs(save_dir, exist_ok=True)  # 自动创建目录（不存在则创建，存在则跳过）
            # save_path = os.path.join(save_dir, f"pc_iter_{iteration}.ply")  # 完整路径（目录+文件名）
            # # 2. 补全save_path：目录+文件名（原代码只有目录，没有具体文件名）
            # save_path = os.path.join(save_dir, f"pc_iter_{iteration}.ply")  # 生成如"pc_iter_600.ply"
            # if iteration == 2000 or iteration == 9999:
            #     # 3. 传入pc对象（而非points和opacity_values），匹配函数参数要求
            #     if points is not None and points.shape[0] > 0:
            #         # save_point_cloud_ply1(pc, save_path, iteration)  # 正确传参：pc、save_path、iteration
            #         save_point_cloud_ply2(pc, save_path, iteration, opacity_values)
            #     else:
            #         print(f"[Iteration {iteration}] 没有有效的点云数据，跳过保存")

        # 初始化 drop_rates
        drop_rates = torch.zeros(scales.shape[0], device=scales.device, dtype=scales.dtype)

        if iteration < 600:
            # 第一阶段不drop
            pass
        elif iteration < 2000:
            # 第二阶段：随迭代进度增加，基于点云特性动态计算
            phase_progress = (iteration - 600) / (2000 - 600)  # [0,1]
            base_drop = base_factor * phase_progress  # 仅由点云特性和迭代进度决定
            drop_rates = base_drop * drop_bias
        else:
            # 第三阶段：完全基于点云特性，不再随迭代变化
            phase_progress = (iteration - 2000) / (10000 - 2000)  # [0,1]
            base_drop = base_factor * phase_progress * 0.7   # 0.7是相对比例，非固定基准值
            drop_rates = base_drop * drop_bias

        # 应用drop（保持原有逻辑）
        if torch.any(drop_rates > 0):
            rand_values = torch.rand_like(drop_rates)
            drop_mask = rand_values < drop_rates
            compensation = torch.where(drop_mask,
                                        torch.zeros_like(drop_rates),
                                        torch.ones_like(drop_rates) / (1 - drop_rates + 1e-8))
            opacity = opacity * compensation[:, None]

        # if iteration is not None:
        #     save_drop_rate(iteration, drop_rates)


    # # DropGaussian
    # if is_train:
    #
    #     # drop_rates = 0
    #     # # Create initial compensation factor (1 for each Gaussian)
    #     # compensation = torch.ones(opacity.shape[0], dtype=torch.float32, device="cuda")
    #     #
    #     # if iteration < 600:
    #     #     # 第一阶段不drop
    #     #     pass
    #     # elif iteration < 2000:
    #     #     # 第二阶段：随迭代进度增加，基于点云特性动态计算
    #     #     phase_progress = (iteration - 600) / (2000 - 600)  # [0,1]
    #     #     # base_drop = base_factor * phase_progress  # 仅由点云特性和迭代进度决定
    #     #     drop_rates = phase_progress
    #     # else:
    #     #     # 第三阶段：完全基于点云特性，不再随迭代变化
    #     #     phase_progress = (iteration - 2000) / (10000 - 2000)  # [0,1]
    #     #     # base_drop = base_factor * phase_progress * 0.7   # 0.7是相对比例，非固定基准值
    #     #     drop_rates = phase_progress * 0.3
    #
    #     # Create initial compensation factor (1 for each Gaussian)
    #     compensation = torch.ones(opacity.shape[0], dtype=torch.float32, device="cuda")
    #
    #     # Apply DropGaussian with compensation
    #     drop_rate = 0.2 * (iteration / 10000)
    #
    #     d = torch.nn.Dropout(p=drop_rate)
    #     compensation = d(compensation)
    #
    #     # Apply to opacity
    #     opacity = opacity * compensation[:, None]


    # if is_train:
    #     with torch.no_grad():
    #         # 使用当前不透明度值作为重要性指标
    #         opacity_values = opacity.squeeze()
    #         # 不透明度越低，说明点越不重要，drop率越高
    #         importance = (opacity_values - opacity_values.min()) / (opacity_values.max() - opacity_values.min() + 1e-8)
    #         drop_bias = 1.0 - importance  # 重要性越低，drop倾向越高
    #
    #     # 初始化 drop_rates 为与opacity相同形状的零张量
    #     drop_rates = torch.zeros(scales.shape[0], device=scales.device, dtype=scales.dtype)
    #
    #     if iteration < 600:
    #         # 第一阶段不drop，保持为0
    #         pass
    #     elif iteration < 2000:
    #         phase_progress = (iteration - 600) / (2000 - 600)
    #         base_drop = 0.25 * phase_progress
    #         drop_rates = base_drop * (0.3 + 0.7 * drop_bias)
    #     else:
    #         base_drop = 0.35
    #         drop_rates = base_drop * (0.3 + 0.7 * drop_bias)
    #
    #     # 应用drop
    #     if torch.any(drop_rates > 0):
    #         rand_values = torch.rand_like(drop_rates)
    #         drop_mask = rand_values < drop_rates
    #         compensation = torch.where(drop_mask,
    #                                    torch.zeros_like(drop_rates),
    #                                    torch.ones_like(drop_rates) / (1 - drop_rates + 1e-8))
    #         opacity = opacity * compensation[:, None]

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
        "depth": depth  # ✅ 添加这一行
    }


    return out


def save_point_cloud_ply2(pc, save_path, iteration, opacity_values):
    """
    保存带不透明度着色的点云到PLY文件（三阶段颜色映射）：
    - 不透明度 > 0.6 → 红色 (255, 0, 0)
    - 0.3 ≤ 不透明度 ≤ 0.6 → 蓝色 (0, 0, 255)
    - 不透明度 < 0.3 → 绿色 (0, 255, 0)
    Args:
        pc: 点云对象（含xyz坐标）
        save_path: 保存路径（如 "./output_point_clouds/pc_iter_2000.ply"）
        iteration: 迭代次数（仅日志用）
        opacity_values: 每个点的不透明度（shape[N,]，范围[0,1]）
    """
    # 1. 获取点云坐标（确保是[N,3]的numpy数组，float32）
    points = pc.get_xyz  # 点云对象的坐标属性（torch.Tensor[N,3]）
    if points is None or points.shape[0] == 0:
        print(f"[Iteration {iteration}] 无有效点云，跳过保存")
        return
    points_np = points.detach().cpu().numpy().astype(np.float32)  # 转numpy数组

    # 2. 不透明度映射为RGB颜色（核心修改：三阶段阈值着色）
    opacity_np = opacity_values.detach().cpu().numpy()  # 转numpy，范围[0,1]
    num_points = points_np.shape[0]  # 点的总数

    # 初始化RGB通道数组（默认全0，后续按阈值赋值）
    red = np.zeros(num_points, dtype=np.uint8)
    green = np.zeros(num_points, dtype=np.uint8)
    blue = np.zeros(num_points, dtype=np.uint8)

    # 三阶段阈值判断，赋值对应颜色
    # ① 不透明度 > 0.6 → 红色（红=255，绿=0，蓝=0）
    red[opacity_np > 0.6] = 255
    # ② 0.3 ≤ 不透明度 ≤ 0.6 → 蓝色（红=0，绿=0，蓝=255）
    blue[(opacity_np >= 0.3) & (opacity_np <= 0.6)] = 255
    # ③ 不透明度 < 0.3 → 绿色（红=0，绿=255，蓝=0）
    green[opacity_np < 0.3] = 255

    # 合并RGB通道为[N,3]的颜色数组
    colors = np.stack([red, green, blue], axis=1)  # shape[N,3]，每个点对应一组RGB

    # 3. 写入PLY文件（与原逻辑完全一致，无需修改）
    with open(save_path, 'w') as f:
        # PLY文件头（固定格式）
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        # 逐点写入坐标+颜色
        for i in range(num_points):
            x, y, z = points_np[i]
            r, g, b = colors[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")

    print(f"[Iteration {iteration}] 三阶段着色点云已保存到：{save_path}")


def save_point_cloud_ply1(pc, save_path, iteration, opacity_values):
    """
    保存带不透明度着色的点云到PLY文件
    Args:
        pc: 点云对象（含xyz坐标）
        save_path: 保存路径（如 "./output_point_clouds/pc_iter_600.ply"）
        iteration: 迭代次数（仅日志用）
        opacity_values: 每个点的不透明度（shape[N,]，范围[0,1]，来自你的代码）
    """
    # 1. 获取点云坐标（确保是[N,3]的numpy数组，float32）
    points = pc.get_xyz  # 你的原有逻辑，假设返回torch.Tensor[N,3]
    if points is None or points.shape[0] == 0:
        print(f"[Iteration {iteration}] 无有效点云，跳过保存")
        return
    points_np = points.detach().cpu().numpy().astype(np.float32)  # 转numpy

    # 2. 不透明度映射为RGB颜色（高红低蓝）
    opacity_np = opacity_values.detach().cpu().numpy()  # 转numpy，范围[0,1]
    # 映射逻辑：红通道=不透明度，蓝通道=1-不透明度，绿通道=0（简化渐变，也可加绿通道优化）
    red = (opacity_np * 255).astype(np.uint8)    # 不透明度越高，红色越浓（0→0，1→255）
    green = np.zeros_like(red)                   # 绿通道为0，突出红-蓝过渡
    blue = ((1 - opacity_np) * 255).astype(np.uint8)  # 不透明度越低，蓝色越浓（0→255，1→0）
    colors = np.stack([red, green, blue], axis=1)  # shape[N,3]，每个点的RGB值

    # 3. 写入PLY文件（PLY格式是文本/二进制，这里用文本格式，易调试）
    num_points = points_np.shape[0]
    with open(save_path, 'w') as f:
        # PLY文件头（固定格式，声明点数量和属性）
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")  # 颜色属性：红（0-255）
        f.write("property uchar green\n")# 颜色属性：绿（0-255）
        f.write("property uchar blue\n") # 颜色属性：蓝（0-255）
        f.write("end_header\n")

        # 写入每个点的坐标+颜色
        for i in range(num_points):
            x, y, z = points_np[i]
            r, g, b = colors[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")

    print(f"[Iteration {iteration}] 带颜色点云已保存到：{save_path}")


def save_point_cloud_ply(
    pc,  # 点云对象（即代码中的pc）
    save_path,  # PLY文件保存路径（含文件名）
    iteration  # 当前迭代次数（用于日志）
):
    """
    保存点云为PLY文件，包含坐标、RGB颜色、不透明度、缩放、旋转属性
    """
    # 1. 提取点云核心属性（从GPU迁移到CPU，转numpy数组）
    # 坐标 (N, 3)
    xyz = pc.get_xyz.detach().cpu().numpy()
    # 不透明度 (N, 1) → 转(N,)
    opacity = pc._opacity.detach().cpu().squeeze().numpy()
    # 缩放 (N, 3)
    scales = pc._scaling.detach().cpu().numpy()
    # 旋转（四元数，通常为[N,4]，若为轴角则是[N,3]，根据你的模型调整）
    rotation = pc._rotation.detach().cpu().numpy()
    # 颜色（从DC特征转换：3D Gaussian中_features_dc是[1,3]，需解激活到[0,255]）
    # 假设_features_dc是(N,1,3)，先转(N,3)，再解激活（训练时通常归一化到[0,1]，这里反归一化）
    dc_features = pc._features_dc.detach().cpu().squeeze(1).numpy()  # (N,3)
    color = np.clip((dc_features * 2 - 1) * 127.5 + 127.5, 0, 255).astype(np.uint8)  # 转0-255整数

    # 2. 确认点数量一致（防止数据维度错误）
    num_points = xyz.shape[0]
    assert opacity.shape[0] == num_points, "不透明度数量与点数量不匹配"
    assert scales.shape[0] == num_points, "缩放数量与点数量不匹配"
    assert rotation.shape[0] == num_points, "旋转数量与点数量不匹配"
    assert color.shape[0] == num_points, "颜色数量与点数量不匹配"

    # 3. 写入PLY文件（ASCII格式）
    with open(save_path, "w") as f:
        # PLY表头：定义文件格式、顶点数量、属性
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        # 坐标属性（x/y/z为float）
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        # 颜色属性（red/green/blue为uchar，0-255）
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        # 不透明度属性（float，0-1）
        f.write("property float opacity\n")
        # 缩放属性（scaling_x/y/z为float）
        f.write("property float scaling_x\n")
        f.write("property float scaling_y\n")
        f.write("property float scaling_z\n")
        # 旋转属性（若为四元数则4个float，若为轴角则3个，根据你的rotation维度调整）
        if rotation.shape[1] == 4:
            f.write("property float rotation_x\n")
            f.write("property float rotation_y\n")
            f.write("property float rotation_z\n")
            f.write("property float rotation_w\n")
        else:
            f.write("property float rotation_x\n")
            f.write("property float rotation_y\n")
            f.write("property float rotation_z\n")
        # 表头结束
        f.write("end_header\n")

        # 逐行写入每个点的属性（按表头顺序排列）
        for i in range(num_points):
            # 坐标 + 颜色
            x, y, z = xyz[i]
            r, g, b = color[i]
            # 不透明度 + 缩放
            op = opacity[i]
            sx, sy, sz = scales[i]
            # 旋转（按维度取数）
            if rotation.shape[1] == 4:
                rx, ry, rz, rw = rotation[i]
                line = f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b} {op:.6f} {sx:.6f} {sy:.6f} {sz:.6f} {rx:.6f} {ry:.6f} {rz:.6f} {rw:.6f}\n"
            else:
                rx, ry, rz = rotation[i]
                line = f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b} {op:.6f} {sx:.6f} {sy:.6f} {sz:.6f} {rx:.6f} {ry:.6f} {rz:.6f}\n"
            f.write(line)

    print(f"迭代{iteration}：点云PLY文件已保存至 {save_path}（共{num_points}个点）")



def render1(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
            scaling_modifier=1.0, override_color=None, is_train=False, iteration=None,
            render_gradient_overlay=True):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype,
                                          requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    confidence = pc.confidence if pipe.use_confidence else torch.ones_like(pc.confidence)
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
        confidence=confidence
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # Colors
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3,
                                                            (pc.max_sh_degree + 1) ** 2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # ------------------ DropGaussian ------------------
    if is_train:
        with torch.no_grad():
            opacity_values = opacity.squeeze()
            importance = (opacity_values - opacity_values.min()) / (
                        opacity_values.max() - opacity_values.min() + 1e-8)
            drop_bias = 1.0 - importance

        drop_rates = torch.zeros(scales.shape[0], device=scales.device, dtype=scales.dtype)

        if iteration < 600:
            pass
        elif iteration < 2000:
            phase_progress = (iteration - 600) / (2000 - 600)
            base_drop = 0.25 * phase_progress
            drop_rates = base_drop * (0.3 + 0.7 * drop_bias)
        else:
            base_drop = 0.35
            drop_rates = base_drop * (0.3 + 0.7 * drop_bias)

        if torch.any(drop_rates > 0):
            rand_values = torch.rand_like(drop_rates)
            drop_mask = rand_values < drop_rates
            compensation = torch.where(drop_mask,
                                        torch.zeros_like(drop_rates),
                                        torch.ones_like(drop_rates) / (1 - drop_rates + 1e-8))
            opacity = opacity * compensation[:, None]

    # ------------------ Rasterization ------------------
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

    rendered_image = rendered_image.clamp(0, 1)

    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": (radii > 0).nonzero(),
        "radii": radii,
        "depth": depth
    }

    # ------------------ Gradient overlay ------------------
    if render_gradient_overlay:
        # 优先使用 point_types
        high_grad_mask = None
        if hasattr(pc, "point_types"):
            high_grad_mask = (pc.point_types == 1).nonzero().squeeze(-1)
        elif hasattr(pc, "high_gradient_indices") and pc.high_gradient_indices is not None:
            high_grad_mask = pc.high_gradient_indices

        if high_grad_mask is not None and len(high_grad_mask) > 0:
            print(f"渲染 {len(high_grad_mask)} 个高梯度点的能量图")
            gradient_overlay = render_gradient_energy_map(
                viewpoint_camera, pc, pipe, bg_color, scaling_modifier,
                high_grad_mask=high_grad_mask
            )
            out["gradient_overlay"] = gradient_overlay
        else:
            print("没有高梯度点需要渲染")
            out["gradient_overlay"] = torch.zeros(
                (3, viewpoint_camera.image_height, viewpoint_camera.image_width),
                device="cuda"
            )
    else:
        out["gradient_overlay"] = torch.zeros(
            (3, viewpoint_camera.image_height, viewpoint_camera.image_width),
            device="cuda"
        )

    return out



def render2(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
            scaling_modifier=1.0, override_color=None, is_train=False,
            iteration=None, render_gradient_overlay=True):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype,
                                          requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    confidence = pc.confidence if pipe.use_confidence else torch.ones_like(pc.confidence)
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
        confidence=confidence
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

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

    # DropGaussian with opacity-value-based dropping
    if is_train:
        with torch.no_grad():
            opacity_values = opacity.squeeze()
            importance = (opacity_values - opacity_values.min()) / (
                    opacity_values.max() - opacity_values.min() + 1e-8)
            drop_bias = 1.0 - importance

        drop_rates = torch.zeros(scales.shape[0], device=scales.device, dtype=scales.dtype)

        if iteration < 600:
            pass
        elif iteration < 2000:
            phase_progress = (iteration - 600) / (2000 - 600)
            base_drop = 0.25 * phase_progress
            drop_rates = base_drop * (0.3 + 0.7 * drop_bias)
        else:
            base_drop = 0.35
            drop_rates = base_drop * (0.3 + 0.7 * drop_bias)

        if torch.any(drop_rates > 0):
            rand_values = torch.rand_like(drop_rates)
            drop_mask = rand_values < drop_rates
            compensation = torch.where(drop_mask,
                                       torch.zeros_like(drop_rates),
                                       torch.ones_like(drop_rates) / (1 - drop_rates + 1e-8))
            opacity = opacity * compensation[:, None]

    # Rasterize visible Gaussians to image
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

    rendered_image = rendered_image.clamp(0, 1)

    out = {
        "render": rendered_image.clamp(0, 1),
        "viewspace_points": screenspace_points,
        "visibility_filter": (radii > 0).nonzero(),
        "radii": radii,
        "depth": depth
    }

    # === 高梯度点渲染逻辑 ===
    if render_gradient_overlay:
        # 优先使用 point_types
        if hasattr(pc, "point_types"):
            high_grad_mask = (pc.point_types == 1)  # 例如 type=2 表示高梯度点
        elif hasattr(pc, "high_gradient_mask") and pc.high_gradient_mask is not None:
            high_grad_mask = pc.high_gradient_mask
        else:
            high_grad_mask = None

        if high_grad_mask is not None and high_grad_mask.any():
            num_high_grad = high_grad_mask.sum().item()
            print(f"在渲染图上标记 {num_high_grad} 个高梯度点的位置")

            high_grad_xyz = pc.get_xyz[high_grad_mask]

            if len(high_grad_xyz) > 0:
                hom_points = torch.cat([high_grad_xyz, torch.ones_like(high_grad_xyz[:, :1])], dim=1)
                screen_points = torch.matmul(hom_points, viewpoint_camera.full_proj_transform)
                screen_points = screen_points[:, :2] / screen_points[:, 3:4]

                screen_points[:, 0] = (screen_points[:, 0] + 1) * 0.5 * viewpoint_camera.image_width
                screen_points[:, 1] = (1 - screen_points[:, 1]) * 0.5 * viewpoint_camera.image_height

                visible_mask = (screen_points[:, 0] >= 0) & \
                               (screen_points[:, 0] < viewpoint_camera.image_width) & \
                               (screen_points[:, 1] >= 0) & \
                               (screen_points[:, 1] < viewpoint_camera.image_height)

                visible_points = screen_points[visible_mask].long()

                marked_image = rendered_image.clone()
                for x, y in visible_points:
                    if (0 <= x < viewpoint_camera.image_width and
                            0 <= y < viewpoint_camera.image_height):
                        marked_image[0, y, x] = 1.0  # 红色
                        marked_image[1, y, x] = 0.0
                        marked_image[2, y, x] = 0.0

                out["gradient_overlay"] = marked_image
            else:
                out["gradient_overlay"] = rendered_image.clone()
        else:
            print("没有高梯度点信息")
            out["gradient_overlay"] = rendered_image.clone()
    else:
        out["gradient_overlay"] = rendered_image.clone()

    return out

def render_gradient_energy_map(viewpoint_camera, pc, pipe, bg_color, scaling_modifier, high_grad_mask=None):
    """渲染高梯度点的能量图，体现椭球高斯分布 (中心亮，边缘暗)"""

    device = bg_color.device
    H, W = viewpoint_camera.image_height, viewpoint_camera.image_width

    # ---------------------------- 处理高梯度点索引 ----------------------------
    if high_grad_mask is None:
        if hasattr(pc, 'point_types'):
            high_grad_mask = (pc.point_types == 1)
        elif hasattr(pc, 'high_gradient_indices') and pc.high_gradient_indices is not None:
            high_grad_mask = pc.high_gradient_indices
        else:
            return torch.zeros((3, H, W), device=device)

    # 转成索引形式
    if high_grad_mask.dtype == torch.bool:
        render_indices = high_grad_mask.nonzero().squeeze(-1)
    else:
        render_indices = high_grad_mask

    if len(render_indices) == 0:
        return torch.zeros((3, H, W), device=device)

    num_points = len(render_indices)
    num_to_render = max(1, int(num_points * 0.22))  # 可调节采样比例
    rand_indices = torch.randperm(num_points, device=device)[:num_to_render]
    render_indices = render_indices[rand_indices]

    print(f"渲染 {len(render_indices)} 个高梯度点的能量图")

    # ---------------------------- Rasterization 设置 ----------------------------
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype,
                                          requires_grad=False, device=device)

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    energy_bg_color = torch.zeros(3, device=device)  # 黑色背景

    confidence = pc.confidence if pipe.use_confidence else torch.ones_like(pc.confidence)
    raster_settings = GaussianRasterizationSettings(
        image_height=H,
        image_width=W,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=energy_bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=0,  # 禁用 SH，直接用颜色
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        confidence=confidence
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # ---------------------------- 获取高梯度点属性 ----------------------------
    means3D = pc.get_xyz[render_indices]
    means2D = screenspace_points[render_indices]
    opacity = pc.get_opacity[render_indices]

    # 能量颜色 = 常数 1（白色/灰度），真正的扩散由高斯决定
    colors_precomp = torch.ones_like(means3D)

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)[render_indices]
        scales, rotations = None, None
    else:
        cov3D_precomp = None
        scales = pc.get_scaling[render_indices]
        rotations = pc.get_rotation[render_indices]

    # ---------------------------- 渲染 ----------------------------
    rendered_energy_map, _, _, _ = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=None,
        colors_precomp=colors_precomp,
        opacities=opacity,          # 由 opacity 控制整体亮度
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp
    )

    # gamma 校正，让中心更亮边缘更暗
    gamma = 2.2
    rendered_energy_map = torch.pow(rendered_energy_map, 1.0 / gamma)

    # 转灰度保持一致
    gray_value = rendered_energy_map.mean(dim=0, keepdim=True)
    rendered_energy_map = gray_value.repeat(3, 1, 1)

    return rendered_energy_map.clamp(0, 1)


def render_gradient_energy_map1(viewpoint_camera, pc, pipe, bg_color, scaling_modifier, high_grad_mask=None):
    """渲染高梯度点的能量图，使用黑白亮度映射
       high_grad_mask: tensor of point indices or boolean mask for high-gradient points
    """
    device = bg_color.device
    H, W = viewpoint_camera.image_height, viewpoint_camera.image_width

    # ---------------------------- 处理高梯度点索引 ----------------------------
    if high_grad_mask is None:
        if hasattr(pc, 'point_types'):
            high_grad_mask = (pc.point_types == 1)
        elif hasattr(pc, 'high_gradient_indices') and pc.high_gradient_indices is not None:
            high_grad_mask = pc.high_gradient_indices
        else:
            # 没有高梯度点
            return torch.zeros((3, H, W), device=device)

    # 转成索引形式
    if high_grad_mask.dtype == torch.bool:
        render_indices = high_grad_mask.nonzero().squeeze(-1)
    else:
        render_indices = high_grad_mask

    if len(render_indices) == 0:
        return torch.zeros((3, H, W), device=device)

    num_points = len(render_indices)
    # 可选：随机采样一部分高梯度点以加速渲染
    num_to_render = max(1, int(num_points * 0.22))
    rand_indices = torch.randperm(num_points, device=device)[:num_to_render]
    render_indices = render_indices[rand_indices]

    print(f"渲染 {len(render_indices)} 个高梯度点的能量图")

    # ---------------------------- Rasterization 设置 ----------------------------
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype,
                                          requires_grad=False, device=device)
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    energy_bg_color = torch.zeros(3, device=device)  # 黑色背景

    confidence = pc.confidence if pipe.use_confidence else torch.ones_like(pc.confidence)
    raster_settings = GaussianRasterizationSettings(
        image_height=H,
        image_width=W,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=energy_bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        confidence=confidence
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # ---------------------------- 获取高梯度点属性 ----------------------------
    means3D = pc.get_xyz[render_indices]
    means2D = screenspace_points[render_indices]
    opacity = pc.get_opacity[render_indices]

    # 能量值 = opacity 归一化
    energy_values = opacity.squeeze()
    if energy_values.max() > energy_values.min():
        energy_values = (energy_values - energy_values.min()) / (energy_values.max() - energy_values.min())
    else:
        energy_values = torch.ones_like(energy_values)

    # 颜色映射到灰度
    colors_precomp = torch.zeros_like(means3D)
    colors_precomp[:, 0] = energy_values
    colors_precomp[:, 1] = energy_values
    colors_precomp[:, 2] = energy_values

    # 增强不透明度
    enhanced_opacity = opacity * 2.0

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)[render_indices]
        scales, rotations = None, None
    else:
        cov3D_precomp = None
        scales = pc.get_scaling[render_indices]
        rotations = pc.get_rotation[render_indices]

    # 禁用 SH
    shs = None

    # ---------------------------- 渲染 ----------------------------
    rendered_energy_map, radii, depth, alpha = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=enhanced_opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp
    )

    # 伽马校正
    gamma = 1.5
    rendered_energy_map = torch.pow(rendered_energy_map, 1.0 / gamma)

    # 保持灰度一致
    gray_value = rendered_energy_map.mean(dim=0, keepdim=True)
    rendered_energy_map = gray_value.repeat(3, 1, 1)

    return rendered_energy_map.clamp(0, 1)

def render_gradient_points_only1(viewpoint_camera, pc, pipe, bg_color, scaling_modifier):
    """只渲染高梯度点中最重要的前10%，并保持和 render 一致的逻辑"""
    if not hasattr(pc, 'high_gradient_indices') or pc.high_gradient_indices is None:
        return torch.zeros((3, viewpoint_camera.image_height, viewpoint_camera.image_width), device="cuda")

    if len(pc.high_gradient_indices) == 0:
        return torch.zeros((3, viewpoint_camera.image_height, viewpoint_camera.image_width), device="cuda")

    # 选择前10%的高梯度点
    num_points = len(pc.high_gradient_indices)
    num_to_render = max(1, int(num_points * 0.28))

    opacity_scores = pc._opacity[pc.high_gradient_indices].squeeze()
    _, sorted_indices = torch.sort(opacity_scores, descending=True)
    selected_indices = sorted_indices[:num_to_render]
    render_indices = pc.high_gradient_indices[selected_indices]

    print(f"从 {num_points} 个高梯度点中选择渲染不透明度最高的 {num_to_render} 个点")

    # ----------------------------
    # 和 render1 一致的变量准备
    # ----------------------------
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype,
                                          requires_grad=False, device="cuda")
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    confidence = pc.confidence if pipe.use_confidence else torch.ones_like(pc.confidence)
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
        confidence=confidence
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # 只取前 10% 高梯度点的属性
    means3D = pc.get_xyz[render_indices]
    means2D = screenspace_points[render_indices]
    opacity = pc.get_opacity[render_indices]

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)[render_indices]
        scales, rotations = None, None
    else:
        cov3D_precomp = None
        scales = pc.get_scaling[render_indices]
        rotations = pc.get_rotation[render_indices]

    shs, colors_precomp = None, None
    if pipe.convert_SHs_python:
        shs_view = pc.get_features[render_indices].transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
        dir_pp = (pc.get_xyz[render_indices] - viewpoint_camera.camera_center.repeat(len(render_indices), 1))
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    else:
        shs = pc.get_features[render_indices]

    # ----------------------------
    # 调用 rasterizer
    # ----------------------------
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

    return rendered_image.clamp(0, 1)

def render_gradient_points_only(viewpoint_camera, pc, pipe, bg_color, scaling_modifier):
    """随机渲染高梯度点中 28% 的点，并保持和 render 一致的逻辑"""
    if not hasattr(pc, 'high_gradient_indices') or pc.high_gradient_indices is None:
        return torch.zeros((3, viewpoint_camera.image_height, viewpoint_camera.image_width), device="cuda")

    if len(pc.high_gradient_indices) == 0:
        return torch.zeros((3, viewpoint_camera.image_height, viewpoint_camera.image_width), device="cuda")

    num_points = len(pc.high_gradient_indices)
    num_to_render = max(1, int(num_points * 0.28))  # 至少渲染 1 个点

    # ✅ 随机选取 28% 的点
    rand_indices = torch.randperm(num_points, device="cuda")[:num_to_render]
    render_indices = pc.high_gradient_indices[rand_indices]

    print(f"从 {num_points} 个高梯度点中随机选择 {num_to_render} 个点进行渲染")

    # ----------------------------
    # 和 render1 一致的变量准备
    # ----------------------------
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype,
                                          requires_grad=False, device="cuda")
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    confidence = pc.confidence if pipe.use_confidence else torch.ones_like(pc.confidence)
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
        confidence=confidence
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz[render_indices]
    means2D = screenspace_points[render_indices]
    opacity = pc.get_opacity[render_indices]

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)[render_indices]
        scales, rotations = None, None
    else:
        cov3D_precomp = None
        scales = pc.get_scaling[render_indices]
        rotations = pc.get_rotation[render_indices]

    shs, colors_precomp = None, None
    if pipe.convert_SHs_python:
        shs_view = pc.get_features[render_indices].transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
        dir_pp = (pc.get_xyz[render_indices] - viewpoint_camera.camera_center.repeat(len(render_indices), 1))
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    else:
        shs = pc.get_features[render_indices]

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

    return rendered_image.clamp(0, 1)


