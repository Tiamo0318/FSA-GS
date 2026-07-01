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
import os
import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
import cv2
import struct
from scipy.spatial.transform import Rotation
from torch_cluster import knn_graph
import torch.nn.functional as F
import math
from torchvision import transforms
from pytorch3d.transforms import quaternion_to_matrix
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import torch.distributions as dist
import csv

class GaussianModel():

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree: int, args):
        self.args = args
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        self.confidence = torch.empty(0)


        self.point_types = torch.zeros(self._xyz.shape[0], dtype=torch.long, device="cuda")

        self.history = {
            'high_gradient': [],  # 高梯度区域点的统计
            'low_gradient': []  # 低梯度区域点的统计
        }
        self.current_iter = 0

        self.progressive_exposure = {
            "indices": torch.tensor([], dtype=torch.long, device="cuda"),
            "step": torch.tensor([], dtype=torch.float32, device="cuda")
        }

        # 其他初始化逻辑...
        self.point_ids = torch.empty(0, dtype=torch.int64, device="cuda")  # 每个点的唯一ID
        self.tracked_flags = torch.zeros(0, dtype=torch.bool, device="cuda")  # 是否为跟踪点（True表示被跟踪）
        self.split_history = []  # 分裂日志：[(iter, point_id, is_parent, parent_id), ...]
        self.next_id = 0  # 下一个可用的点ID


    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
         self._xyz,
         self._features_dc,
         self._features_rest,
         self._scaling,
         self._rotation,
         self._opacity,
         self.max_radii2D,
         xyz_gradient_accum,
         denom,
         opt_dict,
         self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda())[0], 1e-7)

        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.confidence = torch.ones_like(opacities, device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        # === 保存点类型信息（新格式） ===
        if hasattr(self, 'point_types') and self.point_types is not None:
            point_type_path = path.replace('.ply', '_point_types.pt')
            torch.save({
                'point_types': self.point_types.detach().cpu(),
                'format_version': 'types_v1',
                'current_iter': getattr(self, 'current_iter', 0),
            }, point_type_path)
            counts = torch.bincount(self.point_types.detach().cpu())
            print(f"保存点类型到: {point_type_path}")
            print("  各类型点数量:", {i: int(c) for i, c in enumerate(counts)})

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # === 转为 nn.Parameter，保持梯度 ===
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

        # === 新的点类型加载 ===
        point_type_path = path.replace('.ply', '_point_types.pt')
        if os.path.exists(point_type_path):
            try:
                type_data = torch.load(point_type_path)
                self.point_types = type_data['point_types'].to("cuda")
                self.current_iter = type_data.get('current_iter', 0)

                counts = torch.bincount(self.point_types.detach().cpu())
                print(f"成功加载点类型: {point_type_path}")
                print("  各类型点数量:", {i: int(c) for i, c in enumerate(counts)})

            except Exception as e:
                print(f"加载点类型失败: {e}")
                self.point_types = None
        else:
            print(f"未找到点类型文件: {point_type_path}")
            self.point_types = None

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask, iteration):
        # 初始化 prune_mask
        prune_mask = mask
        # === 白名单保护新生成的点 ===
        if hasattr(self, "new_point_mask"):
            prune_mask = torch.logical_and(prune_mask, self.new_point_mask)
            del self.new_point_mask  # 防止影响下一轮

        valid_points_mask = ~prune_mask  # 保留未被标记为删除的点
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.confidence = self.confidence[valid_points_mask]

        if hasattr(self, "point_types"):
            self.point_types = self.point_types[valid_points_mask]


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                    dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                                                       dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                              new_rotation):
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.max_radii2D = torch.cat(
            [self.max_radii2D, torch.zeros(self.get_xyz.shape[0] - self.max_radii2D.shape[0]).to(self.max_radii2D)])
        self.confidence = torch.cat([self.confidence, torch.ones(new_opacities.shape, device="cuda")], 0)

    def compute_frequency_maps(self, image_dir, image_size=(640, 480), cutoff_ratio=0.3, order=2):
        """
        使用巴特沃斯高通滤波（Butterworth High-Pass Filter）提取图像高频分量，
        得到类似梯度的高频响应图。

        Args:
            image_dir: 图像文件夹路径
            image_size: (width, height)，与 cv2.resize 一致
            cutoff_ratio: 截止频率占图像最大频率的比例（0~0.5 常见）
            order: 巴特沃斯高通滤波器阶数

        Returns:
            torch.Tensor: 高频/梯度近似图 [num_cameras, H, W]（CUDA float）
        """
        freq_maps = []
        filenames = sorted(os.listdir(image_dir))

        for fname in filenames:
            img_path = os.path.join(image_dir, fname)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"无法读取图像：{img_path}")

            img = cv2.resize(img, image_size).astype(np.float32) / 255.0  # [0,1]
            H, W = img.shape

            # --- 计算FFT并中心化 ---
            fft = np.fft.fft2(img)
            fft_shift = np.fft.fftshift(fft)

            # --- 构建巴特沃斯高通滤波器 ---
            u = np.arange(-W // 2, W // 2)
            v = np.arange(-H // 2, H // 2)
            U, V = np.meshgrid(u, v)
            D = np.sqrt(U ** 2 + V ** 2)
            D0 = cutoff_ratio * np.sqrt(H ** 2 + W ** 2)
            H_hp = 1 / (1 + (D0 / (D + 1e-6)) ** (2 * order))

            # --- 应用高通滤波器 ---
            fft_hp = fft_shift * H_hp
            img_hp = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_hp)))

            # --- 归一化 ---
            img_min, img_max = img_hp.min(), img_hp.max()
            img_norm = (img_hp - img_min) / (img_max - img_min + 1e-6)

            freq_maps.append(torch.from_numpy(img_norm.astype(np.float32)))

        return torch.stack(freq_maps).to("cuda").float()

    def compute_gradient_maps(self, image_dir, image_size=(640, 480)):
        gradient_maps = []
        filenames = sorted(os.listdir(image_dir))
        for fname in filenames:
            img = cv2.imread(os.path.join(image_dir, fname), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, image_size)

            # sobelx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
            # sobely = cv2.Sobel(img, cv2.CV_32F, 0, 1)
            # grad_mag = np.sqrt(sobelx ** 2 + sobely ** 2)

            scharrx = cv2.Scharr(img, cv2.CV_32F, 1, 0)
            scharry = cv2.Scharr(img, cv2.CV_32F, 0, 1)
            grad_mag = np.sqrt(scharrx ** 2 + scharry ** 2)

            gradient_maps.append(torch.from_numpy(grad_mag))

        return torch.stack(gradient_maps).to("cuda").float()  # [24, H, W]

    def get_detail_weight(self, xyz, projected_coords, gradient_maps):
        if xyz.numel() == 0:
            return torch.zeros((0,), device=xyz.device)

        gradients = []
        # gradient_maps shape: [num_cameras, H, W]
        num_cameras, H, W = gradient_maps.shape

        for i in range(num_cameras):
            # 获取当前相机的投影坐标 (N, 2)，浮点型(x, y)
            coords = projected_coords[:, i]  # [N, 2]，未取整的浮点坐标

            # 分离整数部分和小数部分
            x = coords[:, 0]  # x坐标（列方向）
            y = coords[:, 1]  # y坐标（行方向）

            # 计算周围4个像素的坐标（整数）
            x0 = torch.floor(x).long()  # 左像素列索引
            x1 = x0 + 1  # 右像素列索引
            y0 = torch.floor(y).long()  # 上像素行索引
            y1 = y0 + 1  # 下像素行索引

            # 边界处理：确保坐标在图像范围内
            x0 = torch.clamp(x0, 0, W - 1)
            x1 = torch.clamp(x1, 0, W - 1)
            y0 = torch.clamp(y0, 0, H - 1)
            y1 = torch.clamp(y1, 0, H - 1)

            # 计算小数部分（插值权重）
            wx = x - x0.float()  # x方向小数部分（右像素权重）
            wy = y - y0.float()  # y方向小数部分（下像素权重）

            # 获取4个周围像素的梯度值
            grad00 = gradient_maps[i][y0, x0]  # 左上像素
            grad01 = gradient_maps[i][y0, x1]  # 右上像素
            grad10 = gradient_maps[i][y1, x0]  # 左下像素
            grad11 = gradient_maps[i][y1, x1]  # 右下像素

            # 双线性插值计算
            # 先在x方向插值
            grad_x0 = grad00 * (1 - wx) + grad01 * wx  # 上边缘插值
            grad_x1 = grad10 * (1 - wx) + grad11 * wx  # 下边缘插值
            # 再在y方向插值
            grad_interp = grad_x0 * (1 - wy) + grad_x1 * wy  # 最终插值结果

            gradients.append(grad_interp)

        # 计算所有相机的平均梯度
        mean_grad = torch.stack(gradients, dim=1).mean(dim=1)
        # 归一化到[0, 1]范围
        return (mean_grad - mean_grad.min()) / (mean_grad.max() - mean_grad.min() + 1e-6)

    @staticmethod
    def load_camera_params(cameras_path, images_path):

        def read_next(f, fmt):
            return struct.unpack(fmt, f.read(struct.calcsize(fmt)))

        # --- 读取 cameras.bin ---
        cameras = {}
        with open(cameras_path, 'rb') as f:
            num_cameras = read_next(f, "<Q")[0]
            for _ in range(num_cameras):
                camera_id, model_id = read_next(f, "<ii")
                width, height = read_next(f, "<ii")
                params_len = 4 if model_id == 0 else 5  # e.g. SIMPLE_PINHOLE=3, PINHOLE=4
                params = read_next(f, "<" + "d" * params_len)
                cameras[camera_id] = {
                    'model_id': model_id,
                    'width': width,
                    'height': height,
                    'params': np.array(params[:4], dtype=np.float32)  # fx, fy, cx, cy
                }

        # --- 读取 images.bin ---
        intrinsics_list = []
        extrinsics_list = []
        with open(images_path, 'rb') as f:
            num_images = read_next(f, "<Q")[0]
            for _ in range(num_images):
                image_id = read_next(f, "<i")[0]
                qvec = np.array(read_next(f, "<dddd"), dtype=np.float32)  # quaternion
                tvec = np.array(read_next(f, "<ddd"), dtype=np.float32)  # translation
                camera_id = read_next(f, "<i")[0]
                name = b""
                while True:
                    c = f.read(1)
                    if c == b'\x00':
                        break
                    name += c

                # 跳过 2D 特征点部分
                num_points2D = read_next(f, "<Q")[0]
                f.read(24 * num_points2D)

                # --- 计算 intrinsics ---
                fx, fy, cx, cy = cameras[camera_id]['params']
                intrinsics = np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ], dtype=np.float32)
                intrinsics_list.append(intrinsics)

                # --- 计算 extrinsics ---
                # 将 qvec 转换为旋转矩阵
                R = Rotation.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]]).as_matrix()
                extrinsic = np.hstack((R, tvec.reshape(3, 1))).astype(np.float32)
                extrinsics_list.append(extrinsic)

        return intrinsics_list, extrinsics_list

    def project_points_to_image(self, points_3d, camera_intrinsics, camera_extrinsics):
        # 3D点投影到2D图像
        # points_3d: [N, 3] 3D点
        # camera_intrinsics: [3, 3] 内参矩阵
        # camera_extrinsics: [3, 4] 外参矩阵，包含旋转和平移

        # 添加齐次坐标
        ones = torch.ones((points_3d.shape[0], 1), device=points_3d.device)
        points_3d_homogeneous = torch.cat((points_3d, ones), dim=1)  # [N, 4]

        # 投影到相机坐标系
        points_2d_homogeneous = torch.matmul(points_3d_homogeneous, camera_extrinsics.T)  # [N, 4]

        # 投影到图像平面
        points_2d_homogeneous = torch.matmul(points_2d_homogeneous, camera_intrinsics.T)  # [N, 3]

        # 除以齐次坐标，得到最终的2D像素坐标
        points_2d = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2:3]  # [N, 2]
        return points_2d

    def generate_pixel_coords_per_image(self, points_3d, camera_intrinsics_list, camera_extrinsics_list):
        # points_3d: [N, 3]
        pixel_coords_per_image = []
        for i in range(len(camera_intrinsics_list)):
            intrinsics = torch.tensor(camera_intrinsics_list[i], device=points_3d.device)
            extrinsics = torch.tensor(camera_extrinsics_list[i], device=points_3d.device)
            pixel_coords = self.project_points_to_image(points_3d, intrinsics, extrinsics)  # [N, 2]
            pixel_coords_per_image.append(pixel_coords)  # list of [N, 2]

        # pixel_coords_per_image: list of V x [N, 2]
        # 最终返回 shape: [N, V, 2]
        return torch.stack(pixel_coords_per_image, dim=0).permute(1, 0, 2)

    @staticmethod
    def gaussian_kernel(t, sigma):
        return torch.exp(- (t ** 2) / (2 * sigma ** 2))

    def gaussian_interpolation_adaptive(self,
                                        p1, p2,
                                        intrinsics, extrinsics, gradient_maps,
                                        num_line_samples=10,
                                        steps_min=1
                                        ):
        B = p1.shape[0]
        new_points_list = []
        point_types_list = []
        eps = 1e-4

        for i in range(B):
            dir_vec = p2[i] - p1[i]
            src_point = p1[i]
            tgt_point = p2[i]

            # 区域类型判断
            line_t_temp = torch.linspace(0, 1, steps=num_line_samples, device=p1.device).unsqueeze(1)
            sample_pts_temp = src_point + line_t_temp * dir_vec
            pixel_coords_temp = self.generate_pixel_coords_per_image(sample_pts_temp, intrinsics,
                                                                     extrinsics)  # 函数功能：将点投影胡图像
            line_weights_temp = self.get_detail_weight(sample_pts_temp, pixel_coords_temp, gradient_maps)  # 函数功能，计算梯度值

            if torch.max(line_weights_temp) <= 0.0:
                # 低梯度区域
                num_points = 6
                sigma = 0.10

                def normal_pdf(x, mu, sigma):
                    return torch.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * torch.sqrt(torch.tensor(2 * torch.pi)))

                t_candidates = torch.linspace(eps, 1.0 - eps, steps=10, device=p1.device)
                pdf_p1 = normal_pdf(t_candidates, 0.0, sigma)
                pdf_p2 = normal_pdf(t_candidates, 1.0, sigma)
                mixed_pdf = pdf_p1 + pdf_p2
                mixed_pdf = mixed_pdf / torch.sum(mixed_pdf)

                indices = torch.multinomial(mixed_pdf, num_points, replacement=False)
                t_vals = t_candidates[indices]
                t_vals, _ = torch.sort(t_vals)

                start_point = src_point.reshape(3)
                end_point = tgt_point.reshape(3)
                dir_vec_safe = end_point - start_point
                if torch.norm(dir_vec_safe) > eps:
                    dir_vec_safe = dir_vec_safe / torch.norm(dir_vec_safe)
                else:
                    dir_vec_safe = torch.tensor([1.0, 0.0, 0.0], device=p1.device)

                new_pts = start_point.unsqueeze(0) + t_vals.unsqueeze(1) * dir_vec_safe.unsqueeze(0)
                point_types = torch.full((num_points,), 0, device=p1.device)

            else:
                eps = 1e-8  # 避免边界重叠

                num_intervals = 15  # 8个等分区间
                total_sample_points = 10  # 总共插入10个点
                interval_step = 1.0 / num_intervals

                # 1. 生成8个区间的中心点t值和3D坐标
                interval_centers_t = torch.linspace(
                    interval_step / 2, 1 - interval_step / 2,
                    steps=num_intervals, device=p1.device
                )
                center_3d_pts = src_point + interval_centers_t.unsqueeze(1) * dir_vec

                # 2. 计算每个区间中心点的梯度权重
                center_pixel_coords = self.generate_pixel_coords_per_image(center_3d_pts, intrinsics, extrinsics)
                interval_weights = self.get_detail_weight(center_3d_pts, center_pixel_coords, gradient_maps)

                # 3. 新增线段端点（t=0和t=1），梯度权重设为0
                weights_t = torch.cat([
                    torch.tensor([0.0], device=p1.device),  # 起点t=0
                    interval_centers_t,
                    torch.tensor([1.0], device=p1.device)  # 终点t=1
                ], dim=0)

                weights = torch.cat([
                    torch.tensor([0.0], device=p1.device),  # 起点梯度设为0
                    interval_weights,
                    torch.tensor([0.0], device=p1.device)  # 终点梯度设为0
                ], dim=0)

                # 4. 权重归一化并计算CDF（累积分布函数）
                w = weights + 1e-6  # 避免梯度为0导致概率为0
                w = w / w.sum()  # 权重归一化（总和=1）
                cdf = torch.cumsum(w, dim=0)

                # 5. 生成采样点的CDF查询值（0到1均匀分布）
                t_samples = torch.linspace(0, 1, steps=total_sample_points, device=p1.device)  # [10,]

                # 6. 查找插值索引
                base_grid = weights_t  # 权重节点的t坐标（有序：0→0.05→...→0.95→1.0）
                # 找到每个t_sample在CDF中对应的区间索引
                inds = torch.searchsorted(cdf, t_samples, right=True).clamp(max=len(base_grid) - 1)
                inds0 = (inds - 1).clamp(min=0)  # 区间左端点索引
                inds1 = inds  # 区间右端点索引

                # 7. 线性插值计算最终t值
                cdf0, cdf1 = cdf[inds0], cdf[inds1]  # 区间两端的CDF值
                grid0, grid1 = base_grid[inds0], base_grid[inds1]  # 区间两端的t坐标
                denom = (cdf1 - cdf0 + 1e-8)  # 避免分母为0
                t_vals = grid0 + (t_samples - cdf0) / denom * (grid1 - grid0)  # 线性插值得到t值

                # 8. 确保t值有序
                t_vals, _ = torch.sort(t_vals)

                # 9. 生成高梯度区域插值点
                new_pts = src_point.unsqueeze(0) + t_vals.unsqueeze(1) * dir_vec.unsqueeze(0)  # [10, 3]
                point_types = torch.full((total_sample_points,), 1, device=p1.device)  # 1表示高梯度区域

            new_points_list.append(new_pts)
            point_types_list.append(point_types)

        new_points = torch.cat(new_points_list, dim=0)
        point_types = torch.cat(point_types_list, dim=0)

        return new_points, point_types

    def proximity_fsgs(self, model: 'GaussianModel', scene_extent, current_iter, N=3):
        dist, nearest_indices = distCUDA2(self.get_xyz)
        selected_pts_mask = torch.logical_and(dist > (5. * scene_extent),
                                              torch.max(self.get_scaling, dim=1).values > (scene_extent))

        new_indices = nearest_indices[selected_pts_mask].reshape(-1).long()
        source_xyz = self._xyz[selected_pts_mask].repeat(1, N, 1).reshape(-1, 3)
        target_xyz = self._xyz[new_indices]
        new_xyz = (source_xyz + target_xyz) / 2
        new_scaling = self._scaling[new_indices]
        new_rotation = torch.zeros_like(self._rotation[new_indices])
        new_rotation[:, 0] = 1
        new_features_dc = torch.zeros_like(self._features_dc[new_indices])
        new_features_rest = torch.zeros_like(self._features_rest[new_indices])
        new_opacity = self._opacity[new_indices]
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

    def proximity(self, model: 'GaussianModel', scene_extent, current_iter, N=3):
        # Step 1: 计算图像梯度图（用于 detail_weight）
        image_dir = "/media/junz/4TB-11/liyangeng/papers/DropGaussian/dataset/nerf_llff_data/leaves/3_views/images"
        # gradient_maps = self.compute_frequency_maps(image_dir, cutoff_ratio=0.3, order=2)
        gradient_maps = self.compute_gradient_maps(image_dir)

        # Step 1.1:
        cameras_path = "/media/junz/4TB-11/liyangeng/papers/DropGaussian/dataset/nerf_llff_data/leaves/3_views/triangulated/cameras.bin"
        images_path = "/media/junz/4TB-11/liyangeng/papers/DropGaussian/dataset/nerf_llff_data/leaves/3_views/triangulated/images.bin"
        intrinsics, extrinsics = self.load_camera_params(cameras_path, images_path)

        # Step 2: 找出需要插值的点（距离远 且 尺度大）
        dist, nearest_indices = distCUDA2(model.get_xyz)
        selected_pts_mask = torch.logical_and(
            dist > (5. * scene_extent),
            torch.max(model.get_scaling, dim=1).values > scene_extent
        )
        source_xyz = model._xyz[selected_pts_mask].repeat(1, N, 1).reshape(-1, 3)
        if source_xyz.shape[0] == 0:
            return

        new_indices = nearest_indices[selected_pts_mask].reshape(-1).long()
        target_xyz = model._xyz[new_indices]

        # Step 3: 计算图像细节权重（图像梯度）
        pixel_coords_per_image = self.generate_pixel_coords_per_image(source_xyz, intrinsics, extrinsics)
        detail_weight = self.get_detail_weight(source_xyz, pixel_coords_per_image, gradient_maps)  # [B]

        # Step 4: 基于梯度分布的自适应高斯插值生成新点
        new_xyz, new_point_types = self.gaussian_interpolation_adaptive(
            p1=source_xyz,
            p2=target_xyz,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            gradient_maps=gradient_maps,
            num_line_samples=20,

            steps_min=1
        )

        num_new_points = new_xyz.shape[0]
        if num_new_points == 0:
            return

        self.current_iter = current_iter

        # Step 5: 创建新点的属性（初始化为单位/零）
        new_scaling = torch.ones((num_new_points, 3), device="cuda")
        new_rotation = torch.zeros((num_new_points, 4), device="cuda")
        new_rotation[:, 0] = 1
        new_features_dc = torch.zeros((num_new_points, 1, 3), device="cuda")
        new_features_rest = torch.zeros((num_new_points, 15, 3), device="cuda")
        new_opacity = torch.ones((num_new_points, 1), device="cuda")

        # Step 6: 插入新点
        self.densification_postfix(
            new_xyz, new_features_dc, new_features_rest,
            new_opacity, new_scaling, new_rotation
        )

        # Step 7: 更新 point_types，保证和 _xyz 对齐
        if not hasattr(self, 'point_types'):
            # 初始化（老点全设为低梯度=0）
            self.point_types = torch.zeros(self.get_xyz.shape[0] - num_new_points,
                                           dtype=torch.long, device="cuda")
        self.point_types = torch.cat([self.point_types, new_point_types])

        # Step 8: 设置白名单，保护新点不被立即 Drop
        full_mask = torch.ones(self.get_xyz.shape[0], dtype=torch.bool, device=self.get_xyz.device)
        full_mask[-num_new_points:] = False
        self.new_point_mask = full_mask


    def densify_and_split1(
            self, grads, grad_threshold, scene_extent, iter,
            N=2,
            image_dir="/media/junz/4TB-11/liyangeng/papers/DropGaussian/dataset/nerf_llff_data/leaves/3_views/images",
            cameras_path="/media/junz/4TB-11/liyangeng/papers/DropGaussian/dataset/nerf_llff_data/leaves/3_views/triangulated/cameras.bin",
            images_path="/media/junz/4TB-11/liyangeng/papers/DropGaussian/dataset/nerf_llff_data/leaves/3_views/triangulated/images.bin"
    ):
        n_init_points = self.get_xyz.shape[0]

        # # === 初始化新点大小存储列表（只初始化一次）===
        # if not hasattr(self, "new_points_size_history"):
        #     # 存储格式：[(迭代次数, 新点大小数组), ...]
        #     self.new_points_size_history = []
        #     # 控制记录频率：可以设置为每n轮记录一次，这里默认每次都记录（后续统计时可以筛选）
        #     self.record_frequency = 1  # 1=每次都记录，10=每10轮记录一次

        # === 设置初始的分裂掩码 ===
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = padded_grad >= grad_threshold

        # 增加约束：point size也要足够大
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent
        )

        # SFS的第二阶段选点方式
        dist, _ = distCUDA2(self.get_xyz)
        selected_pts_mask2 = torch.logical_and(
            dist > (self.args.dist_thres * scene_extent),
            torch.max(self.get_scaling, dim=1).values > scene_extent
        )
        selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask2)

        # === 获取被选中的高斯属性 ===
        selected_xyz = self.get_xyz[selected_pts_mask]
        selected_scaling = self.get_scaling[selected_pts_mask]
        selected_rotation = self._rotation[selected_pts_mask]
        selected_dc = self._features_dc[selected_pts_mask]
        selected_rest = self._features_rest[selected_pts_mask]
        selected_opacity = self._opacity[selected_pts_mask]

        # === 计算 detail_weight（图像梯度）===
        if image_dir and cameras_path and images_path and selected_xyz.shape[0] > 0:
            gradient_maps = self.compute_gradient_maps(image_dir)
            intrinsics, extrinsics = self.load_camera_params(cameras_path, images_path)

            source_xyz = selected_xyz.repeat(1, N, 1).reshape(-1, 3)
            pixel_coords = self.generate_pixel_coords_per_image(source_xyz, intrinsics, extrinsics)
            detail_weight = self.get_detail_weight(source_xyz, pixel_coords, gradient_maps)
        else:
            detail_weight = None

        # === 基于 detail_weight 控制分裂尺度 ===
        if detail_weight is not None:
            selected_detail = detail_weight
            scaling_factor = 0.6 + (1.0 - selected_detail).unsqueeze(1) * 0.4  # [0.6, 1.0]
        else:
            scaling_factor = 1.0

        # === 分裂新点 ===
        stds = selected_scaling.repeat(N, 1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)

        rots = build_rotation(selected_rotation).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + selected_xyz.repeat(N, 1)

        new_scaling = self.scaling_inverse_activation(
            selected_scaling.repeat(N, 1) / (0.8 * N * scaling_factor )
        )

        new_rotation = selected_rotation.repeat(N, 1)
        new_features_dc = selected_dc.repeat(N, 1, 1)
        new_features_rest = selected_rest.repeat(N, 1, 1)
        new_opacity = selected_opacity.repeat(N, 1)

        self.densification_postfix(
            new_xyz, new_features_dc, new_features_rest,
            new_opacity, new_scaling, new_rotation
        )

        # # === 记录新点大小（控制记录频率）===
        # if new_xyz.shape[0] > 0 and (iter % self.record_frequency == 0):
        #     # 计算每个新点的大小：使用scaling的最大值（也可以用L2范数或平均值）
        #     new_points_size = torch.max(new_scaling, dim=1).values  # [num_new_points]
        #     # 转移到CPU并转为numpy数组，避免占用GPU内存
        #     new_points_size_np = new_points_size.cpu().detach().numpy()
        #     # 保存迭代次数和对应的新点大小
        #     self.new_points_size_history.append((iter, new_points_size_np))

        # === 同步更新 point_types ===
        if not hasattr(self, "point_types"):
            # 初始化所有老点为低梯度(0)
            self.point_types = torch.zeros(self.get_xyz.shape[0] - new_xyz.shape[0],
                                           dtype=torch.long, device="cuda")

        # 父点类型
        parent_point_types = self.point_types[selected_pts_mask]
        # 新点继承父点类型（每个父点分裂 N 个）
        new_point_types = parent_point_types.repeat_interleave(N, dim=0)

        # 拼接 point_types
        self.point_types = torch.cat([self.point_types, new_point_types], dim=0)

        prune_filter = torch.cat((
            selected_pts_mask,
            torch.zeros(new_xyz.shape[0], device="cuda", dtype=torch.bool)
        ))

        self.prune_points(prune_filter, iter)

        # # === 第9999轮时统计并绘制直方图 ===
        # if iter == 9000:
        #     self.plot_new_points_size_histogram()

    def plot_new_points_size_histogram(self):
        """Plot histogram of the size distribution of all recorded new points"""
        if not hasattr(self, "new_points_size_history") or len(self.new_points_size_history) == 0:
            print("No recorded new point size data, cannot plot histogram")
            return

        # Collect all recorded new point sizes
        all_sizes = []
        all_iters = []
        for iter_num, sizes in self.new_points_size_history:
            all_sizes.extend(sizes.tolist())
            all_iters.append(iter_num)

        all_sizes = np.array(all_sizes)
        print(
            f"\n=== New Point Size Distribution Statistics (Recorded {len(self.new_points_size_history)} splitting rounds, {len(all_sizes)} new points) ===")
        print(f"Size Range: [{all_sizes.min():.6f}, {all_sizes.max():.6f}]")
        print(f"Mean: {all_sizes.mean():.6f}")
        print(f"Median: {np.median(all_sizes):.6f}")
        print(f"Standard Deviation: {all_sizes.std():.6f}")

        # Plot histogram
        plt.figure(figsize=(12, 6))

        # Main histogram
        plt.subplot(1, 2, 1)
        n, bins, patches = plt.hist(all_sizes, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        plt.axvline(all_sizes.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {all_sizes.mean():.6f}')
        plt.axvline(np.median(all_sizes), color='orange', linestyle='--', linewidth=2,
                    label=f'Median: {np.median(all_sizes):.6f}')
        plt.xlabel('New Point Size (Max Scaling Value)')
        plt.ylabel('Number of Points')
        plt.title('Size Distribution Histogram of All Split New Points')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Per-iteration statistics (optional)
        plt.subplot(1, 2, 2)
        # Calculate per-iteration new point size statistics
        iter_means = []
        iter_stds = []
        for iter_num, sizes in self.new_points_size_history:
            iter_means.append(np.mean(sizes))
            iter_stds.append(np.std(sizes))

        iter_means = np.array(iter_means)
        iter_stds = np.array(iter_stds)
        plt.errorbar(all_iters, iter_means, yerr=iter_stds, fmt='o-', color='darkred',
                     capsize=3, markersize=4, linewidth=1)
        plt.xlabel('Iteration Number')
        plt.ylabel('New Point Size Mean ± Std')
        plt.title('Statistical Variation of New Point Sizes Across Splitting Rounds')
        plt.grid(True, alpha=0.3)

        # Adjust layout and save/display
        plt.tight_layout()

        # Save image (recommended)
        save_path = "new_points_size_histogram.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nHistogram saved to: {save_path}")

        # Uncomment below to display the plot if running locally
        # plt.show()

        # Optional: Save raw data to npz file for further analysis
        np.savez("new_points_size_data.npz",
                 all_sizes=all_sizes,
                 iter_numbers=np.array(all_iters),
                 per_iter_sizes=[sizes for _, sizes in self.new_points_size_history])
        print("Raw data saved to: new_points_size_data.npz")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values > self.percent_dense * scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        # === 同步更新 point_types ===
        if not hasattr(self, "point_types"):
            # 初始化所有老点为低梯度(0)
            self.point_types = torch.zeros(self.get_xyz.shape[0] - new_xyz.shape[0],
                                           dtype=torch.long, device="cuda")

        # 父点类型
        parent_point_types = self.point_types[selected_pts_mask]
        # 新点继承父点类型（每个父点分裂 N 个）
        new_point_types = parent_point_types.repeat_interleave(N, dim=0)

        # 拼接 point_types
        self.point_types = torch.cat([self.point_types, new_point_types], dim=0)

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter, iter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,  # self.percent_dense = 0.01
                                              torch.max(self.get_scaling,
                                                        dim=1).values <= self.percent_dense * scene_extent)

        # Step 1: 初始化/校准 point_types（确保与当前高斯点数量匹配，老点默认低梯度=0）
        if not hasattr(self, "point_types") or self.point_types.numel() != self._xyz.shape[0]:
            self.point_types = torch.zeros(self._xyz.shape[0], dtype=torch.long, device="cuda")

        # Step 2: 获取父点类型（被选中克隆的点的类型）
        parent_point_types = self.point_types[selected_pts_mask]

        # Step 3: 生成新点类型（克隆点完全继承父点类型，1:1克隆无需repeat）
        new_point_types = parent_point_types.clone()  # 克隆点与父点类型完全一致

        # Step 4: 提取属性并生成新点（原有逻辑不变）
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        # Step 5: 添加新点到模型（原有逻辑不变）
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                                   new_rotation)

        # ==================================
        # 补全部分：同步更新全局 point_types
        # ==================================
        # 拼接原有point_types和新克隆点的类型（与新点的xyz顺序对齐）
        self.point_types = torch.cat([self.point_types, new_point_types], dim=0)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, iter):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        if iter <= 2000:
            self.densify_and_clone(grads, max_grad, extent)
            self.densify_and_split1(grads, max_grad, extent, iter)
            self.proximity(self, extent, iter, N=3)
        else:
            self.densify_and_clone(grads, max_grad, extent)
            self.densify_and_split1(grads, max_grad, extent, iter)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        self.prune_points(prune_mask, iter)
        torch.cuda.empty_cache()


    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1,
                                                             keepdim=True)
        self.denom[update_filter] += 1