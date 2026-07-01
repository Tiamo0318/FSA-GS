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
from torch import nn
import numpy as np
from math import tan  # 关键导入
from utils.graphics_utils import getWorld2View2, getProjectionMatrix


def extract_intrinsics_extrinsics_from_cameras(camera_list):
    """
    从3DGS的Camera实例列表，提取内参列表和外参列表（适配之前的代码）
    :param camera_list: list[Camera/MiniCam/PseudoCamera]，3DGS的相机实例列表
    :return: intrinsics_list (list[np.ndarray(3,3)]), extrinsics_list (list[np.ndarray(3,4)])
    """
    intrinsics_list = []
    extrinsics_list = []

    for cam in camera_list:
        # ---------------------- 1. 推导内参矩阵（fx, fy, cx, cy）----------------------
        # 3DGS默认主点（cx, cy）在图像中心，无需额外校准
        cx = cam.image_width / 2.0
        cy = cam.image_height / 2.0

        # 由FoVx推导fx：fx = (图像宽度/2) / tan(FoVx/2)（弧度制）
        fx = (cam.image_width / 2.0) / tan(cam.FoVx / 2.0)
        # 由FoVy推导fy：fy = (图像高度/2) / tan(FoVy/2)（弧度制）
        fy = (cam.image_height / 2.0) / tan(cam.FoVy / 2.0)

        # 构建内参矩阵（3x3）
        intrinsics = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        intrinsics_list.append(intrinsics)

        # ---------------------- 2. 提取外参矩阵（R:3x3, T:3x1 → 3x4）----------------------
        # 方式1：直接从Camera的R、T属性构建（最直接，前提是Camera保留了R、T的numpy数组）
        R = cam.R  # 3x3旋转矩阵（世界→相机）
        T = cam.T  # 3x1平移向量（世界→相机）
        extrinsic = np.hstack((R, T.reshape(3, 1))).astype(np.float32)

        # 方式2：从world_view_transform（W2C矩阵）提取（若R、T不可直接访问时用）
        # W2C = cam.world_view_transform.cpu().numpy().T  # 因为Camera中做了transpose(0,1)，需转回来
        # R = W2C[:3, :3]
        # T = W2C[:3, 3:4]
        # extrinsic = np.hstack((R, T)).astype(np.float32)

        extrinsics_list.append(extrinsic)

    return intrinsics_list, extrinsics_list

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid, bounds,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        self.zfar = 100.0
        self.znear = 0.01
        self.trans = trans
        self.scale = scale

        self.bounds=bounds

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()                                         #Extrinsic
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()        #NOT Intrinsic!
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)                     #E @ P
        self.camera_center = self.world_view_transform.inverse()[3, :3]


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

class PseudoCamera(nn.Module):
    def __init__(self, R, T, FoVx, FoVy, width, height, trans=np.array([0.0, 0.0, 0.0]), scale=1.0 ):
        super(PseudoCamera, self).__init__()

        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_width = width
        self.image_height = height

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
