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

       
        self.current_iter = 0

        self.progressive_exposure = {
            "indices": torch.tensor([], dtype=torch.long, device="cuda"),
            "step": torch.tensor([], dtype=torch.float32, device="cuda")
        }


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

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]


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
        
        prune_mask = mask
        
        if hasattr(self, "new_point_mask"):
            prune_mask = torch.logical_and(prune_mask, self.new_point_mask)
            del self.new_point_mask  

        valid_points_mask = ~prune_mask  
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


    def compute_gradient_maps(self, image_dir, image_size=(640, 480)):
        gradient_maps = []
        filenames = sorted(os.listdir(image_dir))
        for fname in filenames:
            img = cv2.imread(os.path.join(image_dir, fname), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, image_size)
            scharrx = cv2.Scharr(img, cv2.CV_32F, 1, 0)
            scharry = cv2.Scharr(img, cv2.CV_32F, 0, 1)
            grad_mag = np.sqrt(scharrx ** 2 + scharry ** 2)

            gradient_maps.append(torch.from_numpy(grad_mag))

        return torch.stack(gradient_maps).to("cuda").float()  # [24, H, W]

    def get_detail_weight(self, xyz, projected_coords, gradient_maps):
        if xyz.numel() == 0:
            return torch.zeros((0,), device=xyz.device)

        gradients = []
        num_cameras, H, W = gradient_maps.shape

        for i in range(num_cameras):
            
            coords = projected_coords[:, i]  
            x = coords[:, 0]  
            y = coords[:, 1]  
            x0 = torch.floor(x).long()  
            x1 = x0 + 1  
            y0 = torch.floor(y).long()  
            y1 = y0 + 1 
            x0 = torch.clamp(x0, 0, W - 1)
            x1 = torch.clamp(x1, 0, W - 1)
            y0 = torch.clamp(y0, 0, H - 1)
            y1 = torch.clamp(y1, 0, H - 1)
            wx = x - x0.float() 
            wy = y - y0.float() 
            grad00 = gradient_maps[i][y0, x0] 
            grad01 = gradient_maps[i][y0, x1]  
            grad10 = gradient_maps[i][y1, x0]  
            grad11 = gradient_maps[i][y1, x1] 

            grad_x0 = grad00 * (1 - wx) + grad01 * wx  
            grad_x1 = grad10 * (1 - wx) + grad11 * wx 
  
            grad_interp = grad_x0 * (1 - wy) + grad_x1 * wy  

            gradients.append(grad_interp)

        mean_grad = torch.stack(gradients, dim=1).mean(dim=1)

        return (mean_grad - mean_grad.min()) / (mean_grad.max() - mean_grad.min() + 1e-6)

    @staticmethod
    def load_camera_params(cameras_path, images_path):

        def read_next(f, fmt):
            return struct.unpack(fmt, f.read(struct.calcsize(fmt)))

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

                num_points2D = read_next(f, "<Q")[0]
                f.read(24 * num_points2D)

                fx, fy, cx, cy = cameras[camera_id]['params']
                intrinsics = np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ], dtype=np.float32)
                intrinsics_list.append(intrinsics)

                R = Rotation.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]]).as_matrix()
                extrinsic = np.hstack((R, tvec.reshape(3, 1))).astype(np.float32)
                extrinsics_list.append(extrinsic)

        return intrinsics_list, extrinsics_list

    def project_points_to_image(self, points_3d, camera_intrinsics, camera_extrinsics):

        ones = torch.ones((points_3d.shape[0], 1), device=points_3d.device)
        points_3d_homogeneous = torch.cat((points_3d, ones), dim=1)  # [N, 4]
        points_2d_homogeneous = torch.matmul(points_3d_homogeneous, camera_extrinsics.T)  # [N, 4]
        points_2d_homogeneous = torch.matmul(points_2d_homogeneous, camera_intrinsics.T)  # [N, 3]
        points_2d = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2:3]  # [N, 2]
        return points_2d

    def generate_pixel_coords_per_image(self, points_3d, camera_intrinsics_list, camera_extrinsics_list):
    
        pixel_coords_per_image = []
        for i in range(len(camera_intrinsics_list)):
            intrinsics = torch.tensor(camera_intrinsics_list[i], device=points_3d.device)
            extrinsics = torch.tensor(camera_extrinsics_list[i], device=points_3d.device)
            pixel_coords = self.project_points_to_image(points_3d, intrinsics, extrinsics)  
            pixel_coords_per_image.append(pixel_coords)  

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

            line_t_temp = torch.linspace(0, 1, steps=num_line_samples, device=p1.device).unsqueeze(1)
            sample_pts_temp = src_point + line_t_temp * dir_vec
            pixel_coords_temp = self.generate_pixel_coords_per_image(sample_pts_temp, intrinsics,
                                                                     extrinsics)  
            line_weights_temp = self.get_detail_weight(sample_pts_temp, pixel_coords_temp, gradient_maps) 

            if torch.max(line_weights_temp) <= 0.0:
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
                eps = 1e-8 

                num_intervals = 15  
                total_sample_points = 10  
                interval_step = 1.0 / num_intervals

                interval_centers_t = torch.linspace(
                    interval_step / 2, 1 - interval_step / 2,
                    steps=num_intervals, device=p1.device
                )
                center_3d_pts = src_point + interval_centers_t.unsqueeze(1) * dir_vec

                center_pixel_coords = self.generate_pixel_coords_per_image(center_3d_pts, intrinsics, extrinsics)
                interval_weights = self.get_detail_weight(center_3d_pts, center_pixel_coords, gradient_maps)
                weights_t = torch.cat([
                    torch.tensor([0.0], device=p1.device), 
                    interval_centers_t,
                    torch.tensor([1.0], device=p1.device)  
                ], dim=0)

                weights = torch.cat([
                    torch.tensor([0.0], device=p1.device), 
                    interval_weights,
                    torch.tensor([0.0], device=p1.device)  
                ], dim=0)

                w = weights + 1e-6  
                w = w / w.sum()  
                cdf = torch.cumsum(w, dim=0)
                t_samples = torch.linspace(0, 1, steps=total_sample_points, device=p1.device)  # [10,]

                base_grid = weights_t 
                inds = torch.searchsorted(cdf, t_samples, right=True).clamp(max=len(base_grid) - 1)
                inds0 = (inds - 1).clamp(min=0)  
                inds1 = inds  

              
                cdf0, cdf1 = cdf[inds0], cdf[inds1] 
                grid0, grid1 = base_grid[inds0], base_grid[inds1]  
                denom = (cdf1 - cdf0 + 1e-8)  
                t_vals = grid0 + (t_samples - cdf0) / denom * (grid1 - grid0) 

                
                t_vals, _ = torch.sort(t_vals)

                
                new_pts = src_point.unsqueeze(0) + t_vals.unsqueeze(1) * dir_vec.unsqueeze(0)  # [10, 3]
                point_types = torch.full((total_sample_points,), 1, device=p1.device) 

            new_points_list.append(new_pts)
            point_types_list.append(point_types)

        new_points = torch.cat(new_points_list, dim=0)
        point_types = torch.cat(point_types_list, dim=0)

        return new_points, point_types

    def proximity(self, model: 'GaussianModel', scene_extent, current_iter, N=3):
        
        image_dir = ""
        
        gradient_maps = self.compute_gradient_maps(image_dir)

       
        cameras_path = ""
        images_path = ""
        intrinsics, extrinsics = self.load_camera_params(cameras_path, images_path)

        
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

        pixel_coords_per_image = self.generate_pixel_coords_per_image(source_xyz, intrinsics, extrinsics)
        detail_weight = self.get_detail_weight(source_xyz, pixel_coords_per_image, gradient_maps)  # [B]

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

        new_scaling = torch.ones((num_new_points, 3), device="cuda")
        new_rotation = torch.zeros((num_new_points, 4), device="cuda")
        new_rotation[:, 0] = 1
        new_features_dc = torch.zeros((num_new_points, 1, 3), device="cuda")
        new_features_rest = torch.zeros((num_new_points, 15, 3), device="cuda")
        new_opacity = torch.ones((num_new_points, 1), device="cuda")

        self.densification_postfix(
            new_xyz, new_features_dc, new_features_rest,
            new_opacity, new_scaling, new_rotation
        )

        if not hasattr(self, 'point_types'):
            self.point_types = torch.zeros(self.get_xyz.shape[0] - num_new_points,
                                           dtype=torch.long, device="cuda")
        self.point_types = torch.cat([self.point_types, new_point_types])

        full_mask = torch.ones(self.get_xyz.shape[0], dtype=torch.bool, device=self.get_xyz.device)
        full_mask[-num_new_points:] = False
        self.new_point_mask = full_mask


    def densify_and_split(
            self, grads, grad_threshold, scene_extent, iter,
            N=2,
            image_dir="",
            cameras_path="",
            images_path=""
    ):
        n_init_points = self.get_xyz.shape[0]

        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = padded_grad >= grad_threshold

        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent
        )

        dist, _ = distCUDA2(self.get_xyz)
        selected_pts_mask2 = torch.logical_and(
            dist > (self.args.dist_thres * scene_extent),
            torch.max(self.get_scaling, dim=1).values > scene_extent
        )
        selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask2)

        selected_xyz = self.get_xyz[selected_pts_mask]
        selected_scaling = self.get_scaling[selected_pts_mask]
        selected_rotation = self._rotation[selected_pts_mask]
        selected_dc = self._features_dc[selected_pts_mask]
        selected_rest = self._features_rest[selected_pts_mask]
        selected_opacity = self._opacity[selected_pts_mask]

        if image_dir and cameras_path and images_path and selected_xyz.shape[0] > 0:
            gradient_maps = self.compute_gradient_maps(image_dir)
            intrinsics, extrinsics = self.load_camera_params(cameras_path, images_path)

            source_xyz = selected_xyz.repeat(1, N, 1).reshape(-1, 3)
            pixel_coords = self.generate_pixel_coords_per_image(source_xyz, intrinsics, extrinsics)
            detail_weight = self.get_detail_weight(source_xyz, pixel_coords, gradient_maps)
        else:
            detail_weight = None

        if detail_weight is not None:
            selected_detail = detail_weight
            scaling_factor = 0.6 + (1.0 - selected_detail).unsqueeze(1) * 0.4  # [0.6, 1.0]
        else:
            scaling_factor = 1.0

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

        if not hasattr(self, "point_types"):
            self.point_types = torch.zeros(self.get_xyz.shape[0] - new_xyz.shape[0],
                                           dtype=torch.long, device="cuda")

        parent_point_types = self.point_types[selected_pts_mask]
        new_point_types = parent_point_types.repeat_interleave(N, dim=0)
        self.point_types = torch.cat([self.point_types, new_point_types], dim=0)

        prune_filter = torch.cat((
            selected_pts_mask,
            torch.zeros(new_xyz.shape[0], device="cuda", dtype=torch.bool)
        ))

        self.prune_points(prune_filter, iter)

    

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,  # self.percent_dense = 0.01
                                              torch.max(self.get_scaling,
                                                        dim=1).values <= self.percent_dense * scene_extent)

        if not hasattr(self, "point_types") or self.point_types.numel() != self._xyz.shape[0]:
            self.point_types = torch.zeros(self._xyz.shape[0], dtype=torch.long, device="cuda")

        parent_point_types = self.point_types[selected_pts_mask]

        new_point_types = parent_point_types.clone()  
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                                   new_rotation)

        self.point_types = torch.cat([self.point_types, new_point_types], dim=0)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, iter):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        if iter <= 2000:
            self.densify_and_clone(grads, max_grad, extent)
            self.densify_and_split(grads, max_grad, extent, iter)
            self.proximity(self, extent, iter, N=3)
        else:
            self.densify_and_clone(grads, max_grad, extent)
            self.densify_and_split(grads, max_grad, extent, iter)

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