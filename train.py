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
import torchvision
from random import randint
from utils.loss_utils import l1_loss, ssim
from utils.depth_utils import estimate_depth
from gaussian_renderer import render, network_gui
from torchmetrics.functional.regression import pearson_corrcoef
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from torchvision.utils import save_image
from torch import nn
import copy
import matplotlib.pyplot as plt
import json
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def get_current_psnr(rendered_image, gt_image):
    """
    计算渲染图像与真实图像之间的PSNR
    """
    # 确保图像在相同设备和数据类型
    rendered_image = rendered_image.to(gt_image.device).to(gt_image.dtype)

    # 确保图像值范围在[0, 1]
    rendered_image = torch.clamp(rendered_image, 0, 1)
    gt_image = torch.clamp(gt_image, 0, 1)

    # 计算MSE
    mse = torch.mean((rendered_image - gt_image) ** 2)

    # 避免除零错误
    if mse < 1e-10:
        return float('inf')

    # 计算PSNR
    max_pixel = 1.0
    psnr_value = 20 * torch.log10(max_pixel / torch.sqrt(mse))

    return psnr_value.item()

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree,args)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    test_imgs_dir = os.path.join(args.model_path, "test_imgs/")
    os.makedirs(test_imgs_dir, exist_ok = True)

    # === 核心修改：从测试相机中选固定视角（而非训练相机）===
    psnr_records = []  # 存储格式：[{iteration, split_count, gaussian_num, psnr}, ...]
    split_count = 0  # 分裂操作计数器（仅统计有效分裂）
    fixed_test_cam = None  # 固定测试视角的相机
    fixed_test_gt = None  # 测试视角的真实图（GT）

    # # 1. 获取测试相机集合（scene.getTestCameras() 返回测试视角列表）
    # test_cameras = scene.getTestCameras()
    # if test_cameras:
    #     # 选第0个测试相机作为固定评估视角（可修改索引，如test_cameras[1]）
    #     fixed_test_cam = test_cameras[0]
    #     # 获取测试相机的真实图（GT），确保格式正确（根据你的Scene类实现调整）
    #     # 若test_camera.original_image不存在，可能需要用dataset加载测试图，示例：
    #     # fixed_test_gt = dataset.load_test_image(fixed_test_cam.image_path).cuda()
    #     fixed_test_gt = fixed_test_cam.original_image.cuda()  # 假设测试相机有original_image属性
    #     print(f"Using fixed test camera (index 0) for PSNR evaluation")
    # else:
    #     # 若没有测试相机，提示并跳过PSNR记录
    #     print("Warning: No test cameras found! PSNR recording is skipped.")


    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    bg_mask = None
    loss_accum = 0
    pseudo_stack = None
    for iteration in range(first_iter, opt.iterations + 1):        
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()


        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        gt_image = viewpoint_cam.original_image.cuda()

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, is_train=True, iteration=iteration)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
        render_pkg["visibility_filter"], render_pkg["radii"]

        Ll1 = l1_loss(image, gt_image)
        ssim_value = ssim(image, gt_image)
        loss = Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        if iteration % args.sample_pseudo_interval == 0 and iteration > args.start_sample_pseudo and iteration < args.end_sample_pseudo:
            if not pseudo_stack:
                pseudo_stack = scene.getPseudoCameras().copy()
            pseudo_cam = pseudo_stack.pop(randint(0, len(pseudo_stack) - 1))

            render_pkg_pseudo = render(pseudo_cam, gaussians, pipe, background)
            rendered_depth_pseudo = render_pkg_pseudo["depth"][0]
            midas_depth_pseudo = estimate_depth(render_pkg_pseudo["render"], mode='train')

            rendered_depth_pseudo = rendered_depth_pseudo.reshape(-1, 1)
            midas_depth_pseudo = midas_depth_pseudo.reshape(-1, 1)
            depth_loss_pseudo = (1 - pearson_corrcoef(rendered_depth_pseudo, -midas_depth_pseudo)).mean()

            if torch.isnan(depth_loss_pseudo).sum() == 0:
                loss_scale = min((iteration - args.start_sample_pseudo) / 500., 1)
                loss += loss_scale * args.depth_pseudo_weight * depth_loss_pseudo


        loss.backward()
        iter_end.record()

        with torch.no_grad():

            # Progress bar
            if iteration > opt.densify_from_iter:
                loss_accum += loss.clone().detach().item()

            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(dataset, tb_writer, iteration, loss, l1_loss, iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                     radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = None

                    # # === 新增：记录分裂前的高斯数量（判断是否为有效分裂）===
                    # gaussian_num_before = gaussians.get_xyz.shape[0]


                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold,
                                                iteration)

                    # # === 新增：分裂后计算并记录PSNR ===
                    # gaussian_num_after = gaussians.get_xyz.shape[0]
                    # # === 关键：用测试视角渲染并计算PSNR ===
                    # if gaussian_num_after > gaussian_num_before and fixed_test_cam is not None and fixed_test_gt is not None:
                    #     split_count += 1
                    #     # 用固定测试相机渲染（is_train=False，关闭训练时的随机扰动）
                    #     render_test_pkg = render(
                    #         fixed_test_cam,  # 测试相机（而非训练相机）
                    #         gaussians,
                    #         pipe,
                    #         background,  # 固定背景（非随机）
                    #         is_train=False,
                    #         iteration=iteration
                    #     )
                    #     rendered_test_img = render_test_pkg["render"]
                    #     # 计算测试视角的PSNR
                    #     psnr = get_current_psnr(rendered_test_img, fixed_test_gt)
                    #     # 记录数据（新增测试视角标记，可选）
                    #     psnr_records.append({
                    #         "iteration": iteration,
                    #         "split_count": split_count,
                    #         "gaussian_num": gaussian_num_after,
                    #         "psnr": psnr,
                    #         "view_type": "test"  # 标记是测试视角
                    #     })

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


    # # === 新增：训练结束后保存PSNR记录 ===
    # psnr_save_path = os.path.join(args.model_path, "psnr_split_records.json")
    # with open(psnr_save_path, "w") as f:
    #     json.dump(psnr_records, f, indent=2)
    # print(f"\nPSNR records saved to: {psnr_save_path}")



def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(args, tb_writer, iteration, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        # tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(len(scene.getTrainCameras()))]})

        for config in validation_configs:
            render_path = os.path.join(args.model_path, config['name'], "ours_{}".format(iteration), "renders")
            gts_path = os.path.join(args.model_path, config['name'], "ours_{}".format(iteration), "gt")
            os.makedirs(render_path, exist_ok=True)
            os.makedirs(gts_path, exist_ok=True)
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = render_pkg["render"]
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    torchvision.utils.save_image(image, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
                    torchvision.utils.save_image(gt_image, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5000, 10000,])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
