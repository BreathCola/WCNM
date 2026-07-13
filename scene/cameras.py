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
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.general_utils import PILtoTorch
import cv2

class Camera(nn.Module):
    def __init__(self, resolution, colmap_id, R, T, FoVx, FoVy, depth_params, image, invdepthmap,
                 image_name, uid, normal_prior=None, normal_prior_valid=None, normal_prior_space="camera",
                 specular_mask=None, specular_mask_sha256=None,
                 internal_object_masks=None,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 train_test_exp = False, is_test_dataset = False, is_test_view = False
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.normal_prior_space = normal_prior_space

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        # 将图像转换为torch张量
        resized_image_rgb = PILtoTorch(image, resolution)
        # 提取张量（C,H,W）中C的前三个通道，即RGB通道（纯粹的RGB彩色图像）；并保留H,W；
        gt_image = resized_image_rgb[:3, ...]
        # 初始设置蒙板为None
        self.alpha_mask = None
        # 若图像张量的通道数为4，则提取第四通道作为蒙板（alpha通道）；
        # 否则，创建一个与RGB通道相同大小的全1张量作为蒙板（全不透明）；
        if resized_image_rgb.shape[0] == 4:
            self.alpha_mask = resized_image_rgb[3:4, ...].to(self.data_device)
        else: 
            self.alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...].to(self.data_device))

        # 若训练测试实验且当前视图为测试视图，则根据数据集类型对蒙板进行裁剪：
        # 若为测试数据集，则将蒙板的左半部分设为0（完全透明）；
        # 若为训练数据集，则将蒙板的右半部分设为0（完全透明）；
        if train_test_exp and is_test_view:
            if is_test_dataset:
                self.alpha_mask[..., :self.alpha_mask.shape[-1] // 2] = 0
            else:
                self.alpha_mask[..., self.alpha_mask.shape[-1] // 2:] = 0

        # 确保图像像素值在合理范围（0-1），并将数据迁移到目标计算设备（如GPU）
        self.original_image = gt_image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
        self.normal_prior = None if normal_prior is None else normal_prior.to(self.data_device)
        self.normal_prior_valid = None if normal_prior_valid is None else normal_prior_valid.to(self.data_device)
        self.specular_mask = None if specular_mask is None else specular_mask.to(self.data_device)
        self.specular_mask_sha256 = specular_mask_sha256
        self.internal_object_masks = None
        if internal_object_masks is not None:
            self.internal_object_masks = {
                key: (value.to(self.data_device) if torch.is_tensor(value) else value)
                for key, value in internal_object_masks.items()
            }

        self.invdepthmap = None
        self.depth_reliable = False
        if invdepthmap is not None:
            self.depth_mask = torch.ones_like(self.alpha_mask)
            self.invdepthmap = cv2.resize(invdepthmap, resolution)
            self.invdepthmap[self.invdepthmap < 0] = 0
            self.depth_reliable = True

            if depth_params is not None:
                if depth_params["scale"] < 0.2 * depth_params["med_scale"] or depth_params["scale"] > 5 * depth_params["med_scale"]:
                    self.depth_reliable = False
                    self.depth_mask *= 0
                
                if depth_params["scale"] > 0:
                    self.invdepthmap = self.invdepthmap * depth_params["scale"] + depth_params["offset"]

            if self.invdepthmap.ndim != 2:
                self.invdepthmap = self.invdepthmap[..., 0]
            self.invdepthmap = torch.from_numpy(self.invdepthmap[None]).to(self.data_device)

        # 设置相机的近裁剪平面（znear）和远裁剪平面（zfar）
        self.zfar = 100.0
        self.znear = 0.01

        # 设置相机的转换矩阵（trans）和缩放因子（scale）
        self.trans = trans
        self.scale = scale

        # 设置相机的世界视图变换矩阵（world_view_transform）和投影矩阵（projection_matrix）
        # 这个矩阵负责将 3D 世界坐标系中的点，转换到以相机为中心的坐标系中。
        # 它包含了相机的旋转矩阵（R）、平移向量（T）、转换矩阵（trans）和缩放因子（scale）。
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        # 这个矩阵负责将相机坐标系下的 3D 点，投影到一个 2D 的平面上，并产生透视效果（近大远小）。
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        # 计算相机中心（camera_center），即相机在世界坐标系中的位置。
        # 这是通过取世界视图变换矩阵的逆矩阵的第4行（索引为3）的前3个元素（即平移向量的负方向）得到的。
        # 计算并存储相机在世界坐标系中的确切位置 (x, y, z)。这是通过对视图矩阵求逆得到的
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
