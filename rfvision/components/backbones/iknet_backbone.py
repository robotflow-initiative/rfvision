import torch.nn as nn
import torch
import torch.nn.functional as F
from rflib.runner import BaseModule
from rfvision.components.utils import (normalize_quaternion, quaternion_to_angle_axis,
                                       quaternion_mul, quaternion_inv)
from rfvision.models.builder import BACKBONES


@BACKBONES.register_module()
class IKNetBackbone(BaseModule):
    def __init__(
        self,
        num_joints=21, # 21 joints
        hidden_channels=[256, 512, 1024, 1024, 512, 256],
        out_channels=16 * 4,   # 16 quats
        init_cfg=None
    ):

        super().__init__(init_cfg)
        self.num_joints = num_joints
        self.in_channels = self.num_joints * 3
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        self.channels = [self.in_channels] + hidden_channels

        layers = []
        for i, j in zip(self.channels[:-1], self.channels[1:]):
            layers.append(nn.Linear(i, j))
            layers.append(nn.BatchNorm1d(j))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.channels[-1], self.out_channels))
        self.layers = nn.Sequential(*layers)

    def forward(self, joints_xyz):
        joints_xyz = joints_xyz.contiguous().view(-1, self.num_joints * 3)
        quat = self.layers(joints_xyz)
        quat = quat.view(-1, 16, 4)
        quat = normalize_quaternion(quat)
        so3 = quaternion_to_angle_axis(quat).contiguous()
        so3 = so3.view(-1, 16 * 3)
        return so3, quat

    @staticmethod
    def loss_ik(pred_quat, gt_quat):
        loss_quat_l2 = F.mse_loss(pred_quat, gt_quat)
        pred_quat_inv = quaternion_inv(pred_quat)
        real_part = quaternion_mul(gt_quat, pred_quat_inv)
        real_part = real_part[..., -1]

        loss_quat_cos = F.l1_loss(real_part, torch.ones_like(real_part))
        losses_ik = {'loss_quat_l2': loss_quat_l2,
                     'loss_quat_cos': loss_quat_cos}
        return losses_ik

