import torch.nn as nn
import torch
import torch.nn.functional as F
import logging
from torch.nn.modules.batchnorm import _BatchNorm
from rflib.cnn import kaiming_init, constant_init
from rflib.runner import load_checkpoint
from rfvision.components.utils.handtailor_utils import (normalize_quaternion, quaternion_to_angle_axis,
                                                               quaternion_mul, quaternion_inv)
from rfvision.models.builder import BACKBONES


@BACKBONES.register_module()
class IKNetBackbone(nn.Module):
    def __init__(
        self,
        num_joints=21, # 21 joints
        hidden_channels=[256, 512, 1024, 1024, 512, 256],
        out_channels=16 * 4,   # 16 quats
    ):

        super().__init__()
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

    def init_weights(self, init_cfg=None):
        if isinstance(init_cfg, str):
            logger = logging.getLogger()
            load_checkpoint(self, init_cfg, strict=False, logger=logger)
        elif init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

        else:
            raise TypeError('init_cfg must be a str or None')

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

