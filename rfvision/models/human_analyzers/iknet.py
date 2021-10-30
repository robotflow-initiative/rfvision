from rfvision.models.builder import build_backbone, HUMAN_ANALYZERS
from rfvision.models import Base3DDetector

import torch.nn as nn
import torch
import torch.nn.functional as F
from rfvision.components.utils import (normalize_quaternion, quaternion_to_angle_axis,
                                       quaternion_mul, quaternion_inv)





# @HUMAN_ANALYZERS.register_module()
# class INKVNet(Base3DDetector):
#     def __init__(self,
#                  num_joints=21,
#                  in_channels=21 * 3,
#                  out_channels=16 * 3,
#                  train_cfg=None,
#                  test_cfg=None,
#                  init_cfg=None
#                  ):
#         super().__init__(init_cfg=init_cfg)
#         self.num_joints = num_joints
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.hidden_channels = [256, 512, 1024, 1024, 512, 256]
#         self.init_layers()
#
#     def init_layers(self):
#         self.in_channels = self.num_joints * 3
#         self.channels = [self.in_channels] + self.hidden_channels
#
#         layers = []
#         for i, j in zip(self.channels[:-1], self.channels[1:]):
#             layers.append(nn.Linear(i, j))
#             layers.append(nn.BatchNorm1d(j))
#             layers.append(nn.ReLU())
#         layers.append(nn.Linear(self.channels[-1], self.out_channels))
#         self.layers = nn.Sequential(*layers)
#
#     def forward_train(self, joints_xyz, full_poses):
#         joints_xyz = joints_xyz.contiguous().view(-1, self.num_joints * 3)
#         full_poses = full_poses.contiguous().view(-1, 16, 3)
#         pred_full_poses = self.layers(joints_xyz).view(-1, 16, 3) # (bz, 3 * 16)
#         loss_full_poses = F.mse_loss(pred_full_poses, full_poses)
#         losses = {'loss_so3': loss_full_poses}
#
#
#         # joints_xyz = joints_xyz.contiguous().view(-1, self.num_joints * 3)
#         # # quat_pred = self.layers(joints_xyz).view(-1, 16, 3) # (bz, 3 * 16)
#         # quat_pred = self.layers(joints_xyz) # (bz, 3 * 16)
#         # quat_pred = quat_pred.view(-1, 16, 4)
#         # quat_pred = normalize_quaternion(quat_pred)
#         # # theta = quaternion_to_angle_axis(quat_pred).contiguous()
#         # # theta = theta.view(-1, 16 * 3)
#         #
#         # # loss
#         # loss_quat_l2 = F.mse_loss(quat_pred, quat)
#         # pred_quat_inv = quaternion_inv(quat_pred)
#         # real_part = quaternion_mul(quat, pred_quat_inv)
#         # real_part = real_part[..., -1]
#         # loss_quat_cos = F.l1_loss(real_part, torch.ones_like(real_part))
#         #
#         # # loss_theta = F.mse_loss(theta, torch.zeros_like(theta))
#         #
#         # losses = {'loss_quat_l2': loss_quat_l2,
#         #           'loss_quat_cos': loss_quat_cos,
#         #           # 'loss_theta': loss_theta
#         #           }
#         return losses
#
#     def forward_test(self, joints_xyz):
#         joints_xyz = joints_xyz.contiguous().view(-1, self.num_joints * 3)
#         # If ValueError: Expected more than 1 value per channel when training, got input size [1, 63], model.eval() is needed!
#         full_poses = self.layers(joints_xyz)
#         return full_poses
#         # joints_xyz = joints_xyz.contiguous().view(-1, self.num_joints * 3)
#         # quat = self.layers(joints_xyz)
#         # quat = quat.view(-1, 16, 4)
#         # quat = normalize_quaternion(quat)
#         # theta = quaternion_to_angle_axis(quat).contiguous()
#         # theta = theta.view(-1, 16 * 3)
#
#         # return theta
#
#     def forward(self, return_loss=True, **kwargs):
#         joints_xyz = kwargs['joints_xyz']
#         # quat = kwargs['quat']
#         if return_loss == True:
#             full_poses = kwargs['full_poses']
#             losses = self.forward_train(joints_xyz, full_poses)
#             return losses
#         else:
#             theta = self.forward_test(joints_xyz)
#             return theta
#
#     def train_step(self, data, optimizer):
#         losses = self(**data)
#         loss, log_vars = self._parse_losses(losses)
#         outputs = dict(
#             loss=loss, log_vars=log_vars, num_samples=len(data['full_poses']))
#         return outputs
#
#     def val_step(self, data, optimizer):
#         return self.train_step(data, optimizer)
#
#     def aug_test(self,):
#         pass
#
#     def extract_feat(self,):
#         pass
#
#     def simple_test(self,):
#         pass


@HUMAN_ANALYZERS.register_module()
class INKVNet(Base3DDetector):
    def __init__(self,
                 num_joints=21,
                 in_channels=21 * 3,
                 out_channels=16 * 4,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None
                 ):
        super().__init__(init_cfg=init_cfg)
        self.num_joints = num_joints
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = [256, 512, 1024, 1024, 512, 256]
        self.init_layers()

    def init_layers(self):
        self.in_channels = self.num_joints * 3
        self.channels = [self.in_channels] + self.hidden_channels

        layers = []
        for i, j in zip(self.channels[:-1], self.channels[1:]):
            layers.append(nn.Linear(i, j))
            layers.append(nn.BatchNorm1d(j))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.channels[-1], self.out_channels))
        self.layers = nn.Sequential(*layers)

    def forward_train(self, joints_xyz, quat):
        # joints_xyz = joints_xyz.contiguous().view(-1, self.num_joints * 3)
        # pred_full_poses = self.layers(joints_xyz).view(-1, 16, 3) # (bz, 3 * 16)
        # loss_full_poses = F.mse_loss(pred_full_poses, full_poses)
        # losses = {'loss_so3': loss_full_poses}


        joints_xyz = joints_xyz.contiguous().view(-1, self.num_joints * 3)
        # quat_pred = self.layers(joints_xyz).view(-1, 16, 3) # (bz, 3 * 16)
        quat_pred = self.layers(joints_xyz) # (bz, 3 * 16)
        quat_pred = quat_pred.view(-1, 16, 4)
        quat_pred = normalize_quaternion(quat_pred)
        # theta = quaternion_to_angle_axis(quat_pred).contiguous()
        # theta = theta.view(-1, 16 * 3)

        # loss
        loss_quat_l2 = F.mse_loss(quat_pred, quat)
        pred_quat_inv = quaternion_inv(quat_pred)
        real_part = quaternion_mul(quat, pred_quat_inv)
        real_part = real_part[..., -1]
        loss_quat_cos = F.l1_loss(real_part, torch.ones_like(real_part))

        # loss_theta = F.mse_loss(theta, torch.zeros_like(theta))

        losses = {'loss_quat_l2': loss_quat_l2,
                  'loss_quat_cos': loss_quat_cos,
                  # 'loss_theta': loss_theta
                  }
        return losses

    def forward_test(self, joints_xyz):
        # joints_xyz = joints_xyz.contiguous().view(-1, self.num_joints * 3)
        # # If ValueError: Expected more than 1 value per channel when training, got input size [1, 63], model.eval() is needed!
        # full_poses = self.layers(joints_xyz)
        joints_xyz = joints_xyz.contiguous().view(-1, self.num_joints * 3)
        quat = self.layers(joints_xyz)
        quat = quat.view(-1, 16, 4)
        quat = normalize_quaternion(quat)
        theta = quaternion_to_angle_axis(quat).contiguous()
        theta = theta.view(-1, 16 * 3)
        # return full_poses
        return theta

    def forward(self, return_loss=True, **kwargs):
        joints_xyz = kwargs['joints_xyz']
        # full_poses = kwargs['full_poses']
        if return_loss == True:
            quat = kwargs['quat']
            losses = self.forward_train(joints_xyz, quat)
            return losses
        else:
            theta = self.forward_test(joints_xyz)
            return theta

    def train_step(self, data, optimizer):
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)
        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['quat']))
        return outputs

    def val_step(self, data, optimizer):
        return self.train_step(data, optimizer)

    def aug_test(self,):
        pass

    def extract_feat(self,):
        pass

    def simple_test(self,):
        pass
